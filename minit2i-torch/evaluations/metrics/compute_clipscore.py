from typing import Any, List, Union, cast

import torch
from torch import Tensor
from torchmetrics.functional.multimodal.clip_score import (
    _detect_modality,
    _get_clip_model_and_processor,
    _get_features,
    _process_image_data,
    _process_text_data,
)
from transformers import CLIPModel as _CLIPModel
from transformers import CLIPProcessor as _CLIPProcessor


def _as_feature_tensor(v: Any) -> Tensor:
    """CLIPModel.get_image_features / get_text_features return ``BaseModelOutputWithPooling``; HF writes the
    projected CLIP vector into ``pooler_output`` (see ``transformers`` ``CLIPModel.get_*_features``). Older stacks
    may still return a bare ``Tensor``."""
    if isinstance(v, Tensor):
        return v
    pool = getattr(v, "pooler_output", None)
    if isinstance(pool, Tensor):
        return pool
    raise TypeError(f"Expected Tensor or BaseModelOutputWithPooling.pooler_output; got {type(v)!r}")


def _clip_score_update(
    source: Union[Tensor, List[Tensor], List[str], str],
    target: Union[Tensor, List[Tensor], List[str], str],
    model: _CLIPModel,
    processor: _CLIPProcessor,
) -> tuple[Tensor, int]:
    """Update function for CLIP Score."""
    source_modality = _detect_modality(source)
    target_modality = _detect_modality(target)

    source_data = (
        _process_image_data(cast(Union[Tensor, List[Tensor]], source))
        if source_modality == "image"
        else _process_text_data(cast(Union[str, List[str]], source))
    )
    target_data = (
        _process_image_data(cast(Union[Tensor, List[Tensor]], target))
        if target_modality == "image"
        else _process_text_data(cast(Union[str, List[str]], target))
    )

    if len(source_data) != len(target_data):
        raise ValueError(
            "Expected the number of source and target examples to be the same but got "
            f"{len(source_data)} and {len(target_data)}"
        )

    device = (
        source_data[0].device
        if source_modality == "image" and isinstance(source_data[0], Tensor)
        else target_data[0].device
        if target_modality == "image" and isinstance(target_data[0], Tensor)
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device)

    source_features = _get_features(
        cast(List[Union[Tensor, str]], source_data), source_modality, device, model, processor
    )
    target_features = _get_features(
        cast(List[Union[Tensor, str]], target_data), target_modality, device, model, processor
    )
    source_features = _as_feature_tensor(source_features)
    target_features = _as_feature_tensor(target_features)
    source_features = source_features / source_features.norm(p=2, dim=-1, keepdim=True)
    target_features = target_features / target_features.norm(p=2, dim=-1, keepdim=True)

    # Calculate cosine similarity
    score = 100 * (source_features * target_features).sum(axis=-1)
    score = score.cpu() if source_modality == "text" and target_modality == "text" else score
    return score, len(source_data)

class CLIPScorer:
    """One CLIP model load (``_get_clip_model_and_processor``), then ``_clip_score_update`` per batch — same math as
    ``torchmetrics.functional.multimodal.clip_score`` without reloading weights each call.
    """

    def __init__(
        self,
        model_name_or_path: str = "openai/clip-vit-base-patch16",
        device: str | torch.device | None = None,
    ):
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model, self._processor = _get_clip_model_and_processor(model_name_or_path)
        self._model = self._model.to(self._device).eval()

    @torch.inference_mode()
    def __call__(self, images, prompts: list[str]) -> tuple[int, float]:
        images_int = (images * 255).astype("uint8")
        tensor = torch.from_numpy(images_int).permute(0, 3, 1, 2).to(self._device)
        n = int(tensor.shape[0])
        if len(prompts) != n:
            raise ValueError(f"len(prompts)={len(prompts)} != batch={n}")
        score, _ = _clip_score_update(tensor, prompts, self._model, self._processor)
        score = score.mean(0)
        out = torch.max(score, torch.zeros_like(score))
        return n, round(float(out), 4)

    @torch.inference_mode()
    def score_per_sample(self, images, prompts: list[str]) -> list[float]:
        """Return per-sample CLIP scores (clamped to >=0), same math as __call__ but no mean."""
        images_int = (images * 255).astype("uint8")
        tensor = torch.from_numpy(images_int).permute(0, 3, 1, 2).to(self._device)
        n = int(tensor.shape[0])
        if len(prompts) != n:
            raise ValueError(f"len(prompts)={len(prompts)} != batch={n}")
        scores, _ = _clip_score_update(tensor, prompts, self._model, self._processor)
        scores = torch.clamp(scores, min=0.0)
        return scores.tolist()
