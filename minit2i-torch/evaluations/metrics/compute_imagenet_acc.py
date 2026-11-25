from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Special sentinel model name: the *exact* InceptionV3 checkpoint torch-fidelity uses
# for FID/precision-recall (the classic "inception-2015-12-05" FID-Inception network). Its classification
# head is 1008-way, in Google's internal 2012-challenge node order (not standard ImageNet-1k
# order) -- node 0 and nodes 1001-1007 are unused padding, nodes 1-1000 are the real classes.
# fid_inception_label_map.json is the verified node_id -> standard_label permutation, built by
# cross-referencing imagenet_2012_challenge_label_map_proto.pbtxt against the standard
# imagenet_class_index.json ordering, and empirically confirmed by running real tench/goldfish
# images through the actual checkpoint (node 449 -> tench, node 450 -> goldfish, as predicted).
_FID_INCEPTION_SENTINEL = "fid-inception-v3"
_FID_LABEL_MAP_PATH = Path(__file__).resolve().parent / "fid_inception_label_map.json"


def _looks_like_timm_id(model_name_or_path: str) -> bool:
    """HF ids are ``org/model`` (contain a ``/``); timm ids are a single token
    (e.g. ``eva02_large_patch14_448.mim_m38m_ft_in22k_in1k``, no ``/``)."""
    return "/" not in model_name_or_path


def _load_fid_inception_extractor():
    import json

    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

    extractor = FeatureExtractorInceptionV3(name="fid_inception_classifier", features_list=["logits"])
    node_to_std = {int(k): v for k, v in json.loads(_FID_LABEL_MAP_PATH.read_text()).items()}
    return extractor, node_to_std


class ImageNetClassifier:
    """Load an ImageNet-1k classifier once, then score batches of images against
    ground-truth labels. Backend is auto-detected from ``model_name_or_path``:
    a HF id (``org/model``, e.g. ``facebook/dinov2-large-imagenet1k-1-layer``) uses
    ``transformers``; the sentinel ``"fid-inception-v3"`` loads the *exact* InceptionV3
    checkpoint torch-fidelity uses for FID/precision-recall (see module-level comment);
    anything else (e.g. ``eva02_large_patch14_448.mim_m38m_ft_in22k_in1k``) is loaded via
    ``timm``. All backends assume/remap to standard ImageNet-1k class ordering (0-999),
    matching this codebase's integer ``label`` fields directly (no name lookup).

    Args:
        model_name_or_path: HF model id, local HF path, timm model name, or
            ``"fid-inception-v3"`` for the FID-Inception backend.
        device: target device; defaults to CUDA if available.
    """

    def __init__(
        self,
        model_name_or_path: str = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        device: str | torch.device | None = None,
    ):
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if model_name_or_path == _FID_INCEPTION_SENTINEL:
            self._backend = "fid_inception"
            self._model, self._node_to_std = _load_fid_inception_extractor()
            self._model = self._model.to(self._device).eval()
            self._transform = None
            self._processor = None
            self._label2id = None  # fid-inception: standard ImageNet-1k index order, ints only
            return

        self._backend = "timm" if _looks_like_timm_id(model_name_or_path) else "hf"

        if self._backend == "timm":
            import timm

            self._model = timm.create_model(model_name_or_path, pretrained=True)
            self._model = self._model.to(self._device).eval()
            data_config = timm.data.resolve_model_data_config(self._model)
            self._transform = timm.data.create_transform(**data_config, is_training=False)
            self._processor = None
            self._label2id = None  # timm models: standard ImageNet-1k index order, ints only
        else:
            self._processor = AutoImageProcessor.from_pretrained(model_name_or_path)
            self._model = AutoModelForImageClassification.from_pretrained(model_name_or_path)
            self._model = self._model.to(self._device).eval()
            self._label2id = {v: k for k, v in self._model.config.id2label.items()}
            self._transform = None

    def _label_ids(self, labels: List[Union[int, str]]) -> list[int]:
        if self._label2id is None:
            if any(not isinstance(lbl, int) for lbl in labels):
                raise ValueError("timm backend only supports integer ImageNet-1k class labels")
            return list(labels)
        return [lbl if isinstance(lbl, int) else self._label2id[lbl] for lbl in labels]

    @torch.inference_mode()
    def _predicted_ids(self, images: np.ndarray) -> list[int]:
        if self._backend == "fid_inception":
            # forward() resizes/normalizes internally; just needs uint8 NCHW RGB.
            batch = torch.from_numpy((images * 255).astype("uint8")).permute(0, 3, 1, 2).to(self._device)
            (logits,) = self._model(batch)  # N x 1008, Google-internal node order
            node_ids = logits.argmax(-1).cpu().tolist()
            # -1 for the 8 unused/background nodes -- never matches a real ground-truth label.
            return [self._node_to_std.get(n, -1) for n in node_ids]

        pil_images = [Image.fromarray((img * 255).astype("uint8")) for img in images]
        if self._backend == "timm":
            batch = torch.stack([self._transform(im) for im in pil_images]).to(self._device)
            logits = self._model(batch)
        else:
            inputs = self._processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            logits = self._model(**inputs).logits
        return logits.argmax(-1).cpu().tolist()

    @torch.inference_mode()
    def __call__(
        self,
        images: np.ndarray,
        labels: List[Union[int, str]],
    ) -> list[bool]:
        """Return per-sample top-1 match booleans.

        Args:
            images: float32 numpy array of shape (N, H, W, C) in [0, 1].
            labels: ground-truth class indices (int) or class name strings (HF backend only).

        Returns:
            List of bools, True where predicted top-1 == ground-truth.
        """
        n = images.shape[0]
        if len(labels) != n:
            raise ValueError(f"len(labels)={len(labels)} != batch size={n}")

        label_ids = self._label_ids(labels)
        predicted_ids = self._predicted_ids(images)

        return [pred == gt for pred, gt in zip(predicted_ids, label_ids)]

    @torch.inference_mode()
    def predict_detailed(
        self,
        images: np.ndarray,
        labels: List[Union[int, str]],
    ) -> list[dict]:
        """Return per-sample prediction dicts with hit flag and class names.

        Returns list of dicts with keys:
            hit (bool), pred_id (int), pred_name (str), gt_id (int), gt_name (str)

        Note: class names (``pred_name``/``gt_name``) are only resolved for the HF
        backend (via ``model.config.id2label``); timm backend falls back to the
        raw integer id as the name.
        """
        n = images.shape[0]
        if len(labels) != n:
            raise ValueError(f"len(labels)={len(labels)} != batch size={n}")

        label_ids = self._label_ids(labels)
        predicted_ids = self._predicted_ids(images)

        id2label = self._model.config.id2label if self._backend == "hf" else {}
        return [
            {
                "hit": pred == gt,
                "pred_id": pred,
                "pred_name": id2label.get(pred, str(pred)),
                "gt_id": gt,
                "gt_name": id2label.get(gt, str(gt)),
            }
            for pred, gt in zip(predicted_ids, label_ids)
        ]
