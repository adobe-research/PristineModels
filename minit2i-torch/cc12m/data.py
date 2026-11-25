"""CC12M whole-caption training with local metadata and FlashS3 JPEG fetching.

Set dataset_backend="cc12m_flash_s3" and cc12m_flash_caption_dir to the
whole_prompts/pretrain directory. The existing cc12m_flash_tagged_jsonl field
accepts the unsafe-key JSON settings file (its "unsafe" array); the existing
cc12m_flash_excluded_jsonl field accepts the exclusion settings JSON and applies
every group globally. cc12m_flash_group selects "safe", "unsafe", or all when
unset. Validation settings are deliberately ignored on this path.
S3 clients are created lazily per thread by the original module-level helper.

make_cc12m_twobranch_loaders uses whole captions for safe samples and paired
top-level subject_caption/residual_caption fields from
cc12m_flash_decomposed_caption_dir for unsafe samples. Every retained unsafe
key must have a decomposition; missing entries are errors, not whole-caption
fallbacks. Both groups use the same global exclusions and image retry behavior.

Fetch/decode failures resample within the selected pool without a retry limit,
matching the original training recipe. The replacement image and caption stay
paired. A persistent S3 failure can therefore keep a worker retrying indefinitely.
Failures are reported immediately and every ten failed attempts thereafter.

FlashS3's PyTorch patch must already be installed in the training environment.
Importing this module never installs packages or patches PyTorch.
"""

from __future__ import annotations

import inspect
import io
import itertools
import json
import os
import random
import re
import tarfile
import threading
import time
from contextlib import closing
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Protocol, TypedDict

import boto3
import torch
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    CredentialRetrievalError,
    NoCredentialsError,
)
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from torchvision import transforms
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing_extensions import NotRequired

from mini_t2i.config import TrainConfig
from mini_t2i.data import endless_batches as endless_batches
from mini_t2i.datasets.finetune import source_files_for_root
from mini_t2i.datasets.interface import ImageNetDatasetProvider, ImageNetPristineDatasetProvider, provider_for_config
from mini_t2i.utils import rank, read_token, world_size


def _require_dir(path: str | Path | None, name: str) -> Path:
    if not path:
        raise RuntimeError(f"{name} is required for the selected dataset backend")
    root = Path(path)
    if not root.exists():
        raise RuntimeError(f"{name} does not exist: {root}")
    if not root.is_dir():
        raise RuntimeError(f"{name} must be a directory, got: {root}")
    return root


def _check_local_tensor_chunks(cfg: TrainConfig) -> None:
    root = _require_dir(cfg.local_dataset_dir, "local_dataset_dir")
    files = sorted(root.glob("chunk_*.pt"))
    if not files:
        raise RuntimeError(
            f"no pretraining tensor chunks found in {root}. "
            "Expected files named chunk_*.pt; run tools/prepare_cc12m_chunks.py first."
        )

    first = files[0]
    try:
        sample = torch.load(first, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"failed to read pretraining tensor chunk {first}: {exc}") from exc

    required = {"pixel_values", "input_ids", "attention_mask"}
    missing = sorted(required - set(sample.keys()))
    if missing:
        raise RuntimeError(
            f"pretraining tensor chunk {first} is missing required key(s): {', '.join(missing)}"
        )


def _check_tar_has_sample(path: Path, source: str) -> None:
    if not tarfile.is_tarfile(path):
        raise RuntimeError(f"finetune source {source} has a non-tar shard: {path}")
    try:
        with tarfile.open(path) as tar:
            names = [member.name.lower() for _, member in zip(range(256), tar) if member.isfile()]
    except Exception as exc:
        raise RuntimeError(f"failed to inspect finetune tar shard {path}: {exc}") from exc

    if not names:
        raise RuntimeError(f"finetune tar shard is empty: {path}")
    has_image = any(name.endswith((".jpg", ".jpeg", ".png")) for name in names)
    has_text = any(name.endswith(".txt") for name in names)
    if not has_image or not has_text:
        raise RuntimeError(
            f"finetune tar shard {path} does not look like image/text WebDataset data "
            f"(image={has_image}, txt={has_text})"
        )


def _check_finetune_wds(cfg: TrainConfig) -> None:
    if not cfg.finetune_dataset_dir:
        if getattr(cfg, "finetune_hf_streaming", False):
            return
        raise RuntimeError("finetune_dataset_dir is required for finetune_wds backend")

    root_path = Path(cfg.finetune_dataset_dir)
    if not root_path.exists() and getattr(cfg, "finetune_hf_streaming", False):
        return

    root = _require_dir(cfg.finetune_dataset_dir, "finetune_dataset_dir")
    if not cfg.finetune_sources:
        raise RuntimeError("finetune_sources must contain at least one source")

    missing = []
    first_files = []
    for source in cfg.finetune_sources:
        files = source_files_for_root(root, source)
        if files:
            first_files.append((source, files[0]))
        elif source == "dalle3" and sorted((root / "dalle3_parquet").glob("*.parquet")):
            continue
        else:
            missing.append(source)
    if missing:
        raise RuntimeError(
            f"missing finetune source(s) under {root}: {', '.join(missing)}. "
            "Expected local WebDataset .tar shards; run tools/prepare_120k_mix_layout.py first."
        )

    source, first = first_files[0]
    _check_tar_has_sample(first, source)


def _require_file(path: str | None, name: str) -> None:
    if path and not Path(path).is_file():
        raise RuntimeError(f"{name} does not exist: {path}")


def _check_cc12m(cfg: TrainConfig) -> str:
    root = _require_dir(cfg.cc12m_flash_caption_dir, "cc12m_flash_caption_dir")
    if next(root.rglob("captions_merged.jsonl"), None) is None:
        raise RuntimeError(f"no captions_merged.jsonl found under {root}")
    bucket = cfg.cc12m_flash_s3_bucket
    if not bucket or not bucket.strip() or "/" in bucket or bucket != bucket.strip():
        raise RuntimeError("cc12m_flash_s3_bucket must be a bucket name, not an S3 URL")
    if cfg.cc12m_flash_group not in (None, "safe", "unsafe"):
        raise RuntimeError("cc12m_flash_group must be 'safe', 'unsafe', or unset")
    if cfg.cc12m_flash_group and not cfg.cc12m_flash_tagged_jsonl:
        raise RuntimeError("safe/unsafe selection requires cc12m_flash_tagged_jsonl (unsafe-key settings JSON)")
    for name in ("cc12m_flash_tagged_jsonl", "cc12m_flash_excluded_jsonl"):
        _require_file(getattr(cfg, name), name)
    for name in ("image_size", "prompt_length", "micro_batch_size", "cc12m_flash_num_threads"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RuntimeError(f"{name} must be a positive integer")
    if isinstance(cfg.num_workers, bool) or not isinstance(cfg.num_workers, int) or cfg.num_workers < 0:
        raise RuntimeError("num_workers must be a nonnegative integer")
    limit = cfg.cc12m_flash_max_train_samples
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise RuntimeError("cc12m_flash_max_train_samples must be a positive integer or unset")
    return bucket


def _check_imagenet(cfg: TrainConfig) -> None:
    for name in ("imagenet_captions_dir", "imagenet_val_captions_dir"):
        if name == "imagenet_val_captions_dir" and not cfg.imagenet_val_captions_dir:
            continue
        captions_dir = _require_dir(getattr(cfg, name), name)
        if next(captions_dir.rglob("captions_merged.jsonl"), None) is None:
            raise RuntimeError(f"no captions_merged.jsonl found under {name}={captions_dir}")
    _require_dir(cfg.imagenet_image_dir, "imagenet_image_dir")
    if cfg.imagenet_group not in (None, "safe", "unsafe"):
        raise RuntimeError(f"imagenet_group must be 'safe', 'unsafe' or unset, got {cfg.imagenet_group!r}")
    if cfg.imagenet_group and not cfg.imagenet_setting_labels_path:
        raise RuntimeError("imagenet_group needs imagenet_setting_labels_path")
    _require_file(cfg.imagenet_setting_labels_path, "imagenet_setting_labels_path")
    _require_file(cfg.imagenet_leakage_path, "imagenet_leakage_path")


def _check_imagenet_pristine(cfg: TrainConfig) -> None:
    if not cfg.imagenet_setting_labels_path:
        raise RuntimeError("imagenet_pristine needs imagenet_setting_labels_path to split safe/unsafe classes")
    for name in ("imagenet_whole_captions_dir", "imagenet_whole_val_captions_dir"):
        if getattr(cfg, name):
            _require_dir(getattr(cfg, name), name)
    _check_imagenet(cfg)


def check_training_data(cfg: TrainConfig) -> None:
    """Fail early with clear messages when configured training data is absent."""
    if cfg.dataset_backend == "cc12m_flash_s3":
        _check_cc12m(cfg)
        return
    if cfg.dataset_backend == "local_folder":
        _check_local_tensor_chunks(cfg)
        return
    if cfg.dataset_backend == "finetune_wds":
        _check_finetune_wds(cfg)
        return
    if cfg.dataset_backend == "imagenet":
        _check_imagenet(cfg)
        return
    if cfg.dataset_backend == "imagenet_pristine":
        _check_imagenet_pristine(cfg)
        return
    raise RuntimeError(
        f"unknown dataset_backend={cfg.dataset_backend!r}; supported backends are "
        "'local_folder', 'finetune_wds', 'imagenet', 'imagenet_pristine' and 'cc12m_flash_s3'"
    )


def collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch], dim=0),
        "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch], dim=0),
        "caption": [x["caption"] for x in batch],
        "source": [x.get("source", "") for x in batch],
    }


class CC12MSample(TypedDict):
    key: str
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    caption: str
    source: str
    input_ids_mode: NotRequired[torch.Tensor]
    attention_mask_mode: NotRequired[torch.Tensor]
    input_ids_shared: NotRequired[torch.Tensor]
    attention_mask_shared: NotRequired[torch.Tensor]
    shared_caption: NotRequired[str]


class _S3Body(Protocol):
    def read(self) -> bytes: ...
    def close(self) -> None: ...


class _S3Response(TypedDict):
    Body: _S3Body


class _S3Client(Protocol):
    def get_object(self, *, Bucket: str, Key: str) -> _S3Response: ...


class _CC12MSampleError(RuntimeError):
    pass


_CC12M_KEY = re.compile(r"[0-9]{9}")
_s3_local = threading.local()
_S3_TTL = 3000
_S3_EXPIRED_CODES = {"ExpiredToken", "ExpiredTokenException", "InvalidClientTokenId", "AuthFailure"}


def _s3() -> _S3Client:
    now = time.monotonic()
    if not hasattr(_s3_local, "client") or now - getattr(_s3_local, "ts", 0.0) > _S3_TTL:
        _s3_local.client = boto3.Session().client("s3")
        _s3_local.ts = now
    return _s3_local.client


def _cc12m_key(value: object, location: str) -> int:
    if not isinstance(value, str) or _CC12M_KEY.fullmatch(value) is None:
        raise _CC12MSampleError(f"{location}: expected a nine-digit string image key")
    return int(value)


def _cc12m_caption(line: bytes, location: str) -> tuple[int, str]:
    try:
        row = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _CC12MSampleError(f"{location}: invalid caption JSON: {exc}") from exc
    if not isinstance(row, dict):
        raise _CC12MSampleError(f"{location}: caption row must be an object")
    key = _cc12m_key(row.get("key"), location)
    captions = row.get("captions")
    caption = captions.get("child_caption") if isinstance(captions, dict) else None
    if not isinstance(caption, str) or not caption.strip():
        raise _CC12MSampleError(f"{location}: captions.child_caption must be a nonempty string")
    return key, caption


def _cc12m_decomposed_caption(line: bytes, location: str) -> tuple[int, str, str]:
    try:
        row = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _CC12MSampleError(f"{location}: invalid decomposed caption JSON: {exc}") from exc
    if not isinstance(row, dict):
        raise _CC12MSampleError(f"{location}: expected a decomposed caption object")
    key = _cc12m_key(row.get("key"), location)
    subject = row.get("subject_caption")
    residual = row.get("residual_caption")
    if not isinstance(subject, str) or not subject.strip():
        raise _CC12MSampleError(f"{location}: subject_caption must be a nonempty string")
    if not isinstance(residual, str) or not residual.strip():
        raise _CC12MSampleError(f"{location}: residual_caption must be a nonempty string")
    return key, subject, residual


def check_cc12m_twobranch_data(cfg: TrainConfig) -> None:
    _check_cc12m(cfg)
    if cfg.cc12m_flash_group is not None:
        raise RuntimeError("two-branch training selects both groups; leave cc12m_flash_group unset")
    if not cfg.cc12m_flash_tagged_jsonl:
        raise RuntimeError("two-branch training requires cc12m_flash_tagged_jsonl (unsafe-key settings JSON)")
    root = _require_dir(cfg.cc12m_flash_decomposed_caption_dir, "cc12m_flash_decomposed_caption_dir")
    if next(root.rglob("captions_merged.jsonl"), None) is None:
        raise RuntimeError(f"no decomposed captions_merged.jsonl found under {root}")


def _cc12m_key_groups(path: str | None) -> dict[str, set[int]]:
    if path is None:
        return {}
    try:
        with Path(path).open(encoding="utf-8") as stream:
            groups = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read CC12M key settings {path}: {exc}") from exc
    if not isinstance(groups, dict) or not groups or set(groups) - {"safe", "unsafe"}:
        raise RuntimeError(f"{path}: expected an object containing safe/unsafe key arrays")
    result: dict[str, set[int]] = {}
    seen: set[int] = set()
    for group, values in groups.items():
        if not isinstance(values, list):
            raise RuntimeError(f"{path}: {group} must be an array of image keys")
        keys: set[int] = set()
        for value in values:
            key = _cc12m_key(value, f"{path}:{group}")
            if key in seen:
                raise RuntimeError(f"{path}: duplicate image key {key:09d}")
            keys.add(key)
            seen.add(key)
        result[group] = keys
    return result


def _cc12m_signature(stat: os.stat_result) -> tuple[int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


class CC12MFlashS3Dataset(Dataset[CC12MSample]):
    """Whole captions for every mode; a byte-offset index keeps caption text on disk.

    A sample cap stops the scan after that many retained rows. Exclusions and
    group selection are applied before the cap. Keys and captions in every
    scanned row are validated, including rows rejected by those filters.
    """

    def __init__(self, cfg: TrainConfig, tokenizer: PreTrainedTokenizerBase):
        self.bucket = _check_cc12m(cfg)
        self.tokenizer = tokenizer
        self.prompt_length = cfg.prompt_length
        prefix = cfg.cc12m_flash_s3_prefix.strip("/")
        self.prefix = f"{prefix}/" if prefix else ""
        self.transform = transforms.Compose([
            transforms.Resize(cfg.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(cfg.image_size),
            transforms.PILToTensor(),
        ])
        self.files = sorted(_require_dir(cfg.cc12m_flash_caption_dir, "cc12m_flash_caption_dir").rglob("captions_merged.jsonl"))
        self.signatures = [_cc12m_signature(path.stat()) for path in self.files]
        self._keys: list[int] = []
        self._file_ids: list[int] = []
        self._offsets: list[int] = []
        excluded: set[int] = set()
        for keys in _cc12m_key_groups(cfg.cc12m_flash_excluded_jsonl).values():
            excluded.update(keys)
        unsafe_keys: set[int] = set()
        if cfg.cc12m_flash_group:
            groups = _cc12m_key_groups(cfg.cc12m_flash_tagged_jsonl)
            if "unsafe" not in groups:
                raise RuntimeError("unsafe-key settings must contain the unsafe array")
            unsafe_keys = groups["unsafe"]
        seen: set[int] = set()
        limit = cfg.cc12m_flash_max_train_samples
        for file_id, path in enumerate(self.files):
            with path.open("rb") as stream:
                if _cc12m_signature(os.fstat(stream.fileno())) != self.signatures[file_id]:
                    raise RuntimeError(f"caption metadata changed while indexing: {path}")
                offset = 0
                for line_no, line in enumerate(stream, 1):
                    key, _ = _cc12m_caption(line, f"{path}:{line_no}")
                    if key in seen:
                        raise RuntimeError(f"{path}:{line_no}: duplicate image key {key:09d}")
                    seen.add(key)
                    selected = (
                        cfg.cc12m_flash_group is None
                        or (cfg.cc12m_flash_group == "unsafe") == (key in unsafe_keys)
                    )
                    if key not in excluded and selected:
                        self._keys.append(key)
                        self._file_ids.append(file_id)
                        self._offsets.append(offset)
                    offset += len(line)
                    if limit is not None and len(self) >= limit:
                        break
                if _cc12m_signature(os.fstat(stream.fileno())) != self.signatures[file_id]:
                    raise RuntimeError(f"caption metadata changed while indexing: {path}")
            if limit is not None and len(self) >= limit:
                break
        if not self._keys:
            raise RuntimeError("CC12M training pool is empty after exclusions and group selection")

    def __len__(self) -> int:
        return len(self._keys)

    def key_at(self, index: int) -> str:
        return f"{self._keys[index]:09d}"

    def _image_bytes(self, key: str) -> bytes:
        response = _s3().get_object(Bucket=self.bucket, Key=f"{self.prefix}{key}.jpg")
        with closing(response["Body"]) as body:
            return body.read()

    @staticmethod
    def _parse_captions(line: bytes, location: str) -> tuple[int, str, str | None]:
        key, caption = _cc12m_caption(line, location)
        return key, caption, None

    def _captions_at(self, index: int) -> tuple[str, str | None]:
        file_id = self._file_ids[index]
        path = self.files[file_id]
        with path.open("rb") as stream:
            if _cc12m_signature(os.fstat(stream.fileno())) != self.signatures[file_id]:
                raise _CC12MSampleError(f"caption metadata changed after indexing: {path}")
            stream.seek(self._offsets[index])
            actual_key, caption, shared_caption = self._parse_captions(
                stream.readline(), f"{path}:offset {self._offsets[index]}"
            )
            if _cc12m_signature(os.fstat(stream.fileno())) != self.signatures[file_id]:
                raise _CC12MSampleError(f"caption metadata changed while reading: {path}")
        if actual_key != self._keys[index]:
            raise _CC12MSampleError(f"caption key changed after indexing: expected {self.key_at(index)}")
        return caption, shared_caption

    def __getitem__(self, index: int) -> CC12MSample:
        attempts = 0
        while True:
            key = self.key_at(index)
            try:
                with Image.open(io.BytesIO(self._image_bytes(key))) as image:
                    rgb = image.convert("RGB")
                break
            except (BotoCoreError, ClientError, OSError, ValueError, TypeError, KeyError, Image.DecompressionBombError) as exc:
                attempts += 1
                code = exc.response.get("Error", {}).get("Code", "") if isinstance(exc, ClientError) else ""
                if code in _S3_EXPIRED_CODES or isinstance(exc, (NoCredentialsError, CredentialRetrievalError)):
                    _s3_local.ts = float("-inf")
                if attempts == 1 or attempts % 10 == 0:
                    print(
                        f"[cc12m] WARNING: {attempts} consecutive S3/image failures, "
                        f"last s3://{self.bucket}/{self.prefix}{key}.jpg: {exc}; resampling",
                        flush=True,
                    )
                index = random.randrange(len(self))
        try:
            with rgb:
                pixel_values = self.transform(rgb)
            caption, shared_caption = self._captions_at(index)
            tokens = self.tokenizer(
                caption,
                max_length=self.prompt_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            sample: CC12MSample = {
                "key": key,
                "pixel_values": pixel_values,
                "input_ids": tokens["input_ids"][0].long(),
                "attention_mask": tokens["attention_mask"][0].long(),
                "caption": caption,
                "source": "cc12m",
            }
            if shared_caption is not None:
                shared_tokens = self.tokenizer(
                    shared_caption,
                    max_length=self.prompt_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                sample["input_ids_mode"] = sample["input_ids"]
                sample["attention_mask_mode"] = sample["attention_mask"]
                sample["input_ids_shared"] = shared_tokens["input_ids"][0].long()
                sample["attention_mask_shared"] = shared_tokens["attention_mask"][0].long()
                sample["shared_caption"] = shared_caption
            return sample
        except (OSError, ValueError, TypeError, KeyError, Image.DecompressionBombError) as exc:
            raise _CC12MSampleError(
                f"failed to load CC12M image/caption s3://{self.bucket}/{self.prefix}{key}.jpg: {exc}"
            ) from exc


class CC12MDecomposedFlashS3Dataset(CC12MFlashS3Dataset):
    """Index decompositions in the same retained-key order as the unsafe whole pool."""

    def __init__(self, cfg: TrainConfig, tokenizer: PreTrainedTokenizerBase):
        if cfg.cc12m_flash_group != "unsafe":
            raise RuntimeError("decomposed CC12M loading requires cc12m_flash_group='unsafe'")
        super().__init__(cfg, tokenizer)
        root = _require_dir(cfg.cc12m_flash_decomposed_caption_dir, "cc12m_flash_decomposed_caption_dir")
        self.files = sorted(root.rglob("captions_merged.jsonl"))
        self.signatures = [_cc12m_signature(path.stat()) for path in self.files]
        positions = {key: index for index, key in enumerate(self._keys)}
        self._file_ids = [-1] * len(self)
        self._offsets = [0] * len(self)
        seen: set[int] = set()
        for file_id, path in enumerate(self.files):
            with path.open("rb") as stream:
                if _cc12m_signature(os.fstat(stream.fileno())) != self.signatures[file_id]:
                    raise RuntimeError(f"decomposed metadata changed while indexing: {path}")
                offset = 0
                for line_no, line in enumerate(stream, 1):
                    key, _, _ = _cc12m_decomposed_caption(line, f"{path}:{line_no}")
                    if key in seen:
                        raise RuntimeError(f"{path}:{line_no}: duplicate decomposed image key {key:09d}")
                    seen.add(key)
                    index = positions.get(key)
                    if index is not None:
                        self._file_ids[index] = file_id
                        self._offsets[index] = offset
                    offset += len(line)
                if _cc12m_signature(os.fstat(stream.fileno())) != self.signatures[file_id]:
                    raise RuntimeError(f"decomposed metadata changed while indexing: {path}")
        missing = [self.key_at(index) for index, file_id in enumerate(self._file_ids) if file_id < 0]
        if missing:
            raise RuntimeError(
                f"{len(missing)} retained unsafe CC12M keys have no decomposition; first keys: {missing[:10]}"
            )

    @staticmethod
    def _parse_captions(line: bytes, location: str) -> tuple[int, str, str | None]:
        return _cc12m_decomposed_caption(line, location)


@dataclass(frozen=True)
class _CC12MFailure:
    message: str


class _FlashS3Samples(Dataset[CC12MSample | _CC12MFailure]):
    """Carry known failures to collation because FlashS3 otherwise drops them."""

    def __init__(self, samples: CC12MFlashS3Dataset):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> CC12MSample | _CC12MFailure:
        try:
            return self.samples[index]
        except _CC12MSampleError as exc:
            return _CC12MFailure(str(exc))


def _collate_cc12m(
    batch: list[CC12MSample | _CC12MFailure], *, expected_size: int
) -> dict[str, torch.Tensor | list[str]]:
    samples: list[CC12MSample] = []
    for item in batch:
        if isinstance(item, _CC12MFailure):
            raise _CC12MSampleError(item.message)
        samples.append(item)
    if len(samples) != expected_size:
        raise RuntimeError(
            f"FlashS3 returned an incomplete CC12M batch ({len(samples)}/{expected_size}); "
            "a sample failed unexpectedly, see FlashS3's worker error"
        )
    out: dict[str, torch.Tensor | list[str]] = {
        "pixel_values": torch.stack([item["pixel_values"] for item in samples]),
        "input_ids": torch.stack([item["input_ids"] for item in samples]),
        "attention_mask": torch.stack([item["attention_mask"] for item in samples]),
        "key": [item["key"] for item in samples],
        "caption": [item["caption"] for item in samples],
        "source": [item["source"] for item in samples],
    }
    if any(
        any(key in item for key in (
            "input_ids_mode", "attention_mask_mode", "input_ids_shared", "attention_mask_shared", "shared_caption"
        ))
        for item in samples
    ):
        if not all(
            all(key in item for key in (
                "input_ids_mode", "attention_mask_mode", "input_ids_shared", "attention_mask_shared", "shared_caption"
            ))
            for item in samples
        ):
            raise _CC12MSampleError("CC12M batch mixes whole and incomplete decomposed captions")
        for key in ("input_ids_mode", "attention_mask_mode", "input_ids_shared", "attention_mask_shared"):
            out[key] = torch.stack([item[key] for item in samples])
        out["shared_caption"] = [item["shared_caption"] for item in samples]
    return out


def make_cc12m_loader(cfg: TrainConfig, tokenizer: PreTrainedTokenizerBase | None = None) -> DataLoader:
    """Return one training-only whole-caption loader; no validation keys are read or held out."""
    _check_cc12m(cfg)
    if "num_threads" not in inspect.signature(DataLoader).parameters:
        raise RuntimeError(
            "cc12m_flash_s3 requires FlashS3's patched PyTorch DataLoader with num_threads. "
            "Install/apply FlashS3 in an isolated training environment; this module does not patch PyTorch."
        )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.t5_name, token=read_token(cfg.hf_token_file), model_max_length=cfg.prompt_length
        )
    return _make_cc12m_loader(cfg, CC12MFlashS3Dataset(cfg, tokenizer))


def _make_cc12m_loader(cfg: TrainConfig, samples: CC12MFlashS3Dataset) -> DataLoader:
    dataset = _FlashS3Samples(samples)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size(), rank=rank(), shuffle=True, seed=cfg.seed, drop_last=True
    )
    if len(sampler) < cfg.micro_batch_size:
        raise RuntimeError("CC12M training pool has no complete rank-local training batch")
    num_workers = cfg.num_workers
    return DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        num_threads=cfg.cc12m_flash_num_threads,
        collate_fn=partial(_collate_cc12m, expected_size=cfg.micro_batch_size),
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=cfg.dataloader_prefetch if num_workers > 0 else None,
        multiprocessing_context=cfg.dataloader_multiprocessing_context if num_workers > 0 else None,
        drop_last=True,
    )


PRISTINE_GROUPS = ("safe", "unsafe")


def make_cc12m_twobranch_loaders(
    cfg: TrainConfig, tokenizer: PreTrainedTokenizerBase | None = None
) -> dict[str, DataLoader]:
    """Rank-sharded safe whole-caption and unsafe subject/residual training loaders."""
    check_cc12m_twobranch_data(cfg)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.t5_name, token=read_token(cfg.hf_token_file), model_max_length=cfg.prompt_length
        )
    safe_cfg = replace(cfg, cc12m_flash_group="safe")
    unsafe_cfg = replace(cfg, cc12m_flash_group="unsafe")
    safe = make_cc12m_loader(safe_cfg, tokenizer)
    unsafe = _make_cc12m_loader(unsafe_cfg, CC12MDecomposedFlashS3Dataset(unsafe_cfg, tokenizer))
    return {"safe": safe, "unsafe": unsafe}


def collate_pristine(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    out: dict[str, torch.Tensor | list[str]] = {
        key: torch.stack([x[key] for x in batch], dim=0)
        for key in ("pixel_values", "input_ids_mode", "attention_mask_mode", "input_ids_shared", "attention_mask_shared")
    }
    for key in ("caption", "shared_caption", "source"):
        out[key] = [x[key] for x in batch]
    return out


def _pristine_loader(cfg: TrainConfig, dataset, train: bool, collate_fn=collate_pristine) -> DataLoader:
    sampler = DistributedSampler(
        dataset, num_replicas=world_size(), rank=rank(), shuffle=train, seed=cfg.seed, drop_last=train
    )
    num_workers = cfg.num_workers
    return DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=cfg.dataloader_prefetch if num_workers > 0 else None,
        multiprocessing_context=cfg.dataloader_multiprocessing_context if num_workers > 0 else None,
        drop_last=train,
    )


def make_pristine_loaders(cfg: TrainConfig) -> dict[str, DataLoader]:
    """One shuffled, rank-sharded train loader per group, keyed "safe" / "unsafe"."""
    return {
        group: _pristine_loader(cfg, ImageNetPristineDatasetProvider(cfg, group).build(), train=True)
        for group in PRISTINE_GROUPS
    }


def make_pristine_val_loaders(cfg: TrainConfig) -> dict[str, DataLoader] | None:
    """Finite, unshuffled val loaders per group; None if no val dir."""
    if not cfg.imagenet_val_captions_dir:
        return None
    # The leakage ledger holds train global_row_index values; val rows use their own 0..N index space.
    return {
        group: _pristine_loader(
            cfg,
            ImageNetPristineDatasetProvider(cfg, group, captions_dir=cfg.imagenet_val_captions_dir, apply_leakage=False).build(),
            train=False,
        )
        for group in PRISTINE_GROUPS
    }


def make_twobranch_loaders(cfg: TrainConfig) -> dict[str, DataLoader]:
    """safe: plain whole-caption batches; unsafe: pristine subject/residual batches. Both group-filtered."""
    if not cfg.imagenet_whole_captions_dir:
        raise RuntimeError("two-branch training needs imagenet_whole_captions_dir for the safe group")
    safe = ImageNetDatasetProvider(cfg, captions_dir=cfg.imagenet_whole_captions_dir, group="safe").build()
    unsafe = ImageNetPristineDatasetProvider(cfg, "unsafe").build()
    return {
        "safe": _pristine_loader(cfg, safe, train=True, collate_fn=collate),
        "unsafe": _pristine_loader(cfg, unsafe, train=True),
    }


def make_twobranch_val_loaders(cfg: TrainConfig) -> dict[str, DataLoader] | None:
    if not cfg.imagenet_whole_val_captions_dir or not cfg.imagenet_val_captions_dir:
        return None
    # The leakage ledger holds train global_row_index values; val rows use their own 0..N index space.
    safe = ImageNetDatasetProvider(
        cfg, captions_dir=cfg.imagenet_whole_val_captions_dir, apply_leakage=False, group="safe"
    ).build()
    unsafe = ImageNetPristineDatasetProvider(
        cfg, "unsafe", captions_dir=cfg.imagenet_val_captions_dir, apply_leakage=False
    ).build()
    return {
        "safe": _pristine_loader(cfg, safe, train=False, collate_fn=collate),
        "unsafe": _pristine_loader(cfg, unsafe, train=False),
    }


def make_loader(cfg: TrainConfig) -> DataLoader:
    if cfg.dataset_backend == "cc12m_flash_s3":
        return make_cc12m_loader(cfg)
    dataset = provider_for_config(cfg).build()
    # Iterable streams shard and shuffle themselves; map-style datasets need a sampler.
    sampler = None
    if not isinstance(dataset, IterableDataset):
        sampler = DistributedSampler(
            dataset, num_replicas=world_size(), rank=rank(), shuffle=True, seed=cfg.seed, drop_last=True
        )
    num_workers = cfg.num_workers
    multiprocessing_context = cfg.dataloader_multiprocessing_context if num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=cfg.dataloader_prefetch if num_workers > 0 else None,
        multiprocessing_context=multiprocessing_context,
        drop_last=True,
    )


def make_val_loader(cfg: TrainConfig) -> DataLoader | None:
    """Finite, unshuffled ImageNet val loader with each GPU taking its own slice; None if no val dir."""
    if cfg.dataset_backend != "imagenet" or not cfg.imagenet_val_captions_dir:
        return None
    # The leakage ledger holds train global_row_index values; val rows use their own 0..N index space.
    dataset = ImageNetDatasetProvider(cfg, captions_dir=cfg.imagenet_val_captions_dir, apply_leakage=False).build()
    # drop_last=False pads the last slices with repeats so every GPU runs the same number of batches.
    sampler = DistributedSampler(dataset, num_replicas=world_size(), rank=rank(), shuffle=False, drop_last=False)
    num_workers = cfg.num_workers
    return DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=cfg.dataloader_prefetch if num_workers > 0 else None,
        multiprocessing_context=cfg.dataloader_multiprocessing_context if num_workers > 0 else None,
        drop_last=False,
    )


def take_batches(loader: DataLoader, n: int):
    return itertools.islice(loader, n)
