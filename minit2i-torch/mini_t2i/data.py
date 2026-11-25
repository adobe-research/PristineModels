from __future__ import annotations

import itertools
import tarfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset

from .config import TrainConfig
from .datasets.finetune import FinetuneMixedStream, source_files_for_root
from .datasets.interface import ImageNetDatasetProvider, ImageNetPristineDatasetProvider, LocalTensorChunkStream, provider_for_config
from .utils import rank, world_size


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
    if cfg.finetune_metadata_dir:
        return  # The keyed loader validates the local archives against the metadata.
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
    _check_imagenet(cfg)


def check_training_data(cfg: TrainConfig) -> None:
    """Fail early with clear messages when configured training data is absent."""
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
        "'local_folder', 'finetune_wds', 'imagenet' and 'imagenet_pristine'"
    )


def collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch], dim=0),
        "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch], dim=0),
        "caption": [x["caption"] for x in batch],
        "source": [x.get("source", "") for x in batch],
    }


PRISTINE_GROUPS = ("safe", "unsafe")


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


def make_loader(cfg: TrainConfig, *, decomposed: bool = False) -> DataLoader:
    dataset = provider_for_config(cfg).build()
    if decomposed and isinstance(dataset, LocalTensorChunkStream):
        dataset.decomposed = True
    elif decomposed != (isinstance(dataset, FinetuneMixedStream) and cfg.finetune_metadata_dir is not None and cfg.cc12m_flash_group == "unsafe"):
        raise ValueError("Decomposed loading requires local chunks or the unsafe fine-tune stream")
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
        collate_fn=collate_pristine if decomposed else collate,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=cfg.dataloader_prefetch if num_workers > 0 else None,
        multiprocessing_context=multiprocessing_context,
        drop_last=True,
    )


def endless_batches(loader: DataLoader, start_batch: int = 0, *, start_step: int = 0):
    """Resume from a batch offset, or start a fresh shuffle at a saved step."""
    if start_batch and start_step:
        raise ValueError("Specify only one of start_batch and start_step")
    epoch, offset = divmod(start_batch, len(loader)) if start_batch else (start_step, 0)
    while True:
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        yield from itertools.islice(loader, offset, None)
        offset = 0
        epoch += 1


def local_batches(loader: DataLoader, consumed_batches: int):
    if not hasattr(loader.dataset, "skip_rows") or loader.batch_size is None:
        raise TypeError("local resume requires a batched stream that supports skipping rows")
    # Each worker skips complete rounds; discard the partial round in iterator order.
    rounds, remaining = divmod(consumed_batches, loader.num_workers or 1)
    loader.dataset.skip_rows = rounds * loader.batch_size
    return itertools.islice(endless_batches(loader), remaining, None)


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
