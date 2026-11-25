from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer

from ..config import TrainConfig
from ..utils import rank, read_token, world_size
from .finetune import FinetuneMixedStream
from .imagenet import (
    ImageNetCaptionDataset,
    ImageNetDecomposedCaptionDataset,
    ImageNetPristineTrainingSamples,
    ImageNetTrainingSamples,
    load_class_groups,
    load_leaked_row_ids,
)


class DatasetProvider(Protocol):
    def build(self) -> Dataset:
        ...


class LocalTensorChunkStream(IterableDataset):
    """Local-only path for preprocessed tensor chunks.

    This intentionally assumes all dataset assets already exist under a local
    directory. It performs no downloads and no image decoding. The expected file
    format matches the local tensor chunks used by the pretraining pipeline:
    `chunk_*.pt` files containing `pixel_values`, `input_ids`,
    `attention_mask`, and optional `caption`.

    Set skip_rows per worker before iterating to resume the same shuffle.
    """

    skip_rows = 0
    decomposed = False

    def __init__(self, cfg: TrainConfig):
        if not cfg.local_dataset_dir:
            raise RuntimeError("local_dataset_dir is required for local_folder backend")
        self.cfg = cfg
        self.root = Path(cfg.local_dataset_dir)
        self.files = sorted(self.root.glob("chunk_*.pt"))
        if not self.files:
            raise RuntimeError(f"no local tensor chunks found in {self.root}")

    def _files_for_worker(self) -> list[Path]:
        worker = torch.utils.data.get_worker_info()
        shard_index = rank()
        num_shards = world_size()
        if worker is not None:
            shard_index = rank() * worker.num_workers + worker.id
            num_shards = world_size() * worker.num_workers
        return self.files[shard_index::num_shards]

    def __iter__(self):
        files = self._files_for_worker()
        group = self.cfg.cc12m_flash_group
        excluded = load_leaked_row_ids(ledger) if (ledger := self.cfg.cc12m_flash_excluded_jsonl) else set()
        unsafe = load_leaked_row_ids(self.cfg.cc12m_flash_tagged_jsonl) if group else set()
        rows = {
            path: [
                i for i, key in enumerate(torch.load(path, map_location="cpu", weights_only=False, mmap=True)["key"])
                if int(key) not in excluded and (not group or (int(key) in unsafe) == (group == "unsafe"))
            ]
            for path in files
        } if group or excluded else {}
        files = [path for path in files if not rows or rows[path]]
        if not files:
            raise RuntimeError(f"{self.root}: rank/worker shard has no eligible local rows (group={group!r})")
        rng = random.Random(self.cfg.seed + rank())
        skip_rows = self.skip_rows
        token_keys = (
            ("input_ids_mode", "attention_mask_mode", "input_ids_shared", "attention_mask_shared")
            if self.decomposed else ("input_ids", "attention_mask")
        )
        while True:
            order = list(files)
            rng.shuffle(order)
            for path in order:
                chunk = torch.load(path, map_location="cpu", weights_only=False, mmap=skip_rows > 0)
                indices = rows[path].copy() if rows else list(range(chunk["pixel_values"].shape[0]))
                n = len(indices)
                rng.shuffle(indices)
                if skip_rows:
                    indices = indices[skip_rows:]
                    skip_rows = max(0, skip_rows - n)
                if self.decomposed:
                    missing = set((*token_keys, "subject_caption", "shared_caption")) - chunk.keys()
                    if missing:
                        raise ValueError(f"{path}: missing decomposed caption fields: {', '.join(sorted(missing))}")
                    captions = chunk["subject_caption"]
                else:
                    captions = chunk.get("caption") or [""] * n
                for idx in indices:
                    sample = {
                        "pixel_values": chunk["pixel_values"][idx],
                        "caption": captions[idx],
                        **{key: chunk[key][idx].long() for key in token_keys},
                    }
                    if self.decomposed:
                        sample.update(shared_caption=chunk["shared_caption"][idx], source="local")
                    yield sample


class LocalFolderDatasetProvider:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

    def build(self) -> IterableDataset:
        return LocalTensorChunkStream(self.cfg)


class FinetuneWDSDatasetProvider:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

    def build(self) -> IterableDataset:
        return FinetuneMixedStream(self.cfg)



class ImageNetDatasetProvider:
    def __init__(self, cfg: TrainConfig, captions_dir: str | None = None, apply_leakage: bool = True):
        self.cfg = cfg
        self.captions_dir = captions_dir or cfg.imagenet_captions_dir
        self.apply_leakage = apply_leakage

    def build(self) -> Dataset:
        cfg = self.cfg
        if not self.captions_dir or not cfg.imagenet_image_dir:
            raise RuntimeError("imagenet backend needs imagenet_captions_dir and imagenet_image_dir")
        if cfg.imagenet_group and not cfg.imagenet_setting_labels_path:
            raise RuntimeError("imagenet_group needs imagenet_setting_labels_path")
        class_groups = load_class_groups(cfg.imagenet_setting_labels_path) if cfg.imagenet_group else None
        excluded_row_ids = (
            load_leaked_row_ids(cfg.imagenet_leakage_path) if cfg.imagenet_leakage_path and self.apply_leakage else ()
        )
        samples = ImageNetCaptionDataset(
            captions_dir=self.captions_dir,
            image_dir=cfg.imagenet_image_dir,
            group=cfg.imagenet_group,
            class_groups=class_groups,
            excluded_row_ids=excluded_row_ids,
            max_rows=cfg.imagenet_max_rows,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.t5_name,
            token=read_token(cfg.hf_token_file),
            model_max_length=cfg.prompt_length,
        )
        return ImageNetTrainingSamples(samples, tokenizer, cfg.image_size, cfg.prompt_length)


class ImageNetPristineDatasetProvider:
    """One group's ("safe" or "unsafe") decomposed-caption dataset; the pristine trainer builds one per group."""

    def __init__(self, cfg: TrainConfig, group: str, captions_dir: str | None = None, apply_leakage: bool = True):
        self.cfg = cfg
        self.group = group
        self.captions_dir = captions_dir or cfg.imagenet_captions_dir
        self.apply_leakage = apply_leakage

    def build(self) -> Dataset:
        cfg = self.cfg
        if not self.captions_dir or not cfg.imagenet_image_dir or not cfg.imagenet_setting_labels_path:
            raise RuntimeError(
                "imagenet_pristine backend needs imagenet_captions_dir, imagenet_image_dir "
                "and imagenet_setting_labels_path"
            )
        excluded_row_ids = (
            load_leaked_row_ids(cfg.imagenet_leakage_path) if cfg.imagenet_leakage_path and self.apply_leakage else ()
        )
        samples = ImageNetDecomposedCaptionDataset(
            captions_dir=self.captions_dir,
            image_dir=cfg.imagenet_image_dir,
            group=self.group,
            class_groups=load_class_groups(cfg.imagenet_setting_labels_path),
            excluded_row_ids=excluded_row_ids,
            max_rows=cfg.imagenet_max_rows,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.t5_name,
            token=read_token(cfg.hf_token_file),
            model_max_length=cfg.prompt_length,
        )
        return ImageNetPristineTrainingSamples(samples, tokenizer, cfg.image_size, cfg.prompt_length)





def provider_for_config(cfg: TrainConfig) -> DatasetProvider:
    if cfg.dataset_backend == "local_folder":
        return LocalFolderDatasetProvider(cfg)
    if cfg.dataset_backend == "finetune_wds":
        return FinetuneWDSDatasetProvider(cfg)
    if cfg.dataset_backend == "imagenet":
        return ImageNetDatasetProvider(cfg)
    if cfg.dataset_backend == "imagenet_pristine":
        raise ValueError("imagenet_pristine has one loader per group; use make_pristine_loaders")
    raise ValueError(
        f"unknown dataset_backend={cfg.dataset_backend!r}; supported backends are "
        "'local_folder', 'finetune_wds', 'imagenet' and 'imagenet_pristine'"
    )
