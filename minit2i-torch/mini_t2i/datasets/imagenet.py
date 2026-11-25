from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Literal

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

Group = Literal["safe", "unsafe"]


def read_caption_rows(captions_dir: str | Path) -> list[dict[str, Any]]:
    files = sorted(Path(captions_dir).rglob("captions_merged.jsonl"))
    if not files:
        raise FileNotFoundError(f"no captions_merged.jsonl under {captions_dir}")
    rows: list[dict[str, Any]] = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def load_class_groups(setting_labels_path: str | Path) -> dict[int, Group]:
    """ImageNet class id -> "safe" / "unsafe", from a setting labels file."""
    doc = json.loads(Path(setting_labels_path).read_text(encoding="utf-8"))
    return {int(class_id): group for class_id, group in doc["class_id_to_label"].items()}


def load_leaked_row_ids(leakage_path: str | Path) -> set[int]:
    """All leaked global_row_index values from a leakage file ({"safe": [...], "unsafe": [...]})."""
    doc = json.loads(Path(leakage_path).read_text(encoding="utf-8"))
    return {int(row_id) for row_ids in doc.values() for row_id in row_ids}


class ParquetImageReader:
    """Reads one image by (shard_file, row_in_shard) from ImageNet parquet shards."""

    def __init__(self, image_dir: str | Path):
        self.image_dir = Path(image_dir)
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"image_dir not found: {self.image_dir}")
        # Opened lazily so each DataLoader worker holds its own file handles.
        self._open_shards: dict[str, Any] = {}

    def _shard(self, shard_file: str):
        shard = self._open_shards.get(shard_file)
        if shard is None:
            import pyarrow.parquet as pq

            shard = pq.ParquetFile(self.image_dir / shard_file, memory_map=True)
            self._open_shards[shard_file] = shard
        return shard

    def read(self, shard_file: str, row_in_shard: int) -> Image.Image:
        shard = self._shard(shard_file)
        offset = row_in_shard
        for group_idx in range(shard.metadata.num_row_groups):
            group_rows = shard.metadata.row_group(group_idx).num_rows
            if offset < group_rows:
                break
            offset -= group_rows
        else:
            raise IndexError(f"row_in_shard={row_in_shard} out of range for {shard_file}")
        image = shard.read_row_group(group_idx, columns=["image"]).slice(offset, 1).to_pylist()[0]["image"]
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        return image.convert("RGB")


class ImageNetCaptionDataset(Dataset):
    """ImageNet images paired with captions from captions_merged.jsonl files.

    Each caption row points at its image via (shard_file, row_in_shard) and is
    identified by global_row_index, which is what excluded_row_ids refers to.
    """

    def __init__(
        self,
        captions_dir: str | Path,
        image_dir: str | Path,
        group: Group | None = None,
        class_groups: dict[int, Group] | None = None,
        excluded_row_ids: Iterable[int] = (),
        max_rows: int | None = None,
    ):
        rows = read_caption_rows(captions_dir)
        if group is not None:
            if class_groups is None:
                raise ValueError("group filtering needs class_groups (see load_class_groups)")
            rows = [r for r in rows if class_groups.get(int(r["label"])) == group]
        excluded = set(excluded_row_ids)
        if excluded:
            rows = [r for r in rows if int(r["global_row_index"]) not in excluded]
        if not rows:
            raise ValueError(f"no rows left in {captions_dir} after filtering (group={group!r})")
        self.rows = rows[:max_rows] if max_rows else rows
        self.images = ParquetImageReader(image_dir)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "image": self.images.read(row["shard_file"], int(row["row_in_shard"])),
            "caption": row["captions"]["child_caption"],
            "class_id": int(row["label"]),
            "row_id": int(row["global_row_index"]),
        }


class ImageNetDecomposedCaptionDataset(ImageNetCaptionDataset):
    """ImageNet images paired with decomposed_prompts rows: subject_caption names the class,
    residual_caption describes everything else with the subject anonymized."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "image": self.images.read(row["shard_file"], int(row["row_in_shard"])),
            "subject_caption": row["subject_caption"],
            "residual_caption": row["residual_caption"],
            "class_id": int(row["label"]),
            "row_id": int(row["global_row_index"]),
        }


class ImageNetTrainingSamples(Dataset):
    """Turns (image, caption) samples into the tensors the training loop expects."""

    def __init__(self, samples: ImageNetCaptionDataset, tokenizer, image_size: int, prompt_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.PILToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        tokens = self.tokenizer(
            sample["caption"],
            max_length=self.prompt_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": self.image_transform(sample["image"]),
            "input_ids": tokens["input_ids"][0].long(),
            "attention_mask": tokens["attention_mask"][0].long(),
            "caption": sample["caption"],
            "source": "imagenet",
        }


class ImageNetPristineTrainingSamples(ImageNetTrainingSamples):
    """Decomposed samples -> tensors for the pristine model: subject feeds the mode stream, residual the shared stream."""

    def _tokenize(self, text: str) -> tuple[Any, Any]:
        tokens = self.tokenizer(
            text,
            max_length=self.prompt_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return tokens["input_ids"][0].long(), tokens["attention_mask"][0].long()

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        input_ids_mode, attention_mask_mode = self._tokenize(sample["subject_caption"])
        input_ids_shared, attention_mask_shared = self._tokenize(sample["residual_caption"])
        return {
            "pixel_values": self.image_transform(sample["image"]),
            "input_ids_mode": input_ids_mode,
            "attention_mask_mode": attention_mask_mode,
            "input_ids_shared": input_ids_shared,
            "attention_mask_shared": attention_mask_shared,
            "caption": sample["subject_caption"],
            "shared_caption": sample["residual_caption"],
            "source": "imagenet_pristine",
        }
