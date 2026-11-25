from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


SOURCE_REVISION = "d626240be2538720e83103a0e1178d24aca8b12c"
ARCHIVES = {
    "matched-frequency": "f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c",
    "threshold0.7": "32142bfe998f9cfa4141be691f6a80eeb1e27b24bd532a668eb08c26d98897a2",
    "top-images": "85e4ace30f3f0d660152062254b8759dda66267b27f7ec0fd8d5768f1073a04b",
}
NUM_CLASSES = 1000
IMAGES_PER_CLASS = 10
ROWS_PER_SHARD = 1000
SCHEMA = pa.schema([
    ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("label", pa.int64()),
])


def verify_archive(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"Archive checksum mismatch for {path}: expected {expected}, got {actual}. "
            f"Download vaishaal/ImageNetV2 at revision {SOURCE_REVISION}."
        )


def repackage_archive(path: Path, variant: str, output: Path) -> None:
    expected_count = NUM_CLASSES * IMAGES_PER_CLASS
    shard_count = expected_count // ROWS_PER_SHARD
    counts: Counter[int] = Counter()
    seen: set[str] = set()
    rows = []
    total = 0

    # Archive encounter order, not filename sorting, defines the caption row IDs.
    # The upstream .tar.gz files are uncompressed tar; detect compression by content.
    with tarfile.open(path, mode="r|*") as archive:
        for member in archive:
            if member.isdir():
                continue
            parts = PurePosixPath(member.name).parts
            if (
                not member.isfile()
                or len(parts) != 3
                or parts[0] != f"imagenetv2-{variant}-format-val"
                or not parts[1].isdigit()
                or not parts[2].endswith(".jpeg")
            ):
                raise ValueError(f"Unexpected archive entry: {member.name}")
            label = int(parts[1])
            if not 0 <= label < NUM_CLASSES or member.name in seen:
                raise ValueError(f"Invalid label or duplicate image: {member.name}")
            seen.add(member.name)
            counts[label] += 1
            if counts[label] > IMAGES_PER_CLASS or total >= expected_count:
                raise ValueError(f"Too many images in {path}: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read image: {member.name}")
            with stream:
                image_bytes = stream.read()
            if len(image_bytes) != member.size or not image_bytes:
                raise ValueError(f"Empty or incomplete image: {member.name}")
            rows.append({"image": {"bytes": image_bytes, "path": parts[2]}, "label": label})
            total += 1
            if len(rows) == ROWS_PER_SHARD:
                shard = total // ROWS_PER_SHARD - 1
                target = output / f"{variant}-{shard:05d}-of-{shard_count:05d}.parquet"
                pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), target)
                rows.clear()
                print(f"Wrote {target.name}", flush=True)

    if (
        total != expected_count
        or rows
        or counts != Counter({label: IMAGES_PER_CLASS for label in range(NUM_CLASSES)})
    ):
        raise ValueError(f"Incomplete {variant}: expected {expected_count} images, got {total}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Recreate caption-compatible ImageNet-V2 parquet shards from original archives."
    )
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New output directory; must not exist.")
    parser.add_argument(
        "--delete-archives", action="store_true",
        help="Delete only the three input archives after successful conversion; keep them on conversion failure.",
    )
    args = parser.parse_args(argv)
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    for variant, checksum in ARCHIVES.items():
        source = args.archive_dir / f"imagenetv2-{variant}.tar.gz"
        print(f"Verifying {source.name}", flush=True)
        verify_archive(source, checksum)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".imagenet-v2-", dir=output.parent) as temporary:
        staging = Path(temporary) / "data"
        staging.mkdir()
        for variant in ARCHIVES:
            repackage_archive(args.archive_dir / f"imagenetv2-{variant}.tar.gz", variant, staging)
        if output.exists():
            raise FileExistsError(f"Output appeared during conversion; refusing overwrite: {output}")
        staging.rename(output)
    print(f"Complete: 30,000 images in 30 shards at {output}", flush=True)
    if args.delete_archives:
        for variant in ARCHIVES:
            source = args.archive_dir / f"imagenetv2-{variant}.tar.gz"
            source.unlink()
            print(f"Deleted input archive: {source}", flush=True)


if __name__ == "__main__":
    main()
