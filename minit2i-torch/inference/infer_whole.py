#!/usr/bin/env python3
"""Whole-caption inference with MMJiTB32Text2 over the ImageNet eval benchmarks.

Loads Safe and Full checkpoints. Images are written as
    <output_dir>/<benchmark>[/<source>]/<global_row_index:06d>.png
and existing PNGs are skipped, so re-running the same command resumes.

One GPU:
    python inference/infer_whole.py --checkpoint ckpt.pt --output_dir runs/my_model

All 8 GPUs (each rank takes every 8th row):
    for i in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$i python inference/infer_whole.py --checkpoint ckpt.pt \
            --output_dir runs/my_model --rank $i --world_size 8 &
    done; wait
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mini_t2i.config import TrainConfig
from mini_t2i.diffusion import euler_sample
from mini_t2i.settings import IMAGENET_CAPTIONS_ROOT
from mini_t2i.train import build_model
from mini_t2i.utils import amp_dtype

BENCHMARKS = ("imagenet", "clearbg", "compositional", "implicit")


# ---------------------------------------------------------------------------
# Data: every benchmark becomes rows of {"subdir", "global_row_index", "prompt"}
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_benchmark(name: str, data_root: Path, imagenet_file: Path) -> list[dict]:
    if name == "imagenet":
        rows = _read_jsonl(imagenet_file)
        out = [{"subdir": f"imagenet/{r['source']}", "global_row_index": r["global_row_index"],
                "prompt": r["whole_caption"]} for r in rows]
    elif name == "clearbg":
        rows = _read_jsonl(data_root / "clearbg" / "captions_merged.jsonl")
        out = [{"subdir": "clearbg", "global_row_index": r["global_row_index"],
                "prompt": r["captions"]["child_caption"]} for r in rows]
    elif name == "compositional":
        rows = _read_jsonl(data_root / "compositional" / "benchmark_paper.jsonl")
        out = [{"subdir": "compositional", "global_row_index": r["global_row_index"],
                "prompt": r["whole_prompt"]} for r in rows]
    elif name == "implicit":
        implicit_dir = data_root / "implicit" / "whole_prompts" / "val"
        files = sorted(implicit_dir.rglob("captions_merged.jsonl"))
        if not files:
            raise FileNotFoundError(f"No implicit dataset found: expected captions_merged.jsonl under {implicit_dir}")
        rows = [r for f in files for r in _read_jsonl(f)]
        if not rows:
            raise ValueError(f"Implicit dataset contains no caption rows: {implicit_dir}")
        out = [{"subdir": "implicit", "global_row_index": r["global_row_index"],
                "prompt": r["captions"]["child_caption"]} for r in rows]
    else:
        raise ValueError(f"unknown benchmark {name!r}; choose from {BENCHMARKS}")
    return sorted(out, key=lambda r: (r["subdir"], r["global_row_index"]))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(path: str, weights: str) -> tuple[torch.nn.Module, TrainConfig, int | str]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if weights not in ckpt:
        raise KeyError(f"checkpoint has no '{weights}' weights (has: {[k for k in ckpt if k in ('model', 'ema')]})")
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt[weights].items()}
    if any("_safe." in k for k in state):
        raise ValueError("pristine checkpoint: it has no whole-caption path, use infer_pristine_split.py")

    known = TrainConfig.__dataclass_fields__
    cfg = TrainConfig().update_from_mapping({k: v for k, v in ckpt.get("config", {}).items() if k in known})
    model = build_model(cfg)
    model.load_state_dict(state)
    return model, cfg, ckpt.get("step", "?")


def encode(prompts: list[str], tokenizer, encoder, prompt_length: int, device, dtype):
    tok = tokenizer(prompts, max_length=prompt_length, padding="max_length", truncation=True, return_tensors="pt")
    ids, mask = tok["input_ids"].to(device), tok["attention_mask"].to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
        out = encoder(input_ids=ids, attention_mask=mask)
    return out.last_hidden_state.to(dtype), mask


def init_noise(gidxs: list[int], seed: int, image_size: int, noise_scale: float, device, dtype) -> torch.Tensor:
    """Noise depends only on seed + global_row_index, so every model sees the same noise per prompt."""
    x = torch.empty(len(gidxs), 3, image_size, image_size, device=device, dtype=dtype)
    gen = torch.Generator(device=device)
    for k, gi in enumerate(gidxs):
        gen.manual_seed(seed + int(gi))
        x[k] = torch.randn(3, image_size, image_size, device=device, dtype=dtype, generator=gen) * noise_scale
    return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _is_complete_png(path: Path, image_size: int) -> bool:
    try:
        with Image.open(path) as image:
            return image.format == "PNG" and image.size == (image_size, image_size)
    except (OSError, SyntaxError, ValueError):
        return False


def _save_png_atomic(image: Image.Image, path: Path) -> None:
    temporary_file = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    temporary_path = Path(temporary_file.name)
    try:
        with temporary_file:
            image.save(temporary_file, format="PNG")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="local .pt file")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    ap.add_argument("--imagenet_file", default=str(IMAGENET_CAPTIONS_ROOT / "settings" / "Dog" / "equalized_val.jsonl"),
                    help="jsonl with source, global_row_index, whole_caption per row")
    ap.add_argument("--data_root", default=str(IMAGENET_CAPTIONS_ROOT))
    ap.add_argument("--weights", choices=["model", "ema"], default="model")
    ap.add_argument("--cfg_scale", type=float, default=None, help="default: the checkpoint's cfg_scale")
    ap.add_argument("--n_T", type=int, default=None, help="default: the checkpoint's n_T")
    ap.add_argument("--seed", type=int, default=42, help="noise seed per image = seed + global_row_index")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    args = ap.parse_args()

    if args.world_size <= 0:
        ap.error("--world_size must be greater than zero")
    if not 0 <= args.rank < args.world_size:
        ap.error("--rank must satisfy 0 <= rank < world_size")
    if args.batch_size <= 0:
        ap.error("--batch_size must be greater than zero")

    device = torch.device("cuda")
    tag = f"rank{args.rank}"
    out_root = Path(args.output_dir)

    model, cfg, step = load_model(args.checkpoint, args.weights)
    dtype = amp_dtype(cfg.amp_dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = model.to(device).eval()
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg.cfg_scale
    n_T = args.n_T if args.n_T is not None else cfg.n_T
    tqdm.write(f"[{tag}] step={step} weights={args.weights} cfg={cfg_scale} n_T={n_T} seed={args.seed} "
               f"size={cfg.image_size} noise_scale={cfg.noise_scale} amp_dtype={cfg.amp_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.t5_name)
    encoder = T5EncoderModel.from_pretrained(cfg.t5_name).eval().to(device)
    encoder.requires_grad_(False)

    for name in args.benchmarks:
        rows = load_benchmark(name, Path(args.data_root), Path(args.imagenet_file))[args.rank :: args.world_size]
        todo = [r for r in rows if not _is_complete_png(out_root / r["subdir"] / f"{r['global_row_index']:06d}.png", cfg.image_size)]
        tqdm.write(f"[{tag}] {name}: {len(rows)} rows on this rank, {len(todo)} to generate")
        for subdir in {r["subdir"] for r in todo}:
            (out_root / subdir).mkdir(parents=True, exist_ok=True)

        for start in tqdm(range(0, len(todo), args.batch_size), desc=f"{tag} {name}", position=args.rank, dynamic_ncols=True):
            batch = todo[start : start + args.batch_size]
            gidxs = [r["global_row_index"] for r in batch]
            text, mask = encode([r["prompt"] for r in batch], tokenizer, encoder, cfg.prompt_length, device, dtype)
            noise = init_noise(gidxs, args.seed, cfg.image_size, cfg.noise_scale, device, dtype)
            with torch.no_grad():
                samples = euler_sample(model, text, mask, image_size=cfg.image_size, steps=n_T,
                                       cfg_scale=cfg_scale, noise_scale=cfg.noise_scale, init_noise=noise)
            for r, img in zip(batch, samples):
                arr = ((img.float().cpu() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
                _save_png_atomic(Image.fromarray(arr), out_root / r["subdir"] / f"{r['global_row_index']:06d}.png")

    tqdm.write(f"[{tag}] done.")


if __name__ == "__main__":
    main()
