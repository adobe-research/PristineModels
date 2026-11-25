"""Generate images for DPG-Bench, PRISM-Bench, or the safety benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

from .datasets import PROMPTS, SAFETY_PROMPTS, read_jsonl


def load_prompts(benchmark: str, split: str, limit: int | None):
    from mini_t2i.scripts.generate_benchmark_images import PromptItem

    path = SAFETY_PROMPTS if benchmark == "safety" else PROMPTS / f"{benchmark}.jsonl"
    items = []
    for position, row in enumerate(read_jsonl(path)):
        if split != "all" and row.get("split") != split:
            continue
        if benchmark == "safety":
            index, item_id = row["global_index"], row["key"]
        else:
            index, item_id = position, row["item_id"]
            if benchmark == "prism":
                item_id = "/".join(item_id.rsplit("_", 1))
        items.append((position, PromptItem(index, item_id, row["prompt"])))
    return items[:limit]


def load_model(checkpoint: Path, weights: str):
    import torch

    from mini_t2i.config import TrainConfig
    from mini_t2i.train import build_model

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    known = TrainConfig.__dataclass_fields__
    cfg = TrainConfig().update_from_mapping({k: v for k, v in ckpt["config"].items() if k in known})
    cfg.attention_impl = "sdpa"
    model = build_model(cfg)
    # Pristine's paper eval uses the plain whole-caption path, without the unsafe branch.
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt[weights].items()}
    model.load_state_dict({
        k: v for k, v in state.items() if not k.rsplit(".", 1)[0].endswith("_unsafe")
    })
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("dpg", "prism", "safety"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--split", choices=("all", "safe", "unsafe"), default="all")
    parser.add_argument("--batch-size", type=int, default=256, help="PRISM or safety batch size; DPG uses four crops per prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.benchmark == "safety" and args.split != "all":
        parser.error("Only DPG and PRISM have safe/unsafe splits")
    if args.batch_size < 1 or args.steps < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size, --steps and --limit must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    import torch
    import torch.distributed as dist
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from inference.infer_whole import _is_complete_png, _save_png_atomic, init_noise
    from mini_t2i.diffusion import euler_sample
    from mini_t2i.scripts.generate_benchmark_images import (
        generate_for_prompt, init_dist, make_grid, tensor_to_pil,
    )
    from mini_t2i.train import build_text_encoder, encode_text
    from mini_t2i.utils import amp_dtype, rank, read_token, seed_all, world_size

    items = load_prompts(args.benchmark, args.split, args.limit)
    device = init_dist()
    try:
        seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model, cfg = load_model(args.checkpoint, args.weights)
        cfg.n_T, cfg.cfg_scale = args.steps, args.cfg_scale
        dpg = args.benchmark == "dpg"
        size = cfg.image_size * (2 if dpg else 1)
        batch_size = 1 if dpg else args.batch_size
        todo = [
            item for position, item in items
            if position % world_size() == rank()
            and not _is_complete_png(args.outdir / f"{item.item_id}.png", size)
        ]
        if not todo:
            return
        dtype = amp_dtype(cfg.amp_dtype)
        model.to(device, dtype=dtype).eval()
        encoder = build_text_encoder(cfg, device)
        tokenizer = AutoTokenizer.from_pretrained(cfg.t5_name, token=read_token(cfg.hf_token_file),
                                                  model_max_length=cfg.prompt_length)
        for start in tqdm(range(0, len(todo), batch_size), desc=f"{args.benchmark} rank {rank()}"):
            batch = todo[start : start + batch_size]
            if dpg:
                images = generate_for_prompt(
                    model, encoder, tokenizer, cfg, batch[0], device, dtype,
                    samples_per_prompt=4, batch_size=4, seed=args.seed,
                )
                images = [make_grid(images, cfg.image_size)]
            else:
                tok = tokenizer([item.prompt for item in batch], max_length=cfg.prompt_length,
                                padding="max_length", truncation=True, return_tensors="pt")
                mask = tok["attention_mask"].to(device)
                text = encode_text(encoder, tok["input_ids"].to(device), mask, dtype)
                noise = init_noise([item.index * 1000 for item in batch], args.seed,
                                   cfg.image_size, cfg.noise_scale, device, dtype)
                images = tensor_to_pil(euler_sample(
                    model, text, mask, cfg.image_size, steps=cfg.n_T, cfg_scale=cfg.cfg_scale,
                    noise_scale=cfg.noise_scale, init_noise=noise,
                ))
            for item, image in zip(batch, images):
                path = args.outdir / f"{item.item_id}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                _save_png_atomic(image, path)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
