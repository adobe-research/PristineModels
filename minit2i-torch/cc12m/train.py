from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, T5EncoderModel

from cc12m.data import check_training_data as check_cc12m_training_data
from cc12m.data import make_cc12m_loader
from mini_t2i.config import TrainConfig
from mini_t2i.data import check_training_data, endless_batches, local_batches, make_loader
from mini_t2i.diffusion import training_loss, euler_sample
from mini_t2i.model import MMJiTB32Text2
from mini_t2i.settings import WANDB_API_KEY_FILE
from mini_t2i.utils import (
    amp_dtype,
    atomic_torch_save,
    is_main,
    jax_warmup_constant_lr,
    read_token,
    seed_all,
    update_ema,
    world_size,
)
from mini_t2i.visualize import save_grid


def init_dist() -> tuple[torch.device, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl", timeout=timedelta(hours=6))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        local_rank = 0
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank), local_rank


def build_model(cfg: TrainConfig) -> MMJiTB32Text2:
    return MMJiTB32Text2(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        hidden_size=cfg.hidden_size,
        t5_hidden_size=cfg.t5_hidden_size,
        depth_double=cfg.depth_double,
        text_preamble_depth=cfg.text_preamble_depth,
        num_heads=cfg.num_heads,
        head_dim=cfg.head_dim,
        mlp_ratio=cfg.mlp_ratio,
        pca_channels=cfg.pca_channels,
        final_layer_zero=cfg.final_layer_zero,
        rms_affine=not cfg.compat_checkpoint_arch,
        text_qk_norm=not cfg.compat_checkpoint_arch,
        double_qk_norm=not cfg.compat_checkpoint_arch,
        rope_style="compat" if cfg.compat_checkpoint_arch else "jax",
        attention_impl=cfg.attention_impl,
    )


def build_text_encoder(cfg: TrainConfig, device: torch.device):
    token = read_token(cfg.hf_token_file)
    enc = T5EncoderModel.from_pretrained(cfg.t5_name, token=token)
    enc.eval().to(device)
    if cfg.freeze_t5:
        for p in enc.parameters():
            p.requires_grad_(False)
    return enc


def prepare_images(pixel_values: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = pixel_values.to(device, non_blocking=True)
    if images.dtype == torch.uint8:
        images = images.to(dtype=torch.float32).mul_(1.0 / 127.5).add_(-1.0)
    return images


@torch.no_grad()
def encode_text(text_encoder, input_ids, attention_mask, dtype: torch.dtype) -> torch.Tensor:
    out = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    return out.last_hidden_state.to(dtype=dtype)


def maybe_wandb(cfg: TrainConfig):
    if not is_main() or os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        if "WANDB_API_KEY" not in os.environ:
            key_path = WANDB_API_KEY_FILE
            if Path(key_path).exists():
                os.environ["WANDB_API_KEY"] = Path(key_path).read_text().strip()
        import wandb

        return wandb.init(project=cfg.project, entity=cfg.wandb_entity, name=cfg.run_name, config=cfg.to_dict())
    except Exception as e:
        print(f"[wandb] disabled: {e}", flush=True)
        return None


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    model = model.module if isinstance(model, DDP) else model
    return getattr(model, "_orig_mod", model)


def save_checkpoint(cfg, model, ema_model, opt, step, latest: bool = False):
    """Saves checkpoint_{step}.pt (pruned to save_keep), or overwrites checkpoint_latest.pt when latest=True."""
    if not is_main():
        return
    src = unwrap_model(model)
    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / ("checkpoint_latest.pt" if latest else f"checkpoint_{step:06d}.pt")
    atomic_torch_save(
        {
            "step": step,
            "model": src.state_dict(),
            "ema": unwrap_model(ema_model).state_dict(),
            "optimizer": opt.state_dict(),
            "config": cfg.to_dict(),
        },
        path,
    )
    if latest:
        (ckpt_dir / "checkpoint_latest.txt").write_text(f"{step}\n")
        return
    if cfg.save_keep > 0:
        ckpts = sorted(ckpt_dir.glob("checkpoint_[0-9]*.pt"))
        for old in ckpts[:-cfg.save_keep]:
            old.unlink(missing_ok=True)


def latest_checkpoint(cfg: TrainConfig) -> Path | None:
    """Newest of checkpoint_latest.pt and the highest-step checkpoint_{step}.pt."""
    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    numbered = sorted(ckpt_dir.glob("checkpoint_[0-9]*.pt"))
    best_path = numbered[-1] if numbered else None
    best_step = int(best_path.stem.split("_")[1]) if best_path else -1
    latest_pt, latest_txt = ckpt_dir / "checkpoint_latest.pt", ckpt_dir / "checkpoint_latest.txt"
    if latest_pt.exists() and latest_txt.exists() and int(latest_txt.read_text().strip()) > best_step:
        return latest_pt
    return best_path


def resolve_resume_checkpoint(cfg: TrainConfig) -> Path | None:
    if cfg.resume_from:
        if cfg.resume_from == "latest":
            return latest_checkpoint(cfg)
        return Path(cfg.resume_from)
    if cfg.auto_resume:
        return latest_checkpoint(cfg)
    return None


def load_checkpoint_if_available(cfg, model, ema_model, device) -> tuple[int, dict | None]:
    finetune = cfg.dataset_backend == "finetune_wds" and cfg.finetune_metadata_dir
    path = resolve_resume_checkpoint(cfg)
    if path is None:
        if finetune:
            raise ValueError("Fine-tuning requires a local pretrain or fine-tune checkpoint")
        return 0, None
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "ema" not in ckpt:
        raise ValueError(f"resume checkpoint has no EMA weights: {path}")
    step = int(ckpt["step"])
    optimizer = ckpt.get("optimizer")
    if finetune:
        if not 0 <= cfg.finetune_start_step <= step <= cfg.num_steps or optimizer is None:
            raise ValueError(f"{path}: fine-tuning requires Adam state and a step within the fine-tune stage")
        previous = ckpt.get("config") or {}
        if step > cfg.finetune_start_step and (previous.get("dataset_backend"), previous.get("finetune_start_step")) != (
            "finetune_wds", cfg.finetune_start_step
        ):
            raise ValueError(f"{path}: start from step {cfg.finetune_start_step} or resume this fine-tune stage")
    for name, target in (("model", model), ("ema", ema_model)):
        state = {key.removeprefix("_orig_mod."): value for key, value in ckpt[name].items()}
        unwrap_model(target).load_state_dict(state, strict=True)
    if is_main():
        print(f"[resume] loaded {path} at step={step}", flush=True)
    return step, optimizer


@torch.no_grad()
def run_visualize_captions(cfg, model, text_encoder, tokenizer, captions, device, dtype, step, name, wb=None):
    captions = list(captions)
    tok = tokenizer(
        captions,
        max_length=cfg.prompt_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tok["input_ids"].to(device)
    mask = tok["attention_mask"].to(device)
    text = encode_text(text_encoder, input_ids, mask, dtype)
    src = unwrap_model(model)
    was_training = src.training
    try:
        samples = euler_sample(
            src, text, mask, cfg.image_size, steps=cfg.n_T, cfg_scale=cfg.cfg_scale, noise_scale=cfg.noise_scale
        )
    finally:
        src.train(was_training)
    save_grid(samples, str(Path(cfg.output_dir) / "samples" / f"{name}_{step:06d}.png"), nrow=min(4, len(captions)))
    wandb_log_images(wb, name, samples, captions, step)


def wandb_log_images(wb, key: str, images: torch.Tensor, captions: list[str], step: int):
    """Logs each image as `{key}/{i}` with its caption; accepts uint8 or [-1, 1] float images."""
    if wb is None:
        return
    try:
        import wandb

        if images.dtype != torch.uint8:
            images = ((images.detach().float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        wb.log(
            {
                f"{key}/{i}": wandb.Image(img.permute(1, 2, 0).cpu().numpy(), caption=cap)
                for i, (img, cap) in enumerate(zip(images, captions))
            },
            step=step,
        )
    except Exception as e:
        print(f"[wandb] image log failed for {key}: {e}", flush=True)


def train(cfg: TrainConfig):
    cfg.resolve()
    is_cc12m_flash = cfg.dataset_backend == "cc12m_flash_s3"
    is_finetune = cfg.dataset_backend == "finetune_wds" and bool(cfg.finetune_metadata_dir)
    if is_finetune:
        loader = make_loader(cfg)
    elif is_cc12m_flash:
        check_cc12m_training_data(cfg)
    else:
        check_training_data(cfg)
    device, local_rank = init_dist()
    seed_all(cfg.seed)
    dtype = amp_dtype(cfg.amp_dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    text_encoder = build_text_encoder(cfg, device)
    model = build_model(cfg).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    start_step, opt_state = load_checkpoint_if_available(cfg, model, ema_model, device)
    if cfg.compile_model:
        model = torch.compile(model)
    if world_size() > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.learning_rate,
        betas=(0.9, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    wb = maybe_wandb(cfg)

    if not is_finetune:
        loader = make_cc12m_loader(cfg) if is_cc12m_flash else make_loader(cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.t5_name, token=read_token(cfg.hf_token_file), model_max_length=cfg.prompt_length
    )

    start = time.time()
    last_log_time = start
    last_log_step = start_step
    accum_loss = 0.0
    accum_loss_tensor = torch.zeros((), device=device)
    accum_loss_count = 0
    accum_timing = {k: 0.0 for k in ("data_s", "h2d_s", "t5_s", "train_s", "opt_s")}
    opt.zero_grad(set_to_none=True)
    if cfg.dataset_backend == "local_folder" or is_finetune:
        data_step = start_step - cfg.finetune_start_step if is_finetune else start_step
        data_iter = local_batches(loader, data_step * cfg.grad_accum_steps)
    else:
        data_iter = endless_batches(loader, start_step=start_step)
    for step in range(start_step, cfg.num_steps):
        step_t_data = 0.0
        step_t_h2d = 0.0
        step_t_t5 = 0.0
        step_t_train = 0.0
        for accum in range(cfg.grad_accum_steps):
            t0 = time.time()
            batch = next(data_iter)
            t1 = time.time()
            images = prepare_images(batch["pixel_values"], device)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            t2 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                text = encode_text(text_encoder, input_ids, mask, dtype)
            t3 = time.time()
            sync_context = (
                model.no_sync()
                if hasattr(model, "no_sync") and accum < cfg.grad_accum_steps - 1
                else contextlib.nullcontext()
            )
            with sync_context:
                with torch.autocast("cuda", dtype=dtype):
                    loss, metrics, _ = training_loss(
                        model,
                        images,
                        text,
                        mask,
                        label_drop_rate=cfg.label_drop_rate,
                        t_lognorm_mu=cfg.t_lognorm_mu,
                        t_lognorm_sigma=cfg.t_lognorm_sigma,
                        noise_scale=cfg.noise_scale,
                    )
                    loss = loss / cfg.grad_accum_steps
                loss.backward()
            t4 = time.time()
            step_t_data += t1 - t0
            step_t_h2d += t2 - t1
            step_t_t5 += t3 - t2
            step_t_train += t4 - t3
            if is_main() and cfg.defer_loss_sync:
                accum_loss_tensor = accum_loss_tensor + metrics["loss"].detach()
                accum_loss_count += 1
            elif is_main():
                accum_loss += float(metrics["loss"].detach().cpu())
                accum_loss_count += 1

        opt_t0 = time.time()
        lr = jax_warmup_constant_lr(step, cfg.num_steps, cfg.warmup_steps, cfg.learning_rate)
        for group in opt.param_groups:
            group["lr"] = lr
        parameters = [p for p in model.parameters() if p.grad is not None]
        if cfg.max_grad_norm and cfg.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm)
        else:
            grad_norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(p.grad.detach(), 2) for p in parameters]),
                2,
            ) if parameters else torch.zeros((), device=device)
        opt.step()
        opt.zero_grad(set_to_none=True)
        update_ema(ema_model, unwrap_model(model), cfg.ema_decay)
        torch.cuda.synchronize(device)
        opt_t1 = time.time()
        accum_timing["data_s"] += step_t_data
        accum_timing["h2d_s"] += step_t_h2d
        accum_timing["t5_s"] += step_t_t5
        accum_timing["train_s"] += step_t_train
        accum_timing["opt_s"] += opt_t1 - opt_t0

        if is_main() and (step + 1) % cfg.log_every == 0:
            now = time.time()
            elapsed = now - start
            steps_this_run = step + 1 - start_step
            interval_steps = step + 1 - last_log_step
            interval_elapsed = now - last_log_time
            sps = steps_this_run / max(elapsed, 1e-6)
            log = {
                "loss": (float(accum_loss_tensor.detach().cpu()) if cfg.defer_loss_sync else accum_loss)
                / max(accum_loss_count, 1),
                "lr": lr,
                "grad_norm": float(grad_norm),
                "steps_per_second": sps,
                "interval_steps_per_second": interval_steps / max(interval_elapsed, 1e-6),
                "elapsed_s": elapsed,
                "step": step + 1,
            }
            for key, value in accum_timing.items():
                log[key] = value / max(interval_steps, 1)
            print(json.dumps(log), flush=True)
            if wb:
                wb.log(log, step=step + 1)
            accum_loss = 0.0
            accum_loss_tensor = torch.zeros((), device=device)
            accum_loss_count = 0
            accum_timing = {k: 0.0 for k in accum_timing}
            last_log_time = now
            last_log_step = step + 1

        is_ckpt_step = cfg.ckpt_every > 0 and (step + 1) % cfg.ckpt_every == 0
        is_latest_step = cfg.latest_every > 0 and (
            (step + 1) % cfg.latest_every == 0 or step + 1 == cfg.num_steps
        )
        if is_ckpt_step or is_latest_step:
            # Hold all ranks until rank 0 has finished writing, so nobody runs ahead of a half-written file.
            if dist.is_initialized():
                dist.barrier()
            if is_ckpt_step:
                save_checkpoint(cfg, model, ema_model, opt, step + 1)
            if is_latest_step:
                save_checkpoint(cfg, model, ema_model, opt, step + 1, latest=True)
            if dist.is_initialized():
                dist.barrier()
        if is_main() and (step + 1) % cfg.log_images_every == 0:
            wandb_log_images(wb, "train_images", batch["pixel_values"][:4], batch["caption"][:4], step + 1)
        # Sampling has no collectives, so other ranks just wait for rank 0 at the next gradient sync.
        if is_main() and cfg.sample_prompts and (step + 1) % cfg.sample_every == 0:
            run_visualize_captions(
                cfg, model, text_encoder, tokenizer, cfg.sample_prompts, device, dtype, step + 1, "sample_prompts", wb
            )
            run_visualize_captions(
                cfg, ema_model, text_encoder, tokenizer, cfg.sample_prompts, device, dtype, step + 1, "sample_prompts_ema", wb
            )
    if wb:
        wb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--resume-from")
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--sync-loss-every-step", action="store_true")
    parser.add_argument("--sample-every", type=int)
    parser.add_argument("--log-images-every", type=int)
    parser.add_argument("--ckpt-every", type=int, help="Numbered checkpoint interval in steps; 0 disables.")
    parser.add_argument("--latest-every", type=int, help="Latest checkpoint interval in steps, plus run end; 0 disables.")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--dataloader-prefetch", type=int)
    parser.add_argument("--dataloader-multiprocessing-context", choices=["fork", "spawn", "forkserver"])
    parser.add_argument("--shuffle-buffer", type=int)
    parser.add_argument(
        "--dataset-backend",
        choices=[
            "local_folder",
            "imagenet",
            "imagenet_pristine",
            "cc12m_flash_s3",
        ],
    )
    parser.add_argument("--local-dataset-dir")
    parser.add_argument("--compat-checkpoint-arch", action="store_true")
    parser.add_argument("--attention-impl", choices=["einsum", "sdpa"])
    args = parser.parse_args()
    cfg = TrainConfig()
    if args.config_file:
        cfg.update_from_yaml(args.config_file)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.run_name:
        cfg.run_name = args.run_name
    if args.num_steps is not None:
        cfg.num_steps = args.num_steps
    if args.micro_batch_size is not None:
        cfg.micro_batch_size = args.micro_batch_size
    if args.grad_accum_steps is not None:
        cfg.grad_accum_steps = args.grad_accum_steps
    if args.resume_from:
        cfg.resume_from = args.resume_from
    if args.no_auto_resume:
        cfg.auto_resume = False
    if args.log_every is not None:
        cfg.log_every = args.log_every
    if args.sync_loss_every_step:
        cfg.defer_loss_sync = False
    if args.sample_every is not None:
        cfg.sample_every = args.sample_every
    if args.log_images_every is not None:
        cfg.log_images_every = args.log_images_every
    if args.ckpt_every is not None:
        cfg.ckpt_every = args.ckpt_every
    if args.latest_every is not None:
        cfg.latest_every = args.latest_every
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.dataloader_prefetch is not None:
        cfg.dataloader_prefetch = args.dataloader_prefetch
    if args.dataloader_multiprocessing_context is not None:
        cfg.dataloader_multiprocessing_context = args.dataloader_multiprocessing_context
    if args.shuffle_buffer is not None:
        cfg.shuffle_buffer = args.shuffle_buffer
    if args.dataset_backend is not None:
        cfg.dataset_backend = args.dataset_backend
    if args.local_dataset_dir is not None:
        cfg.local_dataset_dir = args.local_dataset_dir
    if args.compat_checkpoint_arch:
        cfg.compat_checkpoint_arch = True
    if args.attention_impl is not None:
        cfg.attention_impl = args.attention_impl
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    cfg.resolve()
    train(cfg)


if __name__ == "__main__":
    main()
