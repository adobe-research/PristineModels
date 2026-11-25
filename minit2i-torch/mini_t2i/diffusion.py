from __future__ import annotations

import torch
from torch import nn
from tqdm import tqdm

from .utils import is_main


def sample_lognorm(batch: int, mu: float, sigma: float, device: torch.device) -> torch.Tensor:
    z = torch.randn(batch, device=device) * sigma + mu
    return torch.sigmoid(z).clamp(1e-5, 1.0 - 1e-5)


def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    text_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    label_drop_rate: float = 0.1,
    t_lognorm_mu: float = -0.8,
    t_lognorm_sigma: float = 0.8,
    noise_scale: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    b = images.shape[0]
    device = images.device
    if label_drop_rate > 0:
        drop = torch.rand(b, device=device) < label_drop_rate
        attention_mask = torch.where(drop[:, None], torch.zeros_like(attention_mask), attention_mask)
    t = sample_lognorm(b, t_lognorm_mu, t_lognorm_sigma, device)
    noise = torch.randn_like(images) * noise_scale
    x_t = images * t[:, None, None, None] + noise * (1.0 - t[:, None, None, None])
    pred_x0 = model(x_t, t, text_embeddings, attention_mask)
    target = (images - x_t) / (1.0 - t[:, None, None, None]).clamp_min(0.05)
    v_pred = (pred_x0 - x_t) / (1.0 - t[:, None, None, None]).clamp_min(0.05)
    per_sample = (v_pred - target).pow(2).mean(dim=(1, 2, 3))
    loss = per_sample.mean()
    return loss, {"loss": loss.detach(), "loss_monitor": per_sample.mean().detach()}, None


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    text_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    image_size: int,
    steps: int = 100,
    cfg_scale: float = 2.0,
    noise_scale: float = 2.0,
    init_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    model.eval()
    b = text_embeddings.shape[0]
    device = text_embeddings.device
    if init_noise is not None:
        x = init_noise.to(device=device, dtype=text_embeddings.dtype)
    else:
        x = torch.randn(b, 3, image_size, image_size, device=device, dtype=text_embeddings.dtype) * noise_scale
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    null_mask = torch.zeros_like(attention_mask)
    if cfg_scale != 1.0:
        cfg_text = torch.cat([text_embeddings, text_embeddings], dim=0)
        cfg_mask = torch.cat([attention_mask, null_mask], dim=0)
    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        t = t0.expand(b)
        amp_dtype = text_embeddings.dtype if text_embeddings.dtype in (torch.float16, torch.bfloat16) else torch.float32
        with torch.autocast("cuda", dtype=amp_dtype, enabled=x.is_cuda and amp_dtype != torch.float32):
            if cfg_scale != 1.0:
                pred_cond, pred_uncond = model(
                    torch.cat([x, x], dim=0), t0.expand(2 * b), cfg_text, cfg_mask
                ).chunk(2, dim=0)
                pred_x0 = pred_uncond + (pred_cond - pred_uncond) * cfg_scale
            else:
                pred_x0 = model(x, t, text_embeddings, attention_mask)
        v = (pred_x0 - x) / (1.0 - t[:, None, None, None]).clamp_min(0.05)
        x = x + v * (t1 - t0)
    return x.clamp(-1, 1)



def pristine_training_loss(
    model: nn.Module,
    images: torch.Tensor,
    text_mode: torch.Tensor,
    mask_mode: torch.Tensor,
    text_shared: torch.Tensor,
    mask_shared: torch.Tensor,
    mode: str = "safe",
    label_drop_rate: float = 0.1,
    t_lognorm_mu: float = -0.8,
    t_lognorm_sigma: float = 0.8,
    noise_scale: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    b = images.shape[0]
    device = images.device
    if label_drop_rate > 0:
        drop = torch.rand(b, device=device) < label_drop_rate
        mask_mode = torch.where(drop[:, None], torch.zeros_like(mask_mode), mask_mode)
        mask_shared = torch.where(drop[:, None], torch.zeros_like(mask_shared), mask_shared)
    t = sample_lognorm(b, t_lognorm_mu, t_lognorm_sigma, device)
    noise = torch.randn_like(images) * noise_scale
    x_t = images * t[:, None, None, None] + noise * (1.0 - t[:, None, None, None])
    pred_x0 = model(x_t, t, text_mode, mask_mode, text_shared, mask_shared, mode=mode)
    target = (images - x_t) / (1.0 - t[:, None, None, None]).clamp_min(0.05)
    v_pred = (pred_x0 - x_t) / (1.0 - t[:, None, None, None]).clamp_min(0.05)
    per_sample = (v_pred - target).pow(2).mean(dim=(1, 2, 3))
    loss = per_sample.mean()
    return loss, {"loss": loss.detach(), "loss_monitor": per_sample.mean().detach()}, None


@torch.no_grad()
def pristine_euler_sample(
    model: nn.Module,
    text_mode: torch.Tensor,
    mask_mode: torch.Tensor,
    text_shared: torch.Tensor,
    mask_shared: torch.Tensor,
    image_size: int,
    steps: int = 100,
    cfg_scale: float = 2.0,
    noise_scale: float = 2.0,
    mode: str = "safe",
    init_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    model.eval()
    b = text_mode.shape[0]
    device = text_mode.device
    if init_noise is not None:
        x = init_noise.to(device=device, dtype=text_mode.dtype)
    else:
        x = torch.randn(b, 3, image_size, image_size, device=device, dtype=text_mode.dtype) * noise_scale
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    null_mask_mode = torch.zeros_like(mask_mode)
    null_mask_shared = torch.zeros_like(mask_shared)
    for i in tqdm(range(steps), desc="sampling", disable=not is_main(), leave=False):
        t0, t1 = ts[i], ts[i + 1]
        t = torch.full((b,), float(t0), device=device)
        amp_dtype = text_mode.dtype if text_mode.dtype in (torch.float16, torch.bfloat16) else torch.float32
        with torch.autocast("cuda", dtype=amp_dtype, enabled=x.is_cuda and amp_dtype != torch.float32):
            if cfg_scale != 1.0:
                pred_cond = model(x, t, text_mode, mask_mode, text_shared, mask_shared, mode=mode)
                pred_uncond = model(x, t, text_mode, null_mask_mode, text_shared, null_mask_shared, mode=mode)
                pred_x0 = pred_uncond + (pred_cond - pred_uncond) * cfg_scale
            else:
                pred_x0 = model(x, t, text_mode, mask_mode, text_shared, mask_shared, mode=mode)
        v = (pred_x0 - x) / (1.0 - t[:, None, None, None]).clamp_min(0.05)
        x = x + v * (t1 - t0)
    return x.clamp(-1, 1)


def pristine_two_branch_training_loss(
    model: nn.Module,
    images: torch.Tensor,
    text: torch.Tensor,
    mask: torch.Tensor,
    text_unsafe: torch.Tensor | None = None,
    mask_unsafe: torch.Tensor | None = None,
    label_drop_rate: float = 0.1,
    t_lognorm_mu: float = -0.8,
    t_lognorm_sigma: float = 0.8,
    noise_scale: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    b = images.shape[0]
    device = images.device
    if label_drop_rate > 0:
        drop = torch.rand(b, device=device) < label_drop_rate
        mask = torch.where(drop[:, None], torch.zeros_like(mask), mask)
        if mask_unsafe is not None:
            mask_unsafe = torch.where(drop[:, None], torch.zeros_like(mask_unsafe), mask_unsafe)
    t = sample_lognorm(b, t_lognorm_mu, t_lognorm_sigma, device)
    noise = torch.randn_like(images) * noise_scale
    x_t = images * t[:, None, None, None] + noise * (1.0 - t[:, None, None, None])
    pred_x0 = model(x_t, t, text, mask, text_unsafe, mask_unsafe)
    target = (images - x_t) / (1.0 - t[:, None, None, None]).clamp_min(0.05)
    v_pred = (pred_x0 - x_t) / (1.0 - t[:, None, None, None]).clamp_min(0.05)
    per_sample = (v_pred - target).pow(2).mean(dim=(1, 2, 3))
    loss = per_sample.mean()
    return loss, {"loss": loss.detach(), "loss_monitor": per_sample.mean().detach()}, None


@torch.no_grad()
def pristine_two_branch_euler_sample(
    model: nn.Module,
    text: torch.Tensor,
    mask: torch.Tensor,
    text_unsafe: torch.Tensor | None,
    mask_unsafe: torch.Tensor | None,
    image_size: int,
    steps: int = 100,
    cfg_scale: float = 2.0,
    noise_scale: float = 2.0,
    init_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    model.eval()
    b = text.shape[0]
    device = text.device
    if init_noise is not None:
        x = init_noise.to(device=device, dtype=text.dtype)
    else:
        x = torch.randn(b, 3, image_size, image_size, device=device, dtype=text.dtype) * noise_scale
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    null_mask = torch.zeros_like(mask)
    null_mask_unsafe = torch.zeros_like(mask_unsafe) if mask_unsafe is not None else None
    for i in tqdm(range(steps), desc="sampling", disable=not is_main(), leave=False):
        t0, t1 = ts[i], ts[i + 1]
        t = torch.full((b,), float(t0), device=device)
        amp_dtype = text.dtype if text.dtype in (torch.float16, torch.bfloat16) else torch.float32
        with torch.autocast("cuda", dtype=amp_dtype, enabled=x.is_cuda and amp_dtype != torch.float32):
            if cfg_scale != 1.0:
                pred_cond = model(x, t, text, mask, text_unsafe, mask_unsafe)
                pred_uncond = model(x, t, text, null_mask, text_unsafe, null_mask_unsafe)
                pred_x0 = pred_uncond + (pred_cond - pred_uncond) * cfg_scale
            else:
                pred_x0 = model(x, t, text, mask, text_unsafe, mask_unsafe)
        v = (pred_x0 - x) / (1.0 - t[:, None, None, None]).clamp_min(0.05)
        x = x + v * (t1 - t0)
    return x.clamp(-1, 1)
