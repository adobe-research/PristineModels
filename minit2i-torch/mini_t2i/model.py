from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y if self.weight is None else y * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = math.ceil(hidden_dim / 8) * 8
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, 256).to(dtype=next(self.parameters()).dtype))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_1d_rope(x: torch.Tensor, start: int = 0, theta: int = 10000) -> torch.Tensor:
    b, h, n, d = x.shape
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    pos = torch.arange(start, start + n, device=x.device, dtype=torch.float32)
    angles = torch.einsum("n,f->nf", pos, inv)
    angles = torch.cat([angles, angles], dim=-1)
    cos = angles.cos()[None, None].to(x.dtype)
    sin = angles.sin()[None, None].to(x.dtype)
    return x * cos + rotate_half(x) * sin


def apply_2d_rope_flat(x: torch.Tensor, grid: int, theta: int = 10000) -> torch.Tensor:
    b, h, n, d = x.shape
    if n != grid * grid:
        return apply_1d_rope(x, theta=theta)
    rope_dim = d // 2
    inv = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, device=x.device, dtype=torch.float32) / rope_dim))
    t = torch.arange(grid, device=x.device, dtype=torch.float32)
    freqs = torch.einsum("n,f->nf", t, inv)
    f_h, f_w = torch.broadcast_tensors(freqs[:, None, :], freqs[None, :, :])
    angles = torch.cat([f_h, f_w], dim=-1)
    angles = torch.cat([angles, angles], dim=-1).reshape(n, d)
    cos = angles.cos()[None, None].to(x.dtype)
    sin = angles.sin()[None, None].to(x.dtype)
    return x * cos + rotate_half(x) * sin


def apply_2d_rope_flat_compat(x: torch.Tensor, grid: int, theta: int = 10000) -> torch.Tensor:
    b, h, n, d = x.shape
    if n != grid * grid:
        return apply_1d_rope(x, theta=theta)
    half = d // 2
    x_h, x_w = x[..., :half], x[..., half:]
    yy = torch.arange(grid, device=x.device, dtype=torch.float32).repeat_interleave(grid)
    xx = torch.arange(grid, device=x.device, dtype=torch.float32).repeat(grid)
    inv = 1.0 / (theta ** (torch.arange(0, half, 2, device=x.device, dtype=torch.float32) / half))
    ah = torch.einsum("n,f->nf", yy, inv)
    aw = torch.einsum("n,f->nf", xx, inv)
    ah = torch.cat([ah, ah], dim=-1)
    aw = torch.cat([aw, aw], dim=-1)
    x_h = x_h * ah.cos()[None, None].to(x.dtype) + rotate_half(x_h) * ah.sin()[None, None].to(x.dtype)
    x_w = x_w * aw.cos()[None, None].to(x.dtype) + rotate_half(x_w) * aw.sin()[None, None].to(x.dtype)
    return torch.cat([x_h, x_w], dim=-1)


def attention_einsum(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head_dim: int) -> torch.Tensor:
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * (head_dim**-0.5)
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", weights, v)


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_dim: int,
    impl: str,
) -> torch.Tensor:
    if impl == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=head_dim**-0.5)
    if impl == "einsum":
        return attention_einsum(q, k, v, head_dim)
    raise ValueError(f"unknown attention_impl={impl!r}")


class BottleneckPatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, hidden_size: int, pca_channels: int):
        super().__init__()
        self.grid = image_size // patch_size
        self.proj1 = nn.Conv2d(in_channels, pca_channels, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_channels, hidden_size, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.xavier_uniform_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj2(self.proj1(x))
        return x.flatten(2).transpose(1, 2)


def sincos_2d(embed_dim: int, grid: int) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    omega = torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim // 4)
    omega = 1.0 / (10000 ** omega)
    out_y = torch.einsum("n,d->nd", y.flatten().float(), omega)
    out_x = torch.einsum("n,d->nd", x.flatten().float(), omega)
    return torch.cat([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=1)


class PlainTextBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        qk_norm: bool = True,
        rms_affine: bool = True,
        attention_impl: str = "einsum",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qk_norm = qk_norm
        self.attention_impl = attention_impl
        inner = num_heads * head_dim
        self.norm1 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.norm2 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.qkv = nn.Linear(hidden_size, inner * 3)
        self.proj = nn.Linear(inner, hidden_size)
        self.mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, elementwise_affine=rms_affine)
            self.k_norm = RMSNorm(head_dim, elementwise_affine=rms_affine)

    def forward(self, txt: torch.Tensor) -> torch.Tensor:
        b, n, _ = txt.shape
        qkv = self.qkv(self.norm1(txt)).view(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k, v = [z.transpose(1, 2) for z in (q, k, v)]
        q = apply_1d_rope(q)
        k = apply_1d_rope(k)
        out = attention_forward(q, k, v, self.head_dim, self.attention_impl)
        txt = txt + self.proj(out.transpose(1, 2).reshape(b, n, -1))
        return txt + self.mlp(self.norm2(txt))


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        image_grid: int,
        qk_norm: bool = True,
        rms_affine: bool = True,
        rope_style: str = "jax",
        attention_impl: str = "einsum",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.image_grid = image_grid
        self.qk_norm = qk_norm
        self.rope_style = rope_style
        self.attention_impl = attention_impl
        inner = num_heads * head_dim
        self.img_norm1 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.img_norm2 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.txt_norm1 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.txt_norm2 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.img_qkv = nn.Linear(hidden_size, inner * 3)
        self.txt_qkv = nn.Linear(hidden_size, inner * 3)
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, elementwise_affine=rms_affine)
            self.k_norm = RMSNorm(head_dim, elementwise_affine=rms_affine)
        self.img_proj = nn.Linear(inner, hidden_size)
        self.txt_proj = nn.Linear(inner, hidden_size)
        self.img_mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        self.txt_mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))

    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, li, _ = img.shape
        lt = txt.shape[1]
        qi, ki, vi = self.img_qkv(self.img_norm1(img)).view(b, li, 3, self.num_heads, self.head_dim).unbind(2)
        qt, kt, vt = self.txt_qkv(self.txt_norm1(txt)).view(b, lt, 3, self.num_heads, self.head_dim).unbind(2)
        q = torch.cat([qt, qi], dim=1).transpose(1, 2)
        k = torch.cat([kt, ki], dim=1).transpose(1, 2)
        v = torch.cat([vt, vi], dim=1).transpose(1, 2)
        if self.qk_norm:
            q_text, q_img = self.q_norm(q[:, :, :lt]), self.q_norm(q[:, :, lt:])
            k_text, k_img = self.k_norm(k[:, :, :lt]), self.k_norm(k[:, :, lt:])
        else:
            q_text, q_img = q[:, :, :lt], q[:, :, lt:]
            k_text, k_img = k[:, :, :lt], k[:, :, lt:]
        rope2d = apply_2d_rope_flat_compat if self.rope_style == "compat" else apply_2d_rope_flat
        q = torch.cat([apply_1d_rope(q_text), rope2d(q_img, self.image_grid)], dim=2)
        k = torch.cat([apply_1d_rope(k_text), rope2d(k_img, self.image_grid)], dim=2)
        out = attention_forward(q, k, v, self.head_dim, self.attention_impl)
        out = out.transpose(1, 2)
        txt = txt + self.txt_proj(out[:, :lt].reshape(b, lt, -1))
        img = img + self.img_proj(out[:, lt:].reshape(b, li, -1))
        img = img + self.img_mlp(self.img_norm2(img))
        txt = txt + self.txt_mlp(self.txt_norm2(txt))
        return img, txt


class MMJiTB32Text2(nn.Module):
    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 32,
        in_channels: int = 3,
        hidden_size: int = 768,
        t5_hidden_size: int = 1024,
        depth_double: int = 17,
        text_preamble_depth: int = 2,
        num_heads: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 2.6667,
        pca_channels: int = 128,
        final_layer_zero: bool = True,
        rms_affine: bool = True,
        text_qk_norm: bool = True,
        double_qk_norm: bool = True,
        rope_style: str = "jax",
        attention_impl: str = "einsum",
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid = image_size // patch_size
        self.img_embed = BottleneckPatchEmbed(image_size, patch_size, in_channels, hidden_size, pca_channels)
        self.txt_embed = nn.Linear(t5_hidden_size, hidden_size, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, t5_hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)
        self.t_embed = TimestepEmbedder(hidden_size)
        self.pooled_embed = nn.Linear(t5_hidden_size, hidden_size, bias=False)
        self.register_buffer("pos_embed", sincos_2d(hidden_size, self.grid).unsqueeze(0), persistent=False)
        self.txt_blocks = nn.ModuleList(
            [
                PlainTextBlock(
                    hidden_size,
                    num_heads,
                    head_dim,
                    mlp_ratio,
                    qk_norm=text_qk_norm,
                    rms_affine=rms_affine,
                    attention_impl=attention_impl,
                )
                for _ in range(text_preamble_depth)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    hidden_size,
                    num_heads,
                    head_dim,
                    mlp_ratio,
                    self.grid,
                    qk_norm=double_qk_norm,
                    rms_affine=rms_affine,
                    rope_style=rope_style,
                    attention_impl=attention_impl,
                )
                for _ in range(depth_double)
            ]
        )
        self.final_norm = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.final = nn.Linear(hidden_size, patch_size * patch_size * in_channels)
        if final_layer_zero:
            nn.init.zeros_(self.final.weight)
            nn.init.zeros_(self.final.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        p = self.patch_size
        c = self.in_channels
        g = int(n**0.5)
        x = x.view(b, g, g, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, c, g * p, g * p)

    def forward(self, img: torch.Tensor, t: torch.Tensor, context: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        mask = attn_mask.to(dtype=torch.bool)[:, :, None]
        context = torch.where(mask, context, self.mask_token.to(dtype=context.dtype))
        img_tokens = self.img_embed(img) + self.pos_embed.to(device=img.device, dtype=img.dtype)
        txt = self.txt_embed(context)
        pooled = context.mean(dim=1)
        vec = self.t_embed(t).to(dtype=img.dtype) + self.pooled_embed(pooled)
        del vec
        for block in self.txt_blocks:
            txt = block(txt)
        for block in self.blocks:
            img_tokens, txt = block(img_tokens, txt)
        out = self.final(self.final_norm(img_tokens))
        return self.unpatchify(out).float()



class PristineDoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        image_grid: int,
        qk_norm: bool = True,
        rms_affine: bool = True,
        rope_style: str = "jax",
        attention_impl: str = "einsum",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.image_grid = image_grid
        self.qk_norm = qk_norm
        self.rope_style = rope_style
        self.attention_impl = attention_impl
        inner = num_heads * head_dim
        self.img_norm1 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.img_norm2 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.txt_norm1 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.txt_norm2 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.img_qkv_safe = nn.Linear(hidden_size, inner * 3)
        self.img_qkv_unsafe = nn.Linear(hidden_size, inner * 3)
        self.img_qkv_shared = nn.Linear(hidden_size, inner * 3)
        self.txt_qkv_safe = nn.Linear(hidden_size, inner * 3)
        self.txt_qkv_unsafe = nn.Linear(hidden_size, inner * 3)
        self.txt_qkv_shared = nn.Linear(hidden_size, inner * 3)
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, elementwise_affine=rms_affine)
            self.k_norm = RMSNorm(head_dim, elementwise_affine=rms_affine)
        self.img_proj_safe = nn.Linear(inner, hidden_size)
        self.img_proj_unsafe = nn.Linear(inner, hidden_size)
        self.img_proj_shared = nn.Linear(inner, hidden_size)
        self.txt_proj_safe = nn.Linear(inner, hidden_size)
        self.txt_proj_unsafe = nn.Linear(inner, hidden_size)
        self.txt_proj_shared = nn.Linear(inner, hidden_size)

        self.img_mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        self.txt_mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))

    def forward(self, img: torch.Tensor, txt_mode: torch.Tensor, txt_shared: torch.Tensor, mode: str = "safe") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, li, _ = img.shape
        l_shared = txt_shared.shape[1]
        l_mode = txt_mode.shape[1]
        img_n = self.img_norm1(img)
        qi_shared, ki_shared, vi_shared = self.img_qkv_shared(img_n).view(b, li, 3, self.num_heads, self.head_dim).unbind(2)
        qt_shared, kt_shared, vt_shared = self.txt_qkv_shared(self.txt_norm1(txt_shared)).view(b, l_shared, 3, self.num_heads, self.head_dim).unbind(2)
        q_shared = torch.cat([qt_shared, qi_shared], dim=1).transpose(1, 2)
        k_shared = torch.cat([kt_shared, ki_shared], dim=1).transpose(1, 2)
        v_shared = torch.cat([vt_shared, vi_shared], dim=1).transpose(1, 2)
        if self.qk_norm:
            q_shared_text, q_shared_img = self.q_norm(q_shared[:, :, :l_shared]), self.q_norm(q_shared[:, :, l_shared:])
            k_shared_text, k_shared_img = self.k_norm(k_shared[:, :, :l_shared]), self.k_norm(k_shared[:, :, l_shared:])
        else:
            q_shared_text, q_shared_img = q_shared[:, :, :l_shared], q_shared[:, :, l_shared:]
            k_shared_text, k_shared_img = k_shared[:, :, :l_shared], k_shared[:, :, l_shared:]
        rope2d = apply_2d_rope_flat_compat if self.rope_style == "compat" else apply_2d_rope_flat
        q_shared = torch.cat([apply_1d_rope(q_shared_text), rope2d(q_shared_img, self.image_grid)], dim=2)
        k_shared = torch.cat([apply_1d_rope(k_shared_text), rope2d(k_shared_img, self.image_grid)], dim=2)
        out_shared = attention_forward(q_shared, k_shared, v_shared, self.head_dim, self.attention_impl)
        out_shared = out_shared.transpose(1, 2)
        mode_img_qkv = self.img_qkv_safe if mode == "safe" else self.img_qkv_unsafe
        mode_txt_qkv = self.txt_qkv_safe if mode == "safe" else self.txt_qkv_unsafe
        mode_img_proj = self.img_proj_safe if mode == "safe" else self.img_proj_unsafe
        mode_txt_proj = self.txt_proj_safe if mode == "safe" else self.txt_proj_unsafe
        qi_mode, ki_mode, vi_mode = mode_img_qkv(img_n).view(b, li, 3, self.num_heads, self.head_dim).unbind(2)
        qt_mode, kt_mode, vt_mode = mode_txt_qkv(self.txt_norm1(txt_mode)).view(b, l_mode, 3, self.num_heads, self.head_dim).unbind(2)
        q_mode = torch.cat([qt_mode, qi_mode], dim=1).transpose(1, 2)
        k_mode = torch.cat([kt_mode, ki_mode], dim=1).transpose(1, 2)
        v_mode = torch.cat([vt_mode, vi_mode], dim=1).transpose(1, 2)
        if self.qk_norm:
            q_mode_text, q_mode_img = self.q_norm(q_mode[:, :, :l_mode]), self.q_norm(q_mode[:, :, l_mode:])
            k_mode_text, k_mode_img = self.k_norm(k_mode[:, :, :l_mode]), self.k_norm(k_mode[:, :, l_mode:])
        else:
            q_mode_text, q_mode_img = q_mode[:, :, :l_mode], q_mode[:, :, l_mode:]
            k_mode_text, k_mode_img = k_mode[:, :, :l_mode], k_mode[:, :, l_mode:]
        q_mode = torch.cat([apply_1d_rope(q_mode_text), rope2d(q_mode_img, self.image_grid)], dim=2)
        k_mode = torch.cat([apply_1d_rope(k_mode_text), rope2d(k_mode_img, self.image_grid)], dim=2)
        out_mode = attention_forward(q_mode, k_mode, v_mode, self.head_dim, self.attention_impl)
        out_mode = out_mode.transpose(1, 2)
        txt_shared = txt_shared + self.txt_proj_shared(out_shared[:, :l_shared].reshape(b, l_shared, -1))
        img = img + self.img_proj_shared(out_shared[:, l_shared:].reshape(b, li, -1)) + mode_img_proj(out_mode[:, l_mode:].reshape(b, li, -1))
        img = img + self.img_mlp(self.img_norm2(img))
        txt_shared = txt_shared + self.txt_mlp(self.txt_norm2(txt_shared))
        txt_mode = txt_mode + mode_txt_proj(out_mode[:, :l_mode].reshape(b, l_mode, -1))
        txt_mode = txt_mode + self.txt_mlp(self.txt_norm2(txt_mode))
        return img, txt_mode, txt_shared



class PristineMMJiTB32Text2(nn.Module):
    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 32,
        in_channels: int = 3,
        hidden_size: int = 768,
        t5_hidden_size: int = 1024,
        depth_double: int = 17,
        text_preamble_depth: int = 2,
        num_heads: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 2.6667,
        pca_channels: int = 128,
        final_layer_zero: bool = True,
        rms_affine: bool = True,
        text_qk_norm: bool = True,
        double_qk_norm: bool = True,
        rope_style: str = "jax",
        attention_impl: str = "einsum",
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid = image_size // patch_size
        self.img_embed = BottleneckPatchEmbed(image_size, patch_size, in_channels, hidden_size, pca_channels)
        self.txt_embed = nn.Linear(t5_hidden_size, hidden_size, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, t5_hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)
        self.t_embed = TimestepEmbedder(hidden_size)
        self.pooled_embed = nn.Linear(t5_hidden_size, hidden_size, bias=False)
        self.register_buffer("pos_embed", sincos_2d(hidden_size, self.grid).unsqueeze(0), persistent=False)
        self.txt_blocks = nn.ModuleList(
            [
                PlainTextBlock(
                    hidden_size,
                    num_heads,
                    head_dim,
                    mlp_ratio,
                    qk_norm=text_qk_norm,
                    rms_affine=rms_affine,
                    attention_impl=attention_impl,
                )
                for _ in range(text_preamble_depth)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                PristineDoubleStreamBlock(
                    hidden_size,
                    num_heads,
                    head_dim,
                    mlp_ratio,
                    self.grid,
                    qk_norm=double_qk_norm,
                    rms_affine=rms_affine,
                    rope_style=rope_style,
                    attention_impl=attention_impl,
                )
                for _ in range(depth_double)
            ]
        )
        self.final_norm = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.final = nn.Linear(hidden_size, patch_size * patch_size * in_channels)
        if final_layer_zero:
            nn.init.zeros_(self.final.weight)
            nn.init.zeros_(self.final.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        p = self.patch_size
        c = self.in_channels
        g = int(n**0.5)
        x = x.view(b, g, g, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, c, g * p, g * p)

    def forward(self, img: torch.Tensor, t: torch.Tensor, context_mode: torch.Tensor, attn_mask_mode: torch.Tensor, context_shared: torch.Tensor, attn_mask_shared: torch.Tensor, mode: str = "safe") -> torch.Tensor:
        mask_mode = attn_mask_mode.to(dtype=torch.bool)[:, :, None]
        mask_shared = attn_mask_shared.to(dtype=torch.bool)[:, :, None]
        context_mode = torch.where(mask_mode, context_mode, self.mask_token.to(dtype=context_mode.dtype))
        context_shared = torch.where(mask_shared, context_shared, self.mask_token.to(dtype=context_shared.dtype))
        img_tokens = self.img_embed(img) + self.pos_embed.to(device=img.device, dtype=img.dtype)
        txt_mode = self.txt_embed(context_mode)
        txt_shared = self.txt_embed(context_shared)
        pooled_mode = context_mode.mean(dim=1)
        pooled_shared = context_shared.mean(dim=1)
        vec = self.t_embed(t).to(dtype=img.dtype) + self.pooled_embed(pooled_mode) #+ self.pooled_embed(pooled_shared)
        del vec
        for block in self.txt_blocks:
            txt_mode = block(txt_mode)
            txt_shared = block(txt_shared)
        for block in self.blocks:
            img_tokens, txt_mode, txt_shared = block(img_tokens, txt_mode, txt_shared, mode=mode)
        out = self.final(self.final_norm(img_tokens))
        return self.unpatchify(out).float()


class PristineTwoBranchDoubleStreamBlock(nn.Module):
    """Shared and unsafe attention branches with shared MLPs."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        image_grid: int,
        qk_norm: bool = True,
        rms_affine: bool = True,
        rope_style: str = "jax",
        attention_impl: str = "einsum",
        is_fused: bool = False,
    ):
        super().__init__()
        self.is_fused = is_fused
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.image_grid = image_grid
        self.qk_norm = qk_norm
        self.rope_style = rope_style
        self.attention_impl = attention_impl
        inner = num_heads * head_dim
        self.img_norm1 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.img_norm2 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.txt_norm1 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.txt_norm2 = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.img_qkv = nn.Linear(hidden_size, inner * 3)
        self.txt_qkv = nn.Linear(hidden_size, inner * 3)
        self.img_qkv_unsafe = nn.Linear(hidden_size, inner * 3)
        self.txt_qkv_unsafe = nn.Linear(hidden_size, inner * 3)
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, elementwise_affine=rms_affine)
            self.k_norm = RMSNorm(head_dim, elementwise_affine=rms_affine)
        self.img_proj = nn.Linear(inner, hidden_size)
        self.txt_proj = nn.Linear(inner, hidden_size)
        self.img_proj_unsafe = nn.Linear(inner, hidden_size)
        self.txt_proj_unsafe = nn.Linear(inner, hidden_size)
        self.img_mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        self.txt_mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))

    def forward(
        self, img: torch.Tensor, txt: torch.Tensor, txt_unsafe: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        b, li, _ = img.shape
        lt = txt.shape[1]
        img_n = self.img_norm1(img)
        qi, ki, vi = self.img_qkv(img_n).view(b, li, 3, self.num_heads, self.head_dim).unbind(2)
        qt, kt, vt = self.txt_qkv(self.txt_norm1(txt)).view(b, lt, 3, self.num_heads, self.head_dim).unbind(2)
        q = torch.cat([qt, qi], dim=1).transpose(1, 2)
        k = torch.cat([kt, ki], dim=1).transpose(1, 2)
        v = torch.cat([vt, vi], dim=1).transpose(1, 2)
        if self.qk_norm:
            q_text, q_img = self.q_norm(q[:, :, :lt]), self.q_norm(q[:, :, lt:])
            k_text, k_img = self.k_norm(k[:, :, :lt]), self.k_norm(k[:, :, lt:])
        else:
            q_text, q_img = q[:, :, :lt], q[:, :, lt:]
            k_text, k_img = k[:, :, :lt], k[:, :, lt:]
        rope2d = apply_2d_rope_flat_compat if self.rope_style == "compat" else apply_2d_rope_flat
        q = torch.cat([apply_1d_rope(q_text), rope2d(q_img, self.image_grid)], dim=2)
        k = torch.cat([apply_1d_rope(k_text), rope2d(k_img, self.image_grid)], dim=2)
        # In fused mode with the unsafe branch active, this call updates only
        # shared text tokens; fused image attention is computed below.
        shared_query = q[:, :, :lt] if self.is_fused and txt_unsafe is not None else q
        out = attention_forward(shared_query, k, v, self.head_dim, self.attention_impl)
        out = out.transpose(1, 2)
        txt_out = out[:, :lt].reshape(b, lt, -1)
        txt = txt + self.txt_proj(txt_out)

        if txt_unsafe is None:
            img = img + self.img_proj(out[:, lt:].reshape(b, li, -1))
        else:
            lu = txt_unsafe.shape[1]
            qi_u, ki_u, vi_u = self.img_qkv_unsafe(img_n).view(b, li, 3, self.num_heads, self.head_dim).unbind(2)
            qt_u, kt_u, vt_u = self.txt_qkv_unsafe(self.txt_norm1(txt_unsafe)).view(b, lu, 3, self.num_heads, self.head_dim).unbind(2)
            q_u = torch.cat([qt_u, qi_u], dim=1).transpose(1, 2)
            k_u = torch.cat([kt_u, ki_u], dim=1).transpose(1, 2)
            v_u = torch.cat([vt_u, vi_u], dim=1).transpose(1, 2)
            if self.qk_norm:
                q_u_text, q_u_img = self.q_norm(q_u[:, :, :lu]), self.q_norm(q_u[:, :, lu:])
                k_u_text, k_u_img = self.k_norm(k_u[:, :, :lu]), self.k_norm(k_u[:, :, lu:])
            else:
                q_u_text, q_u_img = q_u[:, :, :lu], q_u[:, :, lu:]
                k_u_text, k_u_img = k_u[:, :, :lu], k_u[:, :, lu:]
            q_u = torch.cat([apply_1d_rope(q_u_text), rope2d(q_u_img, self.image_grid)], dim=2)
            k_u = torch.cat([apply_1d_rope(k_u_text), rope2d(k_u_img, self.image_grid)], dim=2)
            # In fused mode, this call updates only unsafe text tokens;
            # joint shared/unsafe image attention is computed below.
            unsafe_query = q_u[:, :, :lu] if self.is_fused else q_u
            out_u = attention_forward(unsafe_query, k_u, v_u, self.head_dim, self.attention_impl)
            out_u = out_u.transpose(1, 2)
            if self.is_fused:
                scores_shared = q[:, :, lt:] @ k.transpose(-2, -1)
                scores_unsafe = q_u[:, :, lu:] @ k_u.transpose(-2, -1)
                weights = torch.softmax(torch.cat([scores_shared, scores_unsafe], dim=-1) * (self.head_dim**-0.5), dim=-1)
                img_out = weights @ torch.cat([v, v_u], dim=-2)
                img_out = img_out.transpose(1, 2).reshape(b, li, -1)
            else:
                img_out = out[:, lt:].reshape(b, li, -1) + out_u[:, lu:].reshape(b, li, -1)
            img = img + self.img_proj_unsafe(img_out)
            txt_unsafe = txt_unsafe + self.txt_proj_unsafe(out_u[:, :lu].reshape(b, lu, -1))
            txt_unsafe = txt_unsafe + self.txt_mlp(self.txt_norm2(txt_unsafe))

        img = img + self.img_mlp(self.img_norm2(img))
        txt = txt + self.txt_mlp(self.txt_norm2(txt))
        return img, txt, txt_unsafe


class PristineTwoBranchMMJiTB32Text2(nn.Module):
    """Two-branch MMJiT with shared/unsafe attention branches and shared MLPs."""

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 32,
        in_channels: int = 3,
        hidden_size: int = 768,
        t5_hidden_size: int = 1024,
        depth_double: int = 17,
        text_preamble_depth: int = 2,
        num_heads: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 2.6667,
        pca_channels: int = 128,
        final_layer_zero: bool = True,
        rms_affine: bool = True,
        text_qk_norm: bool = True,
        double_qk_norm: bool = True,
        rope_style: str = "jax",
        attention_impl: str = "einsum",
        is_fused: bool = False,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid = image_size // patch_size
        self.img_embed = BottleneckPatchEmbed(image_size, patch_size, in_channels, hidden_size, pca_channels)
        self.txt_embed = nn.Linear(t5_hidden_size, hidden_size, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, t5_hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)
        self.t_embed = TimestepEmbedder(hidden_size)
        self.pooled_embed = nn.Linear(t5_hidden_size, hidden_size, bias=False)
        self.register_buffer("pos_embed", sincos_2d(hidden_size, self.grid).unsqueeze(0), persistent=False)
        self.txt_blocks = nn.ModuleList(
            [
                PlainTextBlock(
                    hidden_size,
                    num_heads,
                    head_dim,
                    mlp_ratio,
                    qk_norm=text_qk_norm,
                    rms_affine=rms_affine,
                    attention_impl=attention_impl,
                )
                for _ in range(text_preamble_depth)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                PristineTwoBranchDoubleStreamBlock(
                    hidden_size,
                    num_heads,
                    head_dim,
                    mlp_ratio,
                    self.grid,
                    qk_norm=double_qk_norm,
                    rms_affine=rms_affine,
                    rope_style=rope_style,
                    attention_impl=attention_impl,
                    is_fused=is_fused,
                )
                for _ in range(depth_double)
            ]
        )
        self.final_norm = RMSNorm(hidden_size, elementwise_affine=rms_affine)
        self.final = nn.Linear(hidden_size, patch_size * patch_size * in_channels)
        if final_layer_zero:
            nn.init.zeros_(self.final.weight)
            nn.init.zeros_(self.final.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        p = self.patch_size
        c = self.in_channels
        g = int(n**0.5)
        x = x.view(b, g, g, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, c, g * p, g * p)

    def forward(
        self,
        img: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        attn_mask: torch.Tensor,
        context_unsafe: torch.Tensor | None = None,
        attn_mask_unsafe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = attn_mask.to(dtype=torch.bool)[:, :, None]
        context = torch.where(mask, context, self.mask_token.to(dtype=context.dtype))
        img_tokens = self.img_embed(img) + self.pos_embed.to(device=img.device, dtype=img.dtype)
        txt = self.txt_embed(context)
        pooled = context.mean(dim=1)
        vec = self.t_embed(t).to(dtype=img.dtype) + self.pooled_embed(pooled)
        del vec
        txt_unsafe = None
        if context_unsafe is not None:
            mask_unsafe = attn_mask_unsafe.to(dtype=torch.bool)[:, :, None]
            context_unsafe = torch.where(mask_unsafe, context_unsafe, self.mask_token.to(dtype=context_unsafe.dtype))
            txt_unsafe = self.txt_embed(context_unsafe)
        for block in self.txt_blocks:
            txt = block(txt)
            if txt_unsafe is not None:
                txt_unsafe = block(txt_unsafe)
        for block in self.blocks:
            img_tokens, txt, txt_unsafe = block(img_tokens, txt, txt_unsafe)
        out = self.final(self.final_norm(img_tokens))
        return self.unpatchify(out).float()
