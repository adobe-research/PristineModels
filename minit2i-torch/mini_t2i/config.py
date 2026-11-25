from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import yaml

from . import settings


@dataclass
class TrainConfig:
    project: str = "minit2i"
    wandb_entity: str | None = None  # W&B user or team; None uses the logged-in account's default
    run_name: str = "mmdit-b32-cc12m-from-scratch"
    output_dir: str = settings.path_str(settings.OUTPUT_ROOT / "mmdit_b32_cc12m")
    seed: int = 42

    dataset_backend: str = "local_folder"
    local_dataset_dir: str | None = None
    cc12m_flash_caption_dir: str | None = None
    cc12m_flash_decomposed_caption_dir: str | None = None
    cc12m_flash_s3_bucket: str | None = None
    cc12m_flash_s3_prefix: str = ""
    cc12m_flash_num_threads: int = 8
    cc12m_flash_tagged_jsonl: str | None = None
    cc12m_flash_excluded_jsonl: str | None = None
    cc12m_flash_group: str | None = None  # "safe", "unsafe", or None for all groups
    cc12m_flash_safe_val_keys_json: str | None = None
    cc12m_flash_unsafe_val_keys_json: str | None = None
    cc12m_flash_max_train_samples: int | None = None
    cc12m_flash_max_val_samples: int | None = None
    imagenet_captions_dir: str | None = None
    imagenet_val_captions_dir: str | None = None
    imagenet_image_dir: str | None = None
    imagenet_leakage_path: str | None = None
    imagenet_group: str | None = None  # "safe", "unsafe", or None for all classes
    imagenet_setting_labels_path: str | None = None
    imagenet_max_rows: int | None = None  # caps train and val rows; for quick tests
    finetune_dataset_dir: str | None = None
    finetune_metadata_dir: str | None = None
    finetune_start_step: int = 250_000
    finetune_sources: list[str] = field(default_factory=lambda: ["blip3_ft60k", "dalle3", "sharegpt4o"])
    finetune_mix_weights: list[float] = field(default_factory=lambda: [0.06, 0.016, 0.04])
    finetune_blip3o_repo: str = "BLIP3o/BLIP3o-60k"
    finetune_dalle3_repo: str = "OpenDatasets/dalle-3-dataset"
    finetune_sharegpt4o_repo: str = "FreedomIntelligence/ShareGPT-4o-Image"
    finetune_sharegpt4o_parts: int = 10
    finetune_sharegpt4o_filter_tokenizer: str = "google/flan-t5-base"
    finetune_sharegpt4o_max_tokens: int = 256
    finetune_hf_streaming: bool = True
    hf_token_file: str = settings.path_str(settings.HF_TOKEN_FILE)
    image_size: int = 512
    prompt_length: int = 256
    num_workers: int = 8
    dataloader_prefetch: int = 2
    dataloader_multiprocessing_context: str | None = None
    shuffle_buffer: int = 1_000

    t5_name: str = "google/flan-t5-large"
    freeze_t5: bool = True
    cache_text_encoder: bool = False

    patch_size: int = 32
    hidden_size: int = 768
    text_hidden_size: int = 768
    t5_hidden_size: int = 1024
    depth_double: int = 17
    text_preamble_depth: int = 2
    num_heads: int = 12
    head_dim: int = 64
    mlp_ratio: float = 2.6667
    pca_channels: int = 128
    final_layer_zero: bool = True
    compat_checkpoint_arch: bool = False
    attention_impl: str = "einsum"
    twobranch_fused_attention: bool = False

    prediction: str = "x"
    t_sample_schedule: str = "lognorm"
    t_lognorm_mu: float = -0.8
    t_lognorm_sigma: float = 0.8
    label_drop_rate: float = 0.1
    noise_scale: float = 2.0
    n_T: int = 100
    cfg_scale: float = 2.0

    batch_size: int = 1024
    micro_batch_size: int = 128
    grad_accum_steps: int = 1
    num_steps: int = 250_000
    warmup_steps: int = 5_000
    learning_rate: float = 4e-4
    adam_beta2: float = 0.95
    weight_decay: float = 0.0
    max_grad_norm: float = 0.0
    ema_decay: float = 0.99995
    amp_dtype: str = "bf16"
    compile_model: bool = False
    auto_resume: bool = True
    resume_from: str | None = None

    log_every: int = 100
    defer_loss_sync: bool = True
    sample_every: int = 10_000
    sample_prompts: list[str] = field(default_factory=list)
    log_images_every: int = 1_000
    val_every: int = 10_000
    ckpt_every: int = 10_000
    latest_every: int = 0  # overwrite checkpoints/checkpoint_latest.pt this often for resume; 0 disables
    fid_every: int = 20_000
    save_keep: int = 3  # numbered checkpoints to keep; 0 keeps all

    mscoco_caption_file: str = settings.path_str(settings.MSCOCO_CAPTION_FILE)
    mscoco_stats_file: str = settings.path_str(settings.MSCOCO_STATS_FILE)
    fid_num_samples: int = 30_000
    fid_batch_size: int = 64

    val_metric: str = "mse"
    val_fid_manifest: str = settings.path_str(settings.IMAGENET_CAPTIONS_ROOT / "settings" / "Dog" / "equalized_val.jsonl")
    val_fid_dir: str = "/tmp"
    val_fid_sources: list[str] = field(default_factory=lambda: ["v1", "v2"])
    val_fid_v1_real_dir: str = settings.path_str(settings.IMAGENET_PARQUET_ROOT)
    val_fid_v2_real_dir: str = settings.path_str(settings.DATA_ROOT / "imagenet" / "v2" / "data")
    val_fid_batch_size: int = 16

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def update_from_mapping(self, values: dict[str, Any]) -> "TrainConfig":
        for key, value in values.items():
            if not hasattr(self, key):
                raise KeyError(f"unknown TrainConfig key in config file: {key}")
            setattr(self, key, settings.expand_setting(value))
        return self

    def update_from_yaml(self, path: str | Path) -> "TrainConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            values = yaml.safe_load(f) or {}
        if not isinstance(values, dict):
            raise TypeError(f"config file must contain a YAML mapping: {path}")
        return self.update_from_mapping(settings.expand_setting(values))

    def resolve(self) -> "TrainConfig":
        if self.val_metric not in ("mse", "fid"):
            raise ValueError("val_metric must be 'mse' or 'fid'")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        return self
