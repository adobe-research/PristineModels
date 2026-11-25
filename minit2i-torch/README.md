# PristineModels code

See the [main README](../README.md) for:

- Environment creation and dependency installation.
- Hugging Face caption/reference-data and ImageNet-1k downloads.
- Local dataset/output paths.
- Safe, Full, and three-branch Pristine training commands.
- ImageNet-V2 archive conversion, image generation, and FID/evaluation setup.

Run training and inference commands from this directory after completing setup
in the main README. Use `mini_t2i.train` for Safe/Full and
`mini_t2i.pristine_train` for three-branch Pristine.

## CC12M training

### Download the data

- **Pretraining:** Get the JPEGs in the [CC12M image tars](https://huggingface.co/datasets/pixparse/cc12m-wds/tree/796118f2eabdb9984f23f7f15d1e74d388612fc6) and the whole captions in [LLaVA-NeXT `train.jsonl.gz`](https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext/blob/1c24bdba3f81257082612654ba72fe18a83c7a37/train.jsonl.gz). For example, match `000000001.jpg` to the JSONL row with `key: "000000001"` and read its `caption_llava`.
- **Fine-tuning:** Get the original [BLIP3o-60k tars](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k/tree/f7316b0aa446338ee1707484924aa59457b4bbf3), [DALL-E 3 Parquet files](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset/tree/96822d5c0b1338ec4bfd04b0d5909fa070e301b6), and [ShareGPT-4o text-to-image tars and `text_to_image.json`](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image/tree/f9bcb9494e562f3b6d46585bda0e7f0f0d6ec43f). Keep their original filenames.
- **From us:** The [CC12M metadata bundle](https://huggingface.co/datasets/AniSundar18/PristineModels/tree/main/data/cc12m) supplies subject/residual captions, unsafe IDs, and exclusion lists. Copy `settings/` and `decomposed_prompts/` into `data/cc12m/`. It does not contain images, whole captions, or prepared chunks.

### Put the files in place

Paths below are relative to the repo root. Keep the pretraining tars and
`train.jsonl.gz` anywhere; only the prep step reads them.

```text
data/cc12m/
|-- settings/                                      # from our bundle
|   |-- cc12m_unsafe_keys.json
|   |-- leakage_pristine_merged.json
|   |-- finetune_unsafe_keys.json
|   `-- finetune_leakage_pristine_merged.json
|-- decomposed_prompts/
|   |-- pretrain/part_*/captions_merged.jsonl        # from our bundle
|   `-- finetune/<source>/part_*/captions_merged.jsonl
|-- chunks/chunk_*.pt                               # build for pretraining
|-- finetune_wds/
|   |-- blip3_ft60k/*.tar                           # upstream archives
|   |-- dalle3_parquet/*.parquet
|   `-- sharegpt4o/text_to_image_part_*.tar
`-- whole_prompts/finetune/<source>/part_*/captions_merged.jsonl  # build from upstream captions
```

**Pretraining chunks.** Resize RGB images to 512 on the short side and
center-crop to 512 x 512. Tokenize the matching `caption_llava` text with
`google/flan-t5-large`, padding or truncating to 256 tokens. Save each group
of `N` images as one dictionary with `torch.save`, e.g. `chunk_000000.pt`.

| Field | What to save |
|---|---|
| `key` | List of `N` original image IDs as strings |
| `caption` | List of `N` whole captions |
| `pixel_values` | `uint8` image tensor of shape `[N, 3, 512, 512]` |
| `input_ids`, `attention_mask` | Whole-caption token IDs and masks, each an `int64` tensor of shape `[N, 256]` |

For Pristine, also save these fields in chunks with non-excluded unsafe rows:

| Field | What to save |
|---|---|
| `subject_caption`, `shared_caption` | Lists of `N` strings: the subject and residual text from `decomposed_prompts/pretrain/` |
| `input_ids_mode`, `attention_mask_mode` | Subject token IDs and masks, each `[N, 256]` |
| `input_ids_shared`, `attention_mask_shared` | Residual token IDs and masks, each `[N, 256]` |

Keep the same row order in every field. Split fields can be blank or padded
for safe rows. All three configs read the same `chunks/` folder. The
[existing converter](tools/cc12m_wds_to_chunks.py) does not build this full format.

**Fine-tuning captions.** No `.pt` chunks are needed. Write one JSON object
per image under `whole_prompts/finetune/<source>/part_*/captions_merged.jsonl`:

```json
{"key":"sharegpt4o_shard-000003_22580","captions":{"child_caption":"Example whole caption."}}
```

| `<source>` | Caption source | Key format |
|---|---|---|
| `blip3_ft60k` | Matching `.txt` inside the tar | `blip3_ft60k_<tar-stem>_<image-stem>` |
| `dalle3` | Parquet `caption`, or `synthetic_caption` if empty | `dalle3_<parquet-stem>_<zero-based-row>` |
| `sharegpt4o` | `input_prompt` matched on `output_image` in `text_to_image.json` | `sharegpt4o_shard-<six-digit-tar-part>_<image-stem>` |

### Pretrain

From `minit2i-torch/`:

```bash
torchrun --standalone --nproc-per-node=8 -m cc12m.train \
  --config-file configs/cc12m/full.yaml
```

### Fine-tune

Use the matching 250k training checkpoint produced by pretraining:

```bash
torchrun --standalone --nproc-per-node=8 -m cc12m.train \
  --config-file configs/cc12m/finetune/full.yaml \
  --resume-from outputs/cc12m/full/checkpoints/checkpoint_latest.pt
```

For Safe, use `safe.yaml` and the Safe checkpoint. For Pristine, use
`pristine.yaml`, `-m cc12m.pristine`, and the Pristine checkpoint.

This code builds on [MiniT2I](https://github.com/Hope7Happiness/minit2i-torch).
The upstream [MIT license](LICENSE) is preserved.
