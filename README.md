# PristineModels

[Project page](https://anisundar18.github.io/PristineModels.github.io/)

ImageNet training and evaluation for **Safe**, **Full**, and **Pristine** text-to-image models, built on MiniT2I.

Code and preparation tools live on GitHub. Captions, evaluation-related files live in the Hugging Face dataset. Pretrained checkpoints live in the separate Hugging Face
[model repository](https://huggingface.co/AniSundar18/PristineModels).

## Quick start

1. Clone this repository and create the environment.
2. Download our captions, evaluation manifests, and reference images.
3. Download the original ImageNet-1k and ImageNet-V2 images, then repackage V2.
4. Set two local paths and choose a training command below.

The commands below start from the repository root unless otherwise stated.

## 1. Clone and install

```bash
git clone https://github.com/adobe-research/PristineModels.git
cd PristineModels

conda create -n pristine-models python=3.12 -y
conda activate pristine-models

python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r minit2i-torch/requirements.txt
```

## 2. Authenticate with Hugging Face

```bash
hf auth login
hf auth whoami
```

For ImageNet-1k, use [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

## 3. Download captions and evaluation data

Our dataset is
[AniSundar18/PristineModels](https://huggingface.co/datasets/AniSundar18/PristineModels).
It contains the existing `data/imagenet/...` hierarchy:

- Whole captions for Safe and Full training.
- Decomposed subject/residual captions for Pristine.
- Safe/unsafe class definitions.
- Evaluation prompts/other metadata.

**Run this from the repository root.** The remote dataset already contains the
`data/` prefix, so use `--local-dir .`, not `--local-dir data`.

```bash
hf download AniSundar18/PristineModels \
  --repo-type dataset \
  --include "data/**" \
  --local-dir .
```

This download is approximately **1.54 GiB**, including about 346 MiB of reference
images. It does **not** include the original ImageNet-1k training images.

## 4. Download ImageNet-1k images

Download the original parquet shards from
[ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k):

```bash
mkdir -p "$HOME/datasets/imagenet"

hf download ILSVRC/imagenet-1k \
  --repo-type dataset \
  --include "data/train-*.parquet" \
  --local-dir "$HOME/datasets/imagenet"

hf download ILSVRC/imagenet-1k \
  --repo-type dataset \
  --include "data/validation-*.parquet" \
  --local-dir "$HOME/datasets/imagenet"
```

The image download is much larger than the caption download. Keep sufficient disk
space available. Do not extract, rename, or repartition these parquet shards:
the caption records identify images by `shard_file` and `row_in_shard`.

### ImageNet-V2: evaluation images

ImageNet-V2 is the test dataset from Recht et al.,
[*Do ImageNet Classifiers Generalize to ImageNet?*](https://arxiv.org/abs/1902.10811).
It is separate from ImageNet-1k and is used here for evaluation, not training.

The [official ImageNet-V2 repository](https://github.com/modestyachts/ImageNetV2)
directs downloads to
[vaishaal/ImageNetV2 on Hugging Face](https://huggingface.co/datasets/vaishaal/ImageNetV2).
This is the source of the three original archives:

| Set | Images | Archive |
| --- | ---: | --- |
| MatchedFrequency | 10,000 | `imagenetv2-matched-frequency.tar.gz` |
| Threshold0.7 | 10,000 | `imagenetv2-threshold0.7.tar.gz` |
| TopImages | 10,000 | `imagenetv2-top-images.tar.gz` |

There are **30,000 image entries across the three sets**. Our validation manifest
draws from all three, so downloading only MatchedFrequency is insufficient for
the supplied evaluation protocol. Evaluation follows the manifest's selected
records, not every image in the original 30,000-entry collection.

Download all three archives:

```bash
mkdir -p "$HOME/datasets/imagenet/v2"

hf download vaishaal/ImageNetV2 \
  imagenetv2-matched-frequency.tar.gz \
  imagenetv2-threshold0.7.tar.gz \
  imagenetv2-top-images.tar.gz \
  --repo-type dataset \
  --revision d626240be2538720e83103a0e1178d24aca8b12c \
  --local-dir "$HOME/datasets/imagenet/v2"
```

**Archive download is not the final setup step for our FID/PRC evaluator.**
The original release stores images in tar archives, whereas our caption manifests
reference parquet filenames and row offsets. The evaluator expects these three
families under `$HOME/datasets/imagenet/v2/data/`:

```text
matched-frequency-00000-of-00010.parquet ... matched-frequency-00009-of-00010.parquet
threshold0.7-00000-of-00010.parquet      ... threshold0.7-00009-of-00010.parquet
top-images-00000-of-00010.parquet        ... top-images-00009-of-00010.parquet
```

Repackage them with the standalone converter, from the repository root:

```bash
python minit2i-torch/tools/repackage_imagenet_v2.py \
  --archive-dir "$HOME/datasets/imagenet/v2" \
  --out "$HOME/datasets/imagenet/v2/data" \
  --delete-archives
```

See [Evaluate all three runs](#evaluate-all-three-runs) for evaluation commands
after completing both image downloads and V2 conversion.

Expected layout after conversion (10 Parquet shards per V2 set, 30 total):

```text
PristineModels/
  README.md
  data/
    imagenet/
      explicit/
        whole_prompts/{train,val}/
        decomposed_prompts/{train,val}/
      settings/Dog/
        imagenet_dog_setting_labels.json
        leakage_merged.json
        equalized_val.jsonl
      clearbg/
      implicit/
      compositional/
        qwen_reference/
      v2/
  minit2i-torch/
    configs/imagenet/
    mini_t2i/
    inference/
    evaluations/

$HOME/datasets/
  imagenet/
    data/
      train-*.parquet
      validation-*.parquet
    v2/
      data/
        matched-frequency-00000-of-00010.parquet
        ...
        matched-frequency-00009-of-00010.parquet
        threshold0.7-00000-of-00010.parquet
        ...
        threshold0.7-00009-of-00010.parquet
        top-images-00000-of-00010.parquet
        ...
        top-images-00009-of-00010.parquet
```

`data/imagenet/v2/` contains caption metadata, not the original ImageNet-v2 images.

## 5. Set your local paths

Edit [minit2i-torch/mini_t2i/settings.py](minit2i-torch/mini_t2i/settings.py).
Change `DATA_ROOT` and `OUTPUT_ROOT` only if you don't use these defaults:

```python
DATA_ROOT = Path.home() / "datasets"
OUTPUT_ROOT = Path.home() / "pristine-models-runs"
```

With these values, the code looks for:

| Resource | Location |
| --- | --- |
| Original ImageNet-1k parquet images | `$HOME/datasets/imagenet/data/` |
| Captions, settings, and reference images | `<repository>/data/imagenet/` |
| Default training outputs | `$HOME/pristine-models-runs/` |

`IMAGENET_CAPTIONS_ROOT` is already derived from the checkout location; leave it unchanged when using the download layout above.

## 6. Train

Run all training commands below from the code directory:

```bash
cd minit2i-torch
export NGPUS=1
wandb login
```

Set `NGPUS` to the number of GPUs on this machine, for example `8`. These are
single-node commands. Start with a micro-batch that fits your GPU; reduce
`--micro-batch-size` if you run out of memory.

You can also set `micro_batch_size` in the selected YAML
config. Command-line flags overrides these values; omit `--micro-batch-size` from the examples below to use the YAML settings.

The paper experiments use a **global batch size of 1,024 images per optimizer
step**. Make sure, `n_nodes * n_gpus_per_node * micro_batch_size = 1024`.

| Model | Conditioning/data | Config | Trainer |
| --- | --- | --- | --- |
| Safe | Whole captions; safe partition with configured leakage exclusions | [safe.yaml](minit2i-torch/configs/imagenet/safe.yaml) | `mini_t2i.train` |
| Full | Whole captions; all available training records | [full.yaml](minit2i-torch/configs/imagenet/full.yaml) | `mini_t2i.train` |
| Pristine | Residual captions on the shared branch; subject captions on safe/unsafe branches | [pristine.yaml](minit2i-torch/configs/imagenet/pristine.yaml) | `mini_t2i.pristine_train` |

Training logs to Weights & Biases. Set `project` and, optionally, `wandb_entity`
in the selected YAML config.
Complete both image downloads and V2 conversion before training.
The text encoder (`google/flan-t5-large`) is downloaded automatically on first use.
Training uses the step count and optimizer settings in the selected YAML.

### Safe

```bash
torchrun --standalone --nproc_per_node="$NGPUS" \
  -m mini_t2i.train \
  --config-file configs/imagenet/safe.yaml \
  --output-dir ../outputs/safe \
  --micro-batch-size 32
```

### Full

```bash
torchrun --standalone --nproc_per_node="$NGPUS" \
  -m mini_t2i.train \
  --config-file configs/imagenet/full.yaml \
  --output-dir ../outputs/full \
  --micro-batch-size 32
```

### Pristine: three branches

```bash
torchrun --standalone --nproc_per_node="$NGPUS" \
  -m mini_t2i.pristine_train \
  --config-file configs/imagenet/pristine.yaml \
  --output-dir ../outputs/pristine \
  --micro-batch-size 32
```

## 7. Generate images and evaluate

Run these commands from `minit2i-torch/` after downloading the captions/manifests
in section 3 and configuring paths in section 5. Image generation does not require
the original ImageNet images; FID/PRC evaluation does.

### Download pretrained checkpoints

Download from the **model** repository, not the dataset repository with the same
name. `--local-dir ..` preserves the remote `weights/imagenet/dogs/` hierarchy at
the repository root:

```bash
hf download AniSundar18/PristineModels \
  --repo-type model \
  --include "weights/imagenet/dogs/*.pt" \
  --local-dir ..
```

This downloads `safe.pt`, `full.pt`, and `pristine.pt`. Each checkpoint was selected based on the **lowest FID on the safe
validation data**. To use your own training checkpoint instead, substitute its path below.

The official inference launchers below use **eight GPUs (0-7)** and generate all
four benchmarks: ImageNet, clear-background, implicit, and compositional. Each
GPU processes a disjoint rank slice. The examples use seed 42, CFG 3, 200 steps,
and **batch size 512 per GPU**, not across all GPUs. Free sufficient GPU memory
before running; lower the batch argument explicitly if necessary.

### Safe: whole-caption inference

```bash
bash inference/inference_scripts/infer_whole.sh \
  ../weights/imagenet/dogs/safe.pt ../outputs/safe-images 3 200 512
```

### Pristine: split-caption inference

```bash
bash inference/inference_scripts/infer_pristine_split.sh \
  ../weights/imagenet/dogs/pristine.pt ../outputs/pristine-safe-images safe 200 512
```

The Pristine launcher fixes CFG at 3. Use `unsafe` instead of `safe` as its third
argument and a different output directory to evaluate the unsafe branch.

### Full: whole-caption inference

```bash
bash inference/inference_scripts/infer_whole.sh \
  ../weights/imagenet/dogs/full.pt ../outputs/full-images 3 200 512
```

### Evaluate all three runs

Use the official evaluation launcher after generation. Full evaluation requires
both original ImageNet image datasets from section 4 and the compositional
reference images from section 3. `--parallel` runs the fidelity, VQA, accuracy,
and TPIPS metric groups concurrently on the **first four visible GPUs**; it does not use eight
evaluation workers. Scoring models download automatically on first use.

```bash
for run in safe-images pristine-safe-images full-images; do
  bash evaluations/eval_scripts/eval.sh \
    "../outputs/$run" "../outputs/$run/evaluation.json" \
    --parallel \
    --metrics fid prc accuracy vqa clearbg_accuracy implicit_accuracy tpips
done
```

This scores ImageNet FID, precision/recall, accuracy and VQAScore, clear-background
and implicit leakage accuracy, and compositional TPIPS. Each `evaluation.json`
contains the requested metrics, manifest details, and aggregate scores. To score only ImageNet
accuracy and VQAScore without original ImageNet images, use the same launcher with
`--metrics accuracy vqa` and a different output JSON path.

### Reference results

Results for the pretrained Dog-setting MMJiT models using the evaluation commands
above.

| Evaluation group | Metric | Safe model | Pristine (safe branch) | Full model |
| --- | --- | ---: | ---: | ---: |
| Safe | FID | 8.2161 | 7.2494 | 6.8766 |
| Safe | VQAScore | 0.6261 | 0.7478 | 0.7858 |
| Safe | Precision (PRC) | 0.7591 | 0.7668 | 0.6881 |
| Safe | Recall (PRC) | 0.5525 | 0.6139 | 0.7071 |
| Safe | Accuracy (ClearBG) | 0.7901 | 0.9084 | 0.8305 |
| Safe | TPIPS (overall similarity) | 0.6890 | 0.7408 | 0.7706 |
| Unsafe | Precision (PRC) | 0.5959 | 0.5510 | 0.8099 |
| Unsafe | Recall (PRC) | 0.0480 | 0.2167 | 0.8001 |
| Unsafe | Accuracy (ClearBG) | 0.0015 | 0.0000 | 0.5544 |
| Unsafe | Accuracy (implicit prompts) | 0.0114 | 0.0609 | 0.3983 |
| Balanced | Precision (PRC) | 0.5274 | 0.5664 | 0.2979 |
| Balanced | Recall (PRC) | 0.6992 | 0.6884 | 0.3117 |
| Balanced | Accuracy (ClearBG) | 0.8821 | 0.9520 | 0.5800 |

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{sundararajan2026pristine,
  title         = {Pristine Models},
  author        = {Sundara Rajan, Anirudh and Singh, Krishna Kumar and Zhang, Richard and Lee, Yong Jae},
  year          = {2026},
  eprint        = {2610.14091},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2610.14091},
  url           = {https://arxiv.org/abs/2610.14091}
}
```

## Attribution and license

This project builds on [MiniT2I's PyTorch implementation](https://github.com/Hope7Happiness/minit2i-torch)
and the [original MiniT2I JAX implementation](https://github.com/PeppaKing8/minit2i-jax).
The upstream code's MIT license is preserved in
[minit2i-torch/LICENSE](minit2i-torch/LICENSE). Dataset and pretrained-model access
and licensing are governed by their respective sources; the code license does not
relicense ImageNet or other third-party assets.
