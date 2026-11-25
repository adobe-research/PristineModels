# CC12M evaluations

Run from `minit2i-torch/`. Download the [Safety prompts and benchmark safe IDs](https://huggingface.co/datasets/AniSundar18/PristineModels/tree/main/data/cc12m/eval)
into the checkout's `data/cc12m/eval/`:

```bash
hf download AniSundar18/PristineModels --repo-type dataset \
  --include 'data/cc12m/eval/*' --local-dir ..
```

| Benchmark | Prompts | Output |
|---|---:|---|
| DPG | 875 safe / 190 unsafe | `<item_id>.png`, 2x2 grid |
| PRISM (English) | 540 safe / 160 unsafe | `<track>/<index>.png` |
| Safety | 2,695 (245 per category) | `<key>.png` |

We report DPG and PRISM scores on the prompts in `dpg_safe_ids.txt` and
`prism_safe_ids.txt`. The full DPG and PRISM prompt files are separate local inputs.

Use one JSON object per line, replacing `<prompt text>` with the upstream prompt for that ID:

- DPG (`cc12m/eval/prompts/dpg.jsonl`): `{"item_id": "partiprompts97", "prompt": "<prompt text>", "split": "safe"}`
- PRISM (`cc12m/eval/prompts/prism.jsonl`): `{"item_id": "imagination_0", "prompt": "<prompt text>", "split": "safe"}`

## Inference

```bash
python -m cc12m.eval.generate --benchmark safety \
  --checkpoint /path/to/checkpoint.pt --outdir runs/full/safety
```

Run separately for each model and benchmark.

For safe-only DPG/PRISM scores, generate with `--split safe` into a fresh directory. For PRISM use the Qwen2.5-VL scorer (combined alignment and aesthetics).

## DPG-Bench scoring

Use a separate Python 3.12 environment for [DPG's mPLUG scorer](../../mini_t2i/scripts/evaluate_dpg_bench.py). Keep image generation in the main environment.

```bash
conda create -n dpg python=3.12 pip -y
conda activate dpg
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-dpg.txt
python -m pip check
```

The example uses CUDA 13.0 on Linux x86-64. For CPU-only scoring, use the `/whl/cpu` index instead.

Download the [original test parquet](https://huggingface.co/datasets/Jialuo21/DPG-Bench/resolve/16ab39cdc630e51072efaeeec24c16b2bb39a849/data/test-00000-of-00001.parquet) and pass its local path:

```bash
python -m mini_t2i.scripts.evaluate_dpg_bench \
  --image-root-path runs/full/dpg-safe --data /path/to/dpg-test.parquet \
  --mplug-ckpt iic/mplug_visual-question-answering_coco_large_en
```

`--mplug-ckpt` also accepts a local checkpoint directory.

## Safety judge

With vLLM installed, judge a local image folder using a vision-language model that answers Yes or No:

```bash
python -m cc12m.eval.judge_safety run --images runs/full/safety \
  --model /path/to/judge-checkpoint --outdir runs/judge/full
python -m cc12m.eval.judge_safety score runs/judge/full
```

`judgments.jsonl` stores only image IDs and verdicts. Keep `run.json` alongside it.

The unsafe-generation rate is the percentage of valid judgments where the requested concept was detected. Failed and missing judgments are counted separately. The reported results used Muse-Glimmer-30B with `xhigh` reasoning.
