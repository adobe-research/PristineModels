from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mini_t2i.settings import DATA_ROOT, IMAGENET_CAPTIONS_ROOT, IMAGENET_PARQUET_ROOT

VQA_MODEL = "qwen3.5-9b"
ACCURACY_MODEL = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
SOURCES = ("v1", "v2")
METRICS_DIR = REPO / "evaluations/metrics"
IMAGENET_METRICS = ("fid", "prc", "vqa", "accuracy")
LEAKAGE_METRICS = {"clearbg_accuracy": "clearbg", "implicit_accuracy": "implicit"}
ACCURACY_METRICS = ("accuracy", *LEAKAGE_METRICS)
METRICS = (*IMAGENET_METRICS, *LEAKAGE_METRICS, "tpips")
TPIPS_MATERIALS = ("moss", "hay", "gold", "algae", "resin", "bronze", "foam", "ceramic", "clay", "stone")
TPIPS_FACTORS = {
    "object_sim": "the object's identity, ignoring its texture and surface",
    "material_sim": "the object's texture and surface, ignoring its identity",
    "overall_sim": "Object Identity and object material",
}


def load_samples(output_dir: Path, manifest: Path) -> tuple[list[dict], dict]:
    samples = []
    expected = Counter()
    seen = set()
    with manifest.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            source, group = row["source"], row["group"]
            if source not in SOURCES:
                raise ValueError(f"Unknown source: {source}; expected one of {SOURCES}")
            if group not in {"safe", "unsafe"}:
                raise ValueError(f"Unknown group: {group}")
            index = int(row["global_row_index"])
            key = (source, index)
            if key in seen:
                raise ValueError(f"Duplicate manifest ID: {key}")
            seen.add(key)
            expected[(source, group)] += 1
            path = output_dir / "imagenet" / source / f"{index:06d}.png"
            if not path.is_file():
                raise FileNotFoundError(f"Expected generated image is missing: {path}")
            samples.append({**row, "path": path})
    if not expected:
        raise ValueError("Manifest contains no rows for the requested sources")
    coverage = {
        source: {
            group: expected[(source, group)]
            for group in ("safe", "unsafe")
        }
        for source in SOURCES
    }
    return sorted(samples, key=lambda row: (row["source"], int(row["global_row_index"]))), coverage


def load_leakage_samples(output_dir: Path, benchmark: str, manifests: list[Path], label_setting: Path) -> list[dict]:
    with label_setting.open() as stream:
        categories = json.load(stream)["label_to_categories"]
    label_groups = {}
    for group in ("safe", "unsafe"):
        for label in categories[group]:
            label = int(label)
            if label in label_groups:
                raise ValueError(f"Label {label} appears in both groups in {label_setting}")
            label_groups[label] = group
    if not manifests:
        raise FileNotFoundError(f"No captions_merged.jsonl manifests found for {benchmark}")
    samples, seen = [], set()
    for manifest in manifests:
        with manifest.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                index, label = int(row["global_row_index"]), int(row["label"])
                if index < 0 or not 0 <= label < 1000:
                    raise ValueError(f"Invalid {benchmark} ID or ImageNet label: {index}, {label}")
                if index in seen:
                    raise ValueError(f"Duplicate {benchmark} manifest ID: {index}")
                seen.add(index)
                if label not in label_groups:
                    raise ValueError(f"Label {label} is absent from {label_setting}")
                path = output_dir / benchmark / f"{index:06d}.png"
                if not path.is_file():
                    raise FileNotFoundError(f"Expected generated image is missing: {path}")
                samples.append({**row, "label": label, "group": label_groups[label], "path": path})
    for group in ("safe", "unsafe"):
        if not any(row["group"] == group for row in samples):
            raise ValueError(f"No {group} caption rows for {benchmark}")
    return sorted(samples, key=lambda row: int(row["global_row_index"]))


def balanced_score(safe: float, unsafe: float) -> float:
    suppression = 1.0 - unsafe
    denominator = safe + suppression
    return 2.0 * safe * suppression / denominator if denominator else 0.0


def write_result(path: Path, result: dict) -> None:
    scores = result["scores"]
    for benchmark in (scores, scores.get("clearbg", {}), scores.get("implicit", {})):
        safe, unsafe = benchmark.get("safe", {}), benchmark.get("unsafe", {})
        balanced = {metric: balanced_score(safe[metric], unsafe[metric]) for metric in ("precision", "recall") if metric in safe and metric in unsafe}
        if "accuracy" in safe and "accuracy" in unsafe:
            balanced["accuracy"] = {aggregation: balanced_score(safe["accuracy"][aggregation], unsafe["accuracy"][aggregation]) for aggregation in ("macro", "micro") if aggregation in safe["accuracy"] and aggregation in unsafe["accuracy"]}
        if balanced:
            benchmark["balanced"] = balanced
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_images(samples: list[dict]) -> None:
    for row in samples:
        if not os.path.isfile(row["path"]):
            raise ValueError(f"Missing image: {row['path']}")


def prepare_fidelity(samples: list[dict], roots: dict[str, Path], temporary: Path) -> tuple[Path, Path]:
    from io import BytesIO
    import pyarrow.parquet as parquet
    from PIL import Image

    fake, real = temporary / "fake", temporary / "real"
    fake.mkdir()
    real.mkdir()
    shards = defaultdict(list)
    for row in samples:
        shards[(row["source"], row["shard_file"])].append(row)
        stem = f"{row['source']}_{int(row['global_row_index']):06d}.png"
        (fake / stem).symlink_to(row["path"])
    for (source, shard), rows in sorted(shards.items()):
        table = parquet.read_table(roots[source] / shard, columns=["image"])
        for row in rows:
            index = int(row["row_in_shard"])
            if not 0 <= index < table.num_rows:
                raise ValueError(f"Invalid row_in_shard={index} in {source}/{shard}")
            field = table["image"][index].as_py()
            source_image = BytesIO(field["bytes"]) if isinstance(field, dict) and field.get("bytes") is not None else field["path"] if isinstance(field, dict) else field
            with Image.open(source_image) as original:
                image = original.convert("RGB")
            width, height = image.size
            scale = 256 / min(width, height)
            image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
            left, top = (image.width - 256) // 2, (image.height - 256) // 2
            image = image.crop((left, top, left + 256, top + 256))
            image.save(real / f"{source}_{int(row['global_row_index']):06d}.png")
    return fake, real


def run_val_fid(cfg, model, text_encoder, tokenizer, device, dtype, *, split: bool) -> float:
    import torch
    import torch.distributed as dist
    from PIL import Image
    from inference.infer_whole import encode, init_noise, _save_png_atomic
    from mini_t2i.diffusion import euler_sample, pristine_euler_sample

    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1

    def check_errors(error):
        errors = [error]
        if distributed:
            errors = [None] * world_size
            dist.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError(f"FID validation failed: {errors}")

    temporary = None
    location = [None, None]
    if rank == 0:
        try:
            Path(cfg.val_fid_dir).mkdir(parents=True, exist_ok=True)
            temporary = tempfile.TemporaryDirectory(prefix="imagenet-val-", dir=cfg.val_fid_dir)
            location[0] = str(Path(temporary.name).resolve())
        except Exception as error:
            location[1] = repr(error)
    if distributed:
        dist.broadcast_object_list(location, src=0)
    was_training = model.training
    try:
        check_errors(location[1])
        root = Path(location[0])
        error = None
        try:
            if not root.is_dir():
                raise ValueError("val_fid_dir must be visible to every rank; use a shared directory for multi-node training")
            if cfg.val_fid_batch_size <= 0:
                raise ValueError("val_fid_batch_size must be positive")
            if not cfg.val_fid_sources or set(cfg.val_fid_sources) - set(SOURCES):
                raise ValueError(f"val_fid_sources must select from {SOURCES}")
            rows = []
            with Path(cfg.val_fid_manifest).open() as stream:
                for line in stream:
                    if line.strip():
                        row = json.loads(line)
                        if row["group"] == "safe" and row["source"] in cfg.val_fid_sources:
                            rows.append(row)
            if not rows:
                raise ValueError("FID manifest contains no selected safe rows")
            rows.sort(key=lambda row: (row["source"], int(row["global_row_index"])))
            if len({(row["source"], int(row["global_row_index"])) for row in rows}) != len(rows):
                raise ValueError("FID manifest contains duplicate safe image IDs")
            for row in rows:
                row["path"] = root / "imagenet" / row["source"] / f"{int(row['global_row_index']):06d}.png"
            local_rows = rows[rank::world_size]
            model.eval()
            with torch.no_grad(), torch.random.fork_rng(devices=[device]):
                for start in range(0, len(local_rows), cfg.val_fid_batch_size):
                    batch = local_rows[start:start + cfg.val_fid_batch_size]
                    noise = init_noise([row["global_row_index"] for row in batch], cfg.seed, cfg.image_size, cfg.noise_scale, device, dtype)
                    options = dict(steps=cfg.n_T, cfg_scale=cfg.cfg_scale, noise_scale=cfg.noise_scale, init_noise=noise)
                    if split:
                        text_mode, mask_mode = encode([row["subject_caption"] for row in batch], tokenizer, text_encoder, cfg.prompt_length, device, dtype)
                        text_shared, mask_shared = encode([row["residual_caption"] for row in batch], tokenizer, text_encoder, cfg.prompt_length, device, dtype)
                        samples = pristine_euler_sample(model, text_mode, mask_mode, text_shared, mask_shared, cfg.image_size, mode="safe", **options)
                    else:
                        text, mask = encode([row["whole_caption"] for row in batch], tokenizer, text_encoder, cfg.prompt_length, device, dtype)
                        samples = euler_sample(model, text, mask, cfg.image_size, **options)
                    for row, image in zip(batch, samples):
                        row["path"].parent.mkdir(parents=True, exist_ok=True)
                        pixels = ((image.float().cpu() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
                        _save_png_atomic(Image.fromarray(pixels), row["path"])
        except Exception as exception:
            error = repr(exception)
        check_errors(error)
        result = [None, None]
        if rank == 0:
            try:
                import torch_fidelity

                real_dirs = {source: Path(getattr(cfg, f"val_fid_{source}_real_dir")) for source in cfg.val_fid_sources}
                fake, real = prepare_fidelity(rows, real_dirs, root)
                values = torch_fidelity.calculate_metrics(input1=str(fake), input2=str(real), cuda=True, fid=True, prc=False, isc=False, kid=False, feature_extractor="inception-v3-compat", feature_layer_fid="2048", cache=False, cache_root=str(root / "features"), verbose=True)
                result[0] = float(values["frechet_inception_distance"])
            except Exception as exception:
                result[1] = repr(exception)
        if distributed:
            dist.broadcast_object_list(result, src=0)
        check_errors(result[1])
        return result[0]
    finally:
        model.train(was_training)
        if temporary is not None:
            temporary.cleanup()


def score_accuracy(samples: list[dict], classifier, batch_size: int) -> dict:
    import numpy as np
    from PIL import Image

    hits = defaultdict(list)
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        images = []
        for row in batch:
            with Image.open(row["path"]) as image:
                images.append(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0)
        labels = [int(row["label"]) for row in batch]
        values = classifier(np.stack(images), labels)
        if len(values) != len(batch):
            raise ValueError("ImageNet classifier returned an unexpected number of scores")
        for row, label, hit in zip(batch, labels, values):
            hits[label].append(bool(hit))

    def summarize(classes: dict) -> dict:
        counts = {str(label): {"correct": sum(values), "n": len(values), "accuracy": sum(values) / len(values)} for label, values in sorted(classes.items())}
        total = sum(entry["n"] for entry in counts.values())
        return {
            "macro": sum(entry["accuracy"] for entry in counts.values()) / len(counts),
            "micro": sum(entry["correct"] for entry in counts.values()) / total,
            "n": total,
        }

    return summarize(hits)


def load_tpips_samples(output_dir: Path, manifest: Path, reference_dir: Path) -> list[dict]:
    def read_rows(path: Path) -> list[dict]:
        with path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        indices = [int(row["global_row_index"]) for row in rows]
        pairs = [(int(row["label"]), row["child"], row["material"]) for row in rows]
        if len(set(indices)) != len(rows) or len(set(pairs)) != len(rows):
            raise ValueError(f"Duplicate compositional ID or subject/material pair in {path}")
        return rows

    rows = sorted(read_rows(manifest), key=lambda row: int(row["global_row_index"]))
    subjects = defaultdict(set)
    for row in rows:
        subjects[(int(row["label"]), row["child"])].add(row["material"])
    if len(rows) != 1310 or len(subjects) != 131 or any(materials != set(TPIPS_MATERIALS) for materials in subjects.values()):
        raise ValueError("TPIPS manifest must contain all 131 safe subjects paired with exactly the 10 paper materials")
    if [int(row["global_row_index"]) for row in rows] != list(range(1310)):
        raise ValueError("Paper compositional IDs must be contiguous from 0 through 1309")
    samples = []
    for row in rows:
        path = output_dir / "compositional" / f"{int(row['global_row_index']):06d}.png"
        reference_index = int(row["global_row_index"])
        reference_path = reference_dir / f"{reference_index:06d}.png"
        for image_path in (path, reference_path):
            if not image_path.is_file():
                raise FileNotFoundError(f"Expected compositional image is missing: {image_path}")
        samples.append({**row, "path": path, "reference_path": reference_path, "reference_index": reference_index})
    return samples


def score_tpips(samples: list[dict], model, batch_size: int) -> dict:
    import torch
    from PIL import Image
    from tqdm import tqdm

    records = []
    for start in tqdm(range(0, len(samples), batch_size), desc="TPIPS", unit="batch"):
        batch = samples[start:start + batch_size]
        generated, references = [], []
        for row in batch:
            with Image.open(row["path"]) as image:
                generated.append(image.convert("RGB"))
            with Image.open(row["reference_path"]) as image:
                references.append(image.convert("RGB"))
        batch_records = [{key: row[key] for key in ("global_row_index", "child", "material", "reference_index")} for row in batch]
        with torch.inference_mode():
            for key, factor in TPIPS_FACTORS.items():
                similarities = model.similarity(references, generated, factor=factor)
                if similarities.ndim != 1 or len(similarities) != len(batch) or not torch.isfinite(similarities).all():
                    raise ValueError(f"TPIPS returned invalid scores for {key}")
                distances = 1.0 - similarities
                for record, similarity, distance in zip(batch_records, similarities.tolist(), distances.tolist()):
                    record[key] = float(similarity)
                    record[key.replace("_sim", "_dist")] = float(distance)
        records.extend(batch_records)

    metric_keys = [key for factor in TPIPS_FACTORS for key in (factor, factor.replace("_sim", "_dist"))]

    def summarize(items: list[dict]) -> dict:
        return {"n": len(items), **{key: sum(row[key] for row in items) / len(items) for key in metric_keys}}

    return {"model_type": "early_fusion", "factors": TPIPS_FACTORS, "overall": summarize(records)}


def evaluate(args, samples: list[dict], result: dict) -> None:
    import importlib.util

    dependencies = {"fid": ["torch_fidelity", "pyarrow"], "prc": ["torch_fidelity", "pyarrow"], "vqa": ["t2v_metrics"], "accuracy": ["timm", "transformers"], "tpips": ["tpips"]}
    dependencies.update({metric: dependencies["accuracy"] for metric in LEAKAGE_METRICS})
    missing = sorted({name for metric in args.metrics for name in dependencies[metric] if importlib.util.find_spec(name) is None})
    if missing:
        raise RuntimeError(f"Missing requested metric dependencies: {', '.join(missing)}. Select installed metrics with --metrics.")
    if "vqa" in args.metrics or any(metric in args.metrics for metric in ACCURACY_METRICS):
        sys.path.insert(0, str(args.metrics_dir))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Metric evaluation requires an available CUDA GPU")
    groups = {group: [row for row in samples if row["group"] == group] for group in ("safe", "unsafe")}
    if any(metric in IMAGENET_METRICS for metric in args.metrics):
        for group in ("safe", "unsafe") if "accuracy" in args.metrics or "prc" in args.metrics else ("safe",):
            if not groups[group]:
                raise ValueError(f"No available images for {group}")
    print("Validating all requested benchmark images before scoring", flush=True)
    validate_images(samples)
    for rows in args.leakage_samples.values():
        validate_images(rows)
    validate_images(args.tpips_samples)
    validate_images([{"path": row["reference_path"]} for row in args.tpips_samples])
    scores = result["scores"]
    if "fid" in args.metrics or "prc" in args.metrics:
        import torch_fidelity

        for group in ("safe", "unsafe") if "prc" in args.metrics else ("safe",):
            print(f"Scoring {group} fidelity metrics on pooled v1/v2 images", flush=True)
            with tempfile.TemporaryDirectory(prefix="imagenet-eval-", dir=args.temp_dir) as directory:
                fake, real = prepare_fidelity(groups[group], {"v1": args.v1_real_dir, "v2": args.v2_real_dir}, Path(directory))
                if group == "safe" and "fid" in args.metrics:
                    values = torch_fidelity.calculate_metrics(input1=str(fake), input2=str(real), cuda=True, fid=True, prc=False, isc=False, kid=False, feature_extractor="inception-v3-compat", feature_layer_fid="2048", cache=False, cache_root=str(Path(directory) / "features"), verbose=True)
                    scores[group]["fid"] = float(values["frechet_inception_distance"])
                    write_result(args.output_json, result)
                if "prc" in args.metrics:
                    values = torch_fidelity.calculate_metrics(input1=str(fake), input2=str(real), cuda=True, fid=False, prc=True, isc=False, kid=False, feature_extractor="vgg16", feature_layer_prc="fc2_relu", cache=False, cache_root=str(Path(directory) / "features"), verbose=True)
                    for name in ("precision", "recall"):
                        key = name if name in values else f"precision_recall_{name}"
                        scores[group][name] = float(values[key])
                    write_result(args.output_json, result)
    if "vqa" in args.metrics:
        from compute_vqascore import VQAScorer

        print(f"Scoring safe dense-caption VQAScore with {args.vqa_model}", flush=True)
        scorer = VQAScorer(model_name_or_path=args.vqa_model, device="cuda:0")
        values = scorer.score_per_sample([row["path"] for row in groups["safe"]], [row["whole_caption"] for row in groups["safe"]], batch_size=args.vqa_batch_size)
        if len(values) != len(groups["safe"]):
            raise ValueError("VQAScore returned an unexpected number of scores")
        scores["safe"]["vqascore"] = sum(values) / len(values)
        del scorer
        torch.cuda.empty_cache()
        write_result(args.output_json, result)
    if any(metric in args.metrics for metric in ACCURACY_METRICS):
        from compute_imagenet_acc import ImageNetClassifier

        print(f"Scoring requested top-1 accuracies with {args.accuracy_model}", flush=True)
        classifier = ImageNetClassifier(model_name_or_path=args.accuracy_model, device="cuda:0")
        if "accuracy" in args.metrics:
            for group, rows in groups.items():
                scores[group]["accuracy"] = score_accuracy(rows, classifier, args.acc_batch_size)
                write_result(args.output_json, result)
        for benchmark, rows in args.leakage_samples.items():
            print(f"Scoring {benchmark} safe/unsafe top-1 accuracy", flush=True)
            scores[benchmark] = {}
            for group in ("safe", "unsafe"):
                selected = [row for row in rows if row["group"] == group]
                scores[benchmark][group] = {"accuracy": score_accuracy(selected, classifier, args.acc_batch_size)}
                write_result(args.output_json, result)
        del classifier
        torch.cuda.empty_cache()
    if "tpips" in args.metrics:
        import tpips

        pairs = args.tpips_samples
        print("Scoring compositional TPIPS with early_fusion", flush=True)
        model = tpips.load_model("early_fusion", device="cuda:0")
        scores["compositional"] = {"tpips": score_tpips(pairs, model, args.tpips_batch_size)}
        write_result(args.output_json, result)


def evaluate_parallel(args) -> None:
    import subprocess

    groups = (
        ("fidelity", ("fid", "prc")),
        ("vqa", ("vqa",)),
        ("accuracy", ACCURACY_METRICS),
        ("tpips", ("tpips",)),
    )
    parts_dir = args.output_json.parent / f"{args.output_json.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    forwarded = [argument for argument in sys.argv[1:] if argument != "--parallel"]
    workers = []
    try:
        for name, metrics in groups:
            selected = [metric for metric in args.metrics if metric in metrics]
            if not selected:
                continue
            output = parts_dir / f"{name}.json"
            gpu = args.gpu + len(workers)
            command = [sys.executable, str(Path(__file__).resolve()), *forwarded,
                       "--metrics", *selected, "--gpu", str(gpu), "--output-json", str(output)]
            print(f"Launching {name} on GPU {gpu}: {', '.join(selected)}", flush=True)
            workers.append((name, output, subprocess.Popen(command)))
        failed = [name for name, _, process in workers if process.wait() != 0]
        if failed:
            raise RuntimeError(f"Parallel evaluation failed: {', '.join(failed)}; partial results in {parts_dir}")
    finally:
        for _, _, process in workers:
            if process.poll() is None:
                process.terminate()
        for _, _, process in workers:
            process.wait()

    results = []
    for name, output, _ in workers:
        with output.open() as stream:
            result = json.load(stream)
        if result["status"] != "complete":
            raise RuntimeError(f"Incomplete {name} result: {output}")
        results.append(result)
    merged = next((result for result in results if result["manifest"] is not None), results[0]).copy()
    merged["requested_metrics"] = args.metrics
    merged["coverage"] = {}
    merged["leakage_benchmarks"] = {}
    merged["scores"] = {"safe": {}, "unsafe": {}}
    for result in results:
        merged["coverage"].update(result["coverage"])
        merged["leakage_benchmarks"].update(result["leakage_benchmarks"])
        for group, scores in result["scores"].items():
            merged["scores"].setdefault(group, {}).update(scores)
        if "compositional" in result:
            merged["compositional"] = result["compositional"]
    write_result(args.output_json, merged)
    print(f"Results: {args.output_json}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all four generation benchmarks without resampling; select metrics or score the full package.")
    parser.add_argument("output_dir", type=Path, help="Overall generation root containing imagenet/, clearbg/, implicit/, and compositional/")
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--manifest", type=Path, default=IMAGENET_CAPTIONS_ROOT / "settings/Dog/equalized_val.jsonl")
    parser.add_argument("--clearbg-manifest", type=Path, default=IMAGENET_CAPTIONS_ROOT / "clearbg/captions_merged.jsonl")
    parser.add_argument("--implicit-dir", type=Path, default=IMAGENET_CAPTIONS_ROOT / "implicit/whole_prompts/val")
    parser.add_argument("--label-setting", type=Path, default=IMAGENET_CAPTIONS_ROOT / "settings/Dog/imagenet_dog_setting_labels.json", help="Safe/unsafe ImageNet label partition for leakage accuracy")
    parser.add_argument("--v1-real-dir", type=Path, default=IMAGENET_PARQUET_ROOT)
    parser.add_argument("--v2-real-dir", type=Path, default=DATA_ROOT / "imagenet/v2/data")
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--temp-dir", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--parallel", action="store_true", help="Run metric groups on up to four consecutive GPUs starting at --gpu; merge results after all succeed")
    parser.add_argument("--vqa-model", default=VQA_MODEL)
    parser.add_argument("--accuracy-model", default=ACCURACY_MODEL)
    parser.add_argument("--vqa-batch-size", type=int, default=16)
    parser.add_argument("--acc-batch-size", type=int, default=128)
    parser.add_argument("--tpips-manifest", type=Path, default=IMAGENET_CAPTIONS_ROOT / "compositional/benchmark_paper.jsonl")
    parser.add_argument("--tpips-reference-dir", type=Path, default=IMAGENET_CAPTIONS_ROOT / "compositional/qwen_reference", help="Local paper-numbered Qwen-Image reference PNG directory")
    parser.add_argument("--tpips-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.gpu < 0 or min(args.vqa_batch_size, args.acc_batch_size, args.tpips_batch_size) < 1:
        parser.error("GPU must be nonnegative and batch sizes must be positive")
    args.metrics = list(dict.fromkeys(args.metrics))
    args.output_dir = args.output_dir.resolve()
    args.output_json = args.output_json or args.output_dir / "evaluation/imagenet.json"
    if args.parallel:
        evaluate_parallel(args)
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"Requested metrics: {', '.join(args.metrics)}", flush=True)
    samples, source_coverage = load_samples(args.output_dir, args.manifest) if any(metric in IMAGENET_METRICS for metric in args.metrics) else ([], {})
    coverage = {}
    if source_coverage:
        coverage["imagenet"] = {
            group: sum(counts[group] for counts in source_coverage.values())
            for group in ("safe", "unsafe")
        }
    args.leakage_samples = {}
    leakage_metadata = {}
    for metric, benchmark in LEAKAGE_METRICS.items():
        if metric not in args.metrics:
            continue
        manifests = [args.clearbg_manifest] if benchmark == "clearbg" else sorted(args.implicit_dir.rglob("captions_merged.jsonl"))
        rows = load_leakage_samples(args.output_dir, benchmark, manifests, args.label_setting)
        args.leakage_samples[benchmark] = rows
        coverage[benchmark] = {group: sum(row["group"] == group for row in rows) for group in ("safe", "unsafe")}
        leakage_metadata[benchmark] = {
            "manifests": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in manifests],
            "label_setting": str(args.label_setting.resolve()),
            "label_setting_sha256": hashlib.sha256(args.label_setting.read_bytes()).hexdigest(),
        }
    args.tpips_samples = []
    if "tpips" in args.metrics:
        args.tpips_samples = load_tpips_samples(args.output_dir, args.tpips_manifest, args.tpips_reference_dir)
        coverage["compositional"] = {"pairs": len(args.tpips_samples)}
    result = {
        "status": "running",
        "output_dir": str(args.output_dir),
        "manifest": str(args.manifest.resolve()) if samples else None,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest() if samples else None,
        "sources": SOURCES if samples else (),
        "requested_metrics": args.metrics,
        "coverage": coverage,
        "leakage_benchmarks": leakage_metadata,
        "models": {"fid": "torch-fidelity inception-v3-compat (2048 features)", "prc": "torch-fidelity vgg16 (fc2_relu features)", "vqascore": args.vqa_model, "accuracy": args.accuracy_model},
        "protocol": {"sampling": "exact inference manifest; no additional equalization", "pooling": "v1/v2 pooled, not averaged scores", "vqa_caption": "whole_caption", "accuracy": "dense-caption top-1; macro is primary", "leakage_accuracy": "clearbg explicit and implicit whole-caption top-1 against manifest ImageNet labels; safe/unsafe from label setting; macro is primary; micro and per-class also reported", "real_preprocess": "RGB, shorter side 256, bicubic resize, center crop", "v1_real_dir": str(args.v1_real_dir), "v2_real_dir": str(args.v2_real_dir)},
        "scores": {"safe": {}, "unsafe": {}},
    }
    if "tpips" in args.metrics:
        result["compositional"] = {"manifest": str(args.tpips_manifest.resolve()), "reference_dir": str(args.tpips_reference_dir.resolve()), "materials": TPIPS_MATERIALS, "n": len(args.tpips_samples)}
    print(json.dumps(coverage, indent=2), flush=True)
    write_result(args.output_json, result)
    try:
        evaluate(args, samples, result)
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        write_result(args.output_json, result)
        raise
    write_result(args.output_json, result)
    print(f"Results: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()