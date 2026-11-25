from pathlib import Path

# import t2v_metrics  # commented out: pulls in t2v_metrics at import time even for callers that never use VQAScorer
from tqdm import tqdm


class VQAScorer:
    """Wraps ``t2v_metrics.VQAScore`` behind the same ``score_per_sample`` shape as
    ``CLIPScorer``, except ``images`` here are file paths (t2v_metrics' native input)
    rather than a numpy batch. Uses ``batch_forward`` so pairs are actually batched
    through the VLM instead of one forward pass per pair.
    """

    def __init__(self, model_name_or_path: str = "gemma-4-31b-it", device: str | None = None):
        import t2v_metrics
        self._scorer = t2v_metrics.VQAScore(model=model_name_or_path)

    def score_per_sample(
        self, image_paths: list[str | Path], prompts: list[str], batch_size: int = 16
    ) -> list[float]:
        if len(image_paths) != len(prompts):
            raise ValueError(f"len(image_paths)={len(image_paths)} != len(prompts)={len(prompts)}")
        scores: list[float] = []
        pbar = tqdm(range(0, len(image_paths), batch_size), desc="  VQAScore",
                    unit="batch", dynamic_ncols=True)
        for i in pbar:
            chunk_imgs = image_paths[i : i + batch_size]
            chunk_prompts = prompts[i : i + batch_size]
            dataset = [
                {"images": [str(img)], "texts": [prompt]}
                for img, prompt in zip(chunk_imgs, chunk_prompts)
            ]
            chunk_scores = self._scorer.batch_forward(dataset=dataset, batch_size=len(dataset))  # (n, 1, 1)
            scores.extend(chunk_scores[:, 0, 0].tolist())
            pbar.set_postfix(mean=f"{sum(scores)/len(scores):.3f}", n=len(scores))
        return scores
