"""Paths and JSONL loading for evaluation data."""

import json
from pathlib import Path

PROMPTS = Path(__file__).with_name("prompts")
SAFETY_PROMPTS = Path(__file__).resolve().parents[3] / "data/cc12m/eval/safety.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
