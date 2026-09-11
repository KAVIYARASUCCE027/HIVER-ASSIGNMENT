"""Shared data loading + input construction for the Phase F baselines.

Centralized here so majority.py and tfidf_baseline.py are guaranteed to
receive the exact same input per CLAUDE.md's baseline-fairness requirement:
"Both baselines must receive the same customer-message input."

Two input variants:
  A. customer_message only
  B. preceding conversation context + customer_message, concatenated

Golden set is read only for scoring; nothing here derives a class label from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

GOLDEN_PATH = Path("data/golden/golden_200.jsonl")
INTENTS_PATH = Path("config/intents.yaml")


def load_intent_names() -> list[str]:
    with INTENTS_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [i["name"] for i in data["intents"]]


def load_golden() -> list[dict]:
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def build_input_text(customer_message: str, conversation: list[dict], variant: str) -> str:
    if variant == "A":
        return customer_message
    if variant == "B":
        context = " ".join(m["text"] for m in conversation)
        return f"{context} {customer_message}".strip() if context else customer_message
    raise ValueError(f"Unknown variant: {variant!r}, expected 'A' or 'B'")
