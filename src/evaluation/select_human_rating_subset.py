"""Phase J: select the human-rating subset and build its blinded rating data.

Selection is stratified on signals available BEFORE looking at any reply
content or outcome (retrieval sufficiency, escalation decision, golden-set
sampling stratum) -- never on which version "looks better", so the subset
is not biased toward a desired agreement result. Blinding (which reply
shows as "A" vs "B") is assigned with a fixed, documented seed chosen
before any rating happened, not tuned afterward.

Run:
    python -m src.evaluation.select_human_rating_subset

Writes:
    data/evaluation/human_rating_subset.jsonl (25 examples, blinded)

Reads only frozen artifacts -- no LLM call, no modification to any Phase
G/H/I output.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_PATH = Path("data/golden/golden_200.jsonl")
NO_RAG_PATH = Path("evaluation/results/replies_no_rag.jsonl")
RAG_PATH = Path("evaluation/results/replies_rag.jsonl")
ESCALATION_PATH = Path("evaluation/results/escalation_results.json")
OUT_PATH = Path("data/evaluation/human_rating_subset.jsonl")

SUBSET_SIZE = 25
SELECTION_SEED = 20260910  # fixed before selection; documented, never re-rolled
BLINDING_SEED = 20260910   # fixed before any rating; documented, never re-rolled


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def select_subset_ids(golden: dict, rag: dict, escalation: dict) -> list[str]:
    sufficient = [gid for gid in golden if rag[gid]["retrieved_evidence"]]
    insufficient = [gid for gid in golden if not rag[gid]["retrieved_evidence"]]
    escalated = [gid for gid in golden if escalation[gid]["policy_action"] == "ESCALATE"]
    difficult = [gid for gid in golden if golden[gid]["sampling_stratum"].startswith("difficult:")]
    rare = [gid for gid in golden if golden[gid]["sampling_stratum"].startswith("rare_intent:")]

    rng = random.Random(SELECTION_SEED)
    selected: set[str] = set()

    def take(pool: list[str], n: int) -> None:
        avail = [g for g in pool if g not in selected]
        rng.shuffle(avail)
        selected.update(avail[:n])

    # Stratified draw covering: difficult/ambiguous cases, rare intents,
    # insufficient-evidence cases (where No-RAG and RAG most plausibly
    # diverge), escalated cases, and a plain sufficient-evidence sample --
    # deliberately NOT stratified by "which reply looks better", since that
    # would bias the human study toward a predetermined result.
    take(difficult, 6)
    take(rare, 4)
    take(insufficient, 6)
    take(escalated, 4)
    take(sufficient, 5)
    return sorted(selected)


def build_subset(subset_ids: list[str], golden: dict, no_rag: dict, rag: dict) -> list[dict]:
    rng = random.Random(BLINDING_SEED)
    records = []
    for gid in subset_ids:
        g, nr, rr = golden[gid], no_rag[gid], rag[gid]
        order = ["no_rag", "rag"] if rng.random() < 0.5 else ["rag", "no_rag"]
        records.append({
            "example_id": gid,
            "customer_message": g["customer_message"],
            "conversation": g["conversation"],
            "true_intent": g["intent"],
            "predicted_intent": nr["predicted_intent"],
            "sampling_stratum": g["sampling_stratum"],
            "label_A_version": order[0],
            "label_B_version": order[1],
            "reply_A": nr["reply"] if order[0] == "no_rag" else rr["reply"],
            "reply_B": nr["reply"] if order[1] == "no_rag" else rr["reply"],
            "evidence_A": None if order[0] == "no_rag" else rr["retrieved_evidence"],
            "evidence_B": None if order[1] == "no_rag" else rr["retrieved_evidence"],
        })
    return records


def main() -> None:
    golden = {g["id"]: g for g in _load_jsonl(GOLDEN_PATH)}
    no_rag = {r["example_id"]: r for r in _load_jsonl(NO_RAG_PATH)}
    rag = {r["example_id"]: r for r in _load_jsonl(RAG_PATH)}
    escalation = {r["example_id"]: r for r in json.loads(ESCALATION_PATH.read_text(encoding="utf-8"))["per_example"]}

    subset_ids = select_subset_ids(golden, rag, escalation)
    assert len(subset_ids) == SUBSET_SIZE, f"expected {SUBSET_SIZE}, got {len(subset_ids)}"

    records = build_subset(subset_ids, golden, no_rag, rag)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), OUT_PATH)


if __name__ == "__main__":
    main()
