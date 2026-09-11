"""Merge hand-authored labels into the final golden evaluation set.

This is the one and only place `data/golden/golden_200.jsonl` is produced.
It performs no labeling itself -- it reads a fixed LABELS mapping (id ->
(intent, expected_action, expected_reason, label_notes)) that was produced by
careful, per-example reading of every candidate against
`data/golden/LABELING_GUIDE.md`, and merges it onto
`data/golden/golden_candidates.jsonl`.

Run:
    python -m src.evaluation.build_golden_200
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.evaluation.golden_labels import LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CANDIDATES_PATH = Path("data/golden/golden_candidates.jsonl")
OUT_PATH = Path("data/golden/golden_200.jsonl")


def main() -> None:
    candidates = [json.loads(l) for l in CANDIDATES_PATH.open(encoding="utf-8")]
    ids = {c["id"] for c in candidates}
    missing = ids - set(LABELS)
    extra = set(LABELS) - ids
    if missing:
        raise ValueError(f"{len(missing)} candidates have no label: {sorted(missing)[:5]}...")
    if extra:
        raise ValueError(f"{len(extra)} labels have no matching candidate: {sorted(extra)[:5]}...")

    out = []
    for c in candidates:
        intent, action, reason, notes = LABELS[c["id"]]
        out.append({
            "id": c["id"],
            "conversation_id": c["conversation_id"],
            "conversation": c["conversation"],
            "customer_message": c["customer_message"],
            "intent": intent,
            "expected_action": action,
            "expected_reason": reason,
            "label_notes": notes,
            "suggested_intent": c["suggested_intent"],  # non-authoritative, see LABELING_GUIDE.md
            "sampling_stratum": c["sampling_stratum"],
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d labeled examples to %s", len(out), OUT_PATH)


if __name__ == "__main__":
    main()
