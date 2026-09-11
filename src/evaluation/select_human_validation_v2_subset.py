"""Phase J.2: select the 18-example human-validation subset for the V2
refusal-scoring fix.

Selection is reproducible (fixed seed, declared here before any human
rating happened) and stratified across 5 categories:
  A. Justified RAG refusals (per V2's own Phase J.1 classification)  -- 4
  B. Potentially-unjustified RAG refusals (per V2's Phase J.1 classification) -- 3 (all available)
  C/D. Normal substantive RAG replies (paired with their No-RAG reply,
       reused from Phase J.1's control group)                        -- 4
  E. Difficult/ambiguous cases (golden `difficult:*` sampling stratum,
     drawn from the pool NEVER scored by any judge before)           -- 4
  Plus a general fresh sample (never scored by any judge before)     -- 3

IMPORTANT CAVEAT, stated plainly rather than hidden: categories A/B/C-D
are necessarily drawn from the SAME pool already scored by V2 in Phase
J.1 (42 total RAG refusals exist in the whole 200-example set; there is
no "unseen" refusal population to draw from). Selection within that pool
uses a fixed, pre-declared random seed -- NOT chosen by looking at which
examples make V1/V2 agree or disagree -- but the fact remains that this
script's author (Claude) had already seen V2's scores for that pool before
writing this selection code. The blinding safeguard that matters is
Section 2 of the brief: the HUMAN RATER never sees any prior score while
rating, regardless of what the selector has seen. Categories E and the
general sample ARE drawn from a pool never scored by any judge before,
achieving full blindness at the selection level too.

Run:
    python -m src.evaluation.select_human_validation_v2_subset

Writes:
    data/evaluation/human_validation_v2_subset.jsonl (18 examples, blinded A/B)
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
PHASE_J_SUBSET_PATH = Path("data/evaluation/human_rating_subset.jsonl")
V2_AUDIT_PATH = Path("evaluation/results/judge_rubric_v2_audit.jsonl")
OUT_PATH = Path("data/evaluation/human_validation_v2_subset.jsonl")

SELECTION_SEED = 20260912  # fixed before this script was ever run; documented, not re-rolled
BLINDING_SEED = 20260912

TARGET = {"justified_refusal": 4, "unjustified_refusal": 3, "control_normal": 4, "difficult_fresh": 4, "general_fresh": 3}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def main() -> None:
    golden = {g["id"]: g for g in _load_jsonl(GOLDEN_PATH)}
    no_rag = {r["example_id"]: r for r in _load_jsonl(NO_RAG_PATH)}
    rag = {r["example_id"]: r for r in _load_jsonl(RAG_PATH)}
    phase_j_subset_ids = {r["example_id"] for r in _load_jsonl(PHASE_J_SUBSET_PATH)}
    v2_audit = _load_jsonl(V2_AUDIT_PATH)

    justified_pool = [r["example_id"] for r in v2_audit if r.get("v2_major_issue") == "justified_refusal"]
    unjustified_pool = [r["example_id"] for r in v2_audit if r.get("v2_major_issue") == "unjustified_refusal"]
    control_pool = [r["example_id"] for r in v2_audit if r["group"] == "control"]

    already_touched = phase_j_subset_ids | {r["example_id"] for r in v2_audit}
    fresh_pool = [gid for gid in golden if gid not in already_touched]
    difficult_fresh_pool = [gid for gid in fresh_pool if golden[gid]["sampling_stratum"].startswith("difficult:")]
    general_fresh_pool = [gid for gid in fresh_pool if gid not in difficult_fresh_pool]

    rng = random.Random(SELECTION_SEED)
    selected: dict[str, str] = {}  # example_id -> category

    def take(pool: list[str], category: str, n: int) -> None:
        avail = [g for g in pool if g not in selected]
        rng.shuffle(avail)
        for gid in avail[:n]:
            selected[gid] = category

    take(justified_pool, "justified_refusal", TARGET["justified_refusal"])
    take(unjustified_pool, "unjustified_refusal", TARGET["unjustified_refusal"])
    take(control_pool, "control_normal", TARGET["control_normal"])
    take(difficult_fresh_pool, "difficult_fresh", TARGET["difficult_fresh"])
    take(general_fresh_pool, "general_fresh", TARGET["general_fresh"])

    assert len(selected) == sum(TARGET.values()), f"expected {sum(TARGET.values())}, got {len(selected)}"
    logger.info("Selected %d examples: %s", len(selected), selected)

    blind_rng = random.Random(BLINDING_SEED)
    records = []
    for gid, category in selected.items():
        g, nr, rr = golden[gid], no_rag[gid], rag[gid]
        order = ["no_rag", "rag"] if blind_rng.random() < 0.5 else ["rag", "no_rag"]
        records.append({
            "example_id": gid,
            "selection_category": category,
            "customer_message": g["customer_message"],
            "conversation": g["conversation"],
            "predicted_intent": nr["predicted_intent"],
            "sampling_stratum": g["sampling_stratum"],
            "label_A_version": order[0],
            "label_B_version": order[1],
            "reply_A": nr["reply"] if order[0] == "no_rag" else rr["reply"],
            "reply_B": nr["reply"] if order[1] == "no_rag" else rr["reply"],
            "evidence_A": None if order[0] == "no_rag" else rr["retrieved_evidence"],
            "evidence_B": None if order[1] == "no_rag" else rr["retrieved_evidence"],
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), OUT_PATH)


if __name__ == "__main__":
    main()
