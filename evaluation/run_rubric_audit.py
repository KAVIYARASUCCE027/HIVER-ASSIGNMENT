"""Phase J.1: evaluate the V2 candidate rubric against V1, on the refusal
subset first (NOT the full 400 -- that only happens if V2 is approved).

Run:
    python -m evaluation.run_rubric_audit

Two groups, both scored with V2 and compared against ALREADY-EXISTING V1
scores (evaluation/results/judge_rag.jsonl, frozen, never re-scored):
  1. The 42 RAG "refusal" examples (empty reply: 40 insufficient-evidence
     retrieval-gate cases + 2 generation-declined-despite-evidence cases).
  2. A FIXED, predetermined control group of 15 non-empty RAG replies
     (seed chosen before running V2), to verify V2 does NOT change scores
     for normal replies -- if it did, that would mean V2 is not the
     targeted correction it claims to be.

Writes:
    evaluation/results/judge_rubric_v2_audit.jsonl

Does not modify config/judge_rubric.yaml, src/evaluation/judge_prompt.py,
or any evaluation/results/judge_*.jsonl / rag_vs_no_rag.json file from
Phase J v1.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from src.classification.llm_providers import LLMConfig
from src.evaluation.judge_runner_v2 import judge_reply_v2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_PATH = Path("data/golden/golden_200.jsonl")
RAG_PATH = Path("evaluation/results/replies_rag.jsonl")
JUDGE_RAG_V1_PATH = Path("evaluation/results/judge_rag.jsonl")  # READ ONLY, never modified
OUT_PATH = Path("evaluation/results/judge_rubric_v2_audit.jsonl")

CONTROL_GROUP_SEED = 20260911  # fixed BEFORE running V2, documented, not re-rolled
CONTROL_GROUP_SIZE = 15


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def main() -> None:
    golden = {g["id"]: g for g in _load_jsonl(GOLDEN_PATH)}
    rag = {r["example_id"]: r for r in _load_jsonl(RAG_PATH)}
    v1_scores = {r["example_id"]: r for r in _load_jsonl(JUDGE_RAG_V1_PATH)}

    refusal_ids = [gid for gid, r in rag.items() if not r["reply"].strip()]
    non_refusal_ids = [gid for gid, r in rag.items() if r["reply"].strip()]
    assert len(refusal_ids) == 42, f"expected 42 refusals, found {len(refusal_ids)}"

    rng = random.Random(CONTROL_GROUP_SEED)
    control_ids = rng.sample(non_refusal_ids, CONTROL_GROUP_SIZE)

    logger.info("Refusal subset: %d examples", len(refusal_ids))
    logger.info("Control group (non-empty, fixed seed=%d): %d examples", CONTROL_GROUP_SEED, len(control_ids))

    config = LLMConfig.from_env()
    records = []
    for group, ids in (("refusal", refusal_ids), ("control", control_ids)):
        for gid in ids:
            g, r = golden[gid], rag[gid]
            v2 = judge_reply_v2(
                gid, g["customer_message"], g["conversation"], r["predicted_intent"],
                r["reply"], r["retrieved_evidence"], "rag", config=config,
            )
            v1 = v1_scores[gid]
            records.append({
                "example_id": gid, "group": group,
                "v1_overall_score": v1["overall_score"], "v1_major_issue": v1["major_issue"],
                "v1_scores": {k: v1[k] for k in (
                    "relevance_score", "helpfulness_score", "grounding_score", "factual_safety_score",
                    "resolution_score", "uncertainty_score", "professionalism_score")},
                "v2_overall_score": v2.overall_score, "v2_major_issue": v2.major_issue,
                "v2_scores": {
                    "relevance_score": v2.relevance_score, "helpfulness_score": v2.helpfulness_score,
                    "grounding_score": v2.grounding_score, "factual_safety_score": v2.factual_safety_score,
                    "resolution_score": v2.resolution_score, "uncertainty_score": v2.uncertainty_score,
                    "professionalism_score": v2.professionalism_score,
                },
                "v2_brief_reason": v2.brief_reason, "v2_error_kind": v2.error_kind,
            })
            logger.info("%s %s: v1=%s v2=%s (%s)", group, gid, v1["overall_score"], v2.overall_score, v2.major_issue)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %s (%d records)", OUT_PATH, len(records))


if __name__ == "__main__":
    main()
