"""Rerank raw semantic candidates by combining similarity, intent match, and
historical resolution quality, then cut to TOP_K.

Weights are fixed a priori (documented here and in config/rag.yaml), not
tuned against golden. If the Section 7 development evaluation shows a clear
need to adjust them, the change and its reasoning are recorded in
reports/decision_log.md before the golden evaluation is run -- see that file
for whether this happened.
"""

from __future__ import annotations

from src.retrieval.evidence import Evidence

TOP_K = 5

# Fixed a priori weights (see module docstring). Sum to 1.0 for interpretability.
WEIGHT_SIMILARITY = 0.60
WEIGHT_INTENT_MATCH = 0.25
WEIGHT_RESOLUTION_QUALITY = 0.15

# Minimum top-1 rerank_score for evidence to be considered "sufficiently
# relevant" (Section 11). Set from the Section 7 development evaluation (50
# retrieval_pool dev-holdout queries, never golden): top-1 rerank_score
# across all 50 dev queries ranged 0.608-0.951 (median 0.863). Manual
# inspection found exactly one genuinely bad match in that set -- a query
# whose own text was off-topic banter, not a real support request -- at
# score 0.678; the next-lowest four scores (0.708-0.727) were niche/non-English
# queries whose retrieved evidence was still topically relevant even if not a
# full resolution. 0.70 sits between those two groups: it rejects the one
# clear failure while keeping the harder-but-relevant cases. See
# reports/retrieval_quality.md and reports/decision_log.md.
EVIDENCE_THRESHOLD = 0.70


def rerank_score(evidence: Evidence) -> float:
    return (
        WEIGHT_SIMILARITY * evidence.similarity
        + WEIGHT_INTENT_MATCH * (1.0 if evidence.intent_match else 0.0)
        + WEIGHT_RESOLUTION_QUALITY * evidence.resolution_quality
    )


def rerank(candidates: list[Evidence], top_k: int = TOP_K) -> list[Evidence]:
    scored = sorted(candidates, key=rerank_score, reverse=True)
    return scored[:top_k]
