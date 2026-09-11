"""Frozen, retrieval_pool-only embedding classifier for golden-set sampling.

Phase D found the substring keyword tagger has real recall gaps (see
"Discovered during validation" in reports/intent_discovery.md). For Phase E's
harder sampling requirements -- a *natural*-distribution stratum, meaningful
rare-intent coverage, and a "clearly escalate / clearly auto-handle"
difficulty split -- we need a better per-message intent signal than
substring matching, without doing anything that could be construed as
tuning the taxonomy or the eventual classifier on eval_pool.

This module builds one centroid embedding per intent from the 12,000
`retrieval_pool` messages already clustered in Phase D
(`data/processed/intent_sample_embeddings.npy` /
`intent_sample_meta.jsonl`, produced by `src.analysis.intent_discovery`),
using the same manually-curated `CLUSTER_TO_INTENT` mapping from
`src.analysis.intent_coverage`. Nothing here is re-fit on eval_pool -- the
centroids are entirely a function of already-frozen retrieval_pool work.
Classifying eval_pool messages against these frozen centroids is inference,
not tuning, the same way running any fixed model over new data is.

This is still NOT the project's real intent classifier (that's Phase F/G's
job, trained and evaluated properly). It exists solely to make golden-set
*sampling* more representative than a keyword tagger would allow.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.analysis.intent_coverage import CLUSTER_TO_INTENT

logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = Path("data/processed/intent_sample_embeddings.npy")
META_PATH = Path("data/processed/intent_sample_meta.jsonl")

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.clip(norms, 1e-8, None)


def build_intent_centroids() -> tuple[list[str], np.ndarray]:
    """Returns (intent_names, centroids) where centroids[i] is the L2-normalized
    mean embedding for intent_names[i], built only from retrieval_pool data.
    """
    embeddings = np.load(EMBEDDINGS_PATH)
    meta = [json.loads(l) for l in META_PATH.open(encoding="utf-8")]
    assert len(meta) == embeddings.shape[0]

    weighted: dict[str, list[np.ndarray]] = defaultdict(list)
    for row, m in zip(embeddings, meta):
        mapping = CLUSTER_TO_INTENT[m["cluster_id"]]
        # A fractionally-split cluster's texts go to whichever intent got the
        # largest share -- a documented simplification (see
        # src/analysis/intent_coverage.py CLUSTER_TO_INTENT comments).
        best_intent = max(mapping.items(), key=lambda kv: kv[1])[0]
        if best_intent == "__non_english_out_of_scope__":
            continue
        weighted[best_intent].append(row)

    names = sorted(weighted.keys())
    centroids = np.stack([_l2_normalize(np.mean(weighted[n], axis=0, keepdims=True))[0] for n in names])
    logger.info("Built %d intent centroids from %d retrieval_pool sample messages", len(names), len(meta))
    return names, centroids


def classify_texts(texts: list[str], intent_names: list[str], centroids: np.ndarray, batch_size: int = 64) -> list[dict]:
    """For each text, return {"predicted_intent", "margin", "top2_intent"}.
    `margin` = cosine(top1) - cosine(top2): small margin means the message
    sits between two intents (a genuine ambiguity signal), not just "matched
    zero keywords" like the Phase D tagger's residual bucket.
    """
    model = _get_model()
    emb = model.encode(texts, show_progress_bar=True, batch_size=batch_size)
    emb = _l2_normalize(np.asarray(emb))
    sims = emb @ centroids.T  # (n_texts, n_intents)
    order = np.argsort(-sims, axis=1)
    results = []
    for i in range(len(texts)):
        top1_idx, top2_idx = order[i, 0], order[i, 1]
        results.append({
            "predicted_intent": intent_names[top1_idx],
            "confidence": float(sims[i, top1_idx]),
            "margin": float(sims[i, top1_idx] - sims[i, top2_idx]),
            "second_intent": intent_names[top2_idx],
        })
    return results
