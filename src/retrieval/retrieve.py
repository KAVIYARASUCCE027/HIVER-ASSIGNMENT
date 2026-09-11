"""Semantic retrieval over the historical corpus.

The query incorporates the current customer message AND the Phase G
predicted intent (never the true golden intent, per Section 5 -- at
evaluation time the caller passes whatever Phase G's frozen classifier
output, exactly the information a deployed system would actually have).
The predicted intent is not concatenated into the embedded text (that would
just be a repeated token, adding noise to the embedding); instead it's kept
alongside the query for intent-match scoring in rerank.py.
"""

from __future__ import annotations

import numpy as np

from src.retrieval.evidence import Evidence, build_evidence
from src.retrieval.index import RetrievalIndex, embed_query

RETRIEVE_N = 30  # candidates pulled by semantic similarity, before reranking to TOP_K


def semantic_search(query_text: str, index: RetrievalIndex, n: int = RETRIEVE_N) -> list[tuple[dict, float]]:
    """Returns up to n (document, similarity) pairs, highest similarity first."""
    q = embed_query(query_text)
    sims = index.embeddings @ q  # (n_docs,) cosine similarity, since both sides are L2-normalized
    order = np.argsort(-sims)[:n]
    return [(index.documents[i], float(sims[i])) for i in order]


def retrieve_candidates(query_text: str, predicted_intent: str, index: RetrievalIndex,
                         n: int = RETRIEVE_N) -> list[Evidence]:
    """Raw semantic candidates converted to Evidence objects (not yet
    reranked or filtered by TOP_K -- see src/retrieval/rerank.py)."""
    hits = semantic_search(query_text, index, n=n)
    return [build_evidence(doc, sim, predicted_intent) for doc, sim in hits]
