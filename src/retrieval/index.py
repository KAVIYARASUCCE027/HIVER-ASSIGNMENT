"""Build/load the RAG retrieval index.

Simplest reliable option for this corpus size (~67k documents, 384-dim
embeddings ~= 100MB): a flat numpy matrix of L2-normalized embeddings,
searched via a single matrix multiply for cosine similarity. No FAISS/Chroma
-- at this scale a vector database adds infrastructure without adding
capability (CLAUDE.md: "do not introduce unnecessary infrastructure").

Embeddings are cached to disk (data/processed/rag_corpus_embeddings.npy) so
the ~8-10 minute encoding pass runs once, not on every retrieval call.

Embedding model: `all-MiniLM-L6-v2` -- the same sentence-transformers model
already used for intent clustering (Phase D) and the frozen intent
classifier (Phase E/F), reused here per Section 5's "use the existing
sentence-transformer infrastructure where practical."
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CORPUS_PATH = Path("data/processed/rag_corpus.jsonl")
EMBEDDINGS_PATH = Path("data/processed/rag_corpus_embeddings.npy")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _MODEL


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.clip(norms, 1e-8, None)


@dataclass
class RetrievalIndex:
    documents: list[dict]
    embeddings: np.ndarray  # (n_docs, dim), L2-normalized


def build_index(force: bool = False) -> RetrievalIndex:
    documents = [json.loads(l) for l in CORPUS_PATH.open(encoding="utf-8")]

    if not force and EMBEDDINGS_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)
        if embeddings.shape[0] == len(documents):
            logger.info("Loaded cached corpus embeddings: %s", embeddings.shape)
            return RetrievalIndex(documents=documents, embeddings=embeddings)
        logger.warning("Cached embeddings shape %s doesn't match corpus size %d -- rebuilding.",
                        embeddings.shape, len(documents))

    logger.info("Encoding %d corpus documents with %s ...", len(documents), EMBEDDING_MODEL_NAME)
    model = _get_model()
    texts = [d["customer_text"] for d in documents]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = _l2_normalize(np.asarray(embeddings, dtype=np.float32))

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    logger.info("Wrote %s", EMBEDDINGS_PATH)
    return RetrievalIndex(documents=documents, embeddings=embeddings)


def embed_query(text: str) -> np.ndarray:
    model = _get_model()
    emb = model.encode([text], show_progress_bar=False)
    return _l2_normalize(np.asarray(emb, dtype=np.float32))[0]
