"""Build the historical AppleSupport resolution corpus for RAG retrieval.

Source: ONLY `retrieval_pool` conversations from
`data/processed/applesupport_conversations.jsonl` (never `eval_pool`, never
golden). Only conversations where AppleSupport actually replied are included
(`has_agent_reply=True` per Section 3's "prefer conversations where AppleSupport
actually responded" -- Phase C already showed this is 97.9% of retrieval_pool).

A fixed random subset of retrieval_pool conversation_ids is reserved as a
DEV HOLDOUT (excluded from the indexed corpus) so Section 7's development
retrieval evaluation can query with examples that are not trivially
retrievable as themselves. This holdout is entirely within retrieval_pool --
it has nothing to do with the golden/eval_pool split, which remains untouched.

Each corpus document:
    evidence_id           "apple_conv_<conversation_id>" -- always maps to a
                           real conversation_id, never invented
    conversation_id
    customer_text          root customer message (sanitized)
    agent_response          all agent turns concatenated, in order (sanitized)
    timestamp               root message timestamp
    turn_count
    is_substantive          bool, see resolution_quality.py
    resolution_quality      float in [0, 1], see resolution_quality.py
    intent                  assigned below, via the frozen (retrieval_pool-only)
                            embedding classifier -- NOT the golden labels

Run:
    python -m src.retrieval.corpus
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

from src.evaluation.embedding_intent_classifier import build_intent_centroids, classify_texts
from src.retrieval.resolution_quality import is_substantive, resolution_quality
from src.retrieval.sanitize import sanitize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONVERSATIONS_PATH = Path("data/processed/applesupport_conversations.jsonl")
CORPUS_PATH = Path("data/processed/rag_corpus.jsonl")
DEV_HOLDOUT_PATH = Path("data/processed/rag_dev_holdout.jsonl")
GOLDEN_PATH = Path("data/golden/golden_200.jsonl")

DEV_HOLDOUT_SIZE = 50
DEV_HOLDOUT_SEED = 1234  # distinct from other seeds used elsewhere in this project


_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    t = text.lower()
    t = _NON_ALNUM_RE.sub(" ", t)
    return _WHITESPACE_RE.sub(" ", t).strip()


def load_golden_exact_texts() -> set[str]:
    """Normalized golden customer_message texts. A tiny number of retrieval_pool
    conversations turn out to share byte-for-byte-after-normalization text with
    a golden example (different real customers independently posting the same
    short viral phrase, e.g. the autocorrect-bug "FIX I.T @115858" pattern
    already documented in reports/leakage_check.md from Phase F). This is not
    conversation-level contamination (conversation_ids and tweet_ids never
    overlap -- see src/evaluation/check_rag_leakage.py), but Section 2 of
    Phase H explicitly requires zero exact-text overlap in the indexed corpus,
    so those specific documents are excluded here, conservatively, even though
    the underlying phenomenon is benign. See reports/decision_log.md."""
    if not GOLDEN_PATH.exists():
        return set()
    golden = [json.loads(l) for l in GOLDEN_PATH.open(encoding="utf-8")]
    return {_normalize(g["customer_message"]) for g in golden}


def load_retrieval_pool_with_agent() -> list[dict]:
    convs = []
    with CONVERSATIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["split"] != "retrieval_pool":
                continue
            if any(m["speaker"] == "agent" for m in c["messages"]):
                convs.append(c)
    logger.info("%d retrieval_pool conversations with >=1 agent reply", len(convs))
    return convs


def build_document(conv: dict) -> dict:
    messages = conv["messages"]
    customer_text = sanitize(messages[0]["text"])
    agent_texts = [sanitize(m["text"]) for m in messages if m["speaker"] == "agent"]
    agent_response = " ".join(agent_texts)
    agent_had_last_word = messages[-1]["speaker"] == "agent"
    return {
        "evidence_id": f"apple_conv_{conv['conversation_id']}",
        "conversation_id": conv["conversation_id"],
        "customer_text": customer_text,
        "agent_response": agent_response,
        "timestamp": conv["root_timestamp"],
        "turn_count": len(messages),
        "is_substantive": is_substantive(agent_response),
        "resolution_quality": resolution_quality(agent_response, agent_had_last_word),
    }


def main() -> None:
    convs = load_retrieval_pool_with_agent()

    rng = random.Random(DEV_HOLDOUT_SEED)
    shuffled = convs[:]
    rng.shuffle(shuffled)
    dev_holdout_convs = shuffled[:DEV_HOLDOUT_SIZE]
    dev_holdout_ids = {c["conversation_id"] for c in dev_holdout_convs}
    corpus_convs = [c for c in convs if c["conversation_id"] not in dev_holdout_ids]
    logger.info("Dev holdout: %d conversations (excluded from indexed corpus)", len(dev_holdout_ids))

    golden_texts = load_golden_exact_texts()
    if golden_texts:
        before = len(corpus_convs)
        corpus_convs = [
            c for c in corpus_convs if _normalize(c["messages"][0]["text"]) not in golden_texts
        ]
        excluded = before - len(corpus_convs)
        if excluded:
            logger.info("Excluded %d corpus conversations with exact-text overlap with a golden "
                        "customer_message (see load_golden_exact_texts docstring)", excluded)

    logger.info("Corpus (to be indexed): %d conversations", len(corpus_convs))

    logger.info("Building documents (sanitization + resolution-quality scoring) ...")
    documents = [build_document(c) for c in corpus_convs]

    logger.info("Assigning intents via the frozen retrieval_pool-only embedding classifier ...")
    intent_names, centroids = build_intent_centroids()
    predictions = classify_texts([d["customer_text"] for d in documents], intent_names, centroids)
    for doc, pred in zip(documents, predictions):
        doc["intent"] = pred["predicted_intent"]
        doc["intent_confidence"] = round(pred["confidence"], 4)

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS_PATH.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logger.info("Wrote %d documents to %s", len(documents), CORPUS_PATH)

    dev_docs = [build_document(c) for c in dev_holdout_convs]
    with DEV_HOLDOUT_PATH.open("w", encoding="utf-8") as f:
        for doc in dev_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logger.info("Wrote %d dev-holdout documents (NOT indexed, for retrieval dev eval) to %s",
                len(dev_docs), DEV_HOLDOUT_PATH)

    n_substantive = sum(1 for d in documents if d["is_substantive"])
    logger.info("Corpus stats: %d/%d (%.1f%%) substantive resolutions",
                n_substantive, len(documents), 100 * n_substantive / len(documents))


if __name__ == "__main__":
    main()
