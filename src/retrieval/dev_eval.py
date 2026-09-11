"""Development retrieval evaluation (Section 7) -- NOT the golden set.

Runs retrieval + reranking for every dev-holdout query (conversations reserved
by src/retrieval/corpus.py, excluded from the indexed corpus so a query can't
trivially retrieve itself) and dumps the results to a plain-text file for
manual inspection. This script does not itself judge quality -- that
judgment (reports/retrieval_quality.md) is written by hand after reading this
dump, the same way golden-set labeling was done in Phase E.

Run:
    python -m src.retrieval.dev_eval
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.evaluation.embedding_intent_classifier import build_intent_centroids, classify_texts
from src.retrieval.index import build_index
from src.retrieval.rerank import rerank, rerank_score
from src.retrieval.retrieve import retrieve_candidates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEV_HOLDOUT_PATH = Path("data/processed/rag_dev_holdout.jsonl")
OUT_PATH = Path("reports/retrieval_dev_dump.txt")


def main() -> None:
    dev_docs = [json.loads(l) for l in DEV_HOLDOUT_PATH.open(encoding="utf-8")]
    logger.info("Loaded %d dev-holdout queries", len(dev_docs))

    logger.info("Assigning predicted intents to dev queries (frozen embedding classifier) ...")
    intent_names, centroids = build_intent_centroids()
    predictions = classify_texts([d["customer_text"] for d in dev_docs], intent_names, centroids)

    logger.info("Loading retrieval index ...")
    index = build_index()

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for doc, pred in zip(dev_docs, predictions):
            predicted_intent = pred["predicted_intent"]
            candidates = retrieve_candidates(doc["customer_text"], predicted_intent, index)
            top5 = rerank(candidates)

            f.write(f"\n{'=' * 80}\n")
            f.write(f"DEV QUERY: {doc['conversation_id']}\n")
            f.write(f"customer_text: {doc['customer_text']}\n")
            f.write(f"agent_response (ground truth, for reference only): {doc['agent_response'][:300]}\n")
            f.write(f"predicted_intent: {predicted_intent} (confidence={pred['confidence']:.2f})\n\n")
            for i, e in enumerate(top5, 1):
                f.write(f"  [{i}] {e.evidence_id} | sim={e.similarity:.3f} | intent_match={e.intent_match} | "
                        f"resolution_quality={e.resolution_quality:.2f} | substantive={e.is_substantive} | "
                        f"rerank_score={rerank_score(e):.3f}\n")
                f.write(f"      customer: {e.customer_text[:200]}\n")
                f.write(f"      agent:    {e.agent_response[:300]}\n")
    logger.info("Wrote %s", OUT_PATH)


if __name__ == "__main__":
    main()
