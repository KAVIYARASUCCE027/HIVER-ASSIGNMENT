"""Programmatic leakage check for the RAG retrieval corpus, run before any
retrieval index is used. Fails loudly (raises, non-zero exit) if leakage is
detected -- this is a hard gate, not just a report.

Checks:
  1. golden_conversation_ids ∩ indexed_conversation_ids = empty
  2. golden_tweet_ids ∩ indexed_tweet_ids = empty (tweet_ids from the golden
     set's preserved preceding-context messages; the target message's own
     tweet_id isn't retained in golden_200.jsonl by design, same caveat as
     Phase F's src/evaluation/leakage_check.py)
  3. no exact golden customer_message text appears verbatim in the indexed
     corpus's customer_text field

Run:
    python -m src.evaluation.check_rag_leakage
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONVERSATIONS_PATH = Path("data/processed/applesupport_conversations.jsonl")
CORPUS_PATH = Path("data/processed/rag_corpus.jsonl")
GOLDEN_PATH = Path("data/golden/golden_200.jsonl")
OUT_PATH = Path("reports/rag_leakage_check.md")

WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def normalize(text: str) -> str:
    t = text.lower()
    t = NON_ALNUM_RE.sub(" ", t)
    return WHITESPACE_RE.sub(" ", t).strip()


class RAGLeakageError(RuntimeError):
    """Raised when the RAG corpus contains golden-set leakage. Fatal by design."""


def main() -> dict:
    golden = [json.loads(l) for l in GOLDEN_PATH.open(encoding="utf-8")]
    corpus = [json.loads(l) for l in CORPUS_PATH.open(encoding="utf-8")]

    golden_conv_ids = {g["conversation_id"] for g in golden}
    corpus_conv_ids = {d["conversation_id"] for d in corpus}
    conv_intersection = golden_conv_ids & corpus_conv_ids

    golden_context_tids = {m["tweet_id"] for g in golden for m in g["conversation"]}
    # Corpus documents don't retain per-message tweet_ids (only aggregated
    # text), so we check against the FULL conversation records that were
    # eligible for indexing (retrieval_pool, has agent reply) instead --
    # this is a stricter, superset check.
    all_conv = {}
    with CONVERSATIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["split"] == "retrieval_pool":
                all_conv[c["conversation_id"]] = c
    indexed_tids = {
        m["tweet_id"] for cid in corpus_conv_ids for m in all_conv.get(cid, {}).get("messages", [])
    }
    tid_intersection = golden_context_tids & indexed_tids

    corpus_texts = {normalize(d["customer_text"]) for d in corpus}
    golden_exact_matches = [
        g["id"] for g in golden if normalize(g["customer_message"]) in corpus_texts
    ]

    lines = [
        "# RAG Corpus Leakage Check",
        "",
        f"Golden conversations: {len(golden_conv_ids)} | Indexed corpus conversations: {len(corpus_conv_ids)}",
        "",
        "## 1. Conversation-ID intersection",
        "",
        f"`golden_conversation_ids ∩ indexed_conversation_ids` = **{len(conv_intersection)}** (expected: 0)",
        "",
        "## 2. Tweet-ID intersection",
        "",
        f"`golden_tweet_ids ∩ indexed_tweet_ids` = **{len(tid_intersection)}** (expected: 0; golden tweet_ids "
        "here are the preceding-context messages golden_200.jsonl retains -- the target message's own "
        "tweet_id isn't kept in that file, same caveat as Phase F's leakage check)",
        "",
        "## 3. Exact customer-message text overlap",
        "",
        f"Golden customer messages whose normalized text also appears verbatim as a corpus document's "
        f"`customer_text`: **{len(golden_exact_matches)}** / {len(golden)}",
        "",
    ]
    if golden_exact_matches:
        lines.append(f"IDs: {golden_exact_matches}")
        lines.append("")

    passed = len(conv_intersection) == 0 and len(tid_intersection) == 0 and len(golden_exact_matches) == 0
    lines.append(f"## Result: {'PASSED' if passed else 'FAILED'}")
    lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", OUT_PATH)

    if not passed:
        raise RAGLeakageError(
            f"RAG corpus leakage detected: conv_overlap={len(conv_intersection)}, "
            f"tid_overlap={len(tid_intersection)}, exact_text_overlap={len(golden_exact_matches)}. "
            f"See {OUT_PATH}."
        )
    logger.info("RAG LEAKAGE CHECK PASSED.")
    return {
        "conversation_overlap": len(conv_intersection),
        "tweet_id_overlap": len(tid_intersection),
        "exact_text_overlap": len(golden_exact_matches),
        "passed": passed,
    }


if __name__ == "__main__":
    main()
