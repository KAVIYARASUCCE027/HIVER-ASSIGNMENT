"""Phase H: run the No-RAG vs RAG ablation over the frozen 200-example golden set.

Run:
    python -m evaluation.run_rag

Steps:
    1. Leakage check (fails loudly if the corpus contains golden data).
    2. Load/build the retrieval index (cached after first run).
    3. For each golden example: retrieve+rerank evidence using the customer
       message and the Phase G FROZEN predicted intent (never the true golden
       label -- exactly what a deployed system would have).
    4. Generate a No-RAG reply (no evidence) and a RAG reply (reranked
       top-K, or none if below the evidence threshold) with the same model/
       temperature/base prompt.
    5. Save both variants' results and retrieval-ablation statistics.

Requires LLM_PROVIDER/LLM_MODEL/LLM_API_KEY in `.env` (same as Phase G).
Every (example, variant) pair is cached in data/processed/llm_replies/, so
re-running after an interruption does not re-call the API for completed
examples.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

from src.classification.common import load_golden
from src.classification.llm_providers import LLMConfig, LLMConfigError
from src.evaluation import check_rag_leakage
from src.generation.reply_generator import generate_reply
from src.retrieval.evidence import Evidence
from src.retrieval.index import build_index
from src.retrieval.rerank import EVIDENCE_THRESHOLD, rerank, rerank_score
from src.retrieval.retrieve import retrieve_candidates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NO_RAG_PATH = Path("evaluation/results/replies_no_rag.jsonl")
RAG_PATH = Path("evaluation/results/replies_rag.jsonl")
RETRIEVAL_STATS_PATH = Path("evaluation/results/retrieval_ablation.json")
PHASE_G_RESULTS_PATH = Path("evaluation/results/llm_classifier.json")
CONFIG_VERSION = "phase_h_v1"


def load_phase_g_predicted_intents() -> dict[str, str]:
    """Phase G's FROZEN message+context predictions -- never re-run, never
    the golden true label. Using anything else here would violate Section 5's
    "do not use the true golden intent during evaluation"."""
    data = json.loads(PHASE_G_RESULTS_PATH.read_text(encoding="utf-8"))
    return {r["id"]: r["predicted_intent"] for r in data["per_example"]}


def retrieve_for_example(customer_message: str, predicted_intent: str, index) -> tuple[list[Evidence], bool]:
    """Returns (evidence, evidence_sufficient). If the top reranked score is
    below EVIDENCE_THRESHOLD, evidence is returned empty and
    evidence_sufficient=False -- Section 11's retrieval-level gate, applied
    before generation is even called for the RAG variant."""
    candidates = retrieve_candidates(customer_message, predicted_intent, index)
    top_k = rerank(candidates)
    if not top_k or rerank_score(top_k[0]) < EVIDENCE_THRESHOLD:
        return [], False
    return top_k, True


def run_variant(golden: list[dict], predicted_intents: dict[str, str], index, config: LLMConfig,
                 variant: str, out_path: Path) -> tuple[list[dict], dict]:
    records = []
    retrieval_stats = {"n_sufficient": 0, "n_insufficient": 0, "similarities": [], "intent_match_top1": 0,
                        "intent_match_any_top5": 0, "substantive_top1": 0}

    for i, g in enumerate(golden, 1):
        predicted_intent = predicted_intents[g["id"]]

        if variant == "rag":
            evidence, sufficient = retrieve_for_example(g["customer_message"], predicted_intent, index)
            retrieval_stats["n_sufficient" if sufficient else "n_insufficient"] += 1
            if evidence:
                retrieval_stats["similarities"].append(evidence[0].similarity)
                retrieval_stats["intent_match_top1"] += int(evidence[0].intent_match)
                retrieval_stats["intent_match_any_top5"] += int(any(e.intent_match for e in evidence))
                retrieval_stats["substantive_top1"] += int(evidence[0].is_substantive)
        else:
            evidence, sufficient = [], False

        if variant == "rag" and not sufficient:
            # Section 11: retrieval-level insufficient-evidence short-circuit --
            # no generation call, deterministic safe state.
            result_dict = {
                "reply": "", "evidence_ids": [], "should_answer": False,
                "grounding_note": "No sufficiently relevant historical resolution found.",
                "attempts": 0, "error_kind": None,
            }
        else:
            gen = generate_reply(
                g["id"], g["customer_message"], g["conversation"], predicted_intent,
                evidence, config=config, variant=variant,
            )
            result_dict = gen.to_dict()

        records.append({
            "example_id": g["id"],
            "customer_message": g["customer_message"],
            "context": g["conversation"],
            "predicted_intent": predicted_intent,
            "reply": result_dict["reply"],
            "evidence_ids": result_dict["evidence_ids"],
            "should_answer": result_dict["should_answer"],
            "grounding_note": result_dict["grounding_note"],
            "error_kind": result_dict["error_kind"],
            "model": config.model,
            "configuration_version": CONFIG_VERSION,
            "retrieved_evidence": [e.to_dict() for e in evidence] if variant == "rag" else None,
        })
        if i % 25 == 0 or i == len(golden):
            logger.info("%s: %d/%d", variant, i, len(golden))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), out_path)
    return records, retrieval_stats


def summarize_retrieval_stats(raw: dict, n_total: int) -> dict:
    sims = raw["similarities"]
    return {
        "n_total": n_total,
        "pct_usable_evidence": raw["n_sufficient"] / n_total,
        "pct_no_sufficient_evidence": raw["n_insufficient"] / n_total,
        "avg_similarity_top1": (sum(sims) / len(sims)) if sims else None,
        "median_similarity_top1": statistics.median(sims) if sims else None,
        "intent_match_rate_top1": raw["intent_match_top1"] / max(raw["n_sufficient"], 1),
        "intent_match_rate_any_top5": raw["intent_match_any_top5"] / max(raw["n_sufficient"], 1),
        "substantive_rate_top1": raw["substantive_top1"] / max(raw["n_sufficient"], 1),
        "evidence_threshold": EVIDENCE_THRESHOLD,
        "note": "These are RETRIEVAL metrics (did we find plausible evidence), not reply-quality metrics.",
    }


def main() -> None:
    try:
        config = LLMConfig.from_env()
    except LLMConfigError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc

    logger.info("Step 1/5: RAG corpus leakage check")
    check_rag_leakage.main()

    logger.info("Step 2/5: load/build retrieval index")
    index = build_index()

    golden = load_golden()
    predicted_intents = load_phase_g_predicted_intents()
    missing = [g["id"] for g in golden if g["id"] not in predicted_intents]
    if missing:
        raise RuntimeError(f"{len(missing)} golden examples have no Phase G predicted intent: {missing[:5]}")

    logger.info("Step 3/5: generate No-RAG replies (System A)")
    run_variant(golden, predicted_intents, index, config, "no_rag", NO_RAG_PATH)

    logger.info("Step 4/5: retrieve + generate RAG replies (System B)")
    _, retrieval_stats_raw = run_variant(golden, predicted_intents, index, config, "rag", RAG_PATH)

    logger.info("Step 5/5: save retrieval ablation statistics")
    stats = summarize_retrieval_stats(retrieval_stats_raw, len(golden))
    RETRIEVAL_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Wrote %s: %s", RETRIEVAL_STATS_PATH, stats)


if __name__ == "__main__":
    main()
