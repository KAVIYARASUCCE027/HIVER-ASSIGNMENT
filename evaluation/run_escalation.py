"""Phase I: evaluate the deterministic escalation policy against the frozen
200-example golden set's human `expected_action` labels.

Run:
    python -m evaluation.run_escalation

Reads only already-frozen artifacts -- does not call any LLM, does not
re-run classification or retrieval, does not modify golden labels, Phase G
predictions, or Phase H outputs:
    - data/golden/golden_200.jsonl        (human expected_action -- reference only)
    - evaluation/results/llm_classifier.json  (Phase G: confidence, predicted_intent)
    - evaluation/results/replies_rag.jsonl    (Phase H: retrieved_evidence, should_answer)

Writes:
    - evaluation/results/escalation_results.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.classification.common import load_golden
from src.escalation.policy import EscalationInput, decide

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PHASE_G_RESULTS_PATH = Path("evaluation/results/llm_classifier.json")
PHASE_H_RAG_PATH = Path("evaluation/results/replies_rag.jsonl")
OUT_PATH = Path("evaluation/results/escalation_results.json")


def load_phase_g() -> dict[str, dict]:
    """Frozen Phase G message+context predictions: id -> {predicted_intent, confidence}."""
    data = json.loads(PHASE_G_RESULTS_PATH.read_text(encoding="utf-8"))
    return {
        r["id"]: {"predicted_intent": r["predicted_intent"], "confidence": r["confidence"]}
        for r in data["per_example"]
    }


def load_phase_h_rag() -> dict[str, dict]:
    """Frozen Phase H RAG outputs: id -> {has_retrieved_evidence, should_answer}."""
    out = {}
    with PHASE_H_RAG_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["example_id"]] = {
                "has_retrieved_evidence": bool(r["retrieved_evidence"]),
                "should_answer": r["should_answer"],
            }
    return out


def confusion_matrix(records: list[dict]) -> dict:
    tp = sum(1 for r in records if r["expected_action"] == "escalate" and r["policy_action"] == "ESCALATE")
    tn = sum(1 for r in records if r["expected_action"] == "auto_handle" and r["policy_action"] == "AUTO_HANDLE")
    fp = sum(1 for r in records if r["expected_action"] == "auto_handle" and r["policy_action"] == "ESCALATE")
    fn = sum(1 for r in records if r["expected_action"] == "escalate" and r["policy_action"] == "AUTO_HANDLE")
    return {"tp_escalate": tp, "tn_auto_handle": tn, "fp_escalate_when_auto_handle_expected": fp,
            "fn_auto_handle_when_escalate_expected": fn}


def compute_metrics(records: list[dict]) -> dict:
    n = len(records)
    cm = confusion_matrix(records)
    tp, tn, fp, fn = cm["tp_escalate"], cm["tn_auto_handle"], cm["fp_escalate_when_auto_handle_expected"], cm["fn_auto_handle_when_escalate_expected"]
    accuracy = (tp + tn) / n
    precision_escalate = tp / (tp + fp) if (tp + fp) else None
    recall_escalate = tp / (tp + fn) if (tp + fn) else None
    f1_escalate = (
        2 * precision_escalate * recall_escalate / (precision_escalate + recall_escalate)
        if precision_escalate and recall_escalate and (precision_escalate + recall_escalate) > 0
        else None
    )
    return {
        "n_total": n,
        "accuracy": accuracy,
        "precision_escalate": precision_escalate,
        "recall_escalate": recall_escalate,
        "f1_escalate": f1_escalate,
        "confusion_matrix": cm,
        "false_auto_handle_count": fn,  # human said escalate, policy said AUTO_HANDLE -- the safety-critical error
        "false_escalate_count": fp,     # human said auto_handle, policy said ESCALATE -- the cost/friction error
    }


def main() -> None:
    golden = load_golden()
    phase_g = load_phase_g()
    phase_h = load_phase_h_rag()

    missing_g = [g["id"] for g in golden if g["id"] not in phase_g]
    missing_h = [g["id"] for g in golden if g["id"] not in phase_h]
    if missing_g or missing_h:
        raise RuntimeError(f"Missing Phase G data for {missing_g[:5]}, Phase H data for {missing_h[:5]}")

    records = []
    for g in golden:
        pg, ph = phase_g[g["id"]], phase_h[g["id"]]
        inp = EscalationInput(
            example_id=g["id"],
            customer_message=g["customer_message"],
            predicted_intent=pg["predicted_intent"],
            confidence=pg["confidence"],
            has_retrieved_evidence=ph["has_retrieved_evidence"],
            should_answer=ph["should_answer"],
        )
        decision = decide(inp)
        records.append({
            "example_id": g["id"],
            "customer_message": g["customer_message"],
            "true_intent": g["intent"],
            "predicted_intent": pg["predicted_intent"],
            "confidence": pg["confidence"],
            "has_retrieved_evidence": ph["has_retrieved_evidence"],
            "should_answer": ph["should_answer"],
            "expected_action": g["expected_action"],
            "expected_reason": g["expected_reason"],
            "policy_action": decision.action,
            "policy_reasons": decision.reasons,
        })

    metrics = compute_metrics(records)
    out = {"metrics": metrics, "per_example": records}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", OUT_PATH)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
