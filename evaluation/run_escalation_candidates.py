"""Phase I.1: controlled escalation-policy improvement experiment.

Evaluates several candidate policies against the SAME frozen artifacts Phase
I used (data/golden/golden_200.jsonl human labels, Phase G predictions/
confidence, Phase H RAG retrieved_evidence/should_answer), plus one
additional read-only signal not available in the Phase I frozen output:
the raw rerank_score for ALL 200 examples (including the 40 that fell below
Phase H's 0.70 evidence gate), recomputed via the frozen retrieval index and
frozen rerank weights -- no LLM call, no change to config/rag.yaml, no
change to Phase H's own replies_rag.jsonl.

Does NOT modify config/escalation.yaml, src/escalation/policy.py, or
evaluation/results/escalation_results.json (the Phase I baseline). Writes
only NEW files:
    - evaluation/results/escalation_candidates.json

Run:
    python -m evaluation.run_escalation_candidates

See reports/escalation_policy_improvement.md for the full methodology and
rationale behind every candidate, ablation, and rejected signal.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.classification.common import load_golden
from src.retrieval.index import build_index
from src.retrieval.retrieve import retrieve_candidates
from src.retrieval.rerank import rerank, rerank_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PHASE_G_RESULTS_PATH = Path("evaluation/results/llm_classifier.json")
PHASE_H_RAG_PATH = Path("evaluation/results/replies_rag.jsonl")
OUT_PATH = Path("evaluation/results/escalation_candidates.json")

HIGH_RISK_INTENTS = {"account_apple_id_security", "order_purchase_retail_support"}

SAFETY_KW_RE = re.compile(
    r"\b(swell|swollen|smoke|smoking|burn|burnt|burning|fire|explod|shock|"
    r"electrocut|spark|overheat|too hot)\b", re.IGNORECASE)
HARDWARE_KW_RE = re.compile(
    r"\b(broken|busted|defective|cracked|shattered|delaminat|won.t (turn on|charge|power)|"
    r"stopped working|physically damaged)\b", re.IGNORECASE)
BILLING_KW_RE = re.compile(
    r"\b(charged (me |twice|again)?|duplicate charge|double charged|billing (dispute|issue|problem)|"
    r"refund|declin(ed|ing)|still charg)\b", re.IGNORECASE)
DATALOSS_KW_RE = re.compile(
    r"\b(deleted|erased|disappeared|missing|lost|wiped out)\W+(\w+\W+){0,4}?"
    r"(photo|contact|file|data|message|note)s?\b", re.IGNORECASE)
# Tested and REJECTED (Step 7) -- kept here only so the rejection is
# reproducible, not used by any shipped candidate.
REPEAT_ATTEMPT_KW_RE = re.compile(
    r"\b(already (tried|did|attempted|restarted|updated|reset|rebuilt|wiped)|"
    r"still (not|doesn.t|isn.t|hasn.t|happening|occurring)|persists?|"
    r"again and again|multiple times|several times|repeatedly|"
    r"(second|third|another) time|keeps? happening)\b", re.IGNORECASE)


def load_phase_g() -> dict[str, dict]:
    data = json.loads(PHASE_G_RESULTS_PATH.read_text(encoding="utf-8"))
    return {r["id"]: {"predicted_intent": r["predicted_intent"], "confidence": r["confidence"]}
            for r in data["per_example"]}


def load_phase_h() -> dict[str, dict]:
    out = {}
    with PHASE_H_RAG_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["example_id"]] = {"has_retrieved_evidence": bool(r["retrieved_evidence"]),
                                     "should_answer": r["should_answer"]}
    return out


def recompute_rerank_scores(golden: list[dict], phase_g: dict[str, dict]) -> dict[str, float]:
    """Read-only recomputation of the top-1 rerank score for ALL 200 golden
    examples (including the 40 below Phase H's 0.70 gate, which
    replies_rag.jsonl does not store). Uses the frozen retrieval index and
    frozen rerank weights -- no LLM call, no write to any Phase H artifact."""
    index = build_index()
    scores = {}
    for g in golden:
        predicted_intent = phase_g[g["id"]]["predicted_intent"]
        candidates = retrieve_candidates(g["customer_message"], predicted_intent, index)
        top_k = rerank(candidates)
        scores[g["id"]] = rerank_score(top_k[0]) if top_k else 0.0
    return scores


def decide(
    g: dict, pg: dict, ph: dict, rr_score: float, *,
    confidence_threshold: float = 0.80,
    evidence_threshold: float = 0.70,
    use_high_risk_intent: bool = True,
    use_safety_kw: bool = True,
    use_hardware_kw: bool = False,
    use_billing_kw: bool = False,
    use_dataloss_kw: bool = False,
    use_repeat_attempt_kw: bool = False,  # rejected candidate, off by default
    use_evidence_signal: bool = True,
    use_gen_signal: bool = True,
    use_confidence_signal: bool = True,
    exempt_vague_from_evidence: bool = False,
) -> tuple[str, list[str]]:
    msg = g["customer_message"]
    reasons: list[str] = []

    if use_high_risk_intent and pg["predicted_intent"] in HIGH_RISK_INTENTS:
        reasons.append("high_risk_request")
    if use_safety_kw and SAFETY_KW_RE.search(msg):
        reasons.append("high_risk_request")
    if use_hardware_kw and HARDWARE_KW_RE.search(msg):
        reasons.append("high_risk_request")
    if use_billing_kw and BILLING_KW_RE.search(msg):
        reasons.append("high_risk_request")
    if use_dataloss_kw and DATALOSS_KW_RE.search(msg):
        reasons.append("high_risk_request")
    if use_repeat_attempt_kw and REPEAT_ATTEMPT_KW_RE.search(msg):
        reasons.append("high_risk_request")

    insufficient = rr_score < evidence_threshold
    if exempt_vague_from_evidence and pg["predicted_intent"] == "vague_frustration_needs_clarification":
        insufficient = False
    if use_evidence_signal and insufficient:
        reasons.append("insufficient_historical_evidence")
    elif use_gen_signal and not insufficient and rr_score >= 0.70 and not ph["should_answer"]:
        # should_answer is only known for examples where Phase H's OWN
        # 0.70 gate actually let generation run.
        reasons.append("generation_could_not_be_grounded")

    if use_confidence_signal and pg["confidence"] < confidence_threshold:
        reasons.append("intent_uncertainty")

    seen: set[str] = set()
    deduped = [r for r in reasons if not (r in seen or seen.add(r))]
    return ("ESCALATE" if deduped else "AUTO_HANDLE"), deduped


def evaluate_policy(golden, phase_g, phase_h, rerank_scores, **policy_kwargs) -> dict:
    tp = tn = fp = fn = 0
    per_example = []
    for g in golden:
        gid = g["id"]
        pg, ph, rr = phase_g[gid], phase_h[gid], rerank_scores[gid]
        action, reasons = decide(g, pg, ph, rr, **policy_kwargs)
        exp = g["expected_action"]
        if exp == "escalate" and action == "ESCALATE":
            tp += 1
        elif exp == "auto_handle" and action == "AUTO_HANDLE":
            tn += 1
        elif exp == "auto_handle" and action == "ESCALATE":
            fp += 1
        elif exp == "escalate" and action == "AUTO_HANDLE":
            fn += 1
        per_example.append({"example_id": gid, "expected_action": exp, "policy_action": action, "policy_reasons": reasons})
    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    metrics = {
        "n_total": n, "accuracy": (tp + tn) / n, "precision_escalate": precision,
        "recall_escalate": recall, "f1_escalate": f1,
        "confusion_matrix": {"tp_escalate": tp, "tn_auto_handle": tn, "fp_escalate": fp, "fn_auto_handle": fn},
        "false_auto_handle_count": fn, "false_escalate_count": fp,
    }
    return {"metrics": metrics, "per_example": per_example}


CANDIDATES = {
    "baseline_phase_i": dict(),
    "candidate_A_confidence_075": dict(confidence_threshold=0.75),
    "candidate_B_new_keywords": dict(use_hardware_kw=True, use_billing_kw=True, use_dataloss_kw=True),
    "candidate_C_combined": dict(confidence_threshold=0.75, use_hardware_kw=True, use_billing_kw=True, use_dataloss_kw=True),
    "candidate_D_combined_plus_vague_exempt": dict(
        confidence_threshold=0.75, use_hardware_kw=True, use_billing_kw=True, use_dataloss_kw=True,
        exempt_vague_from_evidence=True,
    ),
    # Step 4: remove-one-signal ablation (each relative to the baseline policy)
    "ablation_remove_high_risk_intent": dict(use_high_risk_intent=False),
    "ablation_remove_high_risk_keyword": dict(use_safety_kw=False),
    "ablation_remove_evidence_signal": dict(use_evidence_signal=False),
    "ablation_remove_gen_signal": dict(use_gen_signal=False),
    "ablation_remove_confidence_signal": dict(use_confidence_signal=False),
    # Step 4: each signal alone
    "ablation_only_high_risk_intent": dict(use_high_risk_intent=True, use_safety_kw=False, use_evidence_signal=False, use_gen_signal=False, use_confidence_signal=False),
    "ablation_only_high_risk_keyword": dict(use_high_risk_intent=False, use_safety_kw=True, use_evidence_signal=False, use_gen_signal=False, use_confidence_signal=False),
    "ablation_only_evidence_signal": dict(use_high_risk_intent=False, use_safety_kw=False, use_evidence_signal=True, use_gen_signal=False, use_confidence_signal=False),
    "ablation_only_gen_signal": dict(use_high_risk_intent=False, use_safety_kw=False, use_evidence_signal=False, use_gen_signal=True, use_confidence_signal=False),
    "ablation_only_confidence_signal": dict(use_high_risk_intent=False, use_safety_kw=False, use_evidence_signal=False, use_gen_signal=False, use_confidence_signal=True),
    # Step 7: rejected-signal reference point (must show a net-negative result)
    "rejected_repeat_attempt_keyword": dict(use_repeat_attempt_kw=True),
    # Step 5: evidence-threshold sensitivity (NOT a deployable candidate -- see report Section 7)
    "sensitivity_evidence_threshold_055": dict(evidence_threshold=0.55),
    "sensitivity_evidence_threshold_060": dict(evidence_threshold=0.60),
    "sensitivity_evidence_threshold_065": dict(evidence_threshold=0.65),
    "sensitivity_evidence_threshold_075": dict(evidence_threshold=0.75),
    "sensitivity_evidence_threshold_080": dict(evidence_threshold=0.80),
    "sensitivity_evidence_threshold_085": dict(evidence_threshold=0.85),
    # Step 5: confidence-threshold grid
    "sensitivity_confidence_060": dict(confidence_threshold=0.60),
    "sensitivity_confidence_065": dict(confidence_threshold=0.65),
    "sensitivity_confidence_070": dict(confidence_threshold=0.70),
    "sensitivity_confidence_075": dict(confidence_threshold=0.75),
    "sensitivity_confidence_085": dict(confidence_threshold=0.85),
    "sensitivity_confidence_090": dict(confidence_threshold=0.90),
    # Step 6: additional conditional-policy experiment -- vague-exemption alone
    "conditional_vague_exempt_alone": dict(exempt_vague_from_evidence=True),
}


def main() -> None:
    golden = load_golden()
    phase_g = load_phase_g()
    phase_h = load_phase_h()

    missing_g = [g["id"] for g in golden if g["id"] not in phase_g]
    missing_h = [g["id"] for g in golden if g["id"] not in phase_h]
    if missing_g or missing_h:
        raise RuntimeError(f"Missing Phase G data for {missing_g[:5]}, Phase H data for {missing_h[:5]}")

    logger.info("Recomputing rerank scores for all 200 golden examples (read-only)...")
    rerank_scores = recompute_rerank_scores(golden, phase_g)

    results = {}
    for name, kwargs in CANDIDATES.items():
        results[name] = evaluate_policy(golden, phase_g, phase_h, rerank_scores, **kwargs)
        m = results[name]["metrics"]
        logger.info(
            "%s: acc=%.3f prec=%.3f rec=%.3f f1=%.3f fn=%d fp=%d",
            name, m["accuracy"], m["precision_escalate"], m["recall_escalate"], m["f1_escalate"],
            m["false_auto_handle_count"], m["false_escalate_count"],
        )

    out = {
        "candidate_configs": {k: v for k, v in CANDIDATES.items()},
        "results": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", OUT_PATH)


if __name__ == "__main__":
    main()
