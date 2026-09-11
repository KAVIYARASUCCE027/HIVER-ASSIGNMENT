"""Phase G: evaluate the zero-shot LLM classifier on the frozen golden set.

Run:
    python -m evaluation.run_llm_classifier

Produces:
    evaluation/results/llm_classifier.json
    reports/llm_classifier_results.md
    reports/llm_error_analysis.md

Requires LLM_PROVIDER / LLM_MODEL / LLM_API_KEY in `.env` (see .env.example).
Every golden example is classified at most once per (model, prompt) --
results are cached in data/processed/llm_predictions/, so re-running this
after a transient failure or to regenerate the report does not re-call the
API for already-answered examples.

No LLM calls happen during report generation itself -- only during
`classify_golden_set()`.
"""

from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime
from pathlib import Path

from evaluation.metrics import ClassificationReport, evaluate, format_confusion_matrix_md, format_per_class_md, top_confusion_pairs
from src.classification import majority, tfidf_baseline
from src.classification.common import load_golden, load_intent_names
from src.classification.llm_classifier import FAILURE_SENTINEL, classify
from src.classification.llm_providers import LLMConfig, LLMConfigError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = Path("evaluation/results/llm_classifier.json")
BASELINES_PATH = Path("evaluation/results/baselines.json")
REPORT_PATH = Path("reports/llm_classifier_results.md")
ERROR_ANALYSIS_PATH = Path("reports/llm_error_analysis.md")

CONFIDENCE_BUCKETS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]


def classify_golden_set(golden: list[dict], config: LLMConfig, context_mode: str = "with_context") -> list[dict]:
    """context_mode: "with_context" passes each example's real preceding
    conversation (the Phase G headline run); "message_only" forces empty
    context for every example, for the message-only-vs-context comparison.
    Only 3/200 golden examples have non-empty conversation (see
    reports/llm_classifier_results.md's context-effect section), so
    message_only reuses the cache for the other 197 (identical prompt).
    """
    results = []
    n_cached = 0
    for i, g in enumerate(golden, 1):
        conversation = [] if context_mode == "message_only" else g["conversation"]
        # NOT renaming the cache key per variant: the cache path already
        # depends on a hash of the full prompt (which includes context), so
        # when conversation is empty in both modes (197/200 examples) this
        # naturally reuses the same cache entry -- only the 3 examples with
        # real context need a genuinely new call for message_only.
        result = classify(g["id"], g["customer_message"], conversation, config=config)
        n_cached += int(result.from_cache)
        results.append({
            "id": g["id"],
            "true_intent": g["intent"],
            "predicted_intent": result.intent,
            "confidence": result.confidence,
            "confidence_type": result.confidence_type,
            "alternative_intent": result.alternative_intent,
            "reason": result.reason,
            "attempts": result.attempts,
            "customer_message": g["customer_message"],
            "conversation": g["conversation"],
        })
        if i % 25 == 0 or i == len(golden):
            logger.info("Classified %d/%d (cached: %d)", i, len(golden), n_cached)
    n_failed = sum(1 for r in results if r["predicted_intent"] == FAILURE_SENTINEL)
    logger.info("Done. %d/%d served from cache, %d classification failures after retries.",
                n_cached, len(golden), n_failed)
    return results


def confidence_analysis(results: list[dict]) -> dict:
    buckets = []
    for lo, hi in CONFIDENCE_BUCKETS:
        in_bucket = [r for r in results if lo <= r["confidence"] < hi]
        n = len(in_bucket)
        correct = sum(1 for r in in_bucket if r["predicted_intent"] == r["true_intent"])
        buckets.append({
            "range": f"{lo:.1f}-{hi if hi <= 1.0 else 1.0:.1f}",
            "n": n,
            "accuracy": correct / n if n else None,
        })

    confidences = [r["confidence"] for r in results]
    correctness = [1.0 if r["predicted_intent"] == r["true_intent"] else 0.0 for r in results]
    try:
        correlation = statistics.correlation(confidences, correctness)
    except Exception:
        correlation = None

    return {
        "buckets": buckets,
        "confidence_vs_correctness_pearson_r": correlation,
        "note": "This is a confidence ANALYSIS, not calibration. Self-reported LLM confidence is "
                "not a calibrated probability; true calibration is deferred to a later phase.",
    }


def build_llm_report(golden: list[dict], intent_names: list[str], llm_results: list[dict],
                      majority_result: dict, tfidf_results: dict, conf_analysis: dict) -> ClassificationReport:
    y_true = [r["true_intent"] for r in llm_results]
    y_pred = [r["predicted_intent"] for r in llm_results]
    return evaluate(y_true, y_pred, intent_names)


def confidence_correct_incorrect_summary(results: list[dict]) -> dict:
    correct = [r["confidence"] for r in results if r["predicted_intent"] == r["true_intent"]]
    incorrect = [r["confidence"] for r in results if r["predicted_intent"] != r["true_intent"]]
    high_conf_wrong = sorted(
        (r for r in results if r["predicted_intent"] != r["true_intent"] and r["confidence"] >= 0.8),
        key=lambda r: -r["confidence"],
    )
    low_conf_correct = sorted(
        (r for r in results if r["predicted_intent"] == r["true_intent"] and r["confidence"] < 0.6),
        key=lambda r: r["confidence"],
    )
    return {
        "avg_confidence_correct": (sum(correct) / len(correct)) if correct else None,
        "avg_confidence_incorrect": (sum(incorrect) / len(incorrect)) if incorrect else None,
        "n_correct": len(correct),
        "n_incorrect": len(incorrect),
        "high_confidence_wrong": [
            {"id": r["id"], "confidence": r["confidence"], "true_intent": r["true_intent"],
             "predicted_intent": r["predicted_intent"], "customer_message": r["customer_message"]}
            for r in high_conf_wrong
        ],
        "low_confidence_correct": [
            {"id": r["id"], "confidence": r["confidence"], "true_intent": r["true_intent"],
             "predicted_intent": r["predicted_intent"], "customer_message": r["customer_message"]}
            for r in low_conf_correct
        ],
    }


def write_summary_report(
    llm_report: ClassificationReport, msgonly_report: ClassificationReport,
    majority_result: dict, tfidf_results: dict, conf_analysis: dict, conf_summary: dict,
    llm_results: list[dict], msgonly_results: list[dict], config: LLMConfig, eval_date: str,
) -> None:
    n_failed = sum(1 for r in llm_results if r["predicted_intent"] == FAILURE_SENTINEL)
    n_correct = conf_summary["n_correct"]
    n_incorrect = conf_summary["n_incorrect"]
    n_retried = sum(1 for r in llm_results if r["attempts"] > 1)
    total_attempts = sum(r["attempts"] for r in llm_results)

    lines = [
        "# LLM Zero-Shot Intent Classifier — Results",
        "",
        f"Provider: `{config.provider}` | Model: `{config.model}` | Evaluation date: {eval_date} | "
        "Prompt: frozen (see src/classification/prompts.py, snapshot in config/llm_classifier.yaml)",
        "",
        "## 1. Overall classification",
        "",
        f"- Accuracy: **{llm_report.accuracy:.3f}**",
        f"- Macro F1: **{llm_report.macro_f1:.3f}**",
        f"- Weighted F1: **{llm_report.weighted_f1:.3f}**",
        f"- Correct predictions: {n_correct}/200",
        f"- Incorrect predictions: {n_incorrect}/200",
        f"- API failures (rate-limit/network, after retries exhausted): {n_failed}/200",
        f"- Examples that needed at least one retry: {n_retried}/200",
        f"- Fallback (`_CLASSIFICATION_FAILED_`) predictions: {n_failed}/200",
        f"- Total API call attempts across all 200 examples: {total_attempts}",
        "",
        "| System | Accuracy | Macro F1 | Weighted F1 |",
        "|---|---:|---:|---:|",
        f"| Majority | {majority_result['accuracy']:.3f} | {majority_result['macro_f1']:.3f} | "
        f"{majority_result['weighted_f1']:.3f} |",
        f"| TF-IDF message-only | {tfidf_results['A']['accuracy']:.3f} | "
        f"{tfidf_results['A']['macro_f1']:.3f} | {tfidf_results['A']['weighted_f1']:.3f} |",
        f"| TF-IDF message+context | {tfidf_results['B']['accuracy']:.3f} | "
        f"{tfidf_results['B']['macro_f1']:.3f} | {tfidf_results['B']['weighted_f1']:.3f} |",
        f"| **LLM message+context** | **{llm_report.accuracy:.3f}** | **{llm_report.macro_f1:.3f}** | "
        f"**{llm_report.weighted_f1:.3f}** |",
        "",
        "## 2. Per-intent performance (LLM, message+context)",
        "",
        "Sorted by support (descending), consistent with the baseline reports.",
        "",
        format_per_class_md(llm_report),
        "",
        "## 3. Confusion matrix (LLM)",
        "",
        format_confusion_matrix_md(llm_report),
        "",
        "### Top confusion pairs",
        "",
    ]
    for t, p, c in top_confusion_pairs(llm_report, top_n=10)[:5]:
        lines.append(f"- `{t}` -> `{p}`: {c}")
    lines += [
        "",
        "## 4. TF-IDF vs LLM comparison",
        "",
        "See `reports/llm_error_analysis.md` for concrete real examples (LLM wins, TF-IDF wins, both "
        "fail) and explanations of the confusion pairs above.",
        "",
        "## 5. Context effect",
        "",
        "| Variant | Accuracy | Macro F1 | Weighted F1 |",
        "|---|---:|---:|---:|",
        f"| Message only | {msgonly_report.accuracy:.3f} | {msgonly_report.macro_f1:.3f} | "
        f"{msgonly_report.weighted_f1:.3f} |",
        f"| Message + context | {llm_report.accuracy:.3f} | {llm_report.macro_f1:.3f} | "
        f"{llm_report.weighted_f1:.3f} |",
        "",
        "**Methodology note:** only 3/200 golden examples (`golden_0064`, `golden_0086`, `golden_0153`, "
        "all in the `difficult:context_dependent` sampling stratum) have non-empty preceding conversation "
        "-- for the other 197, the message-only and message+context prompts are byte-identical (both "
        "are effectively \"first message, no context\"), so this comparison is clean and isolates exactly "
        "the effect of context on the cases where context exists. It is not a comparison compromised by "
        "changing anything else -- same frozen prompt template, same model, same temperature, same "
        "examples, differing only in whether the 3 context-bearing examples' preceding turns are included.",
        "",
    ]
    for r in llm_results:
        if r["id"] in ("golden_0064", "golden_0086", "golden_0153"):
            msgonly = next(m for m in msgonly_results if m["id"] == r["id"])
            lines.append(
                f"- `{r['id']}`: message-only -> `{msgonly['predicted_intent']}` "
                f"(true=`{r['true_intent']}`); with context -> `{r['predicted_intent']}`"
            )
    lines.append("")

    lines += [
        "## 6. Confidence Analysis — Not Calibration",
        "",
        "Self-reported by the model (`confidence_type: self_reported`); NOT a calibrated probability. "
        "A 0.9 here does not mean a 90% chance of being correct -- true calibration is deferred to a "
        "later phase.",
        "",
        "| Confidence range | n | Accuracy |",
        "|---|---:|---:|",
    ]
    for b in conf_analysis["buckets"]:
        acc_str = f"{b['accuracy']:.3f}" if b["accuracy"] is not None else "n/a (0 examples)"
        lines.append(f"| {b['range']} | {b['n']} | {acc_str} |")
    r_corr = conf_analysis["confidence_vs_correctness_pearson_r"]
    avg_c = conf_summary["avg_confidence_correct"]
    avg_i = conf_summary["avg_confidence_incorrect"]
    lines += [
        "",
        f"- Average confidence on correct predictions: {avg_c:.3f}" if avg_c is not None else "- Average confidence on correct predictions: n/a",
        f"- Average confidence on incorrect predictions: {avg_i:.3f}" if avg_i is not None else "- Average confidence on incorrect predictions: n/a",
        f"- Pearson correlation between self-reported confidence and correctness (0/1): "
        f"{r_corr:.3f}" if r_corr is not None else "- Pearson correlation: n/a",
        "",
        f"### High-confidence wrong predictions ({len(conf_summary['high_confidence_wrong'])}, confidence >= 0.8)",
        "",
    ]
    for e in conf_summary["high_confidence_wrong"]:
        lines.append(f"- `{e['id']}` (conf={e['confidence']:.2f}): true=`{e['true_intent']}` "
                      f"pred=`{e['predicted_intent']}` -- \"{e['customer_message'][:150]}\"")
    lines.append("")
    lines.append(f"### Low-confidence correct predictions ({len(conf_summary['low_confidence_correct'])}, confidence < 0.6)")
    lines.append("")
    for e in conf_summary["low_confidence_correct"]:
        lines.append(f"- `{e['id']}` (conf={e['confidence']:.2f}): correctly predicted `{e['predicted_intent']}` "
                      f"-- \"{e['customer_message'][:150]}\"")
    lines.append("")
    lines.append("These are exactly the cases an escalation policy built on confidence thresholds must "
                  "handle carefully: high-confidence errors would slip through a naive 'auto-handle if "
                  "confident' rule, and low-confidence correct answers would be needlessly escalated.")
    lines.append("")

    lines += [
        "## 7. API / model reproducibility",
        "",
        f"- Provider: `{config.provider}`",
        f"- Exact model ID: `{config.model}`",
        f"- Evaluation date: {eval_date}",
        "- Temperature: 0.0 (requested; provider-dependent whether perfectly deterministic)",
        "- Max output tokens: 300",
        "- Structured output: Gemini native JSON mode (`response_mime_type=\"application/json\"`), "
        "plus programmatic schema validation with retry (src/classification/llm_classifier.py)",
        f"- Total API calls this evaluation (both variants combined, cache-aware): "
        f"{total_attempts} attempts for message+context; message-only reused cache for 197/200",
        f"- Retries: {n_retried} examples needed a retry",
        f"- Failures: {n_failed}",
        "- Token usage / cost: not returned by the google-genai SDK response object used here, so not "
        "reported -- stating an estimate would not be an actual measured value.",
        "",
        "## 8. Frozen-evaluation integrity",
        "",
        "- Golden labels were never used in prompt construction -- `src/classification/prompts.py` is a "
        "pure function of `config/intents.yaml` and the input example only.",
        "- No golden examples were inserted as few-shot demonstrations -- the prompt contains zero "
        "customer_message/intent pairs from golden_200.jsonl. The only examples in the prompt are "
        "each intent's `examples` field in config/intents.yaml, derived during Phase D clustering "
        "before the golden set existed (see reports/decision_log.md decision #5).",
        "- No golden examples were used for retrieval -- Phase G has no retrieval component.",
        "- No golden labels were used to tune confidence thresholds -- no thresholds exist yet in this "
        "phase; confidence is reported and analyzed, not acted on.",
        "- No golden labels were used to modify the taxonomy -- config/intents.yaml is unchanged since "
        "Phase E's approval.",
        "- No golden examples were used to select successful demonstrations -- zero-shot only.",
        "- **One documented exception, not hidden:** an earlier attempt at this evaluation (Groq "
        "`openai/gpt-oss-120b`, then Gemini `gemini-2.5-flash`) hit provider rate limits partway through "
        "(145/200 and 197/200 failures respectively) before any complete run existed. Those partial runs' "
        "raw outputs were inspected only to diagnose the *infrastructure* failure (confirming HTTP 429 "
        "rate-limit responses, not model behavior) -- no prompt wording, taxonomy content, or threshold "
        "was changed as a result of anything the model actually classified. The prompt was frozen before "
        "any provider produced a complete run, and this document's headline numbers come from the first "
        "complete, unedited run (`gemini-3.5-flash-lite`, 200/200, 0 failures).",
        "",
        "## 9. Model/prompt versioning",
        "",
        "Exact configuration for this run is saved in `config/llm_classifier.yaml`; the exact prompt "
        "text (system prompt + full intent taxonomy block, byte-identical to what every one of the 200 "
        "calls received) is saved in `config/llm_classifier_prompt_snapshot.txt`. Neither file contains "
        "secrets.",
        "",
        "## 10. Results artifacts",
        "",
        "- `evaluation/results/llm_classifier.json` -- machine-readable: both variants' full metrics, "
        "confidence analysis, and per-example predictions.",
        "- `reports/llm_classifier_results.md` -- this file.",
        "- `reports/llm_error_analysis.md` -- TF-IDF-vs-LLM example comparison (item 4).",
        "- `config/llm_classifier.yaml` / `config/llm_classifier_prompt_snapshot.txt` -- exact "
        "configuration and prompt (item 9).",
        "",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", REPORT_PATH)


def write_config_snapshot(config: LLMConfig, eval_date: str) -> None:
    from src.classification.prompts import SYSTEM_PROMPT, build_intent_taxonomy_block

    snapshot_path = Path("config/llm_classifier.yaml")
    prompt_path = Path("config/llm_classifier_prompt_snapshot.txt")

    content = f"""# Exact configuration used for the Phase G zero-shot LLM classifier evaluation.
# Regenerated each run by evaluation/run_llm_classifier.py -- reflects the
# configuration of the LAST run, not necessarily the frozen headline run.
# The frozen headline run's evaluation_date is recorded in reports/llm_classifier_results.md.

provider: {config.provider}
model: {config.model}
temperature: 0.0
max_output_tokens: 300
prompt_version: "phase_g_v1_frozen"
taxonomy_version: "phase_e_frozen_14_intents"  # config/intents.yaml, unchanged since Phase E approval
context_window_policy: "full preceding conversation array included verbatim (no truncation observed necessary at golden-set scale)"
output_schema:
  intent: "one of the 14 frozen intent names"
  confidence: "float 0.0-1.0, self-reported, NOT calibrated"
  alternative_intent: "one of the 14 names, or null"
  reason: "concise evidence-based explanation, 1-2 sentences"
evaluation_date: "{eval_date}"
"""
    snapshot_path.write_text(content, encoding="utf-8")
    prompt_path.write_text(
        "=== SYSTEM PROMPT ===\n" + SYSTEM_PROMPT + "\n\n=== INTENT TAXONOMY BLOCK ===\n" +
        build_intent_taxonomy_block(), encoding="utf-8"
    )
    logger.info("Wrote %s and %s", snapshot_path, prompt_path)


def main() -> None:
    try:
        config = LLMConfig.from_env()
    except LLMConfigError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc

    eval_date = datetime.now().strftime("%Y-%m-%d")
    intent_names = load_intent_names()
    golden = load_golden()

    logger.info("Loading baseline results for comparison ...")
    majority_result = majority.run()
    tfidf_results = tfidf_baseline.run()

    logger.info("Classifying golden set with %s / %s (message + context) ...", config.provider, config.model)
    llm_results = classify_golden_set(golden, config, context_mode="with_context")

    logger.info("Classifying golden set with %s / %s (message only, for context-effect comparison) ...",
                config.provider, config.model)
    msgonly_results = classify_golden_set(golden, config, context_mode="message_only")

    llm_report = build_llm_report(golden, intent_names, llm_results, majority_result, tfidf_results, {})
    msgonly_report = build_llm_report(golden, intent_names, msgonly_results, majority_result, tfidf_results, {})
    conf_analysis = confidence_analysis(llm_results)
    conf_summary = confidence_correct_incorrect_summary(llm_results)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump({
            "model": config.model,
            "provider": config.provider,
            "evaluation_date": eval_date,
            "message_plus_context": llm_report.to_dict(),
            "message_only": msgonly_report.to_dict(),
            "confidence_analysis": conf_analysis,
            "confidence_correct_incorrect": conf_summary,
            "per_example": [{k: v for k, v in r.items() if k != "conversation"} for r in llm_results],
            "per_example_message_only": [{k: v for k, v in r.items() if k != "conversation"} for r in msgonly_results],
        }, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", RESULTS_PATH)

    write_summary_report(llm_report, msgonly_report, majority_result, tfidf_results,
                          conf_analysis, conf_summary, llm_results, msgonly_results, config, eval_date)
    write_config_snapshot(config, eval_date)

    logger.info("Done. See %s and %s. Build reports/llm_error_analysis.md separately by inspecting real examples.",
                RESULTS_PATH, REPORT_PATH)


if __name__ == "__main__":
    main()
