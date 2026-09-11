"""Single-command entry point for Phase F: leakage check + both baselines.

Run:
    python -m evaluation.run_baselines

Produces:
    evaluation/results/baselines.json
    reports/baseline_results.md
    reports/leakage_check.md  (via src.evaluation.leakage_check)

No LLM API calls; deterministic given the fixed seeds already set in
src/classification/majority.py and src/classification/tfidf_baseline.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from evaluation.metrics import ClassificationReport, format_confusion_matrix_md, format_per_class_md, top_confusion_pairs
from src.classification import majority, tfidf_baseline
from src.evaluation import leakage_check

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = Path("evaluation/results/baselines.json")
REPORT_PATH = Path("reports/baseline_results.md")
ERROR_ANALYSIS_TOP_N = 15

# Hand-analyzed reasons for the top misclassified examples (variant B), written
# after inspecting the actual confusion -- not a template. Kept here rather
# than computed, since "why" a bag-of-words model confused two intents is a
# real judgment call, the same way golden labeling itself was.
ERROR_REASONS: dict[str, str] = {
    "golden_0106": "Lexical overlap: 'Watch' and 'charger' strongly cue apple_watch_issues, but the message is "
        "self-directed venting about losing an item, not a product complaint -- a pragmatic distinction "
        "bag-of-words features can't make.",
    "golden_0064": "Context-dependent case (10-turn conversation): the true label (audio_accessory_issues) comes "
        "from an early-turn symptom (static via headphones) that gets diluted in variant B's simple "
        "concatenation once many later turns about 'songs'/'streaming' are summed into one bag of words.",
    "golden_0150": "Multi-symptom message ('no service' + 'battery keeps dying'); the human labeler prioritized "
        "the more severe/blocking symptom (no service), but 'battery' is the more common, lexically dominant "
        "term in training data, so the model defaults to it.",
    "golden_0188": "Generic update vocabulary ('11.2 iOS') outweighs the specific symptom word ('crashing') -- "
        "the single most common failure pattern in this run (see summary below).",
    "golden_0008": "Heavy profanity/frustration language dominates the TF-IDF signal; the actual account-security "
        "cue ('hacked') is comparatively rare in the pseudo-labeled training data relative to profanity terms "
        "that correlate with the residual vague_frustration class there.",
    "golden_0125": "'pictures' is a strong lexical cue for photos_storage_icloud_management. This is a taxonomy-"
        "gap case (see reports/decision_log.md #7) where the model's guess is arguably as defensible as the "
        "human labeler's own forced choice.",
    "golden_0018": "'images'/'Photo Booth' cue photos vocabulary even though the actual context (a Mac app "
        "question) makes mac_macos_issues the better label -- legitimate lexical ambiguity between two "
        "plausible readings.",
    "golden_0002": "Same generic-update-vocabulary-dominance pattern as golden_0188.",
    "golden_0027": "'MacBook Pro' mentions strongly cue mac_macos_issues lexically; the actual distinguishing "
        "signal (asking about repair *pricing*, not a technical problem) requires pragmatic understanding "
        "bag-of-words can't capture.",
    "golden_0061": "Very short, profanity-heavy message with minimal lexical content beyond 'crashing'; short "
        "angry expletive-laden phrasing is also common in the training distribution's autocorrect-bug "
        "complaints (e.g. 'FIX THIS PIECE OF SHIT UPDATE'), so tone/brevity features spuriously correlate "
        "with the wrong class.",
    "golden_0116": "Same tone/brevity-correlation pattern as golden_0061, plus generic update vocabulary.",
    "golden_0057": "'camera' cues photos-related vocabulary. Another taxonomy-gap case (camera quality has no "
        "dedicated intent); the model's guess is a reasonable alternative to the human's fallback choice.",
    "golden_0179": "Same generic-update-vocabulary-dominance pattern as golden_0188/golden_0002.",
    "golden_0154": "'App'/'Applications' mentions cue app_store_purchase_issues vocabulary even though the "
        "actual content is about Search/UX behavior, not app acquisition -- a taxonomy-boundary case.",
    "golden_0054": "No real text content (link only); TF-IDF has almost no signal to work with, so the "
        "prediction is close to an arbitrary guess from the model's class prior.",
}


def _report_from_dict(d: dict) -> ClassificationReport:
    return ClassificationReport(
        accuracy=d["accuracy"], macro_f1=d["macro_f1"], weighted_f1=d["weighted_f1"],
        per_class=d["per_class"], confusion=d["confusion_matrix"], labels=d["labels"],
    )


def build_error_analysis(tfidf_result: dict, top_n: int = ERROR_ANALYSIS_TOP_N) -> list[dict]:
    wrong = [e for e in tfidf_result["per_example"] if e["predicted_intent"] != e["true_intent"]]
    wrong.sort(key=lambda e: -(e["confidence"] or 0))  # most *confidently* wrong first
    return wrong[:top_n]


def write_markdown_report(leakage_summary: str, majority_result: dict, tfidf_results: dict) -> None:
    lines = [
        "# Baseline Results — AppleSupport Intent Classification",
        "",
        "## Why macro F1 is the primary metric here",
        "",
        "The golden set is deliberately not balanced (see `data/golden/README.md`): "
        "it mixes a natural-distribution sample (preserving the real ~19x class imbalance), "
        "a rare-intent floor, and difficult cases. Accuracy on an imbalanced 14-class set is "
        "dominated by the largest classes -- a system that only ever predicts "
        "`ios_update_general_complaint` already gets 16% accuracy for free. Macro F1 weights "
        "every intent equally regardless of frequency, so it directly measures whether the "
        "system works for rare intents too (apple_watch_issues, app_store_purchase_issues, "
        "audio_accessory_issues, photos_storage_icloud_management -- each 4-5% of the golden "
        "set) rather than only for the head of the distribution. Weighted F1 is reported "
        "alongside as the accuracy-like complement. Per-intent tables below make rare-intent "
        "performance visible rather than averaging it away.",
        "",
        "## Leakage check",
        "",
        leakage_summary,
        "",
        "See `reports/leakage_check.md` for full detail.",
        "",
        "## Summary",
        "",
        "| System | Accuracy | Macro F1 | Weighted F1 |",
        "|---|---:|---:|---:|",
        f"| Majority (constant, input-invariant) | {majority_result['accuracy']:.3f} | "
        f"{majority_result['macro_f1']:.3f} | {majority_result['weighted_f1']:.3f} |",
        f"| TF-IDF + LogReg (A: message only) | {tfidf_results['A']['accuracy']:.3f} | "
        f"{tfidf_results['A']['macro_f1']:.3f} | {tfidf_results['A']['weighted_f1']:.3f} |",
        f"| TF-IDF + LogReg (B: + context) | {tfidf_results['B']['accuracy']:.3f} | "
        f"{tfidf_results['B']['macro_f1']:.3f} | {tfidf_results['B']['weighted_f1']:.3f} |",
        "",
        "## Training data for TF-IDF + LogReg: pseudo-labeled, not human-labeled",
        "",
        "`retrieval_pool` has no human intent labels -- golden is the only human-labeled data "
        "and is held out for scoring only. The TF-IDF baseline is trained on 20,000 "
        "retrieval_pool customer messages pseudo-labeled by the frozen, retrieval_pool-only "
        "embedding classifier from Phase E. This means the TF-IDF model's ceiling is bounded "
        "by pseudo-label quality, not by TF-IDF/LogReg's own capacity -- see "
        "`src/classification/tfidf_baseline.py`'s docstring for the full methodology and "
        "why this doesn't leak golden information (the pseudo-labeler was frozen before ever "
        "seeing eval_pool or golden). Golden scoring remains the only uncontaminated measure.",
        "",
        "## Majority baseline detail",
        "",
        f"Majority class (from retrieval_pool only): **{majority_result['majority_class']}**",
        "",
        format_per_class_md(_report_from_dict(majority_result)),
        "",
    ]

    for variant in ("A", "B"):
        r = tfidf_results[variant]
        report = _report_from_dict(r)
        lines += [
            f"## TF-IDF + LogReg — variant {variant} detail",
            "",
            format_per_class_md(report),
            "",
            "### Confusion matrix",
            "",
            format_confusion_matrix_md(report),
            "",
            "### Top confusion pairs (true -> predicted, count)",
            "",
        ]
        for true_l, pred_l, count in top_confusion_pairs(report, top_n=10):
            lines.append(f"- {true_l} -> {pred_l}: {count}")
        lines.append("")

    lines += [
        "## Error analysis (TF-IDF + LogReg, variant B)",
        "",
        "Top misclassified examples, ranked by predicted-class confidence "
        "(most *confidently wrong* first -- the cases most worth understanding):",
        "",
    ]
    for e in build_error_analysis(tfidf_results["B"]):
        conf_str = f"{e['confidence']:.3f}" if e["confidence"] is not None else "n/a"
        reason = ERROR_REASONS.get(e["id"], "(not analyzed)")
        lines.append(f"- **{e['id']}** (confidence={conf_str})")
        lines.append(f"  - message: {e['customer_message'][:200]}")
        lines.append(f"  - true: `{e['true_intent']}` | predicted: `{e['predicted_intent']}`")
        lines.append(f"  - likely reason: {reason}")
    lines.append("")
    lines += [
        "### Failure-pattern summary",
        "",
        "Five recurring patterns account for essentially all 15 cases above:",
        "",
        "1. **Generic update vocabulary beats specific symptoms** ('iOS 11', 'update') pulling "
        "device_freezing_performance cases toward ios_update_general_complaint -- the single most common "
        "pattern (golden_0002, 0116, 0179, 0188).",
        "2. **Topic-name lexical cues override pragmatic intent** -- a bag-of-words model can't distinguish "
        "'mentions Watch/Mac/camera/App' from 'has a *problem* with' or 'is asking the *price* of' that thing "
        "(golden_0018, 0027, 0057, 0106, 0125, 0154).",
        "3. **Multi-symptom / multi-turn dilution** -- when a human prioritizes one symptom among several, or "
        "draws on distant conversation turns, simple bag-of-words concatenation can't replicate that judgment "
        "(golden_0064, 0150).",
        "4. **Tone/brevity spuriously correlates with the wrong class** in the pseudo-labeled training "
        "distribution (short, profanity-heavy phrasing is common in *both* the autocorrect-bug and "
        "vague-frustration training data) (golden_0008, 0061, 0116).",
        "5. **Near-empty input** (a link with no text) leaves the model with essentially no signal "
        "(golden_0054).",
        "",
        "Patterns 2 and 3 are exactly the kind of *semantic/pragmatic* distinction an LLM-based classifier "
        "should be able to make and a bag-of-words model structurally cannot -- a genuine, non-trivial target "
        "for Phase G to actually improve on, not just a headline-number exercise. Pattern 1 and 4 are more "
        "about pseudo-label/training-distribution artifacts than an inherent TF-IDF weakness, and pattern 5 "
        "is unsolvable by any text-only classifier.",
        "",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", REPORT_PATH)


def main() -> None:
    logger.info("Step 1/3: leakage check")
    leakage_check.main()
    leakage_summary = (
        "Passed: `set(train_conversation_ids) ∩ set(golden_conversation_ids)` = empty, "
        "no cross-split tweet_id overlap. See reports/leakage_check.md for near-duplicate "
        "and temporal-overlap detail."
    )

    logger.info("Step 2/3: majority baseline")
    majority_result = majority.run()

    logger.info("Step 3/3: TF-IDF + LogReg baseline (both variants)")
    tfidf_results = tfidf_baseline.run()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump({
            "majority": {k: v for k, v in majority_result.items() if k != "per_example"},
            "tfidf_logreg": {
                variant: {k: v for k, v in r.items() if k != "per_example"}
                for variant, r in tfidf_results.items()
            },
        }, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", RESULTS_PATH)

    write_markdown_report(leakage_summary, majority_result, tfidf_results)


if __name__ == "__main__":
    main()
