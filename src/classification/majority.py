"""Baseline 1: majority-class classifier.

Always predicts the single most frequent intent, determined ENTIRELY from
`retrieval_pool` (train) data -- specifically the cluster-based frequency
estimate already computed in Phase D
(`src.analysis.intent_coverage.cluster_based_estimate`, built from the
12,000-message retrieval_pool KMeans sample manually mapped to intents).
No golden label is inspected to pick the majority class; golden is read only
in `evaluate()`, for scoring.

Run:
    python -m src.classification.majority
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from evaluation.metrics import evaluate, format_confusion_matrix_md, format_per_class_md
from src.analysis.intent_coverage import cluster_based_estimate
from src.classification.common import load_golden, load_intent_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def majority_class() -> str:
    """The single most frequent intent per the retrieval_pool-only cluster
    estimate. Computed without reference to golden data.
    """
    totals, _ = cluster_based_estimate()
    totals.pop("__non_english_out_of_scope__", None)
    return max(totals.items(), key=lambda kv: kv[1])[0]


def run() -> dict:
    labels = load_intent_names()
    majority = majority_class()
    logger.info("Majority class (from retrieval_pool only): %s", majority)

    golden = load_golden()
    y_true = [g["intent"] for g in golden]
    y_pred = [majority] * len(golden)  # constant prediction, input-invariant by construction

    report = evaluate(y_true, y_pred, labels)
    logger.info("Majority baseline: accuracy=%.3f macro_f1=%.3f weighted_f1=%.3f",
                report.accuracy, report.macro_f1, report.weighted_f1)

    result = report.to_dict()
    result["majority_class"] = majority
    result["note"] = ("Prediction is a constant (the majority class) and is identical "
                       "for input variants A and B -- majority-vote is input-invariant by construction.")
    return result


if __name__ == "__main__":
    result = run()
    print(f"\nMajority class: {result['majority_class']}")
    print(f"Accuracy: {result['accuracy']:.3f} | Macro F1: {result['macro_f1']:.3f} | "
          f"Weighted F1: {result['weighted_f1']:.3f}\n")
    from evaluation.metrics import ClassificationReport
    rep = ClassificationReport(
        accuracy=result["accuracy"], macro_f1=result["macro_f1"], weighted_f1=result["weighted_f1"],
        per_class=result["per_class"], confusion=result["confusion_matrix"], labels=result["labels"],
    )
    print(format_per_class_md(rep))
    print()
    print(format_confusion_matrix_md(rep))
