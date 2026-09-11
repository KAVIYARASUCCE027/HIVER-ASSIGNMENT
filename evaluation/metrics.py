"""Shared classification-metric utilities for the baseline evaluations.

Kept deliberately thin: everything here is a light wrapper over
scikit-learn's metrics so both `src/classification/majority.py` and
`src/classification/tfidf_baseline.py` compute and report metrics the exact
same way (accuracy, macro/weighted F1, per-class P/R/F1, confusion matrix).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class ClassificationReport:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]  # label -> {precision, recall, f1, support}
    confusion: list[list[int]]
    labels: list[str]
    predictions: list[dict] = field(default_factory=list)  # per-example detail for error analysis

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion,
            "labels": self.labels,
        }


def evaluate(y_true: list[str], y_pred: list[str], labels: list[str]) -> ClassificationReport:
    """`labels` should be the full fixed intent taxonomy (all 14), not just
    labels observed in this particular y_true/y_pred, so per-class rows and
    the confusion matrix are stable and comparable across systems even if
    one system never predicts a given rare class.
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return ClassificationReport(
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        per_class=per_class,
        confusion=cm,
        labels=labels,
    )


def top_confusion_pairs(report: ClassificationReport, top_n: int = 10) -> list[tuple[str, str, int]]:
    """Off-diagonal (true, predicted, count) triples, largest first."""
    pairs = []
    for i, true_label in enumerate(report.labels):
        for j, pred_label in enumerate(report.labels):
            if i == j:
                continue
            count = report.confusion[i][j]
            if count > 0:
                pairs.append((true_label, pred_label, count))
    return sorted(pairs, key=lambda x: -x[2])[:top_n]


def format_confusion_matrix_md(report: ClassificationReport) -> str:
    labels = report.labels
    short = [l[:12] for l in labels]
    header = "| true \\ pred | " + " | ".join(short) + " |"
    sep = "|---" * (len(labels) + 1) + "|"
    rows = [header, sep]
    for i, label in enumerate(labels):
        row = [short[i]] + [str(v) for v in report.confusion[i]]
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


def format_per_class_md(report: ClassificationReport) -> str:
    rows = ["| Intent | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for label in report.labels:
        pc = report.per_class[label]
        rows.append(
            f"| {label} | {pc['precision']:.3f} | {pc['recall']:.3f} | {pc['f1']:.3f} | {pc['support']} |"
        )
    return "\n".join(rows)
