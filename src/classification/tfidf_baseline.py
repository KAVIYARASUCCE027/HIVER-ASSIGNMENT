"""Baseline 2: TF-IDF + Logistic Regression.

TRAINING DATA METHODOLOGY (read before trusting these numbers):
`retrieval_pool` has no human intent labels -- the golden set is the only
human-labeled data in this project, and it is held out for evaluation only.
To train a supervised classifier on "only the pre-golden training/retrieval
split" at all, this baseline pseudo-labels a sample of retrieval_pool
customer messages using the frozen, retrieval_pool-only embedding classifier
from Phase E (`src.evaluation.embedding_intent_classifier`, centroids built
from the 12,000-message KMeans sample -- never fit on or adjusted using
golden data). This is weak/distillation supervision, not ground truth: the
model is learning to approximate the embedding classifier's decision
boundary, not "true" intent. Phase D already documented that classifier's
real limitations (e.g. a confident but wrong match on the word "clock").

This bounds what this baseline can prove: it tests whether a cheap
bag-of-words model can approximate a pseudo-label signal derived purely from
train-side data, evaluated for real against human golden labels (the only
uncontaminated measure in this whole pipeline). It is NOT evidence that the
pseudo-labeler itself is accurate.

Hyperparameters are fixed a priori (not tuned against golden):
  TF-IDF: unigrams+bigrams, min_df=2, max_features=20000, English stopwords.
  LogisticRegression: class_weight="balanced" (a standard, principled choice
  given known class imbalance, decided before ever looking at golden
  results), C=1.0, max_iter=1000, random_state=42.

Run:
    python -m src.classification.tfidf_baseline
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from evaluation.metrics import evaluate
from src.classification.common import build_input_text, load_golden, load_intent_names
from src.evaluation.embedding_intent_classifier import build_intent_centroids, classify_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONVERSATIONS_PATH = Path("data/processed/applesupport_conversations.jsonl")
PSEUDO_LABELED_PATH = Path("data/processed/tfidf_pseudo_labeled_train.jsonl")

TRAIN_SAMPLE_SIZE = 20000
SEED = 42

VECTORIZER_PARAMS = dict(max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english")
LOGREG_PARAMS = dict(max_iter=1000, class_weight="balanced", C=1.0, random_state=SEED)


def load_retrieval_pool_customer_turns() -> list[dict]:
    """Every customer message in retrieval_pool, with its preceding context."""
    turns = []
    with CONVERSATIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["split"] != "retrieval_pool":
                continue
            messages = c["messages"]
            for i, m in enumerate(messages):
                if m["speaker"] == "customer":
                    turns.append({"customer_message": m["text"], "conversation": messages[:i]})
    return turns


def build_pseudo_labeled_train_set(sample_size: int = TRAIN_SAMPLE_SIZE, seed: int = SEED) -> list[dict]:
    if PSEUDO_LABELED_PATH.exists():
        logger.info("Reusing cached pseudo-labeled training set: %s", PSEUDO_LABELED_PATH)
        return [json.loads(l) for l in PSEUDO_LABELED_PATH.open(encoding="utf-8")]

    logger.info("Loading retrieval_pool customer turns ...")
    turns = load_retrieval_pool_customer_turns()
    logger.info("%d total customer turns in retrieval_pool", len(turns))

    rng = random.Random(seed)
    sample = rng.sample(turns, min(sample_size, len(turns)))

    logger.info("Building frozen intent centroids (retrieval_pool-only) ...")
    intent_names, centroids = build_intent_centroids()

    logger.info("Pseudo-labeling %d sampled customer turns ...", len(sample))
    texts = [t["customer_message"] for t in sample]
    predictions = classify_texts(texts, intent_names, centroids)

    rows = []
    for t, pred in zip(sample, predictions):
        rows.append({
            "customer_message": t["customer_message"],
            "conversation": t["conversation"],
            "pseudo_intent": pred["predicted_intent"],
        })

    PSEUDO_LABELED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PSEUDO_LABELED_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d pseudo-labeled training rows to %s", len(rows), PSEUDO_LABELED_PATH)
    return rows


def train_and_eval(variant: str, train_rows: list[dict], golden: list[dict], labels: list[str]) -> dict:
    logger.info("Training variant %s ...", variant)
    X_train_text = [build_input_text(r["customer_message"], r["conversation"], variant) for r in train_rows]
    y_train = [r["pseudo_intent"] for r in train_rows]

    vectorizer = TfidfVectorizer(**VECTORIZER_PARAMS)
    X_train = vectorizer.fit_transform(X_train_text)

    clf = LogisticRegression(**LOGREG_PARAMS)
    clf.fit(X_train, y_train)

    X_eval_text = [build_input_text(g["customer_message"], g["conversation"], variant) for g in golden]
    X_eval = vectorizer.transform(X_eval_text)
    y_pred = clf.predict(X_eval)
    y_pred_proba = clf.predict_proba(X_eval)
    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}

    y_true = [g["intent"] for g in golden]
    report = evaluate(y_true, list(y_pred), labels)

    per_example = []
    for g, true, pred, proba in zip(golden, y_true, y_pred, y_pred_proba):
        confidence = float(proba[class_to_idx[pred]]) if pred in class_to_idx else None
        per_example.append({
            "id": g["id"], "true_intent": true, "predicted_intent": pred,
            "confidence": confidence, "customer_message": g["customer_message"],
        })

    result = report.to_dict()
    result["variant"] = variant
    result["per_example"] = per_example
    logger.info("Variant %s: accuracy=%.3f macro_f1=%.3f weighted_f1=%.3f",
                variant, report.accuracy, report.macro_f1, report.weighted_f1)
    return result


def run() -> dict:
    labels = load_intent_names()
    golden = load_golden()
    train_rows = build_pseudo_labeled_train_set()

    results = {}
    for variant in ("A", "B"):
        results[variant] = train_and_eval(variant, train_rows, golden, labels)
    return results


if __name__ == "__main__":
    results = run()
    for variant, r in results.items():
        print(f"\n=== Variant {variant} ===")
        print(f"Accuracy: {r['accuracy']:.3f} | Macro F1: {r['macro_f1']:.3f} | Weighted F1: {r['weighted_f1']:.3f}")
