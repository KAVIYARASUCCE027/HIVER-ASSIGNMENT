"""Exploratory clustering over AppleSupport customer messages, used to derive
the intent taxonomy in `config/intents.yaml` (Phase D).

IMPORTANT (leakage): this script only ever reads `retrieval_pool` conversations
from `data/processed/applesupport_conversations.jsonl`. `eval_pool` must never
inform intent design -- see reports/conversation_reconstruction.md.

This is an exploratory tool, not a production module: it produces a raw
cluster dump (`reports/intent_clusters_raw.md`) for manual inspection. The
actual taxonomy in `config/intents.yaml` is hand-curated from that dump, per
CLAUDE.md's explicit instruction to inspect real examples before deciding
whether a cluster is a valid intent.

Run:
    python -m src.analysis.intent_discovery
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONVERSATIONS_PATH = Path("data/processed/applesupport_conversations.jsonl")
OUT_RAW_CLUSTERS = Path("reports/intent_clusters_raw.md")
OUT_SAMPLE_EMBEDDINGS = Path("data/processed/intent_sample_embeddings.npy")
OUT_SAMPLE_META = Path("data/processed/intent_sample_meta.jsonl")
MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")


def load_retrieval_pool_roots(path: Path) -> list[str]:
    roots = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            conv = json.loads(line)
            if conv["split"] != "retrieval_pool":
                continue
            roots.append(conv["messages"][0]["text"])
    logger.info("Loaded %d retrieval_pool root customer messages", len(roots))
    return roots


def strip_for_clustering(text: str) -> str:
    """Remove @mentions/URLs so clustering groups by topic, not by who was tagged."""
    t = MENTION_RE.sub(" ", text)
    t = URL_RE.sub(" ", t)
    return t.strip()


def overall_tfidf_terms(texts: list[str], top_n: int = 40) -> list[tuple[str, float]]:
    vec = TfidfVectorizer(max_df=0.5, min_df=5, ngram_range=(1, 2), stop_words="english")
    matrix = vec.fit_transform(texts)
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    order = np.argsort(scores)[::-1][:top_n]
    return [(terms[i], float(scores[i])) for i in order]


def cluster_top_terms(vec: TfidfVectorizer, matrix, cluster_idx: np.ndarray, top_n: int = 12) -> list[str]:
    sub = matrix[cluster_idx]
    scores = np.asarray(sub.sum(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    order = np.argsort(scores)[::-1][:top_n]
    return [terms[i] for i in order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=12000)
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples-per-cluster", type=int, default=10)
    args = parser.parse_args()

    roots = load_retrieval_pool_roots(CONVERSATIONS_PATH)
    cleaned = [strip_for_clustering(r) for r in roots]
    non_empty = [(r, c) for r, c in zip(roots, cleaned) if len(c.split()) >= 2]
    logger.info("%d / %d roots have >=2 tokens after stripping mentions/urls", len(non_empty), len(roots))

    logger.info("Top TF-IDF terms across all retrieval_pool roots:")
    top_terms = overall_tfidf_terms([c for _, c in non_empty])
    for term, score in top_terms[:40]:
        logger.info("  %-25s %.1f", term, score)

    rng = random.Random(args.seed)
    sample = rng.sample(non_empty, min(args.sample_size, len(non_empty)))
    raw_texts = [s[0] for s in sample]
    clean_texts = [s[1] for s in sample]

    logger.info("Embedding %d sampled messages ...", len(clean_texts))
    from sentence_transformers import SentenceTransformer  # deferred: slow import

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(clean_texts, show_progress_bar=True, batch_size=64)

    logger.info("Fitting KMeans (k=%d) ...", args.n_clusters)
    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    cluster_labels = km.fit_predict(embeddings)

    vec = TfidfVectorizer(max_df=0.7, min_df=3, ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = vec.fit_transform(clean_texts)

    sizes = Counter(cluster_labels)
    OUT_RAW_CLUSTERS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_RAW_CLUSTERS.open("w", encoding="utf-8") as f:
        f.write("# Raw Intent Cluster Dump (exploratory, for manual curation)\n\n")
        f.write(f"Sampled {len(clean_texts)} retrieval_pool root messages, k={args.n_clusters} KMeans clusters "
                f"over `all-MiniLM-L6-v2` embeddings.\n\n")
        f.write("## Global top TF-IDF terms\n\n")
        f.write(", ".join(t for t, _ in top_terms) + "\n\n")
        for cluster_id, size in sizes.most_common():
            idx = np.nonzero(cluster_labels == cluster_id)[0]
            terms = cluster_top_terms(vec, tfidf_matrix, idx)
            f.write(f"## Cluster {cluster_id} (n={size}, {size / len(clean_texts):.1%})\n\n")
            f.write(f"Top terms: {', '.join(terms)}\n\n")
            ex_idx = rng.sample(list(idx), min(args.examples_per_cluster, len(idx)))
            for i in ex_idx:
                f.write(f"- {raw_texts[i][:200]}\n")
            f.write("\n")
    logger.info("Wrote %s", OUT_RAW_CLUSTERS)

    # Persisted so a downstream, retrieval_pool-only-derived embedding
    # classifier (used for golden-set sampling in Phase E, never for taxonomy
    # design) doesn't need to re-run this expensive clustering step.
    np.save(OUT_SAMPLE_EMBEDDINGS, embeddings)
    with OUT_SAMPLE_META.open("w", encoding="utf-8") as f:
        for text, cid in zip(raw_texts, cluster_labels):
            f.write(json.dumps({"text": text, "cluster_id": int(cid)}, ensure_ascii=False) + "\n")
    logger.info("Wrote %s and %s", OUT_SAMPLE_EMBEDDINGS, OUT_SAMPLE_META)


if __name__ == "__main__":
    main()
