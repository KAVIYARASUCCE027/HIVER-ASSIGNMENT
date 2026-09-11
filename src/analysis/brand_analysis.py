"""Rank candidate brands from the raw Twitter customer-support dataset.

This is a Phase-B "which brand should we build the agent for" analysis, not the
final conversation reconstruction (that is `src/ingestion/reconstruct_threads.py`,
Phase C, and only runs for the brand actually selected here).

Because the raw dataset is a flat set of ~2.8M tweets, "conversations" are
recovered by treating `in_response_to_tweet_id` as an edge back to the parent
tweet and taking connected components of that reply graph (scipy). Each
component is a (usually tree-shaped) thread. This is an approximation used
only for brand ranking:
  - a handful of threads legitimately merge two unrelated conversations when
    a brand tweet is (incorrectly, in the source data) marked as a reply to
    more than one root, or ids collide across a deleted/missing parent; we do
    not attempt to fix that here.
  - "resolved" cannot be verified without a human, so we use an explicit,
    named heuristic (see `resolved_heuristic`) and never call it ground truth.

Run:
    python -m src.analysis.brand_analysis
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_CSV = Path("data/raw/twcs.csv")
OUT_DIR = Path("reports")
MENTION_RE = re.compile(r"^@(\w+)")
STOPWORDS = {
    "the", "a", "to", "i", "you", "for", "and", "is", "in", "of", "my", "on",
    "it", "this", "your", "me", "have", "with", "please", "can", "we", "not",
    "be", "that", "are", "so", "our", "will", "just", "us", "im", "u", "at",
    "get", "но", "amp", "an", "as", "if", "was", "has", "do", "still", "but",
    "no", "how", "when", "why", "what", "or", "up", "out", "now", "all",
}


@dataclass
class BrandStats:
    brand: str
    agent_tweets: int
    customer_tweets_approx: int
    conversations: int
    conversations_with_both_sides: int
    usable_conversations: int
    resolved_heuristic_conversations: int
    avg_conversation_len: float
    top_topics: list[str]


def load_columns(csv_path: Path, columns: list[str]) -> pl.DataFrame:
    logger.info("Scanning %s for columns=%s", csv_path, columns)
    lf = pl.scan_csv(
        csv_path,
        schema_overrides={
            "tweet_id": pl.Int64,
            "author_id": pl.Utf8,
            "inbound": pl.Boolean,
            "created_at": pl.Utf8,
            "text": pl.Utf8,
            "response_tweet_id": pl.Utf8,
            "in_response_to_tweet_id": pl.Int64,
        },
    ).select(columns)
    return lf.collect(engine="streaming")


def get_brand_accounts(df_author_inbound: pl.DataFrame) -> set[str]:
    brands = (
        df_author_inbound.filter(~pl.col("inbound"))["author_id"].unique().to_list()
    )
    logger.info("Found %d distinct brand accounts (outbound author_ids)", len(brands))
    return set(brands)


def agent_tweet_counts(df_author_inbound: pl.DataFrame) -> Counter:
    counts = (
        df_author_inbound.filter(~pl.col("inbound"))
        .group_by("author_id")
        .len()
        .sort("len", descending=True)
    )
    return Counter(dict(zip(counts["author_id"].to_list(), counts["len"].to_list())))


def customer_mention_counts(csv_path: Path, brands: set[str]) -> Counter:
    """Approximate customer-tweets-per-brand via the leading @mention.

    This undercounts brands slightly (a customer reply deeper in a thread
    sometimes drops the @mention), which is exactly why the real pipeline
    (Phase C) uses graph reconstruction instead of this shortcut.
    """
    lf = pl.scan_csv(
        csv_path,
        schema_overrides={
            "tweet_id": pl.Int64, "author_id": pl.Utf8, "inbound": pl.Boolean,
            "created_at": pl.Utf8, "text": pl.Utf8,
            "response_tweet_id": pl.Utf8, "in_response_to_tweet_id": pl.Int64,
        },
    ).filter(pl.col("inbound")).select(
        pl.col("text").str.extract(r"^@(\w+)", 1).alias("mention")
    )
    mentions = lf.collect(engine="streaming")
    counts = mentions.filter(pl.col("mention").is_in(list(brands))).group_by(
        "mention"
    ).len()
    return Counter(dict(zip(counts["mention"].to_list(), counts["len"].to_list())))


def build_components(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (author_id, inbound, component_label, tweet_id) arrays, one entry per row.

    Row position in the source frame *is* the node id (0..n-1). To resolve a
    `in_response_to_tweet_id` value to the row that owns that tweet_id, we sort
    tweet_id once and binary-search it (avoids an O(n) python dict of 3M ids).
    """
    df = load_columns(
        csv_path, ["tweet_id", "author_id", "inbound", "in_response_to_tweet_id"]
    )
    n = df.height
    logger.info("Building reply graph over %d tweets", n)

    tweet_id = df["tweet_id"].to_numpy()
    parent_id = df["in_response_to_tweet_id"].to_numpy()
    inbound = df["inbound"].to_numpy()
    author_id = df["author_id"].to_numpy()

    order = np.argsort(tweet_id)
    sorted_ids = tweet_id[order]  # sorted_ids[k] == tweet_id[order[k]]

    has_parent = ~np.isnan(parent_id)
    parent_id_filled = np.nan_to_num(parent_id, nan=-1.0)
    pos = np.searchsorted(sorted_ids, parent_id_filled)
    pos_clipped = np.clip(pos, 0, n - 1)
    parent_exists = has_parent & (pos < n) & (sorted_ids[pos_clipped] == parent_id_filled)

    child_idx = np.nonzero(parent_exists)[0]
    parent_idx = order[pos[parent_exists]]

    logger.info("Reply edges resolved: %d / %d rows have a parent present in the file", len(child_idx), n)

    rows = np.concatenate([child_idx, parent_idx])
    cols = np.concatenate([parent_idx, child_idx])
    data = np.ones(len(rows), dtype=np.int8)
    adj = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    n_components, labels = connected_components(adj, directed=False)
    logger.info("Found %d connected components (approximate conversations)", n_components)
    return author_id, inbound, labels, tweet_id


def majority_brand_per_component(df: pl.DataFrame) -> pl.DataFrame:
    """Majority (mode) outbound author_id per component -> ["component", "brand"]."""
    return (
        df.filter(~pl.col("inbound"))
        .group_by(["component", "author_id"], maintain_order=True)
        .len()
        .sort("len", descending=True)
        .group_by("component", maintain_order=True)
        .first()
        .select(["component", "author_id"])
        .rename({"author_id": "brand"})
    )


def conversation_stats_per_brand(
    author_id: np.ndarray, inbound: np.ndarray, labels: np.ndarray, brands: set[str]
) -> dict[str, dict]:
    df = pl.DataFrame({"author_id": author_id, "inbound": inbound, "component": labels})

    brand_per_component = majority_brand_per_component(df)

    comp_size = df.group_by("component").len().rename({"len": "size"})
    comp_has_customer = (
        df.group_by("component").agg(pl.col("inbound").any().alias("has_customer"))
    )
    comp_has_agent = (
        df.group_by("component").agg((~pl.col("inbound")).any().alias("has_agent"))
    )

    merged = (
        brand_per_component.join(comp_size, on="component")
        .join(comp_has_customer, on="component")
        .join(comp_has_agent, on="component")
        .filter(pl.col("brand").is_in(list(brands)))
    )

    out: dict[str, dict] = {}
    for brand, sub in merged.group_by("brand"):
        brand = brand[0] if isinstance(brand, tuple) else brand
        n_conv = sub.height
        both_sides = sub.filter(pl.col("has_customer") & pl.col("has_agent")).height
        usable = sub.filter(
            (pl.col("has_customer") & pl.col("has_agent")) & (pl.col("size") >= 2)
        ).height
        avg_len = float(sub["size"].mean()) if n_conv else 0.0
        out[brand] = {
            "conversations": n_conv,
            "conversations_with_both_sides": both_sides,
            "usable_conversations": usable,
            "avg_conversation_len": avg_len,
        }
    return out


def resolved_heuristic_counts(
    csv_path: Path, author_id: np.ndarray, inbound: np.ndarray, labels: np.ndarray,
    tweet_id: np.ndarray, brands: set[str],
) -> Counter:
    """Heuristic: a conversation is 'resolved-ish' if the chronologically last
    message in the thread was sent by the brand (agent had the last word, i.e.
    the customer did not need to follow up again). This is a proxy, not a
    verified resolution label -- see reports/decision_log.md.
    """
    df_time = load_columns(csv_path, ["tweet_id", "created_at"]).with_columns(
        pl.col("created_at").str.strptime(pl.Datetime, "%a %b %d %H:%M:%S %z %Y", strict=False)
    )
    frame = (
        pl.DataFrame({
            "tweet_id": tweet_id, "author_id": author_id, "inbound": inbound, "component": labels,
        })
        .join(df_time, on="tweet_id", how="left")
        .drop_nulls("created_at")
    )

    last_is_agent = frame.group_by("component", maintain_order=True).agg(
        (~pl.col("inbound").sort_by("created_at").last()).alias("last_is_agent")
    )
    brand_per_component = majority_brand_per_component(frame)

    joined = last_is_agent.join(brand_per_component, on="component").filter(
        pl.col("brand").is_in(list(brands)) & pl.col("last_is_agent")
    )
    return Counter(joined["brand"].to_list())


def top_topics(csv_path: Path, brand: str, sample_size: int = 20000, k: int = 12) -> list[str]:
    lf = pl.scan_csv(
        csv_path,
        schema_overrides={
            "tweet_id": pl.Int64, "author_id": pl.Utf8, "inbound": pl.Boolean,
            "created_at": pl.Utf8, "text": pl.Utf8,
            "response_tweet_id": pl.Utf8, "in_response_to_tweet_id": pl.Int64,
        },
    ).filter(pl.col("inbound") & pl.col("text").str.contains(f"(?i)@{brand}")).select("text")
    texts = lf.collect(engine="streaming").head(sample_size)["text"].to_list()

    counter: Counter = Counter()
    for t in texts:
        if not t:
            continue
        cleaned = re.sub(r"https?://\S+", " ", t)
        cleaned = re.sub(r"@\w+", " ", cleaned)
        words = re.findall(r"[a-zA-Z']{3,}", cleaned.lower())
        counter.update(w for w in words if w not in STOPWORDS)
    return [w for w, _ in counter.most_common(k)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=15, help="how many top brands (by agent tweet volume) to fully profile")
    parser.add_argument("--csv", type=Path, default=RAW_CSV)
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"{args.csv} not found. See data/README.md for how to obtain it.")

    df_author_inbound = load_columns(args.csv, ["author_id", "inbound"])
    total_rows = df_author_inbound.height
    total_customer = int(df_author_inbound.filter(pl.col("inbound")).height)
    total_agent = int(df_author_inbound.filter(~pl.col("inbound")).height)
    brands = get_brand_accounts(df_author_inbound)
    agent_counts = agent_tweet_counts(df_author_inbound)
    del df_author_inbound

    top_brands = [b for b, _ in agent_counts.most_common(args.top_n)]
    logger.info("Top-%d brands by agent tweet volume: %s", args.top_n, top_brands)

    logger.info("Approximating customer tweets per brand via leading @mention ...")
    mention_counts = customer_mention_counts(args.csv, set(top_brands))

    logger.info("Building full reply graph for conversation-level stats ...")
    author_id, inbound, labels, tweet_id = build_components(args.csv)
    conv_stats = conversation_stats_per_brand(author_id, inbound, labels, set(top_brands))

    logger.info("Computing 'resolved-ish' heuristic (last message = agent) ...")
    resolved_counts = resolved_heuristic_counts(args.csv, author_id, inbound, labels, tweet_id, set(top_brands))
    del author_id, inbound, labels, tweet_id

    rows: list[BrandStats] = []
    for brand in top_brands:
        cs = conv_stats.get(brand, {})
        logger.info("Extracting topics for %s ...", brand)
        topics = top_topics(args.csv, brand)
        rows.append(
            BrandStats(
                brand=brand,
                agent_tweets=agent_counts.get(brand, 0),
                customer_tweets_approx=mention_counts.get(brand, 0),
                conversations=cs.get("conversations", 0),
                conversations_with_both_sides=cs.get("conversations_with_both_sides", 0),
                usable_conversations=cs.get("usable_conversations", 0),
                resolved_heuristic_conversations=resolved_counts.get(brand, 0),
                avg_conversation_len=round(cs.get("avg_conversation_len", 0.0), 2),
                top_topics=topics,
            )
        )

    rows.sort(key=lambda r: r.usable_conversations, reverse=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "brand_candidates.md"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Candidate Brand Ranking\n\n")
        f.write(f"Total rows in raw dataset: {total_rows:,}\n\n")
        f.write(f"Customer (inbound) tweets: {total_customer:,}\n\n")
        f.write(f"Agent (outbound) tweets: {total_agent:,}\n\n")
        f.write(f"Distinct brand accounts: {len(brands)}\n\n")
        f.write(
            "| Brand | Agent Tweets | Customer Tweets (approx) | Conversations | "
            "Both-Sides | Usable | Resolved-ish | Avg Turns | Top Topics |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r.brand} | {r.agent_tweets:,} | {r.customer_tweets_approx:,} | "
                f"{r.conversations:,} | {r.conversations_with_both_sides:,} | "
                f"{r.usable_conversations:,} | {r.resolved_heuristic_conversations:,} | "
                f"{r.avg_conversation_len} | {', '.join(r.top_topics)} |\n"
            )
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
