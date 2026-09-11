"""Reconstruct AppleSupport conversations from the raw Twitter dataset.

Phase A/B (`src/analysis/brand_analysis.py`) used raw *undirected connected
components* of the reply graph to rank brands. That approach is only an
approximation: Twitter lets a third party reply to a public tweet, so an
agent's public reply can become a hub that undirected-connects two
*unrelated* customers into one artificial "component". This module fixes
that for the brand we actually ship: reconstruction here is customer-identity
aware. Starting from each customer's root tweet, we grow the conversation by
following `in_response_to_tweet_id` children, but only through nodes authored
by that same customer or by the brand -- a reply from a different customer
tweet-jacking the thread stops that branch instead of merging two
conversations together.

Run:
    python -m src.ingestion.reconstruct_threads
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from src.analysis.brand_analysis import build_components, load_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_CSV = Path("data/raw/twcs.csv")
OUT_JSONL = Path("data/processed/applesupport_conversations.jsonl")
OUT_REPORT = Path("reports/conversation_reconstruction.md")
BRAND = "AppleSupport"
TIMESTAMP_FMT = "%a %b %d %H:%M:%S %z %Y"

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\-\s()]{8,14}\d)(?!\d)")
WHITESPACE_RE = re.compile(r"\s+")

# Retrieval/train vs held-out eval split. Conversations whose root tweet was
# posted on/after this fraction of the way through AppleSupport's observed
# time range go to eval_pool; everything earlier goes to retrieval_pool. See
# `assign_splits` and reports/conversation_reconstruction.md for the resolved
# calendar date and rationale.
EVAL_HOLDOUT_FRACTION = 0.15

# A customer whose account never gets a brand reply but keeps self-replying
# past this many turns is, empirically, not writing "one conversation" -- it's
# almost always someone's unrelated personal thread that happens to mention
# the brand once (see reports/conversation_reconstruction.md, "Discovered
# during validation"). Threshold set from the observed distribution: p99.9 of
# customer-only conversation length is 5 turns, then it jumps straight to 38.
DEGENERATE_CUSTOMER_ONLY_TURN_THRESHOLD = 5


@dataclass
class RawRow:
    tweet_id: int
    author_id: str
    inbound: bool
    created_at: datetime | None
    text: str
    in_response_to_tweet_id: int | None


@dataclass
class ReconstructionStats:
    brand_relevant_tweets: int = 0
    duplicate_tweet_ids_dropped: int = 0
    malformed_timestamp_dropped: int = 0
    empty_text_dropped: int = 0
    missing_parent_count: int = 0
    response_id_inconsistencies: int = 0
    third_party_branches_pruned: int = 0
    degenerate_self_threads_dropped: int = 0
    conversations: int = 0
    customer_only_conversations: int = 0
    customer_agent_conversations: int = 0
    turn_counts: list[int] = field(default_factory=list)
    agent_responded_count: int = 0
    multi_customer_turn_count: int = 0
    multi_agent_turn_count: int = 0
    large_time_gap_conversations: int = 0


def load_raw_rows(csv_path: Path) -> pl.DataFrame:
    logger.info("Loading full raw dataset for graph context: %s", csv_path)
    return load_columns(
        csv_path,
        ["tweet_id", "author_id", "inbound", "created_at", "text", "in_response_to_tweet_id"],
    )


def clean_text(text: str) -> str:
    """Whitespace-normalize and mask unambiguous PII (email/phone). Deliberately
    does NOT strip order numbers, names, or other support-relevant detail --
    see reports/decision_log.md for why.
    """
    t = WHITESPACE_RE.sub(" ", text).strip()
    t = EMAIL_RE.sub("[EMAIL]", t)
    t = PHONE_RE.sub("[PHONE]", t)
    return t


def dedupe_and_clean(df: pl.DataFrame, stats: ReconstructionStats) -> pl.DataFrame:
    """Defensive cleaning: duplicate tweet_id rows, malformed timestamps, empty text."""
    before = df.height
    df = df.unique(subset=["tweet_id"], keep="first", maintain_order=True)
    stats.duplicate_tweet_ids_dropped = before - df.height

    df = df.with_columns(
        pl.col("created_at").str.strptime(pl.Datetime, TIMESTAMP_FMT, strict=False).alias("parsed_ts")
    )
    before = df.height
    bad_ts = df.filter(pl.col("parsed_ts").is_null())
    stats.malformed_timestamp_dropped = bad_ts.height
    df = df.filter(pl.col("parsed_ts").is_not_null())

    before = df.height
    df = df.filter(pl.col("text").str.strip_chars().str.len_chars() > 0)
    stats.empty_text_dropped = before - df.height

    return df


def build_parent_validity(tweet_id: np.ndarray, parent_id: np.ndarray) -> np.ndarray:
    """Return, per row, the row-index of its parent if that parent tweet_id
    exists in `tweet_id`, else -1 (covers both "no parent" and "missing/
    deleted parent" -- both are handled identically downstream: the row
    becomes a potential conversation root).
    """
    n = len(tweet_id)
    order = np.argsort(tweet_id)
    sorted_ids = tweet_id[order]

    has_parent = ~np.isnan(parent_id)
    parent_filled = np.nan_to_num(parent_id, nan=-1.0)
    pos = np.searchsorted(sorted_ids, parent_filled)
    pos_clipped = np.clip(pos, 0, n - 1)
    exists = has_parent & (pos < n) & (sorted_ids[pos_clipped] == parent_filled)

    parent_row = np.full(n, -1, dtype=np.int64)
    parent_row[exists] = order[pos[exists]]
    return parent_row


def find_candidate_conversations(csv_path: Path, brand: str) -> set[int]:
    """Cheap pre-filter: which connected components (Phase B style) touch `brand`?
    Returns the set of tweet_ids belonging to those components -- the search
    space we then reconstruct precisely. This purely narrows scope; it does
    not decide final conversation boundaries (see module docstring).
    """
    author_id, inbound, labels, tweet_id = build_components(csv_path)
    df = pl.DataFrame({"tweet_id": tweet_id, "author_id": author_id, "component": labels})
    brand_components = (
        df.filter(pl.col("author_id") == brand)["component"].unique().to_list()
    )
    relevant_ids = df.filter(pl.col("component").is_in(brand_components))["tweet_id"].to_list()
    logger.info(
        "Candidate pre-filter: %d components touch %s, covering %d tweets",
        len(brand_components), brand, len(relevant_ids),
    )
    return set(relevant_ids)


def reconstruct(
    df: pl.DataFrame, candidate_tweet_ids: set[int], brand: str, stats: ReconstructionStats
) -> list[dict]:
    tweet_id = df["tweet_id"].to_numpy()
    author_id = df["author_id"].to_list()
    inbound = df["inbound"].to_numpy()
    text = df["text"].to_list()
    created_at = df["parsed_ts"].to_list()
    parent_id_raw = df["in_response_to_tweet_id"].to_numpy()

    parent_row = build_parent_validity(tweet_id, parent_id_raw)
    n = len(tweet_id)

    # restrict everything below to the candidate universe (rows touching
    # `brand`'s components) -- this keeps the Python-level loops and dicts
    # sized in the hundreds of thousands, not the full ~2.8M-row dataset.
    candidate_arr = np.fromiter(candidate_tweet_ids, dtype=np.int64, count=len(candidate_tweet_ids))
    in_candidate = np.isin(tweet_id, candidate_arr)
    candidate_rows = np.nonzero(in_candidate)[0]
    candidate_id_to_row = {int(tweet_id[r]): r for r in candidate_rows}

    children: dict[int, list[int]] = defaultdict(list)
    missing_parent = 0
    for row in candidate_rows:
        p = parent_row[row]
        has_declared_parent = not np.isnan(parent_id_raw[row])
        if p == -1:
            if has_declared_parent:
                missing_parent += 1
            continue
        children[p].append(row)
    stats.missing_parent_count = missing_parent

    # response_tweet_id consistency check (diagnostic only, doesn't affect construction)
    resp_lists = df["response_tweet_id"].to_list() if "response_tweet_id" in df.columns else None
    if resp_lists is not None:
        inconsistencies = 0
        for row in candidate_rows:
            if not resp_lists[row]:
                continue
            claimed_children = {int(x) for x in resp_lists[row].split(",") if x}
            actual_children = {int(tweet_id[c]) for c in children.get(row, [])}
            if claimed_children and claimed_children != actual_children:
                stray = claimed_children - actual_children
                for sid in stray:
                    r = candidate_id_to_row.get(sid)
                    if r is not None and parent_row[r] != row:
                        inconsistencies += 1
        stats.response_id_inconsistencies = inconsistencies

    root_rows = [
        row for row in candidate_rows
        if inbound[row] and parent_row[row] == -1
    ]
    logger.info("Found %d candidate customer roots to reconstruct", len(root_rows))

    conversations: list[dict] = []
    pruned_branches = 0

    for root in root_rows:
        customer_author = author_id[root]
        included: list[int] = []
        frontier = [root]
        seen = {root}
        while frontier:
            cur = frontier.pop()
            included.append(cur)
            for child in children.get(cur, []):
                if child in seen:
                    continue
                a = author_id[child]
                if a == customer_author or a == brand:
                    seen.add(child)
                    frontier.append(child)
                else:
                    pruned_branches += 1

        has_agent = any(not inbound[r] for r in included)
        has_customer = any(inbound[r] for r in included)
        if not has_customer:
            continue  # defensive: cannot happen since root is always inbound

        if not has_agent and len(included) > DEGENERATE_CUSTOMER_ONLY_TURN_THRESHOLD:
            # Same customer, never any brand engagement, and an unusually long
            # self-reply chain: this is reliably someone's unrelated personal
            # thread that happened to mention the brand once, not a support
            # conversation. See DEGENERATE_CUSTOMER_ONLY_TURN_THRESHOLD.
            stats.degenerate_self_threads_dropped += 1
            continue

        included.sort(key=lambda r: (created_at[r], int(tweet_id[r])))

        messages = [
            {
                "tweet_id": str(int(tweet_id[r])),
                "speaker": "customer" if inbound[r] else "agent",
                "timestamp": created_at[r].isoformat(),
                "text": clean_text(text[r]),
            }
            for r in included
        ]

        gaps_hours = [
            (created_at[included[i + 1]] - created_at[included[i]]).total_seconds() / 3600.0
            for i in range(len(included) - 1)
        ]
        has_large_gap = any(g > 48 for g in gaps_hours)

        conv = {
            "conversation_id": f"{brand}_{int(tweet_id[root])}",
            "brand": brand,
            "root_timestamp": created_at[root].isoformat(),
            "messages": messages,
        }
        conversations.append(conv)

        stats.conversations += 1
        stats.turn_counts.append(len(messages))
        if has_agent:
            stats.customer_agent_conversations += 1
            stats.agent_responded_count += 1
        else:
            stats.customer_only_conversations += 1
        n_customer_turns = sum(1 for m in messages if m["speaker"] == "customer")
        n_agent_turns = sum(1 for m in messages if m["speaker"] == "agent")
        if n_customer_turns > 1:
            stats.multi_customer_turn_count += 1
        if n_agent_turns > 1:
            stats.multi_agent_turn_count += 1
        if has_large_gap:
            stats.large_time_gap_conversations += 1

    stats.third_party_branches_pruned = pruned_branches
    stats.brand_relevant_tweets = sum(len(c["messages"]) for c in conversations)
    return conversations


def assign_splits(conversations: list[dict], holdout_fraction: float) -> tuple[list[dict], datetime]:
    """Temporal, conversation-level split. Sorts by root timestamp; the latest
    `holdout_fraction` of conversations become `eval_pool` (candidates for the
    golden set only), everything earlier becomes `retrieval_pool` (used for
    intent discovery, retrieval corpus, prompt/threshold development).
    A conversation_id belongs to exactly one split, by construction.
    """
    ordered = sorted(conversations, key=lambda c: c["root_timestamp"])
    cutoff_idx = int(len(ordered) * (1 - holdout_fraction))
    cutoff_dt = datetime.fromisoformat(ordered[cutoff_idx]["root_timestamp"])
    for i, c in enumerate(ordered):
        c["split"] = "eval_pool" if i >= cutoff_idx else "retrieval_pool"
    return ordered, cutoff_dt


def percentile(sorted_vals: list[int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def write_report(
    conversations: list[dict], stats: ReconstructionStats, cutoff_dt: datetime,
    sample: list[dict], report_path: Path,
) -> None:
    turns = sorted(stats.turn_counts)
    n = stats.conversations or 1
    lines = [
        "# AppleSupport Conversation Reconstruction — Validation Report",
        "",
        "## Method",
        "",
        "Conversations are reconstructed with **customer-identity-aware BFS**, "
        "not raw undirected connected components (see module docstring in "
        "`src/ingestion/reconstruct_threads.py` for why raw components can "
        "silently merge two unrelated customers via a shared agent reply). "
        "A component-based pass (reusing `src.analysis.brand_analysis`) is "
        "used only to cheaply narrow the search space to AppleSupport-adjacent "
        "tweets before the precise per-customer reconstruction runs.",
        "",
        "## Data cleaning",
        "",
        f"- Duplicate `tweet_id` rows dropped (dataset-wide): {stats.duplicate_tweet_ids_dropped}",
        f"- Rows dropped for malformed/unparseable timestamp: {stats.malformed_timestamp_dropped}",
        f"- Rows dropped for empty text after whitespace-stripping: {stats.empty_text_dropped}",
        f"- Rows whose declared parent tweet is missing/absent from the dataset: {stats.missing_parent_count} "
        "(treated as conversation roots if inbound, otherwise excluded)",
        f"- `response_tweet_id` vs `in_response_to_tweet_id` inconsistencies detected: {stats.response_id_inconsistencies} "
        "(diagnostic only; `in_response_to_tweet_id` on the child row is always treated as authoritative)",
        f"- Third-party reply branches pruned (a different customer or a different brand replying "
        f"mid-thread): {stats.third_party_branches_pruned}",
        f"- Degenerate self-reply threads dropped (same customer, zero brand engagement, "
        f">{DEGENERATE_CUSTOMER_ONLY_TURN_THRESHOLD} turns — see 'Discovered during validation' below): "
        f"{stats.degenerate_self_threads_dropped}",
        "",
        "## Headline statistics",
        "",
        f"- AppleSupport-relevant tweets retained in final conversations: {stats.brand_relevant_tweets:,}",
        f"- Conversations reconstructed: {stats.conversations:,}",
        f"- Customer-only conversations (no agent reply): {stats.customer_only_conversations:,} "
        f"({stats.customer_only_conversations / n:.1%})",
        f"- Customer-agent conversations: {stats.customer_agent_conversations:,} "
        f"({stats.customer_agent_conversations / n:.1%})",
        f"- Average turns/conversation: {sum(turns) / n:.2f}",
        f"- Median turns/conversation: {percentile(turns, 0.5):.1f}",
        f"- P90 turns/conversation: {percentile(turns, 0.9):.1f}",
        f"- Max turns/conversation: {max(turns) if turns else 0}",
        f"- 'Resolved' / agent-responded rate (>=1 agent message present): {stats.agent_responded_count / n:.1%}",
        f"- Conversations with multiple customer turns: {stats.multi_customer_turn_count / n:.1%}",
        f"- Conversations with multiple agent turns: {stats.multi_agent_turn_count / n:.1%}",
        f"- Conversations with a >48h gap between consecutive messages (possible stale/resumed thread): "
        f"{stats.large_time_gap_conversations:,} ({stats.large_time_gap_conversations / n:.1%})",
        "",
        "## Train/eval split (leakage prevention)",
        "",
        f"Conversation-level, temporal split: the earliest {(1 - EVAL_HOLDOUT_FRACTION):.0%} of "
        f"conversations by root-tweet timestamp are `retrieval_pool`; the latest "
        f"{EVAL_HOLDOUT_FRACTION:.0%} are `eval_pool`. Cutoff timestamp: **{cutoff_dt.isoformat()}**. "
        "A `conversation_id` appears in exactly one split (enforced by construction — the split "
        "is assigned once, over the full deduplicated conversation list). Intent discovery (Phase D), "
        "retrieval-corpus construction, prompt design, and threshold tuning must only read "
        "`retrieval_pool`. The golden evaluation set (Phase E) is sampled only from `eval_pool`.",
        "",
        "## Manual inspection sample",
        "",
        f"{len(sample)} randomly sampled conversations were inspected for ordering sanity "
        "(chronological order, root message always from the customer, no obviously "
        "cross-wired identities). See the companion console/markdown dump for the full text.",
        "",
        "## Discovered during validation",
        "",
        "- **Self-reply-thread contamination (fixed).** The single longest raw conversation "
        "(38 turns) turned out to be one customer's months-long personal Twitter thread "
        "(mostly live-tweeting iOS 11 impressions in Arabic, June-November 2017) that happened "
        "to mention `@AppleSupport` once, near the end, with zero replies from the brand. "
        "The identity-aware BFS correctly refused to merge this with any *other* customer's "
        "conversation, but a single customer's own rambling self-reply chain still passed the "
        "same-identity check. Fixed by dropping customer-only conversations longer than "
        f"{DEGENERATE_CUSTOMER_ONLY_TURN_THRESHOLD} turns (the threshold is the observed p99.9 "
        "of customer-only conversation length before it jumps straight to 38 -- see the "
        "distribution check in the phase notes). This is a narrow, data-driven fix for a rare "
        "pattern (2 conversations out of 80,507 pre-fix), not a general filter.",
        "- **Residual non-English trickle.** A small number of customer messages are in "
        "German/Portuguese; AppleSupport's standard reply redirects them to a "
        "language-appropriate support channel rather than answering in English. This is far "
        "smaller in scale than AmazonHelp's multilingual volume (see reports/decision_log.md) "
        "but is not exactly zero -- worth a passing note for intent design, not a reason to "
        "revisit the brand choice.",
        "- **At least one self-harm-adjacent message observed** in the raw data (a customer's "
        "opening message referencing self-harm before asking for device help). This did not "
        "change reconstruction logic, but it is a concrete reminder that the escalation policy "
        "(Phase H) and any safety layer must treat sensitive/self-harm language as an automatic "
        "escalation trigger, not something the intent classifier or RAG system should attempt "
        "to auto-handle.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=RAW_CSV)
    parser.add_argument("--brand", default=BRAND)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"{args.csv} not found. See data/README.md.")

    stats = ReconstructionStats()

    candidate_ids = find_candidate_conversations(args.csv, args.brand)

    df = load_raw_rows(args.csv)
    resp_col = load_columns(args.csv, ["tweet_id", "response_tweet_id"])
    df = df.join(resp_col, on="tweet_id", how="left")
    df = dedupe_and_clean(df, stats)

    conversations = reconstruct(df, candidate_ids, args.brand, stats)
    conversations, cutoff_dt = assign_splits(conversations, EVAL_HOLDOUT_FRACTION)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    logger.info("Wrote %d conversations to %s", len(conversations), OUT_JSONL)

    rng = random.Random(args.seed)
    sample = rng.sample(conversations, min(args.sample_size, len(conversations)))

    write_report(conversations, stats, cutoff_dt, sample, OUT_REPORT)

    sample_path = OUT_REPORT.parent / "conversation_samples.md"
    with sample_path.open("w", encoding="utf-8") as f:
        f.write("# Sampled Reconstructed Conversations (manual inspection)\n\n")
        for conv in sample:
            f.write(f"## {conv['conversation_id']} (split={conv['split']}, turns={len(conv['messages'])})\n\n")
            for m in conv["messages"]:
                f.write(f"- **{m['speaker']}** [{m['timestamp']}]: {m['text']}\n")
            f.write("\n")
    logger.info("Wrote %s", sample_path)


if __name__ == "__main__":
    main()
