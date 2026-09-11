"""Rule-based intent coverage estimate over the FULL retrieval_pool.

This is deliberately NOT a trained classifier -- CLAUDE.md explicitly says
"Do not train the final LLM classifier yet" during Phase D. It is a
transparent, auditable keyword tagger built from the `keywords` field in
`config/intents.yaml` (which was itself derived from real cluster top-terms,
see reports/intent_clusters_raw.md), used only to get real, computed
frequency/coverage/ambiguity numbers for reports/intent_discovery.md instead
of guessing at them from the 12k-sample cluster proportions.

Assignment logic per message:
  1. Check every "specific" intent's keywords (all intents except the two
     residual/fallback ones). A message can match zero, one, or several.
  2. If it matched >=1 specific intent, it's tagged with all of them
     (multi-match = one source of the "ambiguous/multi-intent" stat).
  3. If it matched zero specific intents, fall back to
     `ios_update_general_complaint` if it mentions an update/iOS-version
     term, else `vague_frustration_needs_clarification`.

Leakage: reads only `retrieval_pool` conversations, same as intent_discovery.py.

Run:
    python -m src.analysis.intent_coverage
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONVERSATIONS_PATH = Path("data/processed/applesupport_conversations.jsonl")
INTENTS_PATH = Path("config/intents.yaml")
CLUSTERS_PATH = Path("reports/intent_clusters_raw.md")
OUT_PATH = Path("reports/intent_discovery.md")

FALLBACK_UPDATE_KEYWORDS = ["update", "ios 11", "ios11", "upgrad", "new software", "latest update"]
GENERIC_INTENT = "ios_update_general_complaint"
RESIDUAL_INTENT = "vague_frustration_needs_clarification"

# Manual mapping from the 25 raw KMeans clusters (reports/intent_clusters_raw.md)
# to the curated taxonomy, decided by reading each cluster's example messages
# (see module docstring in src/analysis/intent_discovery.py). Two clusters were
# semantically mixed and are split by an approximate weight estimated from
# their 10 sampled examples each -- documented here, not hidden. Cluster 15 is
# non-English content and is out of scope for the intent taxonomy (routed to
# a language-appropriate channel by AppleSupport, not a "problem type").
CLUSTER_TO_INTENT: dict[int, dict[str, float]] = {
    5: {"ios_update_general_complaint": 1.0},
    23: {"vague_frustration_needs_clarification": 1.0},
    10: {"ios_update_general_complaint": 1.0},
    17: {"ios_update_general_complaint": 0.6, "order_purchase_retail_support": 0.3, "connectivity_network_messaging_issues": 0.1},
    21: {"connectivity_network_messaging_issues": 1.0},
    12: {"ios_update_general_complaint": 1.0},
    7: {"battery_life_drain": 1.0},
    20: {"battery_life_drain": 1.0},
    2: {"autocorrect_keyboard_bug": 1.0},
    24: {"media_music_playback_issues": 1.0},
    18: {"account_apple_id_security": 1.0},
    16: {"autocorrect_keyboard_bug": 1.0},
    13: {"autocorrect_keyboard_bug": 1.0},
    22: {"mac_macos_issues": 1.0},
    8: {"ios_update_general_complaint": 1.0},
    4: {"order_purchase_retail_support": 1.0},
    3: {"autocorrect_keyboard_bug": 0.7, "vague_frustration_needs_clarification": 0.3},
    14: {"device_freezing_performance": 1.0},
    19: {"photos_storage_icloud_management": 1.0},
    11: {"audio_accessory_issues": 1.0},
    15: {"__non_english_out_of_scope__": 1.0},
    9: {"autocorrect_keyboard_bug": 1.0},
    6: {"app_store_purchase_issues": 1.0},
    1: {"autocorrect_keyboard_bug": 1.0},
    0: {"apple_watch_issues": 1.0},
}


def cluster_based_estimate() -> tuple[dict[str, float], int]:
    """Aggregate the real per-cluster sample sizes (parsed from the cluster
    dump) through CLUSTER_TO_INTENT. This is the semantically-grounded
    estimate (embeddings + human-inspected cluster labels), as opposed to the
    cheap substring-keyword tagger below -- see 'Discovered during
    validation' in the written report for why the two disagree.
    """
    text = CLUSTERS_PATH.read_text(encoding="utf-8")
    sizes = {int(cid): int(n) for cid, n in re.findall(r"## Cluster (\d+) \(n=(\d+)", text)}
    totals: dict[str, float] = defaultdict(float)
    for cluster_id, size in sizes.items():
        for intent_name, weight in CLUSTER_TO_INTENT[cluster_id].items():
            totals[intent_name] += size * weight
    return dict(totals), sum(sizes.values())


def load_intents() -> list[dict]:
    with INTENTS_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["intents"]


def load_retrieval_pool_roots() -> list[str]:
    roots = []
    with CONVERSATIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            conv = json.loads(line)
            if conv["split"] == "retrieval_pool":
                roots.append(conv["messages"][0]["text"])
    return roots


def tag_message(text_lower: str, specific_intents: list[dict]) -> list[str]:
    matches = [
        intent["name"] for intent in specific_intents
        if any(kw in text_lower for kw in intent["keywords"])
    ]
    if matches:
        return matches
    if any(kw in text_lower for kw in FALLBACK_UPDATE_KEYWORDS):
        return [GENERIC_INTENT]
    return [RESIDUAL_INTENT]


def main() -> None:
    intents = load_intents()
    by_name = {i["name"]: i for i in intents}
    specific_intents = [i for i in intents if i["name"] not in (GENERIC_INTENT, RESIDUAL_INTENT)]

    roots = load_retrieval_pool_roots()
    logger.info("Tagging %d retrieval_pool root messages against %d intents", len(roots), len(intents))

    primary_counts: Counter = Counter()
    all_match_counts: Counter = Counter()
    multi_match_count = 0
    pair_cooccurrence: Counter = Counter()
    examples_by_intent: dict[str, list[str]] = defaultdict(list)

    for text in roots:
        tags = tag_message(text.lower(), specific_intents)
        for t in tags:
            all_match_counts[t] += 1
            if len(examples_by_intent[t]) < 5:
                examples_by_intent[t].append(text)
        primary_counts[tags[0]] += 1
        if len(tags) > 1:
            multi_match_count += 1
            for a, b in combinations(sorted(tags), 2):
                pair_cooccurrence[(a, b)] += 1

    total = len(roots)
    n_intents = len(intents)
    counts_sorted = sorted(all_match_counts.items(), key=lambda kv: kv[1], reverse=True)

    cluster_totals, cluster_n = cluster_based_estimate()
    non_english = cluster_totals.pop("__non_english_out_of_scope__", 0.0)
    cluster_sorted = sorted(cluster_totals.items(), key=lambda kv: kv[1], reverse=True)
    c_max = cluster_sorted[0][1] if cluster_sorted else 0
    c_min = min((c for _, c in cluster_sorted), default=0)

    lines = [
        "# Intent Discovery — AppleSupport",
        "",
        "## Method",
        "",
        "Taxonomy derived from KMeans clustering (k=25) over sentence embeddings "
        "(`all-MiniLM-L6-v2`) of 12,000 sampled `retrieval_pool` root customer messages, "
        "manually inspected and curated into an operationally meaningful set of intents "
        "(`config/intents.yaml`). See `reports/intent_clusters_raw.md` for the raw cluster "
        "dump this was built from.",
        "",
        "Two different frequency estimates are reported below and they disagree "
        "substantially -- see 'Discovered during validation' for why, and which one to trust.",
        "",
        "## Primary frequency estimate (cluster-based, semantically grounded)",
        "",
        f"Aggregated from the real per-cluster sizes in the {cluster_n:,}-message KMeans sample, "
        "via the manual cluster-to-intent mapping in `src/analysis/intent_coverage.py` "
        "(`CLUSTER_TO_INTENT`). This is the number to trust for 'how common is this intent'.",
        "",
        f"- Non-English messages (out of taxonomy scope): {non_english:.0f} ({non_english / cluster_n:.1%})",
        f"- Class imbalance (English-scope intents): largest is {c_max / max(c_min, 1):.1f}x the smallest",
        "",
        "| Intent | Est. count (of 12,000) | % of corpus | Escalation default |",
        "|---|---|---|---|",
    ]
    for name, count in cluster_sorted:
        esc = by_name[name]["escalation_guidance"].strip().split(".")[0]
        esc_short = "Escalate" if esc.lower().startswith("escalate") else "Auto-handle (conditional)"
        lines.append(f"| {name} | {count:.0f} | {count / cluster_n:.1%} | {esc_short} |")

    lines += [
        "",
        "## Secondary cross-check: full-corpus keyword tagger (recall-limited)",
        "",
        "Same taxonomy, tagged over the **full retrieval_pool** (68,427 root messages) using "
        "each intent's `keywords` list -- a cheap, fully-reproducible, deterministic tagger, "
        "but with much lower recall than semantic clustering (see below). Useful for the "
        "*multi-intent/ambiguity* signal (keyword co-occurrence) and as a full-corpus sanity "
        "check, **not** for headline frequency (use the cluster-based table above for that).",
        "",
        f"- Messages analyzed: {total:,}",
        f"- Messages matching >1 specific intent by keyword (ambiguity signal): "
        f"{multi_match_count:,} ({multi_match_count / total:.1%})",
        "",
        "| Intent | Keyword-tagged | % of corpus |",
        "|---|---|---|",
    ]
    for name, count in counts_sorted:
        primary = primary_counts.get(name, 0)
        lines.append(f"| {name} | {primary:,} | {primary / total:.1%} |")

    lines += ["", "## Why each intent exists / how it differs from neighbors", ""]
    for intent in intents:
        lines.append(f"### {intent['name']}")
        lines.append("")
        lines.append(intent["description"].strip())
        lines.append("")
        lines.append(f"**Confusable with:** {', '.join(intent['confusable_intents']) or 'none identified'}")
        lines.append("")
        lines.append("**Real examples from the data:**")
        for ex in examples_by_intent.get(intent["name"], intent["examples"])[:5]:
            lines.append(f"- {ex[:200]}")
        lines.append("")

    lines += ["## Confusion-risk matrix (data-driven: co-occurring keyword matches)", ""]
    lines.append(
        "Pairs below are messages that matched *both* intents' keyword sets "
        "simultaneously -- a concrete, measured signal for which intents a "
        "classifier is most likely to confuse, complementing the qualitative "
        "`confusable_intents` judgment calls above."
    )
    lines.append("")
    lines.append("```text")
    for (a, b), n in pair_cooccurrence.most_common(15):
        lines.append(f"{a} <-> {b}   ({n} co-occurring messages)")
    lines.append("```")
    lines.append("")

    lines += [
        "## Discovered during validation",
        "",
        "- **The two frequency tables disagree sharply, and the keyword tagger is the wrong "
        f"one to trust for headline numbers.** Cluster-based: "
        f"`vague_frustration_needs_clarification` ≈ "
        f"{cluster_totals.get(RESIDUAL_INTENT, 0) / cluster_n:.1%}. Keyword-tagged: "
        f"{primary_counts.get(RESIDUAL_INTENT, 0) / total:.1%}. Manually inspecting a random "
        "sample of messages the keyword tagger dumped into the residual bucket showed most "
        "are clearly specific intents the keyword list simply didn't have the exact phrasing "
        "for -- e.g. *\"FIX THIS I? SHIT NOOOOOW\"* (autocorrect_keyboard_bug, no literal "
        "\"question mark\"/\"letter i\"), *\"my phone just died while it was charging\"* "
        "(battery_life_drain, no literal \"battery\"), *\"crack in the screen... need to get "
        "it repaired\"* (order_purchase_retail_support, no literal \"warranty\"). Substring "
        "keyword matching has poor recall against the enormous phrasing variety in real "
        "customer text (emoji-mangled words, sarcasm, implied symptoms); semantic embedding "
        "clustering does not have this problem, which is exactly why it -- not a keyword "
        "list -- was used to *design* the taxonomy. The keyword tagger remains useful only "
        "for the full-corpus multi-intent/ambiguity cross-check above, and is intentionally "
        "kept simple rather than tuned to chase recall, since building an accurate classifier "
        "is explicitly out of scope for this phase.",
        "- `ios_update_general_complaint` and `vague_frustration_needs_clarification` are "
        "both residual/fallback buckets by construction and will always look 'confusable' "
        "with everything in the keyword cross-check -- that's expected, not a taxonomy flaw.",
        "- Keyword matching (and, to a lesser extent, single-vector clustering) cannot "
        "distinguish sarcasm, quoted/forwarded text, or a customer listing multiple unrelated "
        "symptoms in one message (a real and common pattern -- see `AppleSupport_300594` in "
        "`reports/conversation_samples.md`, which lists five distinct symptoms across one "
        "burst of messages).",
        f"- Non-English messages (~{non_english / cluster_n:.1%} of the sample) are "
        "out of taxonomy scope by design; AppleSupport redirects them to a "
        "language-appropriate channel rather than answering the substance, so they aren't a "
        "'problem type' the intent taxonomy should model (see reports/conversation_reconstruction.md).",
        "- **Root message is sometimes not the real request.** This analysis uses each "
        "conversation's opening (root) message as the unit of analysis. A small number of "
        "roots are casual, unrelated banter (e.g. \"Sorry for party rockin\", pulled up as an "
        "example under vague_frustration_needs_clarification above) where the actual support "
        "issue only appears in a later turn of the same reply chain, once the identity-aware "
        "BFS reached a message that engaged AppleSupport. This doesn't affect conversation "
        "reconstruction correctness (Phase C keeps the full message list), but it means root-only "
        "intent analysis slightly overstates vague/off-topic content and undercounts the true "
        "intent. The golden set (below) should label intent from the customer's actual "
        "request, using full conversation context where the root is not informative -- not "
        "mechanically from the root message alone.",
        "",
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", OUT_PATH)


if __name__ == "__main__":
    main()
