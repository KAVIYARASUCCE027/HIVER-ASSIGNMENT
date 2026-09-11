"""Stratified sampling of golden-set CANDIDATES from `eval_pool` only.

Revised per explicit user feedback after the first version (which used a flat
10-per-intent floor across all 200 slots) looked "artificially balanced" --
the real AppleSupport intent distribution is ~19x imbalanced between the
largest and smallest intent, and a golden set that hides that isn't
trustworthy. The design now combines three deliberately different strata:

  1. natural_distribution (~125): plain, unstratified random draw from
     eval_pool. This is the simplest and most honest way to "reflect the real
     distribution" -- a uniform random sample preserves whatever the true
     class balance is by construction, without needing an intent estimate at
     all (and therefore without inheriting any classifier's bias).
  2. rare_intent_coverage (~45): the ~6 lowest-frequency specific intents
     (per reports/intent_discovery.md's cluster-based estimate) get a
     meaningful floor so per-intent metrics aren't computed on a handful of
     accidental hits. Needs a real intent signal to find true positives for
     rare classes -- uses the frozen embedding classifier (see
     src/evaluation/embedding_intent_classifier.py), not the weaker keyword
     tagger, since Phase D showed the keyword tagger's recall is
     particularly bad for exactly these lower-frequency intents.
  3. difficult_cases (~30): explicit hard patterns spread across 8
     sub-categories (ambiguous, multi-intent, context-dependent, very short,
     high-frustration, unanswered, clear-escalate, clear-auto-handle).

This script selects *which* conversations/messages a human should label. It
does NOT assign intent, expected_action, expected_reason, or label_notes --
those require real human judgment; CLAUDE.md is explicit that fabricated
labels are unacceptable. Output: data/golden/golden_candidates.jsonl, with
those four fields null and a `suggested_intent` (non-authoritative) for a
human labeler to start from.

LEAKAGE: reads only `eval_pool` conversations for selection. The embedding
classifier's centroids are built entirely from retrieval_pool data (frozen
before this script ever looks at eval_pool) -- see
embedding_intent_classifier.py's docstring for why classifying eval_pool
messages against them is inference, not tuning.

Run:
    python -m src.evaluation.sample_golden
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml

from src.evaluation.embedding_intent_classifier import build_intent_centroids, classify_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONVERSATIONS_PATH = Path("data/processed/applesupport_conversations.jsonl")
INTENTS_PATH = Path("config/intents.yaml")
OUT_PATH = Path("data/golden/golden_candidates.jsonl")

TARGET_TOTAL = 200
N_NATURAL = 125
N_RARE = 45
N_DIFFICULT = 30  # 125+45+30 = 200

# Bottom 6 specific intents by the cluster-based frequency estimate in
# reports/intent_discovery.md (excludes the two residual/catch-all intents,
# which are the *largest*, not rare).
RARE_INTENTS = [
    "apple_watch_issues", "app_store_purchase_issues", "audio_accessory_issues",
    "photos_storage_icloud_management", "device_freezing_performance", "mac_macos_issues",
]

RESIDUAL_INTENT = "vague_frustration_needs_clarification"
AMBIGUITY_MARGIN_THRESHOLD = 0.05  # small top1-top2 cosine gap = genuinely ambiguous

FRUSTRATION_MARKERS = re.compile(
    r"\b(fuck|shit|wtf|damn|ridiculous|pathetic|worst|hate|angry|furious|"
    r"disgusted|awful|terrible|garbage|trash|useless)\b|!{2,}|[A-Z]{4,}",
    re.IGNORECASE,
)


def load_intents() -> list[dict]:
    with INTENTS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["intents"]


def load_eval_pool() -> list[dict]:
    convs = []
    with CONVERSATIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["split"] == "eval_pool":
                convs.append(c)
    logger.info("Loaded %d eval_pool conversations", len(convs))
    return convs


def turn_bucket(n: int) -> str:
    if n <= 1:
        return "short"
    if n <= 4:
        return "medium"
    return "long"


def frustration_score(text: str) -> int:
    return len(FRUSTRATION_MARKERS.findall(text))


def build_candidate(conv: dict, target_idx: int, stratum: str, pred: dict) -> dict:
    messages = conv["messages"]
    target = messages[target_idx]
    return {
        "conversation_id": conv["conversation_id"],
        "conversation": messages[:target_idx],
        "customer_message": target["text"],
        "target_tweet_id": target["tweet_id"],
        "intent": None,
        "expected_action": None,
        "expected_reason": None,
        "label_notes": None,
        "suggested_intent": pred["predicted_intent"],
        "suggested_intent_confidence": round(pred["confidence"], 3),
        "suggested_intent_margin": round(pred["margin"], 3),
        "suggested_intent_second_choice": pred["second_intent"],
        "sampling_stratum": stratum,
        "turn_count": len(messages),
        "has_agent_reply": any(m["speaker"] == "agent" for m in messages),
    }


def main() -> None:
    rng = random.Random(42)

    intents = load_intents()
    escalate_intents = {
        i["name"] for i in intents
        if i["escalation_guidance"].strip().lower().startswith("escalate")
    }

    convs = load_eval_pool()
    root_texts = [c["messages"][0]["text"] for c in convs]

    logger.info("Building frozen intent centroids from retrieval_pool sample ...")
    intent_names, centroids = build_intent_centroids()

    logger.info("Classifying %d eval_pool root messages against frozen centroids ...", len(root_texts))
    predictions = classify_texts(root_texts, intent_names, centroids)

    annotated = []
    for conv, text, pred in zip(convs, root_texts, predictions):
        customer_idxs = [i for i, m in enumerate(conv["messages"]) if m["speaker"] == "customer"]
        is_multi_intent = pred["margin"] < AMBIGUITY_MARGIN_THRESHOLD and pred["second_intent"] != RESIDUAL_INTENT
        annotated.append({
            "conv": conv,
            "pred": pred,
            "is_ambiguous": pred["margin"] < AMBIGUITY_MARGIN_THRESHOLD or pred["predicted_intent"] == RESIDUAL_INTENT,
            "is_multi_intent": is_multi_intent,
            "turn_bucket": turn_bucket(len(conv["messages"])),
            "frustration": frustration_score(text),
            "has_agent": any(m["speaker"] == "agent" for m in conv["messages"]),
            "has_multi_customer_turns": len(customer_idxs) > 1,
            "is_escalate_default": pred["predicted_intent"] in escalate_intents,
        })

    used_ids: set[str] = set()
    candidates: list[dict] = []

    def take(pool: list[dict], n: int, stratum: str, context_dependent: bool = False) -> int:
        available = [a for a in pool if a["conv"]["conversation_id"] not in used_ids]
        rng.shuffle(available)
        picked = available[:n]
        for a in picked:
            used_ids.add(a["conv"]["conversation_id"])
            conv = a["conv"]
            if context_dependent and a["has_multi_customer_turns"]:
                customer_idxs = [i for i, m in enumerate(conv["messages"]) if m["speaker"] == "customer"]
                target_idx = customer_idxs[-1]
            else:
                target_idx = 0
            candidates.append(build_candidate(conv, target_idx, stratum, a["pred"]))
        if len(picked) < n:
            logger.warning("Stratum '%s' wanted %d, only found %d unused candidates", stratum, n, len(picked))
        return len(picked)

    # --- Stratum 1: natural distribution -----------------------------------
    # Plain random draw, taken FIRST from the untouched full pool so it is
    # not distorted by what later strata need. This is what "reflects the
    # real distribution, imbalance included" means operationally.
    take(annotated, N_NATURAL, "natural_distribution")

    # --- Stratum 2: rare-intent coverage ------------------------------------
    by_intent = defaultdict(list)
    for a in annotated:
        by_intent[a["pred"]["predicted_intent"]].append(a)
    per_rare_quota = N_RARE // len(RARE_INTENTS)
    got = 0
    for name in RARE_INTENTS:
        got += take(by_intent.get(name, []), per_rare_quota, f"rare_intent:{name}")
    leftover = N_RARE - got
    if leftover > 0:
        # spread any shortfall across whichever rare intents have more available
        for name in RARE_INTENTS:
            if leftover <= 0:
                break
            leftover -= take(by_intent.get(name, []), leftover, f"rare_intent:{name}")

    # --- Stratum 3: difficult / edge cases (8 sub-categories, ~4 each) ------
    n_each = N_DIFFICULT // 8

    ambiguous_pool = [a for a in annotated if a["is_ambiguous"]]
    take(ambiguous_pool, n_each, "difficult:ambiguous")

    multi_intent_pool = [a for a in annotated if a["is_multi_intent"]]
    take(multi_intent_pool, n_each, "difficult:multi_intent")

    context_candidates = [a for a in annotated if a["has_multi_customer_turns"]]
    take(context_candidates, n_each, "difficult:context_dependent", context_dependent=True)

    short_pool = [a for a in annotated if a["turn_bucket"] == "short"]
    take(short_pool, n_each, "difficult:very_short")

    frustrated_pool = sorted(annotated, key=lambda a: -a["frustration"])[:400]
    take(frustrated_pool, n_each, "difficult:high_frustration")

    unanswered_pool = [a for a in annotated if not a["has_agent"]]
    take(unanswered_pool, n_each, "difficult:unanswered")

    clear_escalate_pool = [
        a for a in annotated
        if a["is_escalate_default"] and a["pred"]["margin"] >= AMBIGUITY_MARGIN_THRESHOLD
    ]
    take(clear_escalate_pool, N_DIFFICULT - n_each * 7, "difficult:clear_escalate")

    clear_auto_pool = [
        a for a in annotated
        if not a["is_escalate_default"]
        and a["pred"]["predicted_intent"] != RESIDUAL_INTENT
        and a["pred"]["margin"] >= AMBIGUITY_MARGIN_THRESHOLD
        and a["frustration"] == 0
        and a["has_agent"]
    ]
    take(clear_auto_pool, N_DIFFICULT - n_each * 7, "difficult:clear_auto_handle")

    # --- Fill to exactly TARGET_TOTAL if any stratum came up short ---------
    remaining = TARGET_TOTAL - len(candidates)
    if remaining > 0:
        rest_pool = [a for a in annotated if a["conv"]["conversation_id"] not in used_ids]
        take(rest_pool, remaining, "fill")

    rng.shuffle(candidates)
    candidates = candidates[:TARGET_TOTAL]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for i, c in enumerate(candidates):
            c["id"] = f"golden_{i:04d}"
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    stratum_counts = defaultdict(int)
    for c in candidates:
        stratum_counts[c["sampling_stratum"]] += 1
    logger.info("Wrote %d candidates to %s", len(candidates), OUT_PATH)
    for s, n in sorted(stratum_counts.items(), key=lambda kv: -kv[1]):
        logger.info("  %-40s %d", s, n)


if __name__ == "__main__":
    main()
