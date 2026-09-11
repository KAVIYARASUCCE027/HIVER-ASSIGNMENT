"""Deterministic AUTO_HANDLE / ESCALATE policy (Phase I, promoted to
Candidate C from the Phase I.1 experiment -- see decision log #20 and
reports/escalation_policy_improvement.md for the full comparison that
justified this promotion).

Every signal below is either read directly from an already-frozen phase's
output or is a minimal, auditable encoding of guidance already written in
config/intents.yaml / data/golden/LABELING_GUIDE.md -- nothing here
re-runs classification or retrieval, and no threshold was chosen by fitting
against the golden `expected_action` labels this policy is evaluated
against. See config/escalation.yaml for the full rationale of each
threshold, including signals that were tested and rejected for lack of
data support.

Confidence alone is deliberately NOT the sole criterion (Phase G showed
only r=0.424 confidence/correctness correlation, with 29 high-confidence
wrong predictions) -- it is one of six independent signals, any of which
can trigger escalation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

AUTO_HANDLE = "AUTO_HANDLE"
ESCALATE = "ESCALATE"

# Promoted from Phase I's original 0.80 to 0.75 (Phase I.1 Candidate C):
# the confidence-threshold sweep in reports/escalation_policy_improvement.md
# Section 7 showed 0.60-0.75 are identical on the golden benchmark and each
# weakly dominates 0.80 (same false-auto-handle, fewer unnecessary
# escalations). Still grounded in Phase G's own confidence-bucket accuracy
# data (config/escalation.yaml "intent_uncertainty"), not reselected to
# fit the escalation labels.
CONFIDENCE_THRESHOLD = 0.75

# The only two intents whose config/intents.yaml escalation_guidance says
# "Escalate by default" (account access / financial-logistics commitment).
HIGH_RISK_INTENTS = {"account_apple_id_security", "order_purchase_retail_support"}

# Directly implements the safety carve-out already written in
# config/intents.yaml (battery_life_drain) and
# data/golden/LABELING_GUIDE.md. Conservative, keyword-gated -- same style
# as src/retrieval/sanitize.py, not a learned classifier.
HIGH_RISK_KEYWORDS_RE = re.compile(
    r"\b(swell|swollen|smoke|smoking|burn|burnt|burning|fire|explod|shock|"
    r"electrocut|spark|overheat|too hot)\b",
    re.IGNORECASE,
)

# --- Added by the Phase I.1 promotion (Candidate C) -----------------------
# Each directly implements a per-intent conditional escalation clause
# already written in config/intents.yaml that the original Phase I policy
# did not encode (a gap flagged in the original reports/escalation_results.md
# Section 12). Tested independently for both catch-rate and false-positive
# cost in reports/escalation_policy_improvement.md before promotion; a
# fourth candidate ("repeat_attempt_language") was tested and rejected
# there for having zero catch rate.
HARDWARE_DAMAGE_KEYWORDS_RE = re.compile(
    r"\b(broken|busted|defective|cracked|shattered|delaminat|won.t (turn on|charge|power)|"
    r"stopped working|physically damaged)\b",
    re.IGNORECASE,
)  # implements: audio_accessory_issues / apple_watch_issues / mac_macos_issues /
   # order_purchase_retail_support "repeated hardware failure" / "physical defect" clauses

BILLING_DISPUTE_KEYWORDS_RE = re.compile(
    r"\b(charged (me |twice|again)?|duplicate charge|double charged|billing (dispute|issue|problem)|"
    r"refund|declin(ed|ing)|still charg)\b",
    re.IGNORECASE,
)  # implements: app_store_purchase_issues / media_music_playback_issues "billing dispute" clauses

DATA_LOSS_KEYWORDS_RE = re.compile(
    r"\b(deleted|erased|disappeared|missing|lost|wiped out)\W+(\w+\W+){0,4}?"
    r"(photo|contact|file|data|message|note)s?\b",
    re.IGNORECASE,
)  # implements: photos_storage_icloud_management "unexpectedly deleted" clause


@dataclass
class EscalationInput:
    """Everything the policy needs, all sourced from already-frozen phases."""

    example_id: str
    customer_message: str
    predicted_intent: str
    confidence: float
    has_retrieved_evidence: bool  # Phase H: retrieved_evidence non-empty
    should_answer: bool  # Phase H: generation-level should_answer


@dataclass
class EscalationDecision:
    action: str
    reasons: list[str] = field(default_factory=list)


def decide(inp: EscalationInput) -> EscalationDecision:
    reasons: list[str] = []

    if inp.predicted_intent in HIGH_RISK_INTENTS:
        reasons.append("high_risk_request")
    if HIGH_RISK_KEYWORDS_RE.search(inp.customer_message):
        reasons.append("high_risk_request")
    if HARDWARE_DAMAGE_KEYWORDS_RE.search(inp.customer_message):
        reasons.append("high_risk_request")
    if BILLING_DISPUTE_KEYWORDS_RE.search(inp.customer_message):
        reasons.append("high_risk_request")
    if DATA_LOSS_KEYWORDS_RE.search(inp.customer_message):
        reasons.append("high_risk_request")
    if not inp.has_retrieved_evidence:
        reasons.append("insufficient_historical_evidence")
    elif not inp.should_answer:
        # Evidence existed but the generator itself judged it insufficient
        # to actually answer this specific question (Phase H, rag_results.md
        # Section 11 category 2) -- distinct from no evidence at all.
        reasons.append("generation_could_not_be_grounded")
    if inp.confidence < CONFIDENCE_THRESHOLD:
        reasons.append("intent_uncertainty")

    # de-duplicate while preserving order (high_risk_request can be added twice)
    seen = set()
    deduped = [r for r in reasons if not (r in seen or seen.add(r))]

    action = ESCALATE if deduped else AUTO_HANDLE
    return EscalationDecision(action=action, reasons=deduped)
