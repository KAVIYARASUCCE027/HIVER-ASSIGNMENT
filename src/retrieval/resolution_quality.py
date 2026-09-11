"""Heuristic resolution-quality scoring for historical AppleSupport responses.

Distinguishes substantive troubleshooting responses from generic "please DM
us" boilerplate, per Phase H's requirement not to treat a canned handoff as
useful resolution evidence when better historical responses exist. This is a
transparent, documented heuristic (not tuned against golden or any dev set,
per CLAUDE.md's "no unnecessary abstraction" / "simple system" guidance) --
its purpose is corpus *ranking*, not a claim of ground-truth resolution.

Nothing is deleted from the corpus based on this score -- low-scoring
conversations stay indexed and retrievable, just ranked lower (see
src/retrieval/rerank.py).
"""

from __future__ import annotations

import re

# Same canned-handoff patterns identified in Phase D (reports/intent_discovery.md
# / src/analysis/intent_coverage.py context) -- a reply consisting of little
# more than these phrases carries almost no troubleshooting content.
CANNED_HANDOFF_RE = re.compile(
    r"\bsend (?:us )?a (?:direct message|dm|private message)\b"
    r"|\bclick ['‘]?message['’]?\b"
    r"|\bfollow (?:us )?(?:so (?:that|we)?)?\b.*\bdm\b"
    r"|\bdm us\b"
    r"|\breach out (?:to us )?(?:here|via dm)\b",
    re.IGNORECASE,
)

# Generic goodwill filler that carries no actionable content on its own.
FILLER_RE = re.compile(
    r"\bwe(?:'d| would) (?:like|love) to help\b"
    r"|\bwe want to help\b"
    r"|\bthanks for (?:reaching out|letting us know)\b"
    r"|\bwe(?:'re| are) here (?:for you|to help)\b"
    r"|\bsorry to hear (?:that|this)\b"
    r"|\bwe understand your (?:concern|frustration)s?\b",
    re.IGNORECASE,
)

# Verbs/nouns that signal an actual troubleshooting step or factual answer,
# as opposed to an invitation to continue elsewhere.
ACTIONABLE_RE = re.compile(
    r"\b(restart\w*|reset\w*|updat\w*|reinstall\w*|force[- ]?quit\w*|check\w*|try\w*|"
    r"go to|settings?|backup\w*|sign(?:ing|ed)? (?:in|out)|turn(?:ing|ed)? (?:on|off)|"
    r"enabl\w*|disabl\w*|install\w*|tap\w*|swip\w*|connect\w*|charg\w*|reboot\w*|"
    r"recovery mode|dfu|delet\w*|remov\w*|toggl\w*|open\w*|clos\w*|"
    r"battery health|storage|wifi|bluetooth|airplane mode|version\w*)\b",
    re.IGNORECASE,
)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text))


def is_substantive(agent_response: str) -> bool:
    """True if the response contains real troubleshooting/factual content
    beyond a canned DM-handoff or pure goodwill filler."""
    residual = CANNED_HANDOFF_RE.sub(" ", agent_response)
    residual = FILLER_RE.sub(" ", residual)
    return word_count(residual) >= 6 and bool(ACTIONABLE_RE.search(residual))


def resolution_quality(agent_response: str, agent_had_last_word: bool) -> float:
    """Score in [0, 1]. Documented, fixed weights (not tuned against any
    evaluation set):
      0.5 -- is_substantive (contains real troubleshooting content)
      0.3 -- agent_had_last_word (customer did not need to follow up again --
             the same "resolved-ish" heuristic used in Phase C/D, an
             imperfect proxy for resolution, not ground truth)
      0.2 -- length-scaled: response is long enough to plausibly contain
             detail (saturates at 40 words; longer responses don't score
             extra beyond that)
    """
    substantive_score = 0.5 if is_substantive(agent_response) else 0.0
    last_word_score = 0.3 if agent_had_last_word else 0.0
    length_score = 0.2 * min(word_count(agent_response) / 40.0, 1.0)
    return round(substantive_score + last_word_score + length_score, 4)
