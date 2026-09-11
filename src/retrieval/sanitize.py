"""PII sanitization for retrieval corpus text.

Conservative by design: masks unambiguous PII patterns only. Does NOT attempt
named-entity stripping of ordinary names, product terms, or technical detail
(CLAUDE.md is explicit that customer-support language is noisy and that noise
is often useful -- aggressive rewriting would destroy exactly the detail that
makes retrieved evidence useful).

Email and phone numbers are already masked once, at conversation-reconstruction
time (`src.ingestion.reconstruct_threads.clean_text`) -- this module re-applies
those same two patterns defensively (idempotent: masking an already-masked
`[EMAIL]`/`[PHONE]` token is a no-op) and adds patterns specific to the
retrieval-corpus context: case/order/reference numbers and long numeric
identifiers that were not masked at reconstruction time because reconstruction
had no reason to distinguish "a case number" from "an iOS version number".
"""

from __future__ import annotations

import re

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\-\s()]{8,14}\d)(?!\d)")

# "Case# 12345", "Order number: 12345", "reference ABC1234", etc. Requires an
# explicit identifying keyword immediately before the number -- this is what
# keeps it from accidentally masking iOS versions, model numbers, etc.
CASE_OR_ORDER_RE = re.compile(
    r"\b(case|order|reference|confirmation|ticket)\s*(?:number|no\.?|#)?\s*:?\s*([A-Za-z0-9-]{4,})\b",
    re.IGNORECASE,
)

# Long digit runs (13-19 digits) with optional spaces/dashes every 4 -- shaped
# like a payment card number. Rare in this dataset but worth catching.
CARD_LIKE_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def sanitize(text: str) -> str:
    """Returns a copy of `text` with PII patterns masked. Safe to call on
    already-sanitized text (idempotent)."""
    t = EMAIL_RE.sub("[EMAIL]", text)
    t = PHONE_RE.sub("[PHONE]", t)
    t = CASE_OR_ORDER_RE.sub(lambda m: f"{m.group(1)} [REFERENCE_NUMBER]", t)
    t = CARD_LIKE_RE.sub("[NUMBER]", t)
    return t
