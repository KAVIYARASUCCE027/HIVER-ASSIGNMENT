"""Prompt construction for the zero-shot LLM intent classifier.

Frozen for the Phase G zero-shot experiment -- do not iterate this against
golden-set results (see CLAUDE.md's "experimental discipline" section).
`build_intent_taxonomy_block()` and `build_classification_prompt()` are the
only functions the golden evaluation calls; both are pure functions of
`config/intents.yaml` and the input example, so re-running produces the
identical prompt.

IMPORTANT PROVENANCE NOTE: each intent's `examples` field below comes from
`config/intents.yaml`, itself derived from Phase D clustering of
`retrieval_pool` messages -- BEFORE the golden set existed (see
reports/intent_discovery.md and reports/decision_log.md decision #5). These
are taxonomy-definition examples, not golden-set examples, and are not
few-shot demonstrations of golden labels. No golden-set content of any kind
appears in this prompt.
"""

from __future__ import annotations

from pathlib import Path

import yaml

INTENTS_PATH = Path("config/intents.yaml")

SYSTEM_PROMPT = """You are an intent classifier for AppleSupport's customer support Twitter account. \
You will be shown a customer's message (and, if available, the preceding conversation) and must \
classify it into exactly one of a fixed set of 14 intents.

Rules:
- Choose exactly ONE primary intent from the taxonomy below. Never invent a new intent name or \
  modify a name's spelling.
- Use the preceding conversation context when it is provided -- the target message sometimes only \
  makes sense in light of what was said before (e.g. "Yes, I've tried that" answering an earlier \
  question).
- Many intents look similar. Use each intent's "confusable with" list and description to distinguish \
  them: prefer the more specific intent when the message names a concrete symptom, and only use a \
  broad/residual intent (ios_update_general_complaint, vague_frustration_needs_clarification) when \
  the message genuinely lacks a more specific signal.
- If the message raises more than one issue, pick the intent that determines the correct next \
  action (not necessarily whichever is mentioned first).
- If the message is genuinely ambiguous between two intents, pick the best-supported one and name \
  the second-best guess in `alternative_intent`.
- Do not invent information not present in the message or conversation (no policies, no promised \
  actions, no assumed device models unless stated).
- `reason` must be a concise (1-2 sentence), evidence-based explanation citing what in the message \
  drove your choice. Do not include step-by-step reasoning or a chain of thought -- state the \
  conclusion and its basis only.
- `confidence` is your own self-assessed confidence in the chosen intent, a number from 0.0 to 1.0. \
  This is a subjective self-report, not a calibrated probability.
- Return ONLY a single JSON object matching the schema below. No markdown code fences, no text \
  before or after it.

Output schema (exactly these keys, nothing else):
{"intent": "<one of the 14 intent names>", "confidence": <float 0.0-1.0>, \
"alternative_intent": "<one of the 14 intent names, or null if not applicable>", \
"reason": "<concise evidence-based explanation>"}
"""


def load_intents() -> list[dict]:
    with INTENTS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["intents"]


def build_intent_taxonomy_block(intents: list[dict] | None = None) -> str:
    intents = intents or load_intents()
    parts = []
    for i in intents:
        examples = "\n".join(f"    - {ex}" for ex in i["examples"])
        confusable = ", ".join(i["confusable_intents"]) or "none identified"
        parts.append(
            f"- name: {i['name']}\n"
            f"  description: {i['description'].strip()}\n"
            f"  representative examples (from pre-golden Phase D taxonomy design, not golden-set data):\n"
            f"{examples}\n"
            f"  confusable with: {confusable}\n"
            f"  escalation guidance (context only, NOT what you are asked to output): "
            f"{i['escalation_guidance'].strip()}"
        )
    return "\n\n".join(parts)


def build_classification_prompt(customer_message: str, conversation: list[dict]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt)."""
    taxonomy_block = build_intent_taxonomy_block()

    if conversation:
        context_lines = "\n".join(f"[{m['speaker']}]: {m['text']}" for m in conversation)
        context_block = f"Preceding conversation:\n{context_lines}\n"
    else:
        context_block = "Preceding conversation: (none -- this is the first message)\n"

    user_prompt = (
        f"Intent taxonomy:\n\n{taxonomy_block}\n\n"
        f"---\n\n"
        f"{context_block}\n"
        f"Customer message to classify:\n{customer_message}\n\n"
        f"Return the JSON object now."
    )
    return SYSTEM_PROMPT, user_prompt
