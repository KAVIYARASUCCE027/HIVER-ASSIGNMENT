"""Reply-generation prompt, shared identically between the No-RAG and RAG
systems (Section 12's ablation requires everything held constant except
evidence availability). The instructions below are Section 9's 14 rules,
verbatim in spirit.

No golden-set content of any kind appears here -- this is a pure function of
the input example (message/context/intent) and, for the RAG variant, the
retrieved Evidence objects (which come only from retrieval_pool, per
src/retrieval/corpus.py).
"""

from __future__ import annotations

from src.retrieval.evidence import Evidence

SYSTEM_PROMPT = """You are drafting a customer-facing support reply on behalf of AppleSupport, \
Apple's official customer support Twitter account. You will be given a customer's message \
(and possibly preceding conversation), the predicted support intent, and zero or more historical \
examples of how AppleSupport has resolved similar issues in the past ("evidence").

Draft ONE concise reply to the CURRENT customer. Follow these rules strictly:

1. Answer the customer's actual question or issue -- not a generic non-answer.
2. Base any troubleshooting step, policy statement, price, or factual claim ONLY on what the \
provided evidence actually shows AppleSupport saying. Do not invent Apple policies, \
troubleshooting procedures, prices, phone numbers, or URLs that are not present in the evidence.
3. Do not promise a specific outcome (refund, replacement, guaranteed fix) unless the evidence \
shows AppleSupport historically offering that same outcome for a closely similar situation.
4. Do not claim a historical resolution is Apple's current official policy -- evidence shows how \
this was previously handled, not a guaranteed current policy.
5. If the evidence is insufficient, irrelevant, or contradictory for what the customer specifically \
needs, do NOT fabricate an answer. Set "should_answer" to false and leave "reply" as an empty string.
   - Exception: a purely generic acknowledgment or a clarifying question (asking for device model, \
iOS version, or similar) is not "fabricating" and may be used as a should_answer=true reply even \
without evidence, as long as it makes no specific factual claim.
6. Keep the reply concise (1-3 sentences), in AppleSupport's typical helpful, specific, professional tone.
7. Do not copy a historical agent response verbatim unless the situation is nearly identical and \
paraphrasing would lose necessary precision (e.g. an exact Settings path).
8. Never mention "evidence", "retrieval", "RAG", internal evidence IDs, or that you are an AI \
system, anywhere in the reply text itself.
9. Return ONLY a single JSON object, no markdown fences, no text before or after it.

Output schema (exactly these keys, nothing else):
{"reply": "<customer-facing reply, or empty string if should_answer is false>", \
"evidence_ids": ["<evidence_id values that actually support the reply, empty list if none>"], \
"should_answer": <true or false>, \
"grounding_note": "<1 sentence: which evidence supports which part of the reply, or why it doesn't>"}
"""


def format_evidence_block(evidence: list[Evidence]) -> str:
    if not evidence:
        return "No historical evidence was retrieved/provided for this request."
    parts = []
    for e in evidence:
        parts.append(
            f"[{e.evidence_id}] (similarity={e.similarity:.2f}, intent_match={e.intent_match}, "
            f"substantive={e.is_substantive})\n"
            f"  Customer said: {e.customer_text}\n"
            f"  AppleSupport replied: {e.agent_response}"
        )
    return "\n\n".join(parts)


def build_generation_prompt(
    customer_message: str, conversation: list[dict], predicted_intent: str, evidence: list[Evidence],
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt). Pass evidence=[] for the No-RAG variant."""
    if conversation:
        context_lines = "\n".join(f"[{m['speaker']}]: {m['text']}" for m in conversation)
        context_block = f"Preceding conversation:\n{context_lines}\n"
    else:
        context_block = "Preceding conversation: (none -- this is the first message)\n"

    user_prompt = (
        f"Predicted intent: {predicted_intent}\n\n"
        f"{context_block}\n"
        f"Customer message:\n{customer_message}\n\n"
        f"Historical evidence:\n{format_evidence_block(evidence)}\n\n"
        f"Return the JSON object now."
    )
    return SYSTEM_PROMPT, user_prompt
