"""Phase J: a SECOND, methodologically distinct rating pass used ONLY to
fill in the 20/25 human-rating-subset examples the user did not personally
rate (see reports/judge_results.md Section 1 for the exact split and why).

This is explicitly NOT a human rater and must never be reported or treated
as one. It exists purely as a supplementary self-consistency check between
two different LLM-based evaluation passes -- distinct from
src/evaluation/judge_prompt.py in framing and instructions (a different
persona, reasoning-first output order) so it is not simply the judge
grading itself twice with the identical prompt, though both necessarily
use the same underlying rubric (frozen, config/judge_rubric.yaml) and the
same LLM provider this project has configured throughout.

Every record produced by this module is tagged rater_type="claude_simulated"
downstream and kept in a separate file
(data/evaluation/claude_simulated_ratings.jsonl) from the 5 genuine human
ratings (data/evaluation/human_reply_ratings.jsonl).
"""

from __future__ import annotations

from pathlib import Path

from src.evaluation.judge_prompt import DIMENSION_KEYS, RUBRIC, format_evidence_block

MAJOR_ISSUE_CATEGORIES = [
    "none", "hallucinated_troubleshooting", "unsupported_claim", "generic_non_actionable",
    "excessive_refusal", "evidence_mismatch", "copied_without_adapting",
    "failed_to_acknowledge_insufficient_evidence", "wrong_intent_despite_evidence",
    "contradictory_evidence", "other",
]


def _format_rubric_text() -> str:
    lines = []
    for i, key in enumerate(DIMENSION_KEYS, 1):
        dim = RUBRIC["dimensions"][key]
        lines.append(f"{i}. {key} -- {dim['question']}")
        for score in (5, 4, 3, 2, 1):
            lines.append(f"   {score} = {dim['scale'][score]}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are a meticulous, skeptical customer-support quality auditor. You personally review support replies one at a time, on behalf of the support team, independently of any other review process. Your job is to catch problems other reviewers might rush past -- be critical and specific, not generous.

You will see a customer's message (and prior conversation, if any), the predicted intent, and ONE reply to audit. Some replies were written with access to historical support evidence, some without -- when evidence is shown, check the reply against it; when none is shown, judge the reply on general product-support standards alone.

Score the reply on exactly these 7 dimensions, each 1-5:

{_format_rubric_text()}

On grounding: paraphrasing historical guidance in different words is fine and should NOT be penalized -- do not require word-for-word matches. Only mark a claim as poorly grounded if it states something SPECIFIC (a fix, a policy, a number) that the evidence does not actually support, or contradicts it.

If the reply is empty (declined to answer), score all 7 dimensions using the scale definitions literally -- a well-justified decline scores low on relevance/helpfulness/resolution (nothing actionable was said) but high on grounding/factual_safety/uncertainty_handling if declining was genuinely the right call.

First, in your own words, write 2-3 sentences of honest audit notes about this specific reply's strengths and weaknesses. THEN assign your scores. Return a single JSON object with EXACTLY these keys:
{{
  "example_id": "<copy exactly as given>",
  "audit_notes": "<your 2-3 sentence independent assessment, written BEFORE you finalize scores>",
  "relevance_score": <int 1-5>,
  "helpfulness_score": <int 1-5>,
  "grounding_score": <int 1-5>,
  "factual_safety_score": <int 1-5>,
  "resolution_score": <int 1-5>,
  "uncertainty_score": <int 1-5>,
  "professionalism_score": <int 1-5>,
  "overall_score": <float, the mean of the 7 scores above>,
  "brief_reason": "<1-2 sentence summary citing specific reply content>",
  "major_issue": "<one of: {', '.join(MAJOR_ISSUE_CATEGORIES)}>",
  "confidence": <float 0.0-1.0, your own self-reported confidence>
}}

Return ONLY the JSON object, no other text."""


def build_simulated_rating_prompt(
    example_id: str, customer_message: str, conversation: list[dict], predicted_intent: str,
    reply: str, retrieved_evidence: list[dict] | None,
) -> tuple[str, str]:
    context_block = ""
    if conversation:
        context_block = "Prior conversation:\n" + "\n".join(
            f"  {t.get('speaker', '?')}: {t.get('text', '')}" for t in conversation
        ) + "\n\n"

    user_prompt = f"""example_id: {example_id}

{context_block}Customer message: {customer_message}

Predicted intent: {predicted_intent}

{format_evidence_block(retrieved_evidence)}

Reply to audit:
{reply if reply.strip() else "(empty -- the system declined to answer)"}

Audit this reply now."""

    return SYSTEM_PROMPT, user_prompt
