"""Phase J LLM-as-judge prompt construction.

Builds the judge prompt from the FROZEN rubric in config/judge_rubric.yaml
(loaded, not re-typed, so the prompt and the documented rubric can never
drift apart) and formats one reply (No-RAG or RAG) for scoring.

The judge never sees whether it is scoring a "No-RAG" or "RAG" reply by
name in a way that could bias it toward the "expected" answer -- see
build_judge_prompt()'s docstring for exactly what is and isn't shown.
"""

from __future__ import annotations

import yaml
from pathlib import Path

RUBRIC = yaml.safe_load(Path("config/judge_rubric.yaml").read_text(encoding="utf-8"))

MAJOR_ISSUE_CATEGORIES = [
    "none",
    "hallucinated_troubleshooting",
    "unsupported_claim",
    "generic_non_actionable",
    "excessive_refusal",
    "evidence_mismatch",
    "copied_without_adapting",
    "failed_to_acknowledge_insufficient_evidence",
    "wrong_intent_despite_evidence",
    "contradictory_evidence",
    "other",
]

DIMENSION_KEYS = [
    "relevance", "helpfulness", "grounding", "factual_safety",
    "resolution_quality", "uncertainty_handling", "professionalism",
]


def _format_rubric_text() -> str:
    lines = []
    for i, key in enumerate(DIMENSION_KEYS, 1):
        dim = RUBRIC["dimensions"][key]
        lines.append(f"{i}. {key} -- {dim['question']}")
        for score in (5, 4, 3, 2, 1):
            lines.append(f"   {score} = {dim['scale'][score]}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are an impartial evaluator of AI-generated customer-support replies for a Twitter-based Apple product support account.

You will be shown a customer's message (and prior conversation turns, if any), the predicted intent, and ONE candidate reply. Some replies were generated with access to historical support evidence, some without -- when historical evidence is shown to you, use it; when none is shown, evaluate the reply on general product-support standards alone. Do not guess which generation method produced the reply and do not let that guess affect your scoring -- score only what is in front of you.

Score the reply on exactly these 7 dimensions, each 1-5:

{_format_rubric_text()}

IMPORTANT on grounding: a good grounded reply may PARAPHRASE historical guidance in its own words. Do NOT require or reward lexical/word-for-word copying of the historical evidence -- a paraphrased claim that is faithful to what the evidence actually says should score as well-grounded (4-5). Only penalize grounding when a SPECIFIC claim (a fix, a policy, a number, a link's content) is NOT supported by the evidence shown, or contradicts it. Distinguish three kinds of content: (A) claims directly supported by the evidence, (B) reasonable general customer-support guidance that needs no specific evidence (e.g. "please try restarting"), and (C) unsupported or contradictory specific claims -- only (C) should lower the grounding score.

If the reply is empty (the system declined to answer), still score all 7 dimensions using the literal scale definitions above -- an appropriate, well-justified decline will score low on relevance/helpfulness/resolution_quality (there is no actionable content) but should score high on grounding, factual_safety, and uncertainty_handling if declining was genuinely the right call given what's known.

Return your evaluation as a single JSON object with EXACTLY these keys:
{{
  "example_id": "<copy exactly as given>",
  "relevance_score": <int 1-5>,
  "helpfulness_score": <int 1-5>,
  "grounding_score": <int 1-5>,
  "factual_safety_score": <int 1-5>,
  "resolution_score": <int 1-5>,
  "uncertainty_score": <int 1-5>,
  "professionalism_score": <int 1-5>,
  "overall_score": <float, the mean of the 7 scores above, computed by you>,
  "brief_reason": "<1-2 sentences justifying the scores, citing specific reply content>",
  "major_issue": "<one of: {', '.join(MAJOR_ISSUE_CATEGORIES)}>",
  "confidence": <float 0.0-1.0, your own self-reported confidence in this evaluation>
}}

Return ONLY the JSON object, no other text."""


def format_evidence_block(retrieved_evidence: list[dict] | None) -> str:
    if not retrieved_evidence:
        return "No historical evidence was provided for this reply."
    lines = ["Historical evidence provided to the reply generator:"]
    for i, e in enumerate(retrieved_evidence, 1):
        lines.append(
            f"[{i}] Similarity={e['similarity']:.3f}, intent_match={e['intent_match']}\n"
            f"    Historical customer message: {e['customer_text']}\n"
            f"    Historical agent response: {e['agent_response']}"
        )
    return "\n".join(lines)


def build_judge_prompt(
    example_id: str,
    customer_message: str,
    conversation: list[dict],
    predicted_intent: str,
    reply: str,
    retrieved_evidence: list[dict] | None,
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt). Deliberately does NOT tell the
    judge whether this is a "no_rag" or "rag" reply by label -- only
    whether evidence was or wasn't provided, which is the actually
    relevant fact for grounding evaluation."""
    context_block = ""
    if conversation:
        context_block = "Prior conversation:\n" + "\n".join(
            f"  {t.get('speaker', '?')}: {t.get('text', '')}" for t in conversation
        ) + "\n\n"

    user_prompt = f"""example_id: {example_id}

{context_block}Customer message: {customer_message}

Predicted intent: {predicted_intent}

{format_evidence_block(retrieved_evidence)}

Candidate reply to evaluate:
{reply if reply.strip() else "(empty -- the system declined to answer)"}

Evaluate this reply now, following the rubric exactly."""

    return SYSTEM_PROMPT, user_prompt
