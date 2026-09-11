"""Phase J.1 CANDIDATE judge prompt (V2) -- NOT frozen, NOT used for the
full 400-reply evaluation. Loads config/judge_rubric_v2.yaml. V1
(src/evaluation/judge_prompt.py, config/judge_rubric.yaml) is untouched
and remains the Phase J baseline.

The only conceptual change from V1: explicit, per-dimension instructions
for how to score an intentionally EMPTY reply (a decline), replacing V1's
implicit behavior of applying the normal-reply scale literally to absent
text. See config/judge_rubric_v2.yaml's `refusal_override` fields and
reports/judge_rubric_audit.md for the full reasoning. Non-empty replies
are scored identically to V1 -- nothing here changes how a real reply
with actual text is judged.
"""

from __future__ import annotations

import yaml
from pathlib import Path

RUBRIC_V2 = yaml.safe_load(Path("config/judge_rubric_v2.yaml").read_text(encoding="utf-8"))

MAJOR_ISSUE_CATEGORIES = [
    "none", "hallucinated_troubleshooting", "unsupported_claim", "generic_non_actionable",
    "unjustified_refusal", "justified_refusal", "evidence_mismatch", "copied_without_adapting",
    "wrong_intent_despite_evidence", "contradictory_evidence", "other",
]

DIMENSION_KEYS = [
    "relevance", "helpfulness", "grounding", "factual_safety",
    "resolution_quality", "uncertainty_handling", "professionalism",
]


def _format_rubric_text() -> str:
    lines = []
    for i, key in enumerate(DIMENSION_KEYS, 1):
        dim = RUBRIC_V2["dimensions"][key]
        lines.append(f"{i}. {key} -- {dim['question']}")
        for score in (5, 4, 3, 2, 1):
            lines.append(f"   {score} = {dim['scale'][score]}")
        if dim.get("refusal_override") and dim["refusal_override"].strip() != "None":
            lines.append(f"   [IF THE REPLY IS EMPTY/DECLINED, use this instead of the scale above]: {dim['refusal_override'].strip()}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are an impartial evaluator of AI-generated customer-support replies for a Twitter-based Apple product support account.

You will be shown a customer's message (and prior conversation turns, if any), the predicted intent, and ONE candidate reply. Some replies were generated with access to historical support evidence, some without -- when historical evidence is shown to you, use it; when none is shown, evaluate the reply on general product-support standards alone. Do not guess which generation method produced the reply and do not let that guess affect your scoring -- score only what is in front of you.

Score the reply on exactly these 7 dimensions, each 1-5:

{_format_rubric_text()}

IMPORTANT on grounding: a good grounded reply may PARAPHRASE historical guidance in its own words. Do NOT require or reward lexical/word-for-word copying of the historical evidence -- a paraphrased claim that is faithful to what the evidence actually says should score as well-grounded (4-5). Only penalize grounding when a SPECIFIC claim (a fix, a policy, a number, a link's content) is NOT supported by the evidence shown, or contradicts it.

CRITICAL: if the reply is empty (the system declined to answer), you MUST first decide whether the decline was JUSTIFIED or UNJUSTIFIED before scoring, using ONLY the evidence shown to you plus ordinary product-support knowledge:
  - JUSTIFIED: the historical evidence shown (if any) does not actually address this specific request, or no evidence was shown at all and the request genuinely requires brand/account/history-specific knowledge a generalist could not supply.
  - UNJUSTIFIED: the evidence shown DOES address the request, or the request is a generic question any competent support agent could answer with ordinary product knowledge, and declining was an unnecessary refusal to help.
Then apply each dimension's [IF THE REPLY IS EMPTY/DECLINED] instruction above using that judgment. Declining does NOT earn a good score merely by existing -- an unjustified decline must still score poorly on relevance, helpfulness, resolution_quality, and uncertainty_handling. Set major_issue to "justified_refusal" or "unjustified_refusal" (as appropriate) for any empty reply, instead of "none".

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
  "brief_reason": "<1-2 sentences justifying the scores, citing specific reply content, and for an empty reply stating whether you judged the decline justified or unjustified and why>",
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


def build_judge_prompt_v2(
    example_id: str,
    customer_message: str,
    conversation: list[dict],
    predicted_intent: str,
    reply: str,
    retrieved_evidence: list[dict] | None,
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

Candidate reply to evaluate:
{reply if reply.strip() else "(empty -- the system declined to answer)"}

Evaluate this reply now, following the rubric exactly."""

    return SYSTEM_PROMPT, user_prompt
