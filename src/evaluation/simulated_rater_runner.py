"""Runner for the Phase J Claude-simulated rating pass (NOT a human rater --
see src/evaluation/simulated_rater_prompt.py's module docstring). Mirrors
src/evaluation/judge_runner.py's structure and validation strictness.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from src.classification.llm_providers import LLMConfig, complete
from src.evaluation.simulated_rater_prompt import MAJOR_ISSUE_CATEGORIES, build_simulated_rating_prompt

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/processed/claude_simulated_ratings")
MAX_RETRIES = 3
REQUIRED_KEYS = {
    "example_id", "audit_notes", "relevance_score", "helpfulness_score", "grounding_score",
    "factual_safety_score", "resolution_score", "uncertainty_score",
    "professionalism_score", "overall_score", "brief_reason", "major_issue", "confidence",
}
SCORE_KEYS = [
    "relevance_score", "helpfulness_score", "grounding_score", "factual_safety_score",
    "resolution_score", "uncertainty_score", "professionalism_score",
]


@dataclass
class SimulatedRatingResult:
    example_id: str
    version: str
    relevance_score: int | None
    helpfulness_score: int | None
    grounding_score: int | None
    factual_safety_score: int | None
    resolution_score: int | None
    uncertainty_score: int | None
    professionalism_score: int | None
    overall_score: float | None
    audit_notes: str
    brief_reason: str
    major_issue: str
    confidence: float
    attempts: int
    from_cache: bool
    error_kind: str | None = None

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "example_id", "version", "relevance_score", "helpfulness_score", "grounding_score",
            "factual_safety_score", "resolution_score", "uncertainty_score", "professionalism_score",
            "overall_score", "audit_notes", "brief_reason", "major_issue", "confidence", "attempts", "error_kind",
        )}


def _extract_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in response: {text!r}")
    return json.loads(t[start:end + 1])


def _validate(parsed: dict, example_id: str) -> tuple[bool, str | None]:
    missing = REQUIRED_KEYS - set(parsed.keys())
    if missing:
        return False, f"missing keys: {missing}"
    if parsed.get("example_id") != example_id:
        return False, f"example_id mismatch: got {parsed.get('example_id')!r}"
    for k in SCORE_KEYS:
        v = parsed.get(k)
        if not isinstance(v, int) or not (1 <= v <= 5):
            return False, f"{k} is not an integer in [1,5]: {v!r}"
    if parsed.get("major_issue") not in MAJOR_ISSUE_CATEGORIES:
        return False, f"major_issue not in allowed set: {parsed.get('major_issue')!r}"
    conf = parsed.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
        return False, f"confidence is not a float in [0,1]: {conf!r}"
    if not isinstance(parsed.get("audit_notes"), str) or not parsed["audit_notes"].strip():
        return False, "audit_notes is missing or empty"
    if not isinstance(parsed.get("brief_reason"), str) or not parsed["brief_reason"].strip():
        return False, "brief_reason is missing or empty"
    return True, None


def _prompt_hash(system_prompt: str, user_prompt: str) -> str:
    return hashlib.sha256((system_prompt + "\x00" + user_prompt).encode("utf-8")).hexdigest()[:16]


def simulated_rate_reply(
    example_id: str, customer_message: str, conversation: list[dict], predicted_intent: str,
    reply: str, retrieved_evidence: list[dict] | None, version: str,
    config: LLMConfig | None = None, use_cache: bool = True,
) -> SimulatedRatingResult:
    config = config or LLMConfig.from_env()
    system_prompt, user_prompt = build_simulated_rating_prompt(
        example_id, customer_message, conversation, predicted_intent, reply, retrieved_evidence
    )
    phash = _prompt_hash(system_prompt, user_prompt)
    cache_path = CACHE_DIR / version / f"{example_id}__{config.model}__{phash}.json"

    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return SimulatedRatingResult(**cached, from_cache=True)

    attempts = 0
    last_error_kind = None
    correction = ""
    for attempt in range(1, MAX_RETRIES + 1):
        attempts = attempt
        try:
            raw_response = complete(system_prompt, user_prompt + correction, config=config, max_tokens=600)
        except Exception as exc:
            last_error_kind = "api_error"
            logger.warning("Simulated-rater call failed (attempt %d/%d) for %s/%s: %s", attempt, MAX_RETRIES, example_id, version, exc)
            time.sleep(min(2 ** attempt, 8))
            continue
        try:
            parsed = _extract_json(raw_response)
            valid, error = _validate(parsed, example_id)
        except Exception as exc:
            valid, error = False, str(exc)
            parsed = {}
        if valid:
            computed_overall = round(sum(parsed[k] for k in SCORE_KEYS) / len(SCORE_KEYS), 4)
            result = SimulatedRatingResult(
                example_id=example_id, version=version,
                relevance_score=parsed["relevance_score"], helpfulness_score=parsed["helpfulness_score"],
                grounding_score=parsed["grounding_score"], factual_safety_score=parsed["factual_safety_score"],
                resolution_score=parsed["resolution_score"], uncertainty_score=parsed["uncertainty_score"],
                professionalism_score=parsed["professionalism_score"], overall_score=computed_overall,
                audit_notes=parsed["audit_notes"], brief_reason=parsed["brief_reason"],
                major_issue=parsed["major_issue"], confidence=float(parsed["confidence"]),
                attempts=attempts, from_cache=False, error_kind=None,
            )
            if use_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        last_error_kind = "validation_error"
        logger.warning("Invalid simulated-rater response (attempt %d/%d) for %s/%s: %s", attempt, MAX_RETRIES, example_id, version, error)
        correction = f"\n\nYour previous response was invalid ({error}). Return ONLY the JSON object with exactly the required keys."

    logger.error("Simulated rating failed for %s/%s after %d attempts (%s)", example_id, version, attempts, last_error_kind)
    return SimulatedRatingResult(
        example_id=example_id, version=version,
        relevance_score=None, helpfulness_score=None, grounding_score=None, factual_safety_score=None,
        resolution_score=None, uncertainty_score=None, professionalism_score=None, overall_score=None,
        audit_notes="", brief_reason="Simulated rating failed validation after retries.", major_issue="other",
        confidence=0.0, attempts=attempts, from_cache=False, error_kind=last_error_kind,
    )
