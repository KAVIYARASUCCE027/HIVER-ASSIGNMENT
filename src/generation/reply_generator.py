"""Grounded reply generator (Section 9-11).

Mirrors src/classification/llm_classifier.py's structure: build prompt, call
the same provider-agnostic LLM interface, validate strict JSON, retry on
failure, cache every call. The only new validation rule specific to
generation is that every `evidence_id` the model returns must be one that
was actually offered to it -- the model is never allowed to invent an
evidence_id, per Section 8.

Run standalone via evaluation/run_rag.py, not directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from src.classification.llm_providers import LLMConfig, complete
from src.generation.prompts import build_generation_prompt
from src.retrieval.evidence import Evidence

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/processed/llm_replies")
MAX_RETRIES = 3
REQUIRED_KEYS = {"reply", "evidence_ids", "should_answer", "grounding_note"}


@dataclass
class GenerationResult:
    reply: str
    evidence_ids: list[str]
    should_answer: bool
    grounding_note: str
    raw_response: str
    attempts: int
    from_cache: bool
    error_kind: str | None = None  # None (success) | "api_error" | "validation_error"

    def to_dict(self) -> dict:
        return {
            "reply": self.reply,
            "evidence_ids": self.evidence_ids,
            "should_answer": self.should_answer,
            "grounding_note": self.grounding_note,
            "raw_response": self.raw_response,
            "attempts": self.attempts,
            "error_kind": self.error_kind,
        }


# Two distinct failure notes -- these are NOT interchangeable. An api_error
# means the model was never actually asked the question (the call itself
# failed, e.g. a provider rate limit/quota error); it is missing data that
# must be retried, not a model judgment, and must never be counted as a
# genuine "insufficient evidence" decision in reporting. A validation_error
# means the model responded but its output failed schema/grounding checks
# after retries; that IS a safe, real fail-closed outcome.
FAILURE_NOTE_API_ERROR = (
    "Generation call failed after {attempts} attempts due to an API/provider error "
    "({error}); this is missing data from a transient failure (e.g. rate limit or "
    "quota), NOT a model judgment -- must be retried, never counted as insufficient evidence."
)
FAILURE_NOTE_VALIDATION = "Generation failed validation after retries; treated as insufficient evidence."


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


def _validate(parsed: dict, allowed_evidence_ids: set[str]) -> tuple[bool, str | None]:
    missing = REQUIRED_KEYS - set(parsed.keys())
    if missing:
        return False, f"missing keys: {missing}"
    if not isinstance(parsed["should_answer"], bool):
        return False, "should_answer is not a boolean"
    if not isinstance(parsed["reply"], str):
        return False, "reply is not a string"
    if not isinstance(parsed["evidence_ids"], list) or not all(isinstance(e, str) for e in parsed["evidence_ids"]):
        return False, "evidence_ids is not a list of strings"
    invented = set(parsed["evidence_ids"]) - allowed_evidence_ids
    if invented:
        return False, f"invented evidence_id(s) not offered to the model: {invented}"
    if parsed["should_answer"] and not parsed["reply"].strip():
        return False, "should_answer is true but reply is empty"
    if not isinstance(parsed.get("grounding_note"), str) or not parsed["grounding_note"].strip():
        return False, "grounding_note is missing or empty"
    return True, None


def _prompt_hash(system_prompt: str, user_prompt: str) -> str:
    return hashlib.sha256((system_prompt + "\x00" + user_prompt).encode("utf-8")).hexdigest()[:16]


def generate_reply(
    example_id: str, customer_message: str, conversation: list[dict], predicted_intent: str,
    evidence: list[Evidence], config: LLMConfig | None = None, use_cache: bool = True,
    variant: str = "rag",
) -> GenerationResult:
    """`variant` ("rag" or "no_rag") only affects the cache path -- the prompt
    is already fully determined by `evidence` (empty list for no_rag)."""
    config = config or LLMConfig.from_env()
    system_prompt, user_prompt = build_generation_prompt(customer_message, conversation, predicted_intent, evidence)
    allowed_ids = {e.evidence_id for e in evidence}

    phash = _prompt_hash(system_prompt, user_prompt)
    cache_path = CACHE_DIR / variant / f"{example_id}__{config.model}__{phash}.json"

    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return GenerationResult(**cached, from_cache=True)

    attempts = 0
    last_error = None
    last_error_kind = None
    correction = ""
    raw_response = ""

    for attempt in range(1, MAX_RETRIES + 1):
        attempts = attempt
        try:
            raw_response = complete(system_prompt, user_prompt + correction, config=config, max_tokens=400)
        except Exception as exc:
            last_error, last_error_kind = str(exc), "api_error"
            logger.warning("Generation call failed (attempt %d/%d) for %s: %s", attempt, MAX_RETRIES, example_id, exc)
            time.sleep(min(2 ** attempt, 8))
            continue

        try:
            parsed = _extract_json(raw_response)
            valid, error = _validate(parsed, allowed_ids)
        except Exception as exc:
            valid, error = False, str(exc)
            parsed = {}

        if valid:
            result = GenerationResult(
                reply=parsed["reply"], evidence_ids=parsed["evidence_ids"],
                should_answer=parsed["should_answer"], grounding_note=parsed["grounding_note"],
                raw_response=raw_response, attempts=attempts, from_cache=False, error_kind=None,
            )
            if use_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            return result

        last_error, last_error_kind = error, "validation_error"
        logger.warning("Invalid generation response (attempt %d/%d) for %s: %s", attempt, MAX_RETRIES, example_id, error)
        correction = f"\n\nYour previous response was invalid ({error}). Return ONLY the JSON object with exactly the required keys, and only evidence_ids that were actually offered to you above."

    logger.error("Generation failed for %s after %d attempts (%s): %s", example_id, attempts, last_error_kind, last_error)
    # Fail safe, not silent: an unparseable/invalid generation OR a transient
    # API failure is treated as should_answer=False (never a fabricated
    # reply), but the two are recorded distinctly (error_kind) so reporting
    # never mistakes "the API call failed" for "the model judged the
    # evidence insufficient". Neither outcome is cached (see call site),
    # so a later rerun retries these for real rather than reusing this stub.
    if last_error_kind == "api_error":
        note = FAILURE_NOTE_API_ERROR.format(attempts=attempts, error=last_error)
    else:
        note = FAILURE_NOTE_VALIDATION
    return GenerationResult(
        reply="", evidence_ids=[], should_answer=False, grounding_note=note,
        raw_response=raw_response, attempts=attempts, from_cache=False, error_kind=last_error_kind,
    )
