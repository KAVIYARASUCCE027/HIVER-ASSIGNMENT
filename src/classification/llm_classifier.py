"""Standalone zero-shot LLM intent classifier.

Receives: customer message + preceding conversation context + the frozen
14-intent taxonomy (via src/classification/prompts.py). Returns strict JSON:

    {"intent": ..., "confidence": ..., "alternative_intent": ..., "reason": ...}

`confidence` is the model's own self-report -- stored alongside an explicit
`confidence_type: "self_reported"` so nothing downstream mistakes it for a
calibrated probability (see reports/llm_classifier_results.md's confidence
analysis, which is explicitly NOT calibration).

Invalid JSON / schema violations are retried with a corrective instruction
(up to MAX_RETRIES). A response that still fails validation after retries is
returned with intent="_CLASSIFICATION_FAILED_" (guaranteed not to match any
real intent) rather than silently coerced into a valid-looking label.

Every call is cached to `data/processed/llm_predictions/` keyed by example id
+ a hash of the exact prompt, so re-running the evaluation never re-calls the
API for an already-answered, unchanged example.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from src.classification.llm_providers import LLMConfig, complete
from src.classification.prompts import build_classification_prompt, load_intents

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/processed/llm_predictions")
MAX_RETRIES = 3
FAILURE_SENTINEL = "_CLASSIFICATION_FAILED_"

REQUIRED_KEYS = {"intent", "confidence", "alternative_intent", "reason"}


@dataclass
class ClassificationResult:
    intent: str
    confidence: float
    confidence_type: str
    alternative_intent: str | None
    reason: str
    raw_response: str
    attempts: int
    from_cache: bool

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "confidence_type": self.confidence_type,
            "alternative_intent": self.alternative_intent,
            "reason": self.reason,
            "raw_response": self.raw_response,
            "attempts": self.attempts,
        }


def _extract_json(text: str) -> dict:
    """Model is instructed to return only JSON, but be tolerant of stray
    whitespace or an accidental code fence."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in response: {text!r}")
    return json.loads(t[start:end + 1])


def _validate(parsed: dict, intent_names: set[str]) -> tuple[bool, str | None]:
    missing = REQUIRED_KEYS - set(parsed.keys())
    if missing:
        return False, f"missing keys: {missing}"
    if parsed["intent"] not in intent_names:
        return False, f"intent {parsed['intent']!r} is not one of the 14 taxonomy names"
    try:
        conf = float(parsed["confidence"])
    except (TypeError, ValueError):
        return False, "confidence is not a number"
    if not (0.0 <= conf <= 1.0):
        return False, "confidence out of [0,1] range"
    alt = parsed.get("alternative_intent")
    if alt is not None and alt not in intent_names:
        return False, f"alternative_intent {alt!r} is not one of the 14 taxonomy names or null"
    if not isinstance(parsed.get("reason"), str) or not parsed["reason"].strip():
        return False, "reason is missing or empty"
    return True, None


def _prompt_hash(system_prompt: str, user_prompt: str) -> str:
    return hashlib.sha256((system_prompt + "\x00" + user_prompt).encode("utf-8")).hexdigest()[:16]


def _cache_path(example_id: str, model: str, prompt_hash: str) -> Path:
    safe_model = model.replace("/", "_")
    return CACHE_DIR / f"{example_id}__{safe_model}__{prompt_hash}.json"


def classify(
    example_id: str,
    customer_message: str,
    conversation: list[dict],
    config: LLMConfig | None = None,
    use_cache: bool = True,
) -> ClassificationResult:
    config = config or LLMConfig.from_env()
    intents = load_intents()
    intent_names = {i["name"] for i in intents}

    system_prompt, user_prompt = build_classification_prompt(customer_message, conversation)
    phash = _prompt_hash(system_prompt, user_prompt)
    cache_path = _cache_path(example_id, config.model, phash)

    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("intent") != FAILURE_SENTINEL:
            return ClassificationResult(**cached, from_cache=True)
        logger.info("Cached result for %s was a failure (e.g. rate limit) -- retrying instead of reusing it.",
                    example_id)

    attempts = 0
    last_error = None
    correction = ""
    raw_response = ""

    for attempt in range(1, MAX_RETRIES + 1):
        attempts = attempt
        try:
            raw_response = complete(system_prompt, user_prompt + correction, config=config)
        except Exception as exc:  # provider/network error -- retry with backoff
            last_error = str(exc)
            logger.warning("LLM call failed (attempt %d/%d) for %s: %s", attempt, MAX_RETRIES, example_id, exc)
            time.sleep(min(2 ** attempt, 8))
            continue

        try:
            parsed = _extract_json(raw_response)
            valid, error = _validate(parsed, intent_names)
        except Exception as exc:
            valid, error = False, str(exc)
            parsed = {}

        if valid:
            result = ClassificationResult(
                intent=parsed["intent"],
                confidence=float(parsed["confidence"]),
                confidence_type="self_reported",
                alternative_intent=parsed.get("alternative_intent"),
                reason=parsed["reason"],
                raw_response=raw_response,
                attempts=attempts,
                from_cache=False,
            )
            if use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            return result

        last_error = error
        logger.warning("Invalid response (attempt %d/%d) for %s: %s", attempt, MAX_RETRIES, example_id, error)
        correction = (
            f"\n\nYour previous response was invalid ({error}). "
            f"Respond with ONLY the JSON object, using exactly the required keys and one of the "
            f"14 taxonomy intent names."
        )

    logger.error("Classification failed for %s after %d attempts: %s", example_id, attempts, last_error)
    # Deliberately NOT cached: a failure (e.g. rate limit) is a transient
    # infrastructure condition, not a real answer -- caching it would make it
    # permanently unretriable even after the underlying condition clears.
    return ClassificationResult(
        intent=FAILURE_SENTINEL, confidence=0.0, confidence_type="self_reported",
        alternative_intent=None, reason=f"LLM failed to produce valid output: {last_error}",
        raw_response=raw_response, attempts=attempts, from_cache=False,
    )
