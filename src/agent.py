"""Central inference agent -- orchestrates the frozen pipeline stages
(classification, retrieval, generation, escalation) for ONE new customer
message.

This is a minimal demo/inference orchestration layer, not a new model or a
new decision rule: every stage below calls the exact same function the
Phase G/H/I batch evaluations call --

    src.classification.llm_classifier.classify
    src.retrieval.retrieve.retrieve_candidates / src.retrieval.rerank.rerank
    src.generation.reply_generator.generate_reply
    src.escalation.policy.decide

-- with the same config, taxonomy, retrieval index, thresholds, and policy.
No classification/retrieval/generation/escalation logic is duplicated or
re-implemented here; this module only wires the four together for a
message that has no golden `id` and no pre-computed upstream outputs to
read (see reports/decision_log.md for why this was added).

Not used by, and does not affect, any frozen evaluation artifact --
evaluation/run_*.py never imports this module, and this module never
writes to evaluation/results/ or data/golden/.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from src.classification.llm_classifier import FAILURE_SENTINEL, classify
from src.classification.llm_providers import LLMConfig
from src.escalation.policy import EscalationInput, decide
from src.generation.reply_generator import GenerationResult, generate_reply
from src.retrieval.index import RetrievalIndex, build_index
from src.retrieval.rerank import EVIDENCE_THRESHOLD, rerank, rerank_score
from src.retrieval.retrieve import retrieve_candidates

logger = logging.getLogger(__name__)

_INDEX: RetrievalIndex | None = None


def _get_index() -> RetrievalIndex:
    """Lazily load the retrieval index once per process. Reads the already
    -cached corpus/embeddings on disk (data/processed/rag_corpus*.{jsonl,npy})
    -- does not rebuild or re-embed the corpus."""
    global _INDEX
    if _INDEX is None:
        _INDEX = build_index()
    return _INDEX


def _make_example_id(customer_message: str) -> str:
    """Deterministic cache key derived only from the message text, prefixed
    `adhoc_` so it can never collide with, or be mistaken for, a golden-set
    id (data/golden/golden_200.jsonl uses `golden_NNNN`). Running the exact
    same new message twice reuses the cached LLM output for the second run,
    the same idempotent-cache behavior every other phase relies on."""
    digest = hashlib.sha256(customer_message.encode("utf-8")).hexdigest()[:16]
    return f"adhoc_{digest}"


def run_agent(
    customer_message: str,
    conversation: list[dict] | None = None,
    config: LLMConfig | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Run one new customer message end-to-end through the frozen
    classify -> retrieve -> generate -> escalate pipeline and return one
    structured result dict.

    Raises ValueError for an empty/missing customer_message, and
    src.classification.llm_providers.LLMConfigError if LLM_PROVIDER /
    LLM_MODEL / LLM_API_KEY are not set in the environment.
    """
    if not customer_message or not customer_message.strip():
        raise ValueError("customer_message must be a non-empty string")

    conversation = conversation or []
    config = config or LLMConfig.from_env()
    example_id = _make_example_id(customer_message)

    # 1. Classify (src.classification.llm_classifier.classify -- same
    #    taxonomy, prompt, and retry/cache behavior as Phase G).
    cls = classify(example_id, customer_message, conversation, config=config, use_cache=use_cache)

    if cls.intent == FAILURE_SENTINEL:
        # The frozen classifier itself failed validation after retries.
        # src.escalation.policy.decide() has no signal for "no intent was
        # ever produced" -- it assumes a valid taxonomy intent as input --
        # so this orchestration layer (not the frozen policy) fails safe by
        # escalating directly, without attempting retrieval/generation
        # against a meaningless intent. This is new glue logic specific to
        # this demo entry point, not a change to src/escalation/policy.py.
        logger.warning("Classification failed for this message; escalating without retrieval/generation.")
        return {
            "customer_message": customer_message,
            "conversation": conversation,
            "example_id": example_id,
            "intent": None,
            "confidence": None,
            "confidence_type": cls.confidence_type,
            "alternative_intent": None,
            "classification_reason": cls.reason,
            "evidence_sufficient": None,
            "evidence_threshold": EVIDENCE_THRESHOLD,
            "retrieved_evidence": [],
            "should_answer": False,
            "reply": "",
            "grounding_note": "Classification failed; no retrieval or generation attempted.",
            "error_kind": "validation_error",
            "action": "ESCALATE",
            "escalation_reasons": ["classification_failed"],
            "send_automatically": False,
        }

    # 2. Retrieve + rerank (mirrors evaluation/run_rag.py's
    #    retrieve_for_example: same candidate pool, same rerank formula,
    #    same 0.70 evidence threshold -- not redefined here).
    index = _get_index()
    candidates = retrieve_candidates(customer_message, cls.intent, index)
    top_k = rerank(candidates)
    evidence_sufficient = bool(top_k) and rerank_score(top_k[0]) >= EVIDENCE_THRESHOLD
    evidence = top_k if evidence_sufficient else []

    # 3. Generate. Mirrors run_rag.py's RAG-variant behavior exactly: if
    #    evidence is insufficient, no generation call is made at all (the
    #    same deterministic, API-call-free safe state Phase H uses) rather
    #    than inventing a new "fall back to No-RAG generation" behavior
    #    that was never evaluated as part of this system.
    if evidence_sufficient:
        gen = generate_reply(
            example_id, customer_message, conversation, cls.intent, evidence,
            config=config, use_cache=use_cache, variant="rag",
        )
    else:
        # generate_reply() would still call the LLM even with empty
        # evidence -- Phase H's own retrieval-level gate short-circuits
        # BEFORE that call happens at all (evaluation/run_rag.py's
        # retrieve_for_example / reports/rag_results.md Section 11).
        # Reusing that exact short-circuited record here, instead of
        # calling generate_reply(), keeps this demo's behavior identical
        # to the evaluated system and avoids an unnecessary API call.
        gen = GenerationResult(
            reply="", evidence_ids=[], should_answer=False,
            grounding_note="No sufficiently relevant historical resolution found.",
            raw_response="", attempts=0, from_cache=False, error_kind=None,
        )

    # 4. Escalate (src.escalation.policy.decide -- unmodified, same
    #    signals/thresholds/reasons as the promoted Phase I.1 policy).
    decision = decide(EscalationInput(
        example_id=example_id,
        customer_message=customer_message,
        predicted_intent=cls.intent,
        confidence=cls.confidence,
        has_retrieved_evidence=evidence_sufficient,
        should_answer=gen.should_answer,
    ))

    return {
        "customer_message": customer_message,
        "conversation": conversation,
        "example_id": example_id,
        "intent": cls.intent,
        "confidence": cls.confidence,
        "confidence_type": cls.confidence_type,  # "self_reported" -- not a calibrated probability
        "alternative_intent": cls.alternative_intent,
        "classification_reason": cls.reason,
        "evidence_sufficient": evidence_sufficient,
        "evidence_threshold": EVIDENCE_THRESHOLD,
        "retrieved_evidence": [e.to_dict() for e in evidence],
        "should_answer": gen.should_answer,
        "reply": gen.reply,
        "grounding_note": gen.grounding_note,
        "error_kind": gen.error_kind,
        "action": decision.action,
        "escalation_reasons": decision.reasons,
        # Introduced by this orchestration layer, not pre-existing repo
        # terminology: makes explicit that an ESCALATE decision means the
        # drafted `reply` above (if any) is NOT sent to the customer
        # automatically, even though generation may have produced one.
        "send_automatically": decision.action == "AUTO_HANDLE",
    }
