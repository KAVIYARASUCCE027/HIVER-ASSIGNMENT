"""Unit tests for the single-message inference orchestration layer
(src/agent.py). These test the WIRING between the four pipeline stages,
not the stages themselves (each already has its own frozen behavior
exercised by the Phase G/H/I evaluation runs) -- so every external
boundary (the LLM classifier call and the LLM generation call) is mocked.
No test here makes a live API call, and none reads/writes
data/golden/ or evaluation/results/.

The escalation policy (src.escalation.policy.decide) is deliberately NOT
mocked: it is pure, deterministic, offline code (no LLM call), so letting
it run for real against mocked upstream values is what actually verifies
the orchestration produces a correct end-to-end decision.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent import run_agent
from src.classification.llm_classifier import ClassificationResult
from src.generation.reply_generator import GenerationResult
from src.retrieval.evidence import Evidence


def _classification_result(intent="battery_life_drain", confidence=0.95, alt=None):
    return ClassificationResult(
        intent=intent,
        confidence=confidence,
        confidence_type="self_reported",
        alternative_intent=alt,
        reason="Customer describes fast battery drain after an update.",
        raw_response="{}",
        attempts=1,
        from_cache=False,
    )


def _evidence(evidence_id="apple_conv_test_1", similarity=0.85, intent_match=True):
    return Evidence(
        evidence_id=evidence_id,
        conversation_id="test_1",
        customer_text="my battery drains fast after the update",
        agent_response="Try these battery tips: [link]",
        similarity=similarity,
        intent_match=intent_match,
        resolution_quality=0.8,
        is_substantive=True,
    )


def _generation_result(should_answer=True, reply="Here are some battery tips.", evidence_ids=None):
    return GenerationResult(
        reply=reply if should_answer else "",
        evidence_ids=evidence_ids or [],
        should_answer=should_answer,
        grounding_note="Grounded in the retrieved battery-tips evidence." if should_answer else "No usable evidence.",
        raw_response="{}",
        attempts=1,
        from_cache=False,
        error_kind=None,
    )


class TestRunAgentSuccessStructure:
    @patch("src.agent.generate_reply")
    @patch("src.agent.rerank")
    @patch("src.agent.retrieve_candidates")
    @patch("src.agent._get_index")
    @patch("src.agent.classify")
    def test_returns_expected_keys_and_auto_handles_a_routine_message(
        self, mock_classify, mock_get_index, mock_retrieve, mock_rerank, mock_generate,
    ):
        mock_classify.return_value = _classification_result(confidence=0.95)
        mock_get_index.return_value = MagicMock()
        mock_retrieve.return_value = [_evidence()]
        mock_rerank.return_value = [_evidence(similarity=0.85)]  # clears the 0.70 threshold
        mock_generate.return_value = _generation_result(should_answer=True, evidence_ids=["apple_conv_test_1"])

        result = run_agent("My iPhone battery is draining really fast after the latest update.")

        for key in (
            "customer_message", "intent", "confidence", "confidence_type", "action",
            "escalation_reasons", "reply", "retrieved_evidence", "evidence_sufficient",
            "should_answer", "send_automatically",
        ):
            assert key in result, f"missing expected key: {key}"

        assert result["intent"] == "battery_life_drain"
        assert result["confidence"] == 0.95
        assert result["confidence_type"] == "self_reported"
        assert result["evidence_sufficient"] is True
        assert len(result["retrieved_evidence"]) == 1
        assert result["action"] == "AUTO_HANDLE"
        assert result["escalation_reasons"] == []
        assert result["send_automatically"] is True
        assert result["reply"] == "Here are some battery tips."
        mock_generate.assert_called_once()

    @patch("src.agent.generate_reply")
    @patch("src.agent.rerank")
    @patch("src.agent.retrieve_candidates")
    @patch("src.agent._get_index")
    @patch("src.agent.classify")
    def test_insufficient_evidence_skips_the_generation_call(
        self, mock_classify, mock_get_index, mock_retrieve, mock_rerank, mock_generate,
    ):
        """Mirrors evaluation/run_rag.py's retrieval-level short-circuit:
        below the evidence threshold, generate_reply must never be called."""
        mock_classify.return_value = _classification_result(confidence=0.95)
        mock_get_index.return_value = MagicMock()
        mock_retrieve.return_value = [_evidence(similarity=0.40)]
        mock_rerank.return_value = [_evidence(similarity=0.40)]  # below EVIDENCE_THRESHOLD (0.70)

        result = run_agent("Some rare, hard-to-match request.")

        mock_generate.assert_not_called()
        assert result["evidence_sufficient"] is False
        assert result["retrieved_evidence"] == []
        assert result["should_answer"] is False
        assert result["reply"] == ""
        # No historical evidence is itself an escalation trigger in the
        # unmodified policy (src/escalation/policy.py).
        assert result["action"] == "ESCALATE"
        assert "insufficient_historical_evidence" in result["escalation_reasons"]
        assert result["send_automatically"] is False


class TestRunAgentEscalationHandling:
    @patch("src.agent.generate_reply")
    @patch("src.agent.rerank")
    @patch("src.agent.retrieve_candidates")
    @patch("src.agent._get_index")
    @patch("src.agent.classify")
    def test_high_risk_intent_escalates_even_with_a_grounded_reply(
        self, mock_classify, mock_get_index, mock_retrieve, mock_rerank, mock_generate,
    ):
        # account_apple_id_security is one of src.escalation.policy.HIGH_RISK_INTENTS
        mock_classify.return_value = _classification_result(intent="account_apple_id_security", confidence=0.98)
        mock_get_index.return_value = MagicMock()
        mock_retrieve.return_value = [_evidence(intent_match=True)]
        mock_rerank.return_value = [_evidence(intent_match=True, similarity=0.90)]
        mock_generate.return_value = _generation_result(should_answer=True, reply="You can reset your password here.")

        result = run_agent("I can't log into my Apple ID and I think it's been hacked.")

        assert result["action"] == "ESCALATE"
        assert "high_risk_request" in result["escalation_reasons"]
        assert result["send_automatically"] is False
        # The reply is still surfaced (a human agent can see what was
        # drafted) but must not be marked as auto-sendable.
        assert result["reply"] == "You can reset your password here."

    @patch("src.agent.classify")
    def test_classification_failure_escalates_without_retrieval_or_generation(self, mock_classify):
        from src.classification.llm_classifier import FAILURE_SENTINEL

        mock_classify.return_value = ClassificationResult(
            intent=FAILURE_SENTINEL, confidence=0.0, confidence_type="self_reported",
            alternative_intent=None, reason="LLM failed to produce valid output: timeout",
            raw_response="", attempts=3, from_cache=False,
        )

        with patch("src.agent.retrieve_candidates") as mock_retrieve, \
             patch("src.agent.generate_reply") as mock_generate:
            result = run_agent("Some message the classifier couldn't parse.")

        mock_retrieve.assert_not_called()
        mock_generate.assert_not_called()
        assert result["action"] == "ESCALATE"
        assert result["escalation_reasons"] == ["classification_failed"]
        assert result["send_automatically"] is False


class TestRunAgentInputValidation:
    @patch("src.agent.classify")
    def test_empty_customer_message_raises_without_calling_the_pipeline(self, mock_classify):
        with pytest.raises(ValueError):
            run_agent("")
        mock_classify.assert_not_called()

    @patch("src.agent.classify")
    def test_whitespace_only_customer_message_raises(self, mock_classify):
        with pytest.raises(ValueError):
            run_agent("   \n\t  ")
        mock_classify.assert_not_called()
