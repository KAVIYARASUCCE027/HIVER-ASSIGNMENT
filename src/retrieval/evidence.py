"""Evidence object schema.

Every retrieved result is converted to this fixed shape before being shown to
the reply generator or persisted for audit. `evidence_id` always derives from
a real `conversation_id` already present in the indexed corpus -- never
invented (src/retrieval/corpus.py builds it as `f"apple_conv_{conversation_id}"`
at corpus-construction time; this module never mints a new one).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Evidence:
    evidence_id: str
    conversation_id: str
    customer_text: str
    agent_response: str
    similarity: float
    intent_match: bool
    resolution_quality: float
    is_substantive: bool

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "conversation_id": self.conversation_id,
            "customer_text": self.customer_text,
            "agent_response": self.agent_response,
            "similarity": round(self.similarity, 4),
            "intent_match": self.intent_match,
        }


def build_evidence(document: dict, similarity: float, predicted_intent: str) -> Evidence:
    return Evidence(
        evidence_id=document["evidence_id"],
        conversation_id=document["conversation_id"],
        customer_text=document["customer_text"],
        agent_response=document["agent_response"],
        similarity=float(similarity),
        intent_match=document["intent"] == predicted_intent,
        resolution_quality=document["resolution_quality"],
        is_substantive=document["is_substantive"],
    )
