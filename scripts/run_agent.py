"""Minimal CLI for the single-message inference agent (src/agent.py).

Usage:
    python scripts/run_agent.py "My iPhone battery is draining really fast after the latest update."

Requires LLM_PROVIDER / LLM_MODEL / LLM_API_KEY set in `.env` (copy from
`.env.example`) -- same requirement as every evaluation/run_*.py script.

This is a demo/inference entry point for ONE new message. It is not the
golden-set evaluation harness (see evaluation/run_*.py for that) and does
not read or write anything under data/golden/ or evaluation/results/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows terminals often default stdout to a non-UTF-8 codepage (e.g.
# cp1252), which can't encode some characters that appear verbatim in the
# historical Twitter corpus. Re-configuring stdout to UTF-8 (replacing any
# genuinely unencodable character rather than crashing) only affects how
# this CLI prints -- it changes nothing about the retrieved/generated data.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.agent import run_agent  # noqa: E402
from src.classification.llm_providers import LLMConfigError  # noqa: E402


def _print_evidence(evidence: list[dict]) -> None:
    if not evidence:
        print("    (none)")
        return
    for e in evidence:
        print(f"    [{e['evidence_id']}] similarity={e['similarity']:.2f} intent_match={e['intent_match']}")
        print(f"      customer: {e['customer_text'][:160]!r}")
        print(f"      agent:    {e['agent_response'][:160]!r}")


def _print_result(result: dict) -> None:
    print("=" * 72)
    print(" AI Customer Support Agent -- single-message inference (DEMO)")
    print("=" * 72)
    print(f"Customer message: {result['customer_message']!r}")
    if result["conversation"]:
        print(f"Preceding conversation: {len(result['conversation'])} prior message(s)")

    print("\n-- Classification --")
    print(f"  intent:             {result['intent']}")
    print(f"  confidence:         {result['confidence']} ({result['confidence_type']}, NOT calibrated)")
    print(f"  alternative_intent: {result['alternative_intent']}")
    print(f"  reason:             {result['classification_reason']}")

    print("\n-- Retrieval --")
    print(f"  evidence_sufficient: {result['evidence_sufficient']} (threshold={result['evidence_threshold']})")
    print(f"  retrieved_evidence ({len(result['retrieved_evidence'])}):")
    _print_evidence(result["retrieved_evidence"])

    print("\n-- Generation --")
    print(f"  should_answer:   {result['should_answer']}")
    print(f"  error_kind:      {result['error_kind']}")
    print(f"  grounding_note:  {result['grounding_note']}")
    print(f"  reply:           {result['reply']!r}")

    print("\n-- Escalation --")
    print(f"  action:  {result['action']}")
    print(f"  reasons: {result['escalation_reasons']}")
    if result["action"] == "ESCALATE":
        print(
            "  NOTE: this case is routed to a human agent. The drafted reply above "
            "(if any) was NOT sent to the customer automatically."
        )
    else:
        print("  NOTE: this case is eligible for automatic handling; the reply above would be sent.")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one new customer message through the agent pipeline.")
    parser.add_argument("customer_message", help="The new customer support message to process.")
    parser.add_argument(
        "--conversation-json", default=None,
        help='Optional prior conversation as a JSON list, e.g. \'[{"speaker":"customer","text":"..."}]\'.',
    )
    parser.add_argument("--json", action="store_true", help="Print the raw result as JSON instead of the formatted view.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the LLM response cache for this run.")
    args = parser.parse_args()

    conversation = json.loads(args.conversation_json) if args.conversation_json else None

    try:
        result = run_agent(args.customer_message, conversation=conversation, use_cache=not args.no_cache)
    except LLMConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
