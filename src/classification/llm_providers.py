"""Provider-agnostic LLM call interface.

Configured entirely via environment variables (loaded from `.env`, never
committed -- see `.env.example`):
  LLM_PROVIDER  - "anthropic" (only adapter implemented so far)
  LLM_MODEL     - model id, e.g. "claude-sonnet-5"
  LLM_API_KEY   - API key for the selected provider

Nothing above this module (prompts.py, llm_classifier.py) references a
provider by name -- they only call `complete()`, so adding a second
provider (e.g. OpenAI) later means adding one function here, not touching
the classifier or prompt code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


class LLMConfigError(RuntimeError):
    """Raised when required provider configuration (e.g. an API key) is missing."""


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str | None = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
        model = os.environ.get("LLM_MODEL", "").strip()
        api_key = os.environ.get("LLM_API_KEY", "").strip()
        base_url = os.environ.get("LLM_BASE_URL", "").strip() or None
        if not provider:
            raise LLMConfigError(
                "LLM_PROVIDER is not set. Copy .env.example to .env and fill it in."
            )
        if not api_key:
            raise LLMConfigError(
                "LLM_API_KEY is not set. Copy .env.example to .env and add your API key -- "
                "never pass a key on the command line or commit it."
            )
        if not model:
            raise LLMConfigError("LLM_MODEL is not set.")
        return cls(provider=provider, model=model, api_key=api_key, base_url=base_url)


def _complete_anthropic(config: LLMConfig, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.api_key)
    response = client.messages.create(
        model=config.model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        # this SDK version's typed create() doesn't model `temperature` directly;
        # the Messages API itself does, so pass it through raw for determinism.
        extra_body={"temperature": 0.0},
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _complete_groq(config: LLMConfig, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    """Groq exposes an OpenAI-compatible chat completions API."""
    import openai

    client = openai.OpenAI(api_key=config.api_key, base_url=config.base_url or "https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model=config.model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def _complete_gemini(config: LLMConfig, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.api_key)
    response = client.models.generate_content(
        model=config.model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",  # native JSON mode; still validated downstream
        ),
    )
    return response.text or ""


_PROVIDERS = {
    "anthropic": _complete_anthropic,
    "groq": _complete_groq,
    "gemini": _complete_gemini,
}


def complete(system_prompt: str, user_prompt: str, config: LLMConfig | None = None, max_tokens: int = 300) -> str:
    config = config or LLMConfig.from_env()
    if config.provider not in _PROVIDERS:
        raise LLMConfigError(
            f"Unknown LLM_PROVIDER={config.provider!r}. Implemented: {sorted(_PROVIDERS)}"
        )
    return _PROVIDERS[config.provider](config, system_prompt, user_prompt, max_tokens)
