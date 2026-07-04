"""Legacy LLM client wrapper — delegates to the new provider-based architecture.

This module exists for backward compatibility. New code should use the
LLMProvider interface and service layer directly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from resume_pipeline.config import Settings, load_settings
from resume_pipeline.llm.factory import LLMProviderFactory
from resume_pipeline.llm.interfaces import LLMConfig, LLMError

logger = logging.getLogger(__name__)


def call_llm_json(
    prompt: str,
    model: str,
    settings: Settings | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    provider = LLMProviderFactory.from_config(settings.llm)

    config = LLMConfig(
        model=model or provider.default_model,
        temperature=settings.llm.resume_generation.temperature,
        top_p=settings.llm.resume_generation.top_p,
        max_tokens=settings.llm.resume_generation.num_predict,
        stream=False,
    )

    return provider.generate_json(prompt, config)


def call_llm_text(
    prompt: str,
    model: str,
    settings: Settings | None = None,
    payload: dict[str, Any] | None = None,
    on_progress: Callable[[int, float], None] | None = None,
) -> str:
    settings = settings or load_settings()
    provider = LLMProviderFactory.from_config(settings.llm)
    gen = settings.llm.resume_generation

    config = LLMConfig(
        model=model or provider.default_model,
        temperature=gen.temperature,
        top_p=gen.top_p,
        max_tokens=gen.num_predict,
        stream=gen.stream,
        extra_params={
            "repeat_penalty": gen.repeat_penalty,
            "progress_chars": gen.progress_chars,
        },
    )

    response = provider.generate_text(prompt, config)
    return response.content


def default_progress_printer(chars: int, elapsed: float) -> None:
    print(f"  ... {chars:,} chars generated ({elapsed:.0f}s elapsed)", flush=True)


def print_stream_header(model: str, prompt_chars: int, max_tokens: int) -> None:
    print(
        f"Generating LaTeX with {model} "
        f"(prompt ~{prompt_chars:,} chars, up to {max_tokens:,} tokens).",
        flush=True,
    )
    print(
        "Large local models can take 5-20 minutes. Progress updates below:",
        flush=True,
    )


__all__ = [
    "LLMError",
    "call_llm_json",
    "call_llm_text",
    "default_progress_printer",
    "print_stream_header",
]