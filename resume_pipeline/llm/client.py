import json
import logging
import time
from collections.abc import Callable
from typing import Any

import requests

from resume_pipeline.config import Settings, load_settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM call fails after all retries."""


def _post_ollama(
    payload: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    last_exc: Exception | None = None

    for attempt in range(1, settings.llm.max_retries + 1):
        try:
            logger.debug("LLM attempt %d/%d", attempt, settings.llm.max_retries)
            response = requests.post(
                settings.llm.ollama_url,
                json=payload,
                timeout=settings.llm.request_timeout,
            )

            if response.status_code != 200:
                raise LLMError(f"HTTP {response.status_code}: {response.text[:200]}")

            body = response.json()
            logger.debug("LLM call succeeded on attempt %d", attempt)
            return body

        except (requests.RequestException, LLMError) as exc:
            last_exc = exc
            if attempt < settings.llm.max_retries:
                wait = settings.llm.retry_backoff ** attempt
                logger.warning(
                    "LLM attempt %d failed (%s). Retrying in %.1fs...",
                    attempt,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "LLM failed after %d attempts: %s",
                    settings.llm.max_retries,
                    exc,
                )

    raise LLMError(f"LLM unavailable after {settings.llm.max_retries} attempts: {last_exc}")


def _stream_ollama_text(
    payload: dict[str, Any],
    settings: Settings,
    on_progress: Callable[[int, float], None] | None = None,
) -> str:
    payload = {**payload, "stream": True}
    parts: list[str] = []
    started = time.monotonic()
    last_reported = 0
    progress_chars = settings.llm.resume_generation.progress_chars

    try:
        with requests.post(
            settings.llm.ollama_url,
            json=payload,
            stream=True,
            timeout=(30, settings.llm.request_timeout),
        ) as response:
            if response.status_code != 200:
                raise LLMError(f"HTTP {response.status_code}: {response.text[:200]}")

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    parts.append(token)
                    total_chars = sum(len(part) for part in parts)
                    if on_progress and total_chars - last_reported >= progress_chars:
                        on_progress(total_chars, time.monotonic() - started)
                        last_reported = total_chars

                if chunk.get("done"):
                    break

    except (requests.RequestException, json.JSONDecodeError) as exc:
        raise LLMError(f"Streaming LLM call failed: {exc}") from exc

    text = "".join(parts).strip()
    if not text:
        raise LLMError("LLM returned empty text response")

    if on_progress:
        on_progress(len(text), time.monotonic() - started)

    return text


def call_llm_json(
    prompt: str,
    model: str,
    settings: Settings | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()

    request_payload = payload or {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    body = _post_ollama(request_payload, settings)
    raw = body.get("response", "")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"LLM returned invalid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise LLMError("LLM returned non-dict JSON")

    return parsed


def call_llm_text(
    prompt: str,
    model: str,
    settings: Settings | None = None,
    payload: dict[str, Any] | None = None,
    on_progress: Callable[[int, float], None] | None = None,
) -> str:
    settings = settings or load_settings()
    gen = settings.llm.resume_generation

    request_payload = payload or {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "num_predict": gen.num_predict,
            "repeat_penalty": gen.repeat_penalty,
        },
    }

    if not gen.think:
        request_payload["think"] = False

    if gen.stream:
        logger.info(
            "Streaming from %s (max %d tokens; large models can take several minutes)...",
            model,
            gen.num_predict,
        )
        return _stream_ollama_text(request_payload, settings, on_progress)

    body = _post_ollama(request_payload, settings)
    response_text = body.get("response", "")

    if not response_text.strip():
        raise LLMError("LLM returned empty text response")

    return response_text


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
