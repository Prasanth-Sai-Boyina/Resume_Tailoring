from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from resume_pipeline.llm.interfaces import LLMConfig, LLMError, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "qwen2.5-coder:14b",
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        request_timeout: int = 900,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_url = f"{self._base_url}/api/generate"
        self._default_model = default_model
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._request_timeout = request_timeout

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def default_model(self) -> str:
        return self._default_model

    def generate_text(self, prompt: str, config: LLMConfig) -> LLMResponse:
        payload = self._build_payload(prompt, config, json_mode=False)

        if config.stream:
            content = self._stream_text(payload, config)
        else:
            content = self._post_text(payload)

        return LLMResponse(content=content, model=config.model)

    def generate_json(self, prompt: str, config: LLMConfig) -> dict:
        payload = self._build_payload(prompt, config, json_mode=True)
        body = self._post_ollama(payload)
        raw = body.get("response", "")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM returned invalid JSON: {exc}", self.name) from exc

        if not isinstance(parsed, dict):
            raise LLMError("LLM returned non-dict JSON", self.name)

        return parsed

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self._base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _build_payload(self, prompt: str, config: LLMConfig, json_mode: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": config.model,
            "prompt": prompt,
            "stream": config.stream,
        }

        if json_mode:
            payload["format"] = "json"
        else:
            options = {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "num_predict": config.max_tokens,
            }
            if config.extra_params:
                options.update(config.extra_params)
            payload["options"] = options

        return payload

    def _post_ollama(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug("Ollama attempt %d/%d", attempt, self._max_retries)
                response = requests.post(
                    self._api_url,
                    json=payload,
                    timeout=self._request_timeout,
                )

                if response.status_code != 200:
                    raise LLMError(
                        f"HTTP {response.status_code}: {response.text[:200]}",
                        self.name,
                        response.status_code,
                    )

                body = response.json()
                logger.debug("Ollama call succeeded on attempt %d", attempt)
                return body

            except (requests.RequestException, LLMError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    wait = self._retry_backoff ** attempt
                    logger.warning(
                        "Ollama attempt %d failed (%s). Retrying in %.1fs...",
                        attempt,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Ollama failed after %d attempts: %s",
                        self._max_retries,
                        exc,
                    )

        raise LLMError(
            f"Ollama unavailable after {self._max_retries} attempts: {last_exc}",
            self.name,
        )

    def _post_text(self, payload: dict[str, Any]) -> str:
        body = self._post_ollama(payload)
        response_text = body.get("response", "")

        if not response_text.strip():
            raise LLMError("LLM returned empty text response", self.name)

        return response_text

    def _stream_text(self, payload: dict[str, Any], config: LLMConfig) -> str:
        payload = {**payload, "stream": True}
        parts: list[str] = []
        started = time.monotonic()
        last_reported = 0
        progress_chars = config.extra_params.get("progress_chars", 200) if config.extra_params else 200

        try:
            with requests.post(
                self._api_url,
                json=payload,
                stream=True,
                timeout=(30, self._request_timeout),
            ) as response:
                if response.status_code != 200:
                    raise LLMError(
                        f"HTTP {response.status_code}: {response.text[:200]}",
                        self.name,
                        response.status_code,
                    )

                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue

                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        parts.append(token)
                        total_chars = sum(len(part) for part in parts)
                        if total_chars - last_reported >= progress_chars:
                            logger.debug("Generated %d chars so far", total_chars)
                            last_reported = total_chars

                    if chunk.get("done"):
                        break

        except (requests.RequestException, json.JSONDecodeError) as exc:
            raise LLMError(f"Streaming LLM call failed: {exc}", self.name) from exc

        text = "".join(parts).strip()
        if not text:
            raise LLMError("LLM returned empty text response", self.name)

        return text