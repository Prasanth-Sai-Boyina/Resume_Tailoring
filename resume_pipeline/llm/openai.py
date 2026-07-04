from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from resume_pipeline.llm.interfaces import LLMConfig, LLMError, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        request_timeout: int = 600,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._request_timeout = request_timeout

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return self._default_model

    def generate_text(self, prompt: str, config: LLMConfig) -> LLMResponse:
        payload = self._build_payload(prompt, config, json_mode=False)
        response = self._post_with_retry(payload)

        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = response.get("usage")

        return LLMResponse(
            content=content,
            model=config.model,
            usage=usage,
            metadata={"provider": "openai"},
        )

    def generate_json(self, prompt: str, config: LLMConfig) -> dict:
        payload = self._build_payload(prompt, config, json_mode=True)
        response = self._post_with_retry(payload)

        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM returned invalid JSON: {exc}", self.name) from exc

        if not isinstance(parsed, dict):
            raise LLMError("LLM returned non-dict JSON", self.name)

        return parsed

    def is_available(self) -> bool:
        try:
            response = requests.get(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=10,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _build_payload(self, prompt: str, config: LLMConfig, json_mode: bool = False) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "stream": config.stream,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        if config.extra_params:
            payload.update(config.extra_params)

        return payload

    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug("OpenAI attempt %d/%d", attempt, self._max_retries)
                response = requests.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self._request_timeout,
                )

                if response.status_code != 200:
                    raise LLMError(
                        f"HTTP {response.status_code}: {response.text[:200]}",
                        self.name,
                        response.status_code,
                    )

                body = response.json()
                logger.debug("OpenAI call succeeded on attempt %d", attempt)
                return body

            except (requests.RequestException, LLMError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    wait = self._retry_backoff ** attempt
                    logger.warning(
                        "OpenAI attempt %d failed (%s). Retrying in %.1fs...",
                        attempt,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "OpenAI failed after %d attempts: %s",
                        self._max_retries,
                        exc,
                    )

        raise LLMError(
            f"OpenAI unavailable after {self._max_retries} attempts: {last_exc}",
            self.name,
        )