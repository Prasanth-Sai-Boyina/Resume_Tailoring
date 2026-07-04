from __future__ import annotations

from resume_pipeline.llm.interfaces import LLMConfig, LLMError, LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing purposes."""

    def __init__(self, default_model: str = "mock-model"):
        self._default_model = default_model
        self._responses: dict[str, str] = {}
        self._json_responses: dict[str, dict] = {}
        self._call_count = 0

    @property
    def name(self) -> str:
        return "mock"

    @property
    def default_model(self) -> str:
        return self._default_model

    def set_text_response(self, prompt_prefix: str, response: str) -> None:
        self._responses[prompt_prefix] = response

    def set_json_response(self, prompt_prefix: str, response: dict) -> None:
        self._json_responses[prompt_prefix] = response

    def get_call_count(self) -> int:
        return self._call_count

    def generate_text(self, prompt: str, config: LLMConfig) -> LLMResponse:
        self._call_count += 1
        for prefix, response in self._responses.items():
            if prompt.startswith(prefix):
                return LLMResponse(content=response, model=config.model, metadata={"provider": "mock"})
        return LLMResponse(content=f"Mock response for: {prompt[:50]}...", model=config.model, metadata={"provider": "mock"})

    def generate_json(self, prompt: str, config: LLMConfig) -> dict:
        self._call_count += 1
        for prefix, response in self._json_responses.items():
            if prompt.startswith(prefix):
                return response
        return {"mock": True, "response": "mock json response"}

    def is_available(self) -> bool:
        return True