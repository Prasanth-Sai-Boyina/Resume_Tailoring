from __future__ import annotations

from resume_pipeline.config import LLMConfig as ConfigLLMConfig
from resume_pipeline.llm.interfaces import LLMConfig, LLMProvider
from resume_pipeline.llm.mock import MockLLMProvider
from resume_pipeline.llm.ollama import OllamaProvider
from resume_pipeline.llm.openai import OpenAIProvider


class LLMProviderFactory:
    _providers: dict[str, LLMProvider] = {}

    @classmethod
    def create_ollama(
        cls,
        base_url: str = "http://localhost:11434",
        default_model: str = "qwen2.5-coder:14b",
        request_timeout: int = 900,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
    ) -> OllamaProvider:
        return OllamaProvider(
            base_url=base_url,
            default_model=default_model,
            request_timeout=request_timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )

    @classmethod
    def create_openai(
        cls,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "gpt-4o-mini",
        request_timeout: int = 600,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
    ) -> OpenAIProvider:
        return OpenAIProvider(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            request_timeout=request_timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )

    @classmethod
    def create_mock(cls, default_model: str = "mock-model") -> MockLLMProvider:
        return MockLLMProvider(default_model=default_model)

    @classmethod
    def from_config(cls, config: ConfigLLMConfig) -> LLMProvider:
        if config.provider == "ollama":
            return cls.create_ollama(
                base_url=config.ollama_url,
                default_model=config.resume_model,
                max_retries=config.max_retries,
                retry_backoff=config.retry_backoff,
                request_timeout=config.request_timeout,
            )
        elif config.provider == "openai":
            return cls.create_openai(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
                default_model=config.openai_model,
                request_timeout=config.request_timeout,
                max_retries=config.max_retries,
                retry_backoff=config.retry_backoff,
            )
        elif config.provider == "mock":
            return cls.create_mock(default_model=config.resume_model)
        else:
            raise ValueError(f"Unknown LLM provider: {config.provider}")

    @classmethod
    def register_provider(cls, name: str, provider: LLMProvider) -> None:
        cls._providers[name] = provider

    @classmethod
    def get_provider(cls, name: str) -> LLMProvider | None:
        return cls._providers.get(name)


def create_llm_config_from_settings(settings) -> LLMConfig:
    """Create LLMConfig from settings.llm.resume_generation"""
    return LLMConfig(
        model=settings.llm.resume_model,
        temperature=settings.llm.resume_generation.temperature,
        top_p=settings.llm.resume_generation.top_p,
        max_tokens=settings.llm.resume_generation.num_predict,
        stream=settings.llm.resume_generation.stream,
        extra_params={
            "repeat_penalty": settings.llm.resume_generation.repeat_penalty,
            "progress_chars": settings.llm.resume_generation.progress_chars,
            "think": settings.llm.resume_generation.think,
        },
    )