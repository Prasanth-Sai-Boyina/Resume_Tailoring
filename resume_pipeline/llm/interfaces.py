from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 4000
    stream: bool = False
    extra_params: dict[str, Any] | None = None


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class LLMError(Exception):
    def __init__(self, message: str, provider: str, status_code: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class LLMProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        pass

    @abstractmethod
    def generate_text(self, prompt: str, config: LLMConfig) -> LLMResponse:
        pass

    @abstractmethod
    def generate_json(self, prompt: str, config: LLMConfig) -> dict:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass