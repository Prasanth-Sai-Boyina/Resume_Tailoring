from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class InputPaths:
    job_description: str = "jd.txt"
    profile: str = "profile.txt"
    resume: str = "resume.tex"
    latex_template: str = "format.tex"


@dataclass
class OutputPaths:
    analysis: str = "analysis.json"
    resume_analysis: str = "resume_analysis.json"
    final_prompt: str = "final_prompt.txt"
    generated_resume: str = "output_resume.tex"
    resume_markdown: str = "resume.md"


@dataclass
class PathConfig:
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output")
    inputs: InputPaths = field(default_factory=InputPaths)
    outputs: OutputPaths = field(default_factory=OutputPaths)

    def input_path(self, name: str) -> Path:
        filename = getattr(self.inputs, name)
        return self.data_dir / filename

    def output_path(self, name: str) -> Path:
        filename = getattr(self.outputs, name)
        return self.output_dir / filename


@dataclass
class ResumeGenerationConfig:
    temperature: float = 0.1
    top_p: float = 0.9
    num_predict: int = 4000
    repeat_penalty: float = 1.05
    stream: bool = True
    think: bool = False
    progress_chars: int = 200


@dataclass
class LLMConfig:
    provider: str = "ollama"
    ollama_url: str = "http://localhost:11434/api/generate"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    ats_model: str = "deepseek-coder:6.7b"
    resume_model: str = "qwen2.5-coder:14b"
    max_retries: int = 3
    retry_backoff: float = 1.5
    request_timeout: int = 900
    resume_generation: ResumeGenerationConfig = field(default_factory=ResumeGenerationConfig)


@dataclass
class ScoringConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "skills": 0.40,
            "experience": 0.30,
            "tools": 0.20,
            "impact": 0.10,
        }
    )
    match_levels: dict[str, float] = field(
        default_factory=lambda: {
            "high": 100.0,
            "medium": 60.0,
            "low": 20.0,
            "unknown": 40.0,
        }
    )


@dataclass
class LaTeXConfig:
    engine: str = "pdflatex"
    passes: int = 2
    use_bibtex: bool = False


@dataclass
class Settings:
    paths: PathConfig = field(default_factory=PathConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    latex: LaTeXConfig = field(default_factory=LaTeXConfig)


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _apply_env_overrides(settings: Settings) -> Settings:
    settings.llm.ollama_url = os.getenv("OLLAMA_URL", settings.llm.ollama_url)
    settings.llm.ats_model = os.getenv("OLLAMA_MODEL", settings.llm.ats_model)
    settings.llm.resume_model = os.getenv("QWEN_MODEL", settings.llm.resume_model)
    settings.llm.max_retries = int(os.getenv("ATS_MAX_RETRIES", str(settings.llm.max_retries)))
    settings.llm.retry_backoff = float(
        os.getenv("ATS_RETRY_BACKOFF", str(settings.llm.retry_backoff))
    )
    settings.llm.request_timeout = int(
        os.getenv("ATS_REQUEST_TIMEOUT", str(settings.llm.request_timeout))
    )
    return settings


def _build_settings(raw: dict[str, Any]) -> Settings:
    paths_raw = raw.get("paths", {})
    inputs_raw = paths_raw.get("inputs", {})
    outputs_raw = paths_raw.get("outputs", {})
    llm_raw = raw.get("llm", {})
    resume_gen_raw = llm_raw.get("resume_generation", {})
    scoring_raw = raw.get("scoring", {})
    latex_raw = raw.get("latex", {})

    data_dir = _resolve_path(PROJECT_ROOT, paths_raw.get("data_dir", "data"))
    output_dir = _resolve_path(PROJECT_ROOT, paths_raw.get("output_dir", "output"))

    settings = Settings(
        paths=PathConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            inputs=InputPaths(**inputs_raw),
            outputs=OutputPaths(**outputs_raw),
        ),
        llm=LLMConfig(
            provider=llm_raw.get("provider", LLMConfig.provider),
            ollama_url=llm_raw.get("ollama_url", LLMConfig.ollama_url),
            openai_api_key=llm_raw.get("openai_api_key", LLMConfig.openai_api_key),
            openai_base_url=llm_raw.get("openai_base_url", LLMConfig.openai_base_url),
            openai_model=llm_raw.get("openai_model", LLMConfig.openai_model),
            ats_model=llm_raw.get("ats_model", LLMConfig.ats_model),
            resume_model=llm_raw.get("resume_model", LLMConfig.resume_model),
            max_retries=llm_raw.get("max_retries", LLMConfig.max_retries),
            retry_backoff=llm_raw.get("retry_backoff", LLMConfig.retry_backoff),
            request_timeout=llm_raw.get("request_timeout", LLMConfig.request_timeout),
            resume_generation=ResumeGenerationConfig(**resume_gen_raw),
        ),
        scoring=ScoringConfig(
            weights=scoring_raw.get("weights", ScoringConfig().weights),
            match_levels=scoring_raw.get("match_levels", ScoringConfig().match_levels),
        ),
        latex=LaTeXConfig(**latex_raw),
    )
    return _apply_env_overrides(settings)


def load_settings(config_path: Path | str | None = None) -> Settings:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return _apply_env_overrides(Settings())

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return _build_settings(raw)
