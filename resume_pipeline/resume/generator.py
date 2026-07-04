from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from resume_pipeline.config import Settings, load_settings
from resume_pipeline.llm.factory import LLMProviderFactory
from resume_pipeline.llm.interfaces import LLMConfig, LLMProvider
from resume_pipeline.io import read_text, write_text


def generate_latex(
    prompt: str,
    settings: Settings | None = None,
    on_progress: Callable[[int, float], None] | None = None,
    llm_provider: LLMProvider | None = None,
) -> str:
    settings = settings or load_settings()
    provider = llm_provider or LLMProviderFactory.from_config(settings.llm)
    gen = settings.llm.resume_generation

    config = LLMConfig(
        model=settings.llm.resume_model,
        temperature=gen.temperature,
        top_p=gen.top_p,
        max_tokens=gen.num_predict,
        stream=gen.stream,
        extra_params={
            "repeat_penalty": gen.repeat_penalty,
            "progress_chars": gen.progress_chars,
            "think": gen.think,
        },
    )

    response = provider.generate_text(prompt, config)
    return response.content


def generate_resume_from_prompt(
    prompt_path: Path,
    output_path: Path,
    settings: Settings | None = None,
    on_progress: Callable[[int, float], None] | None = None,
    llm_provider: LLMProvider | None = None,
) -> Path:
    settings = settings or load_settings()
    prompt = read_text(prompt_path)
    latex_output = generate_latex(prompt, settings, on_progress=on_progress, llm_provider=llm_provider)
    write_text(output_path, latex_output)
    return output_path