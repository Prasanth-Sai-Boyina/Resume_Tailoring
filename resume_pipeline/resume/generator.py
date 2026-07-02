from collections.abc import Callable
from pathlib import Path

from resume_pipeline.config import Settings, load_settings
from resume_pipeline.io import read_text, write_text
from resume_pipeline.llm.client import call_llm_text, default_progress_printer


def generate_latex(
    prompt: str,
    settings: Settings | None = None,
    on_progress: Callable[[int, float], None] | None = default_progress_printer,
) -> str:
    settings = settings or load_settings()
    gen = settings.llm.resume_generation

    payload = {
        "model": settings.llm.resume_model,
        "prompt": prompt,
        "stream": gen.stream,
        "options": {
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "num_predict": gen.num_predict,
            "repeat_penalty": gen.repeat_penalty,
        },
    }

    if not gen.think:
        payload["think"] = False

    return call_llm_text(
        prompt,
        settings.llm.resume_model,
        settings,
        payload,
        on_progress=on_progress,
    )


def generate_resume_from_prompt(
    prompt_path: Path,
    output_path: Path,
    settings: Settings | None = None,
    on_progress: Callable[[int, float], None] | None = default_progress_printer,
) -> Path:
    settings = settings or load_settings()
    prompt = read_text(prompt_path)
    latex_output = generate_latex(prompt, settings, on_progress=on_progress)
    write_text(output_path, latex_output)
    return output_path
