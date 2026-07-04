from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from resume_pipeline.ats.models import ATSResult
from resume_pipeline.llm.interfaces import LLMConfig, LLMProvider
from resume_pipeline.resume.prompts import build_resume_prompt
from resume_pipeline.config import load_settings


@dataclass
class ATSServiceConfig:
    model: str = "deepseek-coder:6.7b"
    temperature: float = 0.1
    max_tokens: int = 4000


@dataclass
class ResumeServiceConfig:
    model: str = "qwen2.5-coder:14b"
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 4000
    stream: bool = True
    progress_chars: int = 200
    repeat_penalty: float = 1.05


class ATSService:
    def __init__(
        self,
        llm_provider: LLMProvider,
        config: ATSServiceConfig | None = None,
    ):
        self._llm_provider = llm_provider
        self._config = config or ATSServiceConfig()

    def analyze(self, jd: str, profile: str) -> ATSResult:
        from resume_pipeline.ats.prompts import build_ats_prompt
        from resume_pipeline.ats.scoring import compute_score
        from resume_pipeline.ats.models import LLMEvidence
        from resume_pipeline.ats.fallback import fallback_ats
        from resume_pipeline.llm.interfaces import LLMError
        import logging

        logger = logging.getLogger(__name__)

        if not jd or not jd.strip():
            raise ValueError("Job description must not be empty.")
        if not profile or not profile.strip():
            raise ValueError("Candidate profile must not be empty.")

        prompt = build_ats_prompt(jd, profile)
        llm_config = LLMConfig(
            model=self._config.model,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            stream=False,
        )

        try:
            raw = self._llm_provider.generate_json(prompt, llm_config)
            evidence = LLMEvidence.model_validate(raw)
            settings = load_settings()
            score, breakdown = compute_score(evidence, settings.scoring)

            return ATSResult(
                score=score,
                score_breakdown=breakdown,
                matched_skills=evidence.matched_skills,
                missing_skills=evidence.missing_skills,
                transferable_skills=evidence.transferable_skills,
                matched_tools=evidence.matched_tools,
                missing_tools=evidence.missing_tools,
                experience_match=evidence.experience_match,
                impact_match=evidence.impact_match,
                key_gaps=evidence.key_gaps[:10],
                summary=evidence.summary,
                source="llm",
            )
        except (LLMError, Exception) as exc:
            logger.warning(
                "LLM ATS failed (%s: %s) — switching to fallback.",
                type(exc).__name__,
                exc,
            )
            return fallback_ats(jd, profile)


class ResumeGenerationService:
    def __init__(
        self,
        llm_provider: LLMProvider,
        config: ResumeServiceConfig | None = None,
    ):
        self._llm_provider = llm_provider
        self._config = config or ResumeServiceConfig()

    def generate(
        self,
        jd: str,
        profile: str,
        analysis: ATSResult,
        latex_template: str,
        on_progress: Callable[[int, float], None] | None = None,
    ) -> str:
        prompt = build_resume_prompt(jd, profile, analysis, latex_template)

        llm_config = LLMConfig(
            model=self._config.model,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            max_tokens=self._config.max_tokens,
            stream=self._config.stream,
            extra_params={
                "repeat_penalty": self._config.repeat_penalty,
                "progress_chars": self._config.progress_chars,
            },
        )

        response = self._llm_provider.generate_text(prompt, llm_config)
        return response.content


class ResumePipelineService:
    def __init__(
        self,
        ats_service: ATSService,
        resume_service: ResumeGenerationService,
    ):
        self._ats_service = ats_service
        self._resume_service = resume_service

    def run_analysis(
        self,
        jd: str,
        profile: str,
        latex_template: str,
        on_progress: Callable[[int, float], None] | None = None,
    ) -> tuple[ATSResult, str]:
        analysis = self._ats_service.analyze(jd, profile)
        latex_resume = self._resume_service.generate(
            jd, profile, analysis, latex_template, on_progress
        )
        return analysis, latex_resume