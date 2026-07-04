"""Production ATS matching engine."""

from __future__ import annotations

import logging

from pydantic import ValidationError

from resume_pipeline.ats.fallback import fallback_ats
from resume_pipeline.ats.models import ATSResult, LLMEvidence
from resume_pipeline.ats.prompts import build_ats_prompt
from resume_pipeline.ats.scoring import compute_score
from resume_pipeline.config import Settings, load_settings
from resume_pipeline.llm.interfaces import LLMConfig, LLMError, LLMProvider
from resume_pipeline.llm.factory import LLMProviderFactory

logger = logging.getLogger(__name__)


def ats_match(
    jd: str,
    profile: str,
    settings: Settings | None = None,
    llm_provider: LLMProvider | None = None,
) -> ATSResult:
    """
    Evaluate how well a candidate profile matches a job description.

    Returns an ATSResult with a Python-computed weighted score and full evidence.
    Falls back to keyword scoring if the LLM is unavailable.
    """
    settings = settings or load_settings()

    if not jd or not jd.strip():
        raise ValueError("Job description must not be empty.")
    if not profile or not profile.strip():
        raise ValueError("Candidate profile must not be empty.")

    provider = llm_provider or LLMProviderFactory.from_config(settings.llm)
    prompt = build_ats_prompt(jd, profile)

    llm_config = LLMConfig(
        model=settings.llm.ats_model,
        temperature=0.1,
        max_tokens=settings.llm.resume_generation.num_predict,
        stream=False,
    )

    logger.info("Calling LLM for ATS evaluation.")

    try:
        raw = provider.generate_json(prompt, llm_config)
        logger.debug("Raw LLM output: %s", raw)

        evidence = LLMEvidence.model_validate(raw)
        score, breakdown = compute_score(evidence, settings.scoring)
        logger.info("ATS score computed: %d (breakdown: %s)", score, breakdown)

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

    except (LLMError, ValidationError) as exc:
        logger.warning(
            "LLM ATS failed (%s: %s) — switching to fallback.",
            type(exc).__name__,
            exc,
        )
        return fallback_ats(jd, profile)