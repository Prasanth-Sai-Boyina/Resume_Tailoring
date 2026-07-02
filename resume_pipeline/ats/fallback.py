import logging

from resume_pipeline.ats.models import ATSResult, MatchLevel
from resume_pipeline.utils.text import extract_keywords

logger = logging.getLogger(__name__)


def fallback_ats(jd: str, profile: str) -> ATSResult:
    logger.info("Running keyword-based fallback ATS.")
    jd_kw = extract_keywords(jd)
    profile_kw = extract_keywords(profile)

    if not jd_kw:
        logger.warning("JD yielded no keywords — returning zero score.")
        return ATSResult(
            score=0,
            score_breakdown={"skills": 0, "experience": 0, "tools": 0, "impact": 0},
            matched_skills=[],
            missing_skills=[],
            transferable_skills=[],
            matched_tools=[],
            missing_tools=[],
            experience_match=MatchLevel.UNKNOWN,
            impact_match=MatchLevel.UNKNOWN,
            key_gaps=[],
            summary="No keywords could be extracted from the job description.",
            source="fallback",
        )

    matched = sorted(jd_kw & profile_kw)
    missing = sorted(jd_kw - profile_kw)
    raw_score = int((len(matched) / len(jd_kw)) * 100)

    return ATSResult(
        score=raw_score,
        score_breakdown={
            "skills": float(raw_score),
            "experience": 40.0,
            "tools": 40.0,
            "impact": 40.0,
        },
        matched_skills=matched,
        missing_skills=missing,
        transferable_skills=[],
        matched_tools=[],
        missing_tools=[],
        experience_match=MatchLevel.UNKNOWN,
        impact_match=MatchLevel.UNKNOWN,
        key_gaps=missing[:10],
        summary=(
            f"Fallback keyword analysis: {len(matched)} of {len(jd_kw)} "
            f"JD keywords matched. LLM was unavailable."
        ),
        source="fallback",
    )
