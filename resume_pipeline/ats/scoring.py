from resume_pipeline.ats.models import LLMEvidence, MatchLevel
from resume_pipeline.config import ScoringConfig


def match_level_score(level: MatchLevel, scoring: ScoringConfig) -> float:
    return scoring.match_levels.get(level.value, 40.0)


def skills_score(evidence: LLMEvidence) -> float:
    matched = len(evidence.matched_skills)
    missing = len(evidence.missing_skills)
    transferable = len(evidence.transferable_skills)
    total = matched + missing

    if total == 0:
        return 50.0

    base = matched / total
    transferable_bonus = min(transferable * 0.4, missing) / total if missing > 0 else 0
    return min((base + transferable_bonus) * 100, 100.0)


def tools_score(evidence: LLMEvidence) -> float:
    matched = len(evidence.matched_tools)
    missing = len(evidence.missing_tools)
    total = matched + missing

    if total == 0:
        return 50.0
    return (matched / total) * 100


def compute_score(
    evidence: LLMEvidence,
    scoring: ScoringConfig,
) -> tuple[int, dict[str, float]]:
    skills = skills_score(evidence)
    experience = match_level_score(evidence.experience_match, scoring)
    tools = tools_score(evidence)
    impact = match_level_score(evidence.impact_match, scoring)

    weights = scoring.weights
    weighted = (
        skills * weights["skills"]
        + experience * weights["experience"]
        + tools * weights["tools"]
        + impact * weights["impact"]
    )

    breakdown = {
        "skills": round(skills, 1),
        "experience": round(experience, 1),
        "tools": round(tools, 1),
        "impact": round(impact, 1),
    }

    return round(weighted), breakdown
