from enum import Enum

from pydantic import BaseModel, Field, field_validator


class MatchLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class LLMEvidence(BaseModel):
    """Raw evidence extracted by the LLM — Python computes the final score."""

    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    transferable_skills: list[str] = Field(default_factory=list)

    experience_match: MatchLevel = MatchLevel.UNKNOWN
    experience_notes: str = ""

    matched_tools: list[str] = Field(default_factory=list)
    missing_tools: list[str] = Field(default_factory=list)

    impact_match: MatchLevel = MatchLevel.UNKNOWN
    impact_notes: str = ""

    key_gaps: list[str] = Field(default_factory=list)
    summary: str = ""


class ATSResult(BaseModel):
    """Final ATS result with a mathematically computed score."""

    score: int = Field(ge=0, le=100)
    score_breakdown: dict[str, float]

    matched_skills: list[str]
    missing_skills: list[str]
    transferable_skills: list[str]
    matched_tools: list[str]
    missing_tools: list[str]

    experience_match: MatchLevel
    impact_match: MatchLevel

    key_gaps: list[str]
    summary: str

    source: str = Field(default="llm")

    @field_validator("score")
    @classmethod
    def clamp_score(cls, value: int) -> int:
        return max(0, min(100, value))
