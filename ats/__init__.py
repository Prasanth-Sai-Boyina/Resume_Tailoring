"""Backward-compatible import path for legacy scripts."""

from resume_pipeline.ats import ATSResult, LLMEvidence, MatchLevel, ats_match

__all__ = ["ATSResult", "LLMEvidence", "MatchLevel", "ats_match"]
