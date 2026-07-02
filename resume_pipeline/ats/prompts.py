from resume_pipeline.utils.text import sanitize_for_prompt

ATS_PROMPT_TEMPLATE = """
You are an expert ATS (Applicant Tracking System) analyst.

Your task is to extract structured evidence from a job description and a candidate profile.
Do NOT compute a final score — only extract evidence. The score will be computed separately.

---

JOB DESCRIPTION:
{jd}

---

CANDIDATE PROFILE:
{profile}

---

INSTRUCTIONS:

1. matched_skills: Technical skills, languages, frameworks explicitly present in BOTH texts.
2. missing_skills: Skills required by the JD that are absent from the profile.
3. transferable_skills: Skills in the profile that are adjacent/related to JD requirements.
4. experience_match: "high" if the candidate's experience closely aligns with the role level and domain, "medium" if partially, "low" if weak.
5. experience_notes: One sentence explaining your experience_match rating.
6. matched_tools: Specific tools/platforms/services present in both texts (e.g. AWS, Docker, Jira).
7. missing_tools: Tools mentioned in the JD but absent from the profile.
8. impact_match: "high" if the profile demonstrates quantified achievements/impact, "medium" if some, "low" if none.
9. impact_notes: One sentence explaining your impact_match rating.
10. key_gaps: The top 5 most critical gaps between the JD and profile.
11. summary: 2-3 sentence plain-English summary of the overall fit.

CRITICAL RULES:
- Respond ONLY with a valid JSON object. No explanation, no markdown, no code fences.
- All list fields must be arrays of strings. Empty arrays [] are valid.
- experience_match and impact_match must be exactly one of: "high", "medium", "low".
- Do not invent skills not present in either text.

OUTPUT FORMAT:
{{
  "matched_skills": [],
  "missing_skills": [],
  "transferable_skills": [],
  "experience_match": "high | medium | low",
  "experience_notes": "",
  "matched_tools": [],
  "missing_tools": [],
  "impact_match": "high | medium | low",
  "impact_notes": "",
  "key_gaps": [],
  "summary": ""
}}
"""


def build_ats_prompt(jd: str, profile: str) -> str:
    return ATS_PROMPT_TEMPLATE.format(
        jd=sanitize_for_prompt(jd),
        profile=sanitize_for_prompt(profile),
    )
