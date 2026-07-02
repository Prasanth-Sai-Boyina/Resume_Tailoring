from resume_pipeline.ats.models import ATSResult


def build_resume_prompt(
    jd: str,
    profile: str,
    analysis: ATSResult,
    latex_template: str,
) -> str:
    prompt = f"""
You are an expert resume optimization AI specializing in FAANG-level resumes.

==================== INPUTS ====================

1. Job Description:
{jd}

2. Candidate Master Profile:
{profile}

3. ATS Analysis:
{analysis.model_dump_json(indent=2)}

4. LaTeX Resume Template (STRICTLY FOLLOW THIS FORMAT):
{latex_template}

==================== TASK ====================

Your goal is to generate a highly optimized, ATS-friendly resume tailored to the job description.

Follow these rules STRICTLY:

1. TEMPLATE USAGE (VERY IMPORTANT)
- You MUST use the provided LaTeX template exactly
- DO NOT change structure, margins, or formatting
- ONLY replace content inside sections
- Keep LaTeX syntax valid and compilable

2. CONTENT OPTIMIZATION
- Tailor the resume specifically for this job description
- Maximize overlap with ATS keywords
- Prioritize matched skills and relevant experience
- Incorporate missing skills ONLY if they are realistically inferable (DO NOT lie)

3. BULLET POINT IMPROVEMENT
- Use strong action verbs (Built, Designed, Optimized, Scaled, Implemented)
- Quantify impact wherever possible (%, latency, scale, revenue, users)
- Focus on outcomes, not just responsibilities

4. EXPERIENCE FILTERING
- Prioritize backend, AI, and full-stack experience
- De-emphasize irrelevant work
- Keep only high-impact contributions

5. LENGTH & QUALITY
- Must fit into 1 or 2 pages
- Avoid redundancy
- Keep it crisp and high-signal (FAANG standard)

6. OUTPUT FORMAT
- Return ONLY the final LaTeX code
- No explanations, no comments
- Ensure it compiles without errors

==================== OUTPUT ====================

Generate the final LaTeX resume.
"""

    return prompt.strip()
