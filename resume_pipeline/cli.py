"""Unified CLI for the resume optimization pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from collections.abc import Callable

from resume_pipeline.config import Settings, load_settings
from resume_pipeline.convert.latex_markdown import latex_to_markdown
from resume_pipeline.convert.latex_pdf import compile_latex_cli
from resume_pipeline.io import ensure_output_dir, read_text, write_json, write_text
from resume_pipeline.resume.prompts import build_resume_prompt
from resume_pipeline.services import ATSService, ATSServiceConfig, ResumeGenerationService, ResumeServiceConfig
from resume_pipeline.llm.factory import LLMProviderFactory, create_llm_config_from_settings
from resume_pipeline.llm.interfaces import LLMConfig, LLMProvider


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _create_ats_service(settings: Settings) -> ATSService:
    llm_provider = LLMProviderFactory.from_config(settings.llm)
    return ATSService(
        llm_provider=llm_provider,
        config=ATSServiceConfig(
            model=settings.llm.ats_model,
            max_tokens=settings.llm.resume_generation.num_predict,
        ),
    )


def _create_resume_service(settings: Settings) -> ResumeGenerationService:
    llm_provider = LLMProviderFactory.from_config(settings.llm)
    return ResumeGenerationService(
        llm_provider=llm_provider,
        config=ResumeServiceConfig(
            model=settings.llm.resume_model,
            temperature=settings.llm.resume_generation.temperature,
            top_p=settings.llm.resume_generation.top_p,
            max_tokens=settings.llm.resume_generation.num_predict,
            stream=settings.llm.resume_generation.stream,
            progress_chars=settings.llm.resume_generation.progress_chars,
            repeat_penalty=settings.llm.resume_generation.repeat_penalty,
        ),
    )


def run_analysis(settings: Settings, ats_service: ATSService | None = None,
                 resume_service: ResumeGenerationService | None = None) -> None:
    ensure_output_dir(settings)

    ats_service = ats_service or _create_ats_service(settings)
    resume_service = resume_service or _create_resume_service(settings)

    jd = read_text(settings.paths.input_path("job_description"))
    profile = read_text(settings.paths.input_path("profile"))
    latex_format = read_text(settings.paths.input_path("latex_template"))

    print("Running ATS analysis...")
    analysis = ats_service.analyze(jd, profile)
    print("ATS Analysis completed.")

    analysis_path = settings.paths.output_path("analysis")
    write_json(analysis_path, analysis.model_dump())
    print(f"ATS Score: {analysis.score}%")
    print(f"Saved analysis to {analysis_path}")

    prompt = build_resume_prompt(jd, profile, analysis, latex_format)
    prompt_path = settings.paths.output_path("final_prompt")
    write_text(prompt_path, prompt)
    print(f"Prompt saved to {prompt_path}")

    model = settings.llm.resume_model
    max_tokens = settings.llm.resume_generation.num_predict
    print(
        f"Generating LaTeX with {model} "
        f"(prompt ~{len(prompt):,} chars, up to {max_tokens:,} tokens).",
        flush=True,
    )
    print(
        "Large local models can take 5-20 minutes. Progress updates below:",
        flush=True,
    )

    latex_output = resume_service.generate(jd, profile, analysis, latex_format)
    output_path = settings.paths.output_path("generated_resume")
    write_text(output_path, latex_output)
    print(f"LaTeX generation completed. Check {output_path}")


def run_score(settings: Settings, ats_service: ATSService | None = None) -> None:
    ensure_output_dir(settings)

    ats_service = ats_service or _create_ats_service(settings)

    jd = read_text(settings.paths.input_path("job_description"))
    resume_md_path = settings.paths.output_path("resume_markdown")

    print("Converting resume LaTeX to markdown...")
    resume = latex_to_markdown(
        str(settings.paths.input_path("resume")),
        str(resume_md_path),
    )

    print("Running Resume ATS Score...")
    resume_analysis = ats_service.analyze(jd, resume)

    output_path = settings.paths.output_path("resume_analysis")
    write_json(output_path, resume_analysis.model_dump())
    print(f"Resume ATS Score: {resume_analysis.score}%")
    print(f"Saved resume analysis to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume optimization pipeline: ATS scoring and LaTeX generation.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "analyze",
        help="Run ATS analysis, build prompt, and generate optimized LaTeX resume",
    )
    subparsers.add_parser(
        "score",
        help="Score an existing resume against the job description",
    )

    pdf_parser = subparsers.add_parser("pdf", help="Compile a .tex file to PDF")
    pdf_parser.add_argument("tex_file", help="Path to the .tex file")
    pdf_parser.add_argument(
        "--engine",
        default=None,
        choices=["pdflatex", "xelatex", "lualatex"],
        help="LaTeX engine to use",
    )
    pdf_parser.add_argument(
        "--bibtex",
        action="store_true",
        help="Run bibtex for bibliography support",
    )
    pdf_parser.add_argument(
        "--passes",
        type=int,
        default=None,
        help="Number of compiler passes",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    settings = load_settings(args.config)

    if args.command == "analyze":
        run_analysis(settings)
    elif args.command == "score":
        run_score(settings)
    elif args.command == "pdf":
        if args.engine:
            settings.latex.engine = args.engine
        if args.bibtex:
            settings.latex.use_bibtex = True
        if args.passes is not None:
            settings.latex.passes = args.passes
        compile_latex_cli(args.tex_file, settings.latex)


if __name__ == "__main__":
    main()