#!/usr/bin/env python3
"""Compile a LaTeX file to PDF."""

import argparse
import sys
from pathlib import Path

from resume_pipeline.config import load_settings
from resume_pipeline.convert.latex_pdf import compile_latex_cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile a .tex file to PDF.")
    parser.add_argument("tex_file", help="Path to the .tex file")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--engine",
        default=None,
        choices=["pdflatex", "xelatex", "lualatex"],
        help="LaTeX engine to use",
    )
    parser.add_argument(
        "--bibtex",
        action="store_true",
        help="Run bibtex for bibliography support",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=None,
        help="Number of compiler passes",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    if args.engine:
        settings.latex.engine = args.engine
    if args.bibtex:
        settings.latex.use_bibtex = True
    if args.passes is not None:
        settings.latex.passes = args.passes

    try:
        compile_latex_cli(args.tex_file, settings.latex)
    except RuntimeError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
