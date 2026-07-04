from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from resume_pipeline.config import LaTeXConfig


def check_engine_available(engine: str) -> bool:
    return shutil.which(engine) is not None


def run_command(cmd: list[str], cwd: Path) -> tuple[int, str]:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.returncode, result.stdout


def compile_latex(
    tex_path: Path,
    latex_config: LaTeXConfig | None = None,
) -> Path:
    latex_config = latex_config or LaTeXConfig()
    engine = latex_config.engine
    use_bibtex = latex_config.use_bibtex
    passes = latex_config.passes

    cwd = tex_path.parent
    basename = tex_path.stem

    if not check_engine_available(engine):
        raise RuntimeError(
            f"Error: '{engine}' not found on PATH. "
            "Install a LaTeX distribution (TeX Live / MiKTeX / MacTeX) first."
        )

    compile_cmd = [engine, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]

    code, output = run_command(compile_cmd, cwd)
    if code != 0:
        raise RuntimeError(f"{engine} failed on the first pass.\n{output}")

    if use_bibtex:
        if not check_engine_available("bibtex"):
            raise RuntimeError("Error: 'bibtex' not found on PATH.")
        code, output = run_command(["bibtex", basename], cwd)
        if code != 0:
            raise RuntimeError(f"bibtex failed.\n{output}")

    remaining = passes - 1 + (1 if use_bibtex else 0)
    for _ in range(remaining):
        code, output = run_command(compile_cmd, cwd)
        if code != 0:
            raise RuntimeError(f"{engine} failed on a later pass.\n{output}")

    pdf_path = cwd / f"{basename}.pdf"
    if not pdf_path.exists():
        raise RuntimeError("Compilation finished but no PDF was found.")

    return pdf_path


def compile_latex_cli(tex_file: str, latex_config: LaTeXConfig | None = None) -> None:
    tex_path = Path(tex_file).resolve()
    if not tex_path.exists():
        sys.exit(f"Error: file not found: {tex_path}")
    if tex_path.suffix.lower() != ".tex":
        sys.exit("Error: input file must have a .tex extension")

    pdf_path = compile_latex(tex_path, latex_config)
    print(f"\nSuccess! PDF created at: {pdf_path}")
