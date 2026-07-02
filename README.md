# Resume Pipeline

An LLM-powered pipeline for job application prep: score candidate fit against a job description, generate an optimized LaTeX resume, and compile it to PDF.

## What it does

1. **ATS analysis** — Compares a job description to your master profile using a local Ollama model. The LLM extracts structured evidence; Python computes the weighted score.
2. **Resume generation** — Builds a tailored prompt from the ATS analysis and calls a larger model to produce optimized LaTeX.
3. **Resume scoring** — Converts an existing resume to markdown and scores it against the JD.
4. **PDF compilation** — Compiles generated LaTeX to PDF via `pdflatex`.

## Project layout

```
jobs/
├── config.yaml              # Central configuration (paths, models, scoring weights)
├── main.py                  # CLI entry point
├── data/                    # Input files (JD, profile, resume template)
├── output/                  # Generated artifacts (gitignored)
└── resume_pipeline/         # Python package
    ├── cli.py               # Command-line interface
    ├── config.py            # Settings loader (YAML + env vars)
    ├── io.py                # File read/write helpers
    ├── ats/                 # ATS scoring engine
    │   ├── engine.py        # Orchestration
    │   ├── scoring.py       # Weighted score computation
    │   ├── fallback.py      # Keyword fallback when LLM fails
    │   ├── prompts.py       # ATS extraction prompt
    │   └── models.py        # Pydantic models
    ├── llm/                 # Ollama client (JSON + text modes)
    ├── resume/              # Prompt building + LaTeX generation
    ├── convert/             # LaTeX ↔ markdown/PDF utilities
    └── utils/               # Shared text helpers
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com/) and pull the models configured in `config.yaml`:

```bash
ollama pull deepseek-coder:6.7b
ollama pull qwen3:32b
```

Place your inputs in `data/`:

| File | Purpose |
|------|---------|
| `jd.txt` | Job description |
| `profile.txt` | Master candidate profile |
| `resume.tex` | Current resume (for scoring) |
| `format.tex` | LaTeX template for generation |

## Usage

```bash
# Full pipeline: ATS analysis → prompt → LaTeX generation
python main.py analyze

# Score existing resume against JD
python main.py score

# Compile LaTeX to PDF
python main.py pdf output/output_resume.tex
python latex_to_pdf.py output/output_resume.tex
```

## Configuration

Edit `config.yaml` to change file paths, model names, scoring weights, and LaTeX settings. Environment variables override LLM settings:

| Variable | Overrides |
|----------|-----------|
| `OLLAMA_URL` | Ollama API endpoint |
| `OLLAMA_MODEL` | ATS model |
| `QWEN_MODEL` | Resume generation model |
| `ATS_MAX_RETRIES` | Retry count |
| `ATS_REQUEST_TIMEOUT` | Request timeout (seconds) |

Use a separate config file for machine-specific overrides:

```bash
python main.py analyze --config config.local.yaml
```

## Scoring weights

Default weights (configurable in `config.yaml`):

| Component | Weight |
|-----------|--------|
| Skills | 40% |
| Experience | 30% |
| Tools | 20% |
| Impact | 10% |
