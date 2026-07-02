from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from resume_pipeline.config import Settings


def ensure_output_dir(settings: Settings) -> None:
    settings.paths.output_dir.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
