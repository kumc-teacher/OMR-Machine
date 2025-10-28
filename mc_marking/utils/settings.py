"""Simple persistence layer for user-specific application settings."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

_SETTINGS_DIR = Path.home() / ".mc_marking"
_SETTINGS_FILE = _SETTINGS_DIR / "settings.json"


@dataclass
class AppSettings:
    """User-configurable options persisted between sessions."""

    poppler_path: str | None = None
    tesseract_path: str | None = None
    baseline_ink_density: float | None = None
    row_labels: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.row_labels is None:
            self.row_labels = []


def load_settings() -> AppSettings:
    """Load settings from disk, falling back to defaults when missing."""
    try:
        with _SETTINGS_FILE.open("r", encoding="utf-8") as handle:
            data: Dict[str, Any] = json.load(handle)
    except FileNotFoundError:
        return AppSettings()
    except json.JSONDecodeError:
        return AppSettings()
    return AppSettings(**{**AppSettings().__dict__, **data})


def save_settings(settings: AppSettings) -> None:
    """Persist settings to disk, creating folders as needed."""
    _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    with _SETTINGS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(asdict(settings), handle, indent=2)