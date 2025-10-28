"""Dataclasses describing answer keys, responses, and scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from mc_marking.utils.image_utils import BoundingBox


@dataclass
class CellResult:
    """Captured text from a single table cell."""

    row: int
    column: int
    text: str
    confidence: float
    bounding_box: BoundingBox
    ink_density: float = 0.0

    def has_mark(self, threshold: float = 0.03) -> bool:
        """Heuristic to decide if the cell contains a filled mark."""
        normalized_text = self.text.strip()
        if normalized_text:
            mark_chars = {"√", "✓", "✔", "V", "v", "／", "/", "1", "●", "•"}
            if any(char in mark_chars for char in normalized_text):
                return True
        return self.ink_density >= threshold


@dataclass
class TableExtraction:
    """Raw table recognition output."""

    source_path: Path
    page_index: int
    bounding_box: Optional[BoundingBox]
    cells: List[CellResult] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0


@dataclass
class AnswerKey:
    """Normalized answer key extracted from a table."""

    rows: Dict[int, str]
    metadata: Dict[str, str] = field(default_factory=dict)

    def answer_for(self, question_number: int) -> Optional[str]:
        return self.rows.get(question_number)


@dataclass
class PageAnswer:
    """Per-question recognition results for a single page."""

    question: int
    extracted: str
    confidence: float
    is_correct: Optional[bool]


@dataclass
class PageResult:
    """Aggregated scoring data for a page."""

    source_path: Path
    page_index: int
    answers: List[PageAnswer] = field(default_factory=list)

    @property
    def correct_count(self) -> int:
        return sum(1 for answer in self.answers if answer.is_correct)

    @property
    def incorrect_count(self) -> int:
        return sum(1 for answer in self.answers if answer.is_correct is False)

    @property
    def unanswered_count(self) -> int:
        return sum(1 for answer in self.answers if answer.is_correct is None)

    @property
    def total_questions(self) -> int:
        return len(self.answers)
