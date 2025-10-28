"""Helpers for interpreting grid-style multiple-choice tables."""

from __future__ import annotations

from dataclasses import dataclass
import statistics
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from mc_marking.models.answer_sheet import CellResult, TableExtraction


@dataclass
class QuestionMarks:
    """Captures detected marks for a single question column."""

    question: int
    column_index: int
    choices: Dict[str, CellResult]
    marked: List[Tuple[str, CellResult]]


_DEFAULT_ROW_LABELS = ["A", "B", "C", "D"]
_MARK_THRESHOLD = 0.025
_DENSITY_MARGIN = 0.015
_MAX_FORWARD_JUMP = 3


def parse_choice_table(
    extraction: TableExtraction,
    cells: Sequence[CellResult],
    *,
    baseline_density: float | None = None,
    row_labels_override: Optional[List[str]] = None,
) -> List[QuestionMarks]:
    """Interpret a table into per-question mark data."""
    if extraction.row_count <= 0 or extraction.column_count <= 1:
        return []

    grid: List[List[Optional[CellResult]]] = [
        [None for _ in range(extraction.column_count)] for _ in range(extraction.row_count)
    ]
    for cell in cells:
        if 0 <= cell.row < extraction.row_count and 0 <= cell.column < extraction.column_count:
            grid[cell.row][cell.column] = cell

    column_numbers = _infer_column_question_numbers(grid)
    row_labels = _infer_row_labels(grid, row_labels_override=row_labels_override)

    questions: List[QuestionMarks] = []
    for col_index, question_number in column_numbers.items():
        if question_number is None:
            continue
        choices: Dict[str, CellResult] = {}
        marked: List[Tuple[str, CellResult]] = []
        for row_index, label in row_labels.items():
            cell = grid[row_index][col_index] if row_index < len(grid) else None
            if cell is None:
                continue
            choices[label] = cell
        marked = _select_marked_choices(choices, baseline_density=baseline_density)
        questions.append(QuestionMarks(question=question_number, column_index=col_index, choices=choices, marked=marked))

    questions.sort(key=lambda q: (q.question, q.column_index))
    return questions


def enumerate_question_marks(
    extractions: Sequence[TableExtraction],
    tables_cells: Sequence[Sequence[CellResult]],
    *,
    start_number: int = 1,
    baseline_density: float | None = None,
    row_labels_override: Optional[List[str]] = None,
) -> Iterator[Tuple[int, QuestionMarks]]:
    """Yield question data with sequential numbering across multiple tables."""

    current_number = start_number - 1
    for extraction, cells in zip(extractions, tables_cells):
        for question in parse_choice_table(
            extraction,
            cells,
            baseline_density=baseline_density,
            row_labels_override=row_labels_override,
        ):
            expected = current_number + 1
            candidate = question.question
            if candidate >= expected and candidate - expected <= _MAX_FORWARD_JUMP:
                assigned = candidate
            else:
                assigned = expected
            current_number = assigned
            yield assigned, question


def _infer_column_question_numbers(grid: List[List[Optional[CellResult]]]) -> Dict[int, Optional[int]]:
    if not grid:
        return {}
    column_count = len(grid[0])
    numbers: Dict[int, Optional[int]] = {}
    fallback_number: Optional[int] = None
    for col in range(column_count):
        if col == 0:
            numbers[col] = None
            continue
        header_cell = grid[0][col]
        header_text = header_cell.text if header_cell else ""
        parsed = _parse_question_number(header_text)
        if parsed is not None:
            numbers[col] = parsed
            fallback_number = parsed
        else:
            if fallback_number is None:
                previous = numbers.get(col - 1)
                fallback_number = (previous + 1) if isinstance(previous, int) else col
            else:
                fallback_number += 1
            numbers[col] = fallback_number
    return numbers


def _infer_row_labels(
    grid: List[List[Optional[CellResult]]],
    *,
    row_labels_override: Optional[List[str]] = None,
) -> Dict[int, str]:
    row_labels: Dict[int, str] = {}
    row_count = len(grid)
    overrides = [label.strip() for label in row_labels_override or [] if label.strip()]
    for row_index in range(1, row_count):
        override_label: Optional[str] = None
        if overrides and row_index - 1 < len(overrides):
            override_label = overrides[row_index - 1]
        if override_label:
            row_labels[row_index] = override_label
            continue
        header_cell = grid[row_index][0] if grid[row_index] else None
        header_text = header_cell.text if header_cell else ""
        parsed = _parse_choice_label(header_text)
        if parsed is None:
            fallback_index = row_index - 1
            parsed = _DEFAULT_ROW_LABELS[fallback_index] if fallback_index < len(_DEFAULT_ROW_LABELS) else f"Option{row_index}"
        row_labels[row_index] = parsed
    return row_labels


def _parse_question_number(text: str) -> Optional[int]:
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _parse_choice_label(text: str) -> Optional[str]:
    normalized = text.strip().upper()
    for char in normalized:
        if char.isalpha():
            return char
    return None


def _select_marked_choices(
    choices: Dict[str, CellResult],
    *,
    baseline_density: float | None = None,
) -> List[Tuple[str, CellResult]]:
    if not choices:
        return []

    items = list(choices.items())
    baseline = baseline_density or 0.0
    raw_densities = [max(cell.ink_density, 0.0) for _, cell in items]
    adjusted_densities = [max(density - baseline, 0.0) for density in raw_densities]

    if not any(adjusted_densities):
        adjusted_densities = raw_densities
        baseline = 0.0

    median_adjusted = statistics.median(adjusted_densities)
    adaptive_threshold = max(_MARK_THRESHOLD, median_adjusted + _DENSITY_MARGIN)
    absolute_threshold = baseline + adaptive_threshold

    marked: List[Tuple[str, CellResult]] = []
    for (label, cell), adjusted_density in zip(items, adjusted_densities):
        if (
            adjusted_density >= adaptive_threshold
            or cell.ink_density >= absolute_threshold
            or cell.has_mark(absolute_threshold)
        ):
            marked.append((label, cell))

    if marked:
        return marked

    best_index = max(range(len(adjusted_densities)), key=adjusted_densities.__getitem__)
    best_label, best_cell = items[best_index]
    best_adjusted = adjusted_densities[best_index]
    second_adjusted = max(
        (value for idx, value in enumerate(adjusted_densities) if idx != best_index),
        default=0.0,
    )
    gap = best_adjusted - second_adjusted
    if best_adjusted >= adaptive_threshold * 0.6 and gap >= _DENSITY_MARGIN * 0.5:
        if best_cell.ink_density >= baseline + _DENSITY_MARGIN:
            return [(best_label, best_cell)]
    if best_cell.ink_density >= absolute_threshold:
        return [(best_label, best_cell)]
    return []


def extract_textual_labels(question: QuestionMarks) -> List[str]:
    labels: List[str] = []
    for label, cell in question.choices.items():
        normalized = _normalize_choice_text(cell.text)
        if not normalized:
            continue
        first_alpha = _first_alpha(normalized)
        if first_alpha and first_alpha == label.upper():
            labels.append(label)
        elif normalized in {label.upper(), f"{label.upper()}.", f"({label.upper()})"}:
            labels.append(label)
        elif normalized in {"TRUE", "T"} and label.upper() in {"T", "TRUE"}:
            labels.append(label)
        elif normalized in {"FALSE", "F"} and label.upper() in {"F", "FALSE"}:
            labels.append(label)
    return sorted(set(labels))


def _normalize_choice_text(value: str) -> str:
    return value.strip().upper()


def _first_alpha(value: str) -> Optional[str]:
    for char in value:
        if char.isalpha():
            return char
    return None
