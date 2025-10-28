"""Convert OCR outputs into PageResult structures."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from mc_marking.models.answer_sheet import CellResult, PageAnswer, PageResult, TableExtraction
from mc_marking.services.table_parser import enumerate_question_marks, extract_textual_labels


def build_page_result(
    extractions: Sequence[TableExtraction],
    tables_cells: Sequence[List[CellResult]],
    *,
    baseline_density: float | None = None,
    row_labels_override: Optional[List[str]] = None,
) -> PageResult:
    """Aggregate OCR results from one or more tables into a PageResult."""

    answers: List[PageAnswer] = []
    for question_number, question in enumerate_question_marks(
        extractions,
        tables_cells,
        baseline_density=baseline_density,
        row_labels_override=row_labels_override,
    ):
        labels = [label for label, _ in question.marked]
        if not labels:
            labels = extract_textual_labels(question)
        extracted = "".join(labels)
        if question.marked:
            confidence = sum(cell.confidence for _, cell in question.marked) / len(question.marked)
        elif labels:
            confidence = sum(question.choices[label].confidence for label in labels) / len(labels)
        else:
            confidence = 0.0
        answers.append(
            PageAnswer(
                question=question_number,
                extracted=extracted,
                confidence=confidence,
                is_correct=None,
            )
        )

    source_path = extractions[0].source_path if extractions else Path()
    page_index = extractions[0].page_index if extractions else 0
    return PageResult(source_path=source_path, page_index=page_index, answers=answers)
