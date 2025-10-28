"""Convert table OCR results into structured answer keys."""

from __future__ import annotations

from typing import List, Optional, Sequence

from mc_marking.models.answer_sheet import AnswerKey, CellResult, TableExtraction
from mc_marking.services.table_parser import enumerate_question_marks, extract_textual_labels


def normalize_answer_key(
    extractions: Sequence[TableExtraction],
    tables_cells: Sequence[List[CellResult]],
    *,
    baseline_density: float | None = None,
    row_labels_override: Optional[List[str]] = None,
) -> AnswerKey:
    """Normalize OCR output for one or more tables into an answer key."""

    answers = {}
    for question_number, question in enumerate_question_marks(
        extractions,
        tables_cells,
        baseline_density=baseline_density,
        row_labels_override=row_labels_override,
    ):
        labels = sorted({label for label, _ in question.marked})
        if not labels:
            labels = extract_textual_labels(question)
        if not labels:
            continue
        answers[question_number] = "".join(labels)

    # Preserve sorted order by question number for presentation and stability.
    ordered_answers = dict(sorted(answers.items()))
    metadata = {}
    if extractions:
        metadata = {"source": str(extractions[0].source_path), "page": str(extractions[0].page_index)}
    return AnswerKey(rows=ordered_answers, metadata=metadata)
