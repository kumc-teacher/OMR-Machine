"""Compare recognized answers against the answer key."""

from __future__ import annotations

from typing import Dict, List

from mc_marking.models.answer_sheet import AnswerKey, PageAnswer, PageResult


def evaluate_page(page: PageResult, answer_key: AnswerKey) -> PageResult:
    """Return a new PageResult with correctness flags."""
    evaluated_answers: List[PageAnswer] = []
    for answer in page.answers:
        correct_answer = answer_key.answer_for(answer.question)
        is_correct = None
        if correct_answer is not None:
            normalized_correct = _normalize(correct_answer)
            normalized_answer = _normalize(answer.extracted)
            if normalized_answer:
                is_correct = normalized_answer == normalized_correct
        evaluated_answers.append(
            PageAnswer(
                question=answer.question,
                extracted=answer.extracted,
                confidence=answer.confidence,
                is_correct=is_correct,
            )
        )
    return PageResult(source_path=page.source_path, page_index=page.page_index, answers=evaluated_answers)


def _normalize(value: str) -> str:
    return value.strip().upper()
