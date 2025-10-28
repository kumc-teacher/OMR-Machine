"""OCR utilities built on top of Tesseract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import pytesseract

from mc_marking.models.answer_sheet import CellResult, TableExtraction
from mc_marking.utils.image_utils import crop


@dataclass
class OcrConfig:
    """Parameters controlling the OCR pipeline."""

    languages: str = "eng"
    psm_mode: int = 6  # Assume a uniform block of text


def recognise_table_cells(image: np.ndarray, extraction: TableExtraction, config: OcrConfig | None = None) -> List[CellResult]:
    """Apply OCR to each cell in the detected table."""
    cfg = config or OcrConfig()
    results: List[CellResult] = []
    for cell in extraction.cells:
        cell_image = crop(image, cell.bounding_box)
        ocr_ready, ink_mask, enhanced_gray = _preprocess_cell_for_ocr(cell_image)
        base_config = f"--psm {cfg.psm_mode} --oem 3"
        text = pytesseract.image_to_string(ocr_ready, lang=cfg.languages, config=base_config)
        cleaned = _clean_cell_text(text)
        if not cleaned:
            cleaned = _run_fallback_ocr_passes(ocr_ready, cell.column)
        confidence = _estimate_confidence(enhanced_gray)
        ink_density = _estimate_ink_density(cell_image, precomputed_mask=ink_mask)
        results.append(
            CellResult(
                row=cell.row,
                column=cell.column,
                text=cleaned,
                confidence=confidence,
                bounding_box=cell.bounding_box,
                ink_density=ink_density,
            )
        )
    return results


def _estimate_confidence(image: np.ndarray) -> float:
    variance = float(np.var(image))
    normalized = min(1.0, max(0.0, variance / (255.0 ** 2)))
    return normalized


def _estimate_ink_density(image: np.ndarray, *, precomputed_mask: np.ndarray | None = None) -> float:
    if precomputed_mask is not None:
        return float(np.mean(precomputed_mask / 255.0))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float(np.mean(binary / 255.0))


def _preprocess_cell_for_ocr(cell_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(cell_image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 5, 75, 75)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    scale_factor = 2 if min(binary.shape) < 40 else 1
    if scale_factor > 1:
        binary = cv2.resize(binary, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    mask = 255 - binary
    ocr_ready = binary
    return ocr_ready, mask, denoised


def _run_fallback_ocr_passes(image: np.ndarray, column_index: int) -> str:
    attempts = []
    if column_index == 0:
        attempts.append("--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789")
    attempts.extend(
        [
            "--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        ]
    )
    for config in attempts:
        candidate = pytesseract.image_to_string(image, config=config)
        cleaned = _clean_cell_text(candidate)
        if cleaned:
            return cleaned
    return ""


def _clean_cell_text(text: str) -> str:
    normalized = text.replace("\n", " ").strip()
    return normalized
