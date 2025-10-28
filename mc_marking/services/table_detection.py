"""Detect and segment answer tables from scanned pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from mc_marking.models.answer_sheet import CellResult, TableExtraction
from mc_marking.utils.image_utils import BoundingBox, crop


@dataclass
class TableDetectionConfig:
    """Configure morphological operations for table detection."""

    min_table_area_ratio: float = 0.12
    kernel_scale: float = 0.012
    min_cell_size: int = 12


def detect_table(
    image: np.ndarray,
    page_index: int,
    source_path: str,
    config: TableDetectionConfig | None = None,
    roi: BoundingBox | None = None,
) -> Optional[TableExtraction]:
    """Return the most prominent table detected within the provided region."""

    tables = detect_tables(image, page_index, source_path, config=config, roi=roi)
    return tables[0] if tables else None


def detect_tables(
    image: np.ndarray,
    page_index: int,
    source_path: str,
    config: TableDetectionConfig | None = None,
    roi: BoundingBox | None = None,
) -> List[TableExtraction]:
    """Detect all answer tables visible within the image or ROI."""

    cfg = config or TableDetectionConfig()
    search_image = crop(image, roi) if roi else image
    offset_x = roi.x if roi else 0
    offset_y = roi.y if roi else 0

    prepared = _prepare_binary_mask(search_image)
    kernel_size = max(3, int(cfg.kernel_scale * max(search_image.shape[:2])))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    morph = cv2.morphologyEx(prepared, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    h, w = search_image.shape[:2]
    min_area = cfg.min_table_area_ratio * h * w
    candidates = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    tables: List[TableExtraction] = []
    for contour in candidates:
        x, y, width, height = cv2.boundingRect(contour)
        bounding_box = BoundingBox(x=x + offset_x, y=y + offset_y, width=width, height=height)
        extraction = _extract_table(
            image=image,
            bounding_box=bounding_box,
            page_index=page_index,
            source_path=source_path,
            config=cfg,
        )
        if extraction is not None and extraction.cells:
            tables.append(extraction)

    tables.sort(key=lambda table: (table.bounding_box.y, table.bounding_box.x))
    return tables


def _prepare_binary_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12)


def _extract_table(
    *,
    image: np.ndarray,
    bounding_box: BoundingBox,
    page_index: int,
    source_path: str,
    config: TableDetectionConfig,
) -> Optional[TableExtraction]:
    table_roi = crop(image, bounding_box)
    row_positions, col_positions = _estimate_grid(table_roi, config)
    if row_positions is None or col_positions is None:
        return None

    cells: List[CellResult] = []
    for r_idx, (r_start, r_end) in enumerate(_pairwise(row_positions)):
        for c_idx, (c_start, c_end) in enumerate(_pairwise(col_positions)):
            cell_crop = table_roi[r_start:r_end, c_start:c_end]
            if cell_crop.size == 0:
                continue
            average_intensity = float(np.mean(cell_crop))
            cell_box = BoundingBox(
                x=bounding_box.x + c_start,
                y=bounding_box.y + r_start,
                width=c_end - c_start,
                height=r_end - r_start,
            )
            cells.append(
                CellResult(
                    row=r_idx,
                    column=c_idx,
                    text="",
                    confidence=average_intensity / 255.0,
                    bounding_box=cell_box,
                    ink_density=0.0,
                )
            )

    return TableExtraction(
        source_path=Path(source_path),
        page_index=page_index,
        bounding_box=bounding_box,
        cells=cells,
        row_count=len(row_positions) - 1,
        column_count=len(col_positions) - 1,
    )


def _estimate_grid(image: np.ndarray, config: TableDetectionConfig) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, projected_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(image.shape[0] * 0.05))))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(image.shape[1] * 0.05)), 1))
    vertical_lines = cv2.morphologyEx(projected_bin, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    horizontal_lines = cv2.morphologyEx(projected_bin, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    grid_mask = cv2.add(vertical_lines, horizontal_lines)

    edges = cv2.Canny(grid_mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=120,
        minLineLength=config.min_cell_size * 4,
        maxLineGap=config.min_cell_size,
    )

    rows: List[int] = [0, image.shape[0]]
    cols: List[int] = [0, image.shape[1]]
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(x1 - x2) < 10:
                cols.append(min(x1, x2))
            if abs(y1 - y2) < 10:
                rows.append(min(y1, y2))

    rows = _filter_positions(sorted(set(rows)), config.min_cell_size)
    cols = _filter_positions(sorted(set(cols)), config.min_cell_size)

    if len(rows) < 2 or len(cols) < 2:
        return None, None
    return rows, cols


def _pairwise(positions: Sequence[int]) -> List[Tuple[int, int]]:
    return list(zip(positions[:-1], positions[1:]))


def _filter_positions(positions: Sequence[int], min_distance: int) -> List[int]:
    ordered = list(positions)
    if not ordered:
        return []
    filtered: List[int] = [ordered[0]]
    for pos in ordered[1:-1]:
        if abs(pos - filtered[-1]) >= min_distance:
            filtered.append(pos)
    if abs(ordered[-1] - filtered[-1]) >= min_distance:
        filtered.append(ordered[-1])
    elif ordered[-1] != filtered[-1]:
        filtered[-1] = ordered[-1]
    return filtered
