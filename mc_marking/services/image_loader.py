"""Load images and PDF pages into NumPy arrays."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError

from mc_marking.utils.image_utils import load_image


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class LoadedPage:
    image: np.ndarray
    source: Path
    page_index: int


def _load_pdf(path: Path, poppler_path: str | None) -> List[LoadedPage]:
    try:
        images = convert_from_path(path, poppler_path=poppler_path)
    except PDFInfoNotInstalledError as exc:
        hint = f"Ensure Poppler is installed and its bin folder is on PATH{' or set in the application settings' if poppler_path else ''}."
        raise RuntimeError(
            f"Unable to get page count for PDF files. {hint}"
        ) from exc
    except FileNotFoundError as exc:
        if poppler_path:
            raise RuntimeError(f"Poppler binaries were not found at '{poppler_path}'. Update the path in Settings -> Poppler Path.") from exc
        raise RuntimeError(
            "Poppler binaries were not found. Configure the Poppler path in Settings or add it to PATH."
        ) from exc
    except PDFPageCountError as exc:
        raise RuntimeError(f"Failed to read PDF pages from '{path.name}'. The file may be encrypted or corrupted.") from exc
    return [LoadedPage(image=np.array(page.convert("RGB")), source=path, page_index=index) for index, page in enumerate(images)]


def _load_image(path: Path) -> List[LoadedPage]:
    return [LoadedPage(image=load_image(path), source=path, page_index=0)]


def load_pages(paths: Iterable[Path], poppler_path: str | None = None) -> List[LoadedPage]:
    """Return RGB arrays for the provided image or PDF files."""
    effective_poppler = poppler_path or os.environ.get("POPPLER_PATH")
    pages: List[LoadedPage] = []
    for raw_path in paths:
        path = Path(raw_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages.extend(_load_pdf(path, effective_poppler))
        elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
            pages.extend(_load_image(path))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    return pages
