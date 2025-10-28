"""Utility helpers for image handling and conversions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BoundingBox:
    """Simple rectangle description in image coordinate space."""

    x: int
    y: int
    width: int
    height: int

    @property
    def as_slice(self) -> Tuple[slice, slice]:
        """Return NumPy slicing tuple covering this box."""
        return (slice(self.y, self.y + self.height), slice(self.x, self.x + self.width))

    def ensure_within(self, image_shape: Sequence[int]) -> "BoundingBox":
        """Clamp the bounding box to fit within the provided image shape."""
        if len(image_shape) < 2:
            raise ValueError("image_shape must have at least two dimensions")
        max_y, max_x = image_shape[0], image_shape[1]
        x = max(0, min(self.x, max_x - 1))
        y = max(0, min(self.y, max_y - 1))
        width = max(1, min(self.width, max_x - x))
        height = max(1, min(self.height, max_y - y))
        return BoundingBox(x=x, y=y, width=width, height=height)


def load_image(path: Path) -> np.ndarray:
    """Load an image from disk into an RGB NumPy array."""
    with Image.open(path) as pil_img:
        return np.array(pil_img.convert("RGB"))


def save_image(path: Path, image: np.ndarray) -> None:
    """Save a NumPy image array to disk."""
    pil_image = Image.fromarray(image)
    pil_image.save(path)


def crop(image: np.ndarray, box: BoundingBox) -> np.ndarray:
    """Crop an image according to the provided bounding box."""
    bounded = box.ensure_within(image.shape)
    return image[bounded.as_slice]


def combine_bounding_boxes(boxes: Iterable[BoundingBox]) -> BoundingBox | None:
    """Return a bounding box that tightly encloses the provided boxes."""
    xs: List[int] = []
    ys: List[int] = []
    xe: List[int] = []
    ye: List[int] = []
    for box in boxes:
        xs.append(box.x)
        ys.append(box.y)
        xe.append(box.x + box.width)
        ye.append(box.y + box.height)
    if not xs:
        return None
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xe)
    max_y = max(ye)
    return BoundingBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y)
