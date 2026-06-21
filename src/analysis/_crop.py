"""Crop whitespace margins from a generated PDF figure.

Best-effort helper used by the plotting modules. Requires PyMuPDF and Pillow; if
they are missing the caller skips cropping (the figure is just left uncropped).
"""

from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageChops


def _content_bbox(pix):
    """Bounding box of the non-white pixels, or None if the page is blank."""
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    bg = Image.new("RGB", img.size, (255, 255, 255))
    return ImageChops.difference(img, bg).getbbox()


def crop_pdf(path: Path, margin: float = 1.0) -> None:
    """Crop every page of the PDF at `path` to its content plus `margin` points."""
    doc = fitz.open(path)
    scale = 4  # render at 4x for a precise bounding box
    for page in doc:
        bbox = _content_bbox(page.get_pixmap(matrix=fitz.Matrix(scale, scale)))
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        rect = fitz.Rect(x0 / scale - margin, y0 / scale - margin,
                         x1 / scale + margin, y1 / scale + margin) & page.rect
        page.set_cropbox(rect)
    doc.save(str(path), incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()
