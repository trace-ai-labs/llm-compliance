#!/usr/bin/env python3
"""
Crop whitespace from PDF files.

Run this after generating any figure to remove whitespace margins.

Usage:
    python crop_pdf.py file.pdf [file2.pdf ...]   # crop in-place
    python crop_pdf.py --margin 2 *.pdf            # 2pt margin (default: 1pt)
    python crop_pdf.py --dry-run *.pdf             # show what would be cropped
    python crop_pdf.py *.pdf                       # crop all figs after regeneration
"""

import argparse
import sys
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image


def find_content_bbox(pix):
    """Find bounding box of non-white pixels using Pillow."""
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    # Create a white background image for comparison
    bg = Image.new("RGB", img.size, (255, 255, 255))
    # Diff to find non-white pixels
    from PIL import ImageChops
    diff = ImageChops.difference(img, bg)
    return diff.getbbox()  # returns (x0, y0, x1, y1) or None if blank


def crop_pdf(path: Path, margin: float = 1.0, dry_run: bool = False) -> dict:
    """Crop a PDF to its minimal bounding box plus margin.

    Args:
        path: Path to the PDF file.
        margin: Points of padding to keep around the content.
        dry_run: If True, report but don't modify.

    Returns:
        dict with 'file', 'pages', 'old_size', 'new_size', 'saved'.
    """
    doc = fitz.open(path)
    results = []
    scale = 4  # render at 4x for precise bbox detection

    for page in doc:
        old_rect = page.rect
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        bbox = find_content_bbox(pix)

        if bbox is None:
            results.append((old_rect, old_rect))
            continue

        x0, y0, x1, y1 = bbox
        new_rect = fitz.Rect(
            x0 / scale - margin,
            y0 / scale - margin,
            x1 / scale + margin,
            y1 / scale + margin,
        )
        new_rect = new_rect & old_rect
        page.set_cropbox(new_rect)
        results.append((old_rect, new_rect))

    old_w, old_h = results[0][0].width, results[0][0].height
    new_w, new_h = results[0][1].width, results[0][1].height

    info = {
        'file': str(path),
        'pages': len(results),
        'old_size': f'{old_w:.0f}x{old_h:.0f}',
        'new_size': f'{new_w:.1f}x{new_h:.1f}',
        'saved': not dry_run,
    }

    if not dry_run:
        doc.save(str(path), incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)

    doc.close()
    return info


def main():
    parser = argparse.ArgumentParser(description='Crop whitespace from PDFs.')
    parser.add_argument('files', nargs='+', type=Path, help='PDF files to crop')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin in points to keep around content (default: 1)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report without modifying files')
    args = parser.parse_args()

    for f in args.files:
        if not f.exists():
            print(f'SKIP {f} (not found)')
            continue
        if f.suffix.lower() != '.pdf':
            print(f'SKIP {f} (not a PDF)')
            continue
        try:
            info = crop_pdf(f, margin=args.margin, dry_run=args.dry_run)
            status = 'DRY-RUN' if args.dry_run else 'CROPPED'
            print(f'{status} {info["file"]}: {info["old_size"]} -> {info["new_size"]}')
        except Exception as e:
            print(f'ERROR {f}: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
