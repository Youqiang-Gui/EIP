#!/usr/bin/env python3
"""Render the displayed source PDFs as full-page, 2400-pixel-wide PNG previews.

Run from any directory: python /path/to/EIP/docs/scripts/render_figures.py
Dependency: python -m pip install pymupdf
"""

from pathlib import Path

try:
    import pymupdf
except ImportError:
    raise SystemExit("Install the PDF renderer first: python -m pip install pymupdf")


def main():
    figures = Path(__file__).resolve().parents[1] / "assets" / "figures"
    for name in ("teaser", "pipeline", "mask", "data-process"):
        source = figures / f"{name}.pdf"
        output = figures / f"{name}.png"
        with pymupdf.open(source) as document:
            if document.page_count != 1:
                raise SystemExit(f"Expected a single-page figure: {source}")
            page = document[0]
            scale = 2400 / page.rect.width
            # Render the composed page, including overlays and annotations.
            # Extracting embedded images alone would omit masks drawn on top.
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(scale, scale),
                colorspace=pymupdf.csRGB,
                alpha=False,
                annots=True,
            )
            pixmap.save(output)
            print(f"{source.name} → {output.name} ({pixmap.width} × {pixmap.height})")


if __name__ == "__main__":
    main()
