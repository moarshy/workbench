"""
File-format layer for the eval-rules desk: turn a generated workspace's CSV data into
real .xlsx workbooks and its Markdown documents into real .pdf files, the formats the
hidden SkillsBench-style tasks use. The content is unchanged (tests round-trip it), so the
families' verifiers still apply; the instruction's paths are rewritten to the new names.
"""

import csv
import re
import textwrap
from pathlib import Path
from typing import Iterable

_NUMBER = re.compile(r"^-?(0|[1-9]\d*)(\.\d+)?$")


def _cell(value: str):
    """Numbers as numbers (as a real export would have them); everything else, IDs and dates included, as text."""
    if _NUMBER.match(value):
        return float(value) if "." in value else int(value)
    return value


def csv_to_xlsx(src: Path, dest: Path, sheet: str) -> None:
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = sheet[:31]
    with open(src, newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            ws.append(row if i == 0 else [_cell(v) for v in row])
    wb.save(dest)


def md_to_pdf(src: Path, dest: Path) -> None:
    """Render a Markdown file as plain text lines in a PDF (headings, tables and all) with reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(dest), pagesize=A4)
    width, height = A4
    y, size, margin = height - 50, 9, 40
    max_chars = 105
    for raw in src.read_text().splitlines():
        line = raw.rstrip()
        bold = line.startswith("#")
        text = line.lstrip("#").strip() if bold else line
        chunks = textwrap.wrap(text, max_chars, break_long_words=False, break_on_hyphens=False) or [""]
        for chunk in chunks:
            if y < 50:
                c.showPage()
                y = height - 50
            c.setFont("Helvetica-Bold" if bold else "Helvetica", size + (2 if bold else 0))
            c.drawString(margin, y, chunk)
            y -= size + 4
    c.save()


def convert(root: Path, instruction: str, xlsx: Iterable[str] = (), pdf: Iterable[str] = ()) -> str:
    """Convert the given workspace files (relative paths) in place; return the instruction with paths rewritten."""
    for rel in xlsx:
        src = root / rel
        dest = src.with_suffix(".xlsx")
        csv_to_xlsx(src, dest, sheet=src.stem)
        src.unlink()
        instruction = instruction.replace(rel, dest.relative_to(root).as_posix()).replace(src.name, dest.name)
    for rel in pdf:
        src = root / rel
        dest = src.with_suffix(".pdf")
        md_to_pdf(src, dest)
        src.unlink()
        instruction = instruction.replace(rel, dest.relative_to(root).as_posix()).replace(src.name, dest.name)
    return instruction
