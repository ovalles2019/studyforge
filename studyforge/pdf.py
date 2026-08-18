from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import fitz  # PyMuPDF


@dataclass
class PageText:
    page: int  # 1-indexed
    text: str
    source: str


def normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path: str | Path) -> list[PageText]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: list[PageText] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = normalize_whitespace(page.get_text("text") or "")
            if text:
                pages.append(PageText(page=i, text=text, source=path.name))
    return pages
