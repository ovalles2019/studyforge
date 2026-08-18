from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from studyforge.pdf import PageText, normalize_whitespace


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    source: str

    def metadata(self) -> dict[str, Any]:
        return {
            "page_start": self.page_start,
            "page_end": self.page_end,
            "source": self.source,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        pieces = _split_text(page.text, chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{page.source}::p{page.page}::{i}",
                    text=piece,
                    page_start=page.page,
                    page_end=page.page,
                    source=page.source,
                )
            )
    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    packed: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        content = buf.strip()
        if content:
            packed.append(content)
        buf = ""

    for para in paragraphs:
        if not buf:
            buf = para
            continue
        if len(buf) + 2 + len(para) <= chunk_size:
            buf = f"{buf}\n\n{para}"
        else:
            flush()
            if packed and overlap > 0:
                tail = packed[-1][-overlap:]
                buf = f"{tail}\n\n{para}" if tail else para
            else:
                buf = para
    flush()

    final: list[str] = []
    for piece in packed:
        if len(piece) <= int(chunk_size * 1.5):
            final.append(piece)
            continue
        words = piece.split()
        step = max(40, chunk_size // 5)
        i = 0
        while i < len(words):
            final.append(" ".join(words[i : i + step]))
            i += max(1, step - max(1, overlap // 5))
    return final
