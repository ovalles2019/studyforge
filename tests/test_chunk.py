from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from studyforge.chunk import chunk_pages
from studyforge.pdf import extract_pages


def test_extract_and_chunk_sample(tmp_path, monkeypatch):
    from scripts.make_sample_pdf import build_pdf

    pdf = tmp_path / "primer.pdf"
    build_pdf(pdf)
    pages = extract_pages(pdf)
    assert len(pages) == 4
    assert pages[0].page == 1
    assert "RAG" in pages[0].text
    assert "MI300X" in pages[3].text

    chunks = chunk_pages(pages, chunk_size=800, overlap=80)
    assert chunks
    assert {c.page_start for c in chunks} == {1, 2, 3, 4}
    assert all(c.source == "primer.pdf" for c in chunks)


def test_chunk_preserves_short_page():
    from studyforge.pdf import PageText

    pages = [PageText(page=7, text="Short note about cosine similarity.", source="slides.pdf")]
    chunks = chunk_pages(pages, chunk_size=800, overlap=40)
    assert len(chunks) == 1
    assert chunks[0].page_start == 7
    assert chunks[0].chunk_id.startswith("slides.pdf::p7::")
