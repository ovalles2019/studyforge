"""Generate the bundled sample PDF used in Week 0 demos."""

from __future__ import annotations

from pathlib import Path

import fitz

PAGES = [
    (
        "StudyForge primer: what is RAG?",
        """Retrieval-augmented generation (RAG) is a way to make a language model answer
questions about your own documents instead of only its training data.

The idea is simple. First, you break a PDF into chunks and store them in a vector
database. When a student asks a question, you retrieve the most relevant chunks
and put them in the prompt. The model must answer from those notes and cite the
page it used.

StudyForge uses RAG so a student can upload lecture slides or a textbook chapter
and get answers grounded in that material. Without retrieval, the model might
hallucinate a confident-sounding explanation that is not in the notes.

A good RAG answer names the page. Example: "A vector database stores embeddings
so similar chunks can be found by cosine similarity [p. 2]."
""",
    ),
    (
        "Chunking and embeddings",
        """Chunking means splitting a document into pieces small enough to retrieve.
StudyForge keeps chunks on a single page whenever possible so citations stay
honest. A typical chunk is a few paragraphs, with a short overlap so a sentence
near a boundary is not lost.

Embeddings turn each chunk into a list of numbers. Nearby meanings land near
each other in that space. StudyForge uses a small sentence-transformers model
on CPU (all-MiniLM-L6-v2) so embedding never burns GPU credits.

The vector store is ChromaDB, saved on disk under data/chroma. Ingest once, then
ask many questions without re-reading the PDF. If you ingest a second file, both
sources sit in the same collection and each citation still includes the filename.

If a page is empty (for example a slide that is only a diagram), it is skipped.
""",
    ),
    (
        "Citations, quizzes, and explanation modes",
        """Every StudyForge answer should cite pages like [p. 3]. If the notes do not
contain the answer, the model should say so instead of guessing. That is the
difference between a study copilot and a generic chatbot.

Quizzes are generated from the same retrieved notes. A quiz item must be
answerable from the PDF. Each question records a source_page so a student can
flip back to the lecture slide.

Three explanation modes are just different system prompts:
- kid: explain like I am 12, with analogies and almost no jargon.
- exam: precise vocabulary a grader would expect.
- deep: mechanism, edge cases, and nearby topics, still grounded in the notes.

This page exists so retrieval tests have a distinct citation. If you ask
"How do quizzes stay grounded?" the retriever should prefer page 3.
""",
    ),
    (
        "Why this runs on an AMD Instinct MI300X",
        """The retrieval stack above runs on a laptop CPU. The large language model does
not. StudyForge talks to any OpenAI-compatible API. During Week 0 that is a cheap
endpoint such as Fireworks. During Week 2 the same code points at a vLLM server
on an AMD Instinct MI300X in the AMD Developer Cloud.

vLLM on ROCm serves Qwen. A single MI300X has 192 GB of HBM, which easily fits
Qwen3-8B or Qwen3-32B with long context. That is the student-budget model.

The August 12, 2026 Day-0 announcement was for Qwen3.8-2.4T-A95B, a 2.4 trillion
parameter mixture-of-experts model. It needs a multi-GPU node (on the order of
8x MI355X for the MXFP4 recipe), not one droplet and not a $100 credit pack.
StudyForge uses the same serving stack — vLLM + ROCm on Instinct — with a Qwen
that actually fits. Reviewers prefer that honesty over claiming a 2.4T model
on a single card.

Golden rule: shut the GPU droplet down when you step away. Credits expire 30
days after activation and idle hours still count.
""",
    ),
]


def build_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for title, body in PAGES:
        page = doc.new_page(width=612, height=792)
        page.insert_textbox(
            fitz.Rect(72, 64, 540, 110),
            title,
            fontsize=16,
            fontname="helv",
        )
        page.insert_textbox(
            fitz.Rect(72, 120, 540, 720),
            " ".join(line.strip() for line in body.strip().splitlines()),
            fontsize=12,
            fontname="helv",
            align=0,
        )
        page.insert_text((72, 760), f"StudyForge sample  ·  page {page.number + 1} of {len(PAGES)}", fontsize=9)
    doc.save(path)
    doc.close()
    return path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "sample" / "rag_primer.pdf"
    build_pdf(out)
    print(f"Wrote {out}")
