from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from studyforge.chunk import chunk_pages
from studyforge.config import Settings, get_settings
from studyforge import llm, prompts, store
from studyforge.pdf import extract_pages


def ingest_pdf(pdf_path: str | Path, settings: Settings | None = None) -> dict[str, int]:
    settings = settings or get_settings()
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    n = store.upsert_chunks(chunks, settings=settings)
    return {"pages": len(pages), "chunks": n}


def ask(question: str, mode: str = "exam", settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    hits = store.retrieve(question, settings=settings)
    if not hits:
        return {"answer": "No notes ingested yet. Run `ingest` on a PDF first.", "citations": [], "hits": []}

    if not settings.llm_enabled:
        preview = "\n\n".join(
            f"[p. {h['page_start']}] {h['text'][:400]}" for h in hits
        )
        return {
            "answer": (
                "LLM_API_KEY is not set, so here are the retrieved passages instead "
                "of a generated answer:\n\n" + preview
            ),
            "citations": _citations(hits),
            "hits": hits,
        }

    messages = [
        {"role": "system", "content": prompts.answer_system(mode)},
        {"role": "user", "content": prompts.answer_user(question, hits)},
    ]
    answer = llm.chat(messages, settings=settings)
    return {"answer": answer, "citations": _citations(hits), "hits": hits}


def quiz(topic: str = "", settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    query = topic.strip() or "key concepts, definitions, and exam-worthy facts"
    hits = store.retrieve(query, k=max(settings.retrieve_k, 8), settings=settings)
    if not hits:
        return {"questions": [], "error": "No notes ingested yet."}
    if not settings.llm_enabled:
        return {"questions": [], "error": "Set LLM_API_KEY to generate a quiz.", "hits": hits}

    messages = [
        {"role": "system", "content": prompts.QUIZ_SYSTEM},
        {"role": "user", "content": prompts.quiz_user(hits, topic=topic)},
    ]
    raw = llm.chat(messages, temperature=0.4, settings=settings)
    data = _extract_json(raw)
    questions = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(questions, list):
        return {"questions": [], "error": "Model did not return quiz JSON.", "raw": raw, "hits": hits}
    return {"questions": questions, "hits": hits}


def _citations(hits: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    cites = []
    for hit in hits:
        key = (hit.get("source"), hit.get("page_start"))
        if key in seen:
            continue
        seen.add(key)
        cites.append({"source": hit.get("source"), "page": hit.get("page_start")})
    return cites


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
