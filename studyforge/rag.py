from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from studyforge.chunk import chunk_pages
from studyforge.config import ROOT, Settings, get_settings
from studyforge import llm, prompts, store
from studyforge.pdf import extract_pages


def ingest_pdf(pdf_path: str | Path, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    n = store.upsert_chunks(chunks, settings=settings)
    return {"pages": len(pages), "chunks": n, "source": Path(pdf_path).name}


def ingest_bytes(
    filename: str,
    data: bytes,
    *,
    reset: bool = False,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    if reset:
        store.reset_collection(settings)
    dest = ROOT / "data" / "uploads" / Path(filename).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return ingest_pdf(dest, settings=settings)


def ingest_sample(*, reset: bool = True, settings: Settings | None = None) -> dict[str, Any]:
    sample = ROOT / "sample" / "rag_primer.pdf"
    if not sample.exists():
        from scripts.make_sample_pdf import build_pdf

        build_pdf(sample)
    settings = settings or get_settings()
    if reset:
        store.reset_collection(settings)
    return ingest_pdf(sample, settings=settings)


def prepare_answer(question: str, mode: str = "exam", settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    hits = store.retrieve(question, settings=settings)
    citations = _citations(hits)
    if not hits:
        return {
            "kind": "empty",
            "answer": "No notes ingested yet. Upload a PDF or load the sample primer.",
            "citations": [],
            "hits": [],
            "messages": [],
        }

    messages = [
        {"role": "system", "content": prompts.answer_system(mode)},
        {"role": "user", "content": prompts.answer_user(question, hits)},
    ]
    if not settings.llm_enabled:
        preview = "\n\n".join(f"[p. {h['page_start']}] {h['text'][:400]}" for h in hits)
        return {
            "kind": "offline",
            "answer": (
                "LLM_API_KEY is not set, so here are the retrieved passages instead "
                "of a generated answer:\n\n" + preview
            ),
            "citations": citations,
            "hits": hits,
            "messages": messages,
        }
    return {
        "kind": "stream",
        "answer": "",
        "citations": citations,
        "hits": hits,
        "messages": messages,
    }


def ask(question: str, mode: str = "exam", settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    prep = prepare_answer(question, mode=mode, settings=settings)
    if prep["kind"] != "stream":
        return {"answer": prep["answer"], "citations": prep["citations"], "hits": prep["hits"]}
    answer = llm.chat(prep["messages"], settings=settings)
    return {"answer": answer, "citations": prep["citations"], "hits": prep["hits"]}


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
    return {"questions": normalize_questions(questions), "hits": hits}


def normalize_questions(raw: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        choices = item.get("choices") or {}
        if isinstance(choices, list):
            letters = "ABCD"
            choices = {letters[i]: str(c) for i, c in enumerate(choices[:4])}
        if not isinstance(choices, dict) or not choices:
            continue
        choices = {str(k).strip().upper()[:1]: str(v) for k, v in choices.items()}
        answer = str(item.get("answer", "A")).strip().upper()[:1]
        if answer not in choices:
            answer = next(iter(choices))
        try:
            page = int(item.get("source_page") or item.get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        out.append(
            {
                "question": question,
                "choices": choices,
                "answer": answer,
                "explanation": str(item.get("explanation", "")).strip(),
                "source_page": page,
            }
        )
    return out


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
