from __future__ import annotations

MODE_LABELS = {
    "kid": "Explain like I'm 12",
    "exam": "Exam-ready",
    "deep": "Deep dive",
}

EXPLANATION_MODES = {
    "kid": (
        "Explain like I'm 12. Use short sentences, everyday analogies, and no jargon "
        "unless you immediately define it."
    ),
    "exam": (
        "Exam-ready. Be precise, use the course vocabulary, and include the facts a "
        "student would need to write a full-credit answer. Keep it concise."
    ),
    "deep": (
        "Deep dive. Cover mechanism, edge cases, and how this idea connects to nearby "
        "topics. Still stay faithful to the notes — do not invent sources."
    ),
}

CITATION_RULES = (
    "Answer using ONLY the provided notes. Cite page numbers inline like [p. 3]. "
    "If several pages support a claim, cite each one. If the notes do not contain "
    "the answer, say so and do not guess."
)

QUIZ_SYSTEM = """You write practice quizzes from study notes.
Return ONLY valid JSON with this shape:
{
  "questions": [
    {
      "question": "...",
      "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "answer": "A",
      "explanation": "Why this is correct, citing the notes.",
      "source_page": 1
    }
  ]
}
Rules:
- Exactly 5 multiple-choice questions.
- Each question must be answerable from the notes.
- source_page must match a page in the notes.
- Do not wrap the JSON in markdown fences.
"""


def answer_system(mode: str) -> str:
    style = EXPLANATION_MODES.get(mode, EXPLANATION_MODES["exam"])
    return f"You are StudyForge, a study copilot.\n{CITATION_RULES}\nStyle: {style}"


def format_context(hits: list[dict]) -> str:
    blocks = []
    for hit in hits:
        page = hit.get("page_start") or hit.get("page") or "?"
        source = hit.get("source") or "notes"
        blocks.append(f"[{source} p. {page}]\n{hit['text']}")
    return "\n\n".join(blocks)


def answer_user(question: str, hits: list[dict]) -> str:
    return (
        f"Notes:\n{format_context(hits)}\n\n"
        f"Question: {question}\n\n"
        "Answer with inline page citations."
    )


def quiz_user(hits: list[dict], topic: str = "") -> str:
    focus = f"Focus on: {topic}\n\n" if topic.strip() else ""
    return f"{focus}Notes:\n{format_context(hits)}\n\nWrite 5 multiple-choice questions."
