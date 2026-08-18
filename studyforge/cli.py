from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

load_dotenv()

from studyforge.config import get_settings
from studyforge.rag import ask, ingest_pdf, quiz
from studyforge.store import reset_collection


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="studyforge",
        description="StudyForge — ingest a PDF, ask cited questions, generate quizzes.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Extract, chunk, embed, and store a PDF")
    p_ing.add_argument("pdf")
    p_ing.add_argument("--reset", action="store_true", help="Clear the vector store first")

    p_ask = sub.add_parser("ask", help="Ask a question over ingested notes")
    p_ask.add_argument("question")
    p_ask.add_argument("--mode", choices=["kid", "exam", "deep"], default="exam")

    p_quiz = sub.add_parser("quiz", help="Generate a 5-question multiple-choice quiz")
    p_quiz.add_argument("--topic", default="")

    args = parser.parse_args(argv)
    settings = get_settings()

    if args.cmd == "ingest":
        if args.reset:
            reset_collection(settings)
        stats = ingest_pdf(args.pdf, settings=settings)
        print(f"Ingested {stats['pages']} pages → {stats['chunks']} chunks")
        return

    if args.cmd == "ask":
        result = ask(args.question, mode=args.mode, settings=settings)
        print(result["answer"])
        if result.get("citations"):
            cites = ", ".join(f"{c['source']} p.{c['page']}" for c in result["citations"])
            print(f"\nRetrieved from: {cites}")
        return

    if args.cmd == "quiz":
        result = quiz(topic=args.topic, settings=settings)
        if result.get("error"):
            print(result["error"], file=sys.stderr)
            sys.exit(1)
        print(json.dumps({"questions": result["questions"]}, indent=2))


if __name__ == "__main__":
    main()
