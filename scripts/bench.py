#!/usr/bin/env python3
"""Measure TTFT and tokens/sec against whatever LLM_BASE_URL is in .env.

Dry-run now against OpenAI. Swap .env to the MI300X droplet in Week 2 for
the numbers that go in the showcase (do not treat OpenAI numbers as AMD benches).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from studyforge.config import get_settings
from studyforge.llm import _client


PROMPT = (
    "In three short paragraphs, explain retrieval-augmented generation, "
    "why answers should cite page numbers, and why a 192 GB GPU helps with long notes."
)


def one_run(client, model: str, max_tokens: int) -> dict:
    started = time.perf_counter()
    first_token_at: float | None = None
    text_parts: list[str] = []
    completion_tokens = 0

    kwargs = dict(
        model=model,
        temperature=0.2,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": PROMPT}],
        stream=True,
    )
    try:
        stream = client.chat.completions.create(**kwargs, stream_options={"include_usage": True})
    except TypeError:
        stream = client.chat.completions.create(**kwargs)
    for event in stream:
        if event.usage and event.usage.completion_tokens:
            completion_tokens = event.usage.completion_tokens
        if not event.choices:
            continue
        delta = event.choices[0].delta.content
        if not delta:
            continue
        if first_token_at is None:
            first_token_at = time.perf_counter()
        text_parts.append(delta)

    ended = time.perf_counter()
    text = "".join(text_parts)
    if completion_tokens <= 0:
        completion_tokens = max(1, len(text.split()))
    ttft = (first_token_at or ended) - started
    gen = max(ended - (first_token_at or started), 1e-6)
    return {
        "ttft_s": ttft,
        "tokens": completion_tokens,
        "tok_s": completion_tokens / gen,
        "chars": len(text),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TTFT / tokens-per-second harness")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    settings = get_settings()
    if not settings.llm_enabled:
        raise SystemExit("Set LLM_API_KEY in .env (or any non-empty key for vLLM).")

    client = _client(settings)
    print(f"endpoint: {settings.llm_base_url}")
    print(f"model:    {settings.llm_model}")
    print(f"runs:     {args.runs}  max_tokens={args.max_tokens}")
    print()

    rows = []
    for i in range(args.runs):
        row = one_run(client, settings.llm_model, args.max_tokens)
        rows.append(row)
        print(
            f"run {i + 1}:  TTFT {row['ttft_s']:.3f}s   "
            f"{row['tok_s']:.1f} tok/s   {row['tokens']} tokens"
        )

    ttfts = [r["ttft_s"] for r in rows]
    speeds = [r["tok_s"] for r in rows]
    print()
    print(f"TTFT  mean {statistics.mean(ttfts):.3f}s  median {statistics.median(ttfts):.3f}s")
    print(f"tok/s mean {statistics.mean(speeds):.1f}   median {statistics.median(speeds):.1f}")
    if "openai.com" in settings.llm_base_url:
        print("\nThese are OpenAI numbers, not MI300X benches. Re-run after pointing .env at vLLM.")


if __name__ == "__main__":
    main()
