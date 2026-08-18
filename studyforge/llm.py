from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from studyforge.config import Settings, get_settings


def _client(settings: Settings) -> OpenAI:
    # vLLM accepts any non-empty key; Fireworks needs a real one.
    key = settings.llm_api_key.strip() or "not-needed"
    return OpenAI(api_key=key, base_url=settings.llm_base_url)


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    settings: Settings | None = None,
) -> str:
    settings = settings or get_settings()
    resp = _client(settings).chat.completions.create(
        model=settings.llm_model,
        temperature=temperature,
        messages=messages,
    )
    return (resp.choices[0].message.content or "").strip()


def stream_chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    settings: Settings | None = None,
) -> Iterator[str]:
    settings = settings or get_settings()
    stream = _client(settings).chat.completions.create(
        model=settings.llm_model,
        temperature=temperature,
        messages=messages,
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta.content if event.choices else None
        if delta:
            yield delta
