from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from studyforge.config import Settings, get_settings


def _client(settings: Settings) -> OpenAI:
    # vLLM accepts any non-empty key; Fireworks needs a real one.
    key = settings.llm_api_key.strip() or "not-needed"
    return OpenAI(api_key=key, base_url=settings.llm_base_url)


def _extra(settings: Settings) -> dict:
    # Qwen 3.7/3.8 on Fireworks default to long chain-of-thought otherwise.
    if "fireworks.ai" in settings.llm_base_url and "qwen" in settings.llm_model.lower():
        return {"extra_body": {"reasoning_effort": "none"}}
    return {}


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
        **_extra(settings),
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
        **_extra(settings),
    )
    for event in stream:
        delta = event.choices[0].delta.content if event.choices else None
        if delta:
            yield delta
