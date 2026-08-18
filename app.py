from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from studyforge import llm
from studyforge.config import Settings, get_settings
from studyforge.prompts import MODE_LABELS
from studyforge.rag import ingest_bytes, ingest_sample, prepare_answer, quiz
from studyforge.store import count as chunk_count
from studyforge.store import list_sources, reset_collection

SAMPLE_PROMPTS = [
    "What is RAG, and why do answers cite pages?",
    "How do quizzes stay grounded in the notes?",
    "Why run this on an MI300X instead of Qwen 3.8 2.4T?",
]

PROVIDERS = {
    "Fireworks (AMD perk)": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "model": "accounts/fireworks/models/qwen3-235b-a22b",
        "keys": "https://fireworks.ai/account/api-keys",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "keys": "https://platform.openai.com/api-keys",
    },
}


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("quiz", None)
    st.session_state.setdefault("quiz_checked", False)
    st.session_state.setdefault("pending_question", None)


def _css() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');
          html, body, [class*="css"] { font-family: "IBM Plex Sans", sans-serif; }
          h1, h2, h3 { font-family: "Source Serif 4", Georgia, serif !important; letter-spacing: -0.02em; }
          .block-container { padding-top: 1.4rem; max-width: 820px; }
          .sf-kicker { color: #E07A3D; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 0.2rem; }
          .sf-cite { display: inline-block; background: #2A211C; color: #F3D2B3; border: 1px solid #5A3C2A; border-radius: 999px; padding: 0.15rem 0.6rem; margin: 0.15rem 0.25rem 0 0; font-size: 0.78rem; }
          .stChatMessage { border: 1px solid #2A3038; border-radius: 14px; }
          div[data-testid="stSidebar"] { background: #14181E; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _citation_html(citations: list[dict]) -> str:
    if not citations:
        return ""
    chips = "".join(
        f'<span class="sf-cite">{c.get("source", "notes")} · p.{c.get("page")}</span>' for c in citations
    )
    return chips


def _reset_workspace() -> None:
    reset_collection()
    st.session_state.messages = []
    st.session_state.quiz = None
    st.session_state.quiz_checked = False
    st.session_state.pending_question = None


def _provider_name(settings: Settings) -> str:
    url = settings.llm_base_url.lower()
    if "openai.com" in url:
        return "OpenAI"
    return "Fireworks (AMD perk)"


def _live_settings(base: Settings) -> Settings:
    names = list(PROVIDERS)
    env_provider = _provider_name(base)
    # Earlier builds defaulted this widget to Fireworks and Streamlit kept that session.
    if (
        st.session_state.get("llm_provider") == "Fireworks (AMD perk)"
        and env_provider == "OpenAI"
    ):
        st.session_state.llm_provider = "OpenAI"
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = env_provider
    provider = st.sidebar.selectbox("LLM provider", names, key="llm_provider")
    spec = PROVIDERS[provider]
    if base.llm_enabled:
        st.sidebar.caption("Using the API key from `.env`. Leave the box below empty.")
    override = st.sidebar.text_input(
        "API key override (optional)",
        type="password",
        help="Leave blank to use LLM_API_KEY from .env. Paste here only to try a different key this session.",
        key="llm_api_key_input",
        placeholder="already set in .env" if base.llm_enabled else "paste a key",
    )
    model_key = f"model-{provider}"
    if model_key not in st.session_state:
        st.session_state[model_key] = (
            base.llm_model if provider == env_provider else spec["model"]
        )
    model = st.sidebar.text_input("Model", key=model_key)
    updates = {
        "llm_base_url": spec["base_url"],
        "llm_model": (model or "").strip() or spec["model"],
    }
    if override.strip():
        updates["llm_api_key"] = override.strip()
    return base.model_copy(update=updates)


def _render_sidebar(settings) -> tuple[str, str, Settings]:
    st.sidebar.markdown("**StudyForge**")
    st.sidebar.caption("Student study copilot · AMD Instinct MI300X")
    settings = _live_settings(settings)

    llm_ok = settings.llm_enabled
    st.sidebar.markdown(
        f"{'🟢' if llm_ok else '🟡'} {'LLM connected' if llm_ok else 'Retrieval only (paste a key above)'}"
    )
    st.sidebar.caption(settings.llm_model)
    n = chunk_count(settings)
    sources = list_sources(settings)
    st.sidebar.caption(f"{n} chunks · {len(sources) or 0} source(s)")
    if sources:
        st.sidebar.write(", ".join(sources))

    view = st.sidebar.radio("Workspace", ["Ask", "Quiz"])
    mode_label = st.sidebar.selectbox("Explanation mode", list(MODE_LABELS.values()), index=1)
    mode = next(key for key, label in MODE_LABELS.items() if label == mode_label)

    st.sidebar.divider()
    uploaded = st.sidebar.file_uploader("Upload lecture PDF", type=["pdf"])
    replace = st.sidebar.checkbox("Replace existing notes", value=False)

    if uploaded is not None:
        file_id = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("last_upload") != file_id:
            with st.spinner("Embedding pages on CPU… first run downloads the MiniLM model."):
                stats = ingest_bytes(uploaded.name, uploaded.getvalue(), reset=replace, settings=settings)
            st.session_state.last_upload = file_id
            st.session_state.quiz = None
            st.session_state.quiz_checked = False
            st.sidebar.success(f"{stats['source']}: {stats['pages']} pages → {stats['chunks']} chunks")
            st.rerun()

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("Load sample", use_container_width=True):
        with st.spinner("Loading sample primer…"):
            stats = ingest_sample(reset=True, settings=settings)
        st.session_state.messages = []
        st.session_state.quiz = None
        st.session_state.quiz_checked = False
        st.session_state.last_upload = None
        st.sidebar.success(f"Sample ready ({stats['pages']} pages)")
        st.rerun()
    if col_b.button("Clear notes", use_container_width=True):
        _reset_workspace()
        st.rerun()

    return view, mode, settings


def _render_ask(mode: str, settings) -> None:
    st.markdown('<div class="sf-kicker">Ask the notes</div>', unsafe_allow_html=True)
    st.subheader("Cited answers from your PDF")

    if chunk_count(settings) == 0:
        st.info("Load the sample primer or upload a PDF to start the 60-second demo.")
    elif not st.session_state.messages:
        st.caption("Try one of these:")
        for text in SAMPLE_PROMPTS:
            if st.button(text, key=f"hint-{text[:28]}"):
                st.session_state.pending_question = text
                st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                st.markdown(_citation_html(msg["citations"]), unsafe_allow_html=True)
            if msg.get("hits"):
                with st.expander("Retrieved passages"):
                    for hit in msg["hits"]:
                        st.caption(f"{hit['source']} · p.{hit['page_start']}")
                        st.write(hit["text"])

    typed = st.chat_input("Ask about your notes…")
    prompt = st.session_state.pop("pending_question", None) or typed
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        prep = prepare_answer(prompt, mode=mode, settings=settings)
        if prep["kind"] == "stream":
            try:
                answer = st.write_stream(llm.stream_chat(prep["messages"], settings=settings))
            except Exception as exc:
                st.warning(f"Streaming failed ({exc}). Falling back to a single response.")
                answer = llm.chat(prep["messages"], settings=settings)
                st.markdown(answer)
        else:
            answer = prep["answer"]
            st.markdown(answer)
        st.markdown(_citation_html(prep["citations"]), unsafe_allow_html=True)
        if prep["hits"]:
            with st.expander("Retrieved passages"):
                for hit in prep["hits"]:
                    st.caption(f"{hit['source']} · p.{hit['page_start']}")
                    st.write(hit["text"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": prep["citations"],
            "hits": prep["hits"],
        }
    )
    st.rerun()


def _render_quiz(settings) -> None:
    st.markdown('<div class="sf-kicker">Practice</div>', unsafe_allow_html=True)
    st.subheader("Five questions from the same notes")

    if chunk_count(settings) == 0:
        st.info("Ingest notes first, then generate a quiz.")
        return

    topic = st.text_input("Optional topic focus", placeholder="e.g. embeddings, citations, MI300X")
    if st.button("Generate quiz", type="primary"):
        with st.spinner("Writing five multiple-choice questions…"):
            result = quiz(topic=topic, settings=settings)
        if result.get("error"):
            st.error(result["error"])
            st.session_state.quiz = None
        else:
            st.session_state.quiz = result["questions"]
            st.session_state.quiz_checked = False
            for i in range(5):
                st.session_state.pop(f"quiz_pick_{i}", None)

    questions = st.session_state.quiz
    if not questions:
        return

    for i, item in enumerate(questions):
        st.markdown(f"**{i + 1}. {item['question']}**")
        labels = [f"{letter}. {text}" for letter, text in item["choices"].items()]
        st.radio(
            f"Question {i + 1}",
            labels,
            index=None,
            key=f"quiz_pick_{i}",
            label_visibility="collapsed",
        )
        st.caption(f"Source · p.{item['source_page']}")

    if st.button("Check answers"):
        st.session_state.quiz_checked = True

    if not st.session_state.quiz_checked:
        return

    correct = 0
    for i, item in enumerate(questions):
        pick = st.session_state.get(f"quiz_pick_{i}")
        chosen = pick.split(".", 1)[0] if pick else None
        ok = chosen == item["answer"]
        correct += int(ok)
        if ok:
            st.success(f"{i + 1}. Correct — {item['answer']}")
        else:
            st.error(f"{i + 1}. Answer is {item['answer']}" + ("" if chosen else " (no pick)"))
        if item["explanation"]:
            st.caption(item["explanation"])
    st.markdown(f"### Score: {correct} / {len(questions)}")


def main() -> None:
    st.set_page_config(page_title="StudyForge", page_icon="🔥", layout="centered")
    _init_state()
    _css()
    settings = get_settings()

    st.markdown('<div class="sf-kicker">AMD Developer Cloud · Qwen + vLLM + ROCm</div>', unsafe_allow_html=True)
    st.title("StudyForge")
    st.caption("Upload notes. Get cited answers. Drill with a quiz. Built to run on an Instinct MI300X.")

    view, mode, settings = _render_sidebar(settings)
    if view == "Ask":
        _render_ask(mode, settings)
    else:
        _render_quiz(settings)


if __name__ == "__main__":
    main()
