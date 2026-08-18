# StudyForge

A study copilot built by a student, for students, to run on an **AMD Instinct MI300X**.

Upload lecture notes. Ask questions with **page citations**. Generate a quiz. Switch the explanation between "explain like I'm 12," exam-ready, and deep dive.

The retrieval stack (PDF → chunks → embeddings → ChromaDB) runs on a laptop CPU. The language model is any **OpenAI-compatible** endpoint — Fireworks while developing, then **vLLM on ROCm** on the AMD Developer Cloud.

## Honest model note (read this)

AMD's August 12, 2026 Day-0 announcement was **Qwen3.8-2.4T-A95B**, a 2.4-trillion-parameter MoE model. Published recipes want a multi-GPU node (on the order of 8× MI355X for MXFP4), not one student droplet and not a ~$100 credit pack.

StudyForge still showcases the stack AMD wants to see — **Qwen + vLLM + ROCm on Instinct** — with a model that fits a single MI300X (192 GB HBM):

| When | Model | Where |
|---|---|---|
| Week 0–1 (dev, $0 GPU) | `gpt-4o-mini` or a serverless Qwen | Laptop + API |
| Week 2–3 (demo + benches) | `Qwen/Qwen3-8B` or `Qwen/Qwen3-32B` | One MI300X, vLLM |
| Not in budget | Qwen3.8-2.4T | Multi-node; sequel if credits allow |

That gap is a feature in the write-up, not something to hide.

## Status

- [x] Repo + README describing the goal
- [x] PDF → chunks → embeddings → ChromaDB pipeline (CPU)
- [x] RAG client that speaks OpenAI-compatible APIs (Fireworks now, vLLM later)
- [x] Quiz JSON prompt + three explanation modes
- [x] Streamlit UI (upload, streaming cited chat, quiz, explanation modes)
- [x] Cloud credit request submitted (waiting on approval — do not activate yet)
- [ ] MI300X droplet (Week 2)
- [ ] Benchmarks + demo video (Week 3)
- [ ] Showcase post (Week 4)

## While credits are pending

Keep the GPU off. Use this waiting time:

1. Upload a **real course PDF** and confirm citations still make sense.
2. Dry-run the benchmark harness against the current API (OpenAI numbers are not the showcase):

```bash
python scripts/bench.py --runs 3
```

3. Sketch the 60-second demo: Load sample → ask → switch explanation mode → generate quiz.
4. When the approval email arrives, **do not activate** until you are ready to sit at the machine. Credits expire 30 days after activation.

## Quick start (CPU, no GPU)

```bash
cd studyforge
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Click **Load sample** in the sidebar, then ask “What is RAG?” or generate a quiz.

Ingest and retrieval work with an empty `LLM_API_KEY`. Put an OpenAI-compatible key in `.env` for streaming answers and quizzes. Later, point `LLM_BASE_URL` at the MI300X droplet.

CLI still works if you want the notebook/script path:

```bash
python scripts/make_sample_pdf.py
python -m studyforge ingest sample/rag_primer.pdf --reset
python -m studyforge ask "What is RAG?" --mode exam
python -m studyforge quiz --topic "embeddings"
```

Walk the same pipeline in `notebooks/01_rag_pipeline.ipynb`.

## Architecture

```
Streamlit UI
      │
      ▼
Python on your laptop
  PyMuPDF → chunk by page → sentence-transformers → ChromaDB
      │  OpenAI-compatible HTTP
      ▼
vLLM on AMD Developer Cloud MI300X   (Week 2)
  Qwen3-8B or Qwen3-32B  ·  ROCm Quick Start image
```

Pointing the same `openai` client at Fireworks vs vLLM is a `base_url` change. That is how GPU hours stay inside the credit budget: the droplet only does inference.

## Credits (do not burn these early)

- ≈ $100 ≈ 50 hours on one MI300X, **expires 30 days after activation**
- Build against Fireworks / any cheap API first
- Shut the droplet down every time you step away

## Reproduce on MI300X (Week 2)

1. Activate credits only when you can babysit the droplet. Use the **vLLM Quick Start** image.
2. On the droplet:

```bash
chmod +x scripts/serve_mi300x.sh
MODEL=Qwen/Qwen3-8B ./scripts/serve_mi300x.sh
```

3. On your laptop `.env`:

```
LLM_BASE_URL=http://YOUR_DROPLET_IP:8000/v1
LLM_API_KEY=not-needed
LLM_MODEL=Qwen/Qwen3-8B
```

4. Same Streamlit / `ingest` / `ask` / `quiz` commands as above, then:

```bash
python scripts/bench.py --runs 5 --max-tokens 256
```

Shut the droplet down the moment you step away.

## What's next

When credits are approved: activate, serve Qwen with vLLM, point `.env` at the droplet, run `scripts/bench.py`, record a 2–3 minute demo, and post in the AMD Developer Community Showcase.
