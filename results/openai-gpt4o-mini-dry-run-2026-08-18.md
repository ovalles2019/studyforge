# OpenAI dry-run — 2026-08-18

**Not an MI300X result.** This is a harness check against `gpt-4o-mini` so we know
`scripts/bench.py` works before GPU credits arrive. Do not put these numbers on
the AMD showcase chart.

## Setup

| Field | Value |
|---|---|
| Date | 2026-08-18 |
| Machine | laptop (client only) |
| Endpoint | `https://api.openai.com/v1` |
| Model | `gpt-4o-mini` |
| Command | `python scripts/bench.py --runs 3` |
| Max tokens | 256 |
| Prompt | Fixed three-paragraph ask about RAG, page citations, and 192 GB HBM (no PDF retrieval) |

## Results

| Run | TTFT (s) | tok/s | completion tokens |
|---|---|---|---|
| 1 | 2.449 | 102.3 | 256 |
| 2 | 1.101 | 106.4 | 256 |
| 3 | 1.146 | 101.7 | 256 |
| **Mean** | **1.565** | **103.5** | 256 |
| **Median** | **1.146** | **102.3** | 256 |

TTFT = time to first streamed token after the request is sent.
tok/s = completion tokens ÷ time after the first token.
Run 1 is a cold connection; runs 2–3 are the steadier pair.

## How to reproduce

```bash
python scripts/bench.py --runs 3
```

After the droplet is up, point `.env` at vLLM and save a new file under `results/`
named for the GPU model (for example `mi300x-qwen3-8b-YYYY-MM-DD.md`).
