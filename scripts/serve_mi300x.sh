#!/usr/bin/env bash
# Run this ON the AMD Developer Cloud droplet (vLLM Quick Start image).
# Do not start the droplet until credits are approved — and shut it down when you step away.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"

echo "Serving ${MODEL} on 0.0.0.0:${PORT} (vLLM + ROCm)"
echo "Point StudyForge at: LLM_BASE_URL=http://$(hostname -I 2>/dev/null | awk '{print $1}'):${PORT}/v1"

exec vllm serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code
