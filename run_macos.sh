#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_macos.sh [MODEL_ID]
MODEL_ID_ARG=${1:-"Qwen/Qwen2.5-0.5B-Instruct"}

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt

export MODEL_ID="${MODEL_ID_ARG}"
export NUM_THREADS="${NUM_THREADS:-6}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

PORT=$(python - <<'PY'
import socket
s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()
PY
)
echo "Picked free port: $PORT"

( sleep 1; open "http://localhost:${PORT}" ) &

cd backend
uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
