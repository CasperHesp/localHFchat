# Brainbay Local HF Chat — Quick Start

## Docker (recommended)

> macOS (+GPU acceleration) is at the bottom.

```bash
# Build & start (publishes a RANDOM host port for 8000 in the container)
docker compose up --build -d

# Find the URL to open
docker compose port chatapp 8000
# -> http://localhost:XXXXX
```

---

## What it is

A **fully local** chat app: **FastAPI** backend + minimalist **vanilla** frontend, powered by a **public Hugging Face** chat model.

* **Brainbay-aware** assistant: system prompt + `brainbay_company.txt`, plus one extra context via dropdown (**Market/Geography/Matching** from `brainbay_*.txt`).
* **Frontend**: Markdown, “Thinking…” spinner, **Stop** button, loading overlay (auto-reload when model is ready), intro markdown per context, **10s kickstart** prompts, conversation list (rename/delete), export/import, dark mode, **session memory** (summarize & save, downloadable).
* **Backend**: Hugging Face Transformers (no external inference API), tuned decoding (temp 0.3 / top_p 0.95 / top_k 40 / rep 1.05), **auto token budgeting**, **timeout guard** (default 12s) to prevent stalls, endpoints for chat/summarize/memory.
* **macOS Apple Silicon**: MPS (GPU) acceleration supported (see below).

---

## macOS (+GPU/MPS) option

If you prefer running locally without Docker and want Apple GPU acceleration:

```bash
# optional: choose a tiny public HF model
export MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"

# use your requirements.txt (or requirements-macos.txt) for install
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt  # or: pip install -r requirements-macos.txt

# pick a free port and launch
PORT=$(python - <<'PY'
import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()
PY
)
export PYTORCH_ENABLE_MPS_FALLBACK=1
uvicorn backend.app.main:app --host 0.0.0.0 --port "$PORT"
# open http://localhost:$PORT
```

> Tip: use the included `run_macos.sh` (if present) to auto-install from requirements, pick a random free port, enable MPS, and open your browser:
>
> ```bash
> chmod +x run_macos.sh
> MODEL_ID="HuggingFaceTB/SmolLM2-360M-Instruct" ./run_macos.sh
> ```
