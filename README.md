# Brainbay Local HF Chat — Short Manual

## Build, start & run (Docker)

```bash
# Build & start (publishes a RANDOM host port for container port 8000)
docker compose up --build -d

# Find the URL to open (prints something like http://localhost:XXXXX)
docker compose port chatapp 8000

# (optional) force precision: auto (default), full, or 4bit quantized
MODEL_QUANTIZATION=4bit docker compose up --build -d

# (optional) pin a host port instead of a random one
HOST_PORT=5173 docker compose up --build -d
```

### Helper scripts

* **macOS / Linux** – `./run_docker.sh` automatically builds, starts, opens the browser (using `open`, `xdg-open`, or `powershell.exe` on WSL), and tails logs.
* **Windows PowerShell** – `pwsh ./scripts/run-windows.ps1` (or `powershell` on Windows). It mirrors the shell script behaviour and works out-of-the-box with Docker Desktop.

> Tip: copy `.env.example` to `.env` to customise defaults (`MODEL_ID`, `HOST_PORT`, etc.) without editing the compose file.

## What it is

A **fully local** chat app: **FastAPI** backend + minimalist **vanilla** frontend, powered by a **public Hugging Face** chat model. It has augmented awareness of BrainBay & real estate consultancy thanks to edge-(pre-)computing: larger LLM models were used to scrape domain-specific contextual data on Brainbay, the real estate market, its geographical depencencies, and matching patterns. These were condensed into local contextual files, which could be expanded to include more sensitive internal data. The fully functional app has been tested with small models compatible with laptop-level compute (current default: Qwen/Qwen2.5-3B-Instruct), and includes basic UI/UX (Markdown, spinner, Stop, intro/kickstart prompts, session memory). Designed to run in Docker or natively on macOS (Apple Silicon with MPS).

---

## Project structure

```
project-root/
  backend/app/main.py
  frontend/
    index.html
    style.css
    app.js
    conversation_starters.json
    intro_markdown.json
  backend/requirements/base.txt
  brainbay_company.txt
  brainbay_market.txt
  brainbay_geography.txt
  brainbay_matching.txt
  session_memory.txt
  Dockerfile
  docker-compose.yml
  infra/docker/entrypoint.sh
  run_macos.sh
  run_docker.sh
  scripts/run-windows.ps1
  requirements.txt
```

* **backend/app/main.py** — FastAPI app. Loads a small HF chat model; merges context **as data** (system → company → selected dropdown file → optional session memory); adds guardrails (timeout, no-repeat, tuned sampling); serves `/static` + UI.
* **frontend/index.html / style.css / app.js** — Small custom UI: Markdown render, “Thinking…” spinner, **Stop**, intro markdown (per context), **10s kickstart** prompts, export/import history, dark mode, session memory (summarize & save + download), plus a live status bar with hardware/model insights and a fast/balanced/quality mode switcher.
* **backend/requirements/base.txt** — Canonical Python dependency list shared by the Docker image and local scripts.
* **conversation_starters.json** — Content-only list of kickstart prompts per context (edit without touching code).
* **intro_markdown.json** — One intro block per context, shown once at chat start.
* **brainbay_*.txt** — **Demo** context files (company / market / geography / matching). Swappable via env vars.
* **session_memory.txt** — Optional running memory appended by the “Summarize & Save” feature (can be included as context via toggle).
* **Dockerfile / docker-compose.yml** — Containerized run; compose publishes a **random host port** to avoid collisions.
* **run_macos.sh / requirements*.txt** — macOS convenience: install from requirements, pick a random free port, enable MPS, open browser.

### About the data in `brainbay_*.txt`

* The contents were **scraped and inferred from public data using other AI tools** and are meant as **placeholders for internal Brainbay context for demo purposes**.
* They are **not actual sensitive plaintext** in this demo. In future production passes, they can be replaced with more sensitive data, in which case we could (optionally) add **encryption layers** (e.g., client-side + at-rest). Until then, treat these as placeholders showing the intended workflow.

---

## Tech summary

* **Backend:** FastAPI + Uvicorn, Hugging Face **Transformers**, **PyTorch** (+ Apple **MPS** on macOS), Accelerate, optional **bitsandbytes** 4-bit quantization
* **Frontend:** custom vanilla **HTML/CSS/JS** (no UI framework)
* **Container:** Docker, docker-compose + edge-(pre-)computation

---

## Design choices

* **[EDGE:contextually compressed data]**: Behavior steered by editable text/JSON files; no rebuilds to change domain knowledge. Sensitive data can be safely and bijectively pseudonymised elsewhere (i.e.,ensuring security & recoverability).
* **[CORE: on-device ergonomics]:** Tuned decoding (`temperature=0.3`, `top_p=0.95`, `top_k=40`, `repetition_penalty=1.05`), **auto token budgeting**, `no_repeat_ngram_size=3`, **timeout guard** (default 12s) to reduce stalls/loops.
* **[SENSE: responsive UI/UX]**: Markdown, spinner, **Stop**, intro per context, **10s kickstart**, export/import, dark mode, session memory with summarization.
* **[FLOW: adaptive scaling]**: Backend inspects the runtime (CPU vs GPU/MPS) and model size to auto-tune dtype, token/time budgets and three preset “Fast / Balanced / Quality” modes surfaced in the UI via a live status bar (model/device/budget).

---

## macOS option (+GPU/MPS)

```bash
# (optional) choose a small public HF model
export MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"

# (optional) choose precision (full keeps float32/float16; auto detects; 4bit forces quantized)
export MODEL_QUANTIZATION="full"

python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt   # markers avoid installing bitsandbytes on macOS

# run on a free port
PORT=$(python - <<'PY'
import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()
PY
)
export PYTORCH_ENABLE_MPS_FALLBACK=1
uvicorn backend.app.main:app --host 0.0.0.0 --port "$PORT"
# open http://localhost:$PORT
```

### Precision heuristics & health endpoint

* `MODEL_QUANTIZATION` can be `auto` (default), `full`, or `4bit`. Auto enables 4-bit loading when VRAM is low or CPU lacks AVX512/bfloat16 support.
* `/api/health` now reports the active precision & heuristic throughput profile so the UI can surface it in the status bar.


## Concluding note

Various app/model parameter toggles are intentionally hidden in the UI to keep things simple for demo purposes; they can be exposed easily if needed (e.g., temperature, top-p, repetition penalties, token limits, timeouts, context knobs).

Next steps / roadmapfocus on device-, OS-, and cluster-level optimizations, including:

* Edge (pre-)computing for faster local response and reduced bandwidth,
* Federated learning for privacy-preserving adaptation,
* Invertible masking / pseudonymisation to protect sensitive fields during processing.
* These approaches can be explored once implementation requirements and constraints are clearer.
