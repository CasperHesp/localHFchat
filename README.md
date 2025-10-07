# Brainbay Local HF Chat — Short Manual

## Build, start & run (Docker, recommended)

```bash
# Build & start (publishes a RANDOM host port for container port 8000)
docker compose up --build -d

# Find the URL to open
docker compose port chatapp 8000
# -> http://localhost:XXXXX
```

## What it is

A **fully local** chat app: **FastAPI** backend + minimalist **vanilla** frontend, powered by a **public Hugging Face** chat model. It’s Brainbay-aware via simple text files, supports small models, and includes UX niceties (Markdown, spinner, Stop, intro/kickstart prompts, session memory). Designed to run in Docker or natively on macOS (Apple Silicon with MPS).

---

## Project structure — and why each file exists

```
project-root/
  backend/app/main.py
  frontend/
    index.html
    style.css
    app.js
    conversation_starters.json
    intro_markdown.json
  brainbay_company.txt
  brainbay_market.txt
  brainbay_geography.txt
  brainbay_matching.txt
  session_memory.txt
  Dockerfile
  docker-compose.yml
  run_macos.sh
  requirements*.txt
```

* **backend/app/main.py** — FastAPI app. Loads a small HF chat model; merges context **as data** (system → company → selected dropdown file → optional session memory); adds guardrails (timeout, no-repeat, tuned sampling); serves `/static` + UI.
* **frontend/index.html / style.css / app.js** — Small custom UI: Markdown render, “Thinking…” spinner, **Stop**, intro markdown (per context), **10s kickstart** prompts, export/import history, dark mode, session memory (summarize & save + download).
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

## Technologies used

* **Backend:** FastAPI + Uvicorn, Hugging Face **Transformers**, **PyTorch** (+ Apple **MPS** on macOS), Accelerate
* **Frontend:** custom vanilla **HTML/CSS/JS** (no UI framework)
* **Container:** Docker, docker-compose

---

## Creative choices

* **Context as data** — Behavior steered by editable text/JSON files; no rebuilds to change domain knowledge.
* **Small-model ergonomics** — Tuned decoding (`temperature=0.3`, `top_p=0.95`, `top_k=40`, `repetition_penalty=1.05`), **auto token budgeting**, `no_repeat_ngram_size=3`, **timeout guard** (default 12s) to reduce stalls/loops.
* **Helpful UX** — Markdown, spinner, **Stop**, intro per context, **10s kickstart**, export/import, dark mode, session memory with summarization.

---

## macOS option (+GPU/MPS)

```bash
# (optional) choose a small public HF model
export MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"

python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt   # or: requirements-macos.txt

# run on a free port
PORT=$(python - <<'PY'
import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()
PY
)
export PYTORCH_ENABLE_MPS_FALLBACK=1
uvicorn backend.app.main:app --host 0.0.0.0 --port "$PORT"
# open http://localhost:$PORT
```


## Concluding note

Several app/model parameter toggles are intentionally hidden in the UI to keep things simple for demo purposes; they can be exposed easily if needed (e.g., temperature, top-p, repetition penalties, token limits, timeouts, context knobs).

Next steps / roadmapfocus on device-, OS-, and cluster-level optimizations, including:

* Edge (pre-)computing for faster local response and reduced bandwidth,

* Federated learning for privacy-preserving adaptation,

* Invertible masking / pseudonymisation to protect sensitive fields during processing.

* These approaches can be explored once implementation requirements and constraints are clearer.
