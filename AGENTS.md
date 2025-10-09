# Brainbay Local HF Chat — Agent Guidelines

## Scope
These instructions apply to the entire repository unless a nested `AGENTS.md` overrides them.

## Project orientation
- **Backend:** `backend/app/main.py` hosts the FastAPI app that streams completions from a local Hugging Face model. The rest of the backend folder only holds dependency pins.
- **Frontend:** Plain HTML/CSS/JS under `frontend/` powers the chat UI; there is no bundler or framework layer.
- **Domain context:** Text files such as `brainbay_market.txt` act as editable knowledge bases loaded at runtime.
- **Container scripts:** `Dockerfile`, `docker-compose.yml`, and helper scripts bootstrap the stack locally.

Keep these boundaries in mind when adding or moving files; avoid introducing extra build steps unless absolutely necessary.

## Python backend conventions
- Use 4-space indentation, type hints where practical, and prefer descriptive function names.
- Follow the existing pattern of small helper functions plus module-level singletons for model state; do not add global mutable state without a compelling reason.
- Reuse the logging/printing style already present (simple `print(..., flush=True)` statements) for startup or health messages.
- When touching request/response models, update the corresponding Pydantic schemas and ensure FastAPI endpoints continue streaming responses.
- Guard optional dependencies (e.g., bitsandbytes) exactly as shown—never assume they are available.

## Frontend conventions
- Stick to vanilla JavaScript using `const`/`let`; avoid adding frameworks or build tooling.
- Keep indentation at two spaces and follow the existing DOM utility style (`el('#selector')`, functional helpers, top-level state object).
- When updating markdown rendering, sanitisation, or storage logic, maintain backward compatibility with stored conversations in `localStorage`.
- CSS relies on custom properties for light/dark theming; extend the palette via new variables rather than hard-coded colors.

## Data & configuration files
- Context files (`brainbay_*.txt`) should remain UTF-8 text with trailing newlines; document any structural changes in the README.
- JSON content under `frontend/` must stay pretty-printed with two-space indentation and valid UTF-8.
- If you add new environment variables or user-facing modes, update `README.md` so local runners know how to configure them.

## Testing & verification
- At minimum, run `python -m compileall backend` after backend changes to catch syntax errors.
- For runtime checks, prefer `uvicorn backend.app.main:app --host 0.0.0.0 --port 8000` and exercise endpoints manually with the existing frontend.
- The frontend has no automated tests; manually verify major UI flows (new chat, streaming replies, memory toggles) in a browser when UI logic changes.

## Documentation updates
- Keep the README concise—add short subsections if behaviour changes rather than rewriting large sections.
- Note any dependency additions both in the appropriate `requirements*.txt` file and in the README's setup instructions.

When in doubt, favour incremental changes that respect the current lightweight toolchain and local-first deployment story.
