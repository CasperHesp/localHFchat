import os, time, threading, pathlib
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import StoppingCriteria, StoppingCriteriaList

# ====== Speed & device setup ======
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
IS_MPS = torch.backends.mps.is_available()
DEVICE = torch.device("mps") if IS_MPS else torch.device("cpu")
DTYPE = torch.float16 if IS_MPS else torch.float32
torch.set_num_threads(int(os.environ.get("NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Global time cap (can be overridden per request)
DEFAULT_MAX_TIME_S = float(os.environ.get("GENERATION_MAX_TIME_S", "200.0"))

DEFAULT_SYS = "You are a helpful brokerage assistant, proactively helping Brainbay and their affiliates. You are an optimist by nature: when there appears to be doubt, you try to infer what could be useful and you make useful suggestions for next steps."

# ====== App ======
app = FastAPI(title="HF Local Chat API (timeouts patch)", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HERE = pathlib.Path(__file__).resolve()
def _root_guess() -> pathlib.Path:
    for p in [HERE.parent, HERE.parent.parent, HERE.parent.parent.parent]:
        if (p / "frontend").is_dir():
            return p
    if pathlib.Path("/app/frontend").is_dir():
        return pathlib.Path("/app")
    return HERE.parent

PROJECT_ROOT = _root_guess()
STATIC_DIR = PROJECT_ROOT / "frontend"
SESSION_MEMORY_PATH = pathlib.Path(os.environ.get("SESSION_MEMORY_PATH", PROJECT_ROOT / "session_memory.txt")).resolve()

def _read_first(paths):
    for p in paths:
        if p and os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return f.read().strip(), p
            except Exception:
                pass
    return "", None

def _load_company() -> str:
    txt, path = _read_first([
        os.environ.get("COMPANY_INFO_PATH"),
        str(PROJECT_ROOT / "brainbay_company.txt"),
        "/app/brainbay_company.txt",
    ])
    if path:
        print(f"[startup] Company info: {path} ({len(txt)} chars)", flush=True)
    else:
        print("[startup] No company info file found.", flush=True)
    return txt

def _load_contexts() -> Dict[str, Dict[str, str]]:
    mapping = {
        "market": {"label":"Market","env":"BRAINBAY_MARKET_PATH","fallbacks":[str(PROJECT_ROOT/"brainbay_market.txt"),"/app/brainbay_market.txt"]},
        "geography": {"label":"Geography","env":"BRAINBAY_GEOGRAPHY_PATH","fallbacks":[str(PROJECT_ROOT/"brainbay_geography.txt"),"/app/brainbay_geography.txt"]},
        "matching": {"label":"Matching","env":"BRAINBAY_MATCHING_PATH","fallbacks":[str(PROJECT_ROOT/"brainbay_matching.txt"),"/app/brainbay_matching.txt"]},
    }
    out = {}
    for k, meta in mapping.items():
        txt, path = _read_first([os.environ.get(meta["env"], "")] + meta["fallbacks"])
        out[k] = {"label": meta["label"], "text": txt, "path": path or ""}
        print(f"[startup] Context '{k}': {'OK' if path else 'missing'}", flush=True)
    return out

COMPANY_INFO = _load_company()
CONTEXTS = _load_contexts()

# ====== Schemas ======
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    # tuned defaults (hidden from UI)
    temperature: float = 0.3
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    deterministic: bool = False
    max_new_tokens: Optional[int] = None
    no_repeat_ngram_size: int = 3
    max_time_s: float = DEFAULT_MAX_TIME_S
    context_key: Optional[str] = "market"
    refine: Optional[str] = "model"
    refine_iters: int = 1
    use_memory: bool = False

class ChatResponse(BaseModel):
    content: str
    model_id: str
    generated_tokens: int
    time_ms: int
    stop_reason: Optional[str] = None

class SummarizeRequest(BaseModel):
    messages: List[Message]
    context_key: Optional[str] = "market"
    use_memory: bool = True

class SummarizeResponse(BaseModel):
    summary: str

# ====== Stopping criteria ======
class TimeLimitCriteria(StoppingCriteria):
    def __init__(self, start_t: float, limit_s: float):
        self.start_t = start_t
        self.limit_s = max(3.0, min(60.0, float(limit_s)))  # clamp for safety
    def __call__(self, input_ids, scores, **kwargs):
        return (time.time() - self.start_t) >= self.limit_s

# ====== Model ======
_MODEL_LOCK = threading.Lock()
_PIPE = None
_TOKENIZER = None
_READY = False
_LAST_ERROR = None

def _load_pipeline(model_id: str):
    global _PIPE, _TOKENIZER, _READY, _LAST_ERROR
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.eval()
        model.to(DEVICE)
        # Ensure eos/pad are set to avoid hanging on some models
        if model.config.eos_token_id is None and tok.eos_token_id is not None:
            model.config.eos_token_id = tok.eos_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tok.eos_token_id or tok.pad_token_id

        text_gen = pipeline("text-generation", model=model, tokenizer=tok, device=DEVICE)
        _PIPE = text_gen
        _TOKENIZER = tok
        _READY = True
        _LAST_ERROR = None
        print(f"[startup] Model loaded: {model_id} on {DEVICE} (MPS={IS_MPS})", flush=True)
    except Exception as e:
        _READY = False
        _LAST_ERROR = str(e)
        print(f"[startup] Model load FAILED: {e}", flush=True)
        raise

def ensure_ready():
    if _READY: return
    with _MODEL_LOCK:
        if not _READY:
            _load_pipeline(MODEL_ID)

def _read_session_memory() -> str:
    try:
        if SESSION_MEMORY_PATH.exists():
            return SESSION_MEMORY_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""

def _format_dialog(messages: List[Message], key: Optional[str], use_memory: bool) -> str:
    sys_texts = [m.content for m in messages if m.role.lower().strip() == "system"]
    base = "\n\n".join(sys_texts) if sys_texts else DEFAULT_SYS
    if COMPANY_INFO:
        base = f"{base}\n\n[Company context]\n{COMPANY_INFO}"
    k = (key or "market").lower()
    ctx = CONTEXTS.get(k, {}).get("text", "")
    if ctx:
        label = CONTEXTS.get(k, {}).get("label", k.title())
        base = f"{base}\n\n[{label} context]\n{ctx}"
    if use_memory:
        mem = _read_session_memory()
        if mem:
            base = f"{base}\n\n[Session memory]\n{mem}"

    merged = [{"role":"system","content":base}] + [
        {"role": m.role, "content": m.content}
        for m in messages if m.role.lower().strip() != "system"
    ]
    try:
        return _TOKENIZER.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = [f"[SYSTEM] {base}"]
        for m in messages:
            r = m.role.lower().strip()
            if r == "user":
                parts.append(f"[USER] {m.content.strip()}")
            elif r == "assistant":
                parts.append(f"[ASSISTANT] {m.content.strip()}")
        parts.append("[ASSISTANT] ")
        return "\n".join(parts)

def _auto_max_new_tokens(prompt: str) -> int:
    try:
        max_ctx = int(getattr(_PIPE.model.config, "max_position_embeddings", 2048))
    except Exception:
        max_ctx = 2048
    try:
        prompt_len = len(_TOKENIZER(prompt, add_special_tokens=False).input_ids)
    except Exception:
        prompt_len = 512
    budget = max_ctx - prompt_len - 16
    return max(64, min(1024, budget))

# ====== API ======
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model_id": MODEL_ID,
        "ready": _READY,
        "last_error": _LAST_ERROR,
        "device": str(DEVICE),
        "mps": IS_MPS
    }

class ChatResponse(BaseModel):
    content: str
    model_id: str
    generated_tokens: int
    time_ms: int
    stop_reason: Optional[str] = None

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages may not be empty")
    ensure_ready()
    prompt = _format_dialog(req.messages, req.context_key, req.use_memory)

    eos = getattr(_TOKENIZER, "eos_token_id", None)
    pad = getattr(_TOKENIZER, "pad_token_id", eos)

    if req.max_new_tokens is None or req.max_new_tokens <= 0:
        max_new = _auto_max_new_tokens(prompt)
    else:
        max_new = max(8, min(req.max_new_tokens, 1024))

    # Deterministic mode optional; else tuned sampling
    do_sample = not req.deterministic
    temperature = 0.0 if req.deterministic else max(0.0, req.temperature)
    top_p = 1.0 if req.deterministic else max(0.0, min(req.top_p, 1.0))

    # Time limit stopping criteria
    t0 = time.time()
    stop_list = StoppingCriteriaList([TimeLimitCriteria(t0, req.max_time_s or DEFAULT_MAX_TIME_S)])

    params = dict(
        max_new_tokens=max_new,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=40,
        repetition_penalty=max(1.0, req.repetition_penalty),
        no_repeat_ngram_size=max(1, int(req.no_repeat_ngram_size or 3)),
        eos_token_id=eos,
        pad_token_id=pad,
        early_stopping=True,
        return_full_text=False,
        use_cache=True,
        stopping_criteria=stop_list
    )

    out = _PIPE(prompt, **params)[0]["generated_text"]
    dt_ms = int((time.time() - t0) * 1000)

    # Heuristic: detect if we hit the time limit
    stop_reason = "timeout" if (time.time() - t0) >= (req.max_time_s or DEFAULT_MAX_TIME_S) - 0.01 else "eos_or_len"

    content = out.strip()
    tokens = len(content.split())
    return ChatResponse(content=content, model_id=MODEL_ID, generated_tokens=tokens, time_ms=dt_ms, stop_reason=stop_reason)

class SummarizeRequest(BaseModel):
    messages: List[Message]
    context_key: Optional[str] = "market"
    use_memory: bool = True

class SummarizeResponse(BaseModel):
    summary: str

@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    ensure_ready()
    instruction = "Summarize the conversation into concise bullet points (6–10 max) highlighting facts, decisions, blockers, and next actions. Keep it compact."
    msgs = [{"role":"system","content":DEFAULT_SYS}] + [m.dict() for m in req.messages] + [{"role":"user","content":instruction}]
    try:
        prompt = _TOKENIZER.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = "\n".join([f"[{m['role'].upper()}] {m['content']}" for m in msgs]) + "\n[ASSISTANT] "
    eos = getattr(_TOKENIZER, "eos_token_id", None)
    out = _PIPE(prompt, max_new_tokens=256, do_sample=False, temperature=0.0, top_p=1.0, repetition_penalty=1.0, eos_token_id=eos, return_full_text=False)[0]["generated_text"]
    return SummarizeResponse(summary=out.strip())

@app.post("/api/memory/append")
def memory_append(text: str):
    t = (text or "").strip()
    if not t:
        return {"ok": True, "bytes": 0}
    SESSION_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SESSION_MEMORY_PATH, "a", encoding="utf-8") as f:
        f.write("\n\n" + t)
    return {"ok": True, "bytes": len(t)}

@app.get("/api/memory", response_class=PlainTextResponse)
def memory_read():
    if not SESSION_MEMORY_PATH.exists():
        return PlainTextResponse("", status_code=200)
    try:
        return SESSION_MEMORY_PATH.read_text(encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
def _warm():
    threading.Thread(target=lambda: ensure_ready(), daemon=True).start()

# Static & root
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=False), name="static")

@app.get("/", response_class=HTMLResponse)
def _root():
    if (STATIC_DIR / "index.html").is_file():
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse("<h1>HF Local Chat API</h1><p>Frontend not found. Ensure 'frontend/' exists.</p>")
