import os, time, threading, pathlib
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import StoppingCriteria, StoppingCriteriaList
try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dependency at runtime
    BitsAndBytesConfig = None

# ====== Speed & device setup ======
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MODEL_QUANTIZATION = os.environ.get("MODEL_QUANTIZATION", "auto").strip().lower()
if MODEL_QUANTIZATION not in {"auto", "full", "4bit"}:
    MODEL_QUANTIZATION = "auto"
MIN_FULL_PRECISION_VRAM_GB = float(os.environ.get("MIN_FULL_PRECISION_VRAM_GB", "8"))
IS_MPS = torch.backends.mps.is_available()
IS_CUDA = torch.cuda.is_available()

if IS_CUDA:
    DEVICE = torch.device("cuda")
    DEVICE_TYPE = "cuda"
elif IS_MPS:
    DEVICE = torch.device("mps")
    DEVICE_TYPE = "mps"
else:
    DEVICE = torch.device("cpu")
    DEVICE_TYPE = "cpu"

if IS_CUDA or IS_MPS:
    DTYPE = torch.float16
else:
    # Prefer bfloat16 when supported, else fall back to float32
    has_bf16 = getattr(torch.backends.cpu, "has_bfloat16", False)
    DTYPE = torch.bfloat16 if has_bf16 else torch.float32

_NUM_THREADS = int(os.environ.get("NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1))))
torch.set_num_threads(_NUM_THREADS)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _env_flag(name: str, default: str = "1") -> bool:
    val = os.environ.get(name, default)
    return str(val).strip().lower() not in {"0", "false", "off", "no"}


OPTIMIZATION_INFO = {
    "device_type": DEVICE_TYPE,
    "dtype": str(DTYPE),
    "quantization": "none",
    "bettertransformer": False,
    "compiled": False,
    "matmul_precision": None,
    "interop_threads": None,
}

# ====== Dynamic inference profile ======
def _infer_profile(model_id: str) -> Dict[str, Any]:
    mid = (model_id or "").lower()
    hint = os.environ.get("MODEL_PROFILE", "").lower().strip()

    profiles = {
        "cpu-small": {
            "name": "cpu-small",
            "model_size": "sub-1B",
            "defaults": {
                "temperature": 0.35,
                "top_p": 0.92,
                "repetition_penalty": 1.05,
                "no_repeat_ngram_size": 3,
                "top_k": 40,
                "max_new_tokens": 384,
                "max_new_cap": 512,
                "max_time_s": 75.0,
                "default_mode": "balanced",
            },
        },
        "gpu-medium": {
            "name": "gpu-medium",
            "model_size": "1B-5B",
            "defaults": {
                "temperature": 0.3,
                "top_p": 0.95,
                "repetition_penalty": 1.05,
                "no_repeat_ngram_size": 3,
                "top_k": 40,
                "max_new_tokens": 640,
                "max_new_cap": 1024,
                "max_time_s": 110.0,
                "default_mode": "balanced",
            },
        },
    }

    if hint in profiles:
        profile = profiles[hint]
    elif "0.5b" in mid or "500m" in mid or "0_5b" in mid:
        profile = profiles["cpu-small"]
    elif IS_CUDA or IS_MPS:
        profile = profiles["gpu-medium"]
    else:
        profile = profiles["cpu-small"]

    max_cap = profile["defaults"].get("max_new_cap", profile["defaults"].get("max_new_tokens", 512))
    balanced = profile["defaults"].get("max_new_tokens", 512)

    def _cap(val: float) -> int:
        return max(64, min(int(val), max_cap))

    profile["modes"] = {
        "fast": {
            "label": "Fast",
            "description": "Shorter replies and tighter time cap for responsiveness.",
            "max_new_tokens": _cap(balanced * 0.6),
            "max_time_s": max(25.0, profile["defaults"].get("max_time_s", 75.0) * 0.65),
            "temperature": max(0.0, profile["defaults"].get("temperature", 0.3) - 0.05),
            "top_p": min(0.98, profile["defaults"].get("top_p", 0.95) + 0.0),
        },
        "balanced": {
            "label": "Balanced",
            "description": "Default blend of speed and depth.",
            "max_new_tokens": balanced,
            "max_time_s": profile["defaults"].get("max_time_s", 90.0),
            "temperature": profile["defaults"].get("temperature", 0.3),
            "top_p": profile["defaults"].get("top_p", 0.95),
        },
        "quality": {
            "label": "Quality",
            "description": "Longer answers and higher time limit for deeper dives.",
            "max_new_tokens": _cap(balanced * 1.3),
            "max_time_s": min(float(max_cap) / 3.0, profile["defaults"].get("max_time_s", 90.0) * 1.4),
            "temperature": min(0.9, profile["defaults"].get("temperature", 0.3) + 0.05),
            "top_p": min(0.99, profile["defaults"].get("top_p", 0.95) + 0.02),
        },
    }

    return profile


PROFILE_INFO = _infer_profile(MODEL_ID)
PROFILE_DEFAULTS = PROFILE_INFO.get("defaults", {})

DEFAULT_TEMPERATURE = float(os.environ.get("GENERATION_TEMPERATURE", PROFILE_DEFAULTS.get("temperature", 0.3)))
DEFAULT_TOP_P = float(os.environ.get("GENERATION_TOP_P", PROFILE_DEFAULTS.get("top_p", 0.95)))
DEFAULT_REPETITION_PENALTY = float(
    os.environ.get("GENERATION_REPETITION_PENALTY", PROFILE_DEFAULTS.get("repetition_penalty", 1.05))
)
DEFAULT_NO_REPEAT_NGRAM = int(
    os.environ.get("GENERATION_NO_REPEAT_NGRAM", PROFILE_DEFAULTS.get("no_repeat_ngram_size", 3))
)
DEFAULT_TOP_K = int(os.environ.get("GENERATION_TOP_K", PROFILE_DEFAULTS.get("top_k", 40)))
DEFAULT_MAX_NEW_TOKENS = int(
    os.environ.get("GENERATION_MAX_NEW_TOKENS", PROFILE_DEFAULTS.get("max_new_tokens", 512))
)
MAX_NEW_CAP = int(PROFILE_DEFAULTS.get("max_new_cap", max(DEFAULT_MAX_NEW_TOKENS, 512)))
DEFAULT_MODE = PROFILE_DEFAULTS.get("default_mode", "balanced")
DEFAULT_MAX_TIME_S = float(
    os.environ.get("GENERATION_MAX_TIME_S", PROFILE_DEFAULTS.get("max_time_s", 200.0))
)

DEFAULT_SYS = (
    "You are a helpful brokerage assistant, proactively helping Brainbay and their affiliates. You are an optimist by\n"
    "nature: when there appears to be doubt, you try to infer what could be useful and you make useful suggestions for next steps."
)

PROFILE_INFO.update(
    {
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "threads": _NUM_THREADS,
        "mps": IS_MPS,
        "cuda": IS_CUDA,
    }
)
PROFILE_INFO.setdefault("optimizations", dict(OPTIMIZATION_INFO))

MODEL_CTX_LIMIT = int(PROFILE_INFO.get("max_context_tokens", 2048))

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
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    deterministic: bool = False
    max_new_tokens: Optional[int] = None
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM
    max_time_s: float = DEFAULT_MAX_TIME_S
    context_key: Optional[str] = "market"
    refine: Optional[str] = "model"
    refine_iters: int = 1
    use_memory: bool = False
    performance_mode: str = DEFAULT_MODE

class ChatResponse(BaseModel):
    content: str
    model_id: str
    generated_tokens: int
    time_ms: int
    stop_reason: Optional[str] = None
    mode: Optional[str] = None

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
        safe = max(5.0, float(limit_s))
        self.limit_s = min(safe, 240.0)  # guardrail
    def __call__(self, input_ids, scores, **kwargs):
        return (time.time() - self.start_t) >= self.limit_s

# ====== Model ======
_MODEL_LOCK = threading.Lock()
_PIPE = None
_TOKENIZER = None
_READY = False
_LAST_ERROR = None
_ACTIVE_PRECISION = "full"
_HARDWARE_STATE: Dict[str, Optional[float]] = {
    "cpu_capability": None,
    "cpu_bf16": None,
    "vram_gb": None,
    "quantization_mode": MODEL_QUANTIZATION,
    "active_precision": _ACTIVE_PRECISION,
}


def _detect_cpu_capability() -> str:
    try:
        return str(torch.backends.cpu.get_cpu_capability())
    except Exception:
        return "unknown"


def _cpu_supports_bf16() -> bool:
    try:
        return bool(getattr(torch.cpu.amp, "bfloat16_is_available", lambda: False)())
    except Exception:
        return False


def _detect_vram_gb() -> Optional[float]:
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return round(props.total_memory / (1024 ** 3), 2)
    except Exception:
        pass
    return None


def _should_quantize(cpu_cap: str, cpu_bf16: bool, vram_gb: Optional[float]) -> bool:
    if MODEL_QUANTIZATION == "4bit":
        return True
    if MODEL_QUANTIZATION == "full":
        return False

    # AUTO mode heuristics
    if vram_gb is not None and vram_gb < MIN_FULL_PRECISION_VRAM_GB:
        return True

    if not torch.cuda.is_available() and not IS_MPS:
        has_avx512 = "AVX512" in (cpu_cap or "").upper()
        if not (has_avx512 and cpu_bf16):
            return True

    return False


def _infer_profile() -> Dict[str, Optional[float]]:
    precision = _ACTIVE_PRECISION
    if precision == "4bit":
        est_tps = 18.0
        latency_ms = 450.0
    else:
        est_tps = 10.0
        latency_ms = 650.0

    if torch.cuda.is_available():
        est_tps *= 1.4
        latency_ms *= 0.7
    elif IS_MPS:
        est_tps *= 1.2
        latency_ms *= 0.85

    profile = {
        "precision": precision,
        "quantization": MODEL_QUANTIZATION,
        "estimated_tokens_per_second": round(est_tps, 2),
        "estimated_latency_ms": round(latency_ms, 1),
        "cpu_capability": _HARDWARE_STATE.get("cpu_capability"),
        "cpu_bf16": _HARDWARE_STATE.get("cpu_bf16"),
        "vram_gb": _HARDWARE_STATE.get("vram_gb"),
    }
    return profile

def _load_pipeline(model_id: str):
    global _PIPE, _TOKENIZER, _READY, _LAST_ERROR, MODEL_CTX_LIMIT, MAX_NEW_CAP, OPTIMIZATION_INFO
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)

        optimization = {
            "device_type": DEVICE_TYPE,
            "dtype": str(DTYPE),
            "quantization": "none",
            "bettertransformer": False,
            "compiled": False,
            "matmul_precision": None,
            "interop_threads": None,
        }

        if DEVICE_TYPE == "cpu":
            matmul_precision = os.environ.get("TORCH_FLOAT32_MATMUL_PRECISION", "medium")
            try:
                torch.set_float32_matmul_precision(matmul_precision)
                optimization["matmul_precision"] = matmul_precision
            except Exception as err:
                print(f"[startup] torch.set_float32_matmul_precision failed: {err}", flush=True)
            interop_env = os.environ.get("NUM_INTEROP_THREADS")
            try:
                interop_threads = int(interop_env) if interop_env is not None else _NUM_THREADS
            except ValueError:
                print(
                    f"[startup] Invalid NUM_INTEROP_THREADS={interop_env!r}, using {_NUM_THREADS}",
                    flush=True,
                )
                interop_threads = _NUM_THREADS
            if hasattr(torch, "set_num_interop_threads"):
                try:
                    torch.set_num_interop_threads(max(1, interop_threads))
                    optimization["interop_threads"] = max(1, interop_threads)
                except Exception as err:
                    print(f"[startup] torch.set_num_interop_threads failed: {err}", flush=True)

        quantization_candidates: List[str] = []
        if DEVICE_TYPE == "cuda" and BitsAndBytesConfig is not None and _env_flag("ENABLE_BNB_QUANTIZATION", "1"):
            preferred = os.environ.get("QUANTIZATION_MODE", "4bit").strip().lower()
            if preferred in {"4bit", "8bit"}:
                quantization_candidates = [preferred, "8bit" if preferred == "4bit" else "4bit"]
            else:
                quantization_candidates = ["4bit", "8bit"]

        model = None
        quantization_error: Optional[Exception] = None
        for mode in quantization_candidates:
            try:
                if mode == "4bit":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                elif mode == "8bit":
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    continue

                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    quantization_config=quant_config,
                    device_map="auto",
                )
                optimization["quantization"] = f"bnb-{mode}"
                break
            except (ValueError, FileNotFoundError, OSError) as err:
                quantization_error = err
                print(
                    f"[startup] Quantized load ({mode}) unavailable, will retry/fallback: {err}",
                    flush=True,
                )
                model = None
            except Exception as err:
                quantization_error = err
                print(f"[startup] Quantized load ({mode}) failed: {err}", flush=True)
                model = None

        if model is None:
            if quantization_candidates and quantization_error is not None:
                print("[startup] Falling back to full-precision weights.", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=DTYPE,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(DEVICE)

        model.eval()

        if _env_flag("ENABLE_BETTERTRANSFORMER", "1") and hasattr(model, "to_bettertransformer"):
            try:
                model = model.to_bettertransformer()
                optimization["bettertransformer"] = True
            except Exception as err:
                print(f"[startup] to_bettertransformer() failed: {err}", flush=True)

        def _torch_version_at_least(major: int, minor: int) -> bool:
            ver_str = torch.__version__.split("+")[0]
            parts = ver_str.split(".")
            try:
                ver_nums = [int(p) for p in parts[:3]]
            except ValueError:
                ver_nums = [0, 0, 0]
            ver_nums += [0] * (3 - len(ver_nums))
            return (ver_nums[0], ver_nums[1]) >= (major, minor)

        if (
            _env_flag("ENABLE_TORCH_COMPILE", "0")
            and hasattr(torch, "compile")
            and _torch_version_at_least(2, 1)
            and optimization.get("quantization") == "none"
        ):
            try:
                model = torch.compile(model)
                optimization["compiled"] = True
            except Exception as err:
                print(f"[startup] torch.compile failed: {err}", flush=True)

        OPTIMIZATION_INFO = optimization
        PROFILE_INFO["optimizations"] = dict(optimization)
        ctx_limit = int(getattr(model.config, "max_position_embeddings", MODEL_CTX_LIMIT))
        MODEL_CTX_LIMIT = ctx_limit
        ctx_cap = max(128, ctx_limit - 128)
        MAX_NEW_CAP = min(MAX_NEW_CAP, ctx_cap)
        PROFILE_INFO["max_context_tokens"] = ctx_limit
        PROFILE_INFO.setdefault("defaults", PROFILE_DEFAULTS)
        PROFILE_INFO["defaults"]["max_new_cap"] = MAX_NEW_CAP
        PROFILE_INFO["defaults"]["max_new_tokens"] = min(
            PROFILE_INFO["defaults"].get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS), MAX_NEW_CAP
        )
        for meta in PROFILE_INFO.get("modes", {}).values():
            if not isinstance(meta, dict):
                continue
            if "max_new_tokens" in meta:
                meta["max_new_tokens"] = int(min(meta["max_new_tokens"], MAX_NEW_CAP))
            if "max_time_s" in meta:
                meta["max_time_s"] = float(max(5.0, meta["max_time_s"]))

        # Ensure eos/pad are set to avoid hanging on some models
        if model.config.eos_token_id is None and tok.eos_token_id is not None:
            model.config.eos_token_id = tok.eos_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tok.eos_token_id or tok.pad_token_id

        if quantize:
            if torch.cuda.is_available():
                pipe_device = 0
            elif IS_MPS:
                pipe_device = DEVICE
            else:
                pipe_device = -1
        else:
            pipe_device = DEVICE

        text_gen = pipeline("text-generation", model=model, tokenizer=tok, device=pipe_device)
        _PIPE = text_gen
        _TOKENIZER = tok
        _READY = True
        _LAST_ERROR = None
        _HARDWARE_STATE.update(
            {
                "cpu_capability": cpu_cap,
                "cpu_bf16": cpu_bf16,
                "vram_gb": vram_gb,
                "quantization_mode": MODEL_QUANTIZATION,
                "active_precision": _ACTIVE_PRECISION,
            }
        )
        msg = f"[startup] Model loaded: {model_id} (precision={_ACTIVE_PRECISION}, device={DEVICE}, MPS={IS_MPS})"
        if vram_gb is not None:
            msg += f" [VRAM: {vram_gb} GB]"
        msg += f" [CPU: {cpu_cap}, bf16={cpu_bf16}]"
        print(msg, flush=True)
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
    max_ctx = MODEL_CTX_LIMIT or 2048
    try:
        prompt_len = len(_TOKENIZER(prompt, add_special_tokens=False).input_ids)
    except Exception:
        prompt_len = 512
    budget = max_ctx - prompt_len - 32
    return max(64, min(MAX_NEW_CAP, budget))

# ====== API ======
@app.get("/api/health")
def health():
    profile = dict(PROFILE_INFO)
    defaults = dict(profile.get("defaults", {}))
    profile_modes = {}
    for key, meta in profile.get("modes", {}).items():
        if isinstance(meta, dict):
            profile_modes[key] = {
                k: meta[k]
                for k in ["label", "description", "max_new_tokens", "max_time_s", "temperature", "top_p"]
                if k in meta
            }
    profile["defaults"] = defaults
    profile["modes"] = profile_modes
    profile.setdefault("max_context_tokens", MODEL_CTX_LIMIT)
    profile.setdefault("default_mode", DEFAULT_MODE)
    if "optimizations" not in profile:
        profile["optimizations"] = dict(OPTIMIZATION_INFO)
    return {
        "ok": True,
        "model_id": MODEL_ID,
        "ready": _READY,
        "last_error": _LAST_ERROR,
        "device": str(DEVICE),
        "mps": IS_MPS,
        "quantization": MODEL_QUANTIZATION,
        "precision": _ACTIVE_PRECISION,
        "profile": _infer_profile(),
        "dtype": str(DTYPE),
        "threads": _NUM_THREADS,
        "mps": IS_MPS,
        "cuda": IS_CUDA,
        "optimizations": dict(OPTIMIZATION_INFO),
        "profile": profile,
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages may not be empty")
    ensure_ready()
    prompt = _format_dialog(req.messages, req.context_key, req.use_memory)

    eos = getattr(_TOKENIZER, "eos_token_id", None)
    pad = getattr(_TOKENIZER, "pad_token_id", eos)

    mode_key = (req.performance_mode or DEFAULT_MODE or "balanced").lower().strip()
    mode_meta = PROFILE_INFO.get("modes", {}).get(mode_key) or PROFILE_INFO.get("modes", {}).get(DEFAULT_MODE, {})

    mode_max_tokens = int(mode_meta.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    mode_max_tokens = max(8, min(mode_max_tokens, MAX_NEW_CAP))
    if req.max_new_tokens is None or req.max_new_tokens <= 0:
        max_new = _auto_max_new_tokens(prompt)
    else:
        max_new = max(8, int(req.max_new_tokens))
    max_new = max(8, min(max_new, mode_max_tokens))

    # Deterministic mode optional; else tuned sampling
    do_sample = not req.deterministic
    if req.deterministic:
        temperature = 0.0
        top_p = 1.0
    else:
        base_temp = max(0.0, req.temperature if req.temperature is not None else DEFAULT_TEMPERATURE)
        base_top_p = max(0.0, min(req.top_p if req.top_p is not None else DEFAULT_TOP_P, 1.0))
        temperature = max(0.0, mode_meta.get("temperature", base_temp))
        top_p = max(0.0, min(mode_meta.get("top_p", base_top_p), 1.0))

    top_k = int(mode_meta.get("top_k", DEFAULT_TOP_K))
    repetition_penalty = float(req.repetition_penalty or DEFAULT_REPETITION_PENALTY)
    repetition_penalty = max(1.0, mode_meta.get("repetition_penalty", repetition_penalty))
    no_repeat = int(req.no_repeat_ngram_size or DEFAULT_NO_REPEAT_NGRAM)
    no_repeat = max(1, int(mode_meta.get("no_repeat_ngram_size", no_repeat)))

    # Time limit stopping criteria
    requested_time = float(req.max_time_s or DEFAULT_MAX_TIME_S)
    mode_time = float(mode_meta.get("max_time_s", DEFAULT_MAX_TIME_S))
    max_time = max(5.0, min(requested_time, mode_time))
    t0 = time.time()
    stop_list = StoppingCriteriaList([TimeLimitCriteria(t0, max_time)])

    params = dict(
        max_new_tokens=max_new,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat,
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
    stop_reason = "timeout" if (time.time() - t0) >= max_time - 0.01 else "eos_or_len"

    content = out.strip()
    try:
        tokens = len(_TOKENIZER(content, add_special_tokens=False).input_ids)
    except Exception:
        tokens = len(content.split())
    return ChatResponse(
        content=content,
        model_id=MODEL_ID,
        generated_tokens=tokens,
        time_ms=dt_ms,
        stop_reason=stop_reason,
        mode=mode_key,
    )

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
    summary_tokens = min(256, max(64, MAX_NEW_CAP // 2))
    out = _PIPE(
        prompt,
        max_new_tokens=summary_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        eos_token_id=eos,
        return_full_text=False,
    )[0]["generated_text"]
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
