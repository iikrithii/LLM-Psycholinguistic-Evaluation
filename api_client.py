"""
api_client.py

Groq API client with a multi-key round-robin pool and parallel dispatch.

Design overview:
- All API keys are loaded from GROQ_KEY_1 … GROQ_KEY_N entries in a .env file.
- Keys are assigned round-robin across concurrent calls via a thread-safe lock.
- Keys whose daily token quota is exhausted are tombstoned and skipped automatically.
- parallel_call(specs) fires a batch of API calls concurrently across the key pool,
  yielding throughput proportional to the number of available keys.
- Responses are cached to disk using SHA-256-keyed JSON files so repeated prompts
  do not consume quota.
"""

import os
import json
import time
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from groq import Groq
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE_DIR = Path("results/.cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Models that use internal chain-of-thought reasoning tokens and need special handling.
REASONING_MODELS = {
    "openai/gpt-oss-120b": {"min_tokens": 1024, "no_think": False},
    "qwen/qwen3-32b":      {"min_tokens": 32,   "no_think": True},
}


class TPDError(Exception):
    """Raised when a key's daily token quota (tokens-per-day) is exhausted."""


# ---------------------------------------------------------------------------
# Key pool
# ---------------------------------------------------------------------------

def _load_all_keys() -> list:
    """Read GROQ_KEY_1 … GROQ_KEY_N from environment and return as a list."""
    keys = []
    for i in range(1, 50):
        k = os.getenv(f"GROQ_KEY_{i}")
        if k:
            keys.append(k)
        else:
            break
    if not keys:
        raise ValueError(
            "No Groq API keys found. Set GROQ_KEY_1 (and optionally more) in .env."
        )
    return keys


_pool: list = []
_pool_lock = threading.Lock()
_pool_cursor = 0
_dead_keys: set = set()


def _get_pool() -> list:
    global _pool
    if not _pool:
        _pool = _load_all_keys()
    return _pool


def _live_pool() -> list:
    return [k for k in _get_pool() if k not in _dead_keys]


def _next_key() -> str:
    global _pool_cursor
    live = _live_pool()
    if not live:
        raise RuntimeError("All API keys are TPD-exhausted.")
    with _pool_lock:
        key = live[_pool_cursor % len(live)]
        _pool_cursor += 1
    return key


def _mark_dead(key: str):
    _dead_keys.add(key)
    remaining = len(_live_pool())
    log.warning(f"Key …{key[-10:]} TPD-exhausted. {remaining} key(s) remaining.")


# ---------------------------------------------------------------------------
# Per-key Groq client instances
# ---------------------------------------------------------------------------

_clients: dict = {}


def _get_client(key: str) -> Groq:
    if key not in _clients:
        _clients[key] = Groq(api_key=key)
    return _clients[key]


# ---------------------------------------------------------------------------
# Per-key rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple token-bucket rate limiter — enforces a minimum interval between calls."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self.last_call = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            elapsed = time.time() - self.last_call
            gap = self.interval - elapsed
            if gap > 0:
                time.sleep(gap)
            self.last_call = time.time()


_limiters: dict = {}


def _get_limiter(key: str) -> RateLimiter:
    if key not in _limiters:
        rpm = int(os.getenv("RATE_LIMIT_RPM", "30"))
        _limiters[key] = RateLimiter(rpm)
    return _limiters[key]


# ---------------------------------------------------------------------------
# Disk cache (SHA-256 keyed)
# ---------------------------------------------------------------------------

def _cache_key(model: str, prompt: str, system: str, temperature: float) -> str:
    raw = f"{model}||{system}||{prompt}||{temperature}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_cache(ck: str) -> Optional[dict]:
    p = CACHE_DIR / f"{ck}.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(ck: str, data: dict):
    import tempfile
    p = CACHE_DIR / f"{ck}.json"
    # Atomic write: temp file then rename so concurrent readers never see partial data.
    fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Low-level API call (with exponential backoff retry)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception(lambda e: not isinstance(e, TPDError)),
    reraise=True,
)
def _api_call(client: Groq, model: str, messages: list,
              temperature: float, max_tokens: int):
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        s = str(e).lower()
        if "per day" in s or ("rate limit" in s and "day" in s):
            raise TPDError(str(e)) from e
        raise


def _call_with_key(key: str, model: str, prompt: str, system: str,
                   temperature: float, max_tokens: int) -> dict:
    import re as _re

    _get_limiter(key).wait()
    client = _get_client(key)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]
    try:
        resp = _api_call(client, model, messages, temperature, max_tokens)
    except TPDError:
        _mark_dead(key)
        live = _live_pool()
        if not live:
            return {
                "model": model, "prompt": prompt,
                "response_text": "", "response_raw": "",
                "logprobs": None, "cached": False,
                "error": "All keys TPD-exhausted",
            }
        return _call_with_key(_next_key(), model, prompt, system, temperature, max_tokens)
    except Exception as e:
        log.error(f"API error ({model}): {e}")
        return {
            "model": model, "prompt": prompt,
            "response_text": "", "response_raw": "",
            "logprobs": None, "cached": False,
            "error": str(e),
        }

    text = (resp.choices[0].message.content or "").strip()
    # Strip reasoning chain tokens emitted by thinking-capable models.
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
    return {
        "model": model, "prompt": prompt,
        "response_text": text, "response_raw": text,
        "logprobs": None, "cached": False, "error": None,
    }


# ---------------------------------------------------------------------------
# Public API: single call
# ---------------------------------------------------------------------------

def call_model(
    model: str,
    prompt: str,
    system: str = "You are a helpful linguistic research assistant. Follow instructions precisely.",
    temperature: float = 0.0,
    max_tokens: int = 10,
    use_cache: bool = True,
) -> dict:
    """
    Call a single model and return a result dict with ``response_text`` and ``error``.

    Parameters
    ----------
    model : str
        Model identifier as accepted by the Groq API.
    prompt : str
        User-turn prompt text.
    system : str
        System instruction prepended to every call.
    temperature : float
        Sampling temperature (0.0 = greedy).
    max_tokens : int
        Maximum completion tokens.
    use_cache : bool
        If True, serve cached responses and save new ones to disk.

    Returns
    -------
    dict
        Keys: ``model``, ``prompt``, ``response_text``, ``cached``, ``error``.
    """
    reasoning_cfg = REASONING_MODELS.get(model)
    effective_prompt = prompt
    if reasoning_cfg:
        if reasoning_cfg["no_think"]:
            effective_prompt = prompt + " /no_think"
        max_tokens = max(max_tokens, reasoning_cfg["min_tokens"])

    ck = _cache_key(model, effective_prompt, system, temperature)
    if use_cache:
        cached = _load_cache(ck)
        if cached is not None:
            cached["cached"] = True
            return cached

    key = _next_key()
    result = _call_with_key(key, model, effective_prompt, system, temperature, max_tokens)
    if use_cache and not result["error"]:
        _save_cache(ck, result)
    return result


# ---------------------------------------------------------------------------
# Public API: parallel batch call
# ---------------------------------------------------------------------------

def parallel_call(
    specs: list,
    system: str = "You are a helpful linguistic research assistant. Follow instructions precisely.",
) -> list:
    """
    Fire a list of call specifications concurrently, one worker per available key.

    Parameters
    ----------
    specs : list of dict
        Each dict must contain ``model`` and ``prompt``. Optional keys:
        ``max_tokens`` (default 10), ``temperature`` (default 0.0),
        ``use_cache`` (default True).
    system : str
        System instruction shared across all calls in the batch.

    Returns
    -------
    list of dict
        Result dicts in the same order as ``specs``.
    """
    if not specs:
        return []

    live = _live_pool()
    n_workers = min(len(specs), len(live), 32)

    results = [None] * len(specs)

    def _worker(idx: int, spec: dict) -> tuple:
        m  = spec["model"]
        p  = spec["prompt"]
        mt = spec.get("max_tokens", 10)
        t  = spec.get("temperature", 0.0)
        uc = spec.get("use_cache", True)

        reasoning_cfg = REASONING_MODELS.get(m)
        if reasoning_cfg:
            if reasoning_cfg["no_think"]:
                p = p + " /no_think"
            mt = max(mt, reasoning_cfg["min_tokens"])

        ck = _cache_key(m, p, system, t)
        if uc:
            cached = _load_cache(ck)
            if cached is not None:
                cached["cached"] = True
                return idx, cached

        key = live[idx % len(live)]
        r = _call_with_key(key, m, p, system, t, mt)
        if uc and not r.get("error"):
            _save_cache(ck, r)
        return idx, r

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_worker, i, s) for i, s in enumerate(specs)]
        for fut in as_completed(futures):
            i, res = fut.result()
            results[i] = res

    return results


# ---------------------------------------------------------------------------
# Convenience: call all configured models
# ---------------------------------------------------------------------------

def call_all_models(
    prompt: str,
    system: str = "You are a helpful linguistic research assistant. Follow instructions precisely.",
    temperature: float = 0.0,
    max_tokens: int = 10,
    use_cache: bool = True,
) -> dict:
    """Call all three default models and return {model_id: result_dict}."""
    models = [
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "qwen/qwen3-32b",
    ]
    return {
        m: call_model(m, prompt, system=system, temperature=temperature,
                      max_tokens=max_tokens, use_cache=use_cache)
        for m in models
    }


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def parse_ab_response(text: str) -> Optional[str]:
    """
    Extract 'A' or 'B' from a forced-choice response string.

    Returns the first 'A' or 'B' found (case-insensitive), or None if not found.
    """
    text = text.strip().upper()
    if not text:
        return None
    if text[0] in ("A", "B"):
        return text[0]
    for token in text.split():
        t = token.strip(".,!?:;()")
        if t in ("A", "B"):
            return t
    return None


def parse_rating_response(text: str) -> Optional[float]:
    """Extract a 1-7 Likert rating from a response string. Returns None if not found."""
    import re
    m = re.search(r'\b([1-7])\b', text.strip())
    return float(m.group(1)) if m else None


def parse_generation_response(text: str) -> str:
    """
    Extract a generated word from a free-form response.

    Strips common response prefixes and returns the first whitespace-delimited token.
    """
    text = text.strip()
    for prefix in ["Word:", "Answer:", "Invented word:", "My word:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text.split()[0] if text.split() else text
