"""Batched LLM calls with on-disk resume - the pipeline's only API layer.

Every call goes through the `openai` SDK. Open-weights models are served by
whatever OpenAI-compatible endpoint BENCH_BASE_URL names; the closed models in
EXTERNAL are reached at their own providers. Results append to a JSONL keyed
by request id, and rerunning the same batch skips completed ids.

CLI (smoke test):
  python -m evaluation.batch --model glm --prompt "say hi"
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from openai import (APIConnectionError, APITimeoutError, InternalServerError,
                    OpenAI, RateLimitError)

import paths

try:  # optional; environment variables may already be set
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))
except ImportError:
    pass

BASE_URL_ENV = "BENCH_BASE_URL"
API_KEY_ENV = os.environ.get("BENCH_API_KEY_ENV", "BENCH_API_KEY")

# Transient failures worth retrying. Auth/validation errors fail fast.
RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError,
             InternalServerError)

# Short aliases for CLIs; any full model id the endpoint serves passes
# through unchanged.
MODEL_REGISTRY: Dict[str, str] = {
    "deepseek": "deepseek-ai/DeepSeek-V4-Pro",
    "ds":       "deepseek-ai/DeepSeek-V4-Pro",
    "glm":      "zai-org/GLM-5.2",
    "glm5.2":   "zai-org/GLM-5.2",
    "glm5":     "zai-org/GLM-5",
    "glm4.7":   "zai-org/GLM-4.7",
    "kimi":     "moonshotai/Kimi-K2.6",
    "k2.6":     "moonshotai/Kimi-K2.6",
    "kimi-code": "moonshotai/Kimi-K2.7-Code",
    "inkling":  "thinkingmachines/inkling",
    "gptoss":   "openai/gpt-oss-120b",
    "gpt-oss":  "openai/gpt-oss-120b",
    "nemotron": "nvidia/Nemotron-120B-A12B",
    "super":    "nvidia/Nemotron-120B-A12B",
    "ultra":    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
    "minimax":  "MiniMaxAI/MiniMax-M2.5",
    "mistral":  "mistral-7b-instruct",
    "llama":    "llama-3.3-70b-instruct",
    "llama70b": "llama-3.3-70b-instruct",
    "llama8b":  "llama-3.1-8b-instruct",
    "qwen":     "qwen3.5-35b-a3b",
    "qwen35":   "qwen3.5-35b-a3b",
    "qwen36":   "Qwen3.6 27B",
    "qwen3.6":  "Qwen3.6 27B",
    "gemma":    "gemma-4-26b",
    "seedoss":  "Seed-OSS-36B-Instruct",
    "seed-oss": "Seed-OSS-36B-Instruct",
    # Closed-weight models (see EXTERNAL below).
    "haiku":        "claude-haiku-4-5",
    "claude-haiku": "claude-haiku-4-5",
    "luna":         "gpt-5.6-luna",
    "gpt-luna":     "gpt-5.6-luna",
    "grok":         "x-ai/grok-4.3",
    "grok4.3":      "x-ai/grok-4.3",
    "gemini":       "google/gemini-3-flash-preview",
    "gemini-flash": "google/gemini-3-flash-preview",
    "gemini3":      "google/gemini-3-flash-preview",
}


@dataclass
class ProviderSpec:
    """Connection details and request quirks for a model reachable only at
    its own provider."""
    base_url: str
    key_env: str
    served_model: str                          # slug sent as the "model" field
    extra_body: Dict[str, Any] = field(default_factory=dict)
    use_max_completion_tokens: bool = False    # OpenAI reasoning models reject max_tokens
    send_temperature: bool = True              # False = omit it (default-only models)
    timeout: float = 300.0


# Closed-weight models under evaluation: `call_one` routes them to their own
# provider; every other model goes to the shared BENCH_BASE_URL endpoint.
EXTERNAL: Dict[str, "ProviderSpec"] = {
    "claude-haiku-4-5": ProviderSpec(
        base_url="https://api.anthropic.com/v1/",
        key_env="ANTHROPIC_API_KEY",
        served_model="claude-haiku-4-5-20251001"),
    # requires max_completion_tokens and the default temperature only
    "gpt-5.6-luna": ProviderSpec(
        base_url="https://api.openai.com/v1",
        key_env="OPENAI_API_KEY",
        served_model="gpt-5.6-luna",
        use_max_completion_tokens=True,
        send_temperature=False),
    # run with reasoning disabled
    "x-ai/grok-4.3": ProviderSpec(
        base_url="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_API_KEY",
        served_model="x-ai/grok-4.3",
        extra_body={"reasoning": {"enabled": False}}),
    "google/gemini-3-flash-preview": ProviderSpec(
        base_url="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_API_KEY",
        served_model="google/gemini-3-flash-preview"),
}

# The outcome extractor (a reasoning model: callers keep its max_tokens
# generous so hidden reasoning does not starve the visible label).
DEFAULT_EXTRACT_MODEL = "openai/gpt-oss-120b"

# The taxonomy judge ensemble (the three scenario-generator models), applied
# leave-one-out: a model never judges its own response.
TAXONOMY_TRIO = ("zai-org/GLM-5.2", "moonshotai/Kimi-K2.6",
                 "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B")


def resolve_model(alias_or_id: str) -> str:
    return MODEL_REGISTRY.get(alias_or_id, alias_or_id)


def get_base_url() -> str:
    url = os.environ.get(BASE_URL_ENV)
    if not url:
        raise RuntimeError(
            f"Set {BASE_URL_ENV} to an OpenAI-compatible chat-completions "
            f"endpoint that serves the open-weights models, e.g. "
            f"https://your-provider.example/v1 (any provider works; the "
            f"pipeline only needs /chat/completions)")
    return url


def get_api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"Set {API_KEY_ENV} (or put it in .env)")
    return key


_client_lock = threading.Lock()
_clients: Dict[str, OpenAI] = {}


def _get_client(api_key: str, base_url: Optional[str] = None) -> OpenAI:
    """One SDK client per (base_url, key) pair (thread-safe; shared connection
    pool). base_url defaults to the shared BENCH_BASE_URL endpoint; EXTERNAL
    providers pass their own."""
    burl = base_url or get_base_url()
    ck = f"{burl}||{api_key}"
    with _client_lock:
        client = _clients.get(ck)
        if client is None:
            client = _clients[ck] = OpenAI(base_url=burl, api_key=api_key)
        return client


@dataclass
class BatchRequest:
    """One chat completion. `id` must be unique and stable across reruns -
    it is the resume key."""
    id: str
    model: str
    messages: List[Dict[str, str]]        # includes the system message
    max_tokens: int = 1024
    temperature: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)  # carried into the result row


@dataclass
class BatchResult:
    id: str
    ok: bool
    content: str = ""
    error: str = ""
    model: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class _RateLimiter:
    """Smooth per-process request spacing (no bursts). rpm<=0 disables."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = time.monotonic()

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next)
            self._next = scheduled + self.interval
        delay = scheduled - now
        if delay > 0:
            time.sleep(delay)


def _call_external(req: BatchRequest, max_retries: int, timeout: float) -> BatchResult:
    """Call a model at its own provider (EXTERNAL[req.model]), applying the
    provider's request quirks. Same retry/backoff contract as call_one."""
    spec = EXTERNAL[req.model]
    key = os.environ.get(spec.key_env)
    if not key:
        return BatchResult(req.id, False, "",
                           f"model {req.model} needs {spec.key_env} in .env",
                           req.model, req.meta)
    client = _get_client(key, spec.base_url)
    to = max(timeout, spec.timeout)
    kwargs: Dict[str, Any] = {"model": spec.served_model,
                              "messages": req.messages, "timeout": to}
    if spec.use_max_completion_tokens:
        kwargs["max_completion_tokens"] = req.max_tokens
    else:
        kwargs["max_tokens"] = req.max_tokens
    if spec.send_temperature:
        kwargs["temperature"] = req.temperature
    if spec.extra_body:
        kwargs["extra_body"] = spec.extra_body
    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            content = choice.message.content
            if content is None or not str(content).strip():
                last_err = f"empty completion (finish_reason={choice.finish_reason})"
                time.sleep(2.0 * (attempt + 1))
                continue
            return BatchResult(req.id, True, str(content), "", req.model, req.meta)
        except RateLimitError as e:
            last_err = f"rate limited: {e}"
            time.sleep(min(90.0, 10.0 * (2 ** attempt)))
        except RETRYABLE as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(60.0, 5.0 * (2 ** attempt)))
        except Exception as e:   # auth, bad request, model not found - no retry
            return BatchResult(req.id, False, "",
                               f"non-retryable {type(e).__name__}: {e}",
                               req.model, req.meta)
    return BatchResult(req.id, False, "",
                       f"failed after {max_retries} attempts: {last_err}",
                       req.model, req.meta)


def call_one(req: BatchRequest, api_key: str, max_retries: int = 5,
             timeout: float = 180.0) -> BatchResult:
    """One request with exponential-backoff retries on transient failures;
    fails fast on auth/validation errors; never raises. EXTERNAL models route
    to their own provider; everything else uses the shared endpoint + api_key."""
    if req.model in EXTERNAL:
        return _call_external(req, max_retries, timeout)
    client = _get_client(api_key)
    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=req.model,
                messages=req.messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                timeout=timeout,
            )
            choice = resp.choices[0]
            content = choice.message.content
            if content is None or not str(content).strip():
                last_err = (f"empty completion "
                            f"(finish_reason={choice.finish_reason})")
                time.sleep(2.0 * (attempt + 1))
                continue
            return BatchResult(req.id, True, str(content), "", req.model, req.meta)
        except RateLimitError as e:
            last_err = f"rate limited: {e}"
            time.sleep(min(60.0, 10.0 * (2 ** attempt)))
        except RETRYABLE as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(60.0, 5.0 * (2 ** attempt)))
        except Exception as e:  # auth, bad request, model not found - no retry
            return BatchResult(req.id, False, "",
                               f"non-retryable {type(e).__name__}: {e}",
                               req.model, req.meta)
    return BatchResult(req.id, False, "",
                       f"failed after {max_retries} attempts: {last_err}",
                       req.model, req.meta)


def load_done(out_path: str) -> Dict[str, dict]:
    """Successful rows already on disk, keyed by request id (the resume set)."""
    done: Dict[str, dict] = {}
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ok"):
                done[row["id"]] = row
    return done


def run_batch(requests: List[BatchRequest], out_path: str,
              max_workers: int = 8, rpm: int = 0,
              quiet: bool = False, api_key: Optional[str] = None) -> Dict[str, dict]:
    """Run all requests concurrently, appending results to `out_path` (JSONL).
    Ids already successful there are skipped; failed rows are retried on the
    next invocation. Returns {id: result_row} for every successful request."""
    ids = [r.id for r in requests]
    assert len(ids) == len(set(ids)), "duplicate request ids in batch"

    done = load_done(out_path)
    todo = [r for r in requests if r.id not in done]
    if not quiet:
        print(f"batch: {len(requests)} requests, {len(done)} already done, "
              f"{len(todo)} to run -> {out_path}")
    if not todo:
        return done

    if api_key is None:
        api_key = get_api_key()
    limiter = _RateLimiter(rpm)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    lock = threading.Lock()
    n_ok = n_fail = 0

    def _work(req: BatchRequest) -> BatchResult:
        limiter.wait()
        return call_one(req, api_key)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_work, r): r for r in todo}
        try:
            for fut in as_completed(futures):
                res = fut.result()
                row = {"id": res.id, "ok": res.ok, "content": res.content,
                       "error": res.error, "model": res.model, "meta": res.meta}
                with lock:
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if res.ok:
                        done[res.id] = row
                        n_ok += 1
                    else:
                        n_fail += 1
                    if not quiet:
                        total = n_ok + n_fail
                        mark = "ok" if res.ok else f"FAIL ({res.error[:60]})"
                        print(f"  [{total}/{len(todo)}] {res.id} -> {mark}")
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            print(f"\ninterrupted - {n_ok} ok / {n_fail} failed this run; "
                  f"rerun the same command to resume")
            raise SystemExit(1)

    if not quiet and n_fail:
        print(f"batch done with {n_fail} failures - rerun to retry them")
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="glm", type=resolve_model)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("-o", "--output",
                    default=os.path.join(paths.RESULTS, "batch_smoke.jsonl"))
    args = ap.parse_args()
    reqs = [BatchRequest(id=f"smoke.{args.model}.{abs(hash(args.prompt)) % 10**6}",
                         model=args.model, max_tokens=2048,
                         messages=[{"role": "user", "content": args.prompt}])]
    done = run_batch(reqs, args.output)
    print(done.get(reqs[0].id, {}).get("content", "<no result>"))


if __name__ == "__main__":
    main()
