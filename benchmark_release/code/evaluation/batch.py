"""Batched LLM calls with on-disk resume - the pipeline's only API layer.

Every call is an OpenAI-SDK chat completion against OpenRouter
(OPENROUTER_API_KEY). Results append to a JSONL keyed by request id;
rerunning the same batch skips completed ids.

CLI (smoke test):
  python -m evaluation.batch --model glm-5.2 --prompt "say hi"
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

try:  # optional; the key may already be in the environment
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))
except ImportError:
    pass

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"

# Transient failures worth retrying. Auth/validation errors fail fast.
RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError,
             InternalServerError)

# Short aliases for CLIs; any full OpenRouter slug passes through unchanged.
MODEL_REGISTRY: Dict[str, str] = {
    "mistral":    "mistralai/mistral-7b-instruct",
    "llama8b":    "meta-llama/llama-3.1-8b-instruct",
    "llama70b":   "meta-llama/llama-3.3-70b-instruct",
    "seed-oss":   "bytedance/seed-oss-36b-instruct",
    "gpt-oss":    "openai/gpt-oss-120b",
    "glm-4.7":    "z-ai/glm-4.7",
    "minimax":    "minimax/minimax-m2.5",
    "qwen3.5":    "qwen/qwen3.5-35b-a3b",
    "glm-5":      "z-ai/glm-5",
    "gemma":      "google/gemma-4-26b",
    "super":      "nvidia/nemotron-3-super",
    "qwen3.6":    "qwen/qwen3.6-27b",
    "deepseek":   "deepseek/deepseek-v4-pro",
    "kimi":       "moonshotai/kimi-k2.6",
    "kimi-code":  "moonshotai/kimi-k2.7-code",
    "ultra":      "nvidia/nemotron-3-ultra",
    "glm-5.2":    "z-ai/glm-5.2",
    "inkling":    "thinkingmachines/inkling",
    "haiku":      "anthropic/claude-haiku-4.5",
    "gemini":     "google/gemini-3-flash-preview",
    "grok":       "x-ai/grok-4.3",
    "luna":       "openai/gpt-5.6-luna",
}

# The outcome extractor (a reasoning model: callers keep its max_tokens
# generous so hidden reasoning does not starve the visible label).
DEFAULT_EXTRACT_MODEL = "openai/gpt-oss-120b"

# The taxonomy judge ensemble (the three scenario-generator models), applied
# leave-one-out: a model never judges its own response.
TAXONOMY_TRIO = ("z-ai/glm-5.2", "moonshotai/kimi-k2.6",
                 "nvidia/nemotron-3-ultra")


def resolve_model(alias_or_id: str) -> str:
    return MODEL_REGISTRY.get(alias_or_id, alias_or_id)


def get_api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"Set {API_KEY_ENV} (or put it in .env)")
    return key


_client_lock = threading.Lock()
_client: Optional[OpenAI] = None


def _get_client(api_key: str) -> OpenAI:
    global _client
    with _client_lock:
        if _client is None:
            _client = OpenAI(base_url=BASE_URL, api_key=api_key)
        return _client


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


def call_one(req: BatchRequest, api_key: str, max_retries: int = 5,
             timeout: float = 300.0) -> BatchResult:
    """One request with exponential-backoff retries on transient failures;
    fails fast on auth/validation errors; never raises."""
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
            time.sleep(min(90.0, 10.0 * (2 ** attempt)))
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
    ap.add_argument("--model", default="glm-5.2", type=resolve_model)
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
