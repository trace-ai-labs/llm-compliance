"""Batched LLM calls with on-disk resume — the pipeline's only API layer.

OpenAI-shaped in and out: every call goes through the `openai` SDK pointed at
an OpenAI-compatible endpoint (default: Baseten, `BASETEN_API_KEY`). Swap
providers by setting BENCH_BASE_URL / BENCH_API_KEY_ENV — nothing else in the
pipeline knows or cares.

Every stage (sample generation, evaluation runs, post-hoc judging) builds a
list of `BatchRequest`s and hands it to `run_batch`. Results append to a JSONL
keyed by request id; re-running the same batch skips completed ids, so any
stage can be killed and resumed with the same command.

CLI (smoke test):
  python -m src.benchmark.batch --model glm --prompt "say hi"
"""

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from openai import (APIConnectionError, APITimeoutError, InternalServerError,
                    OpenAI, RateLimitError)

try:  # optional; environment variables may already be set
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))
except ImportError:
    pass

BASE_URL = os.environ.get("BENCH_BASE_URL", "https://inference.baseten.co/v1")
API_KEY_ENV = os.environ.get("BENCH_API_KEY_ENV", "BASETEN_API_KEY")

# Transient failures worth retrying. Auth/validation errors fail fast.
RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError,
             InternalServerError)

# Short aliases for CLIs; any full slug on the endpoint passes through
# unchanged. `gptoss` and `nemotron` are reasoning models: they spend hidden
# reasoning tokens before the visible answer, so give them generous
# max_tokens or content can come back empty with finish_reason="length".
MODEL_REGISTRY: Dict[str, str] = {
    "deepseek": "deepseek-ai/DeepSeek-V4-Pro",
    "ds":       "deepseek-ai/DeepSeek-V4-Pro",
    "glm":      "zai-org/GLM-5.2",
    "glm5.2":   "zai-org/GLM-5.2",
    "glm5.1":   "zai-org/GLM-5.1",
    "glm5":     "zai-org/GLM-5",
    "glm4.7":   "zai-org/GLM-4.7",
    "kimi":     "moonshotai/Kimi-K2.6",
    "k2.6":     "moonshotai/Kimi-K2.6",
    "k2.5":     "moonshotai/Kimi-K2.5",
    "kimi-code": "moonshotai/Kimi-K2.7-Code",
    "gptoss":   "openai/gpt-oss-120b",
    "gpt-oss":  "openai/gpt-oss-120b",
    "nemotron": "nvidia/Nemotron-120B-A12B",
    "super":    "nvidia/Nemotron-120B-A12B",
    "ultra":    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
}

# Non-reasoning instruct model: judge prompts cap max_tokens at ~24, which a
# reasoning model would burn on hidden thought and return nothing.
DEFAULT_JUDGE_MODEL = "moonshotai/Kimi-K2.5"

# Outcome extraction (Stage-B judge) only runs on the deterministic extractor's
# residual and is a near-objective "map this reply to one of these visible
# labels, or ESCALATE/UNCLEAR" task — so it uses a small, cheap non-reasoning
# model. It is protected by the judge-swap Kendall-tau gate. Reasoning-honesty
# classification (axis 5) is subtler and instead uses the JUDGE_TRIO ensemble.
# gpt-oss-120b is a reasoning model; the extractor's max_tokens must stay
# generous enough (see runner/judges callers) that hidden reasoning doesn't
# starve the visible label.
DEFAULT_EXTRACT_MODEL = "openai/gpt-oss-120b"

# The three sample generators, reused as the reasoning-honesty judge ensemble.
# Applied leave-one-out: a model never judges its own response (see
# judges.cmd_classify_honesty). Sample authorship is NOT a conflict — the judge
# grades a model-under-test's reply, not the pack it may have generated.
JUDGE_TRIO = ("zai-org/GLM-5.2", "moonshotai/Kimi-K2.6",
              "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B")


def resolve_model(alias_or_id: str) -> str:
    return MODEL_REGISTRY.get(alias_or_id, alias_or_id)


def get_api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"Set {API_KEY_ENV} (or put it in .env)")
    return key


_client_lock = threading.Lock()
_clients: Dict[str, OpenAI] = {}


def _get_client(api_key: str) -> OpenAI:
    """One SDK client per key (thread-safe; shared connection pool)."""
    with _client_lock:
        client = _clients.get(api_key)
        if client is None:
            client = _clients[api_key] = OpenAI(base_url=BASE_URL,
                                                api_key=api_key)
        return client


@dataclass
class BatchRequest:
    """One chat completion. `id` must be unique and stable across reruns —
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
             timeout: float = 180.0) -> BatchResult:
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
            time.sleep(min(60.0, 10.0 * (2 ** attempt)))
        except RETRYABLE as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(60.0, 5.0 * (2 ** attempt)))
        except Exception as e:  # auth, bad request, model not found — no retry
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
              on_result: Optional[Callable[[BatchResult], None]] = None,
              quiet: bool = False) -> Dict[str, dict]:
    """Run all requests concurrently, appending results to `out_path` (JSONL).

    Resumable: ids already successful in `out_path` are skipped. Failed rows are
    recorded but retried on the next invocation. Returns {id: result_row} for
    every successful request (old + new).
    """
    ids = [r.id for r in requests]
    assert len(ids) == len(set(ids)), "duplicate request ids in batch"

    done = load_done(out_path)
    todo = [r for r in requests if r.id not in done]
    if not quiet:
        print(f"batch: {len(requests)} requests, {len(done)} already done, "
              f"{len(todo)} to run -> {out_path}")
    if not todo:
        return done

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
                if on_result is not None:
                    on_result(res)
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            print(f"\ninterrupted — {n_ok} ok / {n_fail} failed this run; "
                  f"rerun the same command to resume")
            raise SystemExit(1)

    if not quiet and n_fail:
        print(f"batch done with {n_fail} failures — rerun to retry them")
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="glm", type=resolve_model)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("-o", "--output", default=os.path.join("results", "benchmark",
                                                           "batch_smoke.jsonl"))
    args = ap.parse_args()
    reqs = [BatchRequest(id=f"smoke.{args.model}.{abs(hash(args.prompt)) % 10**6}",
                         model=args.model, max_tokens=2048,
                         messages=[{"role": "user", "content": args.prompt}])]
    done = run_batch(reqs, args.output)
    print(done.get(reqs[0].id, {}).get("content", "<no result>"))


if __name__ == "__main__":
    main()
