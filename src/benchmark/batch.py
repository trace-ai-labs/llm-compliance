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

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import httpx
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
    "inkling":  "thinkingmachines/inkling",
    "gptoss":   "openai/gpt-oss-120b",
    "gpt-oss":  "openai/gpt-oss-120b",
    "nemotron": "nvidia/Nemotron-120B-A12B",
    "super":    "nvidia/Nemotron-120B-A12B",
    "ultra":    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
    # Dedicated Baseten deployments (see DEDICATED below) - friendly aliases.
    "mistral":  "mistral-7b-instruct",
    "llama":    "llama-3.3-70b-instruct",
    "llama70b": "llama-3.3-70b-instruct",
    "qwen":     "qwen3.5-35b-a3b",
    "qwen35":   "qwen3.5-35b-a3b",
    "gemma":    "gemma-4-26b",
    "seedoss":  "Seed-OSS-36B-Instruct",
    "seed-oss": "Seed-OSS-36B-Instruct",
    "qwen36":   "Qwen3.6 27B",
    "qwen3.6":  "Qwen3.6 27B",
}

# Baseten DEDICATED deployments (a different product from the shared Model APIs
# above): each is its own scale-to-zero deployment reached at
# model-<id>.api.baseten.co with `Api-Key` auth and the DEPLOYED2 workspace key.
# The pipeline model string (left) names the model in trials/leaderboard; the
# value is the Baseten model_id used to build the endpoint URL. The `model`
# field the vLLM server expects is auto-resolved from its own /v1/models.
DEPLOYED2_ENV = "DEPLOYED2"
DEDICATED_AUTH = "Api-Key"
DEDICATED: Dict[str, str] = {
    "mistral-7b-instruct":    "q04z2013",
    "llama-3.3-70b-instruct": "32p1xx0q",
    "qwen3.5-35b-a3b":        "3m5eny2w",
    "gemma-4-26b":            "q414e75q",
    "Seed-OSS-36B-Instruct":  "wpjz08zw",
    "Qwen3.6 27B":            "qklxd483",
}
_dedicated_names: Dict[str, str] = {}   # model_id -> served vLLM model name (cached)

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

# The reasoning-honesty + unclear-reason TAXONOMY judges (axis 5 and the dataset
# taxonomy figures). Deliberately DISTINCT from JUDGE_TRIO above: JUDGE_TRIO is
# kept unchanged as the provenance record of the sample generators/guards, while
# this trio is what actually grades the honesty/unclear taxonomies. Applied
# leave-one-out (a model never judges its own response); trials from a model that
# is itself in this trio therefore get 2 judges instead of 3.
TAXONOMY_TRIO = ("zai-org/GLM-5.2", "moonshotai/Kimi-K2.6",
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


def _dedicated_url(model_id: str, path: str) -> str:
    return (f"https://model-{model_id}.api.baseten.co"
            f"/environments/production/sync/v1{path}")


def _dedicated_served_name(model_id: str, key: str, timeout: float) -> Optional[str]:
    """The model name the dedicated vLLM server expects (its own /v1/models
    entry). Cached. Returns None if the replica is not ready yet (cold)."""
    with _client_lock:
        cached = _dedicated_names.get(model_id)
    if cached:
        return cached
    r = httpx.get(_dedicated_url(model_id, "/models"),
                  headers={"Authorization": f"{DEDICATED_AUTH} {key}"}, timeout=timeout)
    if r.status_code != 200:
        return None
    data = r.json().get("data") or []
    if not data:
        return None
    name = data[0].get("id")
    with _client_lock:
        _dedicated_names[model_id] = name
    return name


def _call_dedicated(req: BatchRequest, max_retries: int, timeout: float) -> BatchResult:
    """A dedicated Baseten deployment (per-model URL, DEPLOYED2 key, Api-Key
    auth). Tolerant of scale-to-zero cold starts: retries through 'model not
    ready / building / deploying' and timeouts with backoff; fails fast on real
    4xx. Auto-resolves the served model name. Never raises."""
    key = os.environ.get(DEPLOYED2_ENV)
    if not key:
        return BatchResult(req.id, False, "",
                           f"dedicated model {req.model} needs {DEPLOYED2_ENV} in .env",
                           req.model, req.meta)
    model_id = DEDICATED[req.model]
    to = max(timeout, 300.0)              # cold starts can take minutes (esp. 70B)
    retries = max(max_retries, 8)
    url = _dedicated_url(model_id, "/chat/completions")
    headers = {"Authorization": f"{DEDICATED_AUTH} {key}"}
    last = ""
    for attempt in range(retries):
        try:
            name = _dedicated_served_name(model_id, key, min(to, 120.0))
            if not name:
                last = "replica not ready (no /v1/models yet)"
                time.sleep(min(30.0, 5.0 * (attempt + 1)))
                continue
            payload = {"model": name, "messages": req.messages,
                       "temperature": req.temperature, "max_tokens": req.max_tokens}
            r = httpx.post(url, headers=headers, json=payload, timeout=to)
            if r.status_code == 200:
                ch = (r.json().get("choices") or [{}])[0]
                content = (ch.get("message") or {}).get("content")
                if content is None or not str(content).strip():
                    last = f"empty completion (finish_reason={ch.get('finish_reason')})"
                    time.sleep(2.0 * (attempt + 1))
                    continue
                return BatchResult(req.id, True, str(content), "", req.model, req.meta)
            body = r.text[:200]
            low = body.lower()
            if r.status_code == 429:
                last = f"rate limited: {body}"
                time.sleep(min(60.0, 10.0 * (2 ** attempt)))
            elif r.status_code >= 500 or (r.status_code == 400 and
                    ("not ready" in low or "building" in low or "deploying" in low)):
                last = f"cold {r.status_code}: {body}"
                time.sleep(min(60.0, 5.0 * (2 ** attempt)))
            else:  # genuine bad request / auth / not-found - don't retry
                return BatchResult(req.id, False, "",
                                   f"non-retryable {r.status_code}: {body}",
                                   req.model, req.meta)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(min(60.0, 5.0 * (2 ** attempt)))
        except Exception as e:
            return BatchResult(req.id, False, "",
                               f"non-retryable {type(e).__name__}: {e}",
                               req.model, req.meta)
    return BatchResult(req.id, False, "",
                       f"failed after {retries} attempts: {last}", req.model, req.meta)


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
    fails fast on auth/validation errors; never raises. Dedicated Baseten
    deployments route to their own endpoint (DEPLOYED2 key); everything else
    (incl. the gpt-oss extractor) uses the shared endpoint + api_key."""
    if req.model in DEDICATED:
        return _call_dedicated(req, max_retries, timeout)
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
              quiet: bool = False, api_key: Optional[str] = None) -> Dict[str, dict]:
    """Run all requests concurrently, appending results to `out_path` (JSONL).

    Resumable: ids already successful in `out_path` are skipped. Failed rows are
    recorded but retried on the next invocation. Returns {id: result_row} for
    every successful request (old + new).

    `api_key` overrides the default (get_api_key() / BASETEN_API_KEY) for the
    shared endpoint - e.g. DEPLOYED2 for model-under-test calls. Dedicated
    deployments ignore it and use their own DEPLOYED2 auth in _call_dedicated.
    It may also be a callable (req) -> key for PER-REQUEST routing (e.g. route the
    gpt-oss taxonomy judge to BASETEN and the other trio judges to DEPLOYED2).
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

    if api_key is None:
        api_key = get_api_key()
    limiter = _RateLimiter(rpm)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    lock = threading.Lock()
    n_ok = n_fail = 0

    def _work(req: BatchRequest) -> BatchResult:
        limiter.wait()
        k = api_key(req) if callable(api_key) else api_key   # per-request routing
        return call_one(req, k)

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
