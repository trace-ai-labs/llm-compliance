"""Run models over the dataset. Paper Section 3.1, "Evaluation protocol".

One trial = (sample, model, rep): send the sample's system prompt + turn 1,
extract the chosen option with the LLM extractor, force once if the reply
committed to no listed option, then - only if the final turn-1 outcome is
comply and the sample carries a pushback message - send the pushback turn
with the full history and resolve it the same way. A turn-1 violation is
terminal. Rows append to {trials_dir}/{model}.jsonl.

Usage:
  python -m evaluation.runner --models glm-5.2 kimi --reps 3
  python -m evaluation.runner --models gemini --groups pressure --modes base
  python -m evaluation.runner --models gemini --dry-run
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import paths
from evaluation.batch import (BatchRequest, DEFAULT_EXTRACT_MODEL, _RateLimiter,
                              call_one, get_api_key, resolve_model)
from evaluation.judges import (FORCE_MSG, OUTCOME_JUDGE_SYS,
                               outcome_judge_request, parse_outcome_judgment,
                               prompt_sha, resolve_outcome)

MODES = ("base", "mandate")
HELD = ("comply", "overcomply")

# Forcing follow-up: generous max_tokens so a reasoning model is not truncated
# before it names its pick.
FORCE_MAX_TOKENS = 8192
FORCE_TEMP = 1.0

# Per-model output budgets used for the paper's runs (appendix, run
# configuration); --max-tokens overrides when explicitly passed.
OUTPUT_BUDGET: Dict[str, int] = {
    "deepseek/deepseek-v4-pro":       8192,
    "openai/gpt-oss-120b":            8192,
    "nvidia/nemotron-3-super":        8192,
    "bytedance/seed-oss-36b-instruct": 8192,
    "qwen/qwen3.6-27b":               8192,
    "minimax/minimax-m2.5":           8192,
    "google/gemini-3-flash-preview":  8192,
    "openai/gpt-5.6-luna":            8192,
    "x-ai/grok-4.3":                  2048,
    "anthropic/claude-haiku-4.5":     2048,
}
DEFAULT_MAX_TOKENS = 1024


def safe_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def trial_id(sample_id: str, rep: int) -> str:
    return f"{sample_id}.{rep}"


def _load_done(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["trial_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# Process-wide rate cap for the outcome extractor, which is shared across
# every eval model's trials. Set in main().
_EXTRACT_LIMITER: Optional["_RateLimiter"] = None

_JUDGE_MAX_TRIES = 3


def _extract(tid: str, turn: int, response: str, sample: dict, judge_model: str,
             api_key: str, context: Optional[Tuple[str, str]] = None
             ) -> Tuple[Optional[str], str]:
    """Extract the option a reply chose. `context` = (earlier reply, follow-up
    message) lets the extractor resolve replies like "I'll stick with it".
    Returns (choice, "llm") or (None, "llm_error") after _JUDGE_MAX_TRIES."""
    names = [o["name"] for o in sample["options"]]
    prior, followup = context if context else (None, None)
    req = outcome_judge_request(tid, response, names, judge_model, turn,
                                turn1_response=prior, followup=followup)
    for _ in range(_JUDGE_MAX_TRIES):
        if _EXTRACT_LIMITER:
            _EXTRACT_LIMITER.wait()
        res = call_one(req, api_key)
        if res.ok:
            return parse_outcome_judgment(res.content, names), "llm"
    return None, "llm_error"


def resolve_turn(tid: str, turn: int, history: List[dict], sample: dict,
                 model: str, judge_model: str, api_key: str, max_tokens: int,
                 temperature: float, limiter: Optional["_RateLimiter"],
                 judge_context: Optional[Tuple[str, str]] = None) -> dict:
    """Send one turn and resolve it to an outcome: model reply -> extractor ->
    if unclear, one forcing follow-up (FORCE_MSG) whose extracted outcome
    becomes the turn's outcome. The same path serves turn 1 and the pushback."""
    if limiter:
        limiter.wait()
    res = call_one(BatchRequest(
        id=f"{tid}.t{turn}", model=model, temperature=temperature,
        max_tokens=max_tokens, messages=history), api_key)
    if not res.ok:
        raise RuntimeError(f"T{turn} failed: {res.error}")
    response = res.content
    choice, judge = _extract(tid, turn, response, sample, judge_model, api_key,
                             context=judge_context)
    outcome = resolve_outcome(choice, sample["gold"])
    forced, force_response, force_choice = False, None, None
    if outcome == "unclear":
        if limiter:
            limiter.wait()
        fres = call_one(BatchRequest(
            id=f"{tid}.t{turn}f", model=model, temperature=FORCE_TEMP,
            max_tokens=FORCE_MAX_TOKENS,
            messages=history + [{"role": "assistant", "content": response},
                                {"role": "user", "content": FORCE_MSG}]),
            api_key)
        if fres.ok:
            forced, force_response = True, fres.content
            force_choice, _ = _extract(tid, turn, force_response, sample,
                                       judge_model, api_key,
                                       context=(response, FORCE_MSG))
            foutcome = resolve_outcome(force_choice, sample["gold"])
            if foutcome != "unclear":
                choice, outcome, judge = force_choice, foutcome, "llm_forced"
            else:
                judge = "llm_forced_unclear"
    return {"response": response, "choice": choice, "outcome": outcome,
            "judge": judge, "forced": forced, "force_response": force_response,
            "force_choice": force_choice}


def run_trial(sample: dict, model: str, rep: int, judge_model: str,
              api_key: str, max_tokens: int, temperature: float,
              limiter: Optional["_RateLimiter"] = None,
              do_t2: bool = True) -> dict:
    tid = trial_id(sample["id"], rep)
    history = [{"role": "system", "content": sample["system_prompt"]},
               {"role": "user", "content": sample["turn1"]}]
    t1 = resolve_turn(tid, 1, history, sample, model, judge_model, api_key,
                      max_tokens, temperature, limiter)

    row = {
        "trial_id": tid, "sample": sample["id"], "item": sample["item"],
        "scenario": sample["scenario"], "domain": sample["domain"],
        "group": sample["group"], "binds": sample["binds"],
        "pressure": sample["pressure"],
        "model": model, "mode": sample["mode"], "rep": rep,
        "t1_response": t1["response"], "t1_choice": t1["choice"],
        "t1_outcome": t1["outcome"], "t1_judge": t1["judge"],
        "t1_forced": t1["forced"], "t1_force_response": t1["force_response"],
        "t1_force_choice": t1["force_choice"],
        "t2_script": "none", "t2_message": None, "t2_response": None,
        "t2_choice": None, "t2_outcome": None, "t2_judge": None,
        "t2_forced": False, "t2_force_response": None, "t2_force_choice": None,
        "judge_model": judge_model,
        "judge_prompt_sha": prompt_sha(OUTCOME_JUDGE_SYS), "ts": time.time(),
    }

    t2_message = sample.get("pushback")
    if not do_t2 or t1["outcome"] not in HELD or not t2_message:
        return row
    row["t2_script"] = "pushback"
    row["t2_message"] = t2_message

    # The pushback replies to the full history; a forced-comply turn 1 includes
    # its forcing exchange so the committed pick is on the record.
    history += [{"role": "assistant", "content": t1["response"]}]
    if t1["forced"] and t1["force_response"]:
        history += [{"role": "user", "content": FORCE_MSG},
                    {"role": "assistant", "content": t1["force_response"]}]
    history += [{"role": "user", "content": t2_message}]
    committed = t1["force_response"] if t1["forced"] else t1["response"]
    t2 = resolve_turn(tid, 2, history, sample, model, judge_model, api_key,
                      max_tokens, temperature, limiter,
                      judge_context=(committed, t2_message))
    row.update(t2_response=t2["response"], t2_choice=t2["choice"],
               t2_outcome=t2["outcome"], t2_judge=t2["judge"],
               t2_forced=t2["forced"], t2_force_response=t2["force_response"],
               t2_force_choice=t2["force_choice"])
    return row


def grid_jobs(samples: List[dict], reps: int, done: set) -> List[Tuple[dict, int]]:
    return [(s, rep) for s in samples for rep in range(1, reps + 1)
            if trial_id(s["id"], rep) not in done]


def run_model(model: str, samples: List[dict], reps: int,
              judge_model: str, workers: int, max_tokens: Optional[int],
              temperature: float, trials_dir: str = paths.NEW_TRIALS_DIR,
              dry_run: bool = False, rpm: int = 0) -> None:
    out_path = os.path.join(trials_dir, f"{safe_name(model)}.jsonl")
    done = _load_done(out_path)
    jobs = grid_jobs(samples, reps, done)
    budget = max_tokens or OUTPUT_BUDGET.get(model, DEFAULT_MAX_TOKENS)
    print(f"{model}: {len(jobs)} trials to run "
          f"({len(done)} already done, max_tokens={budget}) -> {out_path}")
    if dry_run or not jobs:
        return

    api_key = get_api_key()
    os.makedirs(trials_dir, exist_ok=True)
    lock = threading.Lock()
    n_done = 0
    limiter = _RateLimiter(rpm)   # caps calls to this model across all workers

    def _work(sample: dict, rep: int) -> dict:
        return run_trial(sample, model, rep, judge_model, api_key,
                         budget, temperature, limiter)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_work, s, rep): (s, rep) for s, rep in jobs}
        try:
            for fut in as_completed(futures):
                s, rep = futures[fut]
                tid = trial_id(s["id"], rep)
                try:
                    row = fut.result()
                except Exception as e:
                    print(f"  [ERROR] {tid}: {e}")
                    continue
                with lock:
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_done += 1
                    print(f"  [{n_done}/{len(jobs)}] {tid} -> "
                          f"T1 {row['t1_outcome']}"
                          + (f", T2 {row['t2_outcome']}" if row["t2_outcome"] else ""))
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            print(f"\ninterrupted - {n_done} trials saved; rerun to resume")
            raise SystemExit(1)


def filter_samples(samples: List[dict], modes: Optional[List[str]],
                   groups: Optional[List[str]],
                   scenarios: Optional[List[str]]) -> List[dict]:
    out = samples
    if modes:
        out = [s for s in out if s["mode"] in set(modes)]
    if groups:
        out = [s for s in out if s["group"] in set(groups)]
    if scenarios:
        out = [s for s in out if s["scenario"] in set(scenarios)]
    return out


def main() -> None:
    from generation.items import load_samples

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True,
                    help="model aliases or full OpenRouter slugs")
    ap.add_argument("--samples", default=paths.PACT)
    ap.add_argument("--reps", type=int, default=3,
                    help="independent runs per sample")
    ap.add_argument("--modes", nargs="+", default=None, choices=list(MODES),
                    help="restrict to one system-prompt mode")
    ap.add_argument("--groups", nargs="*", default=None,
                    help="restrict to sample groups (e.g. neutral pressure)")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--judge-model", default=DEFAULT_EXTRACT_MODEL,
                    type=resolve_model, help="the LLM outcome extractor")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rpm", type=int, default=120,
                    help="max requests/min per model under test; 0 disables")
    ap.add_argument("--extract-rpm", type=int, default=1000,
                    help="max requests/min to the shared extractor; 0 disables")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="output budget for the model under test (default: the "
                         "per-model OUTPUT_BUDGET used for the paper's runs)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--trials-dir", default=paths.NEW_TRIALS_DIR)
    ap.add_argument("--dry-run", action="store_true", help="count trials, no calls")
    args = ap.parse_args()

    global _EXTRACT_LIMITER
    _EXTRACT_LIMITER = _RateLimiter(args.extract_rpm)
    samples = filter_samples(load_samples(args.samples), args.modes,
                             args.groups, args.scenarios)
    print(f"{len(samples)} samples after filters")
    for m in args.models:
        run_model(resolve_model(m), samples, args.reps,
                  args.judge_model, args.workers, args.max_tokens,
                  args.temperature, args.trials_dir, args.dry_run, args.rpm)


if __name__ == "__main__":
    main()
