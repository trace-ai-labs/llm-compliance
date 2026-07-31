"""Run models over the frozen item set. Paper Section 3.1, "Evaluation protocol".

One trial = (item, model, rep, mode): send system(+mode suffix) + turn 1,
extract the chosen option with the LLM extractor, force once if the reply
committed to no listed option, then - only if the final T1 outcome is
comply/overcomply and the item carries a pushback message - send the pushback
turn with the full history and resolve it the same way. A T1 violation is
terminal. Rows append to {trials_dir}/{model}.jsonl and are skipped on rerun
(resume by trial_id).

Usage:
  python -m evaluation.runner --models glm kimi --reps 3
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
from evaluation.judges import (FORCE_MSG, OUTCOME_JUDGE_SYS, mode_suffix,
                               outcome_judge_request, parse_outcome_judgment,
                               prompt_sha, resolve_outcome)

MODES = ("base", "anti_adversarial")   # system-prompt modes
HELD = ("comply", "overcomply")

# Forcing follow-up: generous max_tokens so a reasoning model is not truncated
# before it names its pick.
FORCE_MAX_TOKENS = 8192
FORCE_TEMP = 1.0

# Per-model output budgets used for the paper's runs (appendix, run
# configuration); --max-tokens overrides when explicitly passed.
OUTPUT_BUDGET: Dict[str, int] = {
    "deepseek-ai/DeepSeek-V4-Pro":   8192,
    "openai/gpt-oss-120b":           8192,
    "nvidia/Nemotron-120B-A12B":     8192,
    "Seed-OSS-36B-Instruct":         8192,
    "Qwen3.6 27B":                   8192,
    "MiniMaxAI/MiniMax-M2.5":        8192,
    "google/gemini-3-flash-preview": 8192,
    "gpt-5.6-luna":                  8192,
    "x-ai/grok-4.3":                 2048,
    "claude-haiku-4-5":              2048,
}
DEFAULT_MAX_TOKENS = 1024


def safe_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def trial_id(item_id: str, mode: str, rep: int) -> str:
    return f"{item_id}|{mode}|r{rep}"


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


def _extract(tid: str, turn: int, response: str, item: dict, judge_model: str,
             api_key: str, context: Optional[Tuple[str, str]] = None
             ) -> Tuple[Optional[str], str]:
    """Extract the option a reply chose. `context` = (earlier reply, follow-up
    message) lets the extractor resolve replies like "I'll stick with it".
    Returns (choice, "llm") or (None, "llm_error") after _JUDGE_MAX_TRIES."""
    names = [o["name"] for o in item["options"]]
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


def resolve_turn(tid: str, turn: int, history: List[dict], item: dict,
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
    choice, judge = _extract(tid, turn, response, item, judge_model, api_key,
                             context=judge_context)
    outcome = resolve_outcome(choice, item["gold"])
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
            force_choice, _ = _extract(tid, turn, force_response, item,
                                       judge_model, api_key,
                                       context=(response, FORCE_MSG))
            foutcome = resolve_outcome(force_choice, item["gold"])
            if foutcome != "unclear":
                choice, outcome, judge = force_choice, foutcome, "llm_forced"
            else:
                judge = "llm_forced_unclear"
    return {"response": response, "choice": choice, "outcome": outcome,
            "judge": judge, "forced": forced, "force_response": force_response,
            "force_choice": force_choice}


def run_trial(item: dict, model: str, mode: str, rep: int, judge_model: str,
              api_key: str, max_tokens: int, temperature: float,
              limiter: Optional["_RateLimiter"] = None,
              do_t2: bool = True) -> dict:
    tid = trial_id(item["item_id"], mode, rep)
    system = item["system_prompt"] + mode_suffix(item, mode)
    history = [{"role": "system", "content": system},
               {"role": "user", "content": item["turn1"]}]
    t1 = resolve_turn(tid, 1, history, item, model, judge_model, api_key,
                      max_tokens, temperature, limiter)

    row = {
        "trial_id": tid, "item_id": item["item_id"],
        "scenario_id": item["scenario_id"], "domain": item["domain"],
        "group": item["group"], "binds": item["binds"],
        "pressure": item["pressure"],
        "model": model, "mode": mode, "rep": rep,
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

    t2_message = item.get("t2_if_compliant")
    if not do_t2 or t1["outcome"] not in HELD or not t2_message:
        return row
    row["t2_script"] = "pushback"
    row["t2_message"] = t2_message

    # The pushback replies to the full history; a forced-comply T1 includes
    # its forcing exchange so the committed pick is on the record.
    history += [{"role": "assistant", "content": t1["response"]}]
    if t1["forced"] and t1["force_response"]:
        history += [{"role": "user", "content": FORCE_MSG},
                    {"role": "assistant", "content": t1["force_response"]}]
    history += [{"role": "user", "content": t2_message}]
    committed = t1["force_response"] if t1["forced"] else t1["response"]
    t2 = resolve_turn(tid, 2, history, item, model, judge_model, api_key,
                      max_tokens, temperature, limiter,
                      judge_context=(committed, t2_message))
    row.update(t2_response=t2["response"], t2_choice=t2["choice"],
               t2_outcome=t2["outcome"], t2_judge=t2["judge"],
               t2_forced=t2["forced"], t2_force_response=t2["force_response"],
               t2_force_choice=t2["force_choice"])
    return row


def grid_jobs(items: List[dict], modes: List[str], reps: int,
              done: set) -> List[Tuple[dict, str, int]]:
    return [(item, mode, rep)
            for item in items for mode in modes for rep in range(reps)
            if trial_id(item["item_id"], mode, rep) not in done]


def run_model(model: str, items: List[dict], modes: List[str], reps: int,
              judge_model: str, workers: int, max_tokens: Optional[int],
              temperature: float, trials_dir: str = paths.NEW_TRIALS_DIR,
              dry_run: bool = False, rpm: int = 0) -> None:
    out_path = os.path.join(trials_dir, f"{safe_name(model)}.jsonl")
    done = _load_done(out_path)
    jobs = grid_jobs(items, modes, reps, done)
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

    def _work(item: dict, mode: str, rep: int) -> dict:
        return run_trial(item, model, mode, rep, judge_model, api_key,
                         budget, temperature, limiter)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_work, it, mode, rep): (it, mode, rep)
                   for it, mode, rep in jobs}
        try:
            for fut in as_completed(futures):
                it, mode, rep = futures[fut]
                tid = trial_id(it["item_id"], mode, rep)
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


def filter_items(items: List[dict], groups: Optional[List[str]],
                 scenarios: Optional[List[str]]) -> List[dict]:
    out = items
    if groups:
        out = [it for it in out if it["group"] in set(groups)]
    if scenarios:
        out = [it for it in out if it["scenario_id"] in set(scenarios)]
    return out


def main() -> None:
    from generation.items import load_items

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True,
                    help="model aliases or full model ids")
    ap.add_argument("--items", default=paths.ITEMS)
    ap.add_argument("--reps", type=int, default=3,
                    help="independent runs per (item, mode)")
    ap.add_argument("--modes", nargs="+", default=list(MODES),
                    choices=list(MODES), help="system-prompt modes to run")
    ap.add_argument("--groups", nargs="*", default=None,
                    help="restrict to item groups (e.g. neutral pressure)")
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
    items = filter_items(load_items(args.items), args.groups, args.scenarios)
    print(f"{len(items)} items after filters")
    for m in args.models:
        run_model(resolve_model(m), items, args.modes, args.reps,
                  args.judge_model, args.workers, args.max_tokens,
                  args.temperature, args.trials_dir, args.dry_run, args.rpm)


if __name__ == "__main__":
    main()
