"""Stage 3 — run models over the frozen item set (two turns, two arms).

One trial = (item, model, rep, arm). The worker sends system+T1, judges T1
(deterministic extractor first, LLM judge on the residual), sends the pushback
T2 when the model complied (a T1 violation is terminal - the challenge branch
is retired), judges T2, and appends one row to
results/benchmark/trials/{model}.jsonl. Trials already on disk are skipped, so
a killed run resumes with the same command.

Arms (axis 4, spec §5): the anti_adversarial arm appends the scenario's own
generated hard directive (item field `hard_directive`, the company-style
guardrail mandate phrased for that bot's role/company;
registry.MAXIMAL_DIRECTIVE as fallback) to the system prompt — nothing else
differs. The optional moderate arm (intensity contrast, spec §5 open
decision) appends MODERATE_DIRECTIVE.

Trivial agents (`trivial:always_comply|always_cheapest|always_escalate|random`)
are synthesized without API calls — the permanent gameability rows.

`--top-up-floor N` implements the spec's power gate: identify the model's
floor-defining cells (bottom-quartile base-arm pressure cells by compliance
rate) from the trials already on disk and add reps until each has ≥N.

Usage:
  python -m src.benchmark.runner --models gemini gpt --reps 5
  python -m src.benchmark.runner --models trivial:always_comply trivial:random
  python -m src.benchmark.runner --models kimi --groups pressure neutral --arms base
  python -m src.benchmark.runner --models gemini --arms base anti_adversarial moderate
  python -m src.benchmark.runner --models gemini --top-up-floor 50
  python -m src.benchmark.runner --models gemini --dry-run
"""

import argparse
import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from src.benchmark.batch import (BatchRequest, DEFAULT_EXTRACT_MODEL, _RateLimiter,
                                 call_one, get_api_key, resolve_model)
from src.benchmark.items import DEFAULT_OUT as ITEMS_PATH
from src.benchmark.items import load_items
from src.benchmark.judges import (OUTCOME_JUDGE_SYS, extract_choice,
                                  outcome_judge_request,
                                  parse_outcome_judgment, prompt_sha,
                                  resolve_outcome)
from src.benchmark.registry import (ARMS, MAXIMAL_DIRECTIVE,
                                    MODERATE_DIRECTIVE)

TRIALS_DIR = os.path.join("results", "benchmark", "trials")

TRIVIAL_PREFIX = "trivial:"
TRIVIAL_AGENTS = ("always_comply", "always_cheapest", "always_escalate", "random")

ALL_ARMS = ("base", "anti_adversarial", "moderate")


def arm_suffix(item: dict, arm: str) -> str:
    """What the arm appends to the item's system prompt. The anti_adversarial
    arm uses the scenario's own generated hard directive (the company-style
    guardrail mandate, phrased for that bot's role and company);
    registry.MAXIMAL_DIRECTIVE is the fallback for items rendered before
    per-scenario directives existed."""
    if arm == "anti_adversarial":
        return item.get("hard_directive") or MAXIMAL_DIRECTIVE
    if arm == "moderate":
        return MODERATE_DIRECTIVE
    return ""

FLOOR_ALPHA = 0.25   # bottom quartile defines the floor set (spec §5 axis 2)


def safe_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def trial_id(item_id: str, arm: str, rep: int) -> str:
    return f"{item_id}|{arm}|r{rep}"


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


# ── Trivial agents (gameability rows; no API) ────────────────────────────────

def trivial_response(policy: str, item: dict, rng: random.Random) -> str:
    options = item["options"]
    compliant = [o["name"] for o in options if o["compliant"]]
    violating = [o["name"] for o in options if not o["compliant"]]
    binds = item["binds"]
    if policy == "always_comply":
        # A degenerate OVER-CAUTIOUS agent that always applies the rule. On a
        # binding cell that is the compliant pick; on a NON-BINDING guard,
        # "applying the rule anyway" is the needless-escalation (gold-violating)
        # option, so it must FAIL rule-scope discernment rather than look like a
        # real discerner. Being binds-aware is what keeps this axis un-gameable.
        pool = compliant if binds else violating
        return pool[0] if pool else "I need to escalate this to a human."
    if policy == "always_cheapest":
        # Always takes the local-objective winner: the rule-violating dominant
        # pick on a binding cell, but the fast just-do-it (gold-compliant) pick
        # on a non-binding guard (there is no rule to pay for).
        pool = violating if binds else compliant
        return pool[0] if pool else options[0]["name"]
    if policy == "always_escalate":
        return "I need to escalate this to a human before proceeding."
    if policy == "random":
        return rng.choice(options)["name"]
    raise ValueError(f"unknown trivial agent: {policy}")


# ── One trial ────────────────────────────────────────────────────────────────

# Process-wide limiter for the gpt-oss extractor. The extractor is a SHARED model
# across every eval model's trials, so its own rpm cap is a global bottleneck;
# this caps extractor calls across the whole run, independent of the per-model
# limiter. Set in main(); only the residual trials (deterministic extraction
# missed) ever hit it, so it rarely binds when models run one at a time.
_EXTRACT_LIMITER: Optional["_RateLimiter"] = None


_JUDGE_MAX_TRIES = 3   # retry the gpt-oss extractor up to 3x on error


def _judge_turn(tid: str, turn: int, response: str, item: dict,
                judge_model: str, api_key: Optional[str],
                turn1_response: Optional[str] = None,
                followup: Optional[str] = None
                ) -> Tuple[Optional[str], str]:
    """Returns (choice, judge_used) where judge_used ∈ rule|llm|llm_error.

    The LLM extractor is ALWAYS the judge for real models: the deterministic
    regex mis-scored replies that discuss substring-overlapping option names
    (e.g. "CardTap" inside "CardTap+Cash"), so it is no longer trusted even as a
    fast path. Trivial agents (api_key is None) emit canned single-option replies
    and stay on the free deterministic extractor. For a T2 reply, pass
    turn1_response/followup so the judge can resolve "I'll stick with it" against
    the actual recommendation. On extractor error we retry up to _JUDGE_MAX_TRIES;
    if it still fails the turn is recorded unclear (judge='llm_error')."""
    names = [o["name"] for o in item["options"]]
    if api_key is None:                     # trivial agents: no API, canned reply
        return extract_choice(response, names), "rule"
    req = outcome_judge_request(tid, response, names, judge_model, turn,
                                turn1_response=turn1_response, followup=followup)
    # The extractor (gpt-oss) ALWAYS runs on BASETEN_API_KEY, even when the model
    # under test is generated via a different key (e.g. DEPLOYED2 via
    # BENCH_API_KEY_ENV). Only the extractor should ever touch BASETEN_API_KEY;
    # all model-under-test generation goes to the configured provider key.
    xkey = os.environ.get("BASETEN_API_KEY") or api_key
    for _ in range(_JUDGE_MAX_TRIES):
        if _EXTRACT_LIMITER:
            _EXTRACT_LIMITER.wait()
        res = call_one(req, xkey)
        if res.ok:
            return parse_outcome_judgment(res.content, names), "llm"
    return None, "llm_error"


def run_trial(item: dict, model: str, arm: str, rep: int, judge_model: str,
              api_key: Optional[str], max_tokens: int,
              temperature: float, limiter: Optional["_RateLimiter"] = None,
              do_t2: bool = True) -> dict:
    tid = trial_id(item["item_id"], arm, rep)
    system = item["system_prompt"] + arm_suffix(item, arm)
    trivial = model.startswith(TRIVIAL_PREFIX)
    rng = random.Random(hash((tid, model)) & 0xFFFFFFFF)

    # ── T1 ──
    if trivial:
        t1_response = trivial_response(model[len(TRIVIAL_PREFIX):], item, rng)
    else:
        if limiter:
            limiter.wait()          # cap calls to the model under test at --rpm
        res = call_one(BatchRequest(
            id=f"{tid}.t1", model=model, temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": item["turn1"]}]), api_key)
        if not res.ok:
            raise RuntimeError(f"T1 failed: {res.error}")
        t1_response = res.content
    t1_choice, t1_judge = _judge_turn(tid, 1, t1_response, item, judge_model,
                                      None if trivial else api_key)
    t1_outcome = resolve_outcome(t1_choice, item["gold"], item["group"])

    row = {
        "trial_id": tid, "item_id": item["item_id"],
        "scenario_id": item["scenario_id"], "domain": item["domain"],
        "group": item["group"], "binds": item["binds"],
        "pressure": item["pressure"], "split": item["split"],
        "model": model, "arm": arm, "rep": rep,
        "t1_response": t1_response, "t1_choice": t1_choice,
        "t1_outcome": t1_outcome, "t1_judge": t1_judge,
        "t2_script": "none", "t2_message": None, "t2_response": None,
        "t2_choice": None, "t2_outcome": None, "t2_judge": None,
        "judge_model": judge_model,
        "judge_prompt_sha": prompt_sha(OUTCOME_JUDGE_SYS), "ts": time.time(),
    }

    # Real-model T2 is generated by `rejudge --rerun-t2` (only where it is scored:
    # comply at T1 after forcing, with the correct post-forcing history), so skip
    # the wasteful first-pass here. Trivial agents keep their synthetic T2 - the
    # rerun-t2 pass skips trivials, so this is their only source.
    if not do_t2 and not trivial:
        return row

    # ── T2 (pushback only) ── The challenge branch is retired: a T1 violation
    # is terminal, nothing downstream ever scored a challenge reply, and its
    # labels were never maintained to the extractor standard. Old trial files
    # still contain t2_script == "challenge" rows; every consumer ignores them.
    held = t1_outcome in ("comply", "overcomply")
    t2_message = item.get("t2_if_compliant") if held else None
    if t2_message is None or t1_outcome == "unclear":
        return row
    row["t2_script"] = "pushback"
    row["t2_message"] = t2_message

    if trivial:
        t2_response = trivial_response(model[len(TRIVIAL_PREFIX):], item, rng)
    else:
        if limiter:
            limiter.wait()
        res = call_one(BatchRequest(
            id=f"{tid}.t2", model=model, temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": item["turn1"]},
                      {"role": "assistant", "content": t1_response},
                      {"role": "user", "content": t2_message}]), api_key)
        if not res.ok:
            raise RuntimeError(f"T2 failed: {res.error}")
        t2_response = res.content
    t2_choice, t2_judge = _judge_turn(tid, 2, t2_response, item, judge_model,
                                      None if trivial else api_key,
                                      turn1_response=t1_response,
                                      followup=t2_message)
    row.update(t2_response=t2_response, t2_choice=t2_choice,
               t2_outcome=resolve_outcome(t2_choice, item["gold"], item["group"]),
               t2_judge=t2_judge)
    return row


# ── Driver ───────────────────────────────────────────────────────────────────

def grid_jobs(items: List[dict], arms: List[str], reps: int,
              done: set) -> List[Tuple[dict, str, int]]:
    """The standard (item × arm × rep) grid."""
    return [(item, arm, rep)
            for item in items for arm in arms for rep in range(reps)
            if trial_id(item["item_id"], arm, rep) not in done]


def floor_topup_jobs(items: List[dict], out_path: str,
                     target_n: int) -> List[Tuple[dict, str, int]]:
    """Power-gate jobs (spec §5/§6): find this model's floor-defining cells —
    the bottom-quartile base-arm pressure cells by T1 compliance rate over the
    trials already on disk — and top each up to ≥target_n reps."""
    per_cell: Dict[str, List[str]] = {}
    max_rep: Dict[str, int] = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("arm") != "base" or row.get("group") != "pressure":
                    continue
                iid = row["item_id"]
                per_cell.setdefault(iid, []).append(row.get("t1_outcome", "unclear"))
                max_rep[iid] = max(max_rep.get(iid, -1), int(row.get("rep", 0)))
    rates = {}
    for iid, outs in per_cell.items():
        decided = [o for o in outs if o != "unclear"]
        if decided:
            rates[iid] = sum(o == "comply" for o in decided) / len(decided)
    if not rates:
        print("  no judged base-arm pressure cells on disk yet — run the grid first")
        return []
    k = max(1, math.ceil(FLOOR_ALPHA * len(rates)))
    floor_set = sorted(sorted(rates, key=lambda i: rates[i])[:k])
    by_id = {it["item_id"]: it for it in items}
    jobs: List[Tuple[dict, str, int]] = []
    for iid in floor_set:
        item = by_id.get(iid)
        if item is None:
            continue
        have = len(per_cell[iid])
        need = target_n - have
        start = max_rep[iid] + 1
        jobs.extend((item, "base", rep) for rep in range(start, start + max(0, need)))
    print(f"  floor set: {len(floor_set)} cells (bottom {FLOOR_ALPHA:.0%}), "
          f"{len(jobs)} top-up trials to reach n>={target_n}")
    return jobs


def run_model(model: str, items: List[dict], arms: List[str], reps: int,
              judge_model: str, workers: int, max_tokens: int,
              temperature: float, trials_dir: str = TRIALS_DIR,
              dry_run: bool = False, top_up_floor: int = 0, rpm: int = 0,
              do_t2: bool = True) -> None:
    out_path = os.path.join(trials_dir, f"{safe_name(model)}.jsonl")
    done = _load_done(out_path)
    if top_up_floor:
        jobs = [(it, arm, rep) for it, arm, rep in floor_topup_jobs(items, out_path, top_up_floor)
                if trial_id(it["item_id"], arm, rep) not in done]
    else:
        jobs = grid_jobs(items, arms, reps, done)
    print(f"{model}: {len(jobs)} trials to run "
          f"({len(done)} already done) -> {out_path}")
    if dry_run or not jobs:
        return

    trivial = model.startswith(TRIVIAL_PREFIX)
    api_key = None if trivial else get_api_key()
    os.makedirs(trials_dir, exist_ok=True)
    lock = threading.Lock()
    n_done = 0
    # one limiter per model: run_model handles a single model at a time, so this
    # caps calls TO THAT MODEL at --rpm (default 120) across all worker threads,
    # which keeps us under a 120 rpm provider cap and avoids failed gens.
    limiter = _RateLimiter(rpm) if not trivial else None

    def _work(item: dict, arm: str, rep: int) -> dict:
        return run_trial(item, model, arm, rep, judge_model, api_key,
                         max_tokens, temperature, limiter, do_t2=do_t2)

    with ThreadPoolExecutor(max_workers=1 if trivial else workers) as ex:
        futures = {ex.submit(_work, it, arm, rep): (it, arm, rep)
                   for it, arm, rep in jobs}
        try:
            for fut in as_completed(futures):
                it, arm, rep = futures[fut]
                tid = trial_id(it["item_id"], arm, rep)
                try:
                    row = fut.result()
                except Exception as e:
                    print(f"  [ERROR] {tid}: {e}")
                    continue
                with lock:
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_done += 1
                    if not trivial or n_done % 200 == 0:
                        print(f"  [{n_done}/{len(jobs)}] {tid} -> "
                              f"T1 {row['t1_outcome']}"
                              + (f", T2 {row['t2_outcome']}" if row["t2_outcome"] else ""))
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            print(f"\ninterrupted — {n_done} trials saved; rerun to resume")
            raise SystemExit(1)


def filter_items(items: List[dict], groups: Optional[List[str]],
                 scenarios: Optional[List[str]], split: Optional[str]) -> List[dict]:
    out = items
    if groups:
        out = [it for it in out if it["group"] in set(groups)]
    if scenarios:
        out = [it for it in out if it["scenario_id"] in set(scenarios)]
    if split:
        out = [it for it in out if it["split"] == split]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True,
                    help="model aliases / OpenRouter ids / trivial:<policy>")
    ap.add_argument("--items", default=ITEMS_PATH)
    ap.add_argument("--reps", type=int, default=3,
                    help="independent runs per (item, arm); each saved as its "
                         "own trial row for variance measurement")
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    choices=list(ALL_ARMS),
                    help="base/directive are the scored arms; moderate is the "
                         "optional intensity contrast")
    ap.add_argument("--top-up-floor", type=int, default=0, metavar="N",
                    help="instead of the grid, top up bottom-quartile pressure "
                         "cells to n>=N reps (power gate)")
    ap.add_argument("--groups", nargs="*", default=None,
                    help="restrict to item groups (e.g. neutral pressure)")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--split", default=None,
                    choices=["dev", "public_test", "private_holdout"])
    ap.add_argument("--judge-model", default=DEFAULT_EXTRACT_MODEL, type=resolve_model,
                    help="Stage-B outcome extractor (small/cheap; tau-gated)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rpm", type=int, default=120,
                    help="max requests/min PER MODEL under test (provider cap is "
                         "120 on some models); 0 disables the limiter")
    ap.add_argument("--extract-rpm", type=int, default=1000,
                    help="max requests/min to the shared gpt-oss extractor "
                         "(global across all models); 0 disables")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--trials-dir", default=TRIALS_DIR)
    ap.add_argument("--dry-run", action="store_true", help="count trials, no calls")
    ap.add_argument("--no-t2", action="store_true",
                    help="T1 only for real models: skip the first-pass conditional T2. "
                         "Use with the final pipeline - `rejudge --rerun-t2` then generates "
                         "the pushback T2 only where it is scored (comply after forcing), with "
                         "no wasted challenge-branch T2. Trivial agents still get their T2.")
    args = ap.parse_args()

    global _EXTRACT_LIMITER
    _EXTRACT_LIMITER = _RateLimiter(args.extract_rpm)
    items = filter_items(load_items(args.items), args.groups, args.scenarios,
                         args.split)
    print(f"{len(items)} items after filters")
    for m in args.models:
        model = m if m.startswith(TRIVIAL_PREFIX) else resolve_model(m)
        if m.startswith(TRIVIAL_PREFIX):
            assert m[len(TRIVIAL_PREFIX):] in TRIVIAL_AGENTS, f"unknown {m}"
        run_model(model, items, args.arms, args.reps, args.judge_model,
                  args.workers, args.max_tokens, args.temperature,
                  args.trials_dir, args.dry_run, args.top_up_floor, args.rpm,
                  do_t2=not args.no_t2)


if __name__ == "__main__":
    main()
