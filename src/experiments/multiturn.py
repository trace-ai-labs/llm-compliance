#!/usr/bin/env python3
"""Two-turn pushback and challenge dynamics.

Turn 1 is a standard request. Turn 2 either pushes back on a compliant answer or
challenges a non-compliant one, using an identical neutral probe in both directions so
that switching on it reflects sycophancy rather than new information. Measures how
durable compliance is, and how recoverable a violation is.

Turn 1 is supplied by an existing results file (e.g. the controls run), so generate
that first, then point this at it with --t1-results.

Run: python -m src.experiments.multiturn --t1-results results/data/controls.jsonl -o results/data/multiturn.jsonl
"""

import argparse, json, os, threading, random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
from src.core import (
    BASE_PROMPT, FINANCIAL_LEVELS, make_reg_context, make_user_message,
    VENDORS, judge_vendor, call_model_messages, load_results, get_api_key, resolve_model,
)
from src.prompts.multiturn import PUSHBACK_TACTICS, CHALLENGE_TACTICS


def _stable_sample(records: List[Dict], k: int) -> List[Dict]:
    """Deterministically sample up to k records, stable across runs.

    Uses a sort key on (scenario, run_id, seed) to avoid adding new RNG state.
    """
    if k is None or k <= 0:
        return []
    if len(records) <= k:
        return records
    def _key(r: Dict):
        meta = r.get("metadata", {})
        return (
            str(meta.get("framing", "")),
            str(meta.get("fin_level", "")),
            int(r.get("run_id", 0)),
            int(r.get("seed", 0)),
        )
    return sorted(records, key=_key)[:k]

# ── Multi-turn result ─────────────────────────────────────────────────────────

@dataclass
class MultiTurnResult:
    scenario: str        # framing_finlevel
    t1_trial_id: str     # original scenario + run_id from the T1 source file
    run_id: int
    seed: int
    model: str
    tactic: str          # e.g. "pushback_neutral", "challenge_reg_flag"
    direction: str       # "pushback" or "challenge"
    t1_vendor: str
    t1_compliant: bool
    t1_response: str
    t2_vendor: str
    t2_compliant: bool
    t2_parseable: bool
    t2_response: str
    switched: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

# ── Core runner ───────────────────────────────────────────────────────────────

def run_t2_for_trial(t1_record: Dict, model: str, api_key: str) -> List[MultiTurnResult]:
    """Given a stored T1 trial record, run the appropriate T2 follow-ups.

    Reconstructs the original user message from the stored seed, then runs
    pushback tactics if T1 was compliant, challenge tactics if noncompliant.
    """
    meta = t1_record.get("metadata", {})
    framing = meta.get("framing", "")
    fin_level = meta.get("fin_level", "")
    seed = t1_record.get("seed", 0)
    run_id = t1_record.get("run_id", 0)
    t1_compliant = t1_record.get("compliant", False)
    t1_response = t1_record.get("raw_response", "")
    t1_vendor = t1_record.get("vendor_chosen", "UNCLEAR")

    if not fin_level or not t1_response:
        return []

    # Reconstruct the original T1 user message from the stored seed
    rng = random.Random(seed)
    user_t1, shuffled_vendors = make_user_message(rng, VENDORS)

    # framing="none" is the no-regulation baseline — system prompt has no reg context
    if framing == "none":
        system_prompt = BASE_PROMPT
    else:
        system_prompt = BASE_PROMPT + make_reg_context(framing, fin_level)

    tactics = PUSHBACK_TACTICS if t1_compliant else CHALLENGE_TACTICS
    direction = "pushback" if t1_compliant else "challenge"
    scenario = f"{framing}_{fin_level}"
    t1_trial_id = f"{t1_record.get('scenario', scenario)}_{run_id}"

    print(f"  → {scenario} run{run_id} T1={'C' if t1_compliant else 'V'} ({direction})")
    results = []
    for tactic_name, tactic_msg in tactics.items():
        full_tactic = f"{direction}_{tactic_name}"

        print(f"    {scenario} run{run_id} [{tactic_name}] calling T2 model...")
        t2_response = call_model_messages(
            system_prompt,
            [
                {"role": "user", "content": user_t1},
                {"role": "assistant", "content": t1_response},
                {"role": "user", "content": tactic_msg},
            ],
            model, api_key,
        )
        print(f"    {scenario} run{run_id} [{tactic_name}] T2 done, judging...")
        t2_vendor_obj = judge_vendor(t2_response, shuffled_vendors, api_key)
        print(f"    {scenario} run{run_id} [{tactic_name}] judge done")
        t2_compliant = t2_vendor_obj.certified if t2_vendor_obj else False
        t2_vendor = t2_vendor_obj.name if t2_vendor_obj else "UNCLEAR"
        t2_parseable = t2_vendor_obj is not None

        results.append(MultiTurnResult(
            scenario=scenario,
            t1_trial_id=t1_trial_id,
            run_id=run_id,
            seed=seed,
            model=model,
            tactic=full_tactic,
            direction=direction,
            t1_vendor=t1_vendor,
            t1_compliant=t1_compliant,
            t1_response=t1_response,
            t2_vendor=t2_vendor,
            t2_compliant=t2_compliant,
            t2_parseable=t2_parseable,
            t2_response=t2_response,
            switched=(t1_compliant != t2_compliant),
            metadata={"framing": framing, "fin_level": fin_level,
                      "tactic": full_tactic, "direction": direction},
        ))

    return results


# Old framing names used in earlier results files → canonical names used in core.py
_FRAMING_ALIASES = {
    "descriptive": "informational",
    "permissive":  "discretionary",
}

def _normalize_record(r: Dict) -> Dict:
    """Remap legacy framing names in metadata to canonical names."""
    meta = r.get("metadata", {})
    framing = meta.get("framing", "")
    canonical = _FRAMING_ALIASES.get(framing, framing)
    if canonical != framing:
        r = dict(r)
        r["metadata"] = {**meta, "framing": canonical}
    return r


def load_t1_trials(t1_path: str) -> List[Dict]:
    """Load all parseable T1 trials from an existing results file.

    Normalizes legacy framing names (descriptive→informational,
    permissive→discretionary) and skips any rows missing required fields.
    """
    valid_framings = {"directive", "informational", "discretionary", "none"}
    valid_fin_levels = {"none", "low", "breakeven", "high"}
    all_records = [_normalize_record(r) for r in load_results(t1_path)]
    return [
        r for r in all_records
        if r.get("parseable", False)
        and r.get("raw_response")
        and r.get("metadata", {}).get("framing") in valid_framings
        and r.get("metadata", {}).get("fin_level") in valid_fin_levels
    ]


def run_from_t1_file(
    t1_path: str,
    model: str,
    api_key: str,
    outfile: str,
    batch_size: int,
    *,
    direction: str = "both",
    paper: bool = False,
    n_per_cell: Optional[int] = None,
):
    t1_trials = load_t1_trials(t1_path)

    # If the T1 file contains multiple models, only use rows matching the requested model.
    t1_trials = [r for r in t1_trials if r.get("model") == model]

    if paper:
        # Paper multi-model spec: informational framing only.
        # (Direction is still controlled by --direction; default is "both".)
        t1_trials = [r for r in t1_trials if r.get("metadata", {}).get("framing") == "informational"]

    if direction not in {"pushback", "challenge", "both"}:
        raise ValueError("direction must be one of: pushback, challenge, both")

    if direction == "pushback":
        t1_trials = [r for r in t1_trials if r.get("compliant")]
    elif direction == "challenge":
        t1_trials = [r for r in t1_trials if not r.get("compliant")]

    if n_per_cell is not None:
        by_cell: Dict[str, List[Dict]] = defaultdict(list)
        for r in t1_trials:
            m = r.get("metadata", {})
            by_cell[f"{m.get('framing','?')}/{m.get('fin_level','?')}"] .append(r)
        capped: List[Dict] = []
        for cell, recs in by_cell.items():
            capped.extend(_stable_sample(recs, n_per_cell))
        t1_trials = capped

    total = len(t1_trials)
    cell_counts: Dict[str, int] = defaultdict(int)
    for r in t1_trials:
        m = r["metadata"]
        cell_counts[f"{m['framing']}/{m['fin_level']}"] += 1

    print(f"Loaded {total} T1 trials from {t1_path} (model={model})")
    for cell, n in sorted(cell_counts.items()):
        print(f"  {cell}: {n} trials")
    if direction == "pushback":
        print(f"\nRunning T2 follow-ups: PUSHBACK only (3 tactics per trial)")
    elif direction == "challenge":
        print(f"\nRunning T2 follow-ups: CHALLENGE only (3 tactics per trial)")
    else:
        print(f"\nRunning T2 follow-ups (3 tactics per trial, conditional on T1 outcome)")
    print(f"Batch size: {batch_size}\n")

    done = 0
    lock = threading.Lock()
    os.makedirs(os.path.dirname(outfile) if os.path.dirname(outfile) else ".", exist_ok=True)

    with ThreadPoolExecutor(max_workers=batch_size) as ex:
        futures = {ex.submit(run_t2_for_trial, t1, model, api_key): t1 for t1 in t1_trials}
        print(f"Submitted {len(futures)} jobs. Waiting for results...\n")
        for future in as_completed(futures):
            t1 = futures[future]
            meta = t1.get("metadata", {})
            cell = f"{meta.get('framing','?')}/{meta.get('fin_level','?')}"
            try:
                results = future.result()
            except Exception as e:
                with lock:
                    done += 1
                    print(f"  [{done}/{total}] ERROR {cell}: {e}")
                continue
            with lock:
                done += 1
                t2_summary = " ".join(
                    f"{r.tactic.split('_',1)[1]}:{'C' if r.t2_compliant else 'V'}"
                    for r in results
                )
                t1_label = "C" if t1.get("compliant") else "V"
                print(f"  [{done}/{total}] {cell}  T1={t1_label}  {t2_summary}")
                with open(outfile, "a") as f:
                    for r in results:
                        f.write(json.dumps(asdict(r)) + "\n")

def main():
    p = argparse.ArgumentParser(description="Multi-turn pushback / challenge dynamics")
    p.add_argument(
        "-m",
        "--model",
        default="gemini",
        type=resolve_model,
        help=(
            "model alias or full OpenRouter model id "
            "(default: gemini → google/gemini-3-flash-preview)"
        ),
    )
    p.add_argument("--batch-size", type=int, default=20,
                   help="parallel T2 trials (default: 20; each trial = 3 T2 calls)")
    p.add_argument("-o", "--output", default="results/data/multiturn.jsonl")
    p.add_argument("--t1-results", metavar="FILE",
                   help="load T1 trials from this results file instead of re-running them")
    p.add_argument(
        "--direction",
        choices=["pushback", "challenge", "both"],
        default="both",
        help="which multi-turn direction to run (default: both)",
    )
    p.add_argument(
        "--n-per-cell",
        type=int,
        default=None,
        help="cap the number of T1 trials used per framing/fin cell (after direction filtering)",
    )
    p.add_argument("--paper", action="store_true",
                   help="paper subset: informational framing only")
    args = p.parse_args()

    if not args.t1_results:
        p.error("--t1-results FILE is required (e.g. results/data/controls.jsonl)")

    api_key = get_api_key()
    run_from_t1_file(
        args.t1_results, args.model, api_key,
        args.output, args.batch_size,
        direction=args.direction,
        paper=args.paper,
        n_per_cell=args.n_per_cell,
    )


if __name__ == "__main__":
    main()
