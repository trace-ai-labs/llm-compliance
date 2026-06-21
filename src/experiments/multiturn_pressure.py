#!/usr/bin/env python3
"""Two-turn dynamics seeded from the pressure experiment.

Takes Turn-1 outcomes from `pressure` (under the anti-adversarial mandate) and applies
the same Turn-2 pushback/challenge probes, testing whether a mandate makes compliance
more durable in conversation.

Generate the pressure Turn-1 results first, then point this at them with --t1-results.

Run: python -m src.experiments.multiturn_pressure --t1-results results/data/pressure.jsonl -o results/data/multiturn_pressure.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.core import (
    BASE_PROMPT, VENDORS, call_model_messages, get_api_key, judge_vendor,
    load_results, make_reg_context, make_user_message, resolve_model,
)
from src.prompts.pressure import MANDATE_LEVELS, EMPLOYEE_PRESSURES
from src.prompts.multiturn import PUSHBACK_TACTICS, CHALLENGE_TACTICS
from src.experiments import pressure as pressure_exp  # for the paper-subset scenario keys


def _stable_sample(records: List[Dict], k: int) -> List[Dict]:
    """Deterministically sample up to k records, stable across runs."""
    if k is None or k <= 0:
        return []
    if len(records) <= k:
        return records

    def _key(r: Dict):
        meta = r.get("metadata", {})
        return (
            str(meta.get("mandate", "")),
            str(meta.get("pressure", "")),
            str(meta.get("framing", "")),
            str(meta.get("fin_level", "")),
            int(r.get("run_id", 0)),
            int(r.get("seed", 0)),
        )

    return sorted(records, key=_key)[:k]


@dataclass
class MultiTurnResult:
    scenario: str
    t1_trial_id: str
    run_id: int
    seed: int
    model: str
    tactic: str
    direction: str
    t1_vendor: str
    t1_compliant: bool
    t1_response: str
    t2_vendor: str
    t2_compliant: bool
    t2_parseable: bool
    t2_response: str
    switched: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


_FRAMING_ALIASES = {
    "descriptive": "informational",
    "permissive": "discretionary",
}


def _normalize_record(r: Dict) -> Dict:
    meta = r.get("metadata", {})
    framing = meta.get("framing", "")
    canonical = _FRAMING_ALIASES.get(framing, framing)
    if canonical != framing:
        r = dict(r)
        r["metadata"] = {**meta, "framing": canonical}
    return r


def load_t1_trials(t1_path: str) -> List[Dict]:
    """Load all parseable pressure-experiment T1 trials from an existing results file."""
    valid_framings = {"directive", "informational", "discretionary", "none"}
    valid_fin_levels = {"none", "low", "breakeven", "high"}
    all_records = [_normalize_record(r) for r in load_results(t1_path)]

    out: List[Dict] = []
    for r in all_records:
        if not r.get("parseable", False):
            continue
        if not r.get("raw_response"):
            continue
        meta = r.get("metadata", {}) or {}
        if meta.get("framing") not in valid_framings:
            continue
        if meta.get("fin_level") not in valid_fin_levels:
            continue
        if meta.get("mandate") not in MANDATE_LEVELS:
            continue
        if meta.get("pressure") not in EMPLOYEE_PRESSURES:
            continue
        out.append(r)
    return out


def _t1_system_prompt(*, mandate: str, framing: str, fin_level: str) -> str:
    if framing == "none":
        return BASE_PROMPT
    return BASE_PROMPT + MANDATE_LEVELS.get(mandate, "") + make_reg_context(framing, fin_level)


def run_t2_for_trial(t1_record: Dict, model: str, api_key: str) -> List[MultiTurnResult]:
    meta = t1_record.get("metadata", {}) or {}
    mandate = str(meta.get("mandate", "")).strip()
    pressure = str(meta.get("pressure", "")).strip()
    framing = str(meta.get("framing", "")).strip()
    fin_level = str(meta.get("fin_level", "")).strip()

    seed = int(t1_record.get("seed", 0) or 0)
    run_id = int(t1_record.get("run_id", 0) or 0)
    t1_compliant = bool(t1_record.get("compliant", False))
    t1_response = str(t1_record.get("raw_response", "") or "")
    t1_vendor = str(t1_record.get("vendor_chosen", "UNCLEAR") or "UNCLEAR")
    t1_model = resolve_model(str(t1_record.get("model", "")).strip()) if t1_record.get("model") else ""

    if not fin_level or not t1_response:
        return []

    system_prompt = _t1_system_prompt(mandate=mandate, framing=framing, fin_level=fin_level)

    rng = random.Random(seed)
    user_t1, shuffled_vendors = make_user_message(rng, VENDORS)
    user_t1 = user_t1 + EMPLOYEE_PRESSURES.get(pressure, "")

    tactics = PUSHBACK_TACTICS if t1_compliant else CHALLENGE_TACTICS
    direction = "pushback" if t1_compliant else "challenge"

    scenario = str(t1_record.get("scenario", "")).strip() or f"{mandate}_{pressure}_{framing}_{fin_level}"
    t1_trial_id = f"{scenario}_{run_id}"

    results: List[MultiTurnResult] = []
    for tactic_name, tactic_msg in tactics.items():
        full_tactic = f"{direction}_{tactic_name}"
        t2_response = call_model_messages(
            system_prompt,
            [
                {"role": "user", "content": user_t1},
                {"role": "assistant", "content": t1_response},
                {"role": "user", "content": tactic_msg},
            ],
            model,
            api_key,
        )

        t2_vendor_obj = judge_vendor(t2_response, shuffled_vendors, api_key)
        t2_compliant = bool(t2_vendor_obj.certified) if t2_vendor_obj else False
        t2_vendor = t2_vendor_obj.name if t2_vendor_obj else "UNCLEAR"
        t2_parseable = t2_vendor_obj is not None

        results.append(
            MultiTurnResult(
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
                metadata={
                    "mandate": mandate,
                    "pressure": pressure,
                    "framing": framing,
                    "fin_level": fin_level,
                    "tactic": full_tactic,
                    "direction": direction,
                    "t1_source": "exp8",
                    "t1_model": t1_model,
                    "t2_model": model,
                },
            )
        )

    return results


def run_from_t1_file(
    t1_path: str,
    model: Optional[str],
    api_key: str,
    outfile: str,
    batch_size: int,
    *,
    direction: str = "both",
    paper: bool = False,
    n_per_cell: Optional[int] = None,
) -> None:
    t1_trials = load_t1_trials(t1_path)

    # If the T1 file contains multiple models, you can optionally restrict to one model.
    # If not provided, we preserve each row's own model and run T2 with that same model.
    if model is not None:
        target = str(model).strip()
        t1_trials = [
            r for r in t1_trials
            if resolve_model(str(r.get("model", "")).strip()) == target
        ]

    if paper:
        keep = set(pressure_exp.build_scenarios(paper=True).keys())
        t1_trials = [r for r in t1_trials if str(r.get("scenario", "")).strip() in keep]
        t1_trials = [
            r for r in t1_trials
            if str(r.get("metadata", {}).get("mandate", "")).strip() == "anti_adversarial"
        ]

    if direction not in {"pushback", "challenge", "both"}:
        raise ValueError("direction must be one of: pushback, challenge, both")

    if direction == "pushback":
        t1_trials = [r for r in t1_trials if r.get("compliant")]
    elif direction == "challenge":
        t1_trials = [r for r in t1_trials if not r.get("compliant")]

    if n_per_cell is not None:
        by_cell: Dict[str, List[Dict]] = defaultdict(list)
        for r in t1_trials:
            m = r.get("metadata", {}) or {}
            by_cell[f"{m.get('mandate','?')}/{m.get('pressure','?')}/{m.get('fin_level','?')}"] .append(r)
        capped: List[Dict] = []
        for _, recs in by_cell.items():
            capped.extend(_stable_sample(recs, n_per_cell))
        t1_trials = capped

    total = len(t1_trials)
    os.makedirs(os.path.dirname(outfile) if os.path.dirname(outfile) else ".", exist_ok=True)

    done = 0
    lock = threading.Lock()

    def _model_for_row(r: Dict[str, Any]) -> str:
        # Prefer per-row model so multi-model combined T1 files produce matching T2 models.
        if model is not None:
            return str(model)
        m = r.get("model")
        if not m:
            raise ValueError(
                "T1 record missing 'model'. Provide --model to run this file or regenerate T1 results with model provenance."
            )
        return resolve_model(str(m))

    with ThreadPoolExecutor(max_workers=batch_size) as ex:
        futures = {
            ex.submit(run_t2_for_trial, t1, _model_for_row(t1), api_key): t1
            for t1 in t1_trials
        }
        for future in as_completed(futures):
            try:
                results = future.result()
            except Exception as e:
                with lock:
                    done += 1
                    print(f"  [{done}/{total}] ERROR: {e}")
                continue

            with lock:
                done += 1
                with open(outfile, "a", encoding="utf-8") as f:
                    for r in results:
                        f.write(json.dumps(asdict(r)) + "\n")

def main() -> None:
    p = argparse.ArgumentParser(description="Multi-turn follow-ups seeded from the pressure experiment")
    p.add_argument(
        "-m",
        "--model",
        default=None,
        type=lambda s: resolve_model(str(s)),
        help=(
            "Optional model alias or full OpenRouter model id. "
            "If omitted, uses the per-row T1 'model' from --t1-results for each trial."
        ),
    )
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("-o", "--output", default="results/data/multiturn_pressure.jsonl")
    p.add_argument("--t1-results", metavar="FILE", required=True)
    p.add_argument(
        "--direction",
        choices=["pushback", "challenge", "both"],
        default="both",
    )
    p.add_argument(
        "--n-per-cell",
        type=int,
        default=None,
        help="cap T1 trials used per (mandate/pressure/fin) cell",
    )
    p.add_argument("--paper", action="store_true", help="match the pressure paper subset")
    args = p.parse_args()

    api_key = get_api_key()
    run_from_t1_file(
        args.t1_results,
        args.model,
        api_key,
        args.output,
        args.batch_size,
        direction=args.direction,
        paper=args.paper,
        n_per_cell=args.n_per_cell,
    )


if __name__ == "__main__":
    main()
