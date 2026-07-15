"""Feature-slice analysis: does compliance move with a scenario's design tags?

Every ScenarioSeed carries four pre-registered feature codes (see registry):

  crispness  how clear-cut the rule is             hi / med / lo
  benefit    what breaking the rule rewards         money / time / engagement / relationship
  victim     who is harmed by breaking it           identifiable / diffuse / none
  penalty    the kind of real-world consequence     fine / liability / license / criminal

These are analysis dimensions only; the generator never sees them. This module
slices the binding-cell compliance rate (comply / decided, at turn 1) by each
feature value, pooled over the real-model panel and optionally per model, with
Wilson 95% CIs, so you can ask things like "is compliance lower when the rule is
fuzzy (crispness=lo)?" or "does a named victim raise compliance?".

Only binding cells count by default (there, comply == held the rule, one clean
meaning); pass --all-groups to fold in the non-binding cells too, where comply
means the opposite (correct stand-down), so read that mix with care.

Usage:
  python -m src.benchmark.feature_analysis [--trials-dir DIR] [--arm base]
      [--by-model] [--all-groups] [--md PATH] [--csv PATH]
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional

from src.benchmark import metrics as M
from src.benchmark.judges import load_trials
from src.benchmark.registry import SCENARIO_BY_ID
from src.benchmark.runner import TRIALS_DIR, TRIVIAL_PREFIX

FEATURE_DIMS = ("crispness", "benefit", "victim", "penalty")

DIM_BLURB = {
    "crispness": "how clear-cut the rule is",
    "benefit": "what breaking the rule rewards",
    "victim": "who is harmed by breaking it",
    "penalty": "the real-world consequence type",
}

# Stable display order for the values inside each dimension; anything unexpected
# is appended alphabetically after these.
VALUE_ORDER = {
    "crispness": ["hi", "med", "lo"],
    "benefit": ["money", "time", "engagement", "relationship"],
    "victim": ["identifiable", "diffuse", "none"],
    "penalty": ["fine", "liability", "license", "criminal"],
}

OUT_MD = os.path.join("results", "benchmark", "feature_slices.md")
OUT_CSV = os.path.join("results", "benchmark", "feature_slices.csv")


def _feature(scenario_id: str, dim: str) -> Optional[str]:
    seed = SCENARIO_BY_ID.get(scenario_id)
    return seed.features[dim] if seed else None


def _ordered(dim: str, values) -> List[str]:
    pref = VALUE_ORDER.get(dim, [])
    known = [v for v in pref if v in values]
    extra = sorted(v for v in values if v not in pref)
    return known + extra


def select_cells(cells_map: Dict[M.CellKey, M.Cell], arm: str,
                 model: Optional[str] = None, binding_only: bool = True
                 ) -> List[M.Cell]:
    """Real-model cells for one arm (optionally one model), binding by default."""
    out = []
    for (m, a, _), c in cells_map.items():
        if a != arm or m.startswith(TRIVIAL_PREFIX):
            continue
        if model is not None and m != model:
            continue
        if binding_only and not c.binds:
            continue
        out.append(c)
    return out


def slice_dimension(cells: List[M.Cell], dim: str) -> Dict[str, dict]:
    """Pool decided turn-1 outcomes across cells by this dimension's value.
    Returns {value: {k, n, rate, lo, hi, n_cells, n_scenarios}}."""
    k_by: Dict[str, int] = defaultdict(int)
    n_by: Dict[str, int] = defaultdict(int)
    cells_by: Dict[str, int] = defaultdict(int)
    scen_by: Dict[str, set] = defaultdict(set)
    for c in cells:
        val = _feature(c.scenario_id, dim)
        if val is None:
            continue
        decided = c.decided  # list[bool], comply == True, unclear dropped
        if not decided:
            continue
        k_by[val] += sum(decided)
        n_by[val] += len(decided)
        cells_by[val] += 1
        scen_by[val].add(c.scenario_id)
    out: Dict[str, dict] = {}
    for val in n_by:
        k, n = k_by[val], n_by[val]
        lo, hi = M.wilson_ci(k, n)
        out[val] = {"k": k, "n": n, "rate": k / n if n else None,
                    "lo": lo, "hi": hi, "n_cells": cells_by[val],
                    "n_scenarios": len(scen_by[val])}
    return out


def analyze(cells_map: Dict[M.CellKey, M.Cell], arm: str,
            model: Optional[str], binding_only: bool
            ) -> Dict[str, Dict[str, dict]]:
    cells = select_cells(cells_map, arm, model, binding_only)
    return {dim: slice_dimension(cells, dim) for dim in FEATURE_DIMS}


# ── Output ───────────────────────────────────────────────────────────────────

def _fmt_rate(s: dict) -> str:
    return (f"{s['rate']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]"
            if s["rate"] is not None else "-")


def print_console(label: str, sliced: Dict[str, Dict[str, dict]]) -> None:
    print(f"\n=== {label} ===")
    for dim in FEATURE_DIMS:
        rows = sliced[dim]
        if not rows:
            continue
        print(f"\n{dim}  ({DIM_BLURB[dim]})")
        for val in _ordered(dim, rows):
            s = rows[val]
            print(f"  {val:<13} compliance {_fmt_rate(s):<26} "
                  f"n={s['n']:<5} ({s['n_scenarios']} scenarios)")


def write_md(path: str, blocks: List, arm: str, binding_only: bool) -> None:
    scope = "binding cells only" if binding_only else "all cells (binding + non-binding)"
    out = ["# ComplianceBench - compliance by scenario feature",
           "",
           f"Turn-1 compliance rate (comply / decided) on the `{arm}` arm, {scope}, "
           "sliced by each pre-registered scenario feature code. Wilson 95% CIs. "
           "These four codes are analysis tags only; the generator never sees "
           "them.",
           ""]
    for label, sliced in blocks:
        out += ["", "---", "", f"## {label}", ""]
        for dim in FEATURE_DIMS:
            rows = sliced[dim]
            if not rows:
                continue
            out += [f"### {dim} - {DIM_BLURB[dim]}", "",
                    "| value | compliance [95% CI] | n decided | scenarios |",
                    "|---|---|---|---|"]
            for val in _ordered(dim, rows):
                s = rows[val]
                out.append(f"| {val} | {_fmt_rate(s)} | {s['n']} | {s['n_scenarios']} |")
            out.append("")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")


def write_csv(path: str, blocks: List) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "dimension", "value", "n_scenarios", "n_cells",
                    "n_decided", "k_comply", "rate", "wilson_lo", "wilson_hi"])
        for label, sliced in blocks:
            for dim in FEATURE_DIMS:
                for val in _ordered(dim, sliced[dim]):
                    s = sliced[dim][val]
                    w.writerow([label, dim, val, s["n_scenarios"], s["n_cells"],
                                s["n"], s["k"], s["rate"], s["lo"], s["hi"]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials-dir", default=TRIALS_DIR)
    ap.add_argument("--arm", default="base",
                    help="which steerability arm to slice (default base)")
    ap.add_argument("--by-model", action="store_true",
                    help="also emit a per-model breakdown, not just the pool")
    ap.add_argument("--all-groups", action="store_true",
                    help="include non-binding cells (comply flips meaning there)")
    ap.add_argument("--md", default=OUT_MD)
    ap.add_argument("--csv", default=OUT_CSV)
    args = ap.parse_args()

    trials = load_trials(args.trials_dir)
    if not trials:
        raise SystemExit(f"no trials found in {args.trials_dir} - run "
                         "src.benchmark.runner first")
    binding_only = not args.all_groups
    cells_map = M.build_cells(trials)

    blocks = [("all models (pooled)",
               analyze(cells_map, args.arm, None, binding_only))]
    if args.by_model:
        models = sorted({m for (m, a, _) in cells_map
                         if a == args.arm and not m.startswith(TRIVIAL_PREFIX)})
        for m in models:
            blocks.append((m, analyze(cells_map, args.arm, m, binding_only)))

    for label, sliced in blocks:
        print_console(label, sliced)
    write_md(args.md, blocks, args.arm, binding_only)
    write_csv(args.csv, blocks)
    print(f"\nwrote {args.md}, {args.csv}")


if __name__ == "__main__":
    main()
