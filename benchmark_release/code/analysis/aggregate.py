"""Per-model six-axis profiles, PACTScore, and pairwise contrasts. Writes
results/metrics.csv, cells.csv, and contrasts.csv; --figures redraws the
profile figures via analysis.figures.

Usage: python -m analysis.aggregate [--fast] [--figures]
"""

import argparse
import csv
import os
from typing import Dict, List, Optional

import paths
from analysis import metrics as M
from evaluation.judges import load_trials, load_transparency_votes

OUT_CSV = os.path.join(paths.RESULTS, "metrics.csv")
OUT_CELLS = os.path.join(paths.RESULTS, "cells.csv")
OUT_CONTRASTS = os.path.join(paths.RESULTS, "contrasts.csv")

AXES = ("default_compliance", "pressure_resistance", "pushback_resistance",
        "steerability", "transparency", "rule_scope_discernment")


def _fmt(v: Optional[float]) -> str:
    return f"{v:.3f}" if v is not None else "--"


def profile_all(trials: List[dict], trans_votes: Dict, quorum: float = 0.6,
                fast: bool = False) -> Dict[str, Dict]:
    """Six axes + PACTScore per model. `fast=True` skips the PACTScore
    bootstrap CI."""
    cells = M.build_cells(trials)
    models = sorted({t["model"] for t in trials})
    core = M.common_core(cells, models, quorum) if models else []

    profiles: Dict[str, Dict] = {}
    for model in models:
        pr = M.pressure_resistance(cells, model)
        pb = M.pushback_resistance(cells, model, core)
        st = M.steerability(cells, model)
        tr = M.transparency(trials, trans_votes, model)
        rd = M.rule_scope_discernment(cells, model)
        pact = M.pact_score(trials, model)
        pact_ci = (None, None) if fast else M.pact_score_ci(trials, model)
        axes = {
            "default_compliance": M.default_compliance(cells, model),
            "pressure_resistance": pr,
            "pushback_resistance": pb.value,
            "steerability": st.net,
            "transparency": tr.value if tr.defined else None,
            "rule_scope_discernment": rd.value,
        }
        profiles[model] = {
            "axes": axes,
            "pact": pact,
            "pact_ci": pact_ci,
            "abstention": M.abstention_rate(cells, model),
            "detail": {"pushback": pb, "steer": st,
                       "transparency": tr, "discernment": rd},
        }
    profiles["_meta"] = {"core_items": core, "panel": models}
    return profiles


def write_cells_csv(cells: Dict[M.CellKey, M.Cell], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "mode", "item_id", "scenario_id", "domain", "group",
                    "pressure", "binds", "n_reps", "n_decided", "k_comply",
                    "rate", "wilson_lo", "wilson_hi", "unclear_rate"])
        for (model, mode, _), c in sorted(cells.items()):
            decided = [o for o in c.t1 if o != "unclear"]
            k = sum(o == "comply" for o in decided)
            lo, hi = M.wilson_ci(k, len(decided))
            w.writerow([model, mode, c.item_id, c.scenario_id, c.domain,
                        c.group, c.pressure, c.binds, len(c.t1), len(decided),
                        k, c.rate, lo, hi, c.unclear_rate])


def write_outputs(profiles: Dict[str, Dict], contrasts: List[M.Contrast],
                  out_csv: str, out_contrasts: str) -> None:
    models = [m for m in sorted(profiles) if m != "_meta"]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "pact_score", "pact_score_lo", "pact_score_hi",
                    "pact_base", "pact_directed",
                    "pact_steer_gap", "pact_base_t1", "pact_base_t2",
                    "pact_n_items", "pact_n_items_t2", "pact_n_items_full",
                    "pact_n_t2_missing"] + list(AXES)
                   + ["abstention", "transparency_concealed",
                      "transparency_misrepresented", "transparency_n"])
        for m in models:
            p = profiles[m]
            d = p["detail"]
            pact = p["pact"]
            b, dr = pact.base, pact.directed
            pci_lo, pci_hi = p.get("pact_ci") or (None, None)
            w.writerow([m, pact.value, pci_lo, pci_hi,
                        b.value if b else None, dr.value if dr else None,
                        pact.steer_gap,
                        b.t1 if b else None, b.t2 if b else None,
                        b.n_items if b else None,
                        b.n_items_t2 if b else None,
                        b.n_items_full if b else None,
                        b.n_t2_missing if b else None]
                       + [p["axes"][a] for a in AXES] + [
                p["abstention"],
                d["transparency"].concealed_rate,
                d["transparency"].misrepresented_rate,
                d["transparency"].n_labeled])

    with open(out_contrasts, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_a", "model_b", "diff", "n_items", "p", "p_bh"])
        for c in contrasts:
            w.writerow([c.model_a, c.model_b, c.diff, c.n_items, c.p, c.p_bh])


def profile_figures(profiles: Dict[str, Dict]) -> None:
    from analysis import figures
    figures.radar(profiles)
    figures.radar_quad(profiles)
    figures.axis_correlation(profiles)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials-dir", default=paths.TRIALS_DIR)
    ap.add_argument("--transparency", default=paths.TRANSPARENCY,
                    help="axis-5 transparency judge votes")
    ap.add_argument("--quorum", type=float, default=0.6,
                    help="common-core quorum reported alongside axis 3")
    ap.add_argument("--csv", default=OUT_CSV)
    ap.add_argument("--cells-csv", default=OUT_CELLS)
    ap.add_argument("--contrasts-csv", default=OUT_CONTRASTS)
    ap.add_argument("--figures", action="store_true",
                    help="also draw the radar and axis-correlation figures")
    ap.add_argument("--figures-only", action="store_true",
                    help="only redraw those figures (skips CSVs, CIs, contrasts)")
    ap.add_argument("--fast", action="store_true",
                    help="skip the PACTScore bootstrap CI and the paired "
                    "contrasts; contrasts.csv is left header-only")
    args = ap.parse_args()

    trials = load_trials(args.trials_dir)
    if not trials:
        raise SystemExit(f"no trials found in {args.trials_dir}")
    trans_votes = load_transparency_votes(args.transparency)
    print(f"{len(trials)} trials, {len(trans_votes)} transparency-labeled trials")

    if args.figures_only:
        profiles = profile_all(trials, trans_votes, args.quorum, fast=True)
        profile_figures(profiles)
        return

    profiles = profile_all(trials, trans_votes, args.quorum, fast=args.fast)
    cells = M.build_cells(trials)
    panel = profiles["_meta"]["panel"]
    contrasts = {} if args.fast else M.pact_contrasts(trials, panel)
    write_cells_csv(cells, args.cells_csv)

    models = [m for m in sorted(profiles) if m != "_meta"]
    print(f"\n{'model':<38} {'PACT':>7} {'base':>7} {'direct':>7}  "
          + " ".join(f"{a[:9]:>9}" for a in AXES))
    for m in sorted(models, key=lambda m: -(profiles[m]["pact"].value or 0.0)):
        pact = profiles[m]["pact"]
        b, dr = pact.base, pact.directed
        vals = " ".join(f"{_fmt(profiles[m]['axes'][a]):>9}" for a in AXES)
        print(f"{m:<38} {_fmt(pact.value):>7} {_fmt(b.value if b else None):>7} "
              f"{_fmt(dr.value if dr else None):>7}  {vals}")
    for m in models:
        pact = profiles[m]["pact"]
        if set(pact.per_mode) != set(M.PACT_MODES):
            print(f"  [!] {m}: PACTScore averaged over {sorted(pact.per_mode)} only "
                  f"- not comparable to a model scored in both modes")
        b = pact.base
        if b and b.n_t2_missing:
            print(f"  [!] {m}: {b.n_t2_missing} multi-turn items have no T2 outcome "
                  f"(scored turn-1-only)")

    write_outputs(profiles, contrasts, args.csv, args.contrasts_csv)
    meta = profiles["_meta"]
    print(f"\ncommon core: {len(meta['core_items'])} items over a panel "
          f"of {len(meta['panel'])} models")
    print(f"wrote {args.csv}, {args.cells_csv}, {args.contrasts_csv}")
    if args.figures:
        profile_figures(profiles)


if __name__ == "__main__":
    main()
