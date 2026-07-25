"""Distributional breakouts for the appendix: how behavior varies across domains
and pressure families. Pure regroups of the trial records (no API, no re-run of
anything upstream). Emits:

  results/benchmark/dist_domain_axis.csv   domain x 5 axes (panel mean)
  results/benchmark/dist_pressure.csv      pressure family x {t1 comply, t2 hold, steer}
  results/benchmark/figures/domain_pressure_heatmap.{png,pdf}   t1 comply, domain x pressure

Outcome categories in the data are comply / violate / unclear only (no overcomply).
Unclear is dropped from every rate, matching the axis denominators. "Turn 2" here is
the pushback branch (a turn-1 complier is re-argued); the challenge branch is not
analyzed. Steerability is the signed recovery, pooled over the cells in the group.

Run:  python -m src.benchmark.distributions
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from src.benchmark import figstyle as FS
from src.benchmark import metrics as M
from src.benchmark.judges import load_trials
from src.benchmark.registry import DOMAIN_BY_KEY, SCORED_PRESSURES

TRIALS_DIR = os.path.join("results", "benchmark", "trials")
OUT_DIR = os.path.join("results", "benchmark")
FIG_DIR = os.path.join(OUT_DIR, "figures")

# 5 cell-based axes; honesty is omitted per-domain (too few violations per domain
# to be stable). Each entry maps a name to a (cells, model) -> Optional[float].
AXES = [
    ("default_compliance", lambda cs, m: M.default_compliance(cs, m)),
    ("pressure_resistance", lambda cs, m: M.pressure_resistance(cs, m).p3),
    ("pushback_resistance", lambda cs, m: M.pushback_resistance(cs, m, []).value),
    ("steerability", lambda cs, m: M.steerability(cs, m).net),
    ("rule_scope_discernment", lambda cs, m: M.rule_scope_discernment(cs, m).value),
]


def short_domain(key: str) -> str:
    """A compact label for figure/table axes: the first segment of the title."""
    title = DOMAIN_BY_KEY[key].title
    head = title.split(" / ")[0].split(" & ")[0].strip()
    return {"Government services": "Gov services",
            "Customer service": "Customer svc",
            "Advertising": "Advertising",
            "Export controls": "Export ctrl",
            "Pharma medical information": "Pharma",
            "Healthcare admin": "Healthcare",
            "Content moderation": "Moderation"}.get(head, head)


def _pooled_rate(cells: Sequence[M.Cell], attr: str) -> Optional[float]:
    """comply / (comply + violate) over all reps in `cells`, unclear dropped.
    attr is 't1' or 't2_pushback'."""
    comply = decided = 0
    for c in cells:
        outs = getattr(c, attr)
        for o in outs:
            if o == "unclear":
                continue
            decided += 1
            comply += (o == "comply")
    return comply / decided if decided else None


def _pooled_recovery(base: Sequence[M.Cell], directive: Sequence[M.Cell]
                     ) -> Optional[float]:
    """Signed steerability recovery pooled over a cell group: sum(r_dir - r_base)
    / sum(1 - r_base) over items the base model fails (matched by item_id)."""
    b = {c.item_id: c.rate for c in base if c.rate is not None}
    d = {c.item_id: c.rate for c in directive if c.rate is not None}
    gain = mass = 0.0
    for iid, br in b.items():
        if br >= 1.0 or iid not in d:
            continue
        mass += 1.0 - br
        gain += d[iid] - br
    return gain / mass if mass > 0 else None


def domain_axis_table(cells: Dict, models: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    """domain -> axis -> panel mean of the per-model per-domain axis value."""
    domains = sorted({c.domain for c in cells.values()})
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for dom in domains:
        dcells = {k: c for k, c in cells.items() if c.domain == dom}
        row: Dict[str, Optional[float]] = {}
        for name, fn in AXES:
            vals = [fn(dcells, m) for m in models]
            vals = [v for v in vals if v is not None]
            row[name] = sum(vals) / len(vals) if vals else None
        out[dom] = row
    return out


def pressure_table(cells: Dict, models: List[str]
                   ) -> Dict[str, Dict[str, Optional[float]]]:
    """pressure family -> {t1 comply, t2 hold, steerability} pooled over the panel."""
    fams = list(SCORED_PRESSURES)
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for fam in fams:
        base_p = [c for (m, a, _), c in cells.items()
                  if a == "base" and c.pressure == fam and m in models]
        dir_p = [c for (m, a, _), c in cells.items()
                 if a == "anti_adversarial" and c.pressure == fam and m in models]
        out[fam] = {
            "t1_comply": _pooled_rate(base_p, "t1"),
            "t2_hold": _pooled_rate(base_p, "t2_pushback"),
            "steerability": _pooled_recovery(base_p, dir_p),
        }
    return out


def domain_pressure_grid(cells: Dict, models: List[str]):
    """(domain, pressure) -> pooled t1 comply. Returns (domains, fams, grid)."""
    domains = sorted({c.domain for c in cells.values()})
    fams = list(SCORED_PRESSURES)
    grid = []
    for dom in domains:
        row = []
        for fam in fams:
            cs = [c for (m, a, _), c in cells.items()
                  if a == "base" and c.domain == dom and c.pressure == fam and m in models]
            row.append(_pooled_rate(cs, "t1"))
        grid.append(row)
    return domains, fams, grid


def _write_csv(path: str, header: List[str], rows: List[List]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def plot_domain_pressure(domains, fams, grid, path: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    FS.use_paper_style()
    arr = np.array([[v if v is not None else np.nan for v in row] for row in grid])
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    im = ax.imshow(arr, cmap=FS.SCORE_CMAP, vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([f.replace("_", " ") for f in fams], rotation=45, ha="right")
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels([short_domain(d) for d in domains])
    for i in range(len(domains)):
        for j in range(len(fams)):
            v = arr[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=FS.INK if v > 0.7 else "white")
    ax.set_title("Turn-1 compliance by domain and pressure (base mode, panel-pooled)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="compliance")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    trials = load_trials(TRIALS_DIR)
    cells = M.build_cells(trials)
    models = sorted({m for (m, _, _) in cells if not m.startswith("trivial:")})
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. domain x axis
    da = domain_axis_table(cells, models)
    axis_names = [n for n, _ in AXES]
    _write_csv(os.path.join(OUT_DIR, "dist_domain_axis.csv"),
               ["domain"] + axis_names,
               [[dom] + [da[dom][a] for a in axis_names] for dom in da])

    # 2. pressure family x {t1, t2, steer}
    pt = pressure_table(cells, models)
    _write_csv(os.path.join(OUT_DIR, "dist_pressure.csv"),
               ["pressure", "t1_comply", "t2_hold", "steerability"],
               [[p, pt[p]["t1_comply"], pt[p]["t2_hold"], pt[p]["steerability"]]
                for p in pt])

    # 3. domain x pressure heatmap
    domains, fams, grid = domain_pressure_grid(cells, models)
    plot_domain_pressure(domains, fams, grid,
                         os.path.join(FIG_DIR, "domain_pressure_heatmap.png"))

    # headline findings for the prose
    print("\n== domain x axis (panel mean) ==")
    for dom in sorted(da, key=lambda d: (da[d]["default_compliance"] or 9)):
        r = da[dom]
        print(f"  {short_domain(dom):14s} " +
              " ".join(f"{a[:4]}={r[a]:.2f}" if r[a] is not None else f"{a[:4]}=n/a"
                       for a in axis_names))
    print("\n== pressure family ==")
    for p in sorted(pt, key=lambda p: (pt[p]["t1_comply"] or 9)):
        r = pt[p]
        print(f"  {p:24s} t1={r['t1_comply']:.3f} t2={r['t2_hold']:.3f} "
              f"steer={r['steerability']:.3f}")
    flat = [(short_domain(d), f.replace('_', ' '), grid[i][j])
            for i, d in enumerate(domains) for j, f in enumerate(fams)
            if grid[i][j] is not None]
    lo = min(flat, key=lambda t: t[2])
    print(f"\n  weakest domain x pressure cell: {lo[0]} x {lo[1]} = {lo[2]:.3f}")
    print("wrote dist_domain_axis.csv, dist_pressure.csv, domain_pressure_heatmap.{png,pdf}")


if __name__ == "__main__":
    main()
