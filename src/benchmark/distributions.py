"""Distributional breakouts for the appendix: how behavior varies across domains
and pressure families. Pure regroups of the trial records (no API, no re-run of
anything upstream). Emits:

  results/benchmark/dist_domain_axis.csv   domain x 6 axes (cell axes panel-mean;
                                           transparency pooled over panel violations)
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

# The 5 cell-based axes, panel-meaned per domain. Transparency is computed
# separately (transparency_by_domain): per-model per-domain violation counts are
# too thin to average, so it pools every judged violation the panel committed in
# the domain instead. Each entry maps a name to a (cells, model) -> Optional[float].
AXES = [
    ("default_compliance", lambda cs, m: M.default_compliance(cs, m)),
    ("pressure_resistance", lambda cs, m: M.pressure_resistance(cs, m).p3),
    ("pushback_resistance", lambda cs, m: M.pushback_resistance(cs, m, []).value),
    ("steerability", lambda cs, m: M.steerability(cs, m).net),
    ("rule_scope_discernment", lambda cs, m: M.rule_scope_discernment(cs, m).value),
]


def short_domain(key: str) -> str:
    """Formal display label for a domain key (single source: figstyle)."""
    return FS.domain_label(key)


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


def transparency_by_domain(trials: List[dict], models: List[str]
                           ) -> Dict[str, Optional[float]]:
    """domain -> TRANSPARENT vote-share, pooled over every judged violation the
    panel committed in the domain (both arms; each trial weighs its judges'
    votes fractionally, mirroring metrics.transparency's trial selection)."""
    from src.benchmark.judges import load_transparency_votes
    votes = load_transparency_votes()
    keep = set(models)
    shares: Dict[str, List[float]] = defaultdict(list)
    for t in trials:
        if t.get("model") not in keep or not t.get("binds"):
            continue
        if not (t.get("t1_outcome") == "violate"
                or (t.get("t1_outcome") != "violate"
                    and t.get("t2_outcome") == "violate")):
            continue
        v = votes.get(f"{t['model']}||{t['trial_id']}")
        if v and sum(v.values()):
            shares[t["domain"]].append(v.get("TRANSPARENT", 0) / sum(v.values()))
    return {dom: sum(s) / len(s) if s else None for dom, s in shares.items()}


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
    """Annotated heatmap, cell values x100 (no colorbar: the numbers carry the
    value, the shade is a redundant cue). No in-figure title; the caption has it."""
    import matplotlib.pyplot as plt
    import numpy as np
    FS.use_paper_style()
    arr = np.array([[v if v is not None else np.nan for v in row] for row in grid])
    # rows ordered easiest domain at top -> hardest at bottom
    order = np.argsort(-np.nanmean(arr, axis=1))
    arr = arr[order]
    domains = [domains[i] for i in order]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(arr, cmap=FS.SCORE_CMAP, vmin=0.45, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([FS.pressure_label(f) for f in fams], fontsize=12,
                       rotation=38, ha="right")
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels([short_domain(d) for d in domains], fontsize=12.5)
    for i in range(len(domains)):
        for j in range(len(fams)):
            v = arr[i, j]
            if v == v:
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center",
                        fontsize=12.5, color="white" if v >= 0.955 else FS.INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    sa = (sum((x - ma) ** 2 for x in a)) ** 0.5
    sb = (sum((y - mb) ** 2 for y in b)) ** 0.5
    return cov / (sa * sb) if sa and sb else 0.0


def plot_compliance_transparency(da: Dict[str, Dict[str, Optional[float]]],
                                 tbd: Dict[str, Optional[float]],
                                 path: str) -> None:
    """Scatter of per-domain Default Compliance (panel mean) against
    Transparency (pooled over the panel's judged violations in the domain), one
    dot per domain (main-paper fig:comptrans). Shows that where models violate
    most is not where they disclose most: the two most transparent domains are
    the hardest (procurement) and the easiest (government services)."""
    import matplotlib.pyplot as plt
    FS.use_paper_style()
    doms = [d for d in da if da[d]["default_compliance"] is not None
            and tbd.get(d) is not None]
    x = [da[d]["default_compliance"] for d in doms]
    y = [tbd[d] for d in doms]
    r = _pearson(x, y)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter(x, y, s=70, color=FS.BLUE, zorder=3)
    # the right margin is widened so the tight right-hand cluster can hang two
    # labels outside the data range; the rest are nudged against collisions,
    # with thin leader lines wherever a label is displaced far from its dot
    ax.set_xlim(0.735, 1.10)
    nudge = {"Procurement": (8, -12), "Gov. Services": (-9, 0),
             "Finance": (-9, 3), "Moderation": (2, 10),
             "Export Controls": (-28, 18), "Pharma": (7, 3),
             "Customer Service": (22, -13), "Privacy": (-9, -4),
             "AML": (7, -3), "Healthcare": (8, 2), "HR/Hiring": (-9, 0),
             "Advertising": (-9, 5)}
    leader = {"Export Controls", "Customer Service"}
    for d, xi, yi in zip(doms, x, y):
        s = short_domain(d)
        dx, dy = nudge.get(s, (8, 4))
        kw = dict(fontsize=11, xytext=(dx, dy), textcoords="offset points",
                  ha="right" if dx < 0 else "left", va="center", color=FS.INK)
        if s in leader:
            kw["arrowprops"] = dict(arrowstyle="-", lw=0.8, color="#9ca3af",
                                    shrinkA=2, shrinkB=3)
        ax.annotate(s, (xi, yi), **kw)
    ax.set_xlabel("Default Compliance", fontsize=13.5)
    ax.set_ylabel("Transparency", fontsize=13.5)
    ax.set_xticks([0.8, 0.9, 1.0])
    ax.tick_params(labelsize=12)
    ax.grid(True, color=FS.GRID, lw=0.8)
    ax.set_axisbelow(True)
    FS.strip_axes(ax)
    ax.text(0.02, 0.10, f"$r = {r:+.2f}$", transform=ax.transAxes,
            fontsize=12.5, va="bottom", color=FS.INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def significance_summary(contrasts_path: str, metrics_path: str) -> dict:
    """Read the BH-adjusted pairwise contrasts and the PACTScore ranking; report
    how many pairs differ significantly and the size of the top indistinguishable
    cluster (models not significantly below the best)."""
    with open(metrics_path, encoding="utf-8") as f:
        met = {r["model"]: float(r["pact_score"]) for r in csv.DictReader(f)
               if r.get("pact_score") not in ("", None)}
    order = sorted(met, key=lambda m: -met[m])
    with open(contrasts_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pbh = {}
    for r in rows:
        pbh[frozenset((r["model_a"], r["model_b"]))] = float(r["p_bh"])
    n_sig = sum(1 for r in rows if float(r["p_bh"]) < 0.05)
    top = order[0]
    cluster = [top] + [m for m in order[1:]
                       if pbh.get(frozenset((top, m)), 0.0) >= 0.05]
    return {"n_pairs": len(rows), "n_sig": n_sig, "top": top,
            "cluster": cluster, "order": order, "pact": met}


def main() -> None:
    trials = load_trials(TRIALS_DIR)
    cells = M.build_cells(trials)
    models = sorted({m for (m, _, _) in cells if not m.startswith("trivial:")})
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. domain x axis (+ raw turn-2 pushback hold per domain, for the prose).
    # Transparency rides along, pooled per domain over the panel's violations.
    da = domain_axis_table(cells, models)
    axis_names = [n for n, _ in AXES]
    tbd = transparency_by_domain(trials, models)
    t2_by_domain = {
        dom: _pooled_rate([c for (m, a, _), c in cells.items()
                           if a == "base" and c.domain == dom and m in models],
                          "t2_pushback")
        for dom in da}
    _write_csv(os.path.join(OUT_DIR, "dist_domain_axis.csv"),
               ["domain"] + axis_names + ["transparency", "turn2_hold"],
               [[dom] + [da[dom][a] for a in axis_names]
                + [tbd.get(dom), t2_by_domain[dom]]
                for dom in da])

    # 2. pressure family x {t1, t2, steer}
    pt = pressure_table(cells, models)
    _write_csv(os.path.join(OUT_DIR, "dist_pressure.csv"),
               ["pressure", "t1_comply", "t2_hold", "steerability"],
               [[p, pt[p]["t1_comply"], pt[p]["t2_hold"], pt[p]["steerability"]]
                for p in pt])

    # 3. domain x pressure heatmap (+ grid CSVs for the paper's shaded tables:
    # the appendix renders these as leaderboard-style tables, not figures)
    domains, fams, grid = domain_pressure_grid(cells, models)
    plot_domain_pressure(domains, fams, grid,
                         os.path.join(FIG_DIR, "domain_pressure_heatmap.png"))
    _write_csv(os.path.join(OUT_DIR, "dist_domain_pressure.csv"),
               ["domain"] + fams,
               [[dom] + grid[i] for i, dom in enumerate(domains)])

    def _cell_mean(m, want_domain=None, want_pressure=None):
        vals = [c.rate for (mm, a, _), c in cells.items()
                if mm == m and a == "base" and c.rate is not None
                and (want_domain is None or c.domain == want_domain)
                and (want_pressure is None or
                     (c.group == "pressure" and c.pressure == want_pressure))]
        return sum(vals) / len(vals) if vals else None

    dom_keys = [d for d in sorted({c.domain for (_, a, _), c in cells.items()
                                   if a == "base"})]
    _write_csv(os.path.join(OUT_DIR, "dist_model_domain.csv"),
               ["model"] + dom_keys,
               [[m] + [_cell_mean(m, want_domain=d) for d in dom_keys]
                for m in models])
    _write_csv(os.path.join(OUT_DIR, "dist_model_pressure.csv"),
               ["model"] + list(fams),
               [[m] + [_cell_mean(m, want_pressure=f) for f in fams]
                for m in models])

    # 4. per-domain Default Compliance vs Transparency scatter (main paper).
    # The domain steerability-vs-baseline r is still printed for the appendix
    # prose (the headroom check); its scatter was retired for this figure.
    plot_compliance_transparency(
        da, tbd, os.path.join(FIG_DIR, "compliance_transparency.png"))
    dd = [da[d]["default_compliance"] for d in da]
    ss = [da[d]["steerability"] for d in da]
    tt = [tbd[d] for d in da if tbd.get(d) is not None]
    dd_t = [da[d]["default_compliance"] for d in da if tbd.get(d) is not None]
    print(f"\n  r(domain default, steerability) = {_pearson(dd, ss):+.2f}  "
          f"(headroom would predict a negative slope)")
    print(f"  r(domain default, transparency) = {_pearson(dd_t, tt):+.2f}")

    # 5. pairwise significance (uncertainty reporting; the per-axis CI table is
    # built by axis_ci.py + make_paper_assets)
    metrics_path = os.path.join(OUT_DIR, "metrics_v2.csv")
    contrasts_path = os.path.join(OUT_DIR, "contrasts_v2.csv")
    if os.path.exists(metrics_path) and os.path.exists(contrasts_path):
        sig = significance_summary(contrasts_path, metrics_path)
        print(f"\n  pairwise: {sig['n_sig']}/{sig['n_pairs']} significant at BH p<0.05; "
              f"top cluster (tied with {FS.short(sig['top'])}): "
              f"{len(sig['cluster'])} models -> {[FS.short(m) for m in sig['cluster']]}")

    # headline findings for the prose
    print("\n== domain x axis (panel mean) ==")
    for dom in sorted(da, key=lambda d: (da[d]["default_compliance"] or 9)):
        r = da[dom]
        tv = tbd.get(dom)
        # thin-space labels break cp1252 consoles; plain space for the printout
        print(f"  {short_domain(dom).replace(chr(0x2009), ' '):14s} " +
              " ".join(f"{a[:4]}={r[a]:.2f}" if r[a] is not None else f"{a[:4]}=n/a"
                       for a in axis_names) +
              (f" tran={tv:.3f}" if tv is not None else " tran=n/a"))
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
