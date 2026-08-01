"""Domain and pressure-family breakouts (Section 4, 'Compliance varies across
pressures and domains' + appendix 'Difficulty by Domain and Pressure').
Prints the tables and writes results/dist_*.csv.

Usage: python -m analysis.distributions
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import paths
from analysis import metrics as M
from evaluation.judges import load_trials, load_transparency_votes
from generation.registry import SCORED_PRESSURES

# The 5 cell-based axes, panel-meaned per domain; transparency is pooled
# separately (transparency_by_domain) because per-model per-domain violation
# counts are too thin to average.
AXES = [
    ("default_compliance", M.default_compliance),
    ("pressure_resistance", M.pressure_resistance),
    ("pushback_resistance", lambda cs, m: M.pushback_resistance(cs, m, []).value),
    ("steerability", lambda cs, m: M.steerability(cs, m).net),
    ("rule_scope_discernment", lambda cs, m: M.rule_scope_discernment(cs, m).value),
]


def _pooled_rate(cells: Sequence[M.Cell], attr: str) -> Optional[float]:
    """comply / (comply + violate) over all reps in `cells`, unclear dropped.
    attr is 't1' or 't2_pushback'."""
    comply = decided = 0
    for c in cells:
        for o in getattr(c, attr):
            if o == "unclear":
                continue
            decided += 1
            comply += (o == "comply")
    return comply / decided if decided else None


def _pooled_recovery(base: Sequence[M.Cell], directive: Sequence[M.Cell]
                     ) -> Optional[float]:
    """Signed steerability recovery pooled over a cell group, matched by item."""
    b = {c.item: c.rate for c in base if c.rate is not None}
    d = {c.item: c.rate for c in directive if c.rate is not None}
    gain = mass = 0.0
    for iid, br in b.items():
        if br >= 1.0 or iid not in d:
            continue
        mass += 1.0 - br
        gain += d[iid] - br
    return gain / mass if mass > 0 else None


def transparency_by_domain(trials: List[dict], models: List[str]
                           ) -> Dict[str, Optional[float]]:
    """domain -> TRANSPARENT vote-share, pooled over every judged violation
    the panel committed in the domain (both system-prompt modes)."""
    votes = load_transparency_votes(paths.TRANSPARENCY)
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


def domain_axis_table(cells: Dict, models: List[str]
                      ) -> Dict[str, Dict[str, Optional[float]]]:
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
    """pressure family -> {t1 comply, t2 hold, steerability}, panel-pooled."""
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for fam in SCORED_PRESSURES:
        base_p = [c for (m, a, _), c in cells.items()
                  if a == "base" and c.pressure == fam and m in models]
        dir_p = [c for (m, a, _), c in cells.items()
                 if a == "mandate" and c.pressure == fam and m in models]
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
                  if a == "base" and c.domain == dom and c.pressure == fam
                  and m in models]
            row.append(_pooled_rate(cs, "t1"))
        grid.append(row)
    return domains, fams, grid


def _write_csv(path: str, header: List[str], rows: List[List]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    sa = (sum((x - ma) ** 2 for x in a)) ** 0.5
    sb = (sum((y - mb) ** 2 for y in b)) ** 0.5
    return cov / (sa * sb) if sa and sb else 0.0


def significance_summary(contrasts_path: str, metrics_path: str) -> dict:
    """From the BH-adjusted pairwise contrasts and the PACTScore ranking: how
    many pairs differ significantly, and the top indistinguishable cluster."""
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
    trials = load_trials(paths.TRIALS_DIR)
    cells = M.build_cells(trials)
    models = sorted({m for (m, _, _) in cells})
    os.makedirs(paths.RESULTS, exist_ok=True)

    # 1. domain x axis, plus pooled transparency and raw turn-2 hold per domain
    da = domain_axis_table(cells, models)
    axis_names = [n for n, _ in AXES]
    tbd = transparency_by_domain(trials, models)
    t2_by_domain = {
        dom: _pooled_rate([c for (m, a, _), c in cells.items()
                           if a == "base" and c.domain == dom and m in models],
                          "t2_pushback")
        for dom in da}
    _write_csv(os.path.join(paths.RESULTS, "dist_domain_axis.csv"),
               ["domain"] + axis_names + ["transparency", "turn2_hold"],
               [[dom] + [da[dom][a] for a in axis_names]
                + [tbd.get(dom), t2_by_domain[dom]]
                for dom in da])

    # 2. pressure family x {t1, t2, steer}
    pt = pressure_table(cells, models)
    _write_csv(os.path.join(paths.RESULTS, "dist_pressure.csv"),
               ["pressure", "t1_comply", "t2_hold", "steerability"],
               [[p, pt[p]["t1_comply"], pt[p]["t2_hold"], pt[p]["steerability"]]
                for p in pt])

    # 3. domain x pressure and the per-model grids
    domains, fams, grid = domain_pressure_grid(cells, models)
    _write_csv(os.path.join(paths.RESULTS, "dist_domain_pressure.csv"),
               ["domain"] + fams,
               [[dom] + grid[i] for i, dom in enumerate(domains)])

    def _cell_mean(m, want_domain=None, want_pressure=None):
        vals = [c.rate for (mm, a, _), c in cells.items()
                if mm == m and a == "base" and c.rate is not None
                and (want_domain is None or c.domain == want_domain)
                and (want_pressure is None or
                     (c.group == "pressure" and c.pressure == want_pressure))]
        return sum(vals) / len(vals) if vals else None

    _write_csv(os.path.join(paths.RESULTS, "dist_model_domain.csv"),
               ["model"] + domains,
               [[m] + [_cell_mean(m, want_domain=d) for d in domains]
                for m in models])
    _write_csv(os.path.join(paths.RESULTS, "dist_model_pressure.csv"),
               ["model"] + list(fams),
               [[m] + [_cell_mean(m, want_pressure=f) for f in fams]
                for m in models])

    # 4. the domain-level correlations the prose quotes
    dd = [da[d]["default_compliance"] for d in da]
    ss = [da[d]["steerability"] for d in da]
    sc = [da[d]["rule_scope_discernment"] for d in da]
    print(f"\n  r(domain default, steerability) = {_pearson(dd, ss):+.2f}")
    print(f"  r(domain default, scope) = {_pearson(dd, sc):+.2f}")

    # 5. pairwise significance
    metrics_path = os.path.join(paths.RESULTS, "metrics.csv")
    contrasts_path = os.path.join(paths.RESULTS, "contrasts.csv")
    if os.path.exists(metrics_path) and os.path.exists(contrasts_path):
        sig = significance_summary(contrasts_path, metrics_path)
        pct = 100 * sig["n_sig"] / sig["n_pairs"] if sig["n_pairs"] else 0.0
        print(f"\n  pairwise: {sig['n_sig']}/{sig['n_pairs']} ({pct:.1f}%) "
              f"significant at BH p<0.05; "
              f"top cluster (tied with {sig['top']}): "
              f"{len(sig['cluster'])} models -> {sig['cluster']}")

    print("\n== domain x axis (panel mean) ==")
    for dom in sorted(da, key=lambda d: (da[d]["default_compliance"] or 9)):
        r = da[dom]
        tv = tbd.get(dom)
        print(f"  {dom:22s} " +
              " ".join(f"{a[:4]}={r[a]:.2f}" if r[a] is not None else f"{a[:4]}=n/a"
                       for a in axis_names) +
              (f" tran={tv:.3f}" if tv is not None else " tran=n/a"))
    print("\n== pressure family ==")
    for p in sorted(pt, key=lambda p: (pt[p]["t1_comply"] or 9)):
        r = pt[p]
        print(f"  {p:24s} t1={r['t1_comply']:.3f} t2={r['t2_hold']:.3f} "
              f"steer={r['steerability']:.3f}")
    flat = [(d, f, grid[i][j])
            for i, d in enumerate(domains) for j, f in enumerate(fams)
            if grid[i][j] is not None]
    lo = min(flat, key=lambda t: t[2])
    print(f"\n  weakest domain x pressure cell: {lo[0]} x {lo[1]} = {lo[2]:.3f}")
    print("wrote dist_domain_axis.csv, dist_pressure.csv, dist_domain_pressure.csv, "
          "dist_model_domain.csv, dist_model_pressure.csv")


if __name__ == "__main__":
    main()
