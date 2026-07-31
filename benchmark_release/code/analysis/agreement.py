"""Inter-rater agreement for the LLM classifiers (appendix 'Classifier
Agreement'), plus the transparency judge-stability table. Writes
results/agreement/*.csv.

Usage: python -m analysis.agreement
"""
from __future__ import annotations

import csv
import itertools
import json
import os
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import paths
from analysis.metrics import wilson_ci
from evaluation.batch import TAXONOMY_TRIO
from evaluation.judges import (TRANSPARENCY_LABELS, UNCLEAR_LABELS, _model_tag,
                               cohen_kappa, load_trials, parse_transparency,
                               parse_unclear)

AGREE_DIR = os.path.join(paths.RESULTS, "agreement")

TRIO_TAGS = tuple(_model_tag(m) for m in TAXONOMY_TRIO)


def band(k: Optional[float]) -> str:
    """Landis & Koch (1977) interpretation band."""
    if k is None:
        return "n/a"
    if k < 0.0:
        return "poor"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def pairwise_table(per_item: Dict[str, Dict[str, str]]
                   ) -> List[Tuple[str, str, int, float, float]]:
    """One row per rater pair that co-rated >=2 items:
    (rater_a, rater_b, n_overlap, cohen_kappa, raw_agreement)."""
    raters = sorted({r for d in per_item.values() for r in d})
    rows: List[Tuple[str, str, int, float, float]] = []
    for a, b in itertools.combinations(raters, 2):
        la, lb = [], []
        for d in per_item.values():
            if a in d and b in d:
                la.append(d[a])
                lb.append(d[b])
        if len(la) < 2:
            continue
        raw = sum(x == y for x, y in zip(la, lb)) / len(la)
        k = cohen_kappa(la, lb) if len(set(la + lb)) > 1 else 1.0
        rows.append((a, b, len(la), k, raw))
    return rows


def mean_pairwise(rows: List[Tuple[str, str, int, float, float]],
                  only: Optional[set] = None) -> Optional[float]:
    ks = [k for a, b, n, k, raw in rows
          if only is None or (a in only and b in only)]
    return sum(ks) / len(ks) if ks else None


def pooled_raw_agreement(per_item: Dict[str, Dict[str, str]],
                         only: Optional[set] = None) -> Tuple[Optional[float], int]:
    agree = tot = 0
    for labs in per_item.values():
        js = sorted(j for j in labs if only is None or j in only)
        for i in range(len(js)):
            for j in range(i + 1, len(js)):
                tot += 1
                agree += labs[js[i]] == labs[js[j]]
    return ((agree / tot) if tot else None), tot


def unanimity(per_item: Dict[str, Dict[str, str]], only: Optional[set] = None
              ) -> Tuple[Optional[float], Tuple[float, float], int]:
    """Share of items (>=2 raters) on which every rater returned the same
    label, with its Wilson 95% CI: the paper's agreement table."""
    n = k = 0
    for labs in per_item.values():
        vals = [l for j, l in labs.items() if only is None or j in only]
        if len(vals) < 2:
            continue
        n += 1
        k += len(set(vals)) == 1
    if n == 0:
        return None, (0.0, 1.0), 0
    return k / n, wilson_ci(k, n), n


def fleiss_kappa(per_item: Dict[str, Dict[str, str]], raters: set,
                 categories: Tuple[str, ...]) -> Tuple[Optional[float], int]:
    """Fleiss' kappa on the subset of items rated by EXACTLY `raters` (constant
    rater count n, which Fleiss requires). Returns (kappa, n_items)."""
    n = len(raters)
    if n < 2:
        return None, 0
    matrix: List[Counter] = []
    for labs in per_item.values():
        if not raters.issubset(labs.keys()):
            continue
        c = Counter(labs[r] for r in raters)
        matrix.append(c)
    N = len(matrix)
    if N == 0:
        return None, 0
    cats = list(categories)
    total_assign = N * n
    p_j = {c: sum(m.get(c, 0) for m in matrix) / total_assign for c in cats}
    P_bar = 0.0
    for m in matrix:
        s = sum(v * v for v in m.values())
        P_i = (s - n) / (n * (n - 1))
        P_bar += P_i
    P_bar /= N
    P_e = sum(v * v for v in p_j.values())
    if P_e >= 1.0:
        return 1.0, N
    return (P_bar - P_e) / (1 - P_e), N


def read_ensemble(path: str, parse_fn: Callable[[str], Optional[str]]
                  ) -> Dict[str, Dict[str, str]]:
    """{trial_id: {judge_tag: label}} over ALL judges present in the file."""
    per_item: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(path):
        return per_item
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("ok"):
                continue
            label = parse_fn(row.get("content", ""))
            if not label:
                continue
            meta = row.get("meta", {})
            tid = meta.get("trial_id")
            jtag = _model_tag(meta.get("judge_model", "?"))
            if tid:
                per_item.setdefault(tid, {})[jtag] = label
    return per_item


def read_guard_pairs() -> Dict[str, Dict[str, str]]:
    """{component_key: {guard_tag: verdict}} from the guard log; each row
    carries one guard's verdict and its co-guard's, so both are registered."""
    per_item: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(paths.GUARD_LOG):
        return per_item
    with open(paths.GUARD_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (f"{r.get('scenario_id')}|{r.get('generator_model')}|"
                   f"{r.get('component')}|{r.get('pressure_key')}|{r.get('attempt')}")
            g, gv = r.get("guard_model"), r.get("verdict")
            cg, cgv = r.get("co_guard_model"), r.get("co_verdict")
            if g and gv:
                per_item.setdefault(key, {})[_model_tag(g)] = gv
            if cg and cgv:
                per_item.setdefault(key, {})[_model_tag(cg)] = cgv
    return per_item


def build_report(name: str, per_item: Dict[str, Dict[str, str]],
                 categories: Tuple[str, ...], primary: set,
                 binary_map: Optional[Callable[[str], str]] = None) -> dict:
    rows = pairwise_table(per_item)
    fleiss_all, fleiss_n = fleiss_kappa(per_item, primary, categories)
    raw_primary, raw_pairs = pooled_raw_agreement(per_item, only=primary)
    unan, unan_ci, n_items = unanimity(per_item, only=primary)
    dist = Counter()
    for labs in per_item.values():
        for j, l in labs.items():
            if j in primary:
                dist[l] += 1
    rep = {
        "name": name,
        "n_items_multirater": n_items,
        "unanimous": unan,
        "unanimous_ci": unan_ci,
        "mean_pairwise_kappa": mean_pairwise(rows, only=primary),
        "fleiss_kappa": fleiss_all,
        "fleiss_n_items": fleiss_n,
        "pooled_raw_agreement": raw_primary,
        "pairs": rows,
        "primary": primary,
        "label_dist": dict(dist),
    }
    if binary_map is not None:
        bpj = {t: {j: binary_map(l) for j, l in labs.items()}
               for t, labs in per_item.items()}
        brows = pairwise_table(bpj)
        rep["binary_mean_pairwise_kappa"] = mean_pairwise(brows, only=primary)
        bf, _ = fleiss_kappa(bpj, primary, tuple(sorted({binary_map(c) for c in categories})))
        rep["binary_fleiss_kappa"] = bf
        bu, bu_ci, _bn = unanimity(bpj, only=primary)
        rep["binary_unanimous"] = bu
        rep["binary_unanimous_ci"] = bu_ci
    return rep


def write_csv(rep: dict) -> str:
    os.makedirs(AGREE_DIR, exist_ok=True)
    path = os.path.join(AGREE_DIR, f"{rep['name']}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rater_a", "rater_b", "n_overlap", "cohen_kappa",
                    "raw_agreement", "in_primary_set"])
        for a, b, n, k, raw in rep["pairs"]:
            in_prim = a in rep["primary"] and b in rep["primary"]
            w.writerow([a, b, n, round(k, 4), round(raw, 4), in_prim])
        w.writerow([])
        w.writerow(["summary", "value"])
        w.writerow(["n_items_multirater", rep["n_items_multirater"]])
        w.writerow(["unanimous_share (primary)", _r(rep["unanimous"])])
        w.writerow(["unanimous_wilson_lo", _r(rep["unanimous_ci"][0])])
        w.writerow(["unanimous_wilson_hi", _r(rep["unanimous_ci"][1])])
        w.writerow(["mean_pairwise_kappa (primary)",
                    _r(rep["mean_pairwise_kappa"])])
        w.writerow(["fleiss_kappa (primary)", _r(rep["fleiss_kappa"])])
        w.writerow(["fleiss_n_items", rep["fleiss_n_items"]])
        w.writerow(["pooled_raw_agreement (primary)",
                    _r(rep["pooled_raw_agreement"])])
        if "binary_mean_pairwise_kappa" in rep:
            w.writerow(["binary_unanimous_share", _r(rep["binary_unanimous"])])
            w.writerow(["binary_mean_pairwise_kappa",
                        _r(rep["binary_mean_pairwise_kappa"])])
            w.writerow(["binary_fleiss_kappa", _r(rep["binary_fleiss_kappa"])])
        w.writerow([])
        w.writerow(["label", "count (primary raters)"])
        for l, c in sorted(rep["label_dist"].items(), key=lambda x: -x[1]):
            w.writerow([l, c])
    return path


def _r(x: Optional[float]) -> str:
    return "" if x is None else f"{x:.4f}"


def _fmt(rep: dict) -> str:
    def g(k):
        return "n/a" if rep.get(k) is None else f"{rep[k]:.3f}"
    lo, hi = rep["unanimous_ci"]
    lines = [
        f"  n items (>=2 primary raters): {rep['n_items_multirater']:,}",
        f"  unanimous share [95% CI]    : {g('unanimous')} [{lo:.3f}, {hi:.3f}]",
        f"  mean pairwise Cohen kappa   : {g('mean_pairwise_kappa')}  ({band(rep['mean_pairwise_kappa'])})",
        f"  pooled raw pairwise agree   : {g('pooled_raw_agreement')}",
    ]
    if rep["fleiss_kappa"] is not None:
        lines.insert(3, f"  Fleiss kappa (n={rep['fleiss_n_items']:,} full)   : "
                        f"{g('fleiss_kappa')}  ({band(rep['fleiss_kappa'])})")
    if "binary_unanimous" in rep:
        blo, bhi = rep["binary_unanimous_ci"]
        lines.append(f"  binary unanimous share      : {g('binary_unanimous')} "
                     f"[{blo:.3f}, {bhi:.3f}]")
        lines.append(f"  binary mean pairwise kappa  : {g('binary_mean_pairwise_kappa')}  "
                     f"({band(rep['binary_mean_pairwise_kappa'])})")
    return "\n".join(lines)


def transparency_judge_stability(min_shift_n: int = 1) -> dict:
    """Appendix table 'The transparency ranking is stable across judges': over
    trials rated by all three judges, models outside the judge trio, compare
    the per-judge transparent-share leaderboards and the ensemble (fractional
    vote-share) leaderboard - share of model pairs ordered the same, and mean
    absolute rank shift."""
    per_item = read_ensemble(paths.TRANSPARENCY, parse_transparency)
    judges = list(TRIO_TAGS)
    trio_models = set(TAXONOMY_TRIO)
    tr_j: Dict[str, Counter] = {j: Counter() for j in judges}
    tot: Counter = Counter()
    ens_share: Dict[str, float] = Counter()
    for utid, labs in per_item.items():
        if any(j not in labs for j in judges):
            continue
        model = utid.split("||", 1)[0]
        if model in trio_models:
            continue
        tot[model] += 1
        for j in judges:
            if labs[j] == "TRANSPARENT":
                tr_j[j][model] += 1
        ens_share[model] += sum(labs[j] == "TRANSPARENT" for j in judges) / len(judges)
    models = sorted(tot)
    scores: Dict[str, Dict[str, float]] = {}
    for j in judges:
        scores[j] = {m: tr_j[j][m] / tot[m] for m in models}
    scores["ensemble"] = {m: ens_share[m] / tot[m] for m in models}

    def rank(sc: Dict[str, float]) -> Dict[str, int]:
        order = sorted(models, key=lambda m: (-sc[m], m))
        return {m: i + 1 for i, m in enumerate(order)}

    def compare(sa: Dict[str, float], sb: Dict[str, float]) -> Tuple[float, float]:
        same = total = 0
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                total += 1
                da, db = sa[a] - sa[b], sb[a] - sb[b]
                same += (da > 0 and db > 0) or (da < 0 and db < 0) \
                    or (da == 0 and db == 0)
        ra, rb = rank(sa), rank(sb)
        shift = sum(abs(ra[m] - rb[m]) for m in models) / len(models)
        return (same / total if total else 0.0), shift

    out = {"n_models": len(models), "n_trials": sum(tot.values()), "pairs": []}
    rankers = judges + ["ensemble"]
    for i, a in enumerate(rankers):
        for b in rankers[i + 1:]:
            same, shift = compare(scores[a], scores[b])
            out["pairs"].append((a, b, same, shift))
    return out


def main() -> None:
    primary = set(TRIO_TAGS)

    trn = read_ensemble(paths.TRANSPARENCY, parse_transparency)
    rep_t = build_report("transparency", trn, TRANSPARENCY_LABELS, primary,
                         binary_map=lambda l: "transparent" if l == "TRANSPARENT" else "not")
    unc = read_ensemble(paths.UNCLEAR, parse_unclear)
    # keep only turns whose FINAL outcome is unclear; T2 turns are keyed ||t2
    trials = load_trials(paths.TRIALS_DIR)
    final_unclear = {f"{t['model']}||{t['trial_id']}" for t in trials
                     if t.get("t1_outcome") == "unclear"}
    final_unclear |= {f"{t['model']}||{t['trial_id']}||t2" for t in trials
                      if t.get("t2_outcome") == "unclear"
                      and t.get("t2_script") == "pushback"}
    unc = {u: v for u, v in unc.items() if u in final_unclear}
    rep_u = build_report("unclear_reason", unc, UNCLEAR_LABELS, primary)
    grd = read_guard_pairs()
    guard_raters = {r for d in grd.values() for r in d}
    rep_g = build_report("guard_review", grd, ("PASS", "FAIL"), guard_raters)

    print("\n=== classifier inter-rater agreement ===\n")
    print("Transparency (axis 5):")
    print(_fmt(rep_t))
    print("\nUnclear-reason:")
    print(_fmt(rep_u))
    print("\nGuard review (generation stage, two reviewers per component):")
    print(_fmt(rep_g))

    for rep in (rep_t, rep_u, rep_g):
        p = write_csv(rep)
        print(f"\n  wrote {p}")

    st = transparency_judge_stability()
    print(f"\n=== transparency judge stability "
          f"({st['n_models']} models, {st['n_trials']:,} fully-rated trials) ===")
    for a, b, same, shift in st["pairs"]:
        print(f"  {a} x {b}: pairs same order {same:.2f}, "
              f"mean rank shift {shift:.1f}")


if __name__ == "__main__":
    main()
