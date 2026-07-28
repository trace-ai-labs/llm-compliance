"""Inter-rater agreement for PACT's LLM classifiers.

Two families, reported SEPARATELY (spec: keep them distinct):

  Reasoning classifiers -- the honesty (axis 5) and unclear-reason taxonomies.
    Each residual reply is labeled leave-one-out by the TAXONOMY_TRIO
    (GLM-5.2, Kimi-K2.6, Nemotron-3-Ultra); a model never grades its own reply,
    so trio-authored trials get 2 raters and all others 3. gpt-oss-120b
    additionally labeled a stratified subset and is reported here as an EXTERNAL
    cross-check rater (its votes do not enter the shipped ensemble).

  Annotator judge -- the generation-stage guard review (guard_log.jsonl): for
    every authored component two non-authoring trio models independently voted
    PASS / FAIL on scope + authenticity. Agreement is per guard pair.

For each classifier we report, over the trials/components with >=2 raters:
  * per-rater-pair Cohen's kappa (+ overlap n and raw agreement)
  * mean pairwise Cohen's kappa
  * Fleiss' kappa on the fully-rated subset (constant rater count)
  * pooled raw pairwise agreement
  * the label distribution

Outcome judging (which menu option / comply-violate-unclear) is NOT here: it is
a single gpt-oss-120b LLM judge per trial (no ensemble, no regex), so it has no
inter-rater number; its validity is the judge-swap tau-gate in judges.py.

Numbers  -> results/benchmark/agreement/*.csv
Figures  -> results/benchmark/figures/agreement_*.png (also copied to paper/figures/)

    python -m src.benchmark.agreement            # compute all, write csv + figures
    python -m src.benchmark.agreement --no-fig   # numbers only
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import shutil
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

from src.benchmark.batch import TAXONOMY_TRIO
from src.benchmark.judges import (HONESTY_LABELS, HONESTY_PATH, UNCLEAR_LABELS,
                                  UNCLEAR_PATH, _model_tag, cohen_kappa,
                                  parse_honesty, parse_unclear)

RESULTS_DIR = os.path.join("results", "benchmark")
AGREE_DIR = os.path.join(RESULTS_DIR, "agreement")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
PAPER_FIG_DIR = os.path.join("paper", "figures")
GUARD_LOG = os.path.join(RESULTS_DIR, "guard_log.jsonl")

GPT_OSS_TAG = "gpt-oss-120b"
TRIO_TAGS = tuple(_model_tag(m) for m in TAXONOMY_TRIO)


# ── agreement math ────────────────────────────────────────────────────────────

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


def fleiss_kappa(per_item: Dict[str, Dict[str, str]], raters: set,
                 categories: Tuple[str, ...]) -> Tuple[Optional[float], int]:
    """Fleiss' kappa on the subset of items rated by EXACTLY `raters` (constant
    rater count n, which Fleiss requires). Returns (kappa, n_items)."""
    n = len(raters)
    if n < 2:
        return None, 0
    # one Counter per item rated by the full rater set, counting only those raters
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
    # p_j: proportion of all assignments to category j
    total_assign = N * n
    p_j = {c: sum(m.get(c, 0) for m in matrix) / total_assign for c in cats}
    # P_i: extent of agreement for item i
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


# ── data loading ──────────────────────────────────────────────────────────────

def read_ensemble(path: str, parse_fn: Callable[[str], Optional[str]]
                  ) -> Dict[str, Dict[str, str]]:
    """{trial_id: {judge_tag: label}} over ALL judges present in the file
    (trio + any external cross-check rater). No trio filter -- that is applied
    downstream so trio-only and trio+external numbers can both be reported."""
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


def honesty_kappa_by_domain(min_n: int = 30) -> Dict[str, Optional[float]]:
    """{domain: mean pairwise Cohen kappa among the trio honesty raters}, for the
    per-domain judge-reliability column in metrics_v2.md. Domains with fewer than
    `min_n` multi-rater trials return None (too few binding violations to be
    stable). Honesty trial ids look like 'MUT||domain.letter.author...|arm|rep'."""
    per_item = read_ensemble(HONESTY_PATH, parse_honesty)
    primary = set(TRIO_TAGS)
    by_dom: Dict[str, Dict[str, Dict[str, str]]] = {}
    for tid, labs in per_item.items():
        try:
            dom = tid.split("||", 1)[1].split(".", 1)[0]
        except IndexError:
            continue
        by_dom.setdefault(dom, {})[tid] = labs
    out: Dict[str, Optional[float]] = {}
    for dom, pj in by_dom.items():
        n_multi = sum(1 for labs in pj.values()
                      if len([j for j in labs if j in primary]) >= 2)
        out[dom] = mean_pairwise(pairwise_table(pj), only=primary) if n_multi >= min_n else None
    return out


def honesty_judge_rankings(min_n: int = 15
                           ) -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, int]]:
    """Rank models by reasoning-honesty (1 - silent rate) under EACH trio judge
    individually and under the majority-vote ensemble (what the shipped metric
    uses), all from the saved per-judge labels in honesty.jsonl -- no re-judging.

    Honesty trial ids are 'MUT||item...|arm|rep', so the model under test is the
    part before '||'. To keep the comparison strictly apples-to-apples we score
    every judge and the ensemble on the SAME trials: only those labeled by ALL
    THREE trio judges (the fully-rated subset). Under honesty's leave-one-out
    rule a trio model never judges its own responses, so a trio model as MUT is
    only ever labeled by two judges and is therefore excluded from this analysis
    (its honesty score still ships; it just cannot enter a 3-judge comparison).
    For judge J, model M's honesty is 1 - silent_J(M)/n(M) over that identical
    trial set; the ensemble scores a trial silent iff >=2 of the 3 labeled SILENT.

    Returns (rankers, scores, n_by_model): rankers is
    ['GLM-5.2','Kimi-K2.6','NVIDIA-Nemotron-3-Ultra-550B-A55B','ensemble'];
    scores[ranker][model] = honesty; n_by_model[model] = fully-rated trial count.
    Only models with >= min_n fully-rated trials are kept (stable ranking)."""
    per_item = read_ensemble(HONESTY_PATH, parse_honesty)
    judges = list(TRIO_TAGS)
    # per model, over the fully-rated subset: silent count per judge + ensemble
    sil_j: Dict[str, Dict[str, int]] = {j: Counter() for j in judges}
    tot: Counter = Counter()            # identical denominator for every judge
    sil_ens: Counter = Counter()
    for tid, labs in per_item.items():
        if any(j not in labs for j in judges):
            continue                    # require all three trio labels
        model = tid.split("||", 1)[0]
        tot[model] += 1
        for j in judges:
            if labs[j] == "SILENT":
                sil_j[j][model] += 1
        if sum(1 for j in judges if labs[j] == "SILENT") >= 2:   # majority of 3
            sil_ens[model] += 1
    n_by_model = dict(tot)
    models = [m for m in n_by_model if n_by_model[m] >= min_n]
    scores: Dict[str, Dict[str, float]] = {}
    for j in judges:
        scores[j] = {m: 1 - sil_j[j][m] / tot[m] for m in models}
    scores["ensemble"] = {m: 1 - sil_ens[m] / tot[m] for m in models}
    rankers = judges + ["ensemble"]
    return rankers, scores, {m: n_by_model[m] for m in models}


def read_guard_pairs() -> Dict[str, Dict[str, str]]:
    """Reshape guard_log.jsonl into {component_key: {guard_tag: verdict}} so the
    same pairwise machinery applies. Each log row records one guard's verdict and
    its co-guard's verdict on the same component; we register both."""
    per_item: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(GUARD_LOG):
        return per_item
    with open(GUARD_LOG, encoding="utf-8") as f:
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


# ── per-classifier report ─────────────────────────────────────────────────────

def build_report(name: str, per_item: Dict[str, Dict[str, str]],
                 categories: Tuple[str, ...], primary: set,
                 binary_map: Optional[Callable[[str], str]] = None) -> dict:
    rows = pairwise_table(per_item)
    trio_rows = [r for r in rows if r[0] in primary and r[1] in primary]
    fleiss_all, fleiss_n = fleiss_kappa(per_item, primary, categories)
    raw_primary, raw_pairs = pooled_raw_agreement(per_item, only=primary)
    n_items = sum(1 for labs in per_item.values()
                  if len([j for j in labs if j in primary]) >= 2)
    dist = Counter()
    for labs in per_item.values():
        for j, l in labs.items():
            if j in primary:
                dist[l] += 1
    rep = {
        "name": name,
        "n_items_multirater": n_items,
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
        w.writerow(["mean_pairwise_kappa (primary)",
                    _r(rep["mean_pairwise_kappa"])])
        w.writerow(["fleiss_kappa (primary)", _r(rep["fleiss_kappa"])])
        w.writerow(["fleiss_n_items", rep["fleiss_n_items"]])
        w.writerow(["pooled_raw_agreement (primary)",
                    _r(rep["pooled_raw_agreement"])])
        if "binary_mean_pairwise_kappa" in rep:
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


# ── figure (one per classifier) ────────────────────────────────────────────────

def make_figure(rep: dict, title: str, subtitle: str, out_name: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colormaps, colors
    except Exception as e:  # pragma: no cover
        print(f"  (matplotlib unavailable: {e}; skipping figure)")
        return None
    from src.benchmark import figstyle
    figstyle.use_paper_style()

    pairs = sorted(rep["pairs"], key=lambda r: r[3])  # by kappa asc
    labels = [f"{figstyle.short(a)}  x  {figstyle.short(b)}"
              for a, b, n, k, raw in pairs]
    ks = [r[3] for r in pairs]
    ns = [r[2] for r in pairs]
    is_prim = [(r[0] in rep["primary"] and r[1] in rep["primary"]) for r in pairs]

    norm = colors.Normalize(vmin=-0.2, vmax=1.0)
    cmap = colormaps[figstyle.SCORE_CMAP]
    bar_colors = [cmap(norm(k)) for k in ks]

    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(pairs) + 2.2))
    y = range(len(pairs))
    ax.barh(list(y), ks, color=bar_colors, edgecolor="white", height=0.66, zorder=3)
    for i, (k, n, prim) in enumerate(zip(ks, ns, is_prim)):
        ax.text(k + 0.015 if k >= 0 else k - 0.015, i,
                f"{k:.2f}  (n={n:,}{'' if prim else ', ext'})",
                va="center", ha="left" if k >= 0 else "right",
                fontsize=8, color=figstyle.INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(-0.25, 1.15)
    ax.axvline(0.0, color=figstyle.MUTED, lw=1.0, ls=":", zorder=2)

    mk = rep["mean_pairwise_kappa"]
    if mk is not None:
        ax.axvline(mk, color=figstyle.INK, lw=1.4, zorder=4,
                   label=f"mean = {mk:.2f}")
    ax.set_xlabel("Cohen's kappa (per rater pair)")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=10)

    # headline stats as a caption line (collision-free below the plot)
    stat_bits = []
    if mk is not None:
        stat_bits.append(f"mean pairwise kappa {mk:.2f} ({band(mk)})")
    if rep.get("fleiss_kappa") is not None:
        stat_bits.append(f"Fleiss kappa {rep['fleiss_kappa']:.2f} (n={rep['fleiss_n_items']:,})")
    if rep.get("pooled_raw_agreement") is not None:
        stat_bits.append(f"raw agreement {rep['pooled_raw_agreement']*100:.0f}%")
    if rep.get("binary_mean_pairwise_kappa") is not None:
        stat_bits.append(f"SILENT-vs-not kappa {rep['binary_mean_pairwise_kappa']:.2f}")
    fig.text(0.01, 0.075, "   |   ".join(stat_bits), fontsize=9,
             color=figstyle.INK, ha="left", fontweight="bold")
    fig.text(0.01, 0.02, subtitle, fontsize=8, color=figstyle.MUTED, ha="left")

    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=figstyle.GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.11, 1, 1))

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(PAPER_FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, out_name)
    fig.savefig(path, dpi=150)
    # also a vector PDF: the paper's \includegraphics omits the extension and
    # prefers the PDF when both are present (matches make_paper_assets.py).
    pdf_name = out_name[:-4] + ".pdf" if out_name.lower().endswith(".png") else out_name + ".pdf"
    fig.savefig(os.path.join(FIG_DIR, pdf_name))
    plt.close(fig)
    for n in (out_name, pdf_name):
        shutil.copy2(os.path.join(FIG_DIR, n), os.path.join(PAPER_FIG_DIR, n))
    return path


# ── driver ──────────────────────────────────────────────────────────────────

def _fmt(rep: dict) -> str:
    def g(k):
        return "n/a" if rep.get(k) is None else f"{rep[k]:.3f}"
    lines = [
        f"  n items (>=2 primary raters): {rep['n_items_multirater']:,}",
        f"  mean pairwise Cohen kappa   : {g('mean_pairwise_kappa')}  ({band(rep['mean_pairwise_kappa'])})",
        f"  Fleiss kappa (n={rep['fleiss_n_items']:,} full)   : {g('fleiss_kappa')}  ({band(rep['fleiss_kappa'])})",
        f"  pooled raw pairwise agree   : {g('pooled_raw_agreement')}",
    ]
    if "binary_mean_pairwise_kappa" in rep:
        lines.append(f"  binary (SILENT vs not) kappa: {g('binary_mean_pairwise_kappa')}  "
                     f"({band(rep['binary_mean_pairwise_kappa'])})")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-fig", action="store_true", help="numbers/CSV only")
    args = ap.parse_args(argv)

    primary = set(TRIO_TAGS)

    # 1. Reasoning classifier: honesty (axis 5)
    hon = read_ensemble(HONESTY_PATH, parse_honesty)
    rep_h = build_report("reasoning_honesty", hon, HONESTY_LABELS, primary,
                         binary_map=lambda l: "silent" if l == "SILENT" else "non_silent")
    # 2. Reasoning classifier: unclear-reason
    unc = read_ensemble(UNCLEAR_PATH, parse_unclear)
    rep_u = build_report("reasoning_unclear", unc, UNCLEAR_LABELS, primary)
    # 3. Annotator judge: generation guard review
    grd = read_guard_pairs()
    guard_raters = {r for d in grd.values() for r in d}
    rep_g = build_report("annotator_guard", grd, ("PASS", "FAIL"), guard_raters)

    reports = [rep_h, rep_u, rep_g]
    titles = {
        "reasoning_honesty": "Reasoning classifier - Motive taxonomy",
        "reasoning_unclear": "Reasoning classifier - Unclear-reason",
        "annotator_guard": "Annotator judge - Generation guard review",
    }
    print("\n=== PACT classifier inter-rater agreement ===\n")
    print("REASONING CLASSIFIERS (TAXONOMY_TRIO, leave-one-out; gpt-oss = external cross-check)\n")
    print("Honesty (axis 5):")
    print(_fmt(rep_h))
    print("\nUnclear-reason:")
    print(_fmt(rep_u))
    print("\nANNOTATOR JUDGE (generation guard review, non-authoring trio, per pair)\n")
    print(_fmt(rep_g))

    for rep in reports:
        p = write_csv(rep)
        print(f"\n  wrote {p}")

    if not args.no_fig:
        make_figure(rep_h, titles["reasoning_honesty"],
                    "Reasoning classifier. TAXONOMY_TRIO leave-one-out on binding-violation replies; "
                    "'ext' pairs involve the swapped-out gpt-oss-120b cross-check rater.",
                    "agreement_reasoning_honesty.png")
        make_figure(rep_u, titles["reasoning_unclear"],
                    "Reasoning classifier. TAXONOMY_TRIO leave-one-out on first-turn abstentions; "
                    "'ext' pairs involve the swapped-out gpt-oss-120b cross-check rater.",
                    "agreement_reasoning_unclear.png")
        make_figure(rep_g, titles["annotator_guard"],
                    "Annotator judge. Two non-authoring trio models vote PASS/FAIL per generated component.",
                    "agreement_annotator_guard.png")
        print(f"\n  figures -> {FIG_DIR} (and {PAPER_FIG_DIR})")


if __name__ == "__main__":
    main()
