"""Every figure in the paper, one function each; each writes a PNG and a PDF
into figures/. `python -m analysis.figures` regenerates all of them from the
shipped record.
"""

import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import paths
from analysis import figstyle as FS
from analysis.aggregate import AXES

_LABEL_DISPLAY = {"REFUSED_ON_POLICY": "Refused on Policy",
                  "NONCONFORMING_PICK": "Off-Menu Commit",
                  "HANDED_BACK": "Handed Back", "NO_ANSWER": "No Answer",
                  "TRANSPARENT": "Transparent", "CONCEALED": "Concealed",
                  "MISREPRESENTED": "Misrepresented"}


def _save(fig, name: str, dpi: Optional[int] = None) -> None:
    """PNG + PDF into paths.FIGURES."""
    import matplotlib.pyplot as plt
    os.makedirs(paths.FIGURES, exist_ok=True)
    png = os.path.join(paths.FIGURES, name + ".png")
    if dpi is None:
        fig.savefig(png)
    else:
        fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(png[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png} (+pdf)")


def _seed_model_colors(profiles: Dict[str, Dict]) -> List[str]:
    """Fix the per-model palette assignment order (ascending PACTScore) and
    return the model list; keeps each model's color stable across figures."""
    models = [m for m in profiles if m != "_meta"]
    score = lambda m: profiles[m]["pact"].value or 0.0
    for m in sorted(models, key=score):
        FS.model_color(m)
    return models


# -- radar / radar_quad (Section 4 and appendix 'The Six-Axis Profile') -------

def _radar_geometry(profiles: Dict[str, Dict]):
    """Panel order, spoke angles, per-axis min-max normalization, and the
    normalized panel median, shared by radar and radar_quad."""
    import numpy as np
    models = _seed_model_colors(profiles)
    score = lambda m: profiles[m]["pact"].value or 0.0
    panel = sorted(models, key=score, reverse=True)
    N = len(AXES)
    ang = [n / N * 2 * np.pi for n in range(N)] + [0.0]
    sm_models = sorted(panel, key=score, reverse=True)
    amin = {a: min(profiles[m]["axes"][a] or 0.0 for m in sm_models)
            for a in AXES}
    amax = {a: max(profiles[m]["axes"][a] or 0.0 for m in sm_models)
            for a in AXES}

    def _med(a):
        xs = sorted(profiles[m]["axes"][a] or 0.0 for m in sm_models)
        k = len(xs)
        return xs[k // 2] if k % 2 else (xs[k // 2 - 1] + xs[k // 2]) / 2

    def _norm(v, a):
        lo, hi = amin[a], amax[a]
        return (v - lo) / (hi - lo) if hi > lo else 0.5

    medvals = [_norm(_med(a), a) for a in AXES]
    medvals += medvals[:1]
    return score, ang, sm_models, _norm, medvals


_RADAR_ABBR = ["Def", "Prs", "Psh", "Str", "Trn", "Scp"]


def radar(profiles: Dict[str, Dict]) -> None:
    """All-model six-axis small multiples, each axis min-max normalized across
    the panel, with the panel median as a dotted reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FS.use_paper_style()
    score, ang, sm_models, _norm, medvals = _radar_geometry(profiles)
    if not sm_models:
        return
    abbr = _RADAR_ABBR
    ncols = 6
    nrows = -(-len(sm_models) // ncols)
    fig, axarr = plt.subplots(nrows, ncols,
                              figsize=(2.1 * ncols, 2.35 * nrows),
                              subplot_kw=dict(polar=True))
    for idx, m in enumerate(sm_models):
        ax = axarr.flat[idx]
        vals = [_norm(profiles[m]["axes"][a] or 0.0, a) for a in AXES]
        vals += vals[:1]
        col = FS.model_color(m)
        ax.plot(ang, medvals, color=FS.MUTED, lw=1.0, ls=":", zorder=1)
        ax.fill(ang, vals, color=col, alpha=0.25, zorder=2)
        ax.plot(ang, vals, color=col, lw=1.8, zorder=3)
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels(abbr, fontsize=11.5, color=FS.INK)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(f"{FS.short(m)}\n({score(m):.3f})", fontsize=12,
                     color=FS.INK, pad=6)
        ax.tick_params(pad=-3)
        ax.grid(color=FS.GRID, lw=0.7)
        ax.spines["polar"].set_color(FS.GRID)
    for j in range(len(sm_models), nrows * ncols):
        axarr.flat[j].axis("off")
    fig.tight_layout(h_pad=1.6, w_pad=0.8)
    _save(fig, "radar")


def radar_quad(profiles: Dict[str, Dict]) -> None:
    """Main-text strip: four contrasting six-axis profiles in one row, same
    normalization and median reference as `radar`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FS.use_paper_style()
    score, ang, sm_models, _norm, medvals = _radar_geometry(profiles)
    abbr = _RADAR_ABBR
    quad_short = ["Kimi-K2.7-Code", "Claude Haiku 4.5", "GPT-5.6 Luna",
                  "DeepSeek-V4-Pro"]
    quad = [m for s in quad_short for m in sm_models if FS.short(m) == s]
    if len(quad) != 4:
        return
    fig, axarr = plt.subplots(1, 4, figsize=(7.0, 2.35),
                              subplot_kw=dict(polar=True))
    for ax, m in zip(axarr, quad):
        vals = [_norm(profiles[m]["axes"][a] or 0.0, a) for a in AXES]
        vals += vals[:1]
        col = FS.model_color(m)
        ax.plot(ang, medvals, color=FS.MUTED, lw=0.9, ls=":", zorder=1)
        ax.fill(ang, vals, color=col, alpha=0.25, zorder=2)
        ax.plot(ang, vals, color=col, lw=1.8, zorder=3)
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels(abbr, fontsize=13, color=FS.INK)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(FS.short(m), fontsize=12.5, color=FS.INK, pad=17)
        ax.tick_params(pad=2)
        ax.grid(color=FS.GRID, lw=0.6)
        ax.spines["polar"].set_color(FS.GRID)
    fig.tight_layout(w_pad=1.0)
    _save(fig, "radar_quad")


def axis_correlation(profiles: Dict[str, Dict]) -> None:
    """Lower-triangle inter-axis Pearson correlation heatmap across the panel
    (appendix 'The Six-Axis Profile')."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    FS.use_paper_style()
    models = [m for m in profiles if m != "_meta"]
    score = lambda m: profiles[m]["pact"].value or 0.0
    panel = sorted(models, key=score, reverse=True)
    if len(panel) < 3:
        return
    P = np.array([[profiles[m]["axes"][a] for a in AXES] for m in panel],
                 dtype=float)
    keep = [j for j in range(len(AXES)) if not np.isnan(P[:, j]).any()]
    if len(keep) < 2:
        return
    C = np.corrcoef(P[:, keep], rowvar=False)
    labs = [FS.AXIS_LABEL[AXES[j]] for j in keep]
    Mmask = np.ma.masked_where(np.triu(np.ones_like(C), k=1) > 0, C)
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    cmap = plt.get_cmap(FS.CORR_CMAP).copy()
    cmap.set_bad("white")
    ax.imshow(Mmask, cmap=cmap, vmin=0, vmax=1)
    shortlab = ["Default", "Pressure", "Pushback", "Steer.",
                "Transp.", "Scope"]
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels([shortlab[j] for j in keep], fontsize=12.5,
                       rotation=22, ha="right")
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels(labs, fontsize=12.5)
    for i in range(len(keep)):
        for j in range(i + 1):
            v = C[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=14.5,
                    color="white" if v > 0.65 else FS.INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    _save(fig, "axis_correlation")


# -- scope_default (Section 4) -------------------------------------------------

def scope_default(da: Dict[str, Dict[str, Optional[float]]]) -> None:
    """Per-domain Default Compliance vs Rule-Scope Discernment scatter with the
    y = x parity line. `da` comes from distributions.domain_axis_table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FS.use_paper_style()
    doms = [d for d in da if da[d]["default_compliance"] is not None
            and da[d]["rule_scope_discernment"] is not None]
    x = [da[d]["default_compliance"] for d in doms]
    y = [da[d]["rule_scope_discernment"] for d in doms]
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.plot([0.70, 1.005], [0.70, 1.005], ls="--", lw=1.4, color=FS.MUTED,
            zorder=2)
    ax.scatter(x, y, s=70, color=FS.BLUE, zorder=3)
    ax.set_xlim(0.72, 1.10)
    ax.set_ylim(0.70, 1.005)
    nudge = {"Procurement": (8, -6), "Healthcare": (8, -4),
             "HR/Hiring": (-9, 0), "Advertising": (-9, 5),
             "Finance": (-9, 8), "AML": (-9, -3), "Privacy": (8, -3),
             "Customer Service": (-9, 2), "Moderation": (-9, 4),
             "Export Controls": (7, 9), "Pharma": (7, -2),
             "Gov. Services": (7, 2)}
    for d, xi, yi in zip(doms, x, y):
        s = FS.domain_label(d)
        dx, dy = nudge.get(s, (8, 4))
        ax.annotate(s, (xi, yi), fontsize=11, xytext=(dx, dy),
                    textcoords="offset points",
                    ha="right" if dx < 0 else "left", va="center", color=FS.INK)
    ax.set_xlabel("Default Compliance", fontsize=13.5)
    ax.set_ylabel("Rule-Scope Discernment", fontsize=13.5)
    ax.set_xticks([0.8, 0.9, 1.0])
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.tick_params(labelsize=12)
    ax.grid(True, color=FS.GRID, lw=0.8)
    ax.set_axisbelow(True)
    FS.strip_axes(ax)
    fig.tight_layout()
    _save(fig, "scope_default")


# -- dataset_by_domain_group (appendix 'Corpus Generation') ---------------------

def dataset_by_domain_group(items_path: str = paths.ITEMS) -> None:
    """Frozen-set composition: scenario cells per domain, stacked by cell group."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    FS.use_paper_style()

    groups = ["neutral", "pressure", "guard_nonbinding",
              "attack_binding", "attack_nonbinding"]
    colors = dict(zip(groups, [FS.MUTED, FS.BLUE, FS.GREEN, FS.RED, FS.GOLD]))
    per: Dict[str, Counter] = defaultdict(Counter)
    with open(items_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            per[d["domain"]][d["group"]] += 1
    label = {"neutral": "Neutral", "pressure": "Pressure",
             "guard_nonbinding": "Guard", "attack_binding": "Attack (binding)",
             "attack_nonbinding": "Attack (non-binding)"}
    domains = sorted(per)
    x = np.arange(len(domains))
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    bottom = np.zeros(len(domains))
    for g in groups:
        vals = np.array([per[d][g] for d in domains], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=colors[g], label=label[g],
               width=0.74)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([FS.domain_label(d) for d in domains], rotation=32,
                       ha="right", fontsize=11.5)
    ax.set_ylabel("Scenario Cells", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(ncol=3, fontsize=10.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=1.2,
              handlelength=1.2, handletextpad=0.5)
    ax.grid(axis="y", color=FS.GRID, lw=0.8)
    ax.set_axisbelow(True)
    FS.strip_axes(ax)
    fig.tight_layout()
    _save(fig, "dataset_by_domain_group", dpi=150)


# -- taxonomy figures (appendix; transparency and abstention-reason splits) ----

def _stacked_by_model(per_judge, tmodel, labels, colors, sort_key, path):
    """Per-model 100%-stacked horizontal bar of judge vote-share. Rendered
    under default rcParams (all sizes are set explicitly below)."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcdefaults()
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    import numpy as np
    per: Dict[str, Counter] = defaultdict(Counter)
    for utid, labs in per_judge.items():
        m = tmodel.get(utid)
        if m and labs:
            share = 1.0 / len(labs)   # each trial contributes 1.0, split by its votes
            for lab in labs.values():
                per[m][lab] += share
    if not per:
        print(f"  (no data for {path})")
        return
    models = sorted(per, key=lambda m: sort_key(per[m]))
    disp = [FS.short(m) for m in models]
    y = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(6.6, 0.31 * len(models) + 1.3))
    left = np.zeros(len(models))
    for lab in labels:
        vals = np.array([100 * per[m][lab] / (sum(per[m].values()) or 1) for m in models])
        ax.barh(y, vals, left=left, color=colors[lab],
                label=_LABEL_DISPLAY.get(lab, lab), height=0.74)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(disp, fontsize=11.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Judge Vote Share (%)", fontsize=13)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", length=0)
    ax.legend(ncol=min(len(labels), 4), fontsize=10.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=1.1,
              handlelength=1.2, handletextpad=0.5)
    ax.invert_yaxis()
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path} (+pdf)")


def transparency_taxonomy(transparency_path: str = paths.TRANSPARENCY,
                          trials_dir: str = paths.TRIALS_DIR) -> None:
    """Per-model 3-way transparency vote-share on binding violations, sorted by
    transparent share."""
    from analysis.agreement import TRIO_TAGS, read_ensemble
    from evaluation.judges import (TRANSPARENCY_COLORS, TRANSPARENCY_LABELS,
                                   load_trials, parse_transparency)
    trio = set(TRIO_TAGS)
    per = read_ensemble(transparency_path, parse_transparency)
    tpj = {t: {j: l for j, l in labs.items() if j in trio}
           for t, labs in per.items()}
    tmodel = {f"{t['model']}||{t['trial_id']}": t["model"]
              for t in load_trials(trials_dir)}
    os.makedirs(paths.FIGURES, exist_ok=True)
    _stacked_by_model(
        tpj, tmodel, TRANSPARENCY_LABELS, TRANSPARENCY_COLORS,
        sort_key=lambda c: -c.get("TRANSPARENT", 0) / (sum(c.values()) or 1),
        path=os.path.join(paths.FIGURES, "transparency_taxonomy.png"))


def unclear_taxonomy(unclear_path: str = paths.UNCLEAR,
                     trials_dir: str = paths.TRIALS_DIR) -> None:
    """Per-model 4-way abstention-reason vote-share on still-unclear turns
    (T1 + T2), sorted by refused-on-policy share."""
    from evaluation.judges import (UNCLEAR_COLORS, UNCLEAR_LABELS,
                                   aggregate_unclear, load_trials)
    _, upj = aggregate_unclear(unclear_path)
    trials = load_trials(trials_dir)
    tmodel = {f"{t['model']}||{t['trial_id']}": t["model"] for t in trials}
    # keep only turns whose FINAL outcome is unclear; T2 turns are keyed ||t2
    final_unclear = {f"{t['model']}||{t['trial_id']}" for t in trials
                     if t.get("t1_outcome") == "unclear"}
    final_unclear |= {f"{t['model']}||{t['trial_id']}||t2" for t in trials
                      if t.get("t2_outcome") == "unclear"
                      and t.get("t2_script") == "pushback"}
    tmodel.update({f"{t['model']}||{t['trial_id']}||t2": t["model"]
                   for t in trials})
    upj = {u: v for u, v in upj.items() if u in final_unclear}
    os.makedirs(paths.FIGURES, exist_ok=True)
    _stacked_by_model(
        upj, tmodel, UNCLEAR_LABELS, UNCLEAR_COLORS,
        sort_key=lambda c: -c.get("REFUSED_ON_POLICY", 0) / (sum(c.values()) or 1),
        path=os.path.join(paths.FIGURES, "unclear_taxonomy.png"))


# -- eval_awareness (appendix 'Evaluation-Awareness Realism Probe') -------------

def eval_awareness(summary_path: str = paths.AWARENESS_SUMMARY) -> None:
    """Per-model realism-ladder effects as paired diverging bars: the share of
    native non-compliance each altered form removes, (C - C0) / (1 - C0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from evaluation.batch import MODEL_REGISTRY
    FS.use_paper_style()

    with open(summary_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    recs = []
    for r in rows:
        mid = MODEL_REGISTRY.get(r["model"], r["model"])
        c0, c1, c2 = (float(r["C0"]), float(r["C1"]), float(r["C2"]))
        recs.append(dict(name=FS.short(mid),
                         e1=100 * (c1 - c0) / (1 - c0),
                         e2=100 * (c2 - c0) / (1 - c0)))
    recs.sort(key=lambda d: d["e2"])

    y = np.arange(len(recs))
    h = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 0.44 * len(recs) + 1.2))
    ax.barh(y + h / 2, [d["e2"] for d in recs], height=h, color=FS.BLUE,
            label="Announced as a Test", zorder=3)
    ax.barh(y - h / 2, [d["e1"] for d in recs], height=h, color="#9ca3af",
            label="Realism Stripped", zorder=3)
    ax.axvline(0, color=FS.INK, lw=1.0, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([d["name"] for d in recs], fontsize=12)
    ax.set_xlabel("Native Non-Compliance Removed (%)", fontsize=13)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", length=0)
    ax.legend(fontsize=11, loc="lower right", frameon=False,
              handletextpad=0.5, handlelength=1.4)
    ax.grid(axis="x", color=FS.GRID, lw=0.8)
    ax.set_axisbelow(True)
    FS.strip_axes(ax)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    _save(fig, "eval_awareness", dpi=150)


# -- convergence_by_component (appendix 'Corpus Generation') --------------------

def _load_guard_log(path: str) -> List[dict]:
    rows: List[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def convergence_by_component(log_path: str = paths.GUARD_LOG) -> None:
    """Cumulative guard acceptance by authoring attempt, per component group.
    Components never accepted stay in the denominator."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FS.use_paper_style()

    rows = _load_guard_log(log_path)
    _CGROUP = {"persona": "Persona", "task": "Task & menu",
               "t2": "Turn-2 script", "rules": "Rule note",
               "guard_nonbinding": "Non-binding guard", "attacks": "Attack"}

    def _cgroup(comp: str) -> str:
        return "Pressure" if comp.startswith("pressure.") \
            else _CGROUP.get(comp, comp)

    first_acc: Dict[tuple, Optional[int]] = {}
    for r in rows:
        key = (r["scenario_id"], r["generator_model"], r["component"],
               r.get("pressure_key"))
        a = int(r["attempt"])
        if key not in first_acc:
            first_acc[key] = None
        if r["accepted"] and (first_acc[key] is None or a < first_acc[key]):
            first_acc[key] = a
    groups: Dict[str, List[Optional[int]]] = {}
    for key, a in first_acc.items():
        groups.setdefault(_cgroup(key[2]), []).append(a)
    order = ["Rule note", "Persona", "Task & menu", "Turn-2 script",
             "Pressure", "Attack", "Non-binding guard"]
    order = [g for g in order if g in groups]
    colors = dict(zip(order, [FS.GREEN, FS.BLUE, FS.TEAL, FS.PINK,
                              FS.PURPLE, FS.ORANGE, FS.GOLD]))
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    xmax = 6
    for g in order:
        vals = groups[g]
        n = len(vals)
        xs = list(range(1, xmax + 1))
        ys = [100 * sum(1 for a in vals if a is not None and a + 1 <= k) / n
              for k in xs]
        ax.plot(xs, ys, marker="o", ms=5, lw=2.2, color=colors[g],
                solid_capstyle="round", label=g)
    ax.legend(loc="lower right", frameon=False, fontsize=10.5,
              handlelength=1.6, handletextpad=0.5, labelspacing=0.35)
    ax.set_xlim(0.8, xmax + 0.2)
    ax.set_ylim(0, 104)
    ax.set_xticks(range(1, xmax + 1))
    ax.set_xlabel("Authoring Attempts", fontsize=13)
    ax.set_ylabel("Components Accepted (%)", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    os.makedirs(paths.FIGURES, exist_ok=True)
    fig.savefig(os.path.join(paths.FIGURES, "convergence_by_component.png"),
                dpi=200)
    fig.savefig(os.path.join(paths.FIGURES, "convergence_by_component.pdf"))
    plt.close(fig)
    print(f"  wrote {os.path.join(paths.FIGURES, 'convergence_by_component.png')} (+pdf)")


def main() -> None:
    from analysis import distributions as D
    from analysis import metrics as M
    from analysis.aggregate import profile_all
    from evaluation.judges import load_trials, load_transparency_votes

    trials = load_trials(paths.TRIALS_DIR)
    trans_votes = load_transparency_votes(paths.TRANSPARENCY)
    profiles = profile_all(trials, trans_votes, fast=True)
    radar(profiles)
    radar_quad(profiles)
    axis_correlation(profiles)

    cells = M.build_cells(trials)
    models = sorted({m for (m, _, _) in cells})
    scope_default(D.domain_axis_table(cells, models))

    dataset_by_domain_group()
    transparency_taxonomy()
    unclear_taxonomy()
    eval_awareness()
    convergence_by_component()


if __name__ == "__main__":
    main()
