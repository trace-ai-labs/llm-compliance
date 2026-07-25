"""Generate the paper's data-derived assets from the frozen metrics.

Single source of truth for everything in the paper that must track the numbers:

  1. paper/tables/leaderboard.tex  - the 22-model x 6-axis + rollup leaderboard
     `tabular`, sorted by the pass^3 rollup, emitted from results/benchmark/
     metrics_v2.csv. results.tex does `\\input{tables/leaderboard}`.
  2. paper/figures/*.png           - the subset of results/benchmark/figures the
     paper `\\includegraphics`es, copied so the paper tree is self-contained.

It also prints the field-summary statistics the prose quotes (best/median
rollup, how many models clear a given bar, the steerability range), so those
sentences can be checked against the live data rather than hand-maintained.

Run after `aggregate` (which writes metrics_v2.csv and the figures):
  python -m src.benchmark.make_paper_assets
"""

import csv
import os
import shutil
from statistics import median
from typing import Dict, List

from src.benchmark import figstyle as FS
from src.benchmark import registry as REG
from src.benchmark.distributions import short_domain

TRIVIAL_CSV = os.path.join("results", "benchmark", "trivial_v2.csv")
DIST_DOMAIN_CSV = os.path.join("results", "benchmark", "dist_domain_axis.csv")
DIST_PRESSURE_CSV = os.path.join("results", "benchmark", "dist_pressure.csv")


def _tex_escape(s: str) -> str:
    """Escape the LaTeX specials that appear in seed rule text ($ amounts, %, &, _, #)."""
    for a, b in (("\\", r"\textbackslash "), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("~", r"\textasciitilde "),
                 ("^", r"\textasciicircum ")):
        s = s.replace(a, b)
    return s

METRICS_CSV = os.path.join("results", "benchmark", "metrics_v2.csv")
FIG_SRC = os.path.join("results", "benchmark", "figures")
PAPER = "paper"
TABLES_DIR = os.path.join(PAPER, "tables")
FIGURES_DIR = os.path.join(PAPER, "figures")

# axis column -> header label in the leaderboard tabular
AXES = [
    ("default_compliance", "Default"),
    ("pressure_resistance", "Pressure"),
    ("pushback_resistance", "Pushback"),
    ("steerability", "Steer."),
    ("reasoning_honesty", "Honesty"),
    ("rule_scope_discernment", "Scope"),
]

# figures the paper includes, copied into paper/figures/ (name kept identical).
# The six-axis profile is a color-scaled LaTeX table (tables/leaderboard.tex),
# not a figure, so axes_heatmap.png is intentionally not shipped to the paper.
PAPER_FIGURES = [
    "radar_overlay.png",     # results: curated 6-model contrast profile
    "per_pressure.png",      # results: fragility, which lever breaks which model
    "axis_correlation.png",  # methodology: the axes group into three factors
    "radar.png",             # appendix: all-22 small-multiples profile
    "per_domain.png",        # appendix: per-domain base compliance
    "honesty_taxonomy.png",  # appendix: silent / rationalized / defiant split
    "unclear_taxonomy.png",  # appendix: 6-way reason-for-abstention split
    "domain_pressure_heatmap.png",  # appendix: t1 compliance, domain x pressure
]


# Model roster metadata, verified July 2026 against official model cards /
# technical reports / blogs (see aaai2027.bib for the matching citations).
# params: total, plus active for MoE ("1T/32B"); closed models = "--".
# rel: YYYY-MM release. open: True = open-weights. cite: bibkey in aaai2027.bib.
# note: <=~9 words, provider-agnostic (provider is its own column).
MODEL_META = {
    # ---- open-weights ----
    "mistral-7b-instruct": dict(
        provider="Mistral AI", params="7B", rel="2023-09", open=True,
        note="dense; GQA + sliding-window attention", cite="mistral7b"),
    "llama-3.1-8b-instruct": dict(
        provider="Meta", params="8B", rel="2024-07", open=True,
        note="dense; SFT + RLHF, 128K context", cite="llama3herd"),
    "llama-3.3-70b-instruct": dict(
        provider="Meta", params="70B", rel="2024-12", open=True,
        note="dense; SFT + RLHF, multilingual", cite="llama33"),
    "Seed-OSS-36B-Instruct": dict(
        provider="ByteDance Seed", params="36B", rel="2025-08", open=True,
        note="dense; 12T-token pretrain, thinking budget", cite="seedoss"),
    "openai/gpt-oss-120b": dict(
        provider="OpenAI", params="117B/5B", rel="2025-08", open=True,
        note="MoE reasoning model; RL post-training", cite="gptoss"),
    "zai-org/GLM-4.7": dict(
        provider="z.ai (Zhipu)", params="357B/32B", rel="2025-12", open=True,
        note="agentic-coding MoE; interleaved thinking", cite="glm47"),
    "zai-org/GLM-5": dict(
        provider="z.ai (Zhipu)", params="744B/40B", rel="2026-02", open=True,
        note="MoE; sparse attention, async-RL", cite="glm5"),
    "MiniMaxAI/MiniMax-M2.5": dict(
        provider="MiniMax", params="230B/10B", rel="2026-02", open=True,
        note="agentic MoE; RL over 200k+ environments", cite="minimax25"),
    "qwen3.5-35b-a3b": dict(
        provider="Alibaba Qwen", params="35B/3B", rel="2026-02", open=True,
        note="multimodal MoE; early-fusion training", cite="qwen35"),
    "gemma-4-26b": dict(
        provider="Google DeepMind", params="25B/4B", rel="2026-03", open=True,
        note="multimodal MoE; hybrid local/global attn.", cite="gemma4"),
    "nvidia/Nemotron-120B-A12B": dict(
        provider="NVIDIA", params="120B/12B", rel="2026-03", open=True,
        note="hybrid Mamba-2 + MoE; multi-token pred.", cite="nemotron3"),
    "Qwen3.6 27B": dict(
        provider="Alibaba Qwen", params="27B", rel="2026-04", open=True,
        note="dense; Gated-DeltaNet + attention hybrid", cite="qwen36"),
    "moonshotai/Kimi-K2.6": dict(
        provider="Moonshot AI", params="1T/32B", rel="2026-04", open=True,
        note="multimodal agentic MoE; agent-swarm RL", cite="kimik26"),
    "deepseek-ai/DeepSeek-V4-Pro": dict(
        provider="DeepSeek", params="1.6T/49B", rel="2026-04", open=True,
        note="MoE; compressed sparse attention, think modes", cite="deepseekv4"),
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B": dict(
        provider="NVIDIA", params="550B/55B", rel="2026-06", open=True,
        note="hybrid Mamba-2 + MoE; 1M context", cite="nemotron3"),
    "moonshotai/Kimi-K2.7-Code": dict(
        provider="Moonshot AI", params="1T/32B", rel="2026-06", open=True,
        note="K2.6 coding variant; fewer reasoning tokens", cite="kimik27"),
    "zai-org/GLM-5.2": dict(
        provider="z.ai (Zhipu)", params="750B/40B", rel="2026-06", open=True,
        note="MoE; IndexShare sparse attention, 1M ctx", cite="glm52"),
    "thinkingmachines/inkling": dict(
        provider="Thinking Machines", params="975B/41B", rel="2026-07", open=True,
        note="MoE (6/256 experts); hybrid attn., multimodal", cite="inkling"),
    # ---- closed / API-only (no public parameter counts) ----
    "claude-haiku-4-5": dict(
        provider="Anthropic", params="--", rel="2025-10", open=False,
        note="fast tier; extended thinking, computer use", cite="claudehaiku45"),
    "google/gemini-3-flash-preview": dict(
        provider="Google DeepMind", params="--", rel="2025-12", open=False,
        note="multimodal reasoning; configurable thinking", cite="gemini3flash"),
    "x-ai/grok-4.3": dict(
        provider="xAI", params="--", rel="2026-05", open=False,
        note="reasoning model; variable effort, tool use", cite="grok43"),
    "gpt-5.6-luna": dict(
        provider="OpenAI", params="--", rel="2026-07", open=False,
        note="fast, low-cost tier of the GPT-5.6 family", cite="gpt56"),
}


def build_model_table(rows: List[dict]) -> str:
    """Single-column roster of the evaluated models: provider, size, release,
    a one-line note, and a citation. Open-weights first then closed, each block
    chronological by release date. Data from MODEL_META (verified model cards)."""
    present = {r["model"] for r in rows}
    missing = present - set(MODEL_META)
    if missing:
        print(f"  [!] no MODEL_META for: {sorted(missing)}")
    items = [(m, MODEL_META[m]) for m in present if m in MODEL_META]
    # model id is the final, deterministic tiebreak: `present` is a set, so without
    # it same-release rows order by set-hash and churn spuriously between runs.
    items.sort(key=lambda kv: (not kv[1]["open"], kv[1]["rel"], kv[0]))
    lines = [
        "% AUTO-GENERATED by src/benchmark/make_paper_assets.py - do not edit by hand.",
        # Distinct print-legible inks so the two facts packed into the last column
        # read apart at a glance: provider in blue, training method in violet.
        r"\definecolor{providerink}{HTML}{1F5C8B}",
        r"\definecolor{trainink}{HTML}{8A3FA0}",
        r"\begin{tabular}{@{}l l l l@{}}",
        r"\toprule",
        (r"\textbf{Model} & \textbf{Size} & \textbf{Rel.} & "
         r"\textbf{\textcolor{providerink}{Provider}}; "
         r"\textbf{\textcolor{trainink}{training method}} \\"),
        r"\midrule",
        r"\multicolumn{4}{@{}l}{\emph{Open-weights}} \\",
    ]
    seen_closed = False
    for m, meta in items:
        if not meta["open"] and not seen_closed:
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{4}{@{}l}{\emph{Closed / API-only "
                         r"(no published parameter counts)}} \\")
            seen_closed = True
        name = FS.DISPLAY_NAME.get(m, m.split("/")[-1]).replace("&", r"\&")
        prov = meta["provider"].replace("&", r"\&")
        note = meta["note"].replace("&", r"\&")
        lines.append(
            f"\\texttt{{{name}}} & {meta['params']} & {meta['rel']} & "
            f"\\textcolor{{providerink}}{{{prov}}}; "
            f"\\textcolor{{trainink}}{{{note}}}~\\citep{{{meta['cite']}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def load_rows(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(row: dict, key: str) -> float:
    v = row.get(key, "")
    return float(v) if v not in ("", None, "—") else float("nan")


def display(model: str) -> str:
    name = FS.DISPLAY_NAME.get(model, model.split("/")[-1])
    # LaTeX-escape the few names that need it (none currently, but be safe)
    name = name.replace("&", r"\&").replace("_", r"\_")
    cell = f"\\texttt{{{name}}}"  # closed-source models are underlined in the leaderboard
    return f"\\underline{{{cell}}}" if FS.is_closed(model) else cell


def _rdylgn(t: float) -> str:
    """t in [0,1] -> HTML hex on a red-yellow-green ramp (0 red, .5 amber, 1
    green). Lightened toward white so black cell text stays legible."""
    t = max(0.0, min(1.0, t))
    stops = [(0.0, (222, 132, 138)), (0.5, (240, 214, 130)),
             (1.0, (140, 200, 150))]
    for i in range(len(stops) - 1):
        a, ca = stops[i]
        b, cb = stops[i + 1]
        if t <= b:
            f = (t - a) / (b - a) if b > a else 0.0
            rgb = [round(ca[j] + (cb[j] - ca[j]) * f) for j in range(3)]
            return "".join(f"{c:02X}" for c in rgb)
    return "8CC896"


def _colcell(v: float, col_min: float, col_max: float, fmt: str,
             bold: bool = False) -> str:
    """A \\cellcolor cell, color scaled to THIS column's own min..max range."""
    t = (v - col_min) / (col_max - col_min) if col_max > col_min else 0.5
    body = f"\\textbf{{{v:{fmt}}}}" if bold else f"{v:{fmt}}"
    return f"\\cellcolor[HTML]{{{_rdylgn(t)}}} {body}"


def build_table(rows: List[dict]) -> str:
    rows = sorted(rows, key=lambda r: fnum(r, "rollup_p3"), reverse=True)
    cols = [a for a, _ in AXES] + ["rollup_p3"]
    lo = {c: min(fnum(r, c) for r in rows) for c in cols}
    hi = {c: max(fnum(r, c) for r in rows) for c in cols}
    lines = [
        "% AUTO-GENERATED by src/benchmark/make_paper_assets.py - do not edit by hand.",
        "% Regenerate after each `aggregate` run; source: results/benchmark/metrics_v2.csv.",
        "% Cell color is scaled WITHIN each column (per-column min..max), so it shows",
        "% relative standing on that axis; color is NOT comparable across columns.",
        r"\begin{tabular}{l cccccc c}",
        r"\toprule",
        (r"\textbf{Model} & \shortstack{\textbf{Default}\\\textbf{Compliance}} & "
         r"\shortstack{\textbf{Pressure}\\\textbf{Resistance}} & "
         r"\shortstack{\textbf{Pushback}\\\textbf{Resistance}} & \textbf{Steerability} & "
         r"\shortstack{\textbf{Reasoning}\\\textbf{Honesty}} & "
         r"\shortstack{\textbf{Rule-Scope}\\\textbf{Discernment}} & \textbf{Rollup} \\"),
        r"\midrule",
    ]
    for r in rows:
        cells = [_colcell(fnum(r, a), lo[a], hi[a], ".2f") for a, _ in AXES]
        cells.append(_colcell(fnum(r, "rollup_p3"), lo["rollup_p3"],
                              hi["rollup_p3"], ".3f", bold=True))
        lines.append(f"{display(r['model'])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def build_abstention_table(rows: List[dict]) -> str:
    """Per-model turn-1 abstention (unclear) rate, from the `abstention` column of
    metrics_v2.csv. Two model/rate pairs per row to keep the table compact.
    Abstention is a diagnostic, not an axis: `unclear` replies are dropped from
    every axis denominator (they never count as pass or fail), so this rate is the
    guard on how much mass each model removes by declining to commit."""
    pairs = sorted(((display(r["model"]), fnum(r, "abstention"))
                    for r in rows if fnum(r, "abstention") == fnum(r, "abstention")),
                   key=lambda t: (-t[1], t[0]))   # name tiebreak = deterministic order
    lines = [
        "% AUTO-GENERATED by src/benchmark/make_paper_assets.py - do not edit by hand.",
        "% Regenerate after each `aggregate` run; source: results/benchmark/metrics_v2.csv.",
        r"\begin{tabular}{l r @{\hskip 2.2em} l r}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Abstain} & \textbf{Model} & \textbf{Abstain} \\",
        r"\midrule",
    ]
    half = (len(pairs) + 1) // 2
    left, right = pairs[:half], pairs[half:]
    for i in range(half):
        lm, lv = left[i]
        lcell = f"{lm} & {lv * 100:.1f}\\%"
        if i < len(right):
            rm, rv = right[i]
            rcell = f"{rm} & {rv * 100:.1f}\\%"
        else:
            rcell = " & "
        lines.append(f"{lcell} & {rcell} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def build_rules_table() -> str:
    """The verbatim seed rule for all 48 scenarios, straight from registry.py (the
    committed source of truth the generators were conditioned on). Rendered as two
    full-width table* floats because a 48-row table overflows a single float and
    longtable is unsupported in AAAI's twocolumn layout."""
    scen = sorted(REG.SCENARIOS, key=lambda s: (s.domain, s.letter))
    half = (len(scen) + 1) // 2
    parts = [scen[:half], scen[half:]]
    out = ["% AUTO-GENERATED by src/benchmark/make_paper_assets.py - do not edit by hand.",
           "% Source: src/benchmark/registry.py SCENARIOS (the frozen seed rules)."]
    for i, chunk in enumerate(parts):
        of = f"{i + 1} of {len(parts)}"
        out += [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular}{@{}l l p{11.4cm}@{}}",
            r"\toprule",
            r"\textbf{Domain} & \textbf{ID} & \textbf{Rule (verbatim seed text)} \\",
            r"\midrule",
        ]
        for s in chunk:
            dom = _tex_escape(REG.DOMAIN_BY_KEY[s.domain].title)
            out.append(f"{dom} & \\texttt{{{_tex_escape(s.id)}}} & {_tex_escape(s.rule)} \\\\")
        out += [
            r"\bottomrule",
            r"\end{tabular}",
            (r"\caption{The exact standing rule for every scenario (" + of +
             r"), reproduced verbatim from \texttt{registry.py}. This is the frozen "
             r"seed text the generators were conditioned on; the item rendering appends "
             r"it to the persona as a plain informational note, never as a command.}"),
            (r"\label{tab:rules}" if i == 0 else r"\label{tab:rules2}"),
            r"\end{table*}",
            "",
        ]
    return "\n".join(out)


def build_trivial_table(path: str = TRIVIAL_CSV) -> str:
    """Gameability floor: the four trivial agents scored on every axis. If any
    out-ranks a real model on an axis, that axis is broken. Reads trivial_v2.csv
    (written by aggregate); honesty is omitted because the undefined-maps-to-1 rule
    makes it gameable by a non-violating agent by design, so it is not a fair floor."""
    if not os.path.exists(path):
        return ("% trivial_v2.csv missing - run `aggregate` to regenerate it.\n")
    rows = load_rows(path)
    axes = [(a, lbl) for a, lbl in AXES if a != "reasoning_honesty"]
    lines = [
        "% AUTO-GENERATED by src/benchmark/make_paper_assets.py - do not edit by hand.",
        "% Source: results/benchmark/trivial_v2.csv (aggregate's gameability agents).",
        r"\begin{tabular}{l ccccc}",
        r"\toprule",
        r"\textbf{Trivial agent} & \textbf{Default} & \textbf{Pressure} & "
        r"\textbf{Pushback} & \textbf{Steer.} & \textbf{Scope} \\",
        r"\midrule",
    ]
    for r in rows:
        name = r["model"].replace("trivial:", "").replace("_", r"\_")
        cells = []
        for a, _ in axes:
            v = fnum(r, a)
            cells.append("n/a" if v != v else f"{v:.2f}")
        lines.append(f"\\texttt{{{name}}} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def build_domain_axis_table(path: str = DIST_DOMAIN_CSV) -> str:
    """domain x 5 axes (panel mean), color-scaled within each column. Source:
    dist_domain_axis.csv (written by distributions.py). Sorted by default
    compliance so the hardest domains are on top."""
    if not os.path.exists(path):
        return "% dist_domain_axis.csv missing - run `python -m src.benchmark.distributions`.\n"
    rows = load_rows(path)
    cols = ["default_compliance", "pressure_resistance", "pushback_resistance",
            "steerability", "rule_scope_discernment"]
    hdr = ["Default", "Pressure", "Pushback", "Steer.", "Scope"]
    rows.sort(key=lambda r: fnum(r, "default_compliance"))
    lo = {c: min(fnum(r, c) for r in rows) for c in cols}
    hi = {c: max(fnum(r, c) for r in rows) for c in cols}
    lines = [
        "% AUTO-GENERATED by src/benchmark/make_paper_assets.py - do not edit by hand.",
        "% Source: results/benchmark/dist_domain_axis.csv (panel mean per domain).",
        r"\begin{tabular}{l ccccc}",
        r"\toprule",
        r"\textbf{Domain} & " + " & ".join(f"\\textbf{{{h}}}" for h in hdr) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        cells = [_colcell(fnum(r, c), lo[c], hi[c], ".2f") for c in cols]
        lines.append(f"{_tex_escape(short_domain(r['domain']))} & "
                     + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def build_pressure_table(path: str = DIST_PRESSURE_CSV) -> str:
    """Pressure family x {turn-1 comply, turn-2 hold, steerability}, panel-pooled.
    Source: dist_pressure.csv. Sorted by turn-1 compliance (most damaging first)."""
    if not os.path.exists(path):
        return "% dist_pressure.csv missing - run `python -m src.benchmark.distributions`.\n"
    rows = load_rows(path)
    rows.sort(key=lambda r: fnum(r, "t1_comply"))
    lines = [
        "% AUTO-GENERATED by src/benchmark/make_paper_assets.py - do not edit by hand.",
        "% Source: results/benchmark/dist_pressure.csv (panel-pooled).",
        r"\begin{tabular}{@{}l r r r@{}}",
        r"\toprule",
        r"\textbf{Pressure family} & \textbf{Turn-1 comply} & \textbf{Turn-2 hold} "
        r"& \textbf{Steerability} \\",
        r"\midrule",
    ]
    for r in rows:
        name = _tex_escape(r["pressure"].replace("_", " "))
        lines.append(f"{name} & {fnum(r, 't1_comply'):.2f} & "
                     f"{fnum(r, 't2_hold'):.2f} & {fnum(r, 'steerability'):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def field_summary(rows: List[dict]) -> str:
    roll = sorted((fnum(r, "rollup_p3") for r in rows), reverse=True)
    steer = [fnum(r, "steerability") for r in rows]
    hon = [(display(r["model"]), fnum(r, "reasoning_honesty")) for r in rows]
    hon_best = max(hon, key=lambda t: t[1])
    n95 = sum(1 for v in roll if v >= 0.95)
    n90 = sum(1 for v in roll if v >= 0.90)
    out = [
        "-- field summary (quote these in prose) --",
        f"models: {len(rows)}",
        f"best rollup:   {roll[0]:.3f}  (1 - best = {1 - roll[0]:.3f} unreliable share)",
        f"median rollup: {median(roll):.3f}",
        f"worst rollup:  {roll[-1]:.3f}",
        f"models >= 0.95 rollup: {n95}",
        f"models >= 0.90 rollup: {n90}",
        f"steerability range: {min(steer):.2f} - {max(steer):.2f}",
        f"highest honesty: {hon_best[0]} at {hon_best[1]:.2f}",
        f"abstention range: {min(a for r in rows for a in [fnum(r, 'abstention')] if a == a) * 100:.1f}% - "
        f"{max(a for r in rows for a in [fnum(r, 'abstention')] if a == a) * 100:.1f}%  "
        f"(median {median(sorted(a for r in rows for a in [fnum(r, 'abstention')] if a == a)) * 100:.1f}%)",
    ]
    return "\n".join(out)


def copy_figures() -> List[str]:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    copied = []
    for name in PAPER_FIGURES:
        # Ship the PDF (vector) when aggregate has produced one, and the PNG too;
        # the paper includes these extensionless and pdflatex prefers the PDF.
        variants = [name[:-4] + ".pdf", name] if name.endswith(".png") else [name]
        found = False
        for cand in variants:
            src = os.path.join(FIG_SRC, cand)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(FIGURES_DIR, cand))
                copied.append(cand)
                found = True
        if not found:
            print(f"  [!] missing figure {name} - run `aggregate --figures` first")
    return copied


def main() -> None:
    if not os.path.exists(METRICS_CSV):
        raise SystemExit(f"no {METRICS_CSV} - run `aggregate` first")
    rows = load_rows(METRICS_CSV)
    os.makedirs(TABLES_DIR, exist_ok=True)
    tbl_path = os.path.join(TABLES_DIR, "leaderboard.tex")
    with open(tbl_path, "w", encoding="utf-8") as f:
        f.write(build_table(rows))
    print(f"wrote {tbl_path}  ({len(rows)} models)")
    mdl_path = os.path.join(TABLES_DIR, "models.tex")
    with open(mdl_path, "w", encoding="utf-8") as f:
        f.write(build_model_table(rows))
    print(f"wrote {mdl_path}")
    abst_path = os.path.join(TABLES_DIR, "abstention.tex")
    with open(abst_path, "w", encoding="utf-8") as f:
        f.write(build_abstention_table(rows))
    print(f"wrote {abst_path}")
    rules_path = os.path.join(TABLES_DIR, "rules.tex")
    with open(rules_path, "w", encoding="utf-8") as f:
        f.write(build_rules_table())
    print(f"wrote {rules_path}  (48 seed rules)")
    triv_path = os.path.join(TABLES_DIR, "trivial.tex")
    with open(triv_path, "w", encoding="utf-8") as f:
        f.write(build_trivial_table())
    print(f"wrote {triv_path}")
    da_path = os.path.join(TABLES_DIR, "domain_axis.tex")
    with open(da_path, "w", encoding="utf-8") as f:
        f.write(build_domain_axis_table())
    print(f"wrote {da_path}")
    pr_path = os.path.join(TABLES_DIR, "pressure.tex")
    with open(pr_path, "w", encoding="utf-8") as f:
        f.write(build_pressure_table())
    print(f"wrote {pr_path}")
    copied = copy_figures()
    print(f"copied {len(copied)} figures -> {FIGURES_DIR}")
    print()
    print(field_summary(rows))


if __name__ == "__main__":
    main()
