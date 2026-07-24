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
    items.sort(key=lambda kv: (not kv[1]["open"], kv[1]["rel"]))
    lines = [
        "% AUTO-GENERATED by src/benchmark/make_paper_assets.py - do not edit by hand.",
        r"\begin{tabular}{@{}l l l l@{}}",
        r"\toprule",
        (r"\textbf{Model} & \textbf{Size} & \textbf{Rel.} & "
         r"\textbf{Provider; training note} \\"),
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
        lines.append(f"{name} & {meta['params']} & {meta['rel']} & "
                     f"{prov}; {note}~\\citep{{{meta['cite']}}} \\\\")
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
    tag = "C" if FS.is_closed(model) else "O"
    # LaTeX-escape the few names that need it (none currently, but be safe)
    name = name.replace("&", r"\&").replace("_", r"\_")
    return f"{name} ({tag})"


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
        (r"\textbf{Model} & \textbf{Default} & \textbf{Pressure} & "
         r"\textbf{Pushback} & \textbf{Steer.} & \textbf{Honesty} & "
         r"\textbf{Scope} & \textbf{Rollup} \\"),
        r"\midrule",
    ]
    for r in rows:
        cells = [_colcell(fnum(r, a), lo[a], hi[a], ".2f") for a, _ in AXES]
        cells.append(_colcell(fnum(r, "rollup_p3"), lo["rollup_p3"],
                              hi["rollup_p3"], ".3f", bold=True))
        lines.append(f"{display(r['model'])} & " + " & ".join(cells) + r" \\")
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
    ]
    return "\n".join(out)


def copy_figures() -> List[str]:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    copied = []
    for name in PAPER_FIGURES:
        src = os.path.join(FIG_SRC, name)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(FIGURES_DIR, name))
            copied.append(name)
        else:
            print(f"  [!] missing figure {src} - run `aggregate --figures` first")
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
    copied = copy_figures()
    print(f"copied {len(copied)} figures -> {FIGURES_DIR}")
    print()
    print(field_summary(rows))


if __name__ == "__main__":
    main()
