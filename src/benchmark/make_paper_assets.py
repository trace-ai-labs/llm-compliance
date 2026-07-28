"""Generate the paper's data-derived assets from the frozen metrics.

Single source of truth for everything in the paper that must track the numbers:

  1. paper/tables/leaderboard.tex  - the 22-model x 6-axis + PACTScore
     leaderboard `tabular`, sorted by PACTScore, emitted from results/benchmark/
     metrics_v2.csv. results.tex does `\\input{tables/leaderboard}`.
  2. paper/tables/pact_macros.tex  - every PACTScore figure the prose quotes.
  3. paper/figures/*.png           - the subset of results/benchmark/figures the
     paper `\\includegraphics`es, copied so the paper tree is self-contained.

It also prints the field-summary statistics the prose quotes (best/median
PACTScore, how many models clear a given bar, the steerability range), so those
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
    ("transparency", "Transparency"),
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
    "unclear_taxonomy.png",  # appendix: 4-way reason-for-abstention split
    "transparency_by_arm.png",  # appendix: the 3-way split, base vs anti-adversarial
    "unclear_by_arm.png",       # appendix: the 4-way split, base vs anti-adversarial
    "eval_awareness.png",       # appendix: realism-ladder deltas (realism_probe figure)
    "agreement_reasoning_unclear.png",  # appendix: unclear judge pair kappas (4-way)
    "guard_attempts.png",            # appendix gendiag: attempts-to-acceptance dist
    "convergence_by_component.png",  # appendix gendiag: pass prob by revision round
    "dataset_by_domain_group.png",   # appendix gendiag: frozen-set composition
    "domain_pressure_heatmap.png",  # appendix: t1 compliance, domain x pressure
    "steer_baseline.png",           # appendix: steerability vs baseline (headroom check)
    "pact_ci.png",                  # appendix: PACTScore 95% CI caterpillar + top cluster
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
        r"\begin{tabular}{@{}l l l l l@{}}",
        r"\toprule",
        (r"\textbf{Model} & \textbf{Provider} & \textbf{Size} & \textbf{Rel.} & "
         r"\textbf{Training method} \\"),
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\emph{Open-weights}} \\",
    ]
    seen_closed = False
    for m, meta in items:
        if not meta["open"] and not seen_closed:
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{5}{@{}l}{\emph{Closed / API-only "
                         r"(no published parameter counts)}} \\")
            seen_closed = True
        name = FS.DISPLAY_NAME.get(m, m.split("/")[-1]).replace("&", r"\&")
        if "instruct" in m.lower():
            name = f"\\textit{{{name}}}"
        prov = meta["provider"].replace("&", r"\&")
        note = meta["note"].replace("&", r"\&")
        lines.append(
            f"{name} & {prov} & {meta['params']} & {meta['rel']} & "
            f"{note}~\\citep{{{meta['cite']}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def load_rows(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(row: dict, key: str) -> float:
    v = row.get(key, "")
    return float(v) if v not in ("", None, "—") else float("nan")


def display(model: str) -> str:
    """Model names set in roman, not \\texttt: they are proper nouns, not code.
    Monospace is reserved for literal strings, of which the paper now has none."""
    name = FS.DISPLAY_NAME.get(model, model.split("/")[-1])
    # LaTeX-escape the few names that need it (none currently, but be safe)
    name = name.replace("&", r"\&").replace("_", r"\_")
    # models served as explicit instruct-variant ids are italicized
    if "instruct" in model.lower():
        name = f"\\textit{{{name}}}"
    # closed-source models are underlined in the leaderboard
    return f"\\underline{{{name}}}" if FS.is_closed(model) else name


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
    rows = sorted(rows, key=lambda r: fnum(r, "pact_score"), reverse=True)
    cols = [a for a, _ in AXES] + ["pact_score"]
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
         r"\textbf{Transparency} & "
         r"\shortstack{\textbf{Rule-Scope}\\\textbf{Discernment}} & "
         r"\textbf{PACTScore} \\"),
        r"\midrule",
    ]
    for r in rows:
        # three decimals throughout: at n=1737 items the axes resolve well below
        # 0.01, so two decimals was hiding real differences between adjacent models.
        cells = [_colcell(fnum(r, a), lo[a], hi[a], ".3f") for a, _ in AXES]
        cells.append(_colcell(fnum(r, "pact_score"), lo["pact_score"],
                              hi["pact_score"], ".3f", bold=True))
        # Intervals are deliberately NOT a table column. A couple are inlined in the
        # prose to show they exist; the full set and the pairwise tests live in the
        # statistics appendix.
        lines.append(f"{display(r['model'])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


CONTRASTS_CSV = os.path.join("results", "benchmark", "contrasts_v2.csv")
ITEMS_JSONL = os.path.join("results", "benchmark", "items_v1_corrected.jsonl")


def _tex_int(n: int) -> str:
    """Thousands separator LaTeX will not treat as a math comma."""
    return f"{n:,}".replace(",", "{,}")


def item_universe(path: str = ITEMS_JSONL) -> tuple:
    """(items in the frozen set, of those how many carry a second turn). Each
    scenario cell is scored in two system-prompt modes (base / anti_adversarial),
    and the two modes are counted as separate items, so both counts are the cell
    count doubled. Read from the item file rather than any model's denominator:
    per-model counts differ once all-unclear items drop out."""
    import json
    cells = multi_cells = 0
    if not os.path.exists(path):
        return 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            cells += 1
            if d.get("t2_if_compliant"):
                multi_cells += 1
    return cells * 2, multi_cells * 2


def pact_significance(ranked: List[dict], path: str = CONTRASTS_CSV) -> Dict:
    """How many of the pairwise PACTScore contrasts survive BH correction, and
    how many models sit in the top indistinguishable cluster (not separated from
    the leader at BH p<0.05). Returns zeros if contrasts_v2.csv is absent or was
    left header-only by a `--fast` aggregate."""
    out = {"n_pairs": 0, "n_sig": 0, "pct_sig": 0.0, "n_cluster": 1}
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        pairs = [r for r in csv.DictReader(f) if r.get("p_bh") not in ("", None)]
    if not pairs:
        return out
    pbh = {frozenset((r["model_a"], r["model_b"])): float(r["p_bh"]) for r in pairs}
    out["n_pairs"] = len(pairs)
    out["n_sig"] = sum(1 for r in pairs if float(r["p_bh"]) < 0.05)
    out["pct_sig"] = 100.0 * out["n_sig"] / len(pairs)
    top = ranked[0]["model"]
    out["n_cluster"] = 1 + sum(
        1 for r in ranked[1:]
        if pbh.get(frozenset((top, r["model"])), 0.0) >= 0.05)
    return out


def build_pact_macros(rows: List[dict]) -> str:
    """Emit paper/tables/pact_macros.tex: every PACTScore number the prose quotes,
    as a \\newcommand, so the text can never drift from metrics_v2.csv. The prose in
    methodology.tex / results.tex references these and nothing else.

    A hand-written stub of this file ships with provisional values wrapped in
    \\PROVISIONAL (rendered red) for drafting before a full aggregate run; this
    generator overwrites it with the real numbers and drops the wrapper.
    """
    rk = sorted(rows, key=lambda r: fnum(r, "pact_score"), reverse=True)
    lead, second, last = rk[0], rk[1], rk[-1]
    scores = [fnum(r, "pact_score") for r in rk]
    gaps = [fnum(r, "pact_steer_gap") for r in rk]
    halfwidths = [(fnum(r, "pact_score_hi") - fnum(r, "pact_score_lo")) / 2
                  for r in rk]
    sig = pact_significance(rk)

    def _ci(r: dict) -> str:
        return f"[{fnum(r, 'pact_score_lo'):.3f}, {fnum(r, 'pact_score_hi'):.3f}]"

    def _ints(key: str) -> List[int]:
        return [int(float(r.get(key) or 0)) for r in rk]

    # Per-model denominators: an item whose every replication was `unclear` leaves
    # that model's denominator, so these are ranges over the panel, not constants.
    # The frozen universe is read from the item set itself.
    n_universe, n_multi = item_universe()
    # Per-model denominators are per cell; double for the two-mode item basis
    # (each cell is scored as a base item and an anti_adversarial item).
    decided = [2 * x for x in _ints("pact_n_items")]
    decided_t2 = [2 * x for x in _ints("pact_n_items_t2")]
    # T2 degradations are per model; the max is the number that bounds the claim.
    n_t2_missing = 2 * max(_ints("pact_n_t2_missing"))
    # models the mandate makes WORSE, and the biggest base-vs-directed movers
    backfire = [r for r in rk if fnum(r, "pact_steer_gap") < 0]
    by_base = sorted(rk, key=lambda r: fnum(r, "pact_base"), reverse=True)
    base_rank = {r["model"]: i + 1 for i, r in enumerate(by_base)}
    moves = [(r["model"], base_rank[r["model"]] - (i + 1))
             for i, r in enumerate(rk)]
    faller = min(moves, key=lambda kv: kv[1])
    riser = max(moves, key=lambda kv: kv[1])

    def cmd(name: str, val: str) -> str:
        return f"\\newcommand{{\\{name}}}{{{val}}}"

    return "\n".join([
        "% AUTO-GENERATED by src/benchmark/make_paper_assets.py - do not edit by hand.",
        "% Every PACTScore figure quoted in the prose lives here, so the text cannot",
        "% drift from results/benchmark/metrics_v2.csv. Regenerate after each aggregate.",
        cmd("PactN", str(len(rk))),
        cmd("PactLeader", display(lead["model"])),
        cmd("PactLeaderScore", f"{fnum(lead, 'pact_score'):.3f}"),
        cmd("PactLeaderCI", _ci(lead)),
        cmd("PactSecond", display(second["model"])),
        cmd("PactSecondScore", f"{fnum(second, 'pact_score'):.3f}"),
        cmd("PactSecondCI", _ci(second)),
        cmd("PactLast", display(last["model"])),
        cmd("PactLastScore", f"{fnum(last, 'pact_score'):.3f}"),
        cmd("PactLastCI", _ci(last)),
        cmd("PactSpread", f"{max(scores) - min(scores):.3f}"),
        cmd("PactTopGap", f"{fnum(lead, 'pact_score') - fnum(second, 'pact_score'):.3f}"),
        cmd("PactCIHalfMin", f"{min(halfwidths):.3f}"),
        cmd("PactCIHalfMax", f"{max(halfwidths):.3f}"),
        cmd("PactNPairs", str(sig["n_pairs"])),
        cmd("PactNSig", str(sig["n_sig"])),
        cmd("PactPctSig", f"{sig['pct_sig']:.0f}"),
        cmd("PactTopCluster", str(sig["n_cluster"])),
        cmd("PactLeaderFail", f"{100 * (1 - fnum(lead, 'pact_score')):.0f}"),
        cmd("PactLeaderBase", f"{fnum(lead, 'pact_base'):.3f}"),
        cmd("PactLeaderDirected", f"{fnum(lead, 'pact_directed'):.3f}"),
        cmd("PactGapMin", f"{min(gaps):+.3f}"),
        cmd("PactGapMax", f"{max(gaps):+.3f}"),
        cmd("PactNBackfire", str(len(backfire))),
        cmd("PactBiggestFaller", display(faller[0])),
        cmd("PactBiggestFallerPlaces", str(abs(faller[1]))),
        cmd("PactBiggestRiser", display(riser[0])),
        cmd("PactBiggestRiserPlaces", str(abs(riser[1]))),
        cmd("PactNItems", _tex_int(n_universe)),
        cmd("PactNItemsTwo", _tex_int(n_multi)),
        cmd("PactNSingleTurn", _tex_int(n_universe - n_multi)),
        # Per-model denominators after all-unclear items drop out.
        cmd("PactNDecidedMin", _tex_int(min(decided))),
        cmd("PactNDecidedMax", _tex_int(max(decided))),
        cmd("PactNDecidedTwoMin", _tex_int(min(decided_t2))),
        cmd("PactNDecidedTwoMax", _tex_int(max(decided_t2))),
        cmd("PactNTwoMissing", str(n_t2_missing)),
        "",
    ])


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
    """One row per scenario, carrying every field of the frozen `registry.py`
    seed verbatim: title, domain, rule, temptation, and local objective. This
    replaces the earlier pair of tables (a domain-per-row summary and a
    rule-only list), which split the same seed across two places and dropped
    the temptation and local-objective text entirely.

    Rendered as several full-width table* floats because 48 rows of wrapped
    prose overflow a single float and longtable is unsupported in AAAI's
    twocolumn layout. The scenario id is omitted: the title and domain
    identify a row, and the id is an internal key.
    """
    scen = sorted(REG.SCENARIOS, key=lambda s: (s.domain, s.letter))
    per = 12                                   # rows per float
    parts = [scen[i:i + per] for i in range(0, len(scen), per)]
    out = ["% AUTO-GENERATED by src/benchmark/make_paper_assets.py - do not edit by hand.",
           "% Source: src/benchmark/registry.py SCENARIOS. Every column is the frozen",
           "% seed text the generators were conditioned on, reproduced verbatim."]
    for i, chunk in enumerate(parts):
        of = f"{i + 1} of {len(parts)}"
        out += [
            r"\begin{table*}[t]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\renewcommand{\arraystretch}{1.15}",
            r"\begin{tabular}{@{}p{2.55cm} p{1.85cm} p{4.45cm} p{3.05cm} p{4.55cm}@{}}",
            r"\toprule",
            (r"\textbf{Scenario} & \textbf{Domain} & \textbf{Rule} & "
             r"\textbf{Temptation} & \textbf{Local objective} \\"),
            r"\midrule",
        ]
        for s in chunk:
            dom = _tex_escape(REG.DOMAIN_BY_KEY[s.domain].title)
            out.append(
                f"{_tex_escape(s.title)} & {dom} & {_tex_escape(s.rule)} & "
                f"{_tex_escape(s.temptation)} & "
                f"{_tex_escape(s.local_objective)} \\\\")
        out += [
            r"\bottomrule",
            r"\end{tabular}",
            (r"\caption{The 48 scenario seeds (" + of + r"), reproduced verbatim from "
             r"the frozen scenario registry. \emph{Rule} is the standing obligation the agent "
             r"must hold, \emph{temptation} what violating it buys the requester, and "
             r"\emph{local objective} what the convenient option wins in the "
             r"requester's own terms, which is what compliance costs them. The item "
             r"rendering appends the rule to the persona as a plain informational "
             r"note, never as a command.}"),
            (r"\label{tab:rules}" if i == 0 else f"\\label{{tab:rules{i + 1}}}"),
            r"\end{table*}",
            "",
        ]
    return "\n".join(out)


def build_trivial_table(path: str = TRIVIAL_CSV) -> str:
    """Gameability floor: the four trivial agents scored on every axis and on
    PACTScore. If any out-ranks a real model, that number is broken. Reads
    trivial_v2.csv (written by aggregate); honesty is omitted because the
    undefined-maps-to-1 rule makes it gameable by a non-violating agent by design,
    so it is not a fair floor. Three decimals, to match the leaderboard."""
    if not os.path.exists(path):
        return ("% trivial_v2.csv missing - run `aggregate` to regenerate it.\n")
    rows = load_rows(path)
    axes = [(a, lbl) for a, lbl in AXES if a != "transparency"]
    lines = [
        "% AUTO-GENERATED by src/benchmark/make_paper_assets.py - do not edit by hand.",
        "% Source: results/benchmark/trivial_v2.csv (aggregate's gameability agents).",
        r"\begin{tabular}{l ccccc c}",
        r"\toprule",
        r"\textbf{Trivial agent} & \textbf{Default} & \textbf{Pressure} & "
        r"\textbf{Pushback} & \textbf{Steer.} & \textbf{Scope} & \textbf{PACTScore} \\",
        r"\midrule",
    ]
    for r in rows:
        name = r["model"].replace("trivial:", "").replace("_", r"\_")
        cells = []
        for a, _ in axes:
            v = fnum(r, a)
            cells.append("n/a" if v != v else f"{v:.3f}")
        p = fnum(r, "pact_score")
        cells.append("n/a" if p != p else f"\\textbf{{{p:.3f}}}")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
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
        cells = [_colcell(fnum(r, c), lo[c], hi[c], ".3f") for c in cols]
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
        lines.append(f"{name} & {fnum(r, 't1_comply'):.3f} & "
                     f"{fnum(r, 't2_hold'):.3f} & {fnum(r, 'steerability'):.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def field_summary(rows: List[dict]) -> str:
    pact = sorted((fnum(r, "pact_score") for r in rows), reverse=True)
    steer = [fnum(r, "steerability") for r in rows]
    hon = [(display(r["model"]), fnum(r, "transparency")) for r in rows]
    hon_best = max(hon, key=lambda t: t[1])
    n95 = sum(1 for v in pact if v >= 0.95)
    n90 = sum(1 for v in pact if v >= 0.90)
    out = [
        "-- field summary (quote these in prose) --",
        f"models: {len(rows)}",
        f"best PACTScore:   {pact[0]:.3f}  (1 - best = {1 - pact[0]:.3f} unreliable share)",
        f"median PACTScore: {median(pact):.3f}",
        f"worst PACTScore:  {pact[-1]:.3f}",
        f"models >= 0.95 PACTScore: {n95}",
        f"models >= 0.90 PACTScore: {n90}",
        f"steerability range: {min(steer):.2f} - {max(steer):.2f}",
        f"highest transparency: {hon_best[0]} at {hon_best[1]:.2f}",
        f"abstention range: {min(a for r in rows for a in [fnum(r, 'abstention')] if a == a) * 100:.1f}% - "
        f"{max(a for r in rows for a in [fnum(r, 'abstention')] if a == a) * 100:.1f}%  "
        f"(median {median(sorted(a for r in rows for a in [fnum(r, 'abstention')] if a == a)) * 100:.1f}%)",
    ]
    return "\n".join(out)


def dataset_composition_figure(items_path: str = ITEMS_JSONL) -> str:
    """Frozen-set composition: scenario cells per domain, stacked by cell group
    (Appendix fig:datasetcomp). Rendered from the frozen item file so the figure
    can never drift from the corpus the models actually ran."""
    import json
    from collections import Counter, defaultdict
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    FS.use_paper_style()

    groups = ["neutral", "pressure", "guard_nonbinding",
              "attack_binding", "attack_nonbinding"]
    colors = dict(zip(groups, [FS.MUTED, FS.BLUE, FS.GREEN, FS.RED, FS.GOLD]))
    per: Dict[str, Counter] = defaultdict(Counter)
    total = 0
    with open(items_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            per[d["domain"]][d["group"]] += 1
            total += 1
    domains = sorted(per)
    x = np.arange(len(domains))
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    bottom = np.zeros(len(domains))
    for g in groups:
        vals = np.array([per[d][g] for d in domains], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=colors[g], label=g, width=0.72)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("scenario cells")
    ax.set_title(f"PACT frozen set: {total:,} scenario cells by domain and cell group",
                 loc="left", fontsize=11, pad=26)
    ax.legend(ncol=len(groups), fontsize=7.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), frameon=False)
    ax.grid(axis="y", color=FS.GRID, lw=0.8)
    ax.set_axisbelow(True)
    FS.strip_axes(ax)
    fig.tight_layout()
    os.makedirs(FIG_SRC, exist_ok=True)
    out = os.path.join(FIG_SRC, "dataset_by_domain_group.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return out


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
    mac_path = os.path.join(TABLES_DIR, "pact_macros.tex")
    with open(mac_path, "w", encoding="utf-8") as f:
        f.write(build_pact_macros(rows))
    print(f"wrote {mac_path}  (PACTScore figures quoted in the prose)")
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
    fig_path = dataset_composition_figure()
    print(f"wrote {fig_path}")
    copied = copy_figures()
    print(f"copied {len(copied)} figures -> {FIGURES_DIR}")
    print()
    print(field_summary(rows))


if __name__ == "__main__":
    main()
