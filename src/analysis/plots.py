#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matplotlib figure generation for the paper.

Every figure is a multi-model view: an IQR box per condition, the mean as a tick, and
one dot per model. Reads the per-experiment JSONL in results/data and writes PDFs to
results/figures. Run `python -m src.analysis.plots` for all figures, or pass one name
(e.g. `pressure`).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

RESULTS_DIR = str(_SCRIPT_DIR / "../../results/data")
FIGURES_DIR = str(_SCRIPT_DIR / "../../results/figures")

DEFAULT_SINGLE_MODEL = "google/gemini-3-flash-preview"

DEFAULT_RESULTS = {
    "controls":           "controls.jsonl",
    "wording":            "wording.jsonl",
    "authority":          "authority.jsonl",
    "peer_signals":       "peer_signals.jsonl",
    "norms":              "norms.jsonl",
    "pressure":           "pressure.jsonl",
    "multiturn":          "multiturn.jsonl",
    "multiturn_pressure": "multiturn_pressure.jsonl",
    "stakes":             "stakes.jsonl",
    "urgency":            "urgency.jsonl",
}

# ── Axes constants ────────────────────────────────────────────────────────────

FIN_LEVELS  = ["none", "low", "breakeven", "high"]
# Fine amount labels (brevity for x-axes).
FIN_LABELS  = ["None", "Small", "Medium", "Large"]

FRAMINGS = ["directive", "informational", "discretionary"]
FRAMING_LABEL = {
    "directive":     "Imperative",
    "informational": "Informational",
    "discretionary": "Discretionary",
}

# ── Paper-wide styling ────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size":         14,
    "axes.titlesize":    17,
    "axes.labelsize":    17,
    "xtick.labelsize":   15,
    "ytick.labelsize":   15,
    "legend.fontsize":   13,
    "figure.dpi":        300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
})

CONTROL_COLOR   = "#9aa0a6"
ZERO_STUB_PCT   = 2.6

FRAMING_COLOR = {
    "directive":     "#2166ac",
    "informational": "#4dac26",
    "discretionary": "#b2182b",
}

MODEL_DOT_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00",
    "#56B4E9", "#F0E442", "#332288", "#117733", "#AA4499",
    "#88CCEE",
]

MODEL_DOT_SIZE = 22
MODEL_DOT_EDGE_COLOR = "none"
MODEL_DOT_EDGE_WIDTH = 0.0

CONDITION_SERIES_PALETTE = MODEL_DOT_PALETTE

BAR_FIGSIZE_MULTIMODEL   = (10.4, 4.2)
MULTIMODEL_GROUP_GAP     = 1.03
MULTIMODEL_SIDE_PAD      = 0.45
MULTIMODEL_EDGE_PAD      = 0.07

_MODEL_LABEL_MAP = {
    "deepseek/deepseek-v3.2":           "DeepSeek V3.2",
    "google/gemini-3-flash-preview":    "Gemini 3 Flash",
    "google/gemma-4-31b-it":            "Gemma 4 31B",
    "meta-llama/llama-4-maverick":      "Llama 4 Maverick",
    "minimax/minimax-m2.7":             "Minimax M2.7",
    "mistralai/mistral-small-2603":     "Mistral Small 2603",
    "moonshotai/kimi-k2.5":             "Kimi K2.5",
    "openai/gpt-oss-120b":              "GPT OSS 120B",
    "qwen/qwen3.5-flash-02-23":         "Qwen 3.5 Flash",
    "x-ai/grok-4.1-fast":               "Grok 4.1 Fast",
    "z-ai/glm-4.7-flash":               "GLM 4.7 Flash",
    "nvidia/nemotron-3-super-120b-a12b": "Nemotron 3 Super",
}

# Canonical model order for the figures.
MODEL_ORDER: List[str] = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.5-flash-02-23",
    "meta-llama/llama-4-maverick",
    "moonshotai/kimi-k2.5",
    "nvidia/nemotron-3-super-120b-a12b",
    "minimax/minimax-m2.7",
    "mistralai/mistral-small-2603",
    "deepseek/deepseek-v3.2",
    "x-ai/grok-4.1-fast",
    "google/gemini-3-flash-preview",
    "google/gemma-4-31b-it",
    "z-ai/glm-4.7-flash",
]

MODEL_COLOR_MAP: Dict[str, str] = {
    "openai/gpt-oss-120b":              "#0072B2",
    "openai/gpt-oss-120b:free":         "#0072B2",
    "qwen/qwen3.5-flash-02-23":         "#E69F00",
    "meta-llama/llama-4-maverick":      "#009E73",
    "x-ai/grok-4.1-fast":               "#D55E00",
    "deepseek/deepseek-v3.2":           "#CC79A7",
    "google/gemini-3-flash-preview":    "#F0E442",
    "moonshotai/kimi-k2.5":             "#000000",
    "z-ai/glm-4.7-flash":               "#882255",
    "mistralai/mistral-small-2603":     "#44AA99",
    "minimax/minimax-m2.7":             "#EE6677",
    "google/gemma-4-31b-it":            "#999933",
    # Darker than the pastel cyan so it doesn't visually blend with condition fills
    # or table-style compliance shading.
    "nvidia/nemotron-3-super-120b-a12b": "#2F4B7C",
}

# Short display names for axis/legend labels.
# Used in both the model legend (all figures) and violation-reasoning x-axis labels.
MODEL_SHORT_NAMES: Dict[str, str] = {
    "openai/gpt-oss-120b":              "GPT-OSS",
    "qwen/qwen3.5-flash-02-23":         "Qwen 3.5",
    "meta-llama/llama-4-maverick":      "Llama 4",
    "moonshotai/kimi-k2.5":             "Kimi K2.5",
    "x-ai/grok-4.1-fast":               "Grok 4.1",
    "deepseek/deepseek-v3.2":           "DeepSeek",
    "google/gemini-3-flash-preview":    "Gemini 3",
    "z-ai/glm-4.7-flash":               "GLM 4.7",
    "mistralai/mistral-small-2603":     "Mistral Sm.",
    "minimax/minimax-m2.7":             "Minimax M2",
    "google/gemma-4-31b-it":            "Gemma 4",
    "nvidia/nemotron-3-super-120b-a12b": "Nemotron 3",
}

# Group membership for marker shapes (keep in sync with appendix tables)
GROUP_I = {"openai/gpt-oss-120b", "qwen/qwen3.5-flash-02-23", "meta-llama/llama-4-maverick"}


def _model_marker(model: str) -> str:
    m = str(model or "").strip().lower()
    return "o" if m in GROUP_I else "^"


# ── Large-font helpers ─────────────────────────────────────────────────────────
# Applied to all figures EXCEPT exp2 (non-grouped), exp8, and exp9.

_LARGE_AXES_RC: Dict[str, Any] = {
    "axes.titlesize":  22,
    "axes.labelsize":  22,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
}
_LARGE_COND_FONTSIZE: float       = 22.0
_LARGE_COND_TITLE_FONTSIZE: float = 23.0


@contextlib.contextmanager
def _large_font_ctx():
    """Temporarily enlarge axes/tick label sizes for selected figures."""
    old = {k: plt.rcParams[k] for k in _LARGE_AXES_RC}
    plt.rcParams.update(_LARGE_AXES_RC)
    try:
        yield
    finally:
        plt.rcParams.update(old)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def _load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _normalize_framing_name(f: str) -> str:
    f = (f or "").strip().lower()
    return {"descriptive": "informational", "permissive": "discretionary"}.get(f, f)


def _normalize_scenario_name(sc: str) -> str:
    parts = [_normalize_framing_name(p) for p in (sc or "").strip().split("_") if p]
    return "_".join(parts)


def _normalize_rows_for_plotting(rows: Sequence[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for r in rows:
        rr = dict(r)
        if "scenario" in rr:
            rr["scenario"] = _normalize_scenario_name(str(rr.get("scenario", "")))
        meta = rr.get("metadata")
        if isinstance(meta, dict) and meta.get("framing"):
            mm = dict(meta)
            mm["framing"] = _normalize_framing_name(str(mm.get("framing")))
            rr["metadata"] = mm
        out.append(rr)
    return out


def _group_by(rows: Sequence[Dict], key: str) -> Dict[str, List[Dict]]:
    d: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        d[str(r.get(key, ""))].append(r)
    return dict(d)


def _available_models(rows: Sequence[Dict]) -> List[str]:
    return sorted({str(r.get("model", "unknown")) for r in rows})


def _comply_rate(rows: Sequence[Dict]) -> Optional[float]:
    parseable = [r for r in rows if r.get("parseable", True)]
    if not parseable:
        return None
    return sum(1 for r in parseable if r.get("compliant")) / len(parseable)


def _switch_rate(rows: Sequence[Dict]) -> Optional[float]:
    valid = [r for r in rows if r.get("t2_parseable")]
    if not valid:
        return None
    return sum(1 for r in valid if r.get("switched")) / len(valid)


def _per_model_rates(
    rows: Sequence[Dict],
    scenario: str,
    models: Sequence[str],
    *,
    rate_fn=_comply_rate,
) -> Dict[str, float]:
    by_m: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        if r.get("scenario") == scenario:
            by_m[str(r.get("model", "unknown"))].append(r)
    return {m: v for m in models if (v := rate_fn(by_m.get(m, []))) is not None}


# ══════════════════════════════════════════════════════════════════════════════
# STYLE / LAYOUT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ConditionSpec:
    key: str
    label: str


def _pretty_model_name(model: str) -> str:
    m = str(model or "").strip()
    low = m.lower()
    # Short names take priority — keeps legend labels consistent with regime scatter.
    if low in MODEL_SHORT_NAMES:
        return MODEL_SHORT_NAMES[low]
    if low in _MODEL_LABEL_MAP:
        return _MODEL_LABEL_MAP[low]
    core = m.split(":", 1)[0]
    tail = core.split("/", 1)[-1]
    tail = tail.replace("-", " ").replace("_", " ").strip()
    return " ".join(w.capitalize() for w in tail.split())[:32]


def _model_color(model_id: str, index: int = 0) -> str:
    key = str(model_id or "").strip().lower()
    return MODEL_COLOR_MAP.get(key, MODEL_DOT_PALETTE[index % len(MODEL_DOT_PALETTE)])


def _model_dot_offsets(models: Sequence[str], *, span: float = 0.06) -> Dict[str, float]:
    ms = list(models)
    if not ms:
        return {}
    if len(ms) == 1:
        return {ms[0]: 0.0}
    offs = np.linspace(-span, span, num=len(ms))
    return {m: float(o) for m, o in zip(ms, offs)}


def _legend_rows_for_count(n: int, *, max_per_row: int = 6) -> int:
    return max(1, math.ceil(n / max_per_row))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _format_pct_axis(ax: plt.Axes, *, ylabel: str = "Compliance Rate") -> None:
    ax.set_ylim(0, 102)
    ax.set_ylabel(ylabel)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{x}%" for x in [0, 20, 40, 60, 80, 100]])
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", linestyle="-", linewidth=0.7, alpha=0.22, color=CONTROL_COLOR)


def _bar_positions(
    n_groups: int,
    n_bars: int,
    bar_w: float = 0.18,
    group_gap: float = 1.0,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    xs = np.arange(n_groups) * group_gap
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_w
    return xs, [xs + o for o in offsets]


def _group_bar_w(n_bars: int, *, max_halfwidth: float = 0.46, width_ratio: float = 0.85) -> float:
    if n_bars <= 1:
        return 0.22
    denom = ((n_bars - 1) / 2.0) + (width_ratio / 2.0)
    return max(0.06, min(0.22, max_halfwidth / denom))


def _multimodel_bar_w(n_bars: int) -> float:
    max_halfwidth = max(0.06, (MULTIMODEL_GROUP_GAP / 2.0) - MULTIMODEL_EDGE_PAD)
    return _group_bar_w(n_bars, max_halfwidth=max_halfwidth)


def _add_vertical_separators(
    ax: plt.Axes,
    pos: Sequence[np.ndarray],
    *,
    n_groups: int,
    intra_color: str = "#6b6b6b",
    group_color: str = "#4a4a4a",
) -> None:
    if not pos:
        return
    n_bars = len(pos)
    centers = np.array([np.mean([p[j] for p in pos]) for j in range(n_groups)], dtype=float)
    if n_bars >= 2:
        for j in range(n_groups):
            for i in range(n_bars - 1):
                x = (pos[i][j] + pos[i + 1][j]) / 2.0
                ax.axvline(x, color=intra_color, linestyle=(0, (1, 2)),
                           linewidth=0.7, alpha=0.35, zorder=6)
    for j in range(n_groups - 1):
        boundary = (centers[j] + centers[j + 1]) / 2.0
        ax.axvline(boundary, color=group_color, linestyle="-",
                   linewidth=1.45, alpha=0.78, zorder=6)


def _draw_iqr_boxplot(
    ax: plt.Axes,
    *,
    x: float,
    vals_pct: Sequence[float],
    color: str,
    width: float,
) -> None:
    if not vals_pct:
        return
    import matplotlib.colors as mcolors
    r, g, b, _ = mcolors.to_rgba(color)
    face_rgba = (r, g, b, 0.22)        # transparent fill
    dark_rgba = (r * 0.55, g * 0.55, b * 0.55, 1.0)   # darker shade of same color, fully opaque
    STROKE = 1.3   # shared thickness for box border, whiskers, and caps
    ax.boxplot(
        [list(vals_pct)],
        positions=[x],
        widths=[max(0.01, 0.75 * float(width))],
        whis=(0, 100),
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": face_rgba, "edgecolor": dark_rgba, "linewidth": STROKE},
        whiskerprops={"color": dark_rgba, "linewidth": STROKE},
        capprops={"color": dark_rgba, "linewidth": STROKE},
        medianprops={"color": dark_rgba, "linewidth": STROKE},
    )


def _add_stacked_legends(
    fig: plt.Figure,
    *,
    condition_handles: Sequence,
    model_handles: Sequence,
    condition_title: Optional[str] = None,
    condition_ncol: Optional[int] = None,
    condition_fontsize: float = 18.0,
    condition_title_fontsize: float = 19.0,
    model_fontsize: float = 15.0,
    legend_gap: float = 0.010,
    large_fonts: bool = False,
) -> float:
    """Place a conditions legend (1 row) above a models legend (1-2 rows).

    Returns the rect_top value for fig.tight_layout(rect=(0, 0, 1, rect_top)).

    When large_fonts=True the row-height and gap estimates are scaled up to
    match the bigger condition-label text, preventing the model legend from
    overlapping the plot area.
    """
    n_cond   = len(condition_handles)
    n_models = len(model_handles)

    # Per-row height constants — larger when condition labels are bigger.
    _cond_base_h  = 0.120 if large_fonts else 0.095
    _cond_line_h  = 0.054 if large_fonts else 0.042
    _model_row_h  = 0.082 if large_fonts else 0.065
    _gap          = (legend_gap + 0.022) if large_fonts else float(legend_gap)
    _bottom_pad   = 0.030 if large_fonts else 0.015

    COND_ROW_HEIGHT = 0.0
    if n_cond:
        # Default to a wrapped layout for many conditions (prevents off-page legends).
        cond_ncol = max(1, int(condition_ncol)) if condition_ncol else min(n_cond, 6)
        cond_rows = max(1, math.ceil(n_cond / cond_ncol))

        cond_leg = fig.legend(
            handles=list(condition_handles),
            title=condition_title,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            bbox_transform=fig.transFigure,
            ncol=cond_ncol,
            fontsize=condition_fontsize,
            frameon=False,
            handlelength=0.9, handletextpad=0.45, columnspacing=1.2, borderpad=0.5,
        )

        if condition_title and hasattr(cond_leg, "get_title"):
            try:
                cond_leg.get_title().set_fontsize(condition_title_fontsize)
            except Exception:
                pass

        cond_max_lines = 1
        for h in condition_handles:
            if hasattr(h, "get_label"):
                label = str(h.get_label() or "")
                cond_max_lines = max(cond_max_lines, label.count("\n") + 1)

        if condition_title:
            cond_max_lines += 1

        cond_row_height = _cond_base_h + max(0, cond_max_lines - 1) * _cond_line_h
        COND_ROW_HEIGHT = cond_rows * cond_row_height

    MODEL_ROW_HEIGHT = _model_row_h
    LEGEND_GAP       = _gap

    if not model_handles:
        return 1.0 - COND_ROW_HEIGHT - _bottom_pad

    # Keep model legend in 2 rows of ~6.
    max_per_row = 6
    m_rows = max(1, math.ceil(n_models / max_per_row))
    m_ncol = math.ceil(n_models / m_rows)

    fig.legend(
        handles=list(model_handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0 - COND_ROW_HEIGHT - LEGEND_GAP),
        bbox_transform=fig.transFigure,
        ncol=m_ncol,
        fontsize=model_fontsize,
        frameon=True, fancybox=True, framealpha=0.92, edgecolor="#cccccc",
        handlelength=0.8, handletextpad=0.35, columnspacing=0.8, borderpad=0.4,
    )

    total = COND_ROW_HEIGHT + LEGEND_GAP + (m_rows * MODEL_ROW_HEIGHT)
    return 1.0 - total - _bottom_pad


def _save(fig: plt.Figure, out_path: str, fmt: str, dpi: int) -> None:
    fig.savefig(out_path, format=fmt, dpi=dpi)
    plt.close(fig)
    if fmt == "pdf":
        try:
            from src.analysis._crop import crop_pdf as _crop
            _crop(Path(out_path))
        except Exception as e:
            print(f"[WARN] PDF crop skipped: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SHARED MULTIMODEL DRAWING KERNEL
# ══════════════════════════════════════════════════════════════════════════════

def _build_model_legend_handles(model_list: Sequence[str], model_colors: Dict[str, str]) -> List:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markeredgecolor="none",
            markeredgewidth=0.0,
            markerfacecolor=model_colors[m],
            markersize=5,
            label=_pretty_model_name(m),
        )
        for m in model_list
        if m in model_colors
    ]


def _draw_multimodel_cells(
    ax: plt.Axes,
    *,
    rows: Sequence[Dict],
    pos: List[np.ndarray],
    groups: Sequence,          # outer iterable (conditions / framings / …)
    group_colors: Sequence[str],
    group_scenarios: callable,  # (group_item, fin_idx) → scenario str
    model_list: Sequence[str],
    dot_offsets: Dict[str, float],
    model_colors: Dict[str, str],
    bar_w: float,
    rate_fn=_comply_rate,
    fin_levels: Sequence[str] = FIN_LEVELS,
) -> None:
    """Render IQR boxplot + mean tick + per-model dots for every (group, fin) cell."""
    for i, (grp, color) in enumerate(zip(groups, group_colors)):
        for j, fin in enumerate(fin_levels):
            scenario = group_scenarios(grp, j)
            per_m = _per_model_rates(rows, scenario, model_list, rate_fn=rate_fn)
            vals = list(per_m.values())
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            x    = pos[i][j]

            _draw_iqr_boxplot(ax, x=x, vals_pct=[100.0 * v for v in vals],
                              color=color, width=min(0.18, bar_w * 0.82))
            for m, r in per_m.items():
                ax.scatter(
                    [x + dot_offsets.get(m, 0.0)],
                    [100.0 * r],
                    s=MODEL_DOT_SIZE,
                    color=model_colors.get(m, "#333333"),
                    edgecolors=MODEL_DOT_EDGE_COLOR,
                    linewidths=MODEL_DOT_EDGE_WIDTH,
                    alpha=0.95,
                    zorder=2,
                )


def _multimodel_standard_axes(ax: plt.Axes, xs: np.ndarray) -> None:
    ax.set_xlim(xs[0] - MULTIMODEL_SIDE_PAD, xs[-1] + MULTIMODEL_SIDE_PAD)
    ax.set_xticks(xs)
    ax.set_xticklabels(FIN_LABELS)
    ax.set_xlabel("Fine Amount")
    _format_pct_axis(ax)


def _resolve_model_list(
    rows: Sequence[Dict],
    models: Optional[Sequence[str]],
    *,
    exclude: Set[str] = frozenset(),
) -> Tuple[List[str], Dict[str, float], Dict[str, str]]:
    all_models = _available_models(rows)
    model_list = [m for m in (list(models) if models else all_models) if m not in exclude]
    if not model_list:
        raise RuntimeError("No models found in results file")

    if models:
        # Preserve caller-provided order.
        model_order = list(model_list)
    else:
        # Use canonical order, then append any unknown models deterministically.
        model_order = [m for m in MODEL_ORDER if m in model_list]
        extras = sorted([m for m in model_list if m not in set(MODEL_ORDER)])
        model_order.extend(extras)

    dot_offsets  = _model_dot_offsets(model_order)
    model_colors = {m: _model_color(m, i) for i, m in enumerate(model_order)}
    return model_order, dot_offsets, model_colors


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC MULTIMODEL BAR FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def _plot_grouped_bars_multimodel(
    *,
    rows: Sequence[Dict],
    out_path: str,
    conditions: Sequence[ConditionSpec],
    scenario_key: callable,
    models: Optional[Sequence[str]],
    fmt: str,
    dpi: int,
    condition_legend_ncol: Optional[int] = None,
    x_label: str = "Fine Amount",
    figsize: Tuple[float, float] = BAR_FIGSIZE_MULTIMODEL,
    large_fonts: bool = False,
) -> None:
    """Generic multimodel grouped-bar figure (IQR + mean + dots)."""
    cond_fontsize       = _LARGE_COND_FONTSIZE       if large_fonts else 18.0
    cond_title_fontsize = _LARGE_COND_TITLE_FONTSIZE if large_fonts else 19.0
    ctx = _large_font_ctx() if large_fonts else contextlib.nullcontext()

    with ctx:
        model_list, dot_offsets, model_colors = _resolve_model_list(rows, models)

        # Drop conditions that have no data at all.
        def has_data(cond: ConditionSpec) -> bool:
            return any(
                r.get("scenario") == scenario_key(cond.key, fin) and r.get("model") in model_list
                for r in rows for fin in FIN_LEVELS
            )
        conditions = [c for c in conditions if has_data(c)]

        # Assign colors: control → gray, rest → palette in order.
        palette_idx = 0
        group_colors: List[str] = []
        for c in conditions:
            if c.key == "control":
                group_colors.append(CONTROL_COLOR)
            else:
                group_colors.append(CONDITION_SERIES_PALETTE[palette_idx % len(CONDITION_SERIES_PALETTE)])
                palette_idx += 1

        fig, ax = plt.subplots(figsize=figsize)
        n_bars = len(conditions)
        bar_w  = _multimodel_bar_w(n_bars)
        xs, pos = _bar_positions(len(FIN_LEVELS), n_bars, bar_w=bar_w, group_gap=MULTIMODEL_GROUP_GAP)

        _draw_multimodel_cells(
            ax,
            rows=rows,
            pos=pos,
            groups=conditions,
            group_colors=group_colors,
            group_scenarios=lambda cond, j: scenario_key(cond.key, FIN_LEVELS[j]),
            model_list=model_list,
            dot_offsets=dot_offsets,
            model_colors=model_colors,
            bar_w=bar_w,
        )
        _add_vertical_separators(ax, pos, n_groups=len(FIN_LEVELS))
        _multimodel_standard_axes(ax, xs)
        ax.set_xlabel(x_label)

        cond_handles = [
            Patch(facecolor=c, edgecolor=c, alpha=0.25, label=cond.label)
            for cond, c in zip(conditions, group_colors)
        ]
        model_handles = _build_model_legend_handles(model_list, model_colors)
        top_rect = _add_stacked_legends(
            fig,
            condition_handles=cond_handles,
            model_handles=model_handles,
            condition_ncol=condition_legend_ncol,
            condition_fontsize=cond_fontsize,
            condition_title_fontsize=cond_title_fontsize,
            large_fonts=large_fonts,
        )
        fig.tight_layout(rect=(0, 0, 1, top_rect))
        _save(fig, out_path, fmt, dpi)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: CONTROLS (multimodel)
# ══════════════════════════════════════════════════════════════════════════════

def plot_controls(
    rows: Sequence[Dict],
    out_path: str,
    models: Optional[Sequence[str]],
    fmt: str,
    dpi: int,
) -> None:
    with _large_font_ctx():
        model_list, dot_offsets, model_colors = _resolve_model_list(rows, models)
        fig, ax = plt.subplots(figsize=BAR_FIGSIZE_MULTIMODEL)

        n_bars = len(FRAMINGS)
        bar_w  = _multimodel_bar_w(n_bars)
        xs, pos = _bar_positions(len(FIN_LEVELS), n_bars, bar_w=bar_w, group_gap=MULTIMODEL_GROUP_GAP)

        _draw_multimodel_cells(
            ax,
            rows=rows,
            pos=pos,
            groups=FRAMINGS,
            group_colors=[FRAMING_COLOR[f] for f in FRAMINGS],
            group_scenarios=lambda framing, j: f"control_{framing}_{FIN_LEVELS[j]}",
            model_list=model_list,
            dot_offsets=dot_offsets,
            model_colors=model_colors,
            bar_w=bar_w,
        )
        _add_vertical_separators(ax, pos, n_groups=len(FIN_LEVELS))
        _multimodel_standard_axes(ax, xs)

        cond_handles = [
            Patch(facecolor=FRAMING_COLOR[f], edgecolor=FRAMING_COLOR[f], alpha=0.25,
                  label=FRAMING_LABEL[f])
            for f in FRAMINGS
        ]
        model_handles = _build_model_legend_handles(model_list, model_colors)
        top_rect = _add_stacked_legends(
            fig,
            condition_handles=cond_handles,
            model_handles=model_handles,
            condition_fontsize=_LARGE_COND_FONTSIZE,
            condition_title_fontsize=_LARGE_COND_TITLE_FONTSIZE,
            large_fonts=True,
        )
        fig.tight_layout(rect=(0, 0, 1, top_rect))
        _save(fig, out_path, fmt, dpi)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: EXP2 — Obligation wording
# ══════════════════════════════════════════════════════════════════════════════

def _obligation_conditions() -> List[ConditionSpec]:
    return [
        ConditionSpec("requires",           "requires"),
        ConditionSpec("mandates",           "mandates"),
        ConditionSpec("must_use",           "must use"),
        ConditionSpec("should_use",         "should use"),
        ConditionSpec("expects",            "expects"),
        ConditionSpec("recommends",         "recommends"),
        ConditionSpec("encourages",         "encourages"),
        ConditionSpec("informational_ctrl", "informational ctrl"),
    ]


def _wording_scenario_key(cond: str, fin: str) -> str:
    if cond == "requires":
        return f"control_directive_{fin}"
    if cond == "informational_ctrl":
        return f"control_informational_{fin}"
    return f"oblig_{cond}_{fin}"


def plot_wording(
    rows: Sequence[Dict],
    out_path: str,
    models: Optional[Sequence[str]],
    fmt: str,
    dpi: int,
) -> None:
    """Appendix figure: all individual obligation-wording ablations.
    No large fonts — kept at standard sizes.
    """
    _plot_grouped_bars_multimodel(
        rows=rows,
        out_path=out_path,
        conditions=_obligation_conditions(),
        scenario_key=_wording_scenario_key,
        models=models,
        fmt=fmt,
        dpi=dpi,
        condition_legend_ncol=4,
        x_label="Fine amount",
        figsize=(10.6, 4.2),
        large_fonts=False,
    )


# ── Wording, grouped into 4 families ────────────────────────────────────────

def _wording_groups() -> List[Tuple[str, str, List[str]]]:
    """Return [(key, label, [scenario_prefixes]), ...]."""
    return [
        ("directive_required", "Must",          ["control_directive", "oblig_mandates", "oblig_must_use"]),
        ("should_group",       "Should",        ["oblig_should_use", "oblig_expects"]),
        ("recommend_group",    "Recommend",     ["oblig_recommends", "oblig_encourages"]),
        ("informational_ctrl", "Informational (Ctrl)", ["control_informational"]),
    ]


def _wording_group_model_rate(
    rows: Sequence[Dict],
    *,
    model_id: str,
    prefixes: Sequence[str],
    fin_level: str,
) -> Optional[float]:
    matched = [
        r for r in rows
        if r.get("model") == model_id
        and any(str(r.get("scenario", "")) == f"{prefix}_{fin_level}" for prefix in prefixes)
    ]
    return _comply_rate(matched)


def plot_wording_grouped(
    rows: Sequence[Dict],
    out_path: str,
    models: Optional[Sequence[str]],
    fmt: str,
    dpi: int,
) -> None:
    """Main wording figure: 4 wording groups × 4 fine amounts."""
    with _large_font_ctx():
        model_list, dot_offsets, model_colors = _resolve_model_list(rows, models)
        groups = _wording_groups()

        palette_idx = 0
        group_colors: List[str] = []
        for gkey, _, _ in groups:
            if gkey == "informational_ctrl":
                group_colors.append(CONTROL_COLOR)
            else:
                group_colors.append(CONDITION_SERIES_PALETTE[palette_idx % len(CONDITION_SERIES_PALETTE)])
                palette_idx += 1

        fig, ax = plt.subplots(figsize=BAR_FIGSIZE_MULTIMODEL)
        n_bars = len(groups)
        bar_w  = _multimodel_bar_w(n_bars)
        xs, pos = _bar_positions(len(FIN_LEVELS), n_bars, bar_w=bar_w, group_gap=MULTIMODEL_GROUP_GAP)

        for i, ((gkey, glabel, gprefixes), gcolor) in enumerate(zip(groups, group_colors)):
            for j, fin in enumerate(FIN_LEVELS):
                per_m = {
                    m: r for m in model_list
                    if (r := _wording_group_model_rate(rows, model_id=m, prefixes=gprefixes, fin_level=fin)) is not None
                }
                vals = list(per_m.values())
                if not vals:
                    continue
                x = pos[i][j]
                _draw_iqr_boxplot(ax, x=x, vals_pct=[100.0 * v for v in vals],
                                  color=gcolor, width=min(0.18, bar_w * 0.82))
                for m, r in per_m.items():
                    ax.scatter([x + dot_offsets.get(m, 0.0)], [100.0 * r],
                               s=MODEL_DOT_SIZE, color=model_colors.get(m, "#333333"),
                               edgecolors=MODEL_DOT_EDGE_COLOR,
                               linewidths=MODEL_DOT_EDGE_WIDTH,
                               alpha=0.95, zorder=2)

        _add_vertical_separators(ax, pos, n_groups=len(FIN_LEVELS))
        _multimodel_standard_axes(ax, xs)

        cond_handles = [
            Patch(facecolor=c, edgecolor=c, alpha=0.25, label=glabel)
            for (_, glabel, _), c in zip(groups, group_colors)
        ]
        model_handles = _build_model_legend_handles(model_list, model_colors)
        top_rect = _add_stacked_legends(
            fig,
            condition_handles=cond_handles,
            model_handles=model_handles,
            condition_fontsize=_LARGE_COND_FONTSIZE,
            condition_title_fontsize=_LARGE_COND_TITLE_FONTSIZE,
            large_fonts=True,
        )
        fig.tight_layout(rect=(0, 0, 1, top_rect))
        _save(fig, out_path, fmt, dpi)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES: EXP3, EXP4, EXP6 — standard grouped multimodel
# ══════════════════════════════════════════════════════════════════════════════

def _informational_key(cond: str, fin: str) -> str:
    if cond == "baseline":
        # No-regulation baseline is stored as a single scenario with no fin suffix.
        # Plot it only at the "none" fine-amount column.
        return "baseline" if fin == "none" else "__no_data__"
    if cond == "control":
        return f"control_informational_{fin}"
    return f"{cond}_informational_{fin}"


def plot_authority(
    rows: Sequence[Dict],
    out_path: str,
    models: Optional[Sequence[str]],
    fmt: str,
    dpi: int,
) -> None:
    conditions = [
        ConditionSpec("control",       "Control"),
        ConditionSpec("mgr_direct",    "Manager directive"),
        ConditionSpec("mgr_authorize", "Manager authorization"),
        ConditionSpec("board_cost",    "Board cost policy"),
    ]
    _plot_grouped_bars_multimodel(
        rows=rows, out_path=out_path, conditions=conditions,
        scenario_key=_informational_key, models=models, fmt=fmt, dpi=dpi,
        large_fonts=True,
    )


def plot_peer_signals(
    rows: Sequence[Dict],
    out_path: str,
    models: Optional[Sequence[str]],
    fmt: str,
    dpi: int,
) -> None:
    conditions = [
        ConditionSpec("control",        "Control"),
        ConditionSpec("peer_fined",     "Peer fined"),
        ConditionSpec("peer_escaped",   "Peer escaped"),
        ConditionSpec("peer_compliant", "Peer compliant"),
        ConditionSpec("peer_fined_big", "Peer fined (big)"),
    ]
    _plot_grouped_bars_multimodel(
        rows=rows, out_path=out_path, conditions=conditions,
        scenario_key=_informational_key, models=models, fmt=fmt, dpi=dpi,
        large_fonts=True,
    )


def plot_norms(
    rows: Sequence[Dict],
    out_path: str,
    models: Optional[Sequence[str]],
    fmt: str,
    dpi: int,
) -> None:
    conditions = [
        ConditionSpec("baseline",   "No regulation"),
        ConditionSpec("control",    "Control"),
        ConditionSpec("community",  "Community"),
        ConditionSpec("industry",   "Industry"),
        ConditionSpec("media",      "Media"),
    ]

    # Compact layout: a small baseline panel (only one bar) plus a main enforcement panel.
    # This avoids reserving baseline "slots" for the other conditions.
    with _large_font_ctx():
        model_list, dot_offsets, model_colors = _resolve_model_list(rows, models)

        baseline_cond = ConditionSpec("baseline", "No regulation")
        other_conds = [c for c in conditions if c.key != "baseline"]

        def has_baseline_data() -> bool:
            return any(r.get("scenario") == "baseline" and r.get("model") in model_list for r in rows)

        def has_other_data(cond: ConditionSpec) -> bool:
            return any(
                r.get("scenario") == _informational_key(cond.key, fin) and r.get("model") in model_list
                for r in rows for fin in FIN_LEVELS
            )

        other_conds = [c for c in other_conds if has_other_data(c)]
        show_baseline = has_baseline_data()
        if not show_baseline and not other_conds:
            return

        # Colors: control is gray, the rest are palette colors. Baseline is also a palette color.
        palette_idx = 0
        baseline_color = CONDITION_SERIES_PALETTE[palette_idx % len(CONDITION_SERIES_PALETTE)]
        palette_idx += 1
        other_colors: List[str] = []
        for c in other_conds:
            if c.key == "control":
                other_colors.append(CONTROL_COLOR)
            else:
                other_colors.append(CONDITION_SERIES_PALETTE[palette_idx % len(CONDITION_SERIES_PALETTE)])
                palette_idx += 1

        fig, (ax0, ax1) = plt.subplots(
            1,
            2,
            figsize=BAR_FIGSIZE_MULTIMODEL,
            sharey=True,
            # Make the no-regulation panel visually compact.
            gridspec_kw={"width_ratios": [0.55, 6.45]},
        )

        # Baseline panel: single x-group, single bar.
        if show_baseline:
            # Slightly narrower than normal so it doesn't look "fat".
            bar_w0 = min(_multimodel_bar_w(1), 0.12)
            xs0, pos0 = _bar_positions(1, 1, bar_w=bar_w0, group_gap=MULTIMODEL_GROUP_GAP)
            _draw_multimodel_cells(
                ax0,
                rows=rows,
                pos=pos0,
                groups=[baseline_cond],
                group_colors=[baseline_color],
                group_scenarios=lambda _cond, _j: "baseline",
                model_list=model_list,
                dot_offsets=dot_offsets,
                model_colors=model_colors,
                bar_w=bar_w0,
                fin_levels=("none",),
            )
            ax0.set_xlim(xs0[0] - 0.22, xs0[-1] + 0.22)
            ax0.set_xticks(xs0)
            # Norms preference: legend says "No regulation", but the baseline panel is the
            # left-side "Control" reference beside the enforcement grid.
            ax0.set_xticklabels(["Control"])
        else:
            ax0.set_xticks([])

        _format_pct_axis(ax0)
        ax0.set_xlabel("")

        # Enforcement panel: standard none/low/breakeven/high groups for non-baseline conditions.
        if other_conds:
            # Keep control last (legend order preference).
            control_idx = next((i for i, c in enumerate(other_conds) if c.key == "control"), None)
            if control_idx is not None:
                ctrl = other_conds[control_idx]
                ctrl_color = other_colors[control_idx]
                other_conds = [c for i, c in enumerate(other_conds) if i != control_idx] + [ctrl]
                other_colors = [c for i, c in enumerate(other_colors) if i != control_idx] + [ctrl_color]

            n_bars1 = len(other_conds)
            bar_w1 = _multimodel_bar_w(n_bars1)
            xs1, pos1 = _bar_positions(len(FIN_LEVELS), n_bars1, bar_w=bar_w1, group_gap=MULTIMODEL_GROUP_GAP)
            _draw_multimodel_cells(
                ax1,
                rows=rows,
                pos=pos1,
                groups=other_conds,
                group_colors=other_colors,
                group_scenarios=lambda cond, j: _informational_key(cond.key, FIN_LEVELS[j]),
                model_list=model_list,
                dot_offsets=dot_offsets,
                model_colors=model_colors,
                bar_w=bar_w1,
            )
            _add_vertical_separators(ax1, pos1, n_groups=len(FIN_LEVELS))
            _multimodel_standard_axes(ax1, xs1)
        else:
            ax1.set_xticks([])

        ax1.set_ylabel("")

        # Add a divider between panels that matches _add_vertical_separators' group boundary.
        ax0.spines["right"].set_visible(False)
        ax0.plot(
            [1, 1],
            [0, 1],
            transform=ax0.transAxes,
            color="#4a4a4a",
            linewidth=2.2,
            alpha=0.78,
            zorder=6,
            solid_capstyle="butt",
            clip_on=False,
        )

        # Make panels flush: keep exactly one divider line (the baseline right spine).
        ax1.spines["left"].set_visible(False)
        ax1.tick_params(axis="y", left=False, labelleft=False)

        cond_handles: List[Patch] = []
        if show_baseline:
            cond_handles.append(
                Patch(
                    facecolor=baseline_color,
                    edgecolor=baseline_color,
                    alpha=0.25,
                    label=baseline_cond.label,
                )
            )
        cond_handles.extend(
            Patch(facecolor=c, edgecolor=c, alpha=0.25, label=cond.label)
            for cond, c in zip(other_conds, other_colors)
        )
        model_handles = _build_model_legend_handles(model_list, model_colors)
        top_rect = _add_stacked_legends(
            fig,
            condition_handles=cond_handles,
            model_handles=model_handles,
            # Force a single-row conditions legend.
            condition_ncol=len(cond_handles) if cond_handles else None,
            condition_fontsize=_LARGE_COND_FONTSIZE,
            condition_title_fontsize=_LARGE_COND_TITLE_FONTSIZE,
            large_fonts=True,
        )
        fig.tight_layout(rect=(0, 0, 1, top_rect))
        # No inter-panel whitespace: baseline sits directly beside None/Low/Breakeven/High.
        fig.subplots_adjust(wspace=0.0)
        _save(fig, out_path, fmt, dpi)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: EXP8 — Mandate vs. pressure  (standard fonts — no large_font_ctx)
# ══════════════════════════════════════════════════════════════════════════════

def _pressure_axes_present(rows: Sequence[Dict]) -> Tuple[Set[str], Set[str], Set[str]]:
    mandates: Set[str] = set()
    pressures: Set[str] = set()
    fins: Set[str] = set()
    for r in rows:
        meta = r.get("metadata")
        if isinstance(meta, dict) and str(meta.get("framing", "")) == "informational":
            if meta.get("mandate"):  mandates.add(str(meta["mandate"]))
            if meta.get("pressure"): pressures.add(str(meta["pressure"]))
            if meta.get("fin_level"): fins.add(str(meta["fin_level"]))
    if mandates and pressures and fins:
        return mandates, pressures, fins
    # Fallback: parse scenario strings
    for r in rows:
        sc = str(r.get("scenario", ""))
        if "_informational_" not in sc:
            continue
        left, fin = sc.rsplit("_informational_", 1)
        parts = left.split("_", 1)
        if len(parts) == 2:
            mandates.add(parts[0]); pressures.add(parts[1]); fins.add(fin)
    return mandates, pressures, fins


def _pressure_is_reduced_subset(rows: Sequence[Dict]) -> bool:
    mandates, _, fins = _pressure_axes_present(rows)
    if not mandates or not fins:
        return False
    return (mandates.issubset({"none", "anti_adversarial"})
            and fins.issubset({"none", "low"})
            and "anti_adversarial" in mandates)


def _pressure_order_present(rows: Sequence[Dict]) -> List[str]:
    _, pressures, _ = _pressure_axes_present(rows)
    preferred = [
        "none", "cost_pressure", "urgency", "mgr_verbal", "self_authority",
        "risk_minimize", "social_norm", "peer_escaped", "direct_override",
        "mgr_cost", "combo_auth_cost", "combo_social_risk",
    ]
    ordered = [p for p in preferred if p in pressures]
    for p in sorted(pressures):
        if p not in ordered:
            ordered.append(p)
    return ordered


_EXP8_PRESSURE_LABELS_SHORT = {
    "none": "none", "cost_pressure": "cost", "urgency": "urgency",
    "mgr_verbal": "mgr", "self_authority": "self auth", "risk_minimize": "risk",
    "social_norm": "norm", "peer_escaped": "peer", "direct_override": "override",
    "mgr_cost": "mgr+cost", "combo_auth_cost": "auth+cost", "combo_social_risk": "soc+risk",
}


def plot_pressure(
    rows: Sequence[Dict],
    out_path: str,
    models: Optional[Sequence[str]],
    pressure: str,      # legacy arg — kept for CLI compat; ignored
    fmt: str,
    dpi: int,
) -> None:
    if _pressure_is_reduced_subset(rows):
        return _plot_pressure_reduced(rows, out_path, models, fmt, dpi)

    model_list, dot_offsets, model_colors = _resolve_model_list(
        rows, models)

    mandates = [
        ("none", "No mandate"), ("soft", "Soft mandate"),
        ("firm", "Firm mandate"), ("anti_adversarial", "Anti-adversarial"),
    ]
    pressure_order = [
        "none", "cost_pressure", "urgency", "mgr_verbal", "self_authority",
        "risk_minimize", "social_norm", "peer_escaped", "direct_override",
        "combo_auth_cost", "combo_social_risk",
    ]

    def has_pressure_data(p: str) -> bool:
        return any(
            r.get("scenario") == f"{m}_{p}_informational_{fin}" and r.get("model") in model_list
            for m, _ in mandates for fin in FIN_LEVELS for r in rows
        )
    conditions = [ConditionSpec(p, _EXP8_PRESSURE_LABELS_SHORT.get(p, p)) for p in pressure_order]

    palette_idx = 0
    cond_colors: List[str] = []
    for c in conditions:
        if c.key == "none":
            cond_colors.append(CONTROL_COLOR)
        else:
            cond_colors.append(CONDITION_SERIES_PALETTE[palette_idx % len(CONDITION_SERIES_PALETTE)])
            palette_idx += 1

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.6), sharex=True, sharey=True)
    axes_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    n_bars = len(conditions)
    bar_w  = _group_bar_w(n_bars, max_halfwidth=0.46)

    for ax, (m, m_label) in zip(axes_list, mandates):
        _, pos = _bar_positions(len(FIN_LEVELS), n_bars, bar_w=bar_w)
        _draw_multimodel_cells(
            ax,
            rows=rows, pos=pos,
            groups=conditions, group_colors=cond_colors,
            group_scenarios=lambda cond, j: f"{m}_{cond.key}_informational_{FIN_LEVELS[j]}",
            model_list=model_list, dot_offsets=dot_offsets, model_colors=model_colors,
            bar_w=bar_w,
        )
        _add_vertical_separators(ax, pos, n_groups=len(FIN_LEVELS))
        ax.set_title(m_label, pad=10, fontsize=10)
        ax.set_xticks(np.arange(len(FIN_LEVELS)))
        ax.set_xticklabels(FIN_LABELS)
        _format_pct_axis(ax)

    axes[1, 0].set_xlabel("Fine Amount")
    axes[1, 1].set_xlabel("Fine Amount")
    axes[0, 0].set_ylabel("Compliance Rate")
    axes[1, 0].set_ylabel("Compliance Rate")

    pressure_handles = [
        Patch(facecolor=c, edgecolor="black", alpha=0.35, label=cond.label)
        for cond, c in zip(conditions, cond_colors)
    ]
    model_handles = _build_model_legend_handles(model_list, model_colors)
    top_rect = _add_stacked_legends(fig, condition_handles=pressure_handles, model_handles=model_handles)
    fig.tight_layout(rect=(0, 0, 1, top_rect))
    _save(fig, out_path, fmt, dpi)


def _plot_pressure_reduced(
    rows: Sequence[Dict],
    out_path: str,
    models: Optional[Sequence[str]],
    fmt: str,
    dpi: int,
) -> None:
    """Reduced-dimension pressure figure: one panel with mandate×fine bars.
    Standard fonts — no large_font_ctx.
    """
    model_list, dot_offsets, model_colors = _resolve_model_list(
        rows, models)

    fin_levels_local = ["none", "low"]
    mandates = [("none", "None"), ("anti_adversarial", "Anti-adversarial")]
    pressure_order = _pressure_order_present(rows)

    # Filter to pressures with any data
    keep: List[str] = []
    for p in pressure_order:
        ok = any(
            bool(_per_model_rates(rows, f"{m}_{p}_informational_{fin}", model_list))
            for fin in fin_levels_local for m, _ in mandates
        )
        if ok:
            keep.append(p)
    pressure_order = keep
    if not pressure_order:
        raise RuntimeError("No pressure paper-subset rows found")

    combos: List[Tuple[str, str, str]] = []
    # Legend labels are "Mandate / Fine".
    fin_label = {
        "none": FIN_LABELS[FIN_LEVELS.index("none")],
        "low":  FIN_LABELS[FIN_LEVELS.index("low")],
    }
    for mandate, mandate_lbl in mandates:
        for fin in fin_levels_local:
            combos.append((mandate, fin, f"{mandate_lbl} / {fin_label[fin]}"))

    # 4 colored bars per employee-pressure group.
    combo_colors = [
        CONDITION_SERIES_PALETTE[i % len(CONDITION_SERIES_PALETTE)]
        for i in range(len(combos))
    ]

    fig, ax = plt.subplots(1, 1, figsize=(12.6, 4.2))
    n_bars = len(combos)
    bar_w = _multimodel_bar_w(n_bars)
    xs, pos = _bar_positions(
        len(pressure_order),
        n_bars,
        bar_w=bar_w,
        group_gap=MULTIMODEL_GROUP_GAP,
    )

    for i, ((mandate, fin, _), color) in enumerate(zip(combos, combo_colors)):
        for j, p in enumerate(pressure_order):
            sc = f"{mandate}_{p}_informational_{fin}"
            per_m = _per_model_rates(rows, sc, model_list)
            vals = list(per_m.values())
            if not vals:
                continue
            x = pos[i][j]
            _draw_iqr_boxplot(
                ax,
                x=x,
                vals_pct=[100.0 * v for v in vals],
                color=color,
                width=min(0.19, float(bar_w) * 0.86),
            )
            for mm, rr in per_m.items():
                ax.scatter([x + dot_offsets.get(mm, 0.0) * 0.9], [100.0 * rr],
                           s=MODEL_DOT_SIZE, color=model_colors.get(mm, "#333333"),
                           edgecolors=MODEL_DOT_EDGE_COLOR,
                           linewidths=MODEL_DOT_EDGE_WIDTH,
                           alpha=0.95, zorder=2)

    _add_vertical_separators(ax, pos, n_groups=len(pressure_order))
    ax.set_xlim(xs[0] - MULTIMODEL_SIDE_PAD, xs[-1] + MULTIMODEL_SIDE_PAD)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [_EXP8_PRESSURE_LABELS_SHORT.get(p, p) for p in pressure_order],
        rotation=0, ha="center")
    ax.set_xlabel("Employee pressure")
    ax.set_ylabel("Compliance Rate")
    _format_pct_axis(ax)

    combo_handles = [
        Patch(facecolor=c, edgecolor=c, alpha=0.25, label=label)
        for (_, _, label), c in zip(combos, combo_colors)
    ]
    model_handles = _build_model_legend_handles(model_list, model_colors)
    top_rect = _add_stacked_legends(
        fig,
        condition_handles=combo_handles,
        model_handles=model_handles,
        condition_title="Mandate type / Fine amount",
        condition_fontsize=18.0,
        condition_title_fontsize=19.0,
        condition_ncol=4,
        legend_gap=0.045,
    )
    fig.tight_layout(rect=(0, 0, 1, top_rect))
    _save(fig, out_path, fmt, dpi)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: EXP9 — Multi-turn end-state compliance  (standard fonts — no large_font_ctx)
# ══════════════════════════════════════════════════════════════════════════════

def plot_multiturn(
    rows: Sequence[Dict],
    out_path: str,
    models: Optional[Sequence[str]],
    fmt: str,
    dpi: int,
) -> None:
    model_list, dot_offsets, model_colors = _resolve_model_list(rows, models)

    # Tactic orders & labels as requested
    pushback_tactics = ["pushback_neutral", "pushback_cost", "pushback_authority"]
    pushback_labels = ["Neutral probe", "Cost argument", "Claimed manager order"]
    challenge_tactics = ["challenge_neutral", "challenge_reg_flag", "challenge_direct"]
    challenge_labels = ["Neutral probe", "Regulatory concern raised", "Direct correction"]

    def per_model_endstate(
        tactic_key: str,
        *,
        fin_level: str,
        direction: str,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        scenario = f"informational_{fin_level}"
        for m in model_list:
            recs = [
                r for r in rows
                if r.get("model") == m
                and r.get("scenario") == scenario
                and r.get("tactic") == tactic_key
                and r.get("t2_parseable")
                and str(r.get("direction", "")).strip().lower() == direction
            ]
            if not recs:
                continue
            vals = [1.0 if r.get("t2_compliant") else 0.0 for r in recs]
            out[m] = sum(vals) / len(vals)
        return out

    # House style: fine amount on x-axis; color encodes tactic.
    # Neutral probe appears in both directions — keep the same color for both.
    palette = CONDITION_SERIES_PALETTE
    neutral_color = palette[0 % len(palette)]
    push_colors = [
        neutral_color,
        palette[1 % len(palette)],
        palette[2 % len(palette)],
    ]
    ch_colors = [
        neutral_color,
        palette[3 % len(palette)],
        palette[4 % len(palette)],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2), sharey=True)

    def draw_panel(
        ax: plt.Axes,
        *,
        tactics: Sequence[str],
        colors: Sequence[str],
        direction: str,
        title: str,
    ) -> None:
        n_bars = len(tactics)
        bar_w  = _multimodel_bar_w(n_bars)
        xs, pos = _bar_positions(len(FIN_LEVELS), n_bars, bar_w=bar_w, group_gap=MULTIMODEL_GROUP_GAP)

        for i, (tk, color) in enumerate(zip(tactics, colors)):
            for j, fin in enumerate(FIN_LEVELS):
                per_m = per_model_endstate(tk, fin_level=fin, direction=direction)
                vals = list(per_m.values())
                if not vals:
                    continue
                mean = sum(vals) / len(vals)
                x = pos[i][j]

                _draw_iqr_boxplot(
                    ax,
                    x=x,
                    vals_pct=[100.0 * v for v in vals],
                    color=color,
                    width=min(0.18, bar_w * 0.82),
                )
                for mm, rr in per_m.items():
                    ax.scatter(
                        [x + dot_offsets.get(mm, 0.0)],
                        [100.0 * rr],
                        s=MODEL_DOT_SIZE,
                        color=model_colors.get(mm, "#333333"),
                        edgecolors=MODEL_DOT_EDGE_COLOR,
                        linewidths=MODEL_DOT_EDGE_WIDTH,
                        alpha=0.95,
                        zorder=2,
                    )

        _add_vertical_separators(ax, pos, n_groups=len(FIN_LEVELS))
        _multimodel_standard_axes(ax, xs)
        ax.set_title(title, pad=10)

    draw_panel(
        axes[0],
        tactics=pushback_tactics,
        colors=push_colors,
        direction="pushback",
        title="Erosion (Initially Compliant)",
    )
    draw_panel(
        axes[1],
        tactics=challenge_tactics,
        colors=ch_colors,
        direction="challenge",
        title="Recovery (Initially Noncompliant)",
    )

    axes[0].set_ylabel("End-state compliance (%)")
    axes[1].set_ylabel("")

    # Legend: tactics (colors) + models (colors)
    # Neutral probe is shared between the two panels; show it once.
    tactic_handles = [
        Patch(facecolor=neutral_color, edgecolor=neutral_color, alpha=0.25, label="Neutral probe"),
        Patch(facecolor=push_colors[1], edgecolor=push_colors[1], alpha=0.25, label="Cost argument"),
        Patch(facecolor=push_colors[2], edgecolor=push_colors[2], alpha=0.25, label="Manager order"),
        Patch(facecolor=ch_colors[1], edgecolor=ch_colors[1], alpha=0.25, label="Regulatory concern"),
        Patch(facecolor=ch_colors[2], edgecolor=ch_colors[2], alpha=0.25, label="Direct correction"),
    ]
    model_handles = _build_model_legend_handles(model_list, model_colors)
    top_rect = _add_stacked_legends(fig, condition_handles=tactic_handles, model_handles=model_handles)
    fig.tight_layout(rect=(0, 0, 1, top_rect))
    _save(fig, out_path, fmt, dpi)


def plot_multiturn_pressure(
    rows: Sequence[Dict],
    out_path: str,
    models: Optional[Sequence[str]],
    fmt: str,
    dpi: int,
) -> None:
    """Multi-turn-on-pressure plot: pressures on the x-axis, colored by tactic.
    Standard fonts — no large_font_ctx.
    """
    model_list, dot_offsets, model_colors = _resolve_model_list(rows, models)

    def meta(r: Dict) -> Dict:
        m = r.get("metadata")
        return m if isinstance(m, dict) else {}

    # Tactic orders & labels.
    pushback_tactics = ["pushback_neutral", "pushback_cost", "pushback_authority"]
    challenge_tactics = ["challenge_neutral", "challenge_reg_flag", "challenge_direct"]

    palette = CONDITION_SERIES_PALETTE
    neutral_color = palette[0 % len(palette)]
    push_colors = [neutral_color, palette[1 % len(palette)], palette[2 % len(palette)]]
    ch_colors = [neutral_color, palette[3 % len(palette)], palette[4 % len(palette)]]

    pressures_all = _pressure_order_present(rows)
    # Keep only pressures with data for anti_adversarial mandate
    pressures = [p for p in pressures_all if any(
        r.get("t2_parseable") and str(meta(r).get("mandate", "")).strip() == "anti_adversarial" and str(meta(r).get("pressure", "")).strip() == p
        for r in rows
    )]
    if not pressures:
        return

    fin_present = {
        str(meta(r).get("fin_level", "")).strip()
        for r in rows
        if meta(r).get("fin_level")
    }
    fin_levels_local: List[str] = [f for f in ("none", "low") if f in fin_present]
    if not fin_levels_local:
        # Fallback for non-paper runs.
        fin_levels_local = [f for f in FIN_LEVELS if f in fin_present]
    if not fin_levels_local:
        return

    def per_model_endstate(*, pressure: str, fin_level: str, tactic_key: str, direction: str) -> Dict[str, float]:
        """End-state compliance per response-model (filtered to one fin level)."""
        out: Dict[str, float] = {}
        for m in model_list:
            recs = [
                r for r in rows
                if r.get("model") == m
                and str(r.get("tactic", "")) == tactic_key
                and str(r.get("direction", "")).strip().lower() == direction
                and r.get("t2_parseable")
                and str(meta(r).get("mandate", "")).strip() == "anti_adversarial"
                and str(meta(r).get("pressure", "")).strip() == pressure
                and str(meta(r).get("fin_level", "")).strip() == fin_level
            ]
            if not recs:
                continue
            vals = [1.0 if r.get("t2_compliant") else 0.0 for r in recs]
            out[m] = sum(vals) / len(vals)
        return out

    n_rows = len(fin_levels_local)
    fig_h = 4.2 if n_rows == 1 else 7.4
    fig, axes = plt.subplots(n_rows, 2, figsize=(12.6, fig_h), sharey=True, sharex=True)
    if n_rows == 1:
        axes = np.array([axes])

    def draw_axis(
        ax: plt.Axes,
        *,
        direction: str,
        tactics: Sequence[str],
        colors: Sequence[str],
        fin_level: str,
        title: str,
    ) -> None:
        n_bars = len(tactics)
        bar_w = _multimodel_bar_w(n_bars)
        xs, pos = _bar_positions(len(pressures), n_bars, bar_w=bar_w, group_gap=1.05)

        for i, (tk, color) in enumerate(zip(tactics, colors)):
            for j, pressure in enumerate(pressures):
                per_m = per_model_endstate(
                    pressure=pressure,
                    fin_level=fin_level,
                    tactic_key=tk,
                    direction=direction,
                )
                vals = list(per_m.values())
                if not vals:
                    continue
                mean = sum(vals) / len(vals)
                x = pos[i][j]

                _draw_iqr_boxplot(
                    ax,
                    x=x,
                    vals_pct=[100.0 * v for v in vals],
                    color=color,
                    width=min(0.18, bar_w * 0.82),
                )
                for mm, rr in per_m.items():
                    ax.scatter(
                        [x + dot_offsets.get(mm, 0.0)],
                        [100.0 * rr],
                        s=MODEL_DOT_SIZE,
                        color=model_colors.get(mm, "#333333"),
                        edgecolors=MODEL_DOT_EDGE_COLOR,
                        linewidths=MODEL_DOT_EDGE_WIDTH,
                        alpha=0.95,
                        zorder=2,
                    )

        _add_vertical_separators(ax, pos, n_groups=len(pressures))
        ax.set_xlim(xs[0] - 0.55, xs[-1] + 0.55)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [_EXP8_PRESSURE_LABELS_SHORT.get(p, p) for p in pressures],
            rotation=35,
            ha="right",
        )
        ax.set_xlabel("Employee pressure")
        _format_pct_axis(ax, ylabel="")
        ax.set_title(title, pad=8)

    for ri, fin in enumerate(fin_levels_local):
        draw_axis(
            axes[ri][0],
            direction="pushback",
            tactics=pushback_tactics,
            colors=push_colors,
            fin_level=fin,
            title=f"Erosion (fin={fin})",
        )
        draw_axis(
            axes[ri][1],
            direction="challenge",
            tactics=challenge_tactics,
            colors=ch_colors,
            fin_level=fin,
            title=f"Recovery (fin={fin})",
        )

    axes[0][0].set_ylabel("End-state compliance (%)")

    tactic_handles = [
        Patch(facecolor=neutral_color, edgecolor=neutral_color, alpha=0.25, label="Neutral probe"),
        Patch(facecolor=push_colors[1], edgecolor=push_colors[1], alpha=0.25, label="Cost argument"),
        Patch(facecolor=push_colors[2], edgecolor=push_colors[2], alpha=0.25, label="Manager order"),
        Patch(facecolor=ch_colors[1], edgecolor=ch_colors[1], alpha=0.25, label="Regulatory concern"),
        Patch(facecolor=ch_colors[2], edgecolor=ch_colors[2], alpha=0.25, label="Direct correction"),
    ]
    model_handles = _build_model_legend_handles(model_list, model_colors)
    top_rect = _add_stacked_legends(fig, condition_handles=tactic_handles, model_handles=model_handles)
    fig.tight_layout(rect=(0, 0, 1, top_rect))
    _save(fig, out_path, fmt, dpi)


def plot_reasoning(
    rows: Sequence[Dict],
    out_path: str,
    models: Optional[Sequence[str]],
    fmt: str,
    dpi: int,
) -> None:
    """Stacked bars: per-model breakdown of violation reasoning classes.
    Large fonts applied.
    """
    classes = ["HEDGE", "ACKNOWLEDGE", "SILENT", "MANDATE_CITE"]
    class_colors = {
        "HEDGE": CONDITION_SERIES_PALETTE[0 % len(CONDITION_SERIES_PALETTE)],
        "ACKNOWLEDGE": CONDITION_SERIES_PALETTE[1 % len(CONDITION_SERIES_PALETTE)],
        "SILENT": CONDITION_SERIES_PALETTE[2 % len(CONDITION_SERIES_PALETTE)],
        "MANDATE_CITE": CONDITION_SERIES_PALETTE[3 % len(CONDITION_SERIES_PALETTE)],
    }

    def meta(r: Dict) -> Dict:
        m = r.get("metadata")
        return m if isinstance(m, dict) else {}

    # Keep only rows with a valid reasoning class (ignore INVALID/ERROR etc).
    valid_classes = {c.upper() for c in classes}
    rows2 = [
        r for r in rows
        if str(r.get("reasoning_class", "")).strip().upper() in valid_classes
    ]
    if models:
        keep = {str(m) for m in models}
        rows2 = [r for r in rows2 if str(r.get("model", "")) in keep]
    # Split into the two datasets we want to compare side-by-side.
    def is_control(r: Dict) -> bool:
        sc = str(r.get("scenario", ""))
        if sc.startswith("control_"):
            return True
        return not bool(str(meta(r).get("pressure", "")).strip())

    controls_rows = [r for r in rows2 if is_control(r)]
    pressure_rows = [r for r in rows2 if not is_control(r)]

    def by_model(rows_in: Sequence[Dict]) -> Dict[str, List[Dict]]:
        d: Dict[str, List[Dict]] = defaultdict(list)
        for rr in rows_in:
            d[str(rr.get("model", ""))].append(rr)
        return dict(d)

    by_m_ctrl = by_model(controls_rows)
    by_m_pressure = by_model(pressure_rows)

    # Order models using canonical ordering; keep union of both panels.
    all_models = set(by_m_ctrl.keys()) | set(by_m_pressure.keys())
    model_order: List[str] = [m for m in MODEL_ORDER if m in all_models]
    for m in sorted(all_models):
        if m not in model_order:
            model_order.append(m)
    if not model_order:
        return

    # Tighter bars: slightly reduced width and reduced inter-bar spacing.
    x = np.arange(len(model_order), dtype=float) * 0.65
    width = 0.55

    fig_w = max(11.8, 0.62 * len(model_order) + 6.2)

    with _large_font_ctx():
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, 4.6), sharey=True)

        def draw(ax: plt.Axes, *, by_m: Dict[str, List[Dict]], title: str) -> None:
            bottoms = np.zeros(len(model_order), dtype=float)
            for cls in classes:
                heights: List[float] = []
                for m in model_order:
                    rs = by_m.get(m, [])
                    total = len(rs)
                    n = sum(1 for r in rs if str(r.get("reasoning_class", "")).upper() == cls)
                    heights.append(100.0 * n / total if total else 0.0)
                ax.bar(
                    x,
                    heights,
                    bottom=bottoms,
                    width=width,
                    color=class_colors[cls],
                    edgecolor="none",
                    linewidth=0.0,
                    label=cls,
                )
                bottoms += np.array(heights, dtype=float)

            ax.set_ylim(0, 100)
            ax.set_title(title, pad=8)
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_SHORT_NAMES.get(m, m) for m in model_order], rotation=35, ha="right")
            ax.set_xlabel("Response model")
            if len(x) > 0:
                ax.set_xlim(float(x[0]) - 0.45, float(x[-1]) + 0.45)
            ax.set_axisbelow(True)
            ax.grid(axis="y", which="major", linestyle="-", linewidth=0.7, alpha=0.22, color=CONTROL_COLOR)

        draw(axes[0], by_m=by_m_ctrl, title="Shared controls")
        draw(axes[1], by_m=by_m_pressure, title="Mandate \u00d7 pressure (pressure contexts)")
        axes[0].set_ylabel("Violation reasoning class share (%)", labelpad=22)
        axes[1].set_ylabel("")

        class_handles = [
            Patch(facecolor=class_colors[c], edgecolor=class_colors[c], alpha=0.9, label=c)
            for c in classes
        ]
        top_rect = _add_stacked_legends(
            fig,
            condition_handles=class_handles,
            model_handles=[],
            condition_fontsize=_LARGE_COND_FONTSIZE,
            condition_title_fontsize=_LARGE_COND_TITLE_FONTSIZE,
            large_fonts=True,
        )
        # Extra breathing room between the legend and subplot titles.
        top_rect = max(0.78, top_rect - 0.035)
        fig.tight_layout(rect=(0, 0, 1, top_rect))
        # Tighter gap between the two panels.
        fig.subplots_adjust(wspace=0.08)
        _save(fig, out_path, fmt, dpi)
    return


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: EXP_STAKES — Purchase context robustness
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_stakes_rows(rows: Sequence[Dict]) -> List[Dict]:
    """Normalise old prefix-based and new metadata-based exp_stakes formats."""
    out: List[Dict] = []
    for r in rows:
        sc = str(r.get("scenario", ""))
        meta = r.get("metadata") or {}
        stakes = meta.get("stakes") if isinstance(meta, dict) else None

        if sc.startswith("lowstakes_"):
            stakes, sc = "low", sc[len("lowstakes_"):]
        elif sc.startswith("highstakes_"):
            stakes, sc = "high", sc[len("highstakes_"):]

        rr = dict(r)
        rr["scenario"] = sc
        mm = dict(rr.get("metadata") or {})
        if stakes:
            mm["stakes"] = stakes
        rr["metadata"] = mm
        out.append(rr)
    return out


def plot_stakes(
    rows: Sequence[Dict],
    out_path: str,
    models: Optional[Sequence[str]],
    fmt: str,
    dpi: int,
) -> None:
    with _large_font_ctx():
        model_list, dot_offsets, model_colors = _resolve_model_list(rows, models)
        norm_rows = _normalize_stakes_rows(rows)

        stakes_colors = {"low": "#4dac26", "high": "#d73027"}
        conditions = [ConditionSpec("low", "Low-stakes"), ConditionSpec("high", "High-stakes")]

        n_bars  = len(conditions) + 1   # +1 for shared control
        bar_w   = _multimodel_bar_w(n_bars)
        xs, pos = _bar_positions(len(FIN_LEVELS), n_bars, bar_w=bar_w, group_gap=MULTIMODEL_GROUP_GAP)

        fig, ax = plt.subplots(figsize=BAR_FIGSIZE_MULTIMODEL)

        # Stakes conditions
        for i, cond in enumerate(conditions):
            color = stakes_colors[cond.key]
            for j, fin in enumerate(FIN_LEVELS):
                by_model_map: Dict[str, List[Dict]] = defaultdict(list)
                for r in norm_rows:
                    if (r.get("metadata", {}).get("stakes") == cond.key
                            and r.get("scenario") == f"informational_{fin}"):
                        by_model_map[str(r.get("model", "unknown"))].append(r)
                per_m = {m: v for m in model_list if (v := _comply_rate(by_model_map.get(m, []))) is not None}
                vals  = list(per_m.values())
                if not vals:
                    continue
                x = pos[i][j]
                _draw_iqr_boxplot(ax, x=x, vals_pct=[100.0 * v for v in vals],
                                  color=color, width=min(0.18, bar_w * 0.82))
                for m, r in per_m.items():
                    ax.scatter([x + dot_offsets.get(m, 0.0)], [100.0 * r],
                               s=MODEL_DOT_SIZE, color=model_colors.get(m, "#333333"),
                               edgecolors=MODEL_DOT_EDGE_COLOR,
                               linewidths=MODEL_DOT_EDGE_WIDTH,
                               alpha=0.95, zorder=2)

        # Shared control
        for j, fin in enumerate(FIN_LEVELS):
            per_m = _per_model_rates(norm_rows, f"control_informational_{fin}", model_list)
            vals  = list(per_m.values())
            if not vals:
                continue
            x = pos[-1][j]
            _draw_iqr_boxplot(ax, x=x, vals_pct=[100.0 * v for v in vals],
                              color=CONTROL_COLOR, width=min(0.18, bar_w * 0.82))
            for m, r in per_m.items():
                ax.scatter([x + dot_offsets.get(m, 0.0)], [100.0 * r],
                           s=MODEL_DOT_SIZE, color=model_colors.get(m, "#333333"),
                           edgecolors=MODEL_DOT_EDGE_COLOR,
                           linewidths=MODEL_DOT_EDGE_WIDTH,
                           alpha=0.95, zorder=2)

        _add_vertical_separators(ax, pos, n_groups=len(FIN_LEVELS))
        _multimodel_standard_axes(ax, xs)

        cond_handles = [
            Patch(facecolor=stakes_colors[c.key], edgecolor=stakes_colors[c.key], alpha=0.25, label=c.label)
            for c in conditions
        ] + [Patch(facecolor=CONTROL_COLOR, edgecolor=CONTROL_COLOR, alpha=0.25, label="Shared control")]
        model_handles = _build_model_legend_handles(model_list, model_colors)
        top_rect = _add_stacked_legends(
            fig,
            condition_handles=cond_handles,
            model_handles=model_handles,
            condition_fontsize=_LARGE_COND_FONTSIZE,
            condition_title_fontsize=_LARGE_COND_TITLE_FONTSIZE,
            large_fonts=True,
        )
        fig.tight_layout(rect=(0, 0, 1, top_rect))
        _save(fig, out_path, fmt, dpi)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: EXP10 — Urgency follow-up
# ══════════════════════════════════════════════════════════════════════════════

_EXP10_LEGAL_LEVELS = ["none", "anti_adversarial", "clause"]
_EXP10_LEGAL_LABELS = {
    "none": "No mandate", "anti_adversarial": "Anti-adversarial", "clause": "No-exemption clause",
}
_EXP10_FIN_LEVELS = ["none", "low"]


def _urgency_scenario(variant: str, legal_level: str, fin_level: str) -> str:
    if variant == "control":
        return f"control_{legal_level}_{fin_level}"
    return f"urgency_{variant}_{legal_level}_informational_{fin_level}"


def _urgency_condition_specs() -> List[ConditionSpec]:
    return [
        ConditionSpec("control",         "Control"),
        ConditionSpec("generic",         "Generic urgency †"),
        ConditionSpec("deadline",        "Deadline"),
        ConditionSpec("operational",     "Operational"),
        ConditionSpec("sponsor_meeting", "Client visit"),
    ]


def _urgency_condition_color(cond: ConditionSpec, palette_idx: int) -> Tuple[str, int]:
    if cond.key == "control":
        return CONTROL_COLOR, palette_idx
    if cond.key == "generic":
        return "#6c757d", palette_idx
    color = CONDITION_SERIES_PALETTE[palette_idx % len(CONDITION_SERIES_PALETTE)]
    return color, palette_idx + 1


def plot_urgency(
    rows: Sequence[Dict],
    legacy_rows: Sequence[Dict],   # legacy compat; data assumed already merged
    out_path: str,
    models: Optional[Sequence[str]],
    fmt: str,
    dpi: int,
) -> None:
    with _large_font_ctx():
        model_list, dot_offsets, model_colors = _resolve_model_list(rows, models)
        conditions   = _urgency_condition_specs()
        legal_levels = _EXP10_LEGAL_LEVELS

        n_bars = len(conditions)
        bar_w  = _multimodel_bar_w(n_bars)
        xs, pos = _bar_positions(len(legal_levels), n_bars, bar_w=bar_w, group_gap=MULTIMODEL_GROUP_GAP)

        fig, axes = plt.subplots(1, len(_EXP10_FIN_LEVELS), figsize=(12.6, 4.2), sharey=True)

        for ax, fin_level in zip(axes, _EXP10_FIN_LEVELS):
            palette_idx = 0
            for i, cond in enumerate(conditions):
                color, palette_idx = _urgency_condition_color(cond, palette_idx)
                for j, legal_level in enumerate(legal_levels):
                    sc    = _urgency_scenario(cond.key, legal_level, fin_level)
                    per_m = _per_model_rates(rows, sc, model_list)
                    vals  = list(per_m.values())
                    if not vals:
                        continue
                    x = pos[i][j]
                    _draw_iqr_boxplot(ax, x=x, vals_pct=[100.0 * v for v in vals],
                                      color=color, width=min(0.18, bar_w * 0.82))
                    for mm, rr in per_m.items():
                        ax.scatter([x + dot_offsets.get(mm, 0.0)], [100.0 * rr],
                                   s=MODEL_DOT_SIZE, color=model_colors.get(mm, "#333333"),
                                   edgecolors=MODEL_DOT_EDGE_COLOR,
                                   linewidths=MODEL_DOT_EDGE_WIDTH,
                                   alpha=0.95, zorder=2)

            _add_vertical_separators(ax, pos, n_groups=len(legal_levels))
            ax.set_xlim(xs[0] - MULTIMODEL_SIDE_PAD, xs[-1] + MULTIMODEL_SIDE_PAD)
            ax.set_xticks(xs)
            ax.set_xticklabels([_EXP10_LEGAL_LABELS[ll] for ll in legal_levels], rotation=15, ha="right")
            ax.set_xlabel("Legal clarification level")
            fin_lbl = FIN_LABELS[FIN_LEVELS.index(fin_level)] if fin_level in FIN_LEVELS else str(fin_level)
            ax.set_title(f"Fine Amount: {fin_lbl}", pad=8)
            _format_pct_axis(ax, ylabel="Compliance Rate" if fin_level == _EXP10_FIN_LEVELS[0] else "")

        # Build legend handles
        cond_handles = []
        palette_idx = 0
        for cond in conditions:
            color, palette_idx = _urgency_condition_color(cond, palette_idx)
            cond_handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.25, label=cond.label))
        model_handles = _build_model_legend_handles(model_list, model_colors)
        top_rect = _add_stacked_legends(
            fig,
            condition_handles=cond_handles,
            model_handles=model_handles,
            condition_fontsize=_LARGE_COND_FONTSIZE,
            condition_title_fontsize=_LARGE_COND_TITLE_FONTSIZE,
            large_fonts=True,
        )
        fig.tight_layout(rect=(0, 0, 1, top_rect))
        _save(fig, out_path, fmt, dpi)


# ══════════════════════════════════════════════════════════════════════════════
# SUBSET HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _infer_framing(row: Dict) -> Optional[str]:
    meta = row.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("framing"):
        f = _normalize_framing_name(str(meta["framing"]))
        return f if f in FRAMINGS else None
    sc = str(row.get("scenario", ""))
    for token in sc.split("_"):
        t = _normalize_framing_name(token)
        if t in FRAMINGS:
            return t
    return None


def subset_jsonl(
    *,
    in_path: str,
    out_path: str,
    keep_framings: Sequence[str] = ("informational",),
    controls: str = "informational",
    keep_unframed: bool = True,
    keep_baseline: bool = True,
) -> Tuple[int, int]:
    keep_set = {str(x).lower() for x in keep_framings}
    rows = _load_jsonl(in_path)
    kept: List[Dict] = []
    for r in rows:
        sc = str(r.get("scenario", ""))
        if sc == "baseline":
            if keep_baseline:
                kept.append(r)
            continue
        if sc.startswith("control_"):
            if controls == "drop":
                continue
            if controls == "all":
                kept.append(r)
                continue
            if _infer_framing(r) == "informational":
                kept.append(r)
            continue
        f = _infer_framing(r)
        if f is None:
            if keep_unframed:
                kept.append(r)
        elif f in keep_set:
            kept.append(r)
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows), len(kept)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

_ALL_COMMANDS = [
    "all", "subset",
    "controls",
    "wording", "wording_grouped",
    "authority",
    "peer_signals",
    "norms",
    "pressure",
    "multiturn",
    "multiturn_pressure",
    "reasoning",
    "stakes",
    "urgency",
]


def _resolve_results_path(name: str, override: Optional[str], *, results_dir: str) -> str:
    if override:
        return override
    if name not in DEFAULT_RESULTS:
        raise KeyError(f"Unknown dataset '{name}'")
    return os.path.join(results_dir, DEFAULT_RESULTS[name])


def _try_load_rows(path: str) -> Optional[List[Dict]]:
    if not os.path.exists(path):
        return None
    return _normalize_rows_for_plotting(_load_jsonl(path))


def _load_reasoning_rows(results_dir: str) -> Optional[List[Dict]]:
    """Load reasoning-classified rows from the actual classified outputs.

    Prefer the per-experiment classified files produced by the reasoning pass.
    Fall back to the legacy combined file if present.
    """
    candidates = [
        os.path.join(results_dir, "controls_classified.jsonl"),
        os.path.join(results_dir, "pressure_classified.jsonl"),
    ]
    rows: List[Dict] = []
    for path in candidates:
        if os.path.exists(path):
            rows.extend(_load_jsonl(path))
    if rows:
        return _normalize_rows_for_plotting(rows)

    legacy = os.path.join(results_dir, "reasoning_classified.jsonl")
    if os.path.exists(legacy):
        return _normalize_rows_for_plotting(_load_jsonl(legacy))
    return None


def _attach_controls_rows(rows: Sequence[Dict], *, results_dir: str) -> List[Dict]:
    """Merge shared controls rows into experiment rows (controls filter to same model set)."""
    out = list(rows)
    ctrl_path = _resolve_results_path("controls", None, results_dir=results_dir)
    ctrl_rows = _try_load_rows(ctrl_path)
    if ctrl_rows:
        base_models = {str(r.get("model", "")) for r in out}
        out.extend(r for r in ctrl_rows if str(r.get("model", "")) in base_models)
    return out


def _dataset_for_command(cmd: str) -> str:
    """Map a figure name to its dataset key in DEFAULT_RESULTS."""
    if cmd == "reasoning":
        return "reasoning_classified"
    if cmd == "wording_grouped":
        return "wording"
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description="Generate matplotlib figures (multimodel only)")
    p.add_argument("command", nargs="?", default="all", choices=_ALL_COMMANDS)
    p.add_argument("--results-dir", default=RESULTS_DIR)
    p.add_argument("--out-dir",     default=FIGURES_DIR)
    p.add_argument("--results",     default=None, help="override results JSONL path")
    p.add_argument("--models",      nargs="*", default=None,
                   help="models to include (default: all in JSONL)")
    p.add_argument("--format", default="pdf", choices=["png", "pdf", "svg"])
    p.add_argument("--dpi",    type=int, default=300)
    p.add_argument("--out",    default=None, help="output file path (single-command only)")
    p.add_argument("--pressure", default="none", help="(legacy/ignored) exp8 pressure key")
    # subset options
    p.add_argument("--subset-out", default=None)
    p.add_argument("--subset-controls", choices=["drop", "informational", "all"],
                   default="informational")
    p.add_argument("--subset-keep-unframed", action="store_true")
    p.add_argument("--subset-drop-baseline",  action="store_true")
    args = p.parse_args()

    _ensure_dir(args.out_dir)

    def out_path(suffix: str) -> str:
        return args.out or os.path.join(args.out_dir, f"{suffix}.{args.format}")

    def load(dataset: str) -> Optional[List[Dict]]:
        if dataset == "reasoning_classified":
            rows = _load_reasoning_rows(args.results_dir)
            if rows is None:
                print(f"[SKIP] missing reasoning classified files in {args.results_dir}")
            return rows
        rp = _resolve_results_path(dataset, args.results, results_dir=args.results_dir)
        rows = _try_load_rows(rp)
        if rows is None:
            print(f"[SKIP] missing {rp}")
        return rows

    def run(cmd: str) -> None:
        ds   = _dataset_for_command(cmd)
        rows = load(ds)
        if rows is None:
            return
        if cmd in ("wording", "wording_grouped"):
            rows = _attach_controls_rows(rows, results_dir=args.results_dir)

        op = out_path(cmd)

        if   cmd == "controls":
            plot_controls(rows, op, args.models, args.format, args.dpi)
        elif cmd == "wording":
            plot_wording(rows, op, args.models, args.format, args.dpi)
        elif cmd == "wording_grouped":
            plot_wording_grouped(rows, op, args.models, args.format, args.dpi)
        elif cmd == "authority":
            plot_authority(rows, op, args.models, args.format, args.dpi)
        elif cmd == "peer_signals":
            plot_peer_signals(rows, op, args.models, args.format, args.dpi)
        elif cmd == "norms":
            plot_norms(rows, op, args.models, args.format, args.dpi)
        elif cmd == "pressure":
            plot_pressure(rows, op, args.models, args.pressure, args.format, args.dpi)
        elif cmd == "multiturn":
            plot_multiturn(rows, op, args.models, args.format, args.dpi)
        elif cmd == "multiturn_pressure":
            plot_multiturn_pressure(rows, op, args.models, args.format, args.dpi)
        elif cmd == "reasoning":
            plot_reasoning(rows, op, args.models, args.format, args.dpi)
        elif cmd == "stakes":
            plot_stakes(rows, op, args.models, args.format, args.dpi)
        elif cmd == "urgency":
            legacy_path = _resolve_results_path("pressure", None, results_dir=args.results_dir)
            legacy_rows = _try_load_rows(legacy_path) or []
            plot_urgency(rows, legacy_rows, op, args.models, args.format, args.dpi)
        else:
            raise KeyError(f"Unknown command: {cmd}")

        print(f"wrote {op}")

    # ── subset mode ──────────────────────────────────────────────────────────
    if args.command == "subset":
        ds = _dataset_for_command("controls")
        rp = _resolve_results_path(ds, args.results, results_dir=args.results_dir)
        op = args.subset_out
        if not op:
            base, ext = os.path.splitext(rp)
            op = base + "_informational" + (ext or ".jsonl")
        total, kept = subset_jsonl(
            in_path=rp, out_path=op,
            keep_framings=("informational",),
            controls=args.subset_controls,
            keep_unframed=args.subset_keep_unframed,
            keep_baseline=not args.subset_drop_baseline,
        )
        print(f"subset wrote {op} ({kept}/{total} rows)")
        return

    # ── all ──────────────────────────────────────────────────────────────────
    if args.command == "all":
        if args.out:
            raise SystemExit("--out is not supported with command=all; use --out-dir instead.")
        for cmd in [
            "controls",
            "wording", "wording_grouped",
            "authority",
            "peer_signals",
            "norms",
            "pressure",
            "multiturn",
            "multiturn_pressure",
            "reasoning",
            "stakes",
            "urgency",
        ]:
            run(cmd)
        return

    # ── single command ───────────────────────────────────────────────────────
    run(args.command)


if __name__ == "__main__":
    main()