#!/usr/bin/env python3
"""Regime scatter plots: where each model sits on two failure axes.

Produces four views from the controls, pressure, and urgency runs:
  - fragility.pdf        framing sensitivity vs. enforcement sensitivity
  - regime_pressure.pdf  baseline compliance vs. urgency resilience
  - regime_mandate.pdf   baseline compliance vs. mandate recovery
  - regime_combined.pdf  the three views in one 1x3 panel

Models are colored by identity and shaped by training group (I = circle, II = square).

Run: python -m src.analysis.plots_2d
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── Resolve paths relative to this script ────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = str(_SCRIPT_DIR / "../../results/data")
FIGURES_DIR = str(_SCRIPT_DIR / "../../results/figures")

# ── Paper-wide rcParams ───────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 15,
    "axes.titlesize": 19,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

# ── Regime taxonomy ───────────────────────────────────────────────────────────

# Canonical model order.
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

MODEL_REGIMES: Dict[str, str] = {
    "openai/gpt-oss-120b":              "I",
    "qwen/qwen3.5-flash-02-23":         "I",
    "meta-llama/llama-4-maverick":      "I",
    "x-ai/grok-4.1-fast":               "II",
    "deepseek/deepseek-v3.2":           "II",
    "google/gemini-3-flash-preview":    "II",
    "moonshotai/kimi-k2.5":             "II",
    "z-ai/glm-4.7-flash":              "II",
    "mistralai/mistral-small-2603":     "II",
    "minimax/minimax-m2.7":             "II",
    "google/gemma-4-31b-it":            "II",
    "nvidia/nemotron-3-super-120b-a12b": "II",
}

# Regime → marker shape
REGIME_MARKERS: Dict[str, str] = {
    "I":   "o",   # circle
    "II":  "s",   # square
}

REGIME_COLORS: Dict[str, str] = {
    "I":   "#2ca02c",
    "II":  "#ff7f0e",
}

# Per-model colors (colorblind-safe palette).
MODEL_COLORS: Dict[str, str] = {
    "openai/gpt-oss-120b":              "#0072B2",
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
    "nvidia/nemotron-3-super-120b-a12b": "#88CCEE",
}

MODEL_SHORT: Dict[str, str] = {
    "openai/gpt-oss-120b":              "GPT-OSS",
    "qwen/qwen3.5-flash-02-23":         "Qwen 3.5",
    "meta-llama/llama-4-maverick":      "Llama 4",
    "x-ai/grok-4.1-fast":               "Grok 4.1",
    "deepseek/deepseek-v3.2":           "DeepSeek",
    "google/gemini-3-flash-preview":    "Gemini 3",
    "moonshotai/kimi-k2.5":             "Kimi K2.5",
    "z-ai/glm-4.7-flash":              "GLM 4.7",
    "mistralai/mistral-small-2603":     "Mistral Sm.",
    "minimax/minimax-m2.7":             "Minimax M2",
    "google/gemma-4-31b-it":            "Gemma 4",
    "nvidia/nemotron-3-super-120b-a12b": "Nemotron 3 Su",
}

# Per-model label nudges (dx_pct, dy_pct) in axis-percentage units — default (right/above).
LABEL_NUDGE: Dict[str, Tuple[float, float]] = {
    "GPT-OSS":       ( 2.5,  2.5),
    "Qwen 3.5":      ( 2.5, -6.0),
    "Llama 4":       (-9.0,  2.5),
    "Grok 4.1":      ( 2.5,  2.5),
    "DeepSeek":      ( 2.5, -6.0),
    "Gemini 3":      ( 4.5,  2.0),
    "Kimi K2.5":     (-11.0, 2.5),
    "GLM 4.7":       ( 2.5,  2.5),
    "Mistral Sm.":   ( 2.5, -7.0),
    "Minimax M2":    ( 2.5,  2.5),
    "Gemma 4":       (-15.0, 4.0),
    "Nemotron 3 Su": (2.5, 2.5),
}

# Upper-left nudges for the pressure scatter plot.
LABEL_NUDGE_UL: Dict[str, Tuple[float, float]] = {
    "GPT-OSS":       (-13.0,  2.0),
    "Qwen 3.5":      ( -8.0,  1.0),
    "Llama 4":       ( -8.0, -5.0),
    "Grok 4.1":      (-12.0,  3.0),
    "DeepSeek":      (  2.5, -7.0),
    "Gemini 3":      (-12.0,  3.0),
    "Kimi K2.5":     (-11.0,  5.0),
    "GLM 4.7":       (-10.0,  3.0),
    "Mistral Sm.":   (-16.0,  3.0),
    "Minimax M2":    (  2.5,  3.0),
    "Gemma 4":       (  2.5,  3.0),
    "Nemotron 3 Su": (2.5, 2.5),
}

# Custom offsets for mandate scatter.
LABEL_NUDGE_MANDATE: Dict[str, Tuple[float, float]] = {
    "GPT-OSS":       ( 2.5,  2.0),
    "Qwen 3.5":      (-15.0, -7.0),
    "Llama 4":       (-12.0,  2.5),
    "Grok 4.1":      ( 2.5,  2.5),
    "DeepSeek":      ( 2.5, -7.0),
    "Gemini 3":      ( 2.5,  1.5),
    "Kimi K2.5":     (-21.0,  2.5),
    "GLM 4.7":       ( 2.5,  2.5),
    "Mistral Sm.":   ( 6.0,  4.0),
    "Minimax M2":    ( 2.5,  2.5),
    "Gemma 4":       (-19.0,  2.5),
    "Nemotron 3 Su": (2.5, 2.5),
}

# ── JSONL helpers ─────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    rows.append(json.loads(s))
                except json.JSONDecodeError:
                    pass
    return rows


def _normalize_scenario(sc: str) -> str:
    aliases = {"descriptive": "informational", "permissive": "discretionary"}
    parts = sc.strip().split("_")
    return "_".join(aliases.get(p, p) for p in parts)


def _comply_rate(rows: List[Dict]) -> Optional[float]:
    parseable = [r for r in rows if r.get("parseable", True)]
    if not parseable:
        return None
    return sum(1 for r in parseable if r.get("compliant")) / len(parseable)


def _comply_rate_for(rows: List[Dict], model: str, scenario: str) -> Optional[float]:
    subset = [
        r for r in rows
        if r.get("model") == model
        and _normalize_scenario(str(r.get("scenario", ""))) == scenario
        and r.get("parseable", True)
    ]
    if not subset:
        return None
    return sum(1 for r in subset if r.get("compliant")) / len(subset)


def _available_models(rows: List[Dict]) -> List[str]:
    return sorted({str(r.get("model", "")) for r in rows if r.get("model")})


def _canonical_model_order(models: List[str]) -> List[str]:
    """Return models ordered by MODEL_ORDER, with unknowns appended alphabetically."""
    seen = set(models)
    ordered = [m for m in MODEL_ORDER if m in seen]
    extras = sorted([m for m in models if m not in set(MODEL_ORDER)])
    return ordered + extras


# ── Scalar computation ────────────────────────────────────────────────────────

def compute_scalars(
    controls_rows: List[Dict],
    pressure_rows: List[Dict],
) -> Dict[str, Dict[str, Optional[float]]]:
    IMP_NONE_KEYS = [
        "control_directive_none",
        "control_directive_informational_none",
        "control_requires_none",
    ]

    all_models_set = {
        str(r.get("model", ""))
        for r in controls_rows + pressure_rows
        if r.get("model") and str(r.get("model", "")) in MODEL_REGIMES
    }
    all_models = _canonical_model_order(list(all_models_set))

    out: Dict[str, Dict] = {}
    for m in all_models:
        imp_none = None
        for k in IMP_NONE_KEYS:
            imp_none = _comply_rate_for(controls_rows, m, k)
            if imp_none is not None:
                break

        inf_none = _comply_rate_for(controls_rows, m, "control_informational_none")
        inf_low  = _comply_rate_for(controls_rows, m, "control_informational_low")

        urg_none = _comply_rate_for(pressure_rows, m, "none_urgency_informational_low")
        urg_anti = _comply_rate_for(pressure_rows, m, "anti_adversarial_urgency_informational_low")

        if urg_none is None:
            urg_none = _comply_rate_for(pressure_rows, m, "urgency_generic_none_informational_low")
        if urg_anti is None:
            urg_anti = _comply_rate_for(pressure_rows, m, "urgency_generic_anti_adversarial_informational_low")

        framing_sens     = (imp_none - inf_none) if (imp_none is not None and inf_none is not None) else None
        enforcement_sens = (inf_none - inf_low)  if (inf_none is not None and inf_low  is not None) else None
        mandate_recovery = (urg_anti - urg_none) if (urg_anti is not None and urg_none is not None) else None

        out[m] = {
            "imp_none":                imp_none,
            "inf_none":                inf_none,
            "inf_low":                 inf_low,
            "framing_sensitivity":     framing_sens,
            "enforcement_sensitivity": enforcement_sens,
            "urgency_no_mandate":      urg_none,
            "urgency_anti_adv":        urg_anti,
            "mandate_recovery":        mandate_recovery,
        }

    return out


# ── PDF cropping ──────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, out_path: str, fmt: str, dpi: int) -> None:
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, format=fmt, dpi=dpi)
    plt.close(fig)
    if fmt == "pdf":
        try:
            from src.analysis._crop import crop_pdf as _crop
            _crop(Path(out_path))
        except Exception as e:
            print(f"[WARN] PDF crop skipped: {e}")


# ── Shared scatter helpers ────────────────────────────────────────────────────

def _scatter_point(
    ax: plt.Axes,
    x: float,
    y: float,
    model_id: str,
    *,
    size: float = 180,      # was 160 — slightly larger
    zorder: int = 5,
) -> None:
    regime = MODEL_REGIMES.get(model_id, "II")
    color  = MODEL_COLORS.get(model_id, "#333333")
    marker = REGIME_MARKERS.get(regime, "o")
    ax.scatter(
        [x], [y],
        c=[color],
        marker=marker,
        s=size,
        zorder=zorder,
        edgecolors="white",
        linewidths=1.2,     # was 1.0 — slightly thicker white outline
    )


def _annotate_point(
    ax: plt.Axes,
    x: float,
    y: float,
    model_id: str,
    *,
    nudge_dict: Optional[Dict[str, Tuple[float, float]]] = None,
    fontsize: float = 13.0,
) -> None:
    label = MODEL_SHORT.get(model_id, model_id)
    color = MODEL_COLORS.get(model_id, "#333333")
    nd = nudge_dict if nudge_dict is not None else LABEL_NUDGE
    dx, dy = nd.get(label, (2.5, 2.5))
    xt, yt = x + dx, y + dy
    # Draw a subtle leader line only when the label is displaced meaningfully
    dist = math.sqrt(dx**2 + dy**2)
    ap = dict(arrowstyle="-", color=color, lw=0.5, alpha=0.45) if dist > 6 else None
    ann = ax.annotate(
        label,
        xy=(x, y),
        xytext=(xt, yt),
        fontsize=fontsize,
        color=color,
        fontweight="semibold",
        arrowprops=ap,
        zorder=6,
    )
    ann.set_clip_on(False)


def _style_scatter_ax(
    ax: plt.Axes,
    *,
    x_label: str,
    y_label: str,
    title: str = "",
    xlim: Tuple[float, float] = (-5, 105),
    ylim: Tuple[float, float] = (-5, 105),
    midline: bool = True,
    diagonal: bool = False,
    label_fontsize: float = 18,
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=19, pad=9)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticklabels([f"{v}%" for v in [0, 25, 50, 75, 100]])
    ax.set_yticklabels([f"{v}%" for v in [0, 25, 50, 75, 100]])
    ax.set_axisbelow(True)
    ax.grid(axis="both", linestyle="-", linewidth=0.5, alpha=0.18, color="#888888")

    if midline:
        ax.axvline(50, color="#bbbbbb", linewidth=1.0, linestyle="--", zorder=1)
        ax.axhline(50, color="#bbbbbb", linewidth=1.0, linestyle="--", zorder=1)

    if diagonal:
        lo = max(xlim[0], ylim[0])
        hi = min(xlim[1], ylim[1])
        ax.plot([lo, hi], [lo, hi], color="#bbbbbb", linewidth=1.0,
                linestyle="--", zorder=1, label="_nolegend_")


def _quadrant_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    ha: str = "left",
    va: str = "bottom",
) -> None:
    ax.text(
        x, y, text,
        fontsize=12.5,
        color="#9a9a9a",
        ha=ha, va=va,
        style="italic",
        transform=ax.transData,
        zorder=0,
    )


def _spacer() -> Line2D:
    return Line2D([0], [0], color="none", label=" ")


def _regime_handle(r: str) -> Line2D:
    regime_label = {
        "I":  "Safety-aligned",
        "II": "Task-optimized",
    }[r]
    return Line2D(
        [0], [0],
        marker=REGIME_MARKERS[r],
        color="none",
        markerfacecolor="#555555",
        markeredgecolor="white",
        markeredgewidth=0.8,
        markersize=15,
        label=regime_label,
    )


def _model_handle(m: str) -> Line2D:
    regime = MODEL_REGIMES.get(m, "II")
    marker = REGIME_MARKERS.get(regime, "o")   # match the shape used in the scatter
    return Line2D(
        [0], [0],
        marker=marker,
        color="none",
        markerfacecolor=MODEL_COLORS.get(m, "#333333"),
        markeredgecolor="none",     # no outline — consistent with scatter dots
        markersize=13,
        label=MODEL_SHORT[m],
    )


def _build_legend_handles() -> List:
    """Regime shapes then per-model colors, single flat list for side legends."""
    handles = [_regime_handle(r) for r in ("I", "II")]
    handles.append(_spacer())
    model_order = _canonical_model_order(list(MODEL_REGIMES.keys()))
    for m in model_order:
        if m in MODEL_SHORT:
            handles.append(_model_handle(m))
    return handles


def _build_legend_handles_two_rows() -> List:
    """Build handles for a two-row legend.

    Row 1: Regime I, Regime II, [spacers] — pads to exactly ncol
    Row 2: all models in canonical order — fits exactly in one row
    """
    model_order = _canonical_model_order(list(MODEL_REGIMES.keys()))
    model_row = [_model_handle(m) for m in model_order if m in MODEL_SHORT]

    ncol = max(1, len(model_row))
    regime_row = [_regime_handle(r) for r in ("I", "II")]
    while len(regime_row) < ncol:
        regime_row.append(_spacer())

    return regime_row + model_row


def _add_stacked_legends(
    fig: plt.Figure,
    *,
    condition_handles: List,
    model_handles: List,
    condition_title: Optional[str] = None,
    condition_ncol: int,
    model_ncol: int,
    condition_fontsize: float = 15,
    model_fontsize: float = 14,
    condition_title_fontsize: float = 16,
    legend_gap: float = 0.012,
    below_legend_pad: float = 0.015,
) -> float:
    """Add two stacked legends at the top of a figure.

    Matches the shared paper figure style:
    - conditions legend (frameless) above
    - models legend (framed) below
    """
    if not condition_handles and not model_handles:
        return 1.0

    cond_leg = None
    if condition_handles:
        cond_leg = fig.legend(
            handles=list(condition_handles),
            title=(condition_title or None),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            bbox_transform=fig.transFigure,
            ncol=max(1, int(condition_ncol)),
            frameon=False,
            fontsize=condition_fontsize,
            handlelength=1.2,
            handletextpad=0.5,
            columnspacing=1.2,
            borderpad=0.0,
        )
        if condition_title and hasattr(cond_leg, "get_title"):
            try:
                cond_leg.get_title().set_fontsize(condition_title_fontsize)
                cond_leg.get_title().set_fontweight("semibold")
            except Exception:
                pass

    if not model_handles:
        # Reserve space for condition legend only.
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            if cond_leg is not None:
                bbox = cond_leg.get_window_extent(renderer=renderer)
                bbox_fig = fig.transFigure.inverted().transform_bbox(bbox)
                return max(0.0, float(bbox_fig.y0) - float(below_legend_pad))
        except Exception:
            pass
        return 0.92

    # Keep model legend to <= model_ncol per row (typically 6 => two rows of 6).
    max_per_row = max(1, int(model_ncol))
    m_rows = max(1, math.ceil(len(model_handles) / max_per_row))
    m_ncol = math.ceil(len(model_handles) / m_rows)

    # Place model legend directly under the condition legend using the rendered bbox.
    y_model_anchor = 1.0 - float(legend_gap)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        if cond_leg is not None:
            cond_bbox = cond_leg.get_window_extent(renderer=renderer)
            cond_bbox_fig = fig.transFigure.inverted().transform_bbox(cond_bbox)
            y_model_anchor = float(cond_bbox_fig.y0) - float(legend_gap)
    except Exception:
        # Fallback to a conservative anchor.
        y_model_anchor = 0.90

    model_leg = fig.legend(
        handles=list(model_handles),
        loc="upper center",
        bbox_to_anchor=(0.5, y_model_anchor),
        bbox_transform=fig.transFigure,
        ncol=max(1, int(m_ncol)),
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        fontsize=model_fontsize,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=1.0,
        borderpad=0.7,
    )

    # Return a tight_layout top-rect just under the model legend.
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = model_leg.get_window_extent(renderer=renderer)
        bbox_fig = fig.transFigure.inverted().transform_bbox(bbox)
        return max(0.0, float(bbox_fig.y0) - float(below_legend_pad))
    except Exception:
        # Heuristic fallback.
        return 0.88


# ── Figure 1: Framing sensitivity vs Enforcement sensitivity ──────────────────

def plot_sensitivity_scatter(
    scalars: Dict[str, Dict],
    out_path: str,
    fmt: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.6))   # slightly taller than original

    plotted = 0
    for model_id in _canonical_model_order(list(scalars.keys())):
        d = scalars[model_id]
        x = d.get("framing_sensitivity")
        y = d.get("enforcement_sensitivity")
        if x is None or y is None:
            continue
        xp, yp = x * 100, y * 100
        _scatter_point(ax, xp, yp, model_id)
        _annotate_point(ax, xp, yp, model_id, nudge_dict=LABEL_NUDGE_MANDATE)
        plotted += 1

    _style_scatter_ax(
        ax,
        x_label="Framing Sensitivity (%)",
        y_label="Enforcement Sensitivity (%)",
        midline=True,
    )

    _quadrant_label(ax,  2,  2, "stable",             ha="left",  va="bottom")
    _quadrant_label(ax, 98,  2, "framing-dependent",  ha="right", va="bottom")
    _quadrant_label(ax,  2, 98, "penalty-paradox",    ha="left",  va="top")
    _quadrant_label(ax, 98, 98, "doubly fragile",     ha="right", va="top")

    if plotted == 0:
        ax.text(50, 50, "No data found — check --results-dir",
                ha="center", va="center", fontsize=15, color="#cc0000")

    legend_handles = _build_legend_handles()
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        fontsize=14,
        handlelength=1.2,
        handletextpad=0.5,
        borderpad=0.7,
    )

    fig.tight_layout()
    _save(fig, out_path, fmt, dpi)
    print(f"wrote {out_path}")


# ── Figure 2: Baseline compliance vs urgency resilience ───────────────────────

def plot_pressure_scatter(
    scalars: Dict[str, Dict],
    out_path: str,
    fmt: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.4))

    plotted = 0
    for model_id in _canonical_model_order(list(scalars.keys())):
        d = scalars[model_id]
        x = d.get("inf_none")
        y = d.get("urgency_no_mandate")
        if x is None or y is None:
            continue
        xp, yp = x * 100, y * 100
        _scatter_point(ax, xp, yp, model_id)
        _annotate_point(ax, xp, yp, model_id, nudge_dict=LABEL_NUDGE_UL)
        plotted += 1

    _style_scatter_ax(
        ax,
        x_label="Baseline Compliance (%)",
        y_label="Compliance Under Urgency, No Mandate (%)",
        midline=True,
    )

    _quadrant_label(ax,  2,  2, "never compliant",                    ha="left",  va="bottom")
    _quadrant_label(ax, 98,  2, "compliant but folds\nunder urgency", ha="right", va="bottom")
    _quadrant_label(ax,  2, 98, "(logically sparse)",                  ha="left",  va="top")
    _quadrant_label(ax, 98, 98, "robust",                              ha="right", va="top")

    if plotted == 0:
        ax.text(50, 50, "No urgency data found — check --results-dir",
                ha="center", va="center", fontsize=15, color="#cc0000")

    legend_handles = _build_legend_handles()
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        fontsize=14,
        handlelength=1.2,
        handletextpad=0.5,
        borderpad=0.7,
    )

    fig.tight_layout()
    _save(fig, out_path, fmt, dpi)
    print(f"wrote {out_path}")


# ── Figure 3: Mandate recovery scatter ───────────────────────────────────────

def plot_mandate_recovery_scatter(
    scalars: Dict[str, Dict],
    out_path: str,
    fmt: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.4))

    ax.plot(
        [0, 100], [0, 100],
        color="#cccccc", linewidth=1.1, linestyle=":",
        zorder=1,
    )
    ax.text(
        30, 33, "mandate preserved baseline",
        fontsize=12, color="#8f8f8f", style="italic",
        rotation=42, ha="left", va="bottom",
    )

    ax.fill_between(
        [0, 100], [0, 100], [100, 100],
        alpha=0.04, color="#2ca02c", zorder=0,
    )
    ax.text(
        5, 88, "mandate helped ↑",
        fontsize=13, color="#2ca02c", alpha=0.6, style="italic",
    )

    plotted = 0
    for model_id in _canonical_model_order(list(scalars.keys())):
        d = scalars[model_id]
        x = d.get("inf_none")
        y = d.get("urgency_anti_adv")
        if x is None or y is None:
            continue
        xp, yp = x * 100, y * 100
        _scatter_point(ax, xp, yp, model_id)
        _annotate_point(ax, xp, yp, model_id)
        plotted += 1

    _style_scatter_ax(
        ax,
        x_label="Baseline Compliance (%)",
        y_label="Urgency Compliance, Anti-Adversarial Mandate (%)",
        midline=False,
        diagonal=False,
    )

    _quadrant_label(ax,  2,  2, "low baseline;\nlow mandated",       ha="left",  va="bottom")
    _quadrant_label(ax, 98,  2, "baseline high,\nbut folds",         ha="right", va="bottom")
    _quadrant_label(ax,  2, 98, "baseline low,\nbut mandate lifts",  ha="left",  va="top")
    _quadrant_label(ax, 98, 98, "baseline-like\nunder mandate",      ha="right", va="top")

    if plotted == 0:
        ax.text(50, 50, "No urgency data found — check --results-dir",
                ha="center", va="center", fontsize=15, color="#cc0000")

    legend_handles = _build_legend_handles()
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        fontsize=14,
        handlelength=1.2,
        handletextpad=0.5,
        borderpad=0.7,
    )

    fig.tight_layout()
    _save(fig, out_path, fmt, dpi)
    print(f"wrote {out_path}")


# ── Figure 4: Combined 1×3 panel ─────────────────────────────────────────────

def plot_combined_panel(
    scalars: Dict[str, Dict],
    out_path: str,
    fmt: str,
    dpi: int,
) -> None:
    """
    1×3 panel combining all three scatter views.
    Moderate height keeps the combined panel compact.
    """
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18.0, 5.6),
        gridspec_kw={"wspace": 0.35},
    )   # square panels

    # Per-panel nudge dicts — tuned for where data actually clusters in each panel.
    # Panel 1 (framing vs enforcement): models spread across the space, use LABEL_NUDGE_MANDATE.
    # Panel 2 (baseline vs urgency/no-mandate): most models at x≈85-100%, y≈0-30% — fan outward.
    # Panel 3 (baseline vs urgency/anti-adv): most models at x≈85-100%, y≈30-100% — fan left/down.
    NUDGE_P1 = LABEL_NUDGE_MANDATE
    NUDGE_P2: Dict[str, Tuple[float, float]] = {
        "GPT-OSS":       (-13.0,  3.0),
        "Qwen 3.5":      ( -8.0,  3.0),
        "Llama 4":       ( -8.0, -6.0),
        "Grok 4.1":      ( -8.0,  8.0),
        "DeepSeek":      (  3.0, -7.0),
        "Gemini 3":      (-13.0, -6.0),
        "Kimi K2.5":     (  3.0,  4.0),
        "GLM 4.7":       (-10.0, -6.0),
        "Mistral Small": (-15.0,  3.0),
        "MiniMax M2.7":  (-14.0,  8.0),
        "Gemma 4":       ( -8.0, -6.0),
    }
    NUDGE_P3: Dict[str, Tuple[float, float]] = {
        "GPT-OSS":       (  3.0,  3.0),
        "Qwen 3.5":      (-14.0,  4.0),
        "Llama 4":       ( -9.0, -6.0),
        "Grok 4.1":      ( -9.0,  3.0),
        "DeepSeek":      (  3.0, -6.0),
        "Gemini 3":      (-12.0,  3.0),
        "Kimi K2.5":     (  3.0,  3.0),
        "GLM 4.7":       (-10.0, -6.0),
        "Mistral Small": (-15.0, -6.0),
        "MiniMax M2.7":  (-15.0,  3.0),
        "Gemma 4":       (-13.0,  4.0),
    }

    panels = [
        dict(
            ax=axes[0],
            x_key="framing_sensitivity",
            y_key="enforcement_sensitivity",
            x_label="Framing Sensitivity (%)",
            y_label="Enforcement Sensitivity (%)",
            diagonal=False,
            nudge=NUDGE_P1,
            quadrants=[
                ( 2,  2, "stable",           "left",  "bottom"),
                (98,  2, "framing-dep.",      "right", "bottom"),
                ( 2, 98, "penalty-paradox",   "left",  "top"),
                (98, 98, "doubly fragile",    "right", "top"),
            ],
        ),
        dict(
            ax=axes[1],
            x_key="inf_none",
            y_key="urgency_no_mandate",
            x_label="Baseline Compliance (%)",
            y_label="Urgency Compliance,\nNo Mandate (%)",
            diagonal=False,
            nudge=NUDGE_P2,
            quadrants=[
                ( 2,  2, "never\ncompliant",    "left",  "bottom"),
                (98,  2, "folds under\nurgency", "right", "bottom"),
                (98, 98, "robust",               "right", "top"),
            ],
        ),
        dict(
            ax=axes[2],
            x_key="inf_none",
            y_key="urgency_anti_adv",
            x_label="Baseline Compliance (%)",
            y_label="Urgency Compliance,\nAnti-Adv. Mandate (%)",
            diagonal=True,
            nudge=NUDGE_P3,
            quadrants=[
                ( 2,  2, "low baseline;\nlow mandated",   "left",  "bottom"),
                ( 2, 98, "baseline low,\nbut lifted",     "left",  "top"),
                (98, 98, "baseline-like\nunder mandate",  "right", "top"),
            ],
        ),
    ]

    for spec in panels:
        ax = spec["ax"]

        if spec["diagonal"]:
            ax.plot([0, 100], [0, 100], color="#cccccc",
                    linewidth=1.0, linestyle=":", zorder=1)
            # "mandate preserved" diagonal label
            # ax.text(28, 32, "mandate preserved baseline",
            #         fontsize=12, color="#8f8f8f", style="italic",
                    # rotation=42, ha="left", va="bottom")

        plotted = 0
        for model_id in _canonical_model_order(list(scalars.keys())):
            d = scalars[model_id]
            x = d.get(spec["x_key"])
            y = d.get(spec["y_key"])
            if x is None or y is None:
                continue
            xp = x * 100 if x <= 1.0 else x
            yp = y * 100 if y <= 1.0 else y
            _scatter_point(ax, xp, yp, model_id, size=150)
            # Combined panel: no on-plot model labels (legend only).
            plotted += 1

        _style_scatter_ax(
            ax,
            x_label=spec["x_label"],
            y_label=spec["y_label"],
            midline=True,
            diagonal=False,
            label_fontsize=20,
        )
        ax.set_aspect("equal", adjustable="box")   # force square data area

        for qx, qy, qlabel, ha, va in spec["quadrants"]:
            _quadrant_label(ax, qx, qy, qlabel, ha=ha, va=va)

        if plotted == 0:
            ax.text(50, 50, "no data", ha="center", va="center",
                    fontsize=14, color="#cc0000")

    condition_handles = [_regime_handle(r) for r in ("I", "II")]
    model_order = _canonical_model_order(list(MODEL_REGIMES.keys()))
    model_handles = [_model_handle(m) for m in model_order if m in MODEL_SHORT]

    # Top legend: model-group/shape encoding (bigger). Bottom legend: models.
    rect_top = _add_stacked_legends(
        fig,
        condition_handles=condition_handles,
        model_handles=model_handles,
        condition_title=None,
        condition_ncol=2,
        model_ncol=6,
        condition_fontsize=18,
        model_fontsize=17,
        condition_title_fontsize=20,
        legend_gap=0.003,
        below_legend_pad=-0.15,
    )

    # Hard cap: keep a large legend band reserved (tight_layout can still
    # under-estimate figure-level legend space when fonts are huge).
    rect_top = min(rect_top, 0.68)
    fig.tight_layout(rect=(0, 0.0, 1, rect_top), w_pad=0.85)
    fig.subplots_adjust(top=rect_top)
    _save(fig, out_path, fmt, dpi)
    print(f"wrote {out_path}")


# ── Diagnostic: print scalars to stdout ──────────────────────────────────────

def _print_scalars(scalars: Dict[str, Dict]) -> None:
    print("\n-- Computed scalars -------------------------------------------------------")
    header = f"{'Model':<38} {'Regime':>6}  {'imp/0':>6} {'inf/0':>6} {'inf/L':>6}  {'frmSens':>8} {'enfSens':>8}  {'urgNone':>8} {'urgAnti':>8} {'mandRec':>8}"
    print(header)
    print("-" * len(header))
    for m in _canonical_model_order(list(scalars.keys())):
        d = scalars[m]
        def _f(v): return f"{v*100:6.1f}" if v is not None else "     -"
        print(
            f"{MODEL_SHORT.get(m, m):<38} "
            f"{MODEL_REGIMES.get(m,'?'):>6}  "
            f"{_f(d.get('imp_none'))} "
            f"{_f(d.get('inf_none'))} "
            f"{_f(d.get('inf_low'))}  "
            f"{_f(d.get('framing_sensitivity'))} "
            f"{_f(d.get('enforcement_sensitivity'))}  "
            f"{_f(d.get('urgency_no_mandate'))} "
            f"{_f(d.get('urgency_anti_adv'))} "
            f"{_f(d.get('mandate_recovery'))}"
        )
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Regime scatter plots for compliance paper")
    p.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Directory containing combined JSONLs. "
            "Defaults to results/data."
        ),
    )
    p.add_argument(
        "--controls",
        default=None,
        help="Override path to controls.jsonl",
    )
    p.add_argument(
        "--pressure",
        default=None,
        help="Override path to pressure.jsonl (used for urgency scalars)",
    )
    p.add_argument(
        "--urgency",
        default=None,
        help=(
            "Override path to urgency.jsonl (fallback urgency source if the "
            "pressure file lacks a model's data)."
        ),
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to results/figures.",
    )
    p.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--figures",
        nargs="*",
        default=["sensitivity", "pressure", "mandate", "combined"],
        choices=["sensitivity", "pressure", "mandate", "combined"],
        help="Which figures to produce (default: all four).",
    )
    p.add_argument(
        "--print-scalars",
        action="store_true",
        help="Print computed per-model scalars to stdout before plotting.",
    )
    args = p.parse_args()

    # ── Resolve results dir ────────────────────────────────────────────────────
    results_dir = args.results_dir or RESULTS_DIR
    out_dir = args.out_dir or FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)

    fmt = args.format
    dpi = args.dpi

    # ── Load data ──────────────────────────────────────────────────────────────
    controls_path = args.controls or os.path.join(results_dir, "controls.jsonl")
    pressure_path     = args.pressure     or os.path.join(results_dir, "pressure.jsonl")
    urgency_path    = args.urgency    or os.path.join(results_dir, "urgency.jsonl")

    print(f"Loading controls from: {controls_path}")
    print(f"Loading pressure from:     {pressure_path}")
    print(f"Loading urgency from:    {urgency_path}")

    controls_rows = _load_jsonl(controls_path)
    pressure_rows     = _load_jsonl(pressure_path)
    urgency_rows    = _load_jsonl(urgency_path)

    if not controls_rows:
        print(f"[WARN] No rows loaded from controls: {controls_path}")
    if not pressure_rows and not urgency_rows:
        print("[WARN] No rows loaded from pressure or urgency. Urgency scalars will be missing.")

    merged_rows = pressure_rows + urgency_rows

    # ── Compute scalars ────────────────────────────────────────────────────────
    scalars = compute_scalars(controls_rows, merged_rows)

    n_models = sum(1 for d in scalars.values() if d.get("inf_none") is not None)
    print(f"\nFound data for {n_models} models in MODEL_REGIMES.")

    if args.print_scalars or True:
        _print_scalars(scalars)

    # ── Plot ───────────────────────────────────────────────────────────────────
    figure_fns = {
        "sensitivity": (
            plot_sensitivity_scatter,
            os.path.join(out_dir, f"fragility.{fmt}"),
        ),
        "pressure": (
            plot_pressure_scatter,
            os.path.join(out_dir, f"regime_pressure.{fmt}"),
        ),
        "mandate": (
            plot_mandate_recovery_scatter,
            os.path.join(out_dir, f"regime_mandate.{fmt}"),
        ),
        "combined": (
            plot_combined_panel,
            os.path.join(out_dir, f"regime_combined.{fmt}"),
        ),
    }

    for name in args.figures:
        fn, out_path = figure_fns[name]
        fn(scalars, out_path, fmt, dpi)

    print("\nDone.")


if __name__ == "__main__":
    main()