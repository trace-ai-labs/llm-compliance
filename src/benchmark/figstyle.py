"""Shared figure styling for PACT paper figures.

One place for colors, fonts, and sizing so every figure (generation-quality in
generate.py, metrics in aggregate.py) looks consistent in the paper. Convention:
NO titles on figures (LaTeX captions carry them), large axis/tick labels, a fixed
per-model color map, and one score / one correlation colormap everywhere.
"""

import matplotlib

# palette
INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#16a34a"
PURPLE = "#9333ea"
ORANGE = "#ea580c"
TEAL = "#0891b2"
GOLD = "#ca8a04"
PINK = "#be185d"
OLIVE = "#4d7c0f"
PALETTE = [BLUE, GREEN, PURPLE, ORANGE, TEAL, GOLD, PINK, OLIVE]

SCORE_CMAP = "RdYlGn"     # 0..1 scores (higher = greener = better)
CORR_CMAP = "RdBu_r"      # -1..1 correlations

# axis display names (short, title-case, no underscores) for the six metrics
AXIS_LABEL = {
    "default_compliance": "Default\ncompliance",
    "pressure_resistance": "Pressure\nresistance",
    "pushback_resistance": "Pushback\nresistance",
    "steerability": "Steerability",
    "reasoning_honesty": "Reasoning\nhonesty",
    "rule_scope_discernment": "Rule-scope\ndiscernment",
}


def use_paper_style() -> None:
    """Global rcParams for paper figures: large labels, no grid clutter, no
    titles (captions instead), consistent fonts. Call once before plotting."""
    matplotlib.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 14,
        "axes.labelsize": 17,
        "axes.titlesize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.titlesize": 17,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


_MODEL_COLORS: dict = {}


def short(model: str) -> str:
    return model.split("/")[-1] if "/" in model else model


def model_color(model: str) -> str:
    """Stable color per model across all figures (first-seen assignment)."""
    key = short(model)
    if key not in _MODEL_COLORS:
        _MODEL_COLORS[key] = PALETTE[len(_MODEL_COLORS) % len(PALETTE)]
    return _MODEL_COLORS[key]


def strip_axes(ax) -> None:
    """Remove top/right spines and lighten the rest (call per-axes)."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
