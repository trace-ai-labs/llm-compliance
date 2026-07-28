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
    "transparency": "Transparency",
    "reasoning_honesty": "Reasoning\nhonesty",   # legacy diagnostic (motive taxonomy)
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

# Paper display names + open/closed tag, keyed by the raw provider model id in
# metrics_v2.csv. One source of truth for figures AND the generated LaTeX table
# (src/benchmark/make_paper_assets.py imports these). (C) = closed-source,
# (O) = open-weights.
DISPLAY_NAME = {
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "moonshotai/Kimi-K2.7-Code": "Kimi-K2.7-Code",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "Qwen3.6 27B": "Qwen3.6-27B",
    "moonshotai/Kimi-K2.6": "Kimi-K2.6",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B": "Nemotron-3-Ultra",
    "zai-org/GLM-5.2": "GLM-5.2",
    "thinkingmachines/inkling": "Inkling",
    "zai-org/GLM-5": "GLM-5",
    "gemma-4-26b": "Gemma-4-26B",
    "openai/gpt-oss-120b": "GPT-OSS-120B",
    "qwen3.5-35b-a3b": "Qwen3.5-35B",
    "MiniMaxAI/MiniMax-M2.5": "MiniMax-M2.5",
    "llama-3.3-70b-instruct": "Llama-3.3-70B",
    "zai-org/GLM-4.7": "GLM-4.7",
    "deepseek-ai/DeepSeek-V4-Pro": "DeepSeek-V4-Pro",
    "x-ai/grok-4.3": "Grok 4.3",
    "Seed-OSS-36B-Instruct": "Seed-OSS-36B",
    "nvidia/Nemotron-120B-A12B": "Nemotron-3-Super",
    "llama-3.1-8b-instruct": "Llama-3.1-8B",
    "mistral-7b-instruct": "Mistral-7B",
}
CLOSED_SOURCE = {
    "claude-haiku-4-5", "gpt-5.6-luna", "google/gemini-3-flash-preview",
    "x-ai/grok-4.3",
}


def is_closed(model: str) -> bool:
    return model in CLOSED_SOURCE


def short(model: str) -> str:
    if model in DISPLAY_NAME:
        return DISPLAY_NAME[model]
    # judge/guard logs tag models by the last path segment only; resolve those
    # to the same display name so no figure prints a raw slug
    tail = model.split("/")[-1]
    for k, v in DISPLAY_NAME.items():
        if k.split("/")[-1] == tail:
            return v
    return tail


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
