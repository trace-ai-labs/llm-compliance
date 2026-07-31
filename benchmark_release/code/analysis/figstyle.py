"""Shared matplotlib style for the paper's figures."""

import matplotlib

# fonttype 42 = embedded TrueType (AAAI rejects matplotlib's default Type 3)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

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

CORR_CMAP = "Blues"       # correlations: white at 0 -> dark blue at 1

# display names for the six axes
AXIS_LABEL = {
    "default_compliance": "Default\nCompliance",
    "pressure_resistance": "Pressure\nResistance",
    "pushback_resistance": "Pushback\nResistance",
    "steerability": "Steerability",
    "transparency": "Transparency",
    "rule_scope_discernment": "Rule-Scope\nDiscernment",
}


# display names for domain and pressure keys
DOMAIN_LABEL = {
    "advertising": "Advertising",
    "aml_kyc": "AML",
    "customer_service": "Customer Service",
    "export_controls": "Export Controls",
    "finance": "Finance",
    "gov_services": "Gov. Services",
    "healthcare_admin": "Healthcare",
    "hr_hiring": "HR/Hiring",
    "moderation": "Moderation",
    "pharma_medinfo": "Pharma",
    "privacy": "Privacy",
    "procurement": "Procurement",
}
PRESSURE_LABEL = {
    "urgency": "Urgency",
    "mgr_verbal": "Manager",
    "peer_escaped": "Peer Escaped",
    "risk_minimize": "Risk Minimized",
    "cost_pressure": "Cost Pressure",
    "false_clearance": "False Clearance",
    "fait_accompli": "Fait Accompli",
    "sympathetic_beneficiary": "Sympathetic Beneficiary",
    "responsibility_shift": "Responsibility Shift",
}


def domain_label(key: str) -> str:
    return DOMAIN_LABEL.get(key, key.replace("_", " ").title())


def pressure_label(key: str) -> str:
    return PRESSURE_LABEL.get(key, key.replace("_", " ").title())


def use_paper_style() -> None:
    """Set the paper's global rcParams. Call once before plotting."""
    matplotlib.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
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

# display names keyed by the raw model id in results/metrics.csv
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
    # judge/guard logs tag models by the last path segment only
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
