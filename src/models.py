"""The twelve evaluated models, grouped by training orientation.

Group I models are general-purpose, safety-tuned assistants; Group II models are
optimized for agentic task performance. The grouping is assigned a priori from each
developer's stated training emphasis and predicts compliance behavior throughout the
study. Identifiers are OpenRouter model strings.
"""

SAFETY_TUNED = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.5-flash-02-23",
    "meta-llama/llama-4-maverick",
]

TASK_OPTIMIZED = [
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

MODELS = SAFETY_TUNED + TASK_OPTIMIZED
