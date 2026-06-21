#!/usr/bin/env python3
"""Foundational compliance grid.

Runs the three framings (directive, informational, discretionary) against the four
enforcement levels (none, low, breakeven, high) at high N. Every other experiment
reuses these cells as controls. The enforcement paradox surfaces here: under a
directive rule, naming a small fine sharply lowers compliance.

Run: python -m src.experiments.controls -n 50 -o results/data/controls.jsonl
"""

import argparse
from src.core import *


def build_scenarios() -> Dict[str, Scenario]:
    s = {}

    for framing_name in RULE_FRAMINGS:
        for fin_level in FINANCIAL_LEVELS:
            prob, pen = FINANCIAL_LEVELS[fin_level]
            reg_ctx = make_reg_context(framing_name, fin_level)
            name = f"control_{framing_name}_{fin_level}"
            s[name] = Scenario(
                name,
                BASE_PROMPT + reg_ctx,
                prob, pen,
                metadata={"framing": framing_name, "fin_level": fin_level,
                          "condition": "control"}
            )

    s["baseline"] = Scenario(
        "baseline", BASE_PROMPT,
        metadata={"framing": "none", "fin_level": "none", "condition": "control"}
    )

    return s

def main():
    p = argparse.ArgumentParser(description="Shared canonical controls for cross-experiment comparison")
    add_common_args(p)
    args = p.parse_args()

    scenarios = build_scenarios()
    api_key = get_api_key()
    outfile = args.output or "results/data/controls.jsonl"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    print(f"Shared Controls | {len(scenarios)} scenarios × {args.runs} runs")
    print(f"  Output: {outfile}")
    print(f"  Tip: use -n 50 or higher for stable cross-experiment baselines\n")
    run_experiment(scenarios, args.model, api_key, args.runs, outfile,
                   args.classify_reasoning, args.batch_size)


if __name__ == "__main__":
    main()
