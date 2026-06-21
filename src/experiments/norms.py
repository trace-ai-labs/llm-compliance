#!/usr/bin/env python3
"""Non-legal normative pressure.

Adds a social signal unrelated to the law and measures its effect against the
regulation alone: community activism, industry adoption, and media coverage. Also
includes norm-only baselines (the norm present with no regulation at all).

Run: python -m src.experiments.norms -n 25 -o results/data/norms.jsonl
"""

import argparse
from src.core import *
from src.prompts.norms import NORMATIVE_SIGNALS


def build_scenarios(all_framings: bool = False, paper: bool = False) -> Dict[str, Scenario]:
    s = {}
    framings = selected_framings(all_framings=all_framings)

    for norm, fragment in NORMATIVE_SIGNALS.items():
        for framing in framings:
            for fin in FINANCIAL_LEVELS:
                prob, pen = FINANCIAL_LEVELS[fin]
                s[f"{norm}_{framing}_{fin}"] = Scenario(
                    f"{norm}_{framing}_{fin}",
                    BASE_PROMPT + fragment + make_reg_context(framing, fin), prob, pen,
                    metadata={"norm": norm, "framing": framing,
                              "fin_level": fin, "condition": "normative"})

    for framing in framings:
        for fin in FINANCIAL_LEVELS:
            prob, pen = FINANCIAL_LEVELS[fin]
            s[f"control_{framing}_{fin}"] = Scenario(
                f"control_{framing}_{fin}",
                BASE_PROMPT + make_reg_context(framing, fin), prob, pen,
                metadata={"norm": "none", "framing": framing,
                          "fin_level": fin, "condition": "control"})

    # Norm present, no regulation at all.
    for norm, fragment in NORMATIVE_SIGNALS.items():
        s[f"{norm}_only"] = Scenario(f"{norm}_only", BASE_PROMPT + fragment,
            metadata={"norm": norm, "framing": "none",
                      "fin_level": "none", "condition": "norm_only"})

    s["baseline"] = Scenario("baseline", BASE_PROMPT,
        metadata={"norm": "none", "framing": "none",
                  "fin_level": "none", "condition": "control"})

    # Paper subset: norm-only baselines, plus community/industry/media at informational.
    if paper:
        keep_norms = {"community", "industry", "media"}
        s = {k: v for k, v in s.items()
             if v.metadata["condition"] in {"norm_only", "control"}
             or (v.metadata["norm"] in keep_norms and v.metadata["framing"] == "informational")}
    return s


def main():
    p = argparse.ArgumentParser(description="Non-legal normative pressure")
    add_common_args(p)
    p.add_argument("--list", action="store_true", help="print scenarios and exit")
    args = p.parse_args()

    scenarios = build_scenarios(all_framings=args.all_framings, paper=args.paper)
    if args.list:
        for name in scenarios:
            print(" ", name)
        return

    scenarios = filter_for_shared_controls(scenarios, args.shared_controls, model=args.model)
    outfile = args.output or "results/data/norms.jsonl"
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    print(f"Normative pressure | {len(scenarios)} scenarios × {args.runs} runs → {outfile}")
    run_experiment(scenarios, args.model, get_api_key(), args.runs, outfile,
                   args.classify_reasoning, args.batch_size)


if __name__ == "__main__":
    main()
