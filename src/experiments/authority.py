#!/usr/bin/env python3
"""Institutional authority pressure.

Holds the regulation fixed and adds one authority signal to the agent's context:
  mgr_direct     a manager steers toward the cheap, non-certified vendor
  mgr_authorize  a manager grants blanket discretion over vendor choice
  board_cost     a standing board policy makes cost the top priority

Crossed with the framings and enforcement levels, plus no-authority controls.

Run: python -m src.experiments.authority -n 25 -o results/data/authority.jsonl
"""

import argparse
from src.core import *
from src.prompts.authority import AUTHORITY_FRAGMENTS


def build_scenarios(all_framings: bool = False, paper: bool = False) -> Dict[str, Scenario]:
    s = {}
    framings = selected_framings(all_framings=all_framings)

    for auth, fragment in AUTHORITY_FRAGMENTS.items():
        for framing in framings:
            for fin in FINANCIAL_LEVELS:
                prob, pen = FINANCIAL_LEVELS[fin]
                s[f"{auth}_{framing}_{fin}"] = Scenario(
                    f"{auth}_{framing}_{fin}",
                    BASE_PROMPT + fragment + make_reg_context(framing, fin), prob, pen,
                    metadata={"authority": auth, "framing": framing,
                              "fin_level": fin, "condition": "authority"})

    for framing in framings:
        for fin in FINANCIAL_LEVELS:
            prob, pen = FINANCIAL_LEVELS[fin]
            s[f"control_{framing}_{fin}"] = Scenario(
                f"control_{framing}_{fin}",
                BASE_PROMPT + make_reg_context(framing, fin), prob, pen,
                metadata={"authority": "none", "framing": framing,
                          "fin_level": fin, "condition": "control"})

    s["baseline"] = Scenario("baseline", BASE_PROMPT,
        metadata={"authority": "none", "framing": "none",
                  "fin_level": "none", "condition": "control"})

    # Paper subset: blanket authorization and board policy only.
    if paper:
        s = {k: v for k, v in s.items()
             if v.metadata["authority"] in {"mgr_authorize", "board_cost", "none"}}
    return s


def main():
    p = argparse.ArgumentParser(description="Institutional authority pressure")
    add_common_args(p)
    p.add_argument("--list", action="store_true", help="print scenarios and exit")
    args = p.parse_args()

    scenarios = build_scenarios(all_framings=args.all_framings, paper=args.paper)
    if args.list:
        for name in scenarios:
            print(" ", name)
        return

    scenarios = filter_for_shared_controls(scenarios, args.shared_controls, model=args.model)
    outfile = args.output or "results/data/authority.jsonl"
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    print(f"Institutional authority | {len(scenarios)} scenarios × {args.runs} runs → {outfile}")
    run_experiment(scenarios, args.model, get_api_key(), args.runs, outfile,
                   args.classify_reasoning, args.batch_size)


if __name__ == "__main__":
    main()
