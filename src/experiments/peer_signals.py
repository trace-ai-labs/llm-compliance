#!/usr/bin/env python3
"""Peer enforcement outcomes.

Tells the agent what a comparable peer did and what happened to them:
  peer_fined      used a non-certified vendor and was fined
  peer_escaped    used a non-certified vendor and was not caught
  peer_compliant  chose a certified vendor

Crossed with framing and enforcement, plus a no-signal control. These signals move
compliance in both directions.

Run: python -m src.experiments.peer_signals -n 25 -o results/data/peer_signals.jsonl
"""

import argparse
from src.core import *
from src.prompts.peer_signals import SOCIAL_SIGNALS


def build_scenarios(all_framings: bool = False, paper: bool = False) -> Dict[str, Scenario]:
    s = {}
    framings = selected_framings(all_framings=all_framings)

    for signal, suffix in SOCIAL_SIGNALS.items():
        for framing in framings:
            for fin in FINANCIAL_LEVELS:
                prob, pen = FINANCIAL_LEVELS[fin]
                s[f"{signal}_{framing}_{fin}"] = Scenario(
                    f"{signal}_{framing}_{fin}",
                    BASE_PROMPT + make_reg_context(framing, fin), prob, pen,
                    user_suffix=suffix,
                    metadata={"social": signal, "framing": framing,
                              "fin_level": fin, "condition": "social"})

    for framing in framings:
        for fin in FINANCIAL_LEVELS:
            prob, pen = FINANCIAL_LEVELS[fin]
            s[f"control_{framing}_{fin}"] = Scenario(
                f"control_{framing}_{fin}",
                BASE_PROMPT + make_reg_context(framing, fin), prob, pen,
                metadata={"social": "none", "framing": framing,
                          "fin_level": fin, "condition": "control"})

    s["baseline"] = Scenario("baseline", BASE_PROMPT,
        metadata={"social": "none", "framing": "none",
                  "fin_level": "none", "condition": "control"})

    if paper:
        keep = {"peer_escaped", "peer_fined", "peer_compliant", "none"}
        s = {k: v for k, v in s.items() if v.metadata["social"] in keep}
    return s


def main():
    p = argparse.ArgumentParser(description="Peer enforcement outcomes")
    add_common_args(p)
    p.add_argument("--list", action="store_true", help="print scenarios and exit")
    args = p.parse_args()

    scenarios = build_scenarios(all_framings=args.all_framings, paper=args.paper)
    if args.list:
        for name in scenarios:
            print(" ", name)
        return

    scenarios = filter_for_shared_controls(scenarios, args.shared_controls, model=args.model)
    outfile = args.output or "results/data/peer_signals.jsonl"
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    print(f"Peer signals | {len(scenarios)} scenarios × {args.runs} runs → {outfile}")
    run_experiment(scenarios, args.model, get_api_key(), args.runs, outfile,
                   args.classify_reasoning, args.batch_size)


if __name__ == "__main__":
    main()
