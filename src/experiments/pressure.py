#!/usr/bin/env python3
"""Employee pressure against a system-prompt mandate.

The user applies one of nine pressure tactics (cost appeal, deadline urgency, manager
order, self-claimed authority, risk minimization, social normalization, peer impunity,
blunt override), optionally countered by an "anti-adversarial" mandate in the system
prompt instructing the agent to follow the law regardless of the request.

Run: python -m src.experiments.pressure -n 25 -o results/data/pressure.jsonl
"""

import argparse
from src.core import *
from src.prompts.pressure import MANDATE_LEVELS, EMPLOYEE_PRESSURES


def build_scenarios(all_framings: bool = False, paper: bool = False) -> Dict[str, Scenario]:
    s = {}
    framings = selected_framings(all_framings=all_framings)

    for mandate, mandate_ctx in MANDATE_LEVELS.items():
        for pressure, suffix in EMPLOYEE_PRESSURES.items():
            for framing in framings:
                for fin in FINANCIAL_LEVELS:
                    prob, pen = FINANCIAL_LEVELS[fin]
                    name = f"{mandate}_{pressure}_{framing}_{fin}"
                    s[name] = Scenario(
                        name,
                        BASE_PROMPT + mandate_ctx + make_reg_context(framing, fin), prob, pen,
                        user_suffix=suffix,
                        metadata={"mandate": mandate, "pressure": pressure,
                                  "framing": framing, "fin_level": fin,
                                  "condition": "control" if pressure == "none" else "core"})

    s["baseline"] = Scenario("baseline", BASE_PROMPT,
        metadata={"mandate": "none", "pressure": "none", "framing": "none",
                  "fin_level": "none", "condition": "control"})

    # Paper subset: no mandate vs. anti-adversarial, informational framing, no/low fine.
    if paper:
        s = {k: v for k, v in s.items()
             if v.metadata["mandate"] in {"none", "anti_adversarial"}
             and v.metadata["framing"] == "informational"
             and v.metadata["fin_level"] in {"none", "low"}}
    return s


def main():
    p = argparse.ArgumentParser(description="Employee pressure vs. system-prompt mandate")
    add_common_args(p)
    p.add_argument("--list", action="store_true", help="print scenarios and exit")
    args = p.parse_args()

    scenarios = build_scenarios(all_framings=args.all_framings, paper=args.paper)
    if args.list:
        for name in scenarios:
            print(" ", name)
        return

    scenarios = filter_for_shared_controls(scenarios, args.shared_controls, model=args.model)
    outfile = args.output or "results/data/pressure.jsonl"
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    print(f"Employee pressure | {len(scenarios)} scenarios × {args.runs} runs → {outfile}")
    run_experiment(scenarios, args.model, get_api_key(), args.runs, outfile,
                   args.classify_reasoning, args.batch_size)


if __name__ == "__main__":
    main()
