#!/usr/bin/env python3
"""Run experiments across models.

Generate one experiment for one model, an experiment across all models, all
experiments for one model, or everything. Each call dispatches to the experiment's
own module (`python -m src.experiments.<name>`), so this stays a thin orchestrator.

Examples:
    python run.py --experiment pressure --model gemini
    python run.py --experiment all --model openai/gpt-oss-120b
    python run.py --experiment authority --model all -n 25
    python run.py --experiment all --model all --paper
"""

import argparse
import subprocess
import sys

from src.models import MODELS

# Run order matters: the multi-turn experiments read a Turn-1 file produced earlier.
SINGLE_TURN = ["controls", "wording", "authority", "peer_signals", "norms",
               "pressure", "urgency", "stakes"]
MULTI_TURN = {
    "multiturn": "results/data/controls.jsonl",          # T1 = the controls run
    "multiturn_pressure": "results/data/pressure.jsonl",  # T1 = the pressure run
}
EXPERIMENTS = SINGLE_TURN + list(MULTI_TURN)


def command(experiment, model, runs, paper):
    cmd = [sys.executable, "-m", f"src.experiments.{experiment}", "-m", model]
    if experiment in MULTI_TURN:
        cmd += ["--t1-results", MULTI_TURN[experiment]]  # multi-turn takes no -n
    else:
        cmd += ["-n", str(runs)]
    if paper:
        cmd += ["--paper"]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default="all", choices=["all", *EXPERIMENTS])
    p.add_argument("--model", default="gemini",
                   help="OpenRouter model id or alias, or 'all' for every model in src/models.py")
    p.add_argument("-n", "--runs", type=int, default=25, help="runs per scenario (single-turn only)")
    p.add_argument("--paper", action="store_true", help="restrict to the paper's condition subset")
    args = p.parse_args()

    models = MODELS if args.model == "all" else [args.model]
    experiments = EXPERIMENTS if args.experiment == "all" else [args.experiment]

    for model in models:
        for experiment in experiments:
            cmd = command(experiment, model, args.runs, args.paper)
            print(f"\n=== {experiment} · {model} ===\n$ {' '.join(cmd)}")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
