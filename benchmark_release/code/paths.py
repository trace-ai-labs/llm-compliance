"""Filesystem layout of the release. Run every command from the code/
directory, e.g. `python -m analysis.aggregate`."""
import os

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)

# Shipped data: the dataset, per-trial judged outcomes, and judge votes.
DATA = os.path.join(ROOT, "data")
PACT = os.path.join(DATA, "pact.jsonl")
TRIALS_DIR = os.path.join(DATA, "trials")
TRANSPARENCY = os.path.join(DATA, "labels", "transparency.jsonl")
UNCLEAR = os.path.join(DATA, "labels", "unclear_reasons.jsonl")

# Analysis outputs.
RESULTS = os.path.join(ROOT, "results")

# New runs write here, never into data/.
NEW_TRIALS_DIR = os.path.join(RESULTS, "new_trials")
GENERATION_DIR = os.path.join(RESULTS, "generation")
AWARENESS_DIR = os.path.join(RESULTS, "eval_awareness")
