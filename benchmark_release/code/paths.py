"""Filesystem layout of the release. Run every command from this directory
(benchmark_release/code), e.g. `python -m analysis.aggregate`."""
import os

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)

# Shipped record: the frozen corpus, per-trial judged outcomes, and judge votes.
DATA = os.path.join(ROOT, "data")
ITEMS = os.path.join(DATA, "items.jsonl")
TRIALS_DIR = os.path.join(DATA, "trials")
TRANSPARENCY = os.path.join(DATA, "labels", "transparency.jsonl")
UNCLEAR = os.path.join(DATA, "labels", "unclear_reasons.jsonl")
GUARD_LOG = os.path.join(DATA, "labels", "guard_log.jsonl")
AWARENESS_SUMMARY = os.path.join(DATA, "eval_awareness", "summary.csv")

# Analysis outputs.
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(RESULTS, "tables")
FIGURES = os.path.join(ROOT, "figures")

# New runs write here, never over the shipped record in data/.
NEW_TRIALS_DIR = os.path.join(RESULTS, "new_trials")
GENERATION_DIR = os.path.join(RESULTS, "generation")
AWARENESS_DIR = os.path.join(RESULTS, "eval_awareness")
