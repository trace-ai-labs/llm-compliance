"""Stage 6b — package the eval RESULTS (model runs) for Hugging Face.

Companion to `publish.py` (which ships the benchmark *questions*). This ships
what the models *did*: every trial from `results/benchmark/trials/` - one flat
row per (item, model, arm, rep) with both turns' responses, choices, and
outcomes - plus the `metrics_v2` leaderboard as a CSV. Trial rows are already
flat (all scalar fields), so no flattening is needed.

The two user-stopped partial grids (k2.5, glm5.1) were moved out of the trials
dir to `_stopped_partials/`, so they are excluded automatically. Trivial-agent
runs ARE included (they are the published gameability floor).

Usage:
  python -m src.benchmark.publish_results                    # build locally
  python -m src.benchmark.publish_results --push you/pact-v1-results
"""

import argparse
import csv
import glob
import json
import os
from collections import Counter
from typing import Dict, List

from src.benchmark.registry import BENCH_VERSION, CANARY

TRIALS_DIR = os.path.join("results", "benchmark", "trials")
METRICS_CSV = os.path.join("results", "benchmark", "metrics_v2.csv")
RESULTS_HF_DIR = os.path.join("results", "benchmark", "hf_results")

# One flat row per trial. Documented for the dataset card.
COLUMNS: Dict[str, str] = {
    "trial_id":        "unique id: {item_id}.{arm}.{rep}",
    "item_id":         "the benchmark item this trial ran (join key to the PACT items dataset)",
    "scenario_id":     "one of the 48 scenarios, e.g. 'privacy.A'",
    "domain":          "one of the 12 domains",
    "group":           "cell group: neutral | pressure | guard_nonbinding | "
                       "attack_binding | attack_nonbinding",
    "binds":           "bool - whether the rule genuinely applies to this item",
    "pressure":        "pressure/attack tag on turn 1 (family name or 'none')",
    "split":           "'dev' | 'public_test' (private holdout not published)",
    "model":           "the agent under test (provider model id, or 'trivial:<policy>')",
    "arm":             "'base' or 'anti_adversarial' (system prompt + hard_directive)",
    "rep":             "replication index (each item x model x arm run 3x at temp=1.0)",
    "t1_response":     "the model's full turn-1 reply text (final answer only; hidden "
                       "reasoning tokens are NOT captured)",
    "t1_choice":       "the menu option the turn-1 reply settled on (extracted)",
    "t1_outcome":      "comply | violate | overcomply | unclear",
    "t1_judge":        "how t1 was judged: 'rule' (deterministic) or the LLM judge id",
    "t2_script":       "'pushback' (if t1 complied), 'challenge' (if violated), or 'none'",
    "t2_message":      "the turn-2 user message actually sent, or null",
    "t2_response":     "the model's full turn-2 reply text, or null",
    "t2_choice":       "the menu option the turn-2 reply settled on, or null",
    "t2_outcome":      "comply | violate | overcomply | unclear, or null",
    "t2_judge":        "how t2 was judged, or null",
    "judge_model":     "the small LLM judge used for residual (non-deterministic) cases",
    "judge_prompt_sha": "sha256 of the judge prompt (provenance)",
    "ts":              "unix timestamp when the trial was written",
}


def load_all_trials(trials_dir: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(glob.glob(os.path.join(trials_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("excluded"):   # dropped by corpus corrections
                    continue
                rows.append(r)
    return rows


def summarize(trials: List[dict]) -> Dict:
    models = Counter(t["model"] for t in trials)
    return {
        "n_trials": len(trials),
        "models": dict(sorted(models.items())),
        "splits": dict(Counter(t.get("split", "?") for t in trials)),
        "arms": dict(Counter(t.get("arm", "?") for t in trials)),
        "reps": sorted({t.get("rep") for t in trials if t.get("rep") is not None}),
    }


def dataset_card(summary: Dict, has_leaderboard: bool) -> str:
    cols = "\n".join(f"| `{k}` | {v} |" for k, v in COLUMNS.items())
    real = {m: n for m, n in summary["models"].items()
            if not m.startswith("trivial:")}
    triv = {m: n for m, n in summary["models"].items()
            if m.startswith("trivial:")}
    model_lines = "\n".join(f"| `{m}` | {n} |" for m, n in real.items())
    triv_lines = "\n".join(f"| `{m}` | {n} |" for m, n in triv.items())
    lb = ("\nThe six-axis Metrics 2.0 leaderboard (`leaderboard.csv`) accompanies "
          "the trials: default compliance, pressure resistance, pushback "
          "resistance, steerability, reasoning honesty, and rule-scope "
          "discernment, plus the cross-fitted-CVaR scalar rollup.\n"
          if has_leaderboard else "")
    return f"""---
pretty_name: PACT {BENCH_VERSION} - eval results
license: mit
language:
- en
tags:
- safety
- compliance
- agents
- evaluation
- leaderboard
---

# PACT {BENCH_VERSION} - eval results

Model runs on the [PACT {BENCH_VERSION}](.) benchmark: **{summary['n_trials']:,}**
trials, one flat row per (item, model, arm, replication). Each row records both
conversation turns' replies, the extracted menu choice, and the judged outcome.
Join to the PACT *items* dataset on `item_id` to recover the prompts and gold
labels.
{lb}
Every item x model x arm was run **{len(summary['reps'])}x** (temperature 1.0),
so outcomes are statistical - aggregate with Wilson / bootstrap CIs, never bare
point rates. The `anti_adversarial` arm appends each item's steerability
directive to the system prompt; `base` does not.

> Only final-answer text is stored. Hidden reasoning-token traces
> (`reasoning_content`) and per-call token counts were not captured.

## Models

Real agents under test:

| model | trials |
|---|---|
{model_lines}

Trivial reference agents (the published gameability floor - if any real model
fails to beat these on an axis, that axis is broken):

| agent | trials |
|---|---|
{triv_lines}

## Columns

| column | meaning |
|---|---|
{cols}

## Load

```python
from datasets import load_dataset
ds = load_dataset("<this repo>")["train"]
```

## Canary

```
{CANARY}
```
"""


def build(trials_dir: str, metrics_csv: str, out_dir: str) -> List[dict]:
    trials = load_all_trials(trials_dir)
    if not trials:
        raise SystemExit(f"no trials in {trials_dir} - run src.benchmark.runner first")
    os.makedirs(out_dir, exist_ok=True)

    jsonl_path = os.path.join(out_dir, "trials.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in trials:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  trials: {len(trials):,} rows -> {jsonl_path}")

    # best-effort parquet (pandas + pyarrow); JSONL is always written
    try:
        import pandas as pd
        pq = os.path.join(out_dir, "trials.parquet")
        pd.DataFrame(trials).to_parquet(pq, index=False)
        print(f"  parquet -> {pq}")
    except Exception as e:
        print(f"  (parquet skipped: {e})")

    has_lb = os.path.exists(metrics_csv)
    if has_lb:
        import shutil
        dst = os.path.join(out_dir, "leaderboard.csv")
        shutil.copyfile(metrics_csv, dst)
        print(f"  leaderboard -> {dst}")
    else:
        print(f"  (no {metrics_csv} yet - run `aggregate` for the leaderboard)")

    summary = summarize(trials)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(dataset_card(summary, has_lb))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  dataset card -> {os.path.join(out_dir, 'README.md')}")
    real = sum(1 for m in summary["models"] if not m.startswith("trivial:"))
    print(f"  {real} real models, {len(summary['models']) - real} trivial agents")
    return trials


def push(trials: List[dict], repo_id: str, out_dir: str) -> None:
    try:
        from datasets import Dataset
    except ImportError:
        raise SystemExit("pip install datasets huggingface_hub - then rerun with --push")
    Dataset.from_list(trials).push_to_hub(repo_id, private=True)
    from huggingface_hub import HfApi
    api = HfApi()
    for fname in ("README.md", "leaderboard.csv", "summary.json"):
        p = os.path.join(out_dir, fname)
        if not os.path.exists(p):
            continue
        try:
            api.upload_file(path_or_fileobj=p, path_in_repo=fname,
                            repo_id=repo_id, repo_type="dataset")
        except Exception as e:  # best-effort; the trials are already up
            print(f"  ({fname} upload failed: {e})")
    print(f"pushed to https://huggingface.co/datasets/{repo_id} (private)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials-dir", default=TRIALS_DIR)
    ap.add_argument("--metrics-csv", default=METRICS_CSV)
    ap.add_argument("-o", "--out-dir", default=RESULTS_HF_DIR)
    ap.add_argument("--push", default=None, metavar="REPO_ID",
                    help="push to the Hub as a private dataset (needs `datasets` + HF_TOKEN)")
    args = ap.parse_args()

    trials = build(args.trials_dir, args.metrics_csv, args.out_dir)
    if args.push:
        push(trials, args.push, args.out_dir)


if __name__ == "__main__":
    main()
