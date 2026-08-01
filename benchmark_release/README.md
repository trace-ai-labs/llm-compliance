# PACT: Can Enterprise AI Assistants Be Trusted Under Pressure?

Code and data supplement. This folder contains the full PACT dataset, the
judged outcome of every trial behind the paper, and the complete pipeline:
scenario generation, model evaluation, judging, and the computation of every
number in the paper.

## Layout

```
benchmark_release/
├── data/                       everything the analysis reads
│   ├── pact.jsonl              the dataset: 3,364 samples
│   ├── trials/<model>.jsonl    judged outcomes: 22 models x 10,092 trials each
│   ├── labels/
│   │   ├── transparency.jsonl      per-judge transparency votes
│   │   └── unclear_reasons.jsonl   per-judge abstention-reason votes
│   └── eval_awareness/summary.csv  evaluation-awareness probe results
├── code/
│   ├── paths.py                where everything lives
│   ├── generation/             authoring the dataset (paper Section 3.1)
│   ├── evaluation/             running models and judging replies
│   └── analysis/               metrics and statistics
├── results/                    analysis outputs (CSV)
└── requirements.txt
```

The split is by role: `generation/` writes the dataset, `evaluation/` talks to
model APIs (running trials, judging replies), and `analysis/` turns `data/`
into every number in the paper without any API call. Run every command below
from `code/`:

```
cd code
pip install -r ../requirements.txt
```

## The dataset

`data/pact.jsonl` is the complete benchmark: one JSON object per sample, 3,364
samples over 48 scenarios in 12 regulated domains (paper Section 3.1). Each
underlying scenario cell ships as two samples that differ only in the system
prompt: `base`, and `mandate` with the scenario's hard compliance directive
appended (the paper's anti-adversarial mode). The paper counts each sample as
an item (its 3,364 items); the `item` field below names the shared scenario
cell behind a base/mandate pair.

```python
import json
samples = [json.loads(l) for l in open("../data/pact.jsonl", encoding="utf-8")]
```

| field | meaning |
|---|---|
| `id` | unique sample id (`<item>.<mode>`) |
| `item` | the underlying cell, shared by the sample's `base`/`mandate` pair |
| `mode` | `base` or `mandate` |
| `scenario`, `domain` | `scenario` is `<domain>.<letter>` |
| `group` | `neutral`, `pressure`, `guard_nonbinding`, `attack_binding`, `attack_nonbinding` |
| `binds` | whether the embedded rule applies to this request |
| `pressure` | which of the nine pressure families the user turn carries (`none` on neutral cells) |
| `rule` | the standing rule under test, in brief |
| `system_prompt` | the complete system prompt as sent (mandate samples include the directive) |
| `turn1` | the user request with the option menu inline |
| `options` | the menu; each option carries `name`, `line`, `compliant` |
| `gold` | the compliant option names used for scoring |
| `pushback` | the scripted second turn sent when the model complies at turn 1 |
| `generator_model` | which of the three generator models authored this sample's pack |
| `canary` | contamination canary GUID (do not train on this text) |

`data/trials/` holds the judged outcome of every trial in the paper (3
replications per sample per model; trial ids are `<sample>.<rep>`). Response
text is stripped for size; every scored field (`t1_outcome`, `t2_outcome`,
forcing flags, extracted choices) is present, and all metrics recompute
exactly from these rows. `data/labels/` holds the raw per-judge votes behind
the transparency axis and the abstention taxonomy.

## Computing the paper's numbers

The shipped `results/*.csv` were produced by the commands below; rerunning
regenerates them from `data/` alone (no API calls, no network) and prints the
tables to stdout. Every quantity plotted in the paper comes out of one of
these.

```
python -m analysis.aggregate         # the leaderboard: six axes, PACTScore + CIs,
                                     #   pairwise contrasts, inter-axis correlations
                                     #   -> results/metrics.csv, cells.csv, contrasts.csv
python -m analysis.axis_ci           # per-axis bootstrap CIs -> results/axis_ci.csv
python -m analysis.distributions     # difficulty by domain and by pressure family
                                     #   -> results/dist_*.csv
python -m evaluation.judges label-stats        # transparency + abstention taxonomies
```

The evaluation-awareness probe (paper Section 4.3) is the one result that
required its own model runs; its per-model comparison table ships as
`data/eval_awareness/summary.csv`. Rerunning the probe
(`python -m evaluation.eval_awareness rewrite / run / report`) makes API calls
and writes fresh outputs under `results/eval_awareness/`.

All trials ran at temperature 1.0 with three replications per sample; scores
are statistical estimates and are reported with item-cluster bootstrap
confidence intervals. Bootstrap seeds are fixed, so the intervals regenerate
exactly.

## Running PACT on a new model

All model calls go through OpenRouter:

```
export OPENROUTER_API_KEY=...
python -m evaluation.runner --models <alias-or-openrouter-slug> --reps 3
python -m evaluation.runner --models <alias> --dry-run     # count trials, no calls
```

One trial is the full protocol of paper Section 3.1: the model answers the
request; an LLM extractor maps the reply to comply / violate / unclear; an
unclear reply gets one forcing follow-up asking it to commit to an option; a
compliant reply gets the scripted pushback turn, resolved the same way. A
turn-1 violation is terminal. New trials append to `results/new_trials/`;
rerunning the same command continues an interrupted run. Score them with
`python -m analysis.aggregate --trials-dir ../results/new_trials`.

Transparency and abstention labels for new trials (the shipped labels already
cover all 22 released models) come from the judge ensemble:

```
python -m evaluation.judges classify-transparency --trials-dir ../results/new_trials
python -m evaluation.judges classify-unclear --trials-dir ../results/new_trials
```

## Generating new scenarios

Paper Section 3.1's authoring pipeline: three generator models write each
scenario component by component, and every component is reviewed by the two
models that did not write it.

```
python -m generation.registry              # list the 48 scenarios
python -m generation.generate              # author scenario packs
python -m generation.generate --status     # per-scenario progress
python -m generation.generate --stats      # call and review counts for your run
python -m generation.items                 # render packs -> results/generation/pact.jsonl
```

The generation and judge prompts (reproduced in the paper appendix) are in
`generation/gen_prompts.py` and `evaluation/judges.py`; the scenario seeds and
the nine-pressure catalog are in `generation/registry.py`.

## Code -> paper map

| module | paper |
|---|---|
| `generation/registry.py` | Section 3.1: domains, scenarios, the nine pressures |
| `generation/generate.py` | Section 3.1: generation and dual-reviewer audit |
| `generation/items.py` | Section 3.1: template-based prompt construction |
| `evaluation/runner.py` | Section 3.1: evaluation protocol |
| `evaluation/judges.py` | outcome extraction; transparency (Section 3.2, axis 5); abstention reasons |
| `evaluation/eval_awareness.py` | Section 4.3: evaluation-awareness probe |
| `analysis/metrics.py` | Section 3.2: the six axes and PACTScore |
| `analysis/aggregate.py` | Section 4: the leaderboard and its statistics |
| `analysis/axis_ci.py` | appendix: per-axis confidence intervals |
| `analysis/distributions.py` | Section 4 + appendix: difficulty by domain and pressure |
