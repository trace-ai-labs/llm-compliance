# PACT: Can Enterprise AI Assistants Be Trusted Under Pressure?

Code and data supplement. This folder contains the full PACT corpus, the judged
outcome of every trial behind the paper, and the complete pipeline: scenario
generation, model evaluation, judging, metrics, and every figure and table in
the paper.

## Layout

```
benchmark_release/
├── data/                       the shipped record (inputs to all analysis)
│   ├── items.jsonl             the frozen corpus: 1,682 scenario cells
│   ├── trials/<model>.jsonl    judged outcomes: 22 models x 10,092 trials each
│   ├── labels/
│   │   ├── transparency.jsonl      per-judge transparency votes
│   │   ├── unclear_reasons.jsonl   per-judge abstention-reason votes
│   │   └── guard_log.jsonl         generation-time reviewer verdicts
│   └── eval_awareness/summary.csv  evaluation-awareness probe results
├── code/
│   ├── paths.py                where everything lives
│   ├── generation/             authoring the corpus (paper Section 3.1)
│   ├── evaluation/             running models and judging replies
│   └── analysis/               metrics, statistics, figures, tables
├── results/                    analysis outputs (CSV + LaTeX tables)
├── figures/                    regenerated paper figures
└── requirements.txt
```

The split is by role: `generation/` writes the corpus, `evaluation/` talks to
model APIs (running trials, judging replies), and `analysis/` turns the shipped
record in `data/` into every number, figure, and table in the paper without any
API call. Run every command below from `code/`:

```
cd code
pip install -r ../requirements.txt
```

## The dataset

`data/items.jsonl` is the complete benchmark: one JSON object per scenario
cell, 1,682 cells over 48 scenarios in 12 regulated domains (paper Section
3.1). Each cell is evaluated under two system-prompt modes (`base` and
`anti_adversarial`), giving the paper's 3,364 items.

```python
import json
items = [json.loads(l) for l in open("../data/items.jsonl", encoding="utf-8")]
```

Fields that matter most:

| field | meaning |
|---|---|
| `item_id`, `scenario_id`, `domain` | identity; `scenario_id` is `<domain>.<letter>` |
| `group` | `neutral`, `pressure`, `guard_nonbinding`, `attack_binding`, `attack_nonbinding` |
| `binds` | whether the embedded rule applies to this request |
| `pressure` | which of the nine pressure families the user turn carries (`none` on neutral cells) |
| `system_prompt` | the assistant's persona, targets, and standing rules |
| `hard_directive` | the compliance mandate appended in the `anti_adversarial` mode |
| `turn1` | the user request with the option menu inline |
| `options` | the menu; each option carries `name` and `compliant` |
| `gold` | the compliant / violating option names used for scoring |
| `t2_if_compliant` | the scripted pushback sent when the model complies at turn 1 |
| `canary` | contamination canary GUID (do not train on this text) |

`data/trials/` holds the judged outcome of every trial in the paper (3
replications x 2 system-prompt modes x 1,682 cells per model). Response text
is stripped for size; every scored field (`t1_outcome`, `t2_outcome`, forcing
flags, extracted choices) is present, and all metrics recompute exactly from
these rows. `data/labels/` holds the raw per-judge votes behind the
transparency axis and the abstention taxonomy.

## Reproducing the paper

The shipped `results/*.csv` were produced by the pipeline below; rerunning
regenerates them, every figure, and every table from `data/` alone (no API
calls, no network).

```
python -m analysis.aggregate         # six axes, PACTScore, bootstrap CIs, contrasts
                                     #   -> results/metrics.csv, cells.csv, contrasts.csv
python -m analysis.axis_ci           # per-axis bootstrap CIs -> results/axis_ci.csv
python -m analysis.distributions     # per-domain / per-pressure tables -> results/dist_*.csv
python -m analysis.figures           # every figure in the paper -> figures/
python -m analysis.paper_assets      # every LaTeX table -> results/tables/
python -m analysis.agreement         # classifier-agreement numbers (appendix)
```

Where each paper asset comes from:

| paper asset | produced by |
|---|---|
| leaderboard, model list, CI table, in-text macros | `analysis.paper_assets` |
| six-axis profiles (main-text strip + all 22 models) | `analysis.figures` |
| inter-axis correlation heatmap | `analysis.figures` |
| compliance vs rule-scope discernment by domain | `analysis.figures` |
| domain / pressure difficulty tables | `analysis.distributions` + `analysis.paper_assets` |
| transparency taxonomy (figure + per-model table) | `analysis.figures` + `analysis.paper_assets` |
| abstention taxonomy (figure + table) | `analysis.figures` + `analysis.paper_assets` |
| evaluation-awareness figure | `analysis.figures` |
| authoring-convergence figure, generation statistics | `analysis.figures`, `generation.generate --stats` |
| classifier agreement table | `analysis.agreement` |
| pairwise significance (share of separable model pairs) | `analysis.distributions` (printed), `results/contrasts.csv` |

All trials ran at temperature 1.0 with three replications per item; scores are
statistical estimates and are reported with item-cluster bootstrap confidence
intervals. Bootstrap seeds are fixed, so the intervals regenerate exactly.

## Running PACT on a new model

The pipeline talks to any OpenAI-compatible endpoint:

```
export BENCH_BASE_URL=https://<your-endpoint>/v1
export BENCH_API_KEY=...
python -m evaluation.runner --models <alias-or-model-id> --reps 3
python -m evaluation.runner --models <alias> --dry-run     # count trials, no calls
```

One trial is the full protocol of paper Section 3.1: the model answers the
request; an LLM extractor maps the reply to comply / violate / unclear; an
unclear reply gets one forcing follow-up asking it to commit to an option; a
compliant reply gets the scripted pushback turn, resolved the same way. New
trials append to `results/new_trials/` and runs are resumable (rerun the same
command after an interruption). Analyze them with
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
python -m generation.registry              # print the 48-scenario roster
python -m generation.generate              # author scenario packs (resumable)
python -m generation.generate --status     # per-scenario progress
python -m generation.items                 # render packs -> results/generation/items.jsonl
```

The generation and judge prompts (reproduced in the paper appendix) are in
`generation/gen_prompts.py` and `evaluation/judges.py`; the scenario seeds, the
nine-pressure catalog, and the compliance mandate are in
`generation/registry.py`.

## Code -> paper map

| module | paper |
|---|---|
| `generation/registry.py` | Section 3.1: domains, scenarios, the nine pressures |
| `generation/generate.py` | Section 3.1: generation and dual-reviewer audit |
| `generation/items.py` | Section 3.1: template-based item construction |
| `evaluation/runner.py` | Section 3.1: evaluation protocol |
| `evaluation/judges.py` | outcome extraction; transparency (Section 3.2, axis 5); abstention reasons |
| `evaluation/eval_awareness.py` | Section 4.3: evaluation-awareness probe |
| `analysis/metrics.py` | Section 3.2: the six axes and PACTScore |
| `analysis/aggregate.py` | Section 4: the leaderboard and its statistics |
| `analysis/axis_ci.py` | appendix: per-axis confidence intervals |
| `analysis/agreement.py` | appendix: classifier agreement |
| `analysis/distributions.py` | Section 4 + appendix: difficulty by domain and pressure |
| `analysis/figures.py` | every figure in the paper |
| `analysis/paper_assets.py` | every table in the paper |
