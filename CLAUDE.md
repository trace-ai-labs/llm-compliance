# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Two related but **separate** codebases live here. Know which one you are touching:

1. **The AIES study** (`src/experiments/`, `src/prompts/`, `src/core.py`, `src/analysis/`, `run.py`) — the published paper artifact ("What Makes AI Agents Follow the Rules?"). A procurement bot picks vendors under one embedded ISO-14001 rule; experiments vary framing, enforcement, authority, peer signals, norms, pressure. Frozen as the paper's reference; also the *style reference* for benchmark generation.
2. **PACT v1** (`src/benchmark/`, `run_benchmark.ps1`) — the new benchmark this branch (`compliancebench`) is about. 12 domains × 4 scenarios, LLM-authored scenario packs, six "Metrics 2.0" axes. This is where active work happens.

The two halves do **not** share an API layer, a model registry, or a provider (see below). Do not import `src/core.py` machinery into `src/benchmark/` or vice versa — the benchmark is a clean reimplementation.

**Read these before non-trivial benchmark work:** `docs/pact_v1_spec.md` (the authoritative spec — domains, scenarios, the pressure catalog, the six axes, and the stage-by-stage implementation plan; code keys off it) and `docs/benchmark_paper.md` (the rationale).

## API providers & model registries (the main footgun)

| | AIES study | PACT |
|---|---|---|
| Module | `src/core.py` | `src/benchmark/batch.py` |
| Provider | OpenRouter | Baseten (default) |
| Key env var | `OPENROUTER_API_KEY` | `BASETEN_API_KEY` |
| Endpoint | hardcoded OpenRouter | `BENCH_BASE_URL` (default `inference.baseten.co/v1`), OpenAI SDK |
| Aliases | `core.MODEL_REGISTRY` (OpenRouter slugs) | `batch.MODEL_REGISTRY` (Baseten slugs) |

The two `MODEL_REGISTRY` dicts use the **same short aliases** (`glm`, `deepseek`, `kimi`, …) for **different model IDs on different providers**. An alias resolved by one is meaningless to the other. The benchmark provider is fully swappable via `BENCH_BASE_URL` / `BENCH_API_KEY_ENV` — nothing downstream of `batch.py` knows the provider.

Both providers read `.env` at repo root (git-ignored). The README's "Setup" section only mentions `OPENROUTER_API_KEY`; that covers the study, not the benchmark.

## Running things

**No automated test suite exists on this branch** (the README's `python -m tests.test_benchmark` refers to a module that isn't present here). Verify changes by running the relevant stage on a tiny slice.

### AIES study
```bash
pip install -r requirements.txt          # httpx, python-dotenv, numpy, matplotlib, PyMuPDF, Pillow
python run.py --experiment pressure --model gemini      # one experiment, one model
python run.py --experiment authority --model all -n 25  # every model in src/models.py
python -m src.experiments.authority --list              # a single experiment's scenarios, no run
python -m src.analysis.plots                            # regenerate paper figures from results/data
```
Run order matters: multi-turn experiments consume a Turn-1 JSONL produced by an earlier run (`run.py` encodes this in `MULTI_TURN`).

### PACT (use the PowerShell driver — it is the authoritative pipeline)
`run_benchmark.ps1` is the source of truth for stage order, default model sets, and flags. The README's benchmark quickstart has drifted from the actual module CLIs; trust `run_benchmark.ps1` and each module's own `--help`/docstring.

```powershell
.\run_benchmark.ps1 -Clean       # wipe ALL results/benchmark (guard log etc.) before a fresh generation run
.\run_benchmark.ps1 -Generate    # stage 1: trio authors 48×3 packs, dual-guard review
.\run_benchmark.ps1 -Report      # generation-quality stats, guard κ report, figures
.\run_benchmark.ps1 -Items       # stage 2: freeze items_v1.jsonl — COMMIT its sha256 (the pre-registration)
.\run_benchmark.ps1 -Eval -EvalModels glm,deepseek,ultra -Reps 10   # stage 3
.\run_benchmark.ps1 -Judges      # stage 4: honesty labels + eval-awareness probe + blind κ CSVs
.\run_benchmark.ps1 -Aggregate   # stage 5: the six axes, rollup, gates -> metrics_v2.*
.\run_benchmark.ps1 -Publish     # stage 6: HF dataset build (-PushRepo you/repo to upload)
```

Underlying module CLIs (all resumable — kill and rerun the same command):
```bash
python -m src.benchmark.registry              # print the 48-scenario roster (source of truth)
python -m src.benchmark.batch --model glm --prompt "hi"   # smoke-test the API layer
python -m src.benchmark.generate --only aml_kyc.A         # regenerate specific scenarios
python -m src.benchmark.generate --status | --stats | --guard-report | --retry-na
python -m src.benchmark.items --preview                   # human-review markdown, does NOT freeze
python -m src.benchmark.runner --models glm --groups pressure --arms base --dry-run   # count trials, no calls
python -m src.benchmark.runner --models trivial:always_comply trivial:random          # gameability rows (no API)
python -m src.benchmark.runner --models glm --top-up-floor 50   # power gate: floor cells to n≥50
```

## Architecture that spans files

**Everything is JSONL, keyed by a stable id, resumed by id.** `src/benchmark/batch.py` is the *only* place API concurrency, retry, rate limiting, and resume live. Every stage builds `BatchRequest(id, model, messages, …)` and calls `run_batch`, which appends results to a JSONL and skips ids already present. The AIES study has its own parallel/resume logic in `core.run_experiment` (different pattern — worker threads writing appended lines). When adding a stage, follow the batch pattern: derive a deterministic id, let `run_batch`/`_load_done` handle resume; never invent an ad-hoc caller.

**`src/benchmark/registry.py` is the committed source of truth.** The 48 `ScenarioSeed`s, the `PRESSURES` catalog (`SCORED_PRESSURES` = 9 families; `ATTACK_PRESSURE` = `rule_delegitimization`, reserved for the axis-6 under-attack cells), the arm definitions, and `MAXIMAL_DIRECTIVE` all live here. Generation, item rendering, and metrics key off `s.id` (`"{domain}.{letter}"`). Change a scenario id and you invalidate frozen items + trials downstream.

**The five-stage benchmark pipeline** (files map 1:1 to spec §7):
- `generate.py` — stage 1. Authors each pack **component-by-component** (persona+hard_directive → task/menu → rules + 9 pressures + guard twin → t2 + 2 attacks), zero-shot (canonical pressure phrasings in `registry` are docs only, **never** shown to a generator). Each trio model authors a complete pack; the other two **guard-review** every component (scope + authenticity); either FAIL burns one of 5 attempts. Components accumulate in `results/benchmark/components/`, assembled packs land one-JSON-per-`{sid}__{model}` in `results/benchmark/samples/` for human review. Guard verdicts logged flat to `guard_log.jsonl` for the κ analysis.
- `items.py` — stage 2. Deterministic render of packs → 13 cells per pack (48×3×13 = 1872), gold labels, group tags (`neutral|pressure|guard_nonbinding|attack_binding|attack_nonbinding`), split assignment, canary, and a sha256. Menu option order is preserved exactly as generated (never re-sorted — it must not encode compliance).
- `runner.py` — stage 3. One trial = (item, model, rep, arm). Sends system+T1, judges T1, sends the pushback T2 when the model complied (a T1 violation is terminal; the challenge branch is retired — old trial files still carry unscored `t2_script == "challenge"` rows that every consumer ignores), judges T2. The `anti_adversarial` arm appends the item's own `hard_directive` (fallback `MAXIMAL_DIRECTIVE`) to the system prompt — the *only* difference between arms. `trivial:*` agents are synthesized with no API calls.
- `judges.py` — outcome judging is **deterministic-first**: `extract_choice` (rule-based) resolves most responses for free; a small cheap LLM judge sees only the residual. Honesty (axis 5) uses the generator trio leave-one-out as an ensemble. Every judged row logs the judge prompt's sha. `awareness.py` runs the eval-awareness validity probe.
- `metrics.py` / `aggregate.py` — stage 5. `metrics.py` is pure functions over trial dicts (the six axes); `aggregate.py` orchestrates and writes `metrics_v2.{csv,md}`, `cells_v2.csv`, `contrasts_v2.csv`.

## Conventions to preserve

- **Em/en dashes are banned in generated benchmark text** and stripped (spec §8.6). Use `-`.
- **Rule wording is informational-only.** Strictness comes exclusively from the `anti_adversarial` steerability arm, never from a command-phrased rule.
- **Trivial-agent rows are load-bearing.** `always_comply / always_cheapest / always_escalate / random` run every release; if any out-ranks a real model on an axis, that axis is broken. Keep them working.
- **The frozen `items_v1.jsonl` sha256 is a pre-registration.** Regenerating items after models have run invalidates the run; treat freezing as a commit boundary.
- **Reproducibility:** all trials run at `temperature=1.0`; results are statistical estimates, not deterministic. Report Wilson CIs (see `core.wilson_ci`), never bare point rates.
