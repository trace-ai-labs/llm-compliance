# ComplianceBench — session handoff (start here)

Last updated: 2026-07-13. Point Claude at this file to resume.

## TL;DR — where we are
The generation pipeline works. We just ran a 3-domain audit, found the generator
prompts produce mostly realistic, coherent scenarios, swapped the weakest
generator out, tweaked prompts from the audit, and **wiped all generated data**.
Next step is a fresh run on the new trio, then audit again.

## Current config
- **Generator trio** (`generate.DEFAULT_GENERATORS`): `glm`, `kimi`, `ultra`
  = GLM-5.2 / Kimi-K2.6 / Nemotron-3-Ultra. (DeepSeek-V4-Pro was swapped OUT this
  session — it was a poor fit: 5/12 valid, 4 full task-cascades.)
- **Honesty judge trio** (`batch.JUDGE_TRIO`): same three, leave-one-out.
- **Outcome extractor** (`batch.DEFAULT_EXTRACT_MODEL`): Kimi-K2.5 (cheap, separate role).
- Guards are the *other two trio models* per pack (OR-fail dual review). Swapping
  the trio automatically swaps the guards.
- Each scenario is authored by all 3 generators = trio redundancy. 12 scenarios
  (3 domains) x 3 = 36 packs; full 48 x 3 = 144 packs.
- Cost: ~15 components/pack x ~1.5 attempts x (1 gen + 2 guard reviews). A
  3-domain run is ~2.4k API calls; the full 48-scenario run is ~10k.

## Next steps (do this next session)
1. Re-run the same 3 audit domains on the new trio (background, resumable):
   ```
   python -m src.benchmark.generate --only \
     finance.A finance.B finance.C finance.D \
     healthcare_admin.A healthcare_admin.C healthcare_admin.D healthcare_admin.E \
     advertising.A advertising.B advertising.C advertising.D --workers 8
   ```
   NOTE: the background job got killed mid-run twice last time (likely an env
   limit on long background commands). It is resumable — just re-run the same
   command and it skips done calls. Consider one domain at a time if it keeps dying.
2. Check yield: `python -m src.benchmark.generate --status` and `--stats`.
3. Audit coherence (see method below). Compare Kimi's yield vs DeepSeek's old 5/12.
4. If good, run the remaining 9 domains (or the full 48) and freeze items:
   `python -m src.benchmark.items -o results/benchmark/items_v1.jsonl --allow-partial`
   (commit the sha256 it prints before any scoring run = pre-registration).

## How to audit (what "good" means)
Dump valid packs to UTF-8 (Windows console chokes on emoji, so write to file):
render `items.py --preview --allow-partial`, or dump neutral T1 + menu + T2 +
guard-twin per valid pack and read critically. Check each conversation for:
- **Genuine either/or** menu — a real person can't just do several at once.
- **Operationally complete compliant option** — esp. the mishap trap: fix-it
  (resend/correct) must NOT be pitted against report-it; the facts should show
  the fix is already done, leaving report-vs-move-on as the only choice.
- **Coherent metrics** — a real property of the action (time/cost/effort), never
  a bolted-on unit (the old "60 min queue time on a filing" bug).
- **No compliance gloss** — options never say what they're "for"/require/trigger.
- **Plain real-work language**, scenario-fit pressures, no em dashes, no AI tells.

## Known hard scenarios (expect flags / may need --retry-na)
- **finance.B** (expense reclassification) — options are expense categories, which
  invite compliance glosses; nobody got it fully valid last run. Sharpened no-gloss
  rule this session; recheck.
- **healthcare_admin.C** (VIP lookup) — only Nemotron passed; "compliant must cost
  more" is hard when looking someone up is instant. Needs a real handoff cost.
- **advertising.D** (Made in USA) — intrinsically low-yield (most-constrained task).

## What changed this session (see memory file for detail)
- Registry: per-scenario `local_objective` (killed the forced per-domain metric),
  `penalty_anchor` -> `real_world_stakes`, plain-English pressure catalog, em
  dashes swept (except CANARY), clear-labeled seed block.
- Guard + task prompts: decision-coherence / mishap-trap checks, no-gloss covers
  action/category options, plain-English `PRESSURE_GUIDANCE`, un-overscoped
  `peer_escaped`.
- Human view of all 48 seeds: `results/benchmark/scenario_seeds.md`.
- C/B/V/P feature codes (crispness/benefit/victim/penalty) are **removed**: they
  were hand-annotated after the fact, the scenarios were never designed around
  them, and nothing generated keyed off them. Gone from `ScenarioSeed`, the item
  schema, the HF columns, and the spec; `feature_analysis.py` is deleted.

## Key files
- `src/benchmark/registry.py` — the 48 scenario seeds + domains + pressures (source of truth).
- `src/benchmark/gen_prompts.py` — all generator + guard prompts.
- `src/benchmark/generate.py` — stage 1 generation (CLI: --only, --status, --stats, --retry-na).
- `src/benchmark/items.py` — render packs to the frozen item set.
- `src/benchmark/runner.py` — stage 3, run models under test.
- `src/benchmark/metrics.py` + `aggregate.py` — the six axes.
- Auto-memory: `.claude/.../memory/compliancebench-guard-and-menu-decisions.md`.

## Standing constraints
- `.env` holds the API key and is gitignored — never stage/commit it.
- Nothing has been committed or pushed this session; the `compliancebench` branch
  on the remote still has the OLD (pre-swap, pre-tweak) code + the old moderation
  data. Local is ahead. Ask before pushing.
