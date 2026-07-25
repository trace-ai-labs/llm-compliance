# PACT v1 - full pipeline runner (PowerShell 5.1 compatible).
#
# Requires BASETEN_API_KEY *and* DEPLOYED2 in .env. Model-under-test generation
# runs on DEPLOYED2; the gpt-oss outcome extractor runs on BASETEN. See
# docs/RUN_A_NEW_MODEL.md for the full rationale + the finalized methodology.
#
# New-model flow (frozen items already exist), in order:
#   .\run_benchmark.ps1 -Eval -EvalModels <alias> -MaxTokens 8192
#                                         # T1 -> T1-forcing -> T2 -> T2-forcing
#   .\run_benchmark.ps1 -Judges          # honesty + unclear-reason taxonomies
#   .\run_benchmark.ps1 -Aggregate       # Metrics 2.0 + axis + taxonomy figures
#   .\run_benchmark.ps1 -Dashboard       # dashboard_data.json + pact_dashboard.html
#   .\run_benchmark.ps1 -PublishResults -PushRepo you/pact-v1-results   # HF (optional)
#
# Benchmark-construction flow (rarely re-run; items are frozen/pre-registered):
#   -Clean / -Generate / -Report / -Items / -Publish (the QUESTIONS dataset)
#
# Every stage is resumable: a killed run continues with the same command.

param(
    [switch]$Clean,          # delete ALL results/benchmark data (fresh guard log etc.)
    [switch]$Generate,       # stage 1: generate scenario packs (trio cross-check)
    [switch]$Report,         # generation-quality stats + guard report + figures
    [switch]$Items,          # stage 2: render + freeze the item set
    [switch]$Eval,           # stage 3: run model(s) the FINAL way (T1+forcing+T2+T2-forcing)
    [switch]$Judges,         # stage 4: honesty + unclear taxonomies
    [switch]$Aggregate,      # stage 5: Metrics 2.0 axes + figures + taxonomy figures
    [switch]$Dashboard,      # stage 5b: rebuild dashboard_data.json + pact_dashboard.html
    [switch]$Publish,        # publish the QUESTIONS dataset (items) to HF
    [switch]$PublishResults, # publish the RESULTS (trials + leaderboard) to HF
    [string[]]$GenModels = @("glm", "kimi", "ultra"),  # generator trio
    # Finalized eval panel: 15 real models + 4 trivial gameability baselines.
    [string[]]$EvalModels = @("deepseek", "gemma", "llama", "kimi", "kimi-code",
                              "ultra", "nemotron", "gpt-oss", "qwen35", "inkling",
                              "glm4.7", "glm5", "glm", "seedoss", "qwen36",
                              "trivial:always_comply", "trivial:always_cheapest",
                              "trivial:always_escalate", "trivial:random"),
    [int]$Reps = 3,          # independent runs per (item, arm)
    [int]$Workers = 96,      # generation concurrency for the runner
    [int]$Rpm = 0,           # runner per-model rpm cap (0 = uncapped; provider handles it)
    [int]$MaxTokens = 8192,  # model-under-test output cap. 8192 is the safe default:
                             # reasoning models truncate + false-unclear at 1024, and a
                             # non-reasoner still stops early (billed on real output).
                             # See docs/RUN_A_NEW_MODEL.md section 2.
    [double]$HonestyFrac = 1.0,   # fraction of BINDING violations to honesty-judge (1.0 = all)
    [string]$PushRepo = ""   # HF repo id (needs HF_TOKEN + `pip install datasets`)
)

$ErrorActionPreference = "Stop"

if ($Clean) {
    Remove-Item results\benchmark -Recurse -Force -Confirm:$false -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force results\benchmark | Out-Null
    Write-Host "cleared results/benchmark (all past generation + guard data)"
}

if ($Generate) {
    python -m src.benchmark.generate --models @($GenModels) --workers $Workers
    python -m src.benchmark.generate --status
}

if ($Report) {
    python -m src.benchmark.generate --stats
    python -m src.benchmark.generate --guard-report --figures
    Write-Host "CSVs: results/benchmark/guard_report.csv, guard_agreement.csv"
}

if ($Items) {
    # COMMIT items_v1.jsonl's sha256 before running any model (the pre-registration).
    python -m src.benchmark.items
}

if ($Eval) {
    # Stage 3 - the FINAL methodology, same code path that produced every existing
    # model (parity). Three sub-steps:
    #   (a) runner: T1 + rerun-2 inline judge + first-pass conditional T2.
    #       Generation -> DEPLOYED2 (BENCH_API_KEY_ENV); extractor -> BASETEN (hardcoded).
    #   (b) rejudge --force-only: re-ask the model on first-attempt unclears (T1 forcing).
    #   (c) rejudge --rerun-t2: rebuild T2 from the FINAL post-forcing outcome + T2 forcing.
    # (b)/(c) are idempotent over the whole trials dir (force-only skips already-forced;
    # rerun-t2 keeps already-clean pushback), so re-running is safe. NEVER run the bare
    # `rejudge --no-force` (no --only) here - it clobbers other models' T1.
    # --no-t2: real models get T1 only here; rerun-t2 (step c) then generates the
    # pushback T2 ONLY where it is scored (comply after forcing) - no wasted first-
    # pass T2, no unused challenge-branch T2. Trivial agents keep their synthetic T2.
    $env:BENCH_API_KEY_ENV = "DEPLOYED2"
    python -m src.benchmark.runner --models @($EvalModels) --reps $Reps --no-t2 `
        --workers $Workers --rpm $Rpm --extract-rpm 250 --max-tokens $MaxTokens
    Remove-Item Env:\BENCH_API_KEY_ENV -ErrorAction SilentlyContinue   # extractor -> BASETEN for rejudge
    python -m src.benchmark.rejudge --force-only --workers 16 --extract-rpm 1000 --model-rpm 120
    python -m src.benchmark.rejudge --rerun-t2 --workers 24 --extract-rpm 1000 `
        --model-rpm 0 --pushback-max-tokens $MaxTokens
    Write-Host "eval complete. verify parity per docs/RUN_A_NEW_MODEL.md section 5."
}

if ($Judges) {
    # Stage 4 - taxonomies (trio leave-one-out; judges key-routed automatically).
    python -m src.benchmark.judges classify-honesty --frac $HonestyFrac --workers 20
    python -m src.benchmark.judges classify-unclear --frac 1.0 --workers 20
    # Optional human-kappa / awareness / judge-swap gate:
    #   python -m src.benchmark.judges sample-honesty -n 60
    #   python -m src.benchmark.judges judge-swap --judge-model k2.5
}

if ($Aggregate) {
    # Stage 5 - six axes, cross-fitted CVaR rollup + CIs, contrasts, gates, figures.
    python -m src.benchmark.aggregate --figures
    python -m src.benchmark.judges plot-taxonomy
    python -m src.benchmark.distributions   # domain x axis, pressure, domain x pressure heatmap
    Write-Host "outputs: metrics_v2.csv|.md, cells_v2.csv, contrasts_v2.csv, dist_*.csv, figures/*.png"
}

if ($Dashboard) {
    # Stage 5b - rebuild the self-contained results dashboard from the current
    # trials + metrics_v2 + taxonomy_summary. Republish pact_dashboard.html via
    # the Artifact flow to the existing URL to keep the same link.
    python -m src.benchmark.gen_dashboard_data
    python -m src.benchmark.build_dashboard
    Write-Host "dashboard -> results/benchmark/pact_dashboard.html"
}

if ($Publish) {
    # The QUESTIONS dataset (frozen items). Rarely re-run.
    if ($PushRepo) { python -m src.benchmark.publish --push $PushRepo }
    else { python -m src.benchmark.publish }
}

if ($PublishResults) {
    # The RESULTS dataset (every trial + the metrics_v2 leaderboard).
    if ($PushRepo) { python -m src.benchmark.publish_results --push $PushRepo }
    else { python -m src.benchmark.publish_results }
    Write-Host "results build: results/benchmark/hf_results/ (README.md is the dataset card)"
}

if (-not ($Clean -or $Generate -or $Report -or $Items -or $Eval -or $Judges -or $Aggregate -or $Dashboard -or $Publish -or $PublishResults)) {
    Write-Host "No stage selected. New-model flow (items already frozen):"
    Write-Host "  -Eval            run model(s) the final way (-EvalModels, -MaxTokens 8192)"
    Write-Host "  -Judges          honesty + unclear taxonomies"
    Write-Host "  -Aggregate       Metrics 2.0 + figures"
    Write-Host "  -Dashboard       rebuild pact_dashboard.html"
    Write-Host "  -PublishResults  HF results dataset (-PushRepo you/repo)"
    Write-Host "Construction flow: -Clean / -Generate / -Report / -Items / -Publish"
    Write-Host "See docs/RUN_A_NEW_MODEL.md for the full procedure + max_tokens guidance."
}
