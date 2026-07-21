# PACT v1 - frontier (closed / OpenRouter) model runbook.
#
# Adds Claude Haiku 4.5, GPT-5.6 Luna, Grok 4.3, and Gemini 3 Flash Preview to
# the benchmark. These run OFF Baseten via the EXTERNAL provider layer in
# src/benchmark/batch.py (Anthropic / OpenAI / OpenRouter); the gpt-oss outcome
# EXTRACTOR still runs on Baseten (BASETEN_API_KEY), identical to every other
# model. Each model runs as its OWN invocation because the three providers have
# different rate limits and the models need different max_tokens (below).
#
# Provider routing / keys (all in .env; verified 2026-07-20):
#   claude-haiku-4-5            Anthropic  ANTHROPIC_API_KEY2 (primary), ANTHROPIC_API_KEY (fallback)
#   gpt-5.6-luna               OpenAI     OPENAI_API_KEY
#   x-ai/grok-4.3              OpenRouter OPENROUTER_API_KEY   (reasoning DISABLED: reasoning.enabled=false)
#   google/gemini-3-flash-preview OpenRouter OPENROUTER_API_KEY
#
# max_tokens (only a too-LOW cap hurts -> truncation reads as false 'unclear';
# a high cap is free for non-reasoners, billed on real output):
#   haiku  2048  - NOT a reasoning model (no extended thinking by default)
#   grok   2048  - reasoning turned OFF, so it answers directly
#   luna   8192  - GPT-5.x REASONING model (max_completion_tokens covers reasoning+answer)
#   gemini 8192  - Flash THINKS by default on real prompts
#
# Rate limits: OpenRouter (grok/gemini) gets lower concurrency; run_batch retries
# every 429 with exponential backoff, so a transient cap self-heals - these are
# conservative starting points, safe to raise if the provider is comfortable.
#
# Usage (TOMORROW - actually runs API calls):
#   .\run_frontier_models.ps1 -Model haiku      # one model, full T1->forcing->T2->T2-forcing
#   .\run_frontier_models.ps1 -Model all        # all four, sequentially
#   .\run_frontier_models.ps1 -Model grok -DryRun   # count trials, no API calls
# After the models finish, refresh the leaderboard exactly like any other model:
#   .\run_benchmark.ps1 -Judges -Aggregate -Dashboard
#
# Every step is resumable: kill and rerun the same command to continue.

param(
    [ValidateSet("haiku", "luna", "grok", "gemini", "all")]
    [string]$Model = "",
    [int]$Reps = 3,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Per-model settings: alias, unique trials-filename substring for rejudge --only,
# max_tokens, runner concurrency, and per-model rpm cap.
$CFG = [ordered]@{
    haiku  = @{ alias = "haiku";  only = "claude-haiku";            maxtok = 2048; workers = 8; rpm = 200 }
    luna   = @{ alias = "luna";   only = "gpt-5.6-luna";            maxtok = 8192; workers = 6; rpm = 120 }
    grok   = @{ alias = "grok";   only = "grok-4.3";                maxtok = 2048; workers = 6; rpm = 100 }
    gemini = @{ alias = "gemini"; only = "gemini-3-flash-preview";  maxtok = 8192; workers = 6; rpm = 100 }
}

if (-not $Model) {
    Write-Host "Specify -Model haiku|luna|grok|gemini|all. See header for details."
    Write-Host "Per-model settings:"
    foreach ($k in $CFG.Keys) {
        $c = $CFG[$k]
        Write-Host ("  {0,-7} max_tokens={1,-5} workers={2} rpm={3}  (--only '{4}')" -f `
            $k, $c.maxtok, $c.workers, $c.rpm, $c.only)
    }
    return
}

$targets = if ($Model -eq "all") { @($CFG.Keys) } else { @($Model) }

foreach ($t in $targets) {
    $c = $CFG[$t]
    Write-Host ""
    Write-Host "=========================================================="
    Write-Host (" {0}  (alias={1}, max_tokens={2}, workers={3}, rpm={4})" -f `
        $t, $c.alias, $c.maxtok, $c.workers, $c.rpm)
    Write-Host "=========================================================="

    if ($DryRun) {
        python -m src.benchmark.runner --models $c.alias --reps $Reps --no-t2 `
            --max-tokens $c.maxtok --workers $c.workers --rpm $c.rpm --dry-run
        continue
    }

    # (a) T1 only. Generation -> EXTERNAL provider (ignores BENCH_API_KEY_ENV);
    #     the gpt-oss extractor -> BASETEN (hardcoded in runner._judge_turn).
    #     BENCH_API_KEY_ENV=DEPLOYED2 is set for parity with run_benchmark.ps1 -Eval;
    #     external models bypass it, Baseten shared models (none here) would use it.
    $env:BENCH_API_KEY_ENV = "DEPLOYED2"
    python -m src.benchmark.runner --models $c.alias --reps $Reps --no-t2 `
        --max-tokens $c.maxtok --workers $c.workers --rpm $c.rpm --extract-rpm 250
    Remove-Item Env:\BENCH_API_KEY_ENV -ErrorAction SilentlyContinue

    # (b) T1 forcing - re-ask on first-attempt unclears (scoped to THIS model).
    python -m src.benchmark.rejudge --force-only --only $c.only `
        --workers 8 --extract-rpm 1000 --model-rpm $c.rpm

    # (c) T2 rebuild from the final post-forcing outcome + T2 forcing (this model only).
    python -m src.benchmark.rejudge --rerun-t2 --only $c.only `
        --workers $c.workers --extract-rpm 1000 --model-rpm $c.rpm `
        --pushback-max-tokens $c.maxtok

    Write-Host ("{0} done. Verify parity per docs/RUN_A_NEW_MODEL.md section 5." -f $t)
}

Write-Host ""
Write-Host "All requested frontier models complete. Now refresh outputs:"
Write-Host "  .\run_benchmark.ps1 -Judges       # honesty + unclear taxonomies (new models only)"
Write-Host "  .\run_benchmark.ps1 -Aggregate    # Metrics 2.0 + figures over the full panel"
Write-Host "  .\run_benchmark.ps1 -Dashboard    # rebuild pact_dashboard.html"
