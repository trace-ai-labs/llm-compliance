# Per-model max_tokens for the final trial set (back-derived 2026-07-27)

The runner does not store max_tokens in trial rows, so these were reconstructed
after the fact from three evidence sources: (a) the actual runner/driver
invocations recorded in Claude session transcripts, (b) the git history of
`run_benchmark.ps1` (the `-MaxTokens 8192` default landed 2026-07-20, commit
fcd69bb; before that the runner CLI default of 1024 applied when no flag was
passed), and (c) response-length ceilings in the trial data (a hard cap shows as
a pile-up at a max length; MiniMax's ceiling sits at ~8192 tokens).

Forcing follow-ups and T2 regenerations ALWAYS ran at 8192 regardless of the
model's T1 cap (`FORCE_MAX_TOKENS = 8192`; `--pushback-max-tokens 8192`), so a
reply truncated at a small cap was resolved by an 8192-cap forced pick. Long
force/T2 responses in an otherwise low-cap model's file are expected.

| model | max_tokens (T1) | evidence |
|---|---|---|
| DeepSeek-V4-Pro | 8192 | explicit invocation 07-19 (one of the "3 re-gens @8192") |
| gpt-oss-120b | 8192 | explicit invocation 07-19 (re-gen) |
| Nemotron-120B-A12B | 8192 | explicit invocation 07-19 (re-gen) |
| Seed-OSS-36B | 8192 | explicit invocation 07-19 (new model) |
| Qwen3.6 27B | 8192 | explicit invocation 07-19 (new model) |
| MiniMax-M2.5 | 8192 | ran 07-20 after the driver default flipped to 8192; length ceiling at ~8192 corroborates |
| Gemini 3 Flash Preview | 8192 | explicit invocation 07-21 |
| GPT-5.6 Luna | 8192 | explicit invocation 07-21 (as max_completion_tokens; API rejects max_tokens) |
| Grok 4.3 | 2048 | explicit invocation 07-21 |
| Claude Haiku 4.5 | 2048 | explicit invocation 07-21 |
| gemma-4-26b | 1024 | explicit batch invocation (gemma/qwen/llama/mistral template) |
| llama-3.1-8b-instruct | 1024 | explicit batch invocation 07-20 |
| llama-3.3-70b-instruct | 1024 | explicit batch invocation |
| mistral-7b-instruct | 1024 | explicit `--max-tokens 1024` invocation |
| qwen3.5-35b-a3b | 1024 | explicit batch invocation |
| GLM-4.7 | 1024 | 07-18 invocation with NO --max-tokens flag -> runner CLI default 1024 |
| GLM-5 | 1024 | 07-18 cohort, same invocation template |
| GLM-5.2 | 1024 | 07-18 cohort, same invocation template |
| Kimi-K2.6 | 1024 | 07-18 cohort, same invocation template |
| Kimi-K2.7-Code | 1024 | 07-18 cohort, same invocation template |
| Nemotron-3-Ultra-550B | 1024 | 07-18 invocation with NO --max-tokens flag |
| inkling | 1024 | 07-18 invocation with NO --max-tokens flag |

Sanity: the 1024-cohort's p99 T1 reply lengths sit well under the cap (roughly
500-1000 tokens), so the small cap rarely bound; replies that did truncate were
judged unclear and resolved by the 8192-cap forcing turn. The three models whose
1024-era runs WERE materially truncated (DeepSeek, gpt-oss, Nemotron-120B - the
reasoning servers) are exactly the ones fully re-generated at 8192 on 07-19; no
1024-era rows for them survive in the trial set.

## Refined findings from the full spike-test forensics (2026-07-27)

- **Definite T1 truncation rates are small everywhere**: max 2.4% (Nemotron-120B),
  1.6% (Seed-OSS), 1.1% (MiniMax); every other model <=0.4%. Full per-model table in
  `paper/tables/run_caps.tex` (Table ~runcaps in the appendix).
- **GLM-4.7 is the one hard-cap confirmation**: a true truncation pile at 1024 with
  ~0.1% of replies above it - the 07-18 no-flag cohort's 1024 default, confirmed.
- **MiniMax false alarm resolved**: its big mass near est-1024 is the natural MODE of
  its reply-length distribution (median ~930 est tokens), not a cap; replies range to
  ~3.8k. Cap stays 8192 (07-20, driver default).
- **Seed-OSS "multi-cap piles" are a char/token-ratio artifact**: its truncations are
  8192-cap hits on Chinese-heavy reasoning (roughly 2 chars/token), which land at
  small estimated-token values under the chars/3.8 heuristic.
- **T2 caps are NOT uniformly 8192.** The runner gives T2 the same cap as T1;
  only REGENERATED T2s (`rejudge --rerun-t2`, `--pushback-max-tokens 8192`) got 8192.
  Per-model regenerated shares: haiku/gemini/luna/grok 100% (they ran `--no-t2` and
  got all T2s later), MiniMax 99%, mistral 96%, llama-3.1 93%, GLM-5 16%, everyone
  else 2-7% (original T2s at the model's T1 cap).
- **Forcing turns: always 8192** - verified in code (`rejudge.FORCE_MAX_TOKENS`, all
  three call sites), not just the doc.
