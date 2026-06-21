#!/usr/bin/env python3
"""Robustness to item criticality.

Repeats the core grid while varying the purchased item from low-criticality supplies to
safety-critical equipment, confirming that compliance is not an artifact of the fixed
item used elsewhere.

Run: python -m src.experiments.stakes -n 25 -o results/data/stakes.jsonl
"""

import argparse
from src.core import *
from src.prompts.stakes import STAKES_CONTEXTS


def make_user_message_stakes(rng: random.Random, vendors: List[Vendor],
                             stakes: str) -> Tuple[str, List[Vendor], Dict[str, str]]:
    """Build user message sampling randomly from the appropriate stakes pool.

    Returns message text, shuffled vendor list, and a context metadata dict so
    each trial records the sampled item/requester/preamble for later analysis.
    """
    item, requester, preamble = rng.choice(STAKES_CONTEXTS[stakes])
    shuffled = vendors.copy()
    rng.shuffle(shuffled)
    ctx = {
        "stakes_item": item,
        "stakes_requester": requester,
        "stakes_preamble": preamble,
    }
    return f"{preamble}\n\n{vendor_lines(shuffled)}\n\nwhich one should we go with?", shuffled, ctx


def run_trial_stakes(sc: Scenario, run_id: int, model: str, api_key: str,
                     seed: int, stakes: str,
                     classify_reasoning: bool = False) -> TrialResult:
    """run_trial variant that samples from the stakes-appropriate context pool."""
    rng = random.Random(seed)
    user_msg, shuffled_vendors, sampled_ctx = make_user_message_stakes(rng, VENDORS, stakes)
    response = call_model(sc.system_prompt, user_msg + sc.user_suffix, model, api_key)
    chosen = judge_vendor(response, shuffled_vendors, api_key)
    compliant = chosen.certified if chosen else False

    # Make stakes explicit in scenario names for robust unified-file analyses.
    scenario_name = f"{stakes}stakes_{sc.name}"

    reasoning = ReasoningAnalysis()
    reasoning_mode = ""
    if classify_reasoning:
        vendor_name = chosen.name if chosen else "UNCLEAR"
        reasoning = judge_reasoning_full(response, api_key,
                                         vendor_chosen=vendor_name, compliant=compliant)
        reasoning_mode = classify_reasoning_mode(reasoning, compliant)

    return TrialResult(
        scenario=scenario_name, run_id=run_id, seed=seed, model=model,
        vendor_chosen=chosen.name if chosen else "UNCLEAR",
        compliant=compliant, parseable=chosen is not None,
        expected_penalty=sc.ep, ev_verdict=sc.ev, raw_response=response,
        reasoning_category=reasoning.primary if classify_reasoning else "",
        reasoning_secondary=reasoning.secondary if classify_reasoning else "",
        reasoning_mentions_penalty=reasoning.mentions_penalty,
        reasoning_mentions_rule=reasoning.mentions_rule,
        reasoning_cost_benefit_explicit=reasoning.cost_benefit_explicit,
        reasoning_supports_choice=reasoning.reasoning_supports_choice if classify_reasoning else "",
        reasoning_mode=reasoning_mode,
        metadata={**sc.metadata, "stakes": stakes, **sampled_ctx},
    )


def run_experiment_stakes(scenarios: Dict[str, Scenario], model: str, api_key: str,
                          n_runs: int, outfile: str, stakes: str,
                          classify_reasoning: bool = False, batch_size: int = 1):
    """run_experiment variant that passes stakes through to run_trial_stakes."""
    scenario_names = list(scenarios.keys())
    jobs = [(run_id, sc_name)
            for run_id in range(n_runs)
            for sc_name in scenario_names]
    total = len(jobs)
    done = 0
    sc_results = defaultdict(list)
    lock = threading.Lock()

    def _run(run_id: int, sc_name: str) -> TrialResult:
        sc = scenarios[sc_name]
        seed = run_id * 1000 + stable_hash(sc_name + stakes)
        return run_trial_stakes(sc, run_id, model, api_key, seed, stakes, classify_reasoning)

    ex = ThreadPoolExecutor(max_workers=batch_size)
    futures = {ex.submit(_run, run_id, sc_name): (run_id, sc_name)
               for run_id, sc_name in jobs}
    try:
        for future in as_completed(futures):
            run_id, sc_name = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"\n  [ERROR] run{run_id+1} {sc_name}: {e}")
                continue
            with lock:
                sc_results[sc_name].append(result)
                with open(outfile, "a") as f:
                    f.write(json.dumps(asdict(result)) + "\n")
                done += 1
                status = '✓ comply' if result.compliant else ('✗ violate' if result.parseable else '? unclear')
                print(f"  [{done}/{total}] run{run_id+1} {sc_name:<35} [{stakes}] → {status}")
    except KeyboardInterrupt:
        for f in futures:
            f.cancel()
        ex.shutdown(wait=False)
        print(f"\nInterrupted — {done}/{total} results saved to {outfile}")
        raise SystemExit(1)

    print_summary(scenarios, sc_results)
    return sc_results


# ── Scenario builder ──────────────────────────────────────────────────────────

def build_scenarios(all_framings: bool = False,
                    framings: List[str] = None,
                    include_baseline: bool = True) -> Dict[str, Scenario]:
    """Build the control grid scenarios. Stakes condition is passed at run time,
    not baked into the scenario, so the same scenario set is reused for both."""
    s = {}

    if framings is None:
        framings = list(RULE_FRAMINGS.keys()) if all_framings else ["directive", "informational", "discretionary"]

    for framing_name in framings:
        for fin_level in FINANCIAL_LEVELS:
            prob, pen = FINANCIAL_LEVELS[fin_level]
            name = f"{framing_name}_{fin_level}"
            s[name] = Scenario(
                name=name,
                system_prompt=BASE_PROMPT + make_reg_context(framing_name, fin_level),
                audit_prob=prob,
                penalty=pen,
                metadata={"framing": framing_name, "fin_level": fin_level},
            )

    if include_baseline:
        s["baseline"] = Scenario(
            name="baseline",
            system_prompt=BASE_PROMPT,
            metadata={"framing": "none", "fin_level": "none"},
        )

    return s

def main():
    p = argparse.ArgumentParser(description="Purchase context stakes robustness")
    add_common_args(p)
    p.add_argument("--stakes", choices=["low", "high", "both"], default="both",
                   help="which stakes condition to run (default: both)")
    args = p.parse_args()

    api_key = get_api_key()
    if args.paper:
        scenarios = build_scenarios(framings=["informational"], include_baseline=True)
    else:
        scenarios = build_scenarios(all_framings=args.all_framings)
    stakes_to_run = ["low", "high"] if args.stakes == "both" else [args.stakes]

    # Single output file for all stakes and all models; both low/high results append to it.
    # Model and stakes are already tagged in each row's metadata.
    outfile = args.output or "results/data/stakes.jsonl"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    for stakes in stakes_to_run:
        print(f"\nStakes: {stakes} | {len(scenarios)} scenarios × {args.runs} runs on {args.model}\n")
        run_experiment_stakes(
            scenarios, args.model, api_key,
            n_runs=args.runs,
            outfile=outfile,
            stakes=stakes,
            classify_reasoning=args.classify_reasoning,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()