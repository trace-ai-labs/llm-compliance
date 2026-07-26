"""Find the most damning real violation in the trials, for the paper's worked example.

The appendix shows one item end to end (healthcare_admin.D) to explain the item
template. That item is a good template specimen but a mild violation: an omission,
motivated by paperwork time. This finds the opposite, a violation a reader winces at,
and prints it verbatim so it can be quoted.

What counts as damning, in descending weight:

  arm            violating in `anti_adversarial` is worse than in `base`: the system
                 prompt explicitly ordered it to follow every rule without exception
  no pressure    violating a `neutral` item is worse than caving to a pressure; nobody
                 pushed it
  strong model   a violation by a top-PACTScore model matters more than one by a model
                 nobody would deploy
  sensitive      healthcare, privacy, AML, export controls, hiring over advertising
  unanimous      violating on all 3 replications is a property, not a coin flip
  named the rule a RATIONALIZED or DEFIANT_HONEST reply proves the model understood the
                 rule and proceeded anyway, which is the quotable part. A SILENT
                 violation is worse behaviour but a weaker exhibit, so it scores lower
                 here while staying visible in the output.

Usage:
  python -m src.benchmark.find_example                     # top 5 overall
  python -m src.benchmark.find_example --domain healthcare_admin export_controls
  python -m src.benchmark.find_example --scenario healthcare_admin.C --top 3
  python -m src.benchmark.find_example --arm anti_adversarial --tex
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional

from src.benchmark.judges import HONESTY_PATH, load_honesty, load_trials
from src.benchmark.runner import TRIALS_DIR, TRIVIAL_PREFIX

# Domains where a violation carries statutory consequence, weighted by how
# unambiguous the duty is and how directly a person is harmed.
SENSITIVE = {
    "healthcare_admin": 1.0, "privacy": 1.0, "aml_kyc": 1.0,
    "export_controls": 1.0, "hr_hiring": 0.9, "finance": 0.8,
    "pharma_medinfo": 0.8, "gov_services": 0.7, "customer_service": 0.5,
    "procurement": 0.5, "advertising": 0.4, "moderation": 0.4,
}
# A reply that names the rule and proceeds is the exhibit; a silent one is worse
# behaviour but shows the reader nothing.
HONESTY_WEIGHT = {
    "SCOPE_DENIED": 1.0, "RISK_MINIMIZED": 1.0, "FALSE_AUTHORITY": 1.0,
    "DEFIANT_HONEST": 0.9, "SILENT": 0.3,
}


def score_trial(t: dict, rank: float, honesty: Optional[str],
                unanimous: bool) -> float:
    s = 0.0
    s += 3.0 if t["arm"] == "anti_adversarial" else 0.0
    s += 2.5 if t["group"] == "neutral" else (1.0 if t["group"] == "attack_binding" else 0.0)
    s += 2.0 * rank                                   # 1.0 = best model in panel
    s += 2.0 * SENSITIVE.get(t["domain"], 0.5)
    s += 1.5 if unanimous else 0.0
    s += 1.5 * HONESTY_WEIGHT.get(honesty or "", 0.5)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials-dir", default=TRIALS_DIR)
    ap.add_argument("--honesty", default=HONESTY_PATH)
    ap.add_argument("--domain", nargs="*", help="restrict to these domains")
    ap.add_argument("--scenario", nargs="*", help="restrict to these scenario ids")
    ap.add_argument("--arm", choices=["base", "anti_adversarial"])
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--chars", type=int, default=1400, help="response chars to print")
    ap.add_argument("--tex", action="store_true",
                    help="emit the top hit as a LaTeX quote block for the appendix")
    args = ap.parse_args()

    if not os.path.isdir(args.trials_dir):
        raise SystemExit(f"no trials at {args.trials_dir}. This selector needs the "
                         f"eval run; it cannot work from the frozen items alone.")
    trials = load_trials(args.trials_dir)
    honesty = load_honesty(args.honesty)

    real = [t for t in trials if not t["model"].startswith(TRIVIAL_PREFIX)]
    # model strength = share of that model's trials that were correct, as a cheap
    # stand-in for PACTScore so this stays runnable without metrics_v2.csv
    ok: Dict[str, List[bool]] = defaultdict(list)
    for t in real:
        if t["t1_outcome"] != "unclear":
            ok[t["model"]].append(t["t1_outcome"] == "comply")
    strength = {m: sum(v) / len(v) for m, v in ok.items() if v}
    lo, hi = min(strength.values()), max(strength.values())
    rank = {m: (v - lo) / (hi - lo) if hi > lo else 0.5 for m, v in strength.items()}

    # unanimity: did this (model, arm, item) violate on every decided rep?
    cell: Dict[tuple, List[str]] = defaultdict(list)
    for t in real:
        cell[(t["model"], t["arm"], t["item_id"])].append(t["t1_outcome"])

    cands = []
    for t in real:
        if t["t1_outcome"] != "violate" or not t["binds"]:
            continue
        if args.domain and t["domain"] not in args.domain:
            continue
        if args.scenario and t["scenario_id"] not in args.scenario:
            continue
        if args.arm and t["arm"] != args.arm:
            continue
        outs = [o for o in cell[(t["model"], t["arm"], t["item_id"])] if o != "unclear"]
        unanimous = bool(outs) and all(o == "violate" for o in outs)
        h = honesty.get(f"{t['model']}||{t['trial_id']}")
        cands.append((score_trial(t, rank.get(t["model"], 0.5), h, unanimous), h,
                      unanimous, t))
    if not cands:
        raise SystemExit("no violating trials matched those filters")
    cands.sort(key=lambda c: -c[0])
    # One entry per (model, arm, item): the three replications of the same cell are
    # the same exhibit, so keep the best-scoring one and drop its siblings.
    seen = set()
    deduped = []
    for c in cands:
        t = c[3]
        key = (t["model"], t["arm"], t["item_id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    cands = deduped

    print(f"{len(cands)} violating trials matched; top {args.top} by damningness\n")
    for i, (sc, h, unan, t) in enumerate(cands[:args.top], 1):
        print("=" * 78)
        print(f"#{i}  score {sc:.2f}   {t['model']}")
        print(f"    {t['scenario_id']} ({t['domain']})  group={t['group']} "
              f"arm={t['arm']}  pressure={t['pressure']}")
        print(f"    honesty={h or 'unlabelled'}   violated on every rep: {unan}")
        print(f"    rule: {t.get('rule', '(not on trial row)')}")
        print("-" * 78)
        print((t.get("t1_response") or "")[:args.chars])
        if t.get("t2_outcome"):
            print(f"\n    --- turn 2 ({t.get('t2_script')}) -> {t['t2_outcome']} ---")
            print((t.get("t2_response") or "")[:args.chars // 2])
        print()

    if args.tex:
        _, h, _, t = cands[0]
        print("\n" + "=" * 78)
        print("% LaTeX block for the appendix (check the text before pasting):")
        print(r"\begin{quote}\scriptsize\ttfamily")
        for para in (t.get("t1_response") or "").split("\n"):
            if para.strip():
                print(para.replace("&", r"\&").replace("%", r"\%")
                          .replace("$", r"\$").replace("#", r"\#")
                          .replace("_", r"\_"))
                print()
        print(r"\end{quote}")


if __name__ == "__main__":
    main()
