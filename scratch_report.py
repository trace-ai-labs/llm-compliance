"""Ad-hoc yield report: for each (scenario, generator) pack, how many of the
13 rendered cells are backed by fully-valid (non-NA) components.

13-cell battery (items.render_scenario):
  neutral            -> task, rules, persona, t2
  pressure.<k> (x9)  -> task, rules, persona, t2, pressure.<k>
  guard_nonbinding   -> persona, rules, guard_nonbinding
  attack_binding     -> persona, rules, task, attacks
  attack_nonbinding  -> persona, rules, guard_nonbinding, attacks
"""
import os
from src.benchmark.generate import load_components, ALL_COMPONENTS
from src.benchmark.registry import SCORED_PRESSURES

TRIO = ["zai-org/GLM-5.2", "moonshotai/Kimi-K2.6",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B"]
ONLY = ["hr_hiring.A", "privacy.B"]


def cell_deps():
    cells = {}
    cells["neutral"] = ["task", "rules", "persona", "t2"]
    for k in SCORED_PRESSURES:
        cells[f"pressure.{k}"] = ["task", "rules", "persona", "t2", f"pressure.{k}"]
    cells["guard_nonbinding"] = ["persona", "rules", "guard_nonbinding"]
    cells["attack_binding"] = ["persona", "rules", "task", "attacks"]
    cells["attack_nonbinding"] = ["persona", "rules", "guard_nonbinding", "attacks"]
    return cells


def na_set(comps):
    na = set()
    for c in ALL_COMPONENTS:
        v = comps.get(c)
        if not isinstance(v, dict) or v.get("_na"):
            na.add(c)  # missing or NA-flagged
    return na


def main():
    cells = cell_deps()
    print(f"{'scenario':<16}{'model':<26}{'cells/13':>9}  {'components':>11}  bad cells")
    print("-" * 92)
    grand = {}
    for sid in ONLY:
        for model in TRIO:
            comps = load_components(sid, model)
            if not comps:
                print(f"{sid:<16}{model.split('/')[-1]:<26}{'--':>9}  (no components file yet)")
                continue
            na = na_set(comps)
            n_comp_ok = len([c for c in ALL_COMPONENTS if c not in na])
            good = [cell for cell, deps in cells.items() if not (set(deps) & na)]
            bad = sorted(set(cells) - set(good))
            grand[(sid, model)] = len(good)
            tag = model.split("/")[-1]
            badstr = ", ".join(b.replace("pressure.", "p.") for b in bad) if bad else "-"
            print(f"{sid:<16}{tag:<26}{len(good):>7}/13  {n_comp_ok:>7}/{len(ALL_COMPONENTS):<3}  {badstr}")
    if grand:
        avg = sum(grand.values()) / len(grand)
        full = sum(1 for v in grand.values() if v == 13)
        print("-" * 92)
        print(f"mean {avg:.1f}/13 across {len(grand)} packs; "
              f"{full}/{len(grand)} packs fully valid (13/13)")


if __name__ == "__main__":
    main()
