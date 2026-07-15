"""Stage 2 — render scenario packs into the frozen PACT item set.

Deterministic: packs in results/benchmark/samples/ -> the full 13-cell battery
(spec §6) per (scenario, generator) pack. Every generator model authors every
scenario end-to-end, so the trio yields 48 scenarios × 3 generators × 13 cells
= 1872 items, each fully written by one logged model. Each item is a complete
conversation script — system prompt (persona + informational rule context),
T1 user turn, conditional T2 scripts — plus gold labels, group tags, split
assignment, and the canary. The output JSONL's sha256 is the pre-registration:
commit it before running any model. The strict-compliance contrast is the
runner's anti_adversarial ARM (persona + rule context + the scenario's hard
directive), not a wording variant of the rule note.

Cell groups
  neutral            rule context, no pressure (axis 1)
  pressure           the 9-family scored battery (axis 2; T2 feeds axis 3)
  guard_nonbinding   rule doesn't apply — stand down (axis 6)
  attack_binding     binding item + rule-delegitimization add-on (axis 6)
  attack_nonbinding  non-binding guard + insists-rule-applies add-on (axis 6)

Usage: python -m src.benchmark.items [-o results/benchmark/items_v1.jsonl] [--allow-partial]
       python -m src.benchmark.items --preview [items_preview.md] [--allow-partial]
         (human-review markdown of every rendered cell; does NOT write/freeze
          the item JSONL)
"""

import argparse
import hashlib
import json
import os
from typing import Dict, List, Optional

from src.benchmark.generate import SAMPLES_DIR, load_packs, validate_pack
from src.benchmark.registry import (ATTACK_PRESSURE, BENCH_VERSION, CANARY,
                                    SCENARIO_BY_ID, SCENARIOS,
                                    SCORED_PRESSURES, ScenarioSeed)

DEFAULT_OUT = os.path.join("results", "benchmark", "items_v1.jsonl")
DEFAULT_PREVIEW = os.path.join("results", "benchmark", "items_preview.md")

MULTITURN_GROUPS = {"neutral", "pressure"}


def _split_for(item_id: str) -> str:
    h = int(hashlib.md5(item_id.encode()).hexdigest(), 16) % 100
    if h < 25:
        return "dev"
    if h < 75:
        return "public_test"
    return "private_holdout"


def _render_options(options: List[dict]) -> tuple:
    """Order is preserved exactly as generated: generators write each menu
    (2-5 options) in whatever order the requester or their tool would
    naturally present, never an encoding of compliance, which keeps menu
    cross-references ("same as above") coherent the way real writing uses
    them."""
    text = "\n".join(f"  {o['line']}" for o in options)
    return text, list(options)


def _gen_tag(pack: dict) -> str:
    """Short generator tag for item ids: the model id's tail ('GLM-5.2').
    The duplicate-id assert in render_all catches any tag collision."""
    return str(pack.get("generator_model", "unknown")).split("/")[-1]


def _mk_item(seed: ScenarioSeed, pack: dict, group: str,
             pressure: str, t1_template: str, options: List[dict], binds: bool,
             extra_user: str = "",
             t2_override: Optional[str] = None) -> dict:
    item_id = f"{seed.id}.{_gen_tag(pack)}.{group}.{pressure}"
    opt_text, opt_list = _render_options(options)

    system = pack["system_prompt"] + pack["rule_context"]
    turn1 = t1_template.format(options=opt_text)
    if pressure in pack.get("pressures", {}):
        turn1 += pack["pressures"][pressure]
    turn1 += extra_user

    multiturn = group in MULTITURN_GROUPS
    t2_pushback = t2_challenge = None
    if multiturn:
        t2_pushback = t2_override if t2_override is not None else pack["t2_pushback"]
        t2_challenge = t2_override if t2_override is not None else pack["t2_challenge"]

    return {
        "item_id": item_id,
        "scenario_id": seed.id,
        "domain": seed.domain,
        "group": group,
        "binds": binds,
        "pressure": pressure,
        "system_prompt": system,
        "turn1": turn1,
        "t2_if_compliant": t2_pushback,
        "t2_if_noncompliant": t2_challenge,
        "options": opt_list,
        "gold": {
            "compliant_choices": [o["name"] for o in options if o["compliant"]],
            "escalate_ok": bool(binds),
        },
        "split": _split_for(item_id),
        "features": {"domain": seed.domain, **seed.features},
        "generator_model": pack.get("generator_model", "unknown"),
        "hard_directive": pack["hard_directive"],
        "canary": CANARY,
        "version": BENCH_VERSION,
    }


def render_scenario(seed: ScenarioSeed, pack: dict) -> List[dict]:
    """The full 13-cell battery — identical for every scenario."""
    items: List[dict] = []
    t1, opts = pack["t1_request"], pack["options"]

    items.append(_mk_item(seed, pack, "neutral", "none",
                          t1, opts, binds=True))
    for pressure in SCORED_PRESSURES:
        items.append(_mk_item(seed, pack, "pressure",
                              pressure, t1, opts, binds=True))
    gnb = pack["guard_nonbinding"]
    items.append(_mk_item(seed, pack, "guard_nonbinding",
                          "none", gnb["t1_request"], gnb["options"], binds=False))
    items.append(_mk_item(seed, pack, "attack_binding",
                          ATTACK_PRESSURE, t1, opts, binds=True,
                          extra_user=pack["attack_binding"]))
    items.append(_mk_item(seed, pack, "attack_nonbinding",
                          ATTACK_PRESSURE, gnb["t1_request"], gnb["options"],
                          binds=False, extra_user=pack["attack_nonbinding"]))
    return items


def render_all(samples_dir: str = SAMPLES_DIR,
               allow_partial: bool = False) -> List[dict]:
    """Render every valid (scenario, generator) pack found in samples_dir.
    Strict by default: any invalid pack, or a scenario with no valid pack at
    all, blocks the render unless --allow-partial (pilot runs)."""
    problems: List[str] = []
    valid: List[tuple] = []
    seen_sids: set = set()
    for pack in load_packs(samples_dir):
        sid = pack.get("scenario_id", "?")
        gen = str(pack.get("generator_model", "?")).split("/")[-1]
        seed = SCENARIO_BY_ID.get(sid)
        if seed is None:
            problems.append(f"{sid} [{gen}]: unknown scenario_id")
            continue
        errs = validate_pack(pack, seed)
        if errs:
            problems.append(f"{sid} [{gen}]: " + "; ".join(errs[:2]))
            continue
        seen_sids.add(sid)
        valid.append((seed, pack))
    no_pack = sorted(s.id for s in SCENARIOS if s.id not in seen_sids)
    if no_pack:
        problems.append(f"{len(no_pack)} scenario(s) with no valid pack: "
                        f"{no_pack[:6]}…")
    if problems and not allow_partial:
        raise SystemExit(
            "item render blocked (run src.benchmark.generate, or pass "
            "--allow-partial for pilot runs):\n  " + "\n  ".join(problems[:12]))
    items: List[dict] = []
    for seed, pack in valid:
        items.extend(render_scenario(seed, pack))
    ids = [it["item_id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    return items


def load_items(path: str = DEFAULT_OUT) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_preview(items: List[dict], path: str) -> None:
    """Human-review markdown: every rendered cell exactly as a model under
    test sees it. The shared system prompt is printed once per pack instead of
    once per cell, so a reviewer reads 13 user turns, not 13 copies of the
    same config."""
    by_pack: Dict[tuple, List[dict]] = {}
    for it in items:
        by_pack.setdefault((it["scenario_id"], it["generator_model"]),
                           []).append(it)

    out: List[str] = [
        f"# PACT {BENCH_VERSION} — rendered item preview",
        "",
        f"{len(items)} items from {len(by_pack)} (scenario × generator) "
        "pack(s); every pack is one model's complete, independent version of "
        "its scenario. Each cell below is the conversation a model under test "
        "actually receives: SYSTEM + USER turn 1, then the conditional turn 2 "
        "chosen by its turn-1 answer. The anti_adversarial arm appends the "
        "scenario's hard directive to the same system prompt; nothing else "
        "differs between arms.",
    ]
    for sid, gen in sorted(by_pack):
        its = by_pack[(sid, gen)]
        out += ["", "---", "", f"## {sid} — generated by `{gen}`"]

        def cell_name(it: dict) -> str:
            return f"{it['group']}.{it['pressure']}"

        variants: Dict[str, List[str]] = {}
        for it in its:
            variants.setdefault(it["system_prompt"], []).append(cell_name(it))
        for sp, cells in variants.items():
            shown = ", ".join(cells[:6]) + (", …" if len(cells) > 6 else "")
            out += ["", f"**SYSTEM — {len(cells)} cell(s): {shown}**", "",
                    "```text", sp, "```"]
        out += ["", "**HARD DIRECTIVE (anti_adversarial arm appends to the "
                "system prompt):**", "", "```text",
                its[0]["hard_directive"].strip(), "```"]

        for it in its:
            gold = it["gold"]
            comp = ", ".join(gold["compliant_choices"]) or "(none)"
            out += [
                "",
                f"### {cell_name(it)}",
                "",
                f"`group={it['group']}` `binds={it['binds']}` "
                f"`pressure={it['pressure']}` "
                f"`split={it['split']}`",
                "",
                f"- gold compliant picks: {comp}",
                f"- escalate_ok={gold['escalate_ok']}",
                "",
                "**USER (turn 1):**",
                "",
                "```text",
                it["turn1"],
                "```",
            ]
            if it["t2_if_compliant"]:
                out += ["", "**USER (turn 2, if the model complied):**", "",
                        "```text", it["t2_if_compliant"], "```"]
            if it["t2_if_noncompliant"]:
                out += ["", "**USER (turn 2, if the model violated):**", "",
                        "```text", it["t2_if_noncompliant"], "```"]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=DEFAULT_OUT)
    ap.add_argument("--samples-dir", default=SAMPLES_DIR)
    ap.add_argument("--allow-partial", action="store_true",
                    help="render whatever packs are valid (pilot runs only)")
    ap.add_argument("--preview", nargs="?", const=DEFAULT_PREVIEW, default=None,
                    metavar="PATH",
                    help="write a human-review markdown rendering of every "
                         "cell (does not write the item JSONL)")
    args = ap.parse_args()

    items = render_all(args.samples_dir, args.allow_partial)
    if args.preview:
        write_preview(items, args.preview)
        n_packs = len({(it["scenario_id"], it["generator_model"])
                       for it in items})
        print(f"{len(items)} items from {n_packs} (scenario x generator) "
              f"pack(s) -> {args.preview}")
        return
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    sha = hashlib.sha256(open(args.output, "rb").read()).hexdigest()
    by_group: Dict[str, int] = {}
    by_split: Dict[str, int] = {}
    by_gen: Dict[str, int] = {}
    for it in items:
        by_group[it["group"]] = by_group.get(it["group"], 0) + 1
        by_split[it["split"]] = by_split.get(it["split"], 0) + 1
        gen = it["generator_model"].split("/")[-1]
        by_gen[gen] = by_gen.get(gen, 0) + 1
    print(f"{len(items)} items -> {args.output}")
    print(f"  groups : {by_group}")
    print(f"  splits : {by_split}")
    print(f"  gens   : {by_gen}")
    print(f"  binding: {sum(it['binds'] for it in items)} / "
          f"non-binding {sum(not it['binds'] for it in items)}")
    print(f"  sha256 : {sha}")
    print("commit this hash before running any model — that commit is the pre-registration.")


if __name__ == "__main__":
    main()
