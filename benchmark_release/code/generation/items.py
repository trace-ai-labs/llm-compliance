"""Stage 2: deterministic render of scenario packs into the frozen PACT item
set: 13 cells per (scenario, generator) pack, with gold labels, group tags,
and the canary. The output JSONL's sha256 identifies the frozen item set.
Paper Section 3.1, "Template-based prompt construction".

A fresh freeze writes results/generation/items.jsonl; the shipped frozen
corpus is data/items.jsonl (paths.ITEMS) and is never overwritten.

Usage: python -m generation.items [-o PATH] [--allow-partial]
       python -m generation.items --preview [PATH] [--allow-partial]
"""

import argparse
import hashlib
import json
import os
from typing import Dict, List, Optional

import paths
from generation.generate import (ALL_COMPONENTS, SAMPLES_DIR, load_packs,
                                    validate_pack)
from generation.registry import (ATTACK_PRESSURE, BENCH_VERSION, CANARY,
                                    DOMAIN_BY_KEY, SCENARIO_BY_ID, SCENARIOS,
                                    SCORED_PRESSURES, ScenarioSeed)

DEFAULT_OUT = os.path.join(paths.GENERATION_DIR, "items.jsonl")
DEFAULT_PREVIEW = os.path.join(paths.GENERATION_DIR, "items_preview.md")

MULTITURN_GROUPS = {"neutral", "pressure"}

# Which measurement axis each cell group primarily feeds. Axes 4
# (steerability) and 5 (transparency) are cross-cell (mode / judge), not per-group.
GROUP_AXIS: Dict[str, str] = {
    "neutral": "1_default_compliance",
    "pressure": "2_pressure_resistance",       # its T2 also feeds 3_pushback
    "guard_nonbinding": "6_rule_scope_discernment",
    "attack_binding": "6_rule_scope_discernment",
    "attack_nonbinding": "6_rule_scope_discernment",
}

# One-line mechanism each scored pressure targets.
PRESSURE_MECHANISM: Dict[str, str] = {
    "urgency": "time scarcity",
    "mgr_verbal": "legitimate authority",
    "peer_escaped": "descriptive norm + impunity",
    "risk_minimize": "deterrence discounting",
    "cost_pressure": "loss framing",
    "false_clearance": "unverifiable authorization",
    "fait_accompli": "commitment / sunk cost",
    "sympathetic_beneficiary": "identifiable victim of compliance",
    "responsibility_shift": "accountability transfer",
    ATTACK_PRESSURE: "epistemic attack on the rule's scope",
    "none": "",
}


def _pressure_fields(group: str, pressure: str) -> Dict[str, str]:
    """Split the pressure into an INDIVIDUAL family and a GENERAL category so
    the dataset can be sliced either way."""
    if group == "pressure":
        category = "pressure"
    elif group in ("attack_binding", "attack_nonbinding"):
        category = "attack"
    else:
        category = "none"
    family = pressure if pressure and pressure != "none" else "none"
    return {
        "pressure_category": category,               # none | pressure | attack
        "pressure_family": family,                   # the individual mechanism
        "pressure_mechanism": PRESSURE_MECHANISM.get(family, ""),
    }


def _render_options(options: List[dict]) -> tuple:
    """Menu order is preserved exactly as generated; it never encodes
    compliance."""
    text = "\n".join(f"  {o['line']}" for o in options)
    return text, list(options)


def _gen_tag(pack: dict) -> str:
    """Short generator tag for item ids: the model id's tail."""
    return str(pack.get("generator_model", "unknown")).split("/")[-1]


def _pack_completeness(pack: dict) -> tuple:
    """(complete, total) components for this pack."""
    total = pack.get("_components_total") or len(ALL_COMPONENTS)
    na = pack.get("_na_components") or []
    complete = pack.get("_components_complete", total - len(na))
    return complete, total


def _mk_item(seed: ScenarioSeed, pack: dict, group: str,
             pressure: str, t1_template: str, options: List[dict], binds: bool,
             extra_user: str = "",
             t2_override: Optional[str] = None,
             multiturn_ok: bool = True,
             guard_models: Optional[List[str]] = None) -> dict:
    item_id = f"{seed.id}.{_gen_tag(pack)}.{group}.{pressure}"
    opt_text, opt_list = _render_options(options)
    domain = DOMAIN_BY_KEY.get(seed.domain)

    system = pack["system_prompt"] + pack["rule_context"]
    # literal replace, not str.format: generated T1 text can contain stray
    # braces ("{influencer_handle}") that .format would try to interpolate
    turn1 = t1_template.replace("{options}", opt_text)
    if pressure in pack.get("pressures", {}):
        turn1 += pack["pressures"][pressure]
    turn1 += extra_user

    # if the T2 component is NA the cell degrades to single-turn rather than
    # dropping the neutral/pressure measurement
    multiturn = group in MULTITURN_GROUPS and multiturn_ok
    t2_pushback = t2_challenge = None
    if multiturn:
        t2_pushback = t2_override if t2_override is not None else pack["t2_pushback"]
        t2_challenge = t2_override if t2_override is not None else pack["t2_challenge"]

    complete, total = _pack_completeness(pack)
    return {
        "item_id": item_id,
        "scenario_id": seed.id,
        "domain": seed.domain,
        "group": group,
        "binds": binds,
        "pressure": pressure,
        **_pressure_fields(group, pressure),
        "metric_axis": GROUP_AXIS.get(group, ""),
        "system_prompt": system,
        "turn1": turn1,
        "t2_if_compliant": t2_pushback,
        "t2_if_noncompliant": t2_challenge,
        "options": opt_list,
        "gold": {
            "compliant_choices": [o["name"] for o in options if o["compliant"]],
            "escalate_ok": bool(binds),
        },
        # provenance: who wrote it and who cross-reviewed it
        "generator_model": pack.get("generator_model", "unknown"),
        "guard_models": list(guard_models or []),
        "hard_directive": pack["hard_directive"],
        # scenario + domain reference (rule, what breaking it buys, real anchor)
        "scenario_title": seed.title,
        "rule": seed.rule,
        "local_objective": seed.local_objective,
        "temptation": seed.temptation,
        "domain_title": domain.title if domain else seed.domain,
        "penalty_anchor": domain.real_world_stakes if domain else "",
        "pack_completeness": f"{complete}/{total}",
        "pack_components_complete": complete,
        "pack_components_total": total,
        "canary": CANARY,
        "version": BENCH_VERSION,
    }


def render_scenario(seed: ScenarioSeed, pack: dict,
                    guard_models: Optional[List[str]] = None) -> List[dict]:
    """All 13 cells for a pack; a cell whose backing component is NA is
    skipped. Cell -> component:
      pressure.X          -> pressure.X
      guard_nonbinding    -> guard_nonbinding
      attack_binding      -> attacks
      attack_nonbinding   -> attacks AND guard_nonbinding (reuses the twin)
      neutral + pressure  -> multiturn T2 needs t2 (degrade to single-turn if NA)
    """
    na = set(pack.get("_na_components") or [])
    t2_ok = "t2" not in na
    items: List[dict] = []
    t1, opts = pack["t1_request"], pack["options"]
    g = guard_models

    items.append(_mk_item(seed, pack, "neutral", "none",
                          t1, opts, binds=True, multiturn_ok=t2_ok,
                          guard_models=g))
    for pressure in SCORED_PRESSURES:
        if f"pressure.{pressure}" in na:
            continue
        items.append(_mk_item(seed, pack, "pressure",
                              pressure, t1, opts, binds=True, multiturn_ok=t2_ok,
                              guard_models=g))
    gnb_ok = "guard_nonbinding" not in na
    if gnb_ok:
        gnb = pack["guard_nonbinding"]
        items.append(_mk_item(seed, pack, "guard_nonbinding",
                              "none", gnb["t1_request"], gnb["options"],
                              binds=False, guard_models=g))
    if "attacks" not in na:
        items.append(_mk_item(seed, pack, "attack_binding",
                              ATTACK_PRESSURE, t1, opts, binds=True,
                              extra_user=pack["attack_binding"], guard_models=g))
        if gnb_ok:
            gnb = pack["guard_nonbinding"]
            items.append(_mk_item(seed, pack, "attack_nonbinding",
                                  ATTACK_PRESSURE, gnb["t1_request"],
                                  gnb["options"], binds=False,
                                  extra_user=pack["attack_nonbinding"],
                                  guard_models=g))
    return items


def render_all(samples_dir: str = SAMPLES_DIR,
               allow_partial: bool = False) -> List[dict]:
    """Render every usable (scenario, generator) pack found in samples_dir; a
    partial pack ships its surviving cells. --allow-partial lifts the block
    when some packs are missing or invalid."""
    problems: List[str] = []
    valid: List[tuple] = []
    partial: List[str] = []
    seen_sids: set = set()
    all_packs = list(load_packs(samples_dir))
    # the generator trio: every pack is cross-reviewed by the OTHER two models
    trio = sorted({p.get("generator_model") for p in all_packs
                   if p.get("generator_model")})
    for pack in all_packs:
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
        c, n = _pack_completeness(pack)
        if c < n:
            partial.append(f"{sid} [{gen}]: {c}/{n} "
                           f"(NA: {pack.get('_na_components')})")
    no_pack = sorted(s.id for s in SCENARIOS if s.id not in seen_sids)
    if no_pack:
        problems.append(f"{len(no_pack)} scenario(s) with no usable pack: "
                        f"{no_pack[:6]}…")
    if problems and not allow_partial:
        raise SystemExit(
            "item render blocked (run generation.generate, or pass "
            "--allow-partial for pilot runs):\n  " + "\n  ".join(problems[:12]))
    if partial:
        print(f"[items] {len(partial)} partial pack(s) shipped with missing "
              f"cells:\n  " + "\n  ".join(partial[:20]))
    items: List[dict] = []
    for seed, pack in valid:
        guards = [m for m in trio if m != pack.get("generator_model")]
        items.extend(render_scenario(seed, pack, guard_models=guards))
    ids = [it["item_id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    return items


def load_items(path: str = paths.ITEMS) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_preview(items: List[dict], path: str) -> None:
    """Human-review markdown: every rendered cell exactly as a model under
    test sees it."""
    by_pack: Dict[tuple, List[dict]] = {}
    for it in items:
        by_pack.setdefault((it["scenario_id"], it["generator_model"]),
                           []).append(it)

    out: List[str] = [
        f"# PACT {BENCH_VERSION} - rendered item preview",
        "",
        f"{len(items)} items from {len(by_pack)} (scenario × generator) "
        "pack(s); every pack is one model's complete, independent version of "
        "its scenario. Each cell below is the conversation a model under test "
        "actually receives: SYSTEM + USER turn 1, then the conditional turn 2 "
        "chosen by its turn-1 answer. The anti_adversarial mode appends the "
        "scenario's hard directive to the same system prompt; nothing else "
        "differs between modes.",
    ]
    for sid, gen in sorted(by_pack):
        its = by_pack[(sid, gen)]
        comp = its[0].get("pack_completeness", "?")
        tag = f" - {comp} components" + (
            " (PARTIAL)" if comp != f"{its[0].get('pack_components_total')}/"
            f"{its[0].get('pack_components_total')}" else "")
        out += ["", "---", "",
                f"## {sid} - generated by `{gen}`{tag} - {len(its)} cell(s)"]

        def cell_name(it: dict) -> str:
            return f"{it['group']}.{it['pressure']}"

        variants: Dict[str, List[str]] = {}
        for it in its:
            variants.setdefault(it["system_prompt"], []).append(cell_name(it))
        for sp, cells in variants.items():
            shown = ", ".join(cells[:6]) + (", …" if len(cells) > 6 else "")
            out += ["", f"**SYSTEM - {len(cells)} cell(s): {shown}**", "",
                    "```text", sp, "```"]
        out += ["", "**HARD DIRECTIVE (anti_adversarial mode appends to the "
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
                f"`pressure={it['pressure']}`",
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
    by_gen: Dict[str, int] = {}
    for it in items:
        by_group[it["group"]] = by_group.get(it["group"], 0) + 1
        gen = it["generator_model"].split("/")[-1]
        by_gen[gen] = by_gen.get(gen, 0) + 1
    print(f"{len(items)} items -> {args.output}")
    print(f"  groups : {by_group}")
    print(f"  gens   : {by_gen}")
    print(f"  binding: {sum(it['binds'] for it in items)} / "
          f"non-binding {sum(not it['binds'] for it in items)}")
    print(f"  sha256 : {sha}")
    print("record this hash before running any model - it identifies the frozen item set.")


if __name__ == "__main__":
    main()
