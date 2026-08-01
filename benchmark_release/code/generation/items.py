"""Render scenario packs into PACT samples: 13 cells per (scenario,
generator) pack, each in two system-prompt modes - `base`, and `mandate`
with the scenario's hard compliance directive appended to the same system
prompt. Nothing else differs between the two samples of a cell.
Paper Section 3.1, "Template-based prompt construction".

A fresh render writes results/generation/pact.jsonl; the released dataset is
data/pact.jsonl (paths.PACT) and is never overwritten.

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
from generation.registry import (ATTACK_PRESSURE, CANARY, SCENARIO_BY_ID,
                                    SCENARIOS, SCORED_PRESSURES, ScenarioSeed)

DEFAULT_OUT = os.path.join(paths.GENERATION_DIR, "pact.jsonl")
DEFAULT_PREVIEW = os.path.join(paths.GENERATION_DIR, "pact_preview.md")

MULTITURN_GROUPS = {"neutral", "pressure"}
MODES = ("base", "mandate")


def _gen_tag(pack: dict) -> str:
    """Short generator tag for sample ids: the model slug's tail."""
    return str(pack.get("generator_model", "unknown")).split("/")[-1].lower()


def _pack_completeness(pack: dict) -> tuple:
    total = pack.get("_components_total") or len(ALL_COMPONENTS)
    na = pack.get("_na_components") or []
    complete = pack.get("_components_complete", total - len(na))
    return complete, total


def _mk_samples(seed: ScenarioSeed, pack: dict, group: str,
                pressure: str, t1_template: str, options: List[dict],
                binds: bool, extra_user: str = "",
                multiturn_ok: bool = True) -> List[dict]:
    """The two samples (base + mandate) of one cell."""
    item = f"{seed.id}.{_gen_tag(pack)}.{group}.{pressure}"
    opt_text = "\n".join(f"  {o['line']}" for o in options)

    system = pack["system_prompt"] + pack["rule_context"]
    # literal replace, not str.format: generated T1 text can contain stray
    # braces ("{influencer_handle}") that .format would try to interpolate
    turn1 = t1_template.replace("{options}", opt_text)
    if pressure in pack.get("pressures", {}):
        turn1 += pack["pressures"][pressure]
    turn1 += extra_user

    # if the T2 component is NA the cell degrades to single-turn rather than
    # dropping the neutral/pressure measurement
    pushback = (pack["t2_pushback"]
                if group in MULTITURN_GROUPS and multiturn_ok else None)

    samples = []
    for mode in MODES:
        samples.append({
            "id": f"{item}.{mode}",
            "item": item,
            "mode": mode,
            "scenario": seed.id,
            "domain": seed.domain,
            "group": group,
            "binds": binds,
            "pressure": pressure,
            "rule": seed.rule,
            "system_prompt": system + (pack["hard_directive"]
                                       if mode == "mandate" else ""),
            "turn1": turn1,
            # menu order is preserved exactly as generated; it never encodes
            # compliance
            "options": list(options),
            "gold": {
                "compliant_choices": [o["name"] for o in options
                                      if o["compliant"]],
                "escalate_ok": bool(binds),
            },
            "pushback": pushback,
            "generator_model": pack.get("generator_model", "unknown"),
            "canary": CANARY,
        })
    return samples


def render_scenario(seed: ScenarioSeed, pack: dict) -> List[dict]:
    """All samples for a pack; a cell whose backing component is NA is
    skipped. Cell -> component:
      pressure.X          -> pressure.X
      guard_nonbinding    -> guard_nonbinding
      attack_binding      -> attacks
      attack_nonbinding   -> attacks AND guard_nonbinding (reuses the twin)
      neutral + pressure  -> multiturn T2 needs t2 (degrade to single-turn if NA)
    """
    na = set(pack.get("_na_components") or [])
    t2_ok = "t2" not in na
    out: List[dict] = []
    t1, opts = pack["t1_request"], pack["options"]

    out += _mk_samples(seed, pack, "neutral", "none", t1, opts,
                       binds=True, multiturn_ok=t2_ok)
    for pressure in SCORED_PRESSURES:
        if f"pressure.{pressure}" in na:
            continue
        out += _mk_samples(seed, pack, "pressure", pressure, t1, opts,
                           binds=True, multiturn_ok=t2_ok)
    gnb_ok = "guard_nonbinding" not in na
    if gnb_ok:
        gnb = pack["guard_nonbinding"]
        out += _mk_samples(seed, pack, "guard_nonbinding", "none",
                           gnb["t1_request"], gnb["options"], binds=False)
    if "attacks" not in na:
        out += _mk_samples(seed, pack, "attack_binding", ATTACK_PRESSURE,
                           t1, opts, binds=True,
                           extra_user=pack["attack_binding"])
        if gnb_ok:
            gnb = pack["guard_nonbinding"]
            out += _mk_samples(seed, pack, "attack_nonbinding", ATTACK_PRESSURE,
                               gnb["t1_request"], gnb["options"], binds=False,
                               extra_user=pack["attack_nonbinding"])
    return out


def render_all(samples_dir: str = SAMPLES_DIR,
               allow_partial: bool = False) -> List[dict]:
    """Render every usable (scenario, generator) pack found in samples_dir; a
    partial pack ships its surviving cells. --allow-partial lifts the block
    when some packs are missing or invalid."""
    problems: List[str] = []
    valid: List[tuple] = []
    partial: List[str] = []
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
            "render blocked (run generation.generate, or pass "
            "--allow-partial for pilot runs):\n  " + "\n  ".join(problems[:12]))
    if partial:
        print(f"[items] {len(partial)} partial pack(s) kept with missing "
              f"cells:\n  " + "\n  ".join(partial[:20]))
    out: List[dict] = []
    for seed, pack in valid:
        out.extend(render_scenario(seed, pack))
    ids = [s["id"] for s in out]
    assert len(ids) == len(set(ids)), "duplicate sample ids"
    return out


def load_samples(path: str = paths.PACT) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_preview(samples: List[dict], path: str) -> None:
    """Human-review markdown: every cell exactly as a model under test sees
    it. Base mode is shown in full; the mandate sample of the same cell only
    differs by the directive appended to the system prompt."""
    base = [s for s in samples if s["mode"] == "base"]
    mandate = {s["item"]: s for s in samples if s["mode"] == "mandate"}
    by_pack: Dict[tuple, List[dict]] = {}
    for s in base:
        by_pack.setdefault((s["scenario"], s["generator_model"]), []).append(s)

    out: List[str] = [
        "# PACT - rendered sample preview",
        "",
        f"{len(samples)} samples ({len(base)} cells x 2 system-prompt modes) "
        "from {n} (scenario x generator) pack(s); every pack is one model's "
        "complete, independent version of its scenario. Each cell below is "
        "the conversation a model under test receives: SYSTEM + USER turn 1, "
        "then the pushback turn 2 if it complied. The mandate sample appends "
        "the directive shown to the same system prompt; nothing else "
        "differs.".replace("{n}", str(len(by_pack))),
    ]
    for sid, gen in sorted(by_pack):
        cells = by_pack[(sid, gen)]
        out += ["", "---", "",
                f"## {sid} - generated by `{gen}` - {len(cells)} cell(s)"]
        variants: Dict[str, List[str]] = {}
        for s in cells:
            variants.setdefault(s["system_prompt"],
                                []).append(f"{s['group']}.{s['pressure']}")
        for sp, names in variants.items():
            shown = ", ".join(names[:6]) + (", …" if len(names) > 6 else "")
            out += ["", f"**SYSTEM - {len(names)} cell(s): {shown}**", "",
                    "```text", sp, "```"]
        m = mandate.get(cells[0]["item"])
        if m:
            directive = m["system_prompt"][len(cells[0]["system_prompt"]):]
            out += ["", "**DIRECTIVE (mandate mode appends to the system "
                    "prompt):**", "", "```text", directive.strip(), "```"]

        for s in cells:
            comp = ", ".join(s["gold"]["compliant_choices"]) or "(none)"
            out += [
                "",
                f"### {s['group']}.{s['pressure']}",
                "",
                f"`group={s['group']}` `binds={s['binds']}` "
                f"`pressure={s['pressure']}`",
                "",
                f"- gold compliant picks: {comp}",
                f"- escalate_ok={s['gold']['escalate_ok']}",
                "",
                "**USER (turn 1):**",
                "",
                "```text",
                s["turn1"],
                "```",
            ]
            if s["pushback"]:
                out += ["", "**USER (turn 2, if the model complied):**", "",
                        "```text", s["pushback"], "```"]

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
                         "cell (does not write the sample JSONL)")
    args = ap.parse_args()

    samples = render_all(args.samples_dir, args.allow_partial)
    if args.preview:
        write_preview(samples, args.preview)
        n_packs = len({(s["scenario"], s["generator_model"]) for s in samples})
        print(f"{len(samples)} samples from {n_packs} (scenario x generator) "
              f"pack(s) -> {args.preview}")
        return
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    sha = hashlib.sha256(open(args.output, "rb").read()).hexdigest()
    by_group: Dict[str, int] = {}
    for s in samples:
        by_group[s["group"]] = by_group.get(s["group"], 0) + 1
    print(f"{len(samples)} samples -> {args.output}")
    print(f"  groups (x2 modes): {by_group}")
    print(f"  binding: {sum(s['binds'] for s in samples)} / "
          f"non-binding {sum(not s['binds'] for s in samples)}")
    print(f"  sha256 : {sha}")
    print("record this hash before running any model - it identifies this "
          "exact sample set.")


if __name__ == "__main__":
    main()
