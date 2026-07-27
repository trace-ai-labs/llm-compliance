"""Scoped T2 re-extraction (fix for the stale keep/reuse pushback labels).

WHY
---
`rejudge --rerun-t2` re-extracted T1 (+ T1-forcing) under the current (rerun-2)
extractor, but its `keep` branch retained each trial's EXISTING T2 label whenever
the trial's `judge_prompt_sha` (a T1-only field) matched rerun-2 and the old T2
outcome was decided. That gate uses the T1 sha as a proxy for "T2 is current" - but
nothing re-verified the T2 extraction, so T2 labels produced by an earlier/worse
extractor were frozen in (106k `keep` + 2k `reuse` = 108k trials across 18 models).
Measured: ~6.5% of those disagree with a fresh current-extractor pass (vs ~1.4% for
freshly-regenerated T2s and ~1.1% for T1).

WHAT THIS DOES (and only this)
------------------------------
For every in-scope trial - a `pushback` T2 that was NOT freshly regenerated
(`t2_regenerated != True`) - it:
  1. RE-EXTRACTS the *stored* t2_response with the current OUTCOME_JUDGE_SYS
     extractor (BASETEN), using the same context pushback_t2_file uses
     (turn1_response = t1_force_response if t1_forced else t1_response;
      followup = the pushback message the model actually saw).
  2. ONLY IF that comes back `unclear`, runs the T2-push: re-asks the SAME model
     (DEPLOYED2) with the full history + FORCE_MSG, then extracts the forced reply
     (BASETEN). Decided-at-step-1 turns are never force-called.
  3. Writes t2_choice / t2_outcome / t2_judge (+ t2_force_* when forced) and stamps
     `t2_judge_prompt_sha` (a NEW, T2-specific sha) so the label's extractor is now
     auditable and the keep-gate bug cannot silently recur.

It does NOT regenerate the T2 response (the observed reply is kept - only the label
is refreshed), does NOT touch T1, and does NOT touch freshly-regenerated T2s.

CACHE
-----
All intermediate extractor/model results go to a DEDICATED dir
(results/benchmark/reextract_t2/...), so the stale rejudge caches are never read.
`run_batch` still resumes WITHIN this run (kill + rerun the same command safely);
it just never serves a prior round's result. Trial files are backed up to
results/benchmark/trials_prereext/ before rewrite.

TWO PHASES (so the extractor sweep needs no model deployments):
  --phase extract : re-extract every in-scope base t2_response AND any existing forced
                    reply, with the rerun-2 extractor. Extractor ONLY (BASETEN) - no
                    models required. Base turns that come back unclear with NO existing
                    forced reply are flagged `t2_needs_force=True` (label parked at
                    'unclear'/'pending_force') and COUNTED per model.
  --phase force   : for the flagged trials, generate the forced reply (DEPLOYED2) and
                    extract it (BASETEN), then resolve. Needs the models deployed.

So the flow is: extract on Qwen (verify the bug is gone) -> extract on all 18 (surface
per-model PENDING_FORCE counts) -> deploy those models -> force.

KEYS: extractor -> BASETEN_API_KEY, model re-ask (force phase) -> DEPLOYED2.

    python -m src.benchmark.reextract_t2 --phase extract --only "Qwen3.6 27B"   # canary
    python -m src.benchmark.reextract_t2 --phase extract --all                  # all 18, report pending
    python -m src.benchmark.reextract_t2 --phase force  --all                   # after models are up
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
from collections import Counter
from typing import Dict, List, Optional

from src.benchmark.batch import (BatchRequest, DEDICATED, DEFAULT_EXTRACT_MODEL,
                                  resolve_model, run_batch)
from src.benchmark.items import load_items
from src.benchmark.judges import (OUTCOME_JUDGE_SYS, outcome_judge_request,
                                   parse_outcome_judgment, prompt_sha, resolve_outcome)
from src.benchmark.rejudge import FORCE_MAX_TOKENS, FORCE_MSG, FORCE_TEMP
from src.benchmark.runner import TRIALS_DIR, TRIVIAL_PREFIX, arm_suffix

OUT_DIR = os.path.join("results", "benchmark", "reextract_t2")
BACKUP_DIR = os.path.join("results", "benchmark", "trials_prereext")
ITEMS_PATH = os.path.join("results", "benchmark", "items_v1_corrected.jsonl")
RR2_SHA = "2001b3020d01fdd2"   # expected prompt_sha(OUTCOME_JUDGE_SYS) - the rerun-2 extractor

DIFF_COLS = ["trial_id", "arm", "group", "binds", "pressure",
             "old_t2_choice", "old_t2_outcome", "old_t2_judge",
             "new_t2_choice", "new_t2_outcome", "new_t2_judge",
             "reextracted_to_unclear", "forced", "force_resolved"]


def _safe(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _sub(kind: str, safe: str, ext: str = "jsonl") -> str:
    d = os.path.join(OUT_DIR, kind)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{safe}.{ext}")


def _in_scope(t: dict) -> bool:
    """A pushback T2 whose label was NOT freshly regenerated under rerun-2."""
    return (not str(t.get("model", "")).startswith(TRIVIAL_PREFIX)
            and t.get("t2_script") == "pushback"
            and bool(t.get("t2_response"))
            and t.get("t2_regenerated") is not True)


def _names(item: dict) -> List[str]:
    return [o["name"] for o in item["options"]]


def _t1_ctx(t: dict) -> Optional[str]:
    """The T1 pick shown to the T2 judge as the 'earlier reply' - the forced pick
    when T1 was forced, else the original T1 reply (mirrors pushback_t2_file)."""
    return t.get("t1_force_response") if t.get("t1_forced") else t.get("t1_response")


def _pushback_msg(item: dict, t: dict) -> Optional[str]:
    """The exact user turn the stored t2_response was a reply to."""
    return t.get("t2_message") or item.get("t2_if_compliant")


def _force_history(item: dict, t: dict) -> List[dict]:
    """Full conversation up to and including the T2-forcing user turn."""
    system = item["system_prompt"] + arm_suffix(item, t["arm"])
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": item["turn1"]},
            {"role": "assistant", "content": t.get("t1_response")}]
    if t.get("t1_forced") and t.get("t1_force_response"):
        msgs.append({"role": "user", "content": FORCE_MSG})
        msgs.append({"role": "assistant", "content": t["t1_force_response"]})
    msgs.append({"role": "user", "content": _pushback_msg(item, t)})
    msgs.append({"role": "assistant", "content": t["t2_response"]})
    msgs.append({"role": "user", "content": FORCE_MSG})
    return msgs


def _h(*parts: Optional[str]) -> str:
    """Short content hash of the exact text(s) an extraction/generation consumes.
    Baked into every request id so `run_batch`'s resume-by-id can ONLY reuse a
    cached result when the input is byte-identical - a changed response/context/
    history yields a new id and forces a fresh call. This is the structural guard
    against the stale-cache bug (an id keyed on trial_id alone silently served a
    result computed from different text)."""
    m = hashlib.sha256()
    for p in parts:
        m.update((p or "").encode("utf-8"))
        m.update(b"\x1f")
    return m.hexdigest()[:12]


def process_file(path: str, items: Dict[str, dict], extract_model: str,
                 baseten_key: Optional[str], deployed2_key: Optional[str],
                 workers: int, extract_rpm: int, model_rpm: int,
                 max_tokens: int, dry_run: bool, phase: str, limit: int = 0,
                 served: Optional[set] = None) -> dict:
    trials: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trials.append(json.loads(line))
    if not trials:
        return {"file": os.path.basename(path), "trials": 0}
    model = trials[0].get("model")
    safe = _safe(path)
    sha = prompt_sha(OUTCOME_JUDGE_SYS)

    def _rewrite(update_ids: set, apply_fn, extra_cols: bool) -> Counter:
        """Rewrite the file, backing it up first; apply_fn(t, dw) mutates the in-scope
        rows and logs a diff. Rows not in update_ids are copied byte-for-byte."""
        os.makedirs(BACKUP_DIR, exist_ok=True)
        shutil.copy2(path, os.path.join(BACKUP_DIR, os.path.basename(path)))
        st = Counter()
        tmp = path + ".tmp"
        diff_path = _sub("diffs" if phase == "extract" else "diffs_force", safe, ext="csv")
        with open(tmp, "w", encoding="utf-8") as f, \
             open(diff_path, "w", newline="", encoding="utf-8") as df:
            dw = csv.writer(df); dw.writerow(DIFF_COLS)
            for t in trials:
                if t["trial_id"] in update_ids:
                    apply_fn(t, dw, st)
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        return st

    # ======================= PHASE: EXTRACT (extractor only) =======================
    if phase == "extract":
        scope = [t for t in trials if _in_scope(t) and t["item_id"] in items]
        if limit:
            scope = scope[:limit]
        if dry_run:
            print(f"  {safe}: {len(scope)} in-scope pushback T2s (of {len(trials)} rows)")
            return {"file": os.path.basename(path), "model": model,
                    "trials": len(trials), "in_scope": len(scope)}
        if not scope:
            return {"file": os.path.basename(path), "model": model, "in_scope": 0}

        # (A) re-extract each stored t2_response; id hashes the exact input.
        a_reqs, a_id = [], {}
        for t in scope:
            it = items[t["item_id"]]; ctx, foll = _t1_ctx(t), _pushback_msg(it, t)
            r = outcome_judge_request(f"{t['trial_id']}.{_h(t['t2_response'], ctx, foll)}",
                                      t["t2_response"], _names(it), extract_model, 2,
                                      id_prefix=f"rx2j.{safe}", turn1_response=ctx, followup=foll)
            a_reqs.append(r); a_id[t["trial_id"]] = r.id
        a_res = run_batch(a_reqs, _sub("t2j", safe), max_workers=workers,
                          rpm=extract_rpm, api_key=baseten_key, quiet=True)

        def step_a(t):
            it = items[t["item_id"]]
            row = a_res.get(a_id[t["trial_id"]])
            ch = parse_outcome_judgment(row["content"], _names(it)) if row else None
            return ch, resolve_outcome(ch, it["gold"], t["group"])

        # (B) for base-unclear trials that ALREADY have a forced reply, RE-EXTRACT that
        #     observed reply (extractor only). No model calls in this phase.
        unclear = [t for t in scope if step_a(t)[1] == "unclear"]
        efdict: Dict[str, tuple] = {}
        have_force = [t for t in unclear if t.get("t2_force_response")]
        if have_force:
            ef_reqs, ef_id = [], {}
            for t in have_force:
                txt = t["t2_force_response"]
                r = outcome_judge_request(f"{t['trial_id']}.{_h(txt, t['t2_response'], FORCE_MSG)}",
                                          txt, _names(items[t["item_id"]]), extract_model, 2,
                                          id_prefix=f"rx2fj.{safe}",
                                          turn1_response=t["t2_response"], followup=FORCE_MSG)
                ef_reqs.append(r); ef_id[t["trial_id"]] = r.id
            ef_res = run_batch(ef_reqs, _sub("forcej", safe), max_workers=workers,
                               rpm=extract_rpm, api_key=baseten_key, quiet=True)
            for t in have_force:
                it = items[t["item_id"]]
                row = ef_res.get(ef_id[t["trial_id"]])
                fch = parse_outcome_judgment(row["content"], _names(it)) if row else None
                efdict[t["trial_id"]] = (fch, resolve_outcome(fch, it["gold"], t["group"]))

        def apply_extract(t, dw, st):
            it = items[t["item_id"]]
            old_c, old_o, old_j = t.get("t2_choice"), t.get("t2_outcome"), t.get("t2_judge")
            existing_force = t.get("t2_force_response")             # capture before reset
            chA, outA = step_a(t)
            # reset all force/pending state; set below only as warranted
            for k in ("t2_force_response", "t2_force_choice", "t2_outcome_first"):
                t.pop(k, None)
            t["t2_forced"] = False; t["t2_needs_force"] = False
            forced_flag = force_resolved = pending = False
            if outA != "unclear":                                   # base decides it
                new_c, new_o, new_j = chA, outA, "llm"
            elif t["trial_id"] in efdict:                           # re-extract existing force
                fch, fout = efdict[t["trial_id"]]
                t["t2_force_response"] = existing_force             # keep the observed forced reply
                t["t2_force_choice"] = fch
                t["t2_outcome_first"] = "unclear"
                t["t2_forced"] = True
                forced_flag = True; force_resolved = (fout != "unclear")
                new_c, new_o, new_j = ((fch, fout, "llm_forced") if fout != "unclear"
                                       else (None, "unclear", "llm_forced_unclear"))
            else:                                                   # unclear, never forced -> pending
                new_c, new_o, new_j = None, "unclear", "pending_force"
                t["t2_needs_force"] = True; pending = True
            t.setdefault("t2_choice_reext_from", old_c)
            t.setdefault("t2_outcome_reext_from", old_o)
            t.update(t2_choice=new_c, t2_outcome=new_o, t2_judge=new_j,
                     t2_reextracted=True, t2_judge_prompt_sha=sha)
            st["rewritten"] += 1
            if (new_c, new_o) != (old_c, old_o):
                st["changed"] += 1
            st["reextracted_unclear"] += int(outA == "unclear")
            st["force_reextract"] += int(forced_flag)
            st["force_resolved"] += int(force_resolved)
            st["pending_force"] += int(pending)
            dw.writerow([t["trial_id"], t["arm"], t.get("group"), t.get("binds"),
                         t.get("pressure"), old_c, old_o, old_j, new_c, new_o, new_j,
                         outA == "unclear", forced_flag, force_resolved])

        st = _rewrite({t["trial_id"] for t in scope}, apply_extract, False)
        return {"file": os.path.basename(path), "model": model, "in_scope": len(scope), **st}

    # ======================= PHASE: FORCE (model re-ask, DEPLOYED2) ================
    # served != None -> only run models currently live on the shared endpoint; this
    # auto-defers DEDICATED models (not in the shared list) and any down shared model.
    if served is not None and model not in served:
        n = sum(1 for t in trials if t.get("t2_needs_force"))
        if n:
            why = "dedicated" if model in DEDICATED else "not live on shared endpoint"
            print(f"  {safe}: DEFERRED ({why}) - {n} pending forces")
        return {"file": os.path.basename(path), "model": model, "deferred": n}
    pending = [t for t in trials if t.get("t2_needs_force") and t["item_id"] in items]
    if limit:
        pending = pending[:limit]
    if dry_run:
        print(f"  {safe}: {len(pending)} trials pending a force generation")
        return {"file": os.path.basename(path), "model": model, "pending": len(pending)}
    if not pending:
        return {"file": os.path.basename(path), "model": model, "pending": 0}

    m_reqs, m_id = [], {}
    for t in pending:
        hist = _force_history(items[t["item_id"]], t)
        rid = f"rx2m.{safe}.{t['trial_id']}.{_h(json.dumps(hist, ensure_ascii=False))}"
        m_reqs.append(BatchRequest(id=rid, model=model, temperature=FORCE_TEMP,
                                   max_tokens=max_tokens, messages=hist,
                                   meta={"trial_id": t["trial_id"]}))
        m_id[t["trial_id"]] = rid
    m_res = run_batch(m_reqs, _sub("force", safe), max_workers=workers,
                      rpm=model_rpm, api_key=deployed2_key, quiet=True)

    fj_reqs, fj_id, gen_text = [], {}, {}
    for t in pending:
        row = m_res.get(m_id[t["trial_id"]])
        if row is None:
            continue
        gen_text[t["trial_id"]] = row["content"]
        r = outcome_judge_request(f"{t['trial_id']}.{_h(row['content'], t['t2_response'], FORCE_MSG)}",
                                  row["content"], _names(items[t["item_id"]]), extract_model, 2,
                                  id_prefix=f"rx2fj.{safe}", turn1_response=t["t2_response"], followup=FORCE_MSG)
        fj_reqs.append(r); fj_id[t["trial_id"]] = r.id
    fj_res = run_batch(fj_reqs, _sub("forcej", safe), max_workers=workers,
                       rpm=extract_rpm, api_key=baseten_key, quiet=True)

    def apply_force(t, dw, st):
        tid = t["trial_id"]
        if tid not in gen_text:
            st["gen_missing"] += 1
            return
        it = items[t["item_id"]]
        old_c, old_o, old_j = t.get("t2_choice"), t.get("t2_outcome"), t.get("t2_judge")
        row = fj_res.get(fj_id.get(tid, ""))
        fch = parse_outcome_judgment(row["content"], _names(it)) if row else None
        fout = resolve_outcome(fch, it["gold"], t["group"])
        t["t2_force_response"] = gen_text[tid]
        t["t2_force_choice"] = fch
        t["t2_outcome_first"] = "unclear"
        t["t2_forced"] = True
        t["t2_needs_force"] = False
        if fout != "unclear":
            new_c, new_o, new_j = fch, fout, "llm_forced"; st["force_resolved"] += 1
        else:
            new_c, new_o, new_j = None, "unclear", "llm_forced_unclear"
        t.update(t2_choice=new_c, t2_outcome=new_o, t2_judge=new_j, t2_judge_prompt_sha=sha)
        st["forced"] += 1
        if (new_c, new_o) != (old_c, old_o):
            st["changed"] += 1
        dw.writerow([tid, t["arm"], t.get("group"), t.get("binds"), t.get("pressure"),
                     old_c, old_o, old_j, new_c, new_o, new_j, True, True, fout != "unclear"])

    st = _rewrite({t["trial_id"] for t in pending}, apply_force, True)
    return {"file": os.path.basename(path), "model": model, "pending": len(pending), **st}


def main() -> None:
    ap = argparse.ArgumentParser(description="Scoped fresh re-extraction of stale keep/reuse pushback T2 labels.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--only", help="process only trial files whose name contains this substring")
    g.add_argument("--all", action="store_true", help="process every non-trivial trial file")
    ap.add_argument("--phase", choices=("extract", "force"), default="extract",
                    help="extract: re-extract base T2 + existing forced replies (extractor "
                         "ONLY, no models needed) and flag turns that need a new force. "
                         "force: generate the flagged forces (needs the models on DEPLOYED2).")
    ap.add_argument("--trials-dir", default=TRIALS_DIR)
    ap.add_argument("--items", default=ITEMS_PATH)
    ap.add_argument("--dry-run", action="store_true", help="count in-scope/pending trials; no calls, no writes")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--extract-rpm", type=int, default=10000,
                    help="gpt-oss extractor rpm cap (provider limit was raised to 10k)")
    ap.add_argument("--model-rpm", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=FORCE_MAX_TOKENS,
                    help="T2-push (forcing) model output cap; reasoning models need 8192")
    ap.add_argument("--limit", type=int, default=0,
                    help="canary: process at most N in-scope/pending trials per file (0 = all)")
    ap.add_argument("--live-only", action="store_true",
                    help="force phase: poll the shared Baseten /v1/models list and run "
                         "ONLY models currently served there (auto-defers dedicated + down "
                         "shared models). Reports what was deferred.")
    args = ap.parse_args()

    got = prompt_sha(OUTCOME_JUDGE_SYS)
    if got != RR2_SHA:
        print(f"WARNING: OUTCOME_JUDGE_SYS sha {got} != expected rerun-2 sha {RR2_SHA}. "
              f"Refusing to run with a non-rerun-2 extractor unless --dry-run.")
        if not args.dry_run:
            raise SystemExit(1)

    baseten_key = os.environ.get("BASETEN_API_KEY")
    deployed2_key = os.environ.get("DEPLOYED2")
    if not args.dry_run:
        if not baseten_key:
            raise SystemExit("need BASETEN_API_KEY (the gpt-oss extractor) in env")
        if args.phase == "force" and not deployed2_key:
            raise SystemExit("--phase force needs DEPLOYED2 (the models must be deployed) in env")
    extract_model = resolve_model(DEFAULT_EXTRACT_MODEL)

    served = None
    if args.phase == "force" and args.live_only and deployed2_key:
        from openai import OpenAI
        from src.benchmark.batch import BASE_URL
        c = OpenAI(base_url=BASE_URL, api_key=deployed2_key, timeout=30)
        served = {m.id for m in c.models.list().data}
        print(f"live shared /v1/models ({len(served)}): {', '.join(sorted(served))}")

    items = {it["item_id"]: it for it in load_items(args.items)}
    files = sorted(glob.glob(os.path.join(args.trials_dir, "*.jsonl")))
    files = [f for f in files if "trivial" not in os.path.basename(f)]
    if args.only:
        files = [f for f in files if args.only in os.path.basename(f)]
    if not files:
        raise SystemExit("no trial files matched")
    print(f"{'DRY-RUN: ' if args.dry_run else ''}phase={args.phase}; {len(files)} file(s); "
          f"extractor={extract_model} (sha {got})")

    totals = Counter()
    pend_by_model = {}
    for f in files:
        r = process_file(f, items, extract_model, baseten_key, deployed2_key,
                         args.workers, args.extract_rpm, args.model_rpm,
                         args.max_tokens, args.dry_run, args.phase, args.limit,
                         served)
        if not args.dry_run and args.phase == "extract" and r.get("in_scope"):
            pend_by_model[r["model"]] = r.get("pending_force", 0)
            print(f"  {r['file']:44s} in_scope={r.get('in_scope',0):5d} "
                  f"changed={r.get('changed',0):5d} reext_unclear={r.get('reextracted_unclear',0):5d} "
                  f"force_reextract={r.get('force_reextract',0):4d} "
                  f"PENDING_FORCE={r.get('pending_force',0):5d}")
        elif not args.dry_run and args.phase == "force" and r.get("pending"):
            print(f"  {r['file']:44s} pending={r.get('pending',0):5d} "
                  f"forced={r.get('forced',0):5d} force_resolved={r.get('force_resolved',0):5d} "
                  f"gen_missing={r.get('gen_missing',0):4d}")
        for k, v in r.items():
            if isinstance(v, int):
                totals[k] += v
    if args.phase == "extract":
        print(f"\nTOTAL (extract): in_scope={totals['in_scope']} changed={totals['changed']} "
              f"reext_unclear={totals['reextracted_unclear']} "
              f"force_reextract={totals['force_reextract']} (cheap, extractor) "
              f">>> PENDING_FORCE={totals['pending_force']} (need model gen in --phase force)")
        if not args.dry_run and pend_by_model:
            print("\nnew forces needed per model (deploy these on DEPLOYED2, then --phase force):")
            for m in sorted(pend_by_model, key=lambda k: -pend_by_model[k]):
                if pend_by_model[m]:
                    print(f"  {m:44s} {pend_by_model[m]:6d}")
    else:
        print(f"\nTOTAL (force): forced={totals['forced']} "
              f"force_resolved={totals['force_resolved']} gen_missing={totals['gen_missing']} "
              f"deferred(down/dedicated)={totals['deferred']}")
    if not args.dry_run:
        print(f"backups: {BACKUP_DIR}/  diffs: {OUT_DIR}/")


if __name__ == "__main__":
    main()
