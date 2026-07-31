"""One-time export of the PACT release data into benchmark_release/data.

Not part of the release itself. Produces:
  data/items.jsonl                  the frozen 1,682-cell corpus (copied)
  data/trials/<model>.jsonl         judged outcomes, response text stripped
  data/labels/transparency.jsonl    per-judge transparency votes (compact)
  data/labels/unclear_reasons.jsonl per-judge abstention-reason votes (compact)
  data/labels/guard_log.jsonl       generation guard verdicts (feedback stripped)
  data/eval_awareness/summary.csv   realism-probe summary (copied)
  results/*.csv                     canonical pipeline outputs (copied, renamed)
"""
import json
import os
import shutil

REPO = r"C:\Users\mikad\Documents\GitHub\llm-compliance"
SRC = os.path.join(REPO, "results", "benchmark")
REL = os.path.join(REPO, "benchmark_release")

TRIAL_KEEP = [
    "trial_id", "item_id", "scenario_id", "domain", "group", "binds",
    "pressure", "split", "model", "arm", "rep",
    "t1_choice", "t1_outcome", "t1_forced",
    "t2_script", "t2_choice", "t2_outcome", "t2_forced",
]

TRIO_TAGS = {"glm-5.2", "kimi-k2.6", "nvidia-nemotron-3-ultra-550b-a55b"}


def model_tag(model):
    return model.split("/")[-1].lower()


def parse_label(raw, labels):
    raw = (raw or "").upper()
    best = None
    for lab in labels:
        i = raw.rfind(lab)
        if i >= 0 and (best is None or i > best[0]):
            best = (i, lab)
    return best[1] if best else None


def export_trials():
    out_dir = os.path.join(REL, "data", "trials")
    os.makedirs(out_dir, exist_ok=True)
    n_rows = n_excl = 0
    for fn in sorted(os.listdir(os.path.join(SRC, "trials"))):
        if not fn.endswith(".jsonl") or fn.startswith("trivial_"):
            continue
        rows = []
        with open(os.path.join(SRC, "trials", fn), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("excluded"):
                    n_excl += 1
                    continue
                slim = {k: r.get(k) for k in TRIAL_KEEP}
                # challenge-branch T2s were never scored; drop their fields
                if slim["t2_script"] not in ("pushback",):
                    slim["t2_script"] = "none"
                    slim["t2_choice"] = None
                    slim["t2_outcome"] = None
                    slim["t2_forced"] = None
                rows.append(slim)
        with open(os.path.join(out_dir, fn), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_rows += len(rows)
        print(f"  trials/{fn}: {len(rows)} rows")
    print(f"trials: {n_rows} rows kept, {n_excl} excluded rows dropped")


def export_labels(src_name, labels, out_name):
    out = os.path.join(REL, "data", "labels", out_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = skipped = 0
    with open(os.path.join(SRC, src_name), encoding="utf-8") as fin, \
            open(out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("ok"):
                continue
            meta = r.get("meta", {})
            judge = meta.get("judge_model", "")
            if model_tag(judge) not in TRIO_TAGS:
                skipped += 1
                continue
            lab = parse_label(r.get("content", ""), labels)
            if not lab:
                skipped += 1
                continue
            row = {"id": r.get("id"), "ok": True, "content": f"LABEL: {lab}",
                   "meta": {"trial_id": meta.get("trial_id"),
                            "judge_model": judge}}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"labels/{out_name}: {n} votes kept, {skipped} skipped")


def export_guard_log():
    keep = ["scenario_id", "domain", "component", "component_kind",
            "pressure_key", "attempt", "generator_model", "guard_model",
            "verdict", "co_guard_model", "co_verdict", "guards_agree",
            "accepted"]
    out = os.path.join(REL, "data", "labels", "guard_log.jsonl")
    n = 0
    with open(os.path.join(SRC, "guard_log.jsonl"), encoding="utf-8") as fin, \
            open(out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fout.write(json.dumps({k: r.get(k) for k in keep},
                                  ensure_ascii=False) + "\n")
            n += 1
    print(f"labels/guard_log.jsonl: {n} rows")


def main():
    os.makedirs(os.path.join(REL, "data"), exist_ok=True)
    shutil.copyfile(os.path.join(SRC, "items_v1_corrected.jsonl"),
                    os.path.join(REL, "data", "items.jsonl"))
    print("data/items.jsonl copied")

    export_trials()
    export_labels("transparency.jsonl",
                  ("TRANSPARENT", "CONCEALED", "MISREPRESENTED"),
                  "transparency.jsonl")
    export_labels("unclear_reasons.jsonl",
                  ("REFUSED_ON_POLICY", "NONCONFORMING_PICK", "HANDED_BACK",
                   "NO_ANSWER"),
                  "unclear_reasons.jsonl")
    export_guard_log()

    aw = os.path.join(REL, "data", "eval_awareness")
    os.makedirs(aw, exist_ok=True)
    shutil.copyfile(os.path.join(SRC, "realism_probe", "summary.csv"),
                    os.path.join(aw, "summary.csv"))
    print("data/eval_awareness/summary.csv copied")

    res = os.path.join(REL, "results")
    os.makedirs(res, exist_ok=True)
    for src, dst in [("metrics_v2.csv", "metrics.csv"),
                     ("contrasts_v2.csv", "contrasts.csv"),
                     ("axis_ci.csv", "axis_ci.csv"),
                     ("dist_domain_axis.csv", "dist_domain_axis.csv"),
                     ("dist_pressure.csv", "dist_pressure.csv"),
                     ("dist_model_domain.csv", "dist_model_domain.csv"),
                     ("dist_model_pressure.csv", "dist_model_pressure.csv"),
                     ("dist_domain_pressure.csv", "dist_domain_pressure.csv")]:
        shutil.copyfile(os.path.join(SRC, src), os.path.join(res, dst))
        print(f"results/{dst} copied")


if __name__ == "__main__":
    main()
