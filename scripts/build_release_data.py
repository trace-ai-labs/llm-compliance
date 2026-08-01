"""One-time export of the PACT release data into benchmark_release/data.

Not part of the release itself. Produces:
  data/pact.jsonl                   the frozen dataset, one row per sample
                                    (cell x {base, mandate}; the mandate row
                                    materializes the hard directive into the
                                    system prompt)
  data/trials/<model>.jsonl         judged outcomes, response text stripped
  data/labels/transparency.jsonl    per-judge transparency votes (compact)
  data/labels/unclear_reasons.jsonl per-judge abstention-reason votes (compact)
  data/eval_awareness/summary.csv   realism-probe summary (renamed columns)

Model ids are normalized to the OpenRouter slugs the release code uses;
generator tags inside sample ids are normalized the same way; trial ids become
{sample}.{rep} with rep in 1..3; the mode value `anti_adversarial` becomes
`mandate`.
"""
import csv
import json
import os

REPO = r"C:\Users\mikad\Documents\GitHub\llm-compliance"
SRC = os.path.join(REPO, "results", "benchmark")
REL = os.path.join(REPO, "benchmark_release")

SLUG = {
    "deepseek-ai/DeepSeek-V4-Pro": "deepseek/deepseek-v4-pro",
    "zai-org/GLM-5.2": "z-ai/glm-5.2",
    "zai-org/GLM-5": "z-ai/glm-5",
    "zai-org/GLM-4.7": "z-ai/glm-4.7",
    "moonshotai/Kimi-K2.6": "moonshotai/kimi-k2.6",
    "moonshotai/Kimi-K2.7-Code": "moonshotai/kimi-k2.7-code",
    "thinkingmachines/inkling": "thinkingmachines/inkling",
    "openai/gpt-oss-120b": "openai/gpt-oss-120b",
    "nvidia/Nemotron-120B-A12B": "nvidia/nemotron-3-super",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B": "nvidia/nemotron-3-ultra",
    "MiniMaxAI/MiniMax-M2.5": "minimax/minimax-m2.5",
    "mistral-7b-instruct": "mistralai/mistral-7b-instruct",
    "llama-3.3-70b-instruct": "meta-llama/llama-3.3-70b-instruct",
    "llama-3.1-8b-instruct": "meta-llama/llama-3.1-8b-instruct",
    "qwen3.5-35b-a3b": "qwen/qwen3.5-35b-a3b",
    "Qwen3.6 27B": "qwen/qwen3.6-27b",
    "gemma-4-26b": "google/gemma-4-26b",
    "Seed-OSS-36B-Instruct": "bytedance/seed-oss-36b-instruct",
    "claude-haiku-4-5": "anthropic/claude-haiku-4.5",
    "gpt-5.6-luna": "openai/gpt-5.6-luna",
    "x-ai/grok-4.3": "x-ai/grok-4.3",
    "google/gemini-3-flash-preview": "google/gemini-3-flash-preview",
}

# generator tags embedded in item ids
TAG = {
    ".GLM-5.2.": ".glm-5.2.",
    ".Kimi-K2.6.": ".kimi-k2.6.",
    ".NVIDIA-Nemotron-3-Ultra-550B-A55B.": ".nemotron-3-ultra.",
}

MODE = {"base": "base", "anti_adversarial": "mandate"}

# aliases used in the private probe summary
ALIAS_SLUG = {
    "kimi-code": "moonshotai/kimi-k2.7-code", "gpt-oss": "openai/gpt-oss-120b",
    "glm": "z-ai/glm-5.2", "inkling": "thinkingmachines/inkling",
    "deepseek": "deepseek/deepseek-v4-pro", "kimi": "moonshotai/kimi-k2.6",
    "ultra": "nvidia/nemotron-3-ultra", "glm4.7": "z-ai/glm-4.7",
    "haiku": "anthropic/claude-haiku-4.5", "luna": "openai/gpt-5.6-luna",
    "grok": "x-ai/grok-4.3", "gemini": "google/gemini-3-flash-preview",
    "glm5": "z-ai/glm-5", "super": "nvidia/nemotron-3-super",
    "minimax": "minimax/minimax-m2.5",
    "llama8b": "meta-llama/llama-3.1-8b-instruct",
    "mistral": "mistralai/mistral-7b-instruct",
    "llama": "meta-llama/llama-3.3-70b-instruct",
    "qwen": "qwen/qwen3.5-35b-a3b", "gemma": "google/gemma-4-26b",
    "seedoss": "bytedance/seed-oss-36b-instruct", "qwen3.6": "qwen/qwen3.6-27b",
}

TRIO_TAGS = {"glm-5.2", "kimi-k2.6", "nvidia-nemotron-3-ultra-550b-a55b"}


def retag(item_id):
    for old, new in TAG.items():
        if old in item_id:
            return item_id.replace(old, new)
    return item_id


def new_trial_id(old_tid):
    """'item|anti_adversarial|r1' -> 'item.mandate.2' (rep becomes 1-based)."""
    item, mode, rep = old_tid.split("|")
    return f"{retag(item)}.{MODE[mode]}.{int(rep[1:]) + 1}"


def parse_label(raw, labels):
    raw = (raw or "").upper()
    best = None
    for lab in labels:
        i = raw.rfind(lab)
        if i >= 0 and (best is None or i > best[0]):
            best = (i, lab)
    return best[1] if best else None


def export_pact():
    out = os.path.join(REL, "data", "pact.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = 0
    with open(os.path.join(SRC, "items_v1_corrected.jsonl"),
              encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            item = retag(it["item_id"])
            for mode in ("base", "mandate"):
                row = {
                    "id": f"{item}.{mode}",
                    "item": item,
                    "mode": mode,
                    "scenario": it["scenario_id"],
                    "domain": it["domain"],
                    "group": it["group"],
                    "binds": it["binds"],
                    "pressure": it["pressure"],
                    "rule": it["rule"],
                    "system_prompt": it["system_prompt"] + (
                        it["hard_directive"] if mode == "mandate" else ""),
                    "turn1": it["turn1"],
                    "options": it["options"],
                    "gold": it["gold"],
                    "pushback": it["t2_if_compliant"],
                    "generator_model": SLUG[it["generator_model"]],
                    "canary": it["canary"],
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(f"data/pact.jsonl: {n} samples")


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
                item = retag(r["item_id"])
                mode = MODE[r["arm"] if "arm" in r else r["mode"]]
                rep = int(r["rep"]) + 1
                slim = {
                    "trial_id": f"{item}.{mode}.{rep}",
                    "sample": f"{item}.{mode}",
                    "item": item,
                    "scenario": r["scenario_id"],
                    "domain": r["domain"],
                    "group": r["group"],
                    "binds": r["binds"],
                    "pressure": r["pressure"],
                    "model": SLUG[r["model"]],
                    "mode": mode,
                    "rep": rep,
                    "t1_choice": r.get("t1_choice"),
                    "t1_outcome": r.get("t1_outcome"),
                    "t1_forced": r.get("t1_forced"),
                    "t2_script": r.get("t2_script"),
                    "t2_choice": r.get("t2_choice"),
                    "t2_outcome": r.get("t2_outcome"),
                    "t2_forced": r.get("t2_forced"),
                }
                # challenge-branch T2s were never scored; drop their fields
                if slim["t2_script"] != "pushback":
                    slim["t2_script"] = "none"
                    slim["t2_choice"] = None
                    slim["t2_outcome"] = None
                    slim["t2_forced"] = None
                rows.append(slim)
        if not rows:
            continue
        out_fn = rows[0]["model"].replace("/", "_") + ".jsonl"
        with open(os.path.join(out_dir, out_fn), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_rows += len(rows)
        print(f"  trials/{out_fn}: {len(rows)} rows")
    print(f"trials: {n_rows} rows kept, {n_excl} excluded rows dropped")


def export_labels(src_name, labels, out_name, prefix):
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
            if judge.split("/")[-1].lower() not in TRIO_TAGS:
                skipped += 1
                continue
            lab = parse_label(r.get("content", ""), labels)
            if not lab:
                skipped += 1
                continue
            # meta.trial_id = "<model>||<trial>(||t2)?"
            parts = meta["trial_id"].split("||")
            model, tid = SLUG[parts[0]], new_trial_id(parts[1])
            utid = "||".join([model, tid] + parts[2:])
            jslug = SLUG[judge]
            row = {"id": f"{prefix}.{jslug.split('/')[-1]}.{utid}", "ok": True,
                   "content": f"LABEL: {lab}",
                   "meta": {"trial_id": utid, "judge_model": jslug}}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"labels/{out_name}: {n} votes kept, {skipped} skipped")


def export_awareness():
    colmap = {"C0": "native", "C2": "announced", "dR2": "d_announced",
              "C1": "stripped", "dR1": "d_stripped"}

    def newcol(c):
        for old, new in colmap.items():
            if c == old or c.startswith(old + "_"):
                return c.replace(old, new, 1)
        return c

    src = os.path.join(SRC, "realism_probe", "summary.csv")
    out_dir = os.path.join(REL, "data", "eval_awareness")
    os.makedirs(out_dir, exist_ok=True)
    with open(src, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        cols = f.seek(0) or next(csv.reader(open(src, newline="",
                                                 encoding="utf-8")))
    new_cols = [newcol(c) for c in cols]
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(new_cols)
        for r in rows:
            r["model"] = ALIAS_SLUG[r["model"]]
            w.writerow([r[c] for c in cols])
    print(f"data/eval_awareness/summary.csv: {len(rows)} rows")


def main():
    export_pact()
    export_trials()
    export_labels("transparency.jsonl",
                  ("TRANSPARENT", "CONCEALED", "MISREPRESENTED"),
                  "transparency.jsonl", "trans")
    export_labels("unclear_reasons.jsonl",
                  ("REFUSED_ON_POLICY", "NONCONFORMING_PICK", "HANDED_BACK",
                   "NO_ANSWER"),
                  "unclear_reasons.jsonl", "unclear")
    export_awareness()


if __name__ == "__main__":
    main()
