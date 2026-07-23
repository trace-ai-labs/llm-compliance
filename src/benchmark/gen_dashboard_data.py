import json, csv, os, collections
from src.benchmark.judges import load_trials, TRIALS_DIR
TRIV="trivial:"
GROUPS=["neutral","pressure","guard_nonbinding","attack_binding","attack_nonbinding"]
ARMS=["base","anti_adversarial"]
PRESSURES=["cost_pressure","fait_accompli","false_clearance","mgr_verbal","peer_escaped",
           "responsibility_shift","risk_minimize","sympathetic_beneficiary","urgency"]
HELD={"comply","overcomply"}
tr=load_trials()
# trivial agents are excluded from the released dashboard
models=sorted({t["model"] for t in tr if not t["model"].startswith(TRIV)})
# leaderboard from metrics_v2.csv (all axis cols)
lb={}
LBCOLS=["default_compliance","pressure_resistance","pushback_resistance","steerability",
        "reasoning_honesty","rule_scope_discernment","rollup_p3","rollup_p3_lo","rollup_p3_hi"]
for r in csv.DictReader(open("results/benchmark/metrics_v2.csv",encoding='utf-8')):
    lb[r["model"]]={k:r.get(k,"") for k in LBCOLS if k in r}
def block(counter):
    c=counter; n=c["comply"]+c["violate"]+c["unclear"]
    nn=n or 1
    return {"n":n,"comply_pct":round(100*c["comply"]/nn,1),"unclear_pct":round(100*c["unclear"]/nn,1),
            "unclear_first_pct":round(100*c["unclear_first"]/nn,1),"comply":c["comply"],
            "violate":c["violate"],"unclear":c["unclear"],"unclear_first":c["unclear_first"]}
t1=collections.defaultdict(collections.Counter); t2=collections.defaultdict(collections.Counter)
pres=collections.defaultdict(collections.Counter); fu=collections.defaultdict(collections.Counter)
for t in tr:
    m=t["model"]; a=t["arm"]; g=t["group"]
    if m.startswith(TRIV): continue          # trivial agents excluded from the release

    o=t.get("t1_outcome"); oc="comply" if o in HELD else ("violate" if o=="violate" else "unclear")
    k=f"{m}|{a}|{g}"; t1[k][oc]+=1
    if t.get("first_time_unclear"): t1[k]["unclear_first"]+=1
    fu[m]["tot"]+=1
    if t.get("first_time_unclear"): fu[m]["first"]+=1
    if t.get("t1_forced") and t.get("t1_judge")=="llm_forced": fu[m]["forced_ok"]+=1
    if g=="pressure" and t.get("pressure") in PRESSURES:
        pk=f"{m}|{a}|{t['pressure']}"; pres[pk][oc]+=1
        if t.get("first_time_unclear"): pres[pk]["unclear_first"]+=1
    if t.get("t2_script")=="pushback":
        t2o=t.get("t2_outcome"); t2c="comply" if t2o in HELD else ("violate" if t2o=="violate" else "unclear")
        tk=f"{m}|{a}|{g}"; t2[tk][t2c]+=1
        if t.get("t2_first_time_unclear"): t2[tk]["unclear_first"]+=1
first_unclear={}
for m,c in fu.items():
    tot=c["tot"] or 1
    first_unclear[m]={"first_pct":round(100*c["first"]/tot,1),"forced_ok":c["forced_ok"],"first":c["first"]}
tax=json.load(open("results/benchmark/figures/taxonomy_summary.json",encoding='utf-8'))
out={"models":models,"groups":GROUPS,"arms":ARMS,"pressures":PRESSURES,
     "leaderboard":lb,
     "t1":{k:block(v) for k,v in t1.items()},
     "t2":{k:block(v) for k,v in t2.items()},
     "pressure":{k:block(v) for k,v in pres.items()},
     "first_unclear":first_unclear,
     "taxonomy":{"honesty":tax.get("honesty"),"unclear":tax.get("unclear")},
     "agreement":tax.get("agreement")}
json.dump(out,open("results/benchmark/dashboard_data.json","w",encoding='utf-8'),ensure_ascii=False)
print(f"wrote dashboard_data.json: {len(models)} models, {len(t1)} t1 cells, {len(t2)} t2, {len(pres)} pressure")
print("  leaderboard has reasoning_honesty:", "reasoning_honesty" in next(iter(lb.values())))
print("  taxonomy+agreement embedded:", bool(out['taxonomy']['honesty']), bool(out['agreement']))
