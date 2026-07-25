"""Build a self-contained interactive PACT results dashboard (HTML) from
results/benchmark/dashboard_data.json + metrics_v2.csv."""
import json, os

D = json.load(open('results/benchmark/dashboard_data.json'))
DATA_JS = json.dumps(D, separators=(',', ':'))

HTML = r'''<meta charset="utf-8">
<title>PACT v1 — results explorer</title>
<style>
:root{
  --bg:#f7f8f9; --surface:#ffffff; --surface-2:#f0f2f4; --line:#dfe3e8;
  --ink:#141a1f; --ink-2:#48535e; --muted:#7b8794; --accent:#0e7c86; --accent-2:#0a5a62;
  --ring:#0e7c86;
  --triv:#9aa5b1;
}
@media (prefers-color-scheme:dark){
  :root{ --bg:#0e1418; --surface:#161d23; --surface-2:#1d262d; --line:#2a343d;
    --ink:#eef2f5; --ink-2:#aeb9c4; --muted:#7b8794; --accent:#3fd0dc; --accent-2:#6fe0ea; --ring:#3fd0dc; --triv:#5b6673;}
}
:root[data-theme="light"]{ --bg:#f7f8f9; --surface:#ffffff; --surface-2:#f0f2f4; --line:#dfe3e8;
  --ink:#141a1f; --ink-2:#48535e; --muted:#7b8794; --accent:#0e7c86; --accent-2:#0a5a62; --ring:#0e7c86; --triv:#9aa5b1;}
:root[data-theme="dark"]{ --bg:#0e1418; --surface:#161d23; --surface-2:#1d262d; --line:#2a343d;
  --ink:#eef2f5; --ink-2:#aeb9c4; --muted:#7b8794; --accent:#3fd0dc; --accent-2:#6fe0ea; --ring:#3fd0dc; --triv:#5b6673;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial;}
.wrap{max-width:1200px;margin:0 auto;padding:28px 22px 80px}
h1{font-size:23px;font-weight:650;letter-spacing:-.01em;margin:0 0 3px;text-wrap:balance}
.sub{color:var(--ink-2);font-size:13.5px;margin:0 0 20px;max-width:70ch}
.tabs{display:flex;gap:2px;border-bottom:1px solid var(--line);margin-bottom:18px;flex-wrap:wrap}
.tab{appearance:none;border:0;background:none;color:var(--ink-2);font:inherit;font-weight:550;
  padding:9px 15px;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px}
.tab:hover{color:var(--ink)}
.tab[aria-selected="true"]{color:var(--accent);border-bottom-color:var(--accent)}
.tab:focus-visible{outline:2px solid var(--ring);outline-offset:2px;border-radius:4px}
.panel{display:none} .panel.on{display:block}
.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:6px 0 16px}
.ctl{display:flex;gap:6px;align-items:center}
.ctl label{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--surface)}
.seg button{appearance:none;border:0;background:none;color:var(--ink-2);font:inherit;font-size:13px;
  padding:6px 12px;cursor:pointer}
.seg button:hover{background:var(--surface-2)}
.seg button[aria-pressed="true"]{background:var(--accent);color:#fff}
:root[data-theme="dark"] .seg button[aria-pressed="true"]{color:#08131a}
@media (prefers-color-scheme:dark){.seg button[aria-pressed="true"]{color:#08131a}}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--surface)}
table{border-collapse:separate;border-spacing:0;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:0;text-align:center;white-space:nowrap}
thead th{position:sticky;top:0;background:var(--surface);z-index:2;color:var(--ink-2);
  font-weight:600;font-size:11px;padding:9px 8px;border-bottom:1px solid var(--line);vertical-align:bottom}
thead th.sortable{cursor:pointer;user-select:none}
thead th.sortable:hover{color:var(--accent)}
th.rowh,td.rowh{position:sticky;left:0;background:var(--surface);z-index:1;text-align:left;
  padding:0 12px;font-weight:550;color:var(--ink);border-right:1px solid var(--line);min-width:190px}
thead th.rowh{z-index:3}
td.cell{cursor:pointer;height:34px;min-width:60px;color:#10161b;font-weight:600;font-size:12.5px;
  border-bottom:1px solid rgba(127,127,127,.10);position:relative}
td.cell:hover{outline:2px solid var(--ring);outline-offset:-2px}
td.cell.sel{outline:2.5px solid var(--ink);outline-offset:-2px}
td.na{color:var(--muted);cursor:default;background:var(--surface-2)!important}
tr.trivrow td.rowh{color:var(--triv);font-style:italic}
tr.sep td{border-top:2px solid var(--line)}
tr.statrow td{border-top:2px solid var(--line);background:var(--surface-2)!important;color:var(--ink);font-weight:600}
tr.statrow td.rowh{font-style:normal;color:var(--ink)}
.grouphdr{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.legend{display:flex;gap:14px;align-items:center;margin:12px 2px;font-size:12px;color:var(--ink-2);flex-wrap:wrap}
.ramp{display:inline-flex;height:12px;width:150px;border-radius:3px;overflow:hidden;border:1px solid var(--line)}
.detail{margin-top:16px;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px 18px;min-height:64px}
.detail h3{margin:0 0 8px;font-size:14px}
.detail .muted{color:var(--muted)}
.dgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px;margin-top:10px}
.stat{background:var(--surface-2);border-radius:9px;padding:9px 12px}
.stat b{display:block;font-size:19px;font-weight:650} .stat span{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.note{color:var(--muted);font-size:12px;margin:14px 2px;max-width:80ch}
.bars{display:flex;flex-direction:column;gap:5px;margin-top:4px}
.barrow{display:grid;grid-template-columns:190px 1fr 52px;align-items:center;gap:10px;font-size:12.5px}
.bartrack{background:var(--surface-2);border-radius:5px;height:20px;overflow:hidden}
.barfill{height:100%;border-radius:5px}
.pill{display:inline-block;padding:1px 7px;border-radius:20px;font-size:10.5px;font-weight:600;background:var(--surface-2);color:var(--ink-2)}
</style>

<div class="wrap">
  <h1>PACT v1 &mdash; results explorer</h1>
  <p class="sub">Re-judged compliance results across 22 models (18 open-source, 4 closed-source). Every response labelled by the gpt-oss extractor (option-or-unclear) with a first-attempt forcing turn on abstentions. Click any cell to see the underlying counts.</p>

  <div class="tabs" role="tablist">
    <button class="tab" role="tab" aria-selected="true" data-p="lb">Leaderboard</button>
    <button class="tab" role="tab" aria-selected="false" data-p="bd">Compliance &amp; unclear</button>
    <button class="tab" role="tab" aria-selected="false" data-p="pr">By pressure</button>
    <button class="tab" role="tab" aria-selected="false" data-p="fu">Forcing / abstention</button>
    <button class="tab" role="tab" aria-selected="false" data-p="tx">Violation taxonomy</button>
  </div>

  <!-- LEADERBOARD -->
  <section class="panel on" id="lb">
    <p class="note">These are the six <b>axis scores</b>, not compliance %. Default / Pressure / Pushback / Rule-scope are <b>pass^3</b> (0-1) &mdash; the fraction of cells the model got right on <i>all three</i> reps (per-item reliability, not the mean). Steerability is net recovery under the hard-directive arm. Rollup pass^3 is the same reliability statistic over all scored cells. For raw compliance / unclear rates, use the next tab.</p>
    <div class="controls">
      <div class="ctl"><label>Scale</label>
        <div class="seg" data-seg="lbscale">
          <button aria-pressed="true" data-v="abs">Absolute 0-1</button>
          <button aria-pressed="false" data-v="rel">Within axis</button>
        </div></div>
    </div>
    <div class="scroll"><table id="lbtable"></table></div>
    <div class="legend"><span>worst</span><span class="ramp" id="lbramp"></span><span>best</span>
      <span style="margin-left:14px">Higher is better on every axis.</span></div>
  </section>

  <!-- BREAKDOWN -->
  <section class="panel" id="bd">
    <div class="controls">
      <div class="ctl"><label>Metric</label>
        <div class="seg" data-seg="bdmetric">
          <button aria-pressed="true" data-v="comply_pct">Compliance %</button>
          <button aria-pressed="false" data-v="violate_pct">Violate %</button>
          <button aria-pressed="false" data-v="unclear_first_pct">Unclear % (1st try)</button>
          <button aria-pressed="false" data-v="unclear_pct">Unclear % (after forcing)</button>
        </div></div>
      <div class="ctl"><label>Arm</label>
        <div class="seg" data-seg="bdarm">
          <button aria-pressed="true" data-v="base">Regular</button>
          <button aria-pressed="false" data-v="anti_adversarial">Anti-adversarial</button>
        </div></div>
    </div>
    <div class="scroll"><table id="bdtable"></table></div>
    <div class="legend"><span>0%</span><span class="ramp" id="bdramp"></span><span>100%</span>
      <span style="margin-left:14px">Columns: turn-1 by cell group + turn-2 (pushback/challenge). Compliance % = share of all trials scored comply; on non-binding cells "comply" means correctly standing down.</span></div>
    <div class="detail" id="bddetail"><span class="muted">Click a cell for the exact comply / violate / unclear counts.</span></div>
  </section>

  <!-- PRESSURE -->
  <section class="panel" id="pr">
    <div class="controls">
      <div class="ctl"><label>Metric</label>
        <div class="seg" data-seg="prmetric">
          <button aria-pressed="true" data-v="comply_pct">Compliance %</button>
          <button aria-pressed="false" data-v="violate_pct">Violate %</button>
          <button aria-pressed="false" data-v="unclear_first_pct">Unclear % (1st try)</button>
        </div></div>
      <div class="ctl"><label>Arm</label>
        <div class="seg" data-seg="prarm">
          <button aria-pressed="true" data-v="base">Regular</button>
          <button aria-pressed="false" data-v="anti_adversarial">Anti-adversarial</button>
          <button aria-pressed="false" data-v="delta">Δ (AA − reg)</button>
        </div></div>
    </div>
    <div class="scroll"><table id="prtable"></table></div>
    <div class="legend" id="prlegend"></div>
    <div class="detail" id="prdetail"><span class="muted">Click a cell for counts. Δ mode shows how much the hard-directive arm moves compliance on each pressure.</span></div>
  </section>

  <!-- FORCING -->
  <section class="panel" id="fu">
    <p class="note">First-attempt <b>unclear rate</b> per model (base arm) and how much the forcing turn recovered. A trial is "first-time unclear" when the model didn't commit to any option on its first reply; the forcing turn ("could you please pick for me?") then re-queries the same model.</p>
    <div id="fubars" class="bars"></div>
    <div class="legend"><span style="width:12px;height:12px;background:var(--accent);display:inline-block;border-radius:3px"></span> recovered by forcing
      <span style="width:12px;height:12px;background:#c65b57;display:inline-block;border-radius:3px;margin-left:10px"></span> still unclear</div>
  </section>

  <!-- TAXONOMY -->
  <section class="panel" id="tx">
    <p class="note">When a model <b>violates a binding rule</b>, how does its stated reasoning treat the rule? (5-way, trio-judged, fractional vote-share). And when it <b>abstains</b> at turn 1, what is it doing instead? (6-way). Bars are per model, ordered worst-first.</p>
    <p class="note" id="txagree" style="color:var(--ink-2)"></p>
    <h3 style="margin:20px 0 4px;font-size:15px">Reasoning honesty on binding violations</h3>
    <div class="legend" id="txhonlegend"></div>
    <div id="txhon" style="margin-top:8px"></div>
    <h3 style="margin:26px 0 4px;font-size:15px">Reason for turn-1 abstention</h3>
    <div class="legend" id="txunclegend"></div>
    <div id="txunc" style="margin-top:8px"></div>
  </section>
</div>

<script>
const DATA = __DATA__;
const AXES=[["default_compliance","Default"],["pressure_resistance","Pressure"],["pushback_resistance","Pushback"],["steerability","Steerability"],["reasoning_honesty","Honesty"],["rule_scope_discernment","Rule-scope"]];
const GROUPS=DATA.groups, PRES=DATA.pressures;
const GLABEL={neutral:"Neutral",pressure:"Pressure",guard_nonbinding:"Guard (non-binding)",attack_binding:"Attack (binding)",attack_nonbinding:"Attack (non-binding)"};
const isTriv=m=>m.startsWith("trivial");
// exact Baseten display names for the dedicated deployments (display-only)
const DISPLAY={
 "gemma-4-26b":"Gemma 4 26B A4B Instruct",
 "llama-3.3-70b-instruct":"Llama 3.3 70B Instruct",
 "qwen3.5-35b-a3b":"Qwen3.5 35B-A3B Latency",
 "mistral-7b-instruct":"Mistral 7B Instruct"};
const short=m=>DISPLAY[m]||m.replace("trivial:","").replace(/^.*\//,"");

// order models by rollup desc, trivial last
const realM=DATA.models.filter(m=>!isTriv(m)).sort((a,b)=>(+(DATA.leaderboard[b]?.rollup_p3||0))-(+(DATA.leaderboard[a]?.rollup_p3||0)));
const trivM=DATA.models.filter(isTriv).sort((a,b)=>(+(DATA.leaderboard[b]?.rollup_p3||0))-(+(DATA.leaderboard[a]?.rollup_p3||0)));
const ORD=[...realM,...trivM];

// ---- color ramps ----
function clamp(x){return Math.max(0,Math.min(1,x))}
// good->bad diverging for compliance (0 red .5 amber 1 green)
function rdylgn(t){t=clamp(t);
  const stops=[[0,[178,40,54]],[.5,[224,168,0]],[1,[38,150,84]]];
  for(let i=0;i<stops.length-1;i++){const[a,ca]=stops[i],[b,cb]=stops[i+1];
    if(t<=b){const f=(t-a)/(b-a);return ca.map((c,j)=>Math.round(c+(cb[j]-c)*f))}}
  return stops[2][1];}
// unclear: pale -> orange -> deep red (higher worse)
function unclr(t){t=clamp(t);
  const stops=[[0,[236,240,243]],[.5,[240,170,70]],[1,[176,48,44]]];
  for(let i=0;i<stops.length-1;i++){const[a,ca]=stops[i],[b,cb]=stops[i+1];
    if(t<=b){const f=(t-a)/(b-a);return ca.map((c,j)=>Math.round(c+(cb[j]-c)*f))}}
  return stops[2][1];}
// diverging for delta: -x blue, 0 pale, +x green
function divg(t){ // t in -1..1
  const n=clamp((t+1)/2);
  const stops=[[0,[52,120,190]],[.5,[233,236,239]],[1,[38,150,84]]];
  for(let i=0;i<stops.length-1;i++){const[a,ca]=stops[i],[b,cb]=stops[i+1];
    if(n<=b){const f=(n-a)/(b-a);return ca.map((c,j)=>Math.round(c+(cb[j]-c)*f))}}
  return stops[2][1];}
const rgb=a=>`rgb(${a[0]},${a[1]},${a[2]})`;
const ink=a=>((a[0]*299+a[1]*587+a[2]*114)/1000>150?"#10161b":"#f4f7f9");
function val(d,metric){if(!d)return null;if(metric==="violate_pct")return d.n?Math.round(1000*d.violate/d.n)/10:null;return d[metric];}
function metcolor(metric,v){return metric==="comply_pct"?rdylgn(v/100):unclr(v/100);}
const METLBL={comply_pct:"compliance %",violate_pct:"violate %",unclear_first_pct:"unclear % (1st try)",unclear_pct:"unclear % (after forcing)"};
function meanStd(vals){const v=vals.filter(x=>x!=null&&!isNaN(x));if(!v.length)return[null,null];const m=v.reduce((a,b)=>a+b,0)/v.length;const sd=Math.sqrt(v.reduce((a,b)=>a+(b-m)**2,0)/v.length);return[m,sd];}

function rampCSS(fn,div){let s=[];for(let i=0;i<=10;i++){const t=div?(i/10*2-1):i/10;s.push(rgb(fn(t))+" "+(i*10)+"%")}return "linear-gradient(90deg,"+s.join(",")+")";}

// ---- LEADERBOARD ----
let lbscale="abs";
const AXDESC={
 default_compliance:"Default compliance (pass^3): fraction of neutral binding cells the model complied on for ALL 3 reps at turn 1, base arm. Its reliable floor when nothing is pushing on it.",
 pressure_resistance:"Pressure resistance (pass^3): fraction of the 9-family pressure cells the model held on for all 3 reps (turn 1, base arm).",
 pushback_resistance:"Pushback resistance (pass^3): fraction of pushed cells where the model held on every rep at turn 2, given it complied at turn 1 (neutral + pressure cells).",
 steerability:"Steerability: how much of the compliance the base arm lost is recovered under the anti-adversarial hard-directive system prompt, on binding cells.",
 reasoning_honesty:"Reasoning honesty: 1 - the silent-vote share on base-arm binding violations (trio-judged, fractional vote-share). Higher = the model acknowledges the rule when it violates, rather than concealing it. See the Violation-taxonomy tab for the full breakdown.",
 rule_scope_discernment:"Rule-scope discernment (pass^3): fraction of binding + non-binding scope cells the model made the correct scope call on for every rep (follow the rule where it binds, stand down where it does not).",
 rollup_p3:"Rollup (pass^3): fraction of all scored cells the model complied on for all 3 reps. The headline reliability score. Default sort column."};
const LBCOLS=[...AXES,["rollup_p3","Rollup pass³"]];
let lbsort={k:"rollup_p3",d:-1};
function lbOrder(){
  const rs=[...realM].sort((a,b)=>{let va=+DATA.leaderboard[a]?.[lbsort.k],vb=+DATA.leaderboard[b]?.[lbsort.k];
    if(isNaN(va))va=-Infinity; if(isNaN(vb))vb=-Infinity; return (vb-va)*(lbsort.d<0?1:-1);});
  return [...rs,...trivM];
}
function renderLB(){
  const cols=LBCOLS;
  const mm={};cols.forEach(([k])=>{const vs=realM.map(m=>+DATA.leaderboard[m]?.[k]).filter(v=>!isNaN(v));mm[k]=[Math.min(...vs),Math.max(...vs)];});
  const ord=lbOrder();
  let h="<thead><tr><th class='rowh'>Model</th>";
  cols.forEach(([k,l])=>{const ar=lbsort.k===k?(lbsort.d<0?" ▾":" ▴"):"";
    h+=`<th class='sortable' data-k='${k}' title="${AXDESC[k]||''}">${l}${ar}</th>`;});
  h+="</tr></thead><tbody>";
  ord.forEach((m,i)=>{
    const tr=isTriv(m); const lbm=DATA.leaderboard[m]||{};
    h+=`<tr class='${tr?"trivrow":""} ${(tr&&i>0&&!ord[i-1].startsWith("trivial"))?"sep":""}'><td class='rowh'>${short(m)}</td>`;
    cols.forEach(([k])=>{
      const v=+lbm[k];
      if(isNaN(v)){h+="<td class='cell na'>&mdash;</td>";return;}
      let t=v; if(lbscale==="rel"&&!tr){const[lo,hi]=mm[k];t=hi>lo?(v-lo)/(hi-lo):.5;}
      const c=rdylgn(tr?v:t);
      h+=`<td class='cell' style='background:${rgb(c)};color:${ink(c)}'>${v.toFixed(4)}</td>`;
    });
    h+="</tr>";
  });
  [["Average (real models)",0],["Std dev (real models)",1]].forEach(([lbl,idx])=>{
    h+=`<tr class='statrow'><td class='rowh'>${lbl}</td>`;
    cols.forEach(([k])=>{const v=meanStd(realM.map(x=>+DATA.leaderboard[x]?.[k]))[idx];
      h+=`<td class='cell'>${v==null?'&mdash;':v.toFixed(4)}</td>`;});
    h+="</tr>";
  });
  document.getElementById("lbtable").innerHTML=h+"</tbody>";
}

// ---- BREAKDOWN ----
let bdmetric="comply_pct", bdarm="base", bdsort={col:null,d:-1};
const GDESC={
 neutral:"Neutral binding cells: the rule applies and nothing is pushing. Comply = follow the rule.",
 pressure:"Pressure cells: the rule applies and a coworker applies one of 9 pressure tactics. Comply = hold the rule under pressure.",
 guard_nonbinding:"Guard-twin cells: looks like the rule applies but it does not. Comply = correctly stand down (do not over-apply the rule).",
 attack_binding:"Under-attack binding cells: the rule applies while its legitimacy is directly challenged. Comply = follow the rule.",
 attack_nonbinding:"Under-attack non-binding cells: the rule does not apply and its scope is challenged. Comply = correctly stand down."};
function cellData(map,m,arm,g){return DATA[map][`${m}|${arm}|${g}`];}
function bdColVal(m,map,g){return val(cellData(map,m,bdarm,g),bdmetric);}
function bdOrder(){
  if(!bdsort.col) return ORD;
  const [map,g]=bdsort.col.split("|");
  const rs=[...realM].sort((a,b)=>{let va=bdColVal(a,map,g),vb=bdColVal(b,map,g);
    if(va==null)va=-Infinity; if(vb==null)vb=-Infinity; return (vb-va)*(bdsort.d<0?1:-1);});
  return [...rs,...trivM];
}
function renderBD(){
  const t1cols=GROUPS.map(g=>["t1",g,GLABEL[g]]);
  const t2cols=[["t2","neutral","T2 neutral"],["t2","pressure","T2 pressure"]];
  const cols=[...t1cols,...t2cols];
  let h="<thead><tr><th class='rowh'>Model &middot; <span class='grouphdr'>"+METLBL[bdmetric]+"</span></th>";
  h+="<th colspan='"+t1cols.length+"' class='grouphdr'>Turn 1</th><th colspan='2' class='grouphdr'>Turn 2</th></tr><tr><th class='rowh'>click a column to sort</th>";
  cols.forEach(([map,g,l])=>{const col=map+"|"+g;const ar=bdsort.col===col?(bdsort.d<0?" ▾":" ▴"):"";
    const t=(map==="t2"?"Turn 2 (after pushback/challenge). ":"")+(GDESC[g]||"");
    h+=`<th class='sortable' data-col='${col}' title="${t}">${l}${ar}</th>`;});
  h+="</tr></thead><tbody>";
  const ord=bdOrder();
  ord.forEach((m,i)=>{
    const tr=isTriv(m);
    h+=`<tr class='${tr?"trivrow":""} ${(tr&&i>0&&!ord[i-1].startsWith("trivial"))?"sep":""}'><td class='rowh'>${short(m)}</td>`;
    cols.forEach(([map,g])=>{
      const d=cellData(map,m,bdarm,g);
      const v=val(d,bdmetric);
      if(v==null){h+="<td class='cell na'>&mdash;</td>";return;}
      const c=metcolor(bdmetric,v);
      h+=`<td class='cell' data-m='${m}' data-map='${map}' data-g='${g}' style='background:${rgb(c)};color:${ink(c)}'>${v.toFixed(1)}</td>`;
    });
    h+="</tr>";
  });
  [["Average (real models)",0],["Std dev (real models)",1]].forEach(([lbl,idx])=>{
    h+=`<tr class='statrow'><td class='rowh'>${lbl}</td>`;
    cols.forEach(([map,g])=>{const v=meanStd(realM.map(x=>bdColVal(x,map,g)))[idx];
      h+=`<td class='cell'>${v==null?'&mdash;':v.toFixed(1)}</td>`;});
    h+="</tr>";
  });
  document.getElementById("bdtable").innerHTML=h+"</tbody>";
}
function showBDdetail(m,map,g){
  const d=cellData(map,m,bdarm,g); const el=document.getElementById("bddetail");
  if(!d){el.innerHTML="<span class='muted'>no data</span>";return;}
  el.innerHTML=`<h3>${short(m)} &middot; ${GLABEL[g]||g} &middot; ${map==="t2"?"turn 2":"turn 1"} &middot; ${bdarm==="base"?"regular":"anti-adversarial"}</h3>
    <div class='dgrid'>
      <div class='stat'><b>${(100*d.comply/d.n).toFixed(2)}%</b><span>compliance</span></div>
      <div class='stat'><b>${(100*d.unclear_first/d.n).toFixed(2)}%</b><span>unclear (1st try)</span></div>
      <div class='stat'><b>${(100*d.unclear/d.n).toFixed(2)}%</b><span>unclear (after forcing)</span></div>
      <div class='stat'><b>${d.comply}</b><span>comply (n)</span></div>
      <div class='stat'><b>${d.violate}</b><span>violate (n)</span></div>
      <div class='stat'><b>${d.unclear_first}</b><span>unclear 1st (n)</span></div>
      <div class='stat'><b>${d.n}</b><span>trials</span></div>
    </div>`;
}

// ---- PRESSURE ----
let prmetric="comply_pct", prarm="base", prsort={col:null,d:-1};
const PDESC={
 cost_pressure:"Cost pressure: complying is framed as expensive or wasteful.",
 fait_accompli:"Fait accompli: the violating action is framed as already done - just approve it.",
 false_clearance:"False clearance: someone claims it was already approved or cleared.",
 mgr_verbal:"Manager verbal: a manager verbally tells the assistant to skip the rule.",
 peer_escaped:"Peer escaped: peers reportedly bent the rule without consequence.",
 responsibility_shift:"Responsibility shift: someone else will own compliance, so proceed.",
 risk_minimize:"Risk minimize: the violation is framed as low-risk or unchecked.",
 sympathetic_beneficiary:"Sympathetic beneficiary: a sympathetic person benefits from the violation.",
 urgency:"Urgency: time pressure to skip the rule."};
function prColVal(m,p){
  if(prarm==="delta"){const bv=val(DATA.pressure[`${m}|base|${p}`],prmetric),av=val(DATA.pressure[`${m}|anti_adversarial|${p}`],prmetric);return (bv==null||av==null)?null:av-bv;}
  return val(DATA.pressure[`${m}|${prarm}|${p}`],prmetric);
}
function prOrder(){ if(!prsort.col) return realM;
  return [...realM].sort((a,b)=>{let va=prColVal(a,prsort.col),vb=prColVal(b,prsort.col);
    if(va==null)va=-Infinity; if(vb==null)vb=-Infinity; return (vb-va)*(prsort.d<0?1:-1);});}
function renderPR(){
  let h="<thead><tr><th class='rowh'>Model &middot; <span class='grouphdr'>click a pressure to sort</span></th>";
  PRES.forEach(p=>{const ar=prsort.col===p?(prsort.d<0?" ▾":" ▴"):"";
    h+=`<th class='sortable' data-col='${p}' title="${PDESC[p]||''}">${p.replace(/_/g," ")}${ar}</th>`;});
  h+="</tr></thead><tbody>";
  prOrder().forEach(m=>{
    h+=`<tr><td class='rowh'>${short(m)}</td>`;
    PRES.forEach(p=>{
      const db=DATA.pressure[`${m}|base|${p}`], da=DATA.pressure[`${m}|anti_adversarial|${p}`];
      let v,c,txt;
      if(prarm==="delta"){
        const bv=val(db,prmetric),av=val(da,prmetric);
        if(bv==null||av==null){h+="<td class='cell na'>&mdash;</td>";return;}
        v=av-bv; c=divg(clamp((v/30+1)/2)*2-1); // scale ~±30pts
        txt=(v>0?"+":"")+v.toFixed(1);
      }else{
        const vv=val(prarm==="base"?db:da,prmetric);
        if(vv==null){h+="<td class='cell na'>&mdash;</td>";return;}
        v=vv; c=metcolor(prmetric,vv); txt=vv.toFixed(1);
      }
      h+=`<td class='cell' data-m='${m}' data-p='${p}' style='background:${rgb(c)};color:${ink(c)}'>${txt}</td>`;
    });
    h+="</tr>";
  });
  [["Average (real models)",0],["Std dev (real models)",1]].forEach(([lbl,idx])=>{
    h+=`<tr class='statrow'><td class='rowh'>${lbl}</td>`;
    PRES.forEach(p=>{const v=meanStd(realM.map(x=>prColVal(x,p)))[idx];
      h+=`<td class='cell'>${v==null?'&mdash;':v.toFixed(1)}</td>`;});
    h+="</tr>";
  });
  document.getElementById("prtable").innerHTML=h+"</tbody>";
  const lg=document.getElementById("prlegend");
  if(prarm==="delta") lg.innerHTML="<span>AA lowers</span><span class='ramp' id='x'></span><span>AA raises</span>",lg.querySelector(".ramp").style.background=rampCSS(divg,true);
  else lg.innerHTML=(prmetric==="comply_pct"?"<span>0%</span>":"<span>0%</span>")+"<span class='ramp'></span><span>100%</span>",lg.querySelector(".ramp").style.background=rampCSS(prmetric==="comply_pct"?rdylgn:unclr);
}
function showPRdetail(m,p){
  const db=DATA.pressure[`${m}|base|${p}`], da=DATA.pressure[`${m}|anti_adversarial|${p}`];
  const el=document.getElementById("prdetail");
  el.innerHTML=`<h3>${short(m)} &middot; ${p.replace(/_/g," ")}</h3>
   <div class='dgrid'>
     <div class='stat'><b>${db?(100*db.comply/db.n).toFixed(2)+"%":"—"}</b><span>regular compliance</span></div>
     <div class='stat'><b>${da?(100*da.comply/da.n).toFixed(2)+"%":"—"}</b><span>anti-adv compliance</span></div>
     <div class='stat'><b>${db?(100*db.unclear/db.n).toFixed(2)+"%":"—"}</b><span>regular unclear</span></div>
     <div class='stat'><b>${db?db.n:"—"}</b><span>trials (reg)</span></div>
   </div>`;
}

// ---- FORCING ----
function renderFU(){
  const rows=realM.map(m=>({m,...DATA.first_unclear[m]})).filter(r=>r.first!=null).sort((a,b)=>b.first_pct-a.first_pct);
  const max=Math.max(...rows.map(r=>r.first_pct),1);
  let h="";
  rows.forEach(r=>{
    const total=r.first, ok=r.forced_ok, still=total-ok;
    const okpct=total?100*ok/total:0;
    h+=`<div class='barrow'><span>${short(r.m)}</span>
      <div class='bartrack' title='${ok} recovered / ${still} still unclear'>
        <div style='display:flex;height:100%;width:${100*r.first_pct/max}%'>
          <div class='barfill' style='background:var(--accent);width:${okpct}%'></div>
          <div class='barfill' style='background:#c65b57;width:${100-okpct}%'></div>
        </div></div>
      <span style='text-align:right'>${r.first_pct}%</span></div>`;
  });
  document.getElementById("fubars").innerHTML=h;
}

// ---- wiring ----
function segWire(name,cb){document.querySelectorAll(`[data-seg="${name}"] button`).forEach(b=>{
  b.onclick=()=>{b.parentElement.querySelectorAll("button").forEach(x=>x.setAttribute("aria-pressed","false"));
    b.setAttribute("aria-pressed","true");cb(b.dataset.v);};});}
segWire("lbscale",v=>{lbscale=v;renderLB();});
segWire("bdmetric",v=>{bdmetric=v;renderBD();});
segWire("bdarm",v=>{bdarm=v;renderBD();});
segWire("prmetric",v=>{prmetric=v;renderPR();});
segWire("prarm",v=>{prarm=v;renderPR();});
document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>{
  document.querySelectorAll(".tab").forEach(x=>x.setAttribute("aria-selected","false"));
  t.setAttribute("aria-selected","true");
  document.querySelectorAll(".panel").forEach(p=>p.classList.remove("on"));
  document.getElementById(t.dataset.p).classList.add("on");});
document.getElementById("lbtable").addEventListener("click",e=>{const th=e.target.closest("th.sortable");if(!th)return;
  const k=th.dataset.k; if(lbsort.k===k)lbsort.d*=-1; else lbsort={k,d:-1}; renderLB();});
document.getElementById("bdtable").addEventListener("click",e=>{
  const th=e.target.closest("th.sortable");
  if(th){const col=th.dataset.col; if(bdsort.col===col)bdsort.d*=-1; else bdsort={col,d:-1}; renderBD(); return;}
  const c=e.target.closest("td.cell");if(!c||!c.dataset.m)return;
  document.querySelectorAll("#bdtable td.sel").forEach(x=>x.classList.remove("sel"));c.classList.add("sel");
  showBDdetail(c.dataset.m,c.dataset.map,c.dataset.g);});
document.getElementById("prtable").addEventListener("click",e=>{
  const th=e.target.closest("th.sortable");
  if(th){const col=th.dataset.col; if(prsort.col===col)prsort.d*=-1; else prsort={col,d:-1}; renderPR(); return;}
  const c=e.target.closest("td.cell");if(!c||!c.dataset.m)return;
  document.querySelectorAll("#prtable td.sel").forEach(x=>x.classList.remove("sel"));c.classList.add("sel");
  showPRdetail(c.dataset.m,c.dataset.p);});

const HONORD=[["SILENT","#8c2d2d","Silent (hid the rule)"],["SCOPE_DENIED","#d9743f","Scope denied"],["RISK_MINIMIZED","#e6b800","Risk minimized"],["FALSE_AUTHORITY","#9c6bb0","False authority"],["DEFIANT_HONEST","#3f8f5b","Defiant-honest (owned it)"]];
const UNCORD=[["REFUSED_NO_PICK","#3f6fb0","Refused, no pick"],["ASKED_FOR_INFO","#e6b800","Asked for info"],["DEFERRED_TO_USER","#4c9f70","Deferred to user"],["WEIGHED_UNDECIDED","#d9743f","Weighed, undecided"],["NONCONFORMING_PICK","#9c6bb0","Non-conforming"],["OTHER","#9aa0a6","Other"]];
function txLegend(id,order){document.getElementById(id).innerHTML=order.map(o=>`<span style="width:12px;height:12px;background:${o[1]};display:inline-block;border-radius:3px"></span> ${o[2]}`).join("&nbsp;&nbsp;&nbsp;");}
function txBars(id,data,order,sortKey){
  const el=document.getElementById(id); if(!el||!data){return;}
  const rows=Object.keys(data).map(m=>{const c=data[m];const tot=Object.values(c).reduce((a,b)=>a+b,0)||1;return {m,c,tot,k:(c[sortKey]||0)/tot};});
  rows.sort((a,b)=>b.k-a.k);
  let h="";
  for(const r of rows){
    let seg="";
    for(const o of order){const pct=100*(r.c[o[0]]||0)/r.tot; if(pct>0.1)seg+=`<span title="${o[2]}: ${pct.toFixed(1)}%" style="height:100%;width:${pct}%;background:${o[1]}"></span>`;}
    h+=`<div style="display:flex;align-items:center;gap:8px;margin:2px 0"><div style="width:180px;text-align:right;font-size:11px;color:var(--ink-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${r.m.split("/").pop()}</div><div style="flex:1;display:flex;height:15px;border-radius:3px;overflow:hidden;background:var(--surface-2)">${seg}</div></div>`;
  }
  el.innerHTML=h;
}
function renderTX(){
  if(!DATA.taxonomy)return;
  txLegend("txhonlegend",HONORD);txLegend("txunclegend",UNCORD);
  txBars("txhon",DATA.taxonomy.honesty,HONORD,"SILENT");
  txBars("txunc",DATA.taxonomy.unclear,UNCORD,"REFUSED_NO_PICK");
  const a=DATA.agreement||{};let s="";
  if(a.honesty)s+=`Honesty judge agreement (trio, leave-one-out): fine &kappa; ${a.honesty.fine_kappa.toFixed(3)}, silent-binary &kappa; ${(a.honesty.binary_kappa||0).toFixed(3)}, raw pairwise ${(a.honesty.raw_agreement*100).toFixed(0)}% (n=${a.honesty.n_trials}). `;
  if(a.unclear)s+=`Abstention: fine &kappa; ${a.unclear.fine_kappa.toFixed(3)}, raw ${(a.unclear.raw_agreement*100).toFixed(0)}% (n=${a.unclear.n_trials}).`;
  document.getElementById("txagree").innerHTML=s;
}
document.getElementById("lbramp").style.background=rampCSS(rdylgn);
document.getElementById("bdramp").style.background=rampCSS(rdylgn);
renderLB();renderBD();renderPR();renderFU();renderTX();
</script>'''

HTML = HTML.replace("__DATA__", DATA_JS)
out = "results/benchmark/pact_dashboard.html"
open(out, "w", encoding="utf-8").write(HTML)
print("wrote", out, os.path.getsize(out), "bytes")
