# -*- coding: utf-8 -*-
"""Build a self-contained annotate.html labeling UI for the blind human-validation pool.
Reproduces the EXACT seed-13 pool (same rows as worksheet_key_PRIVATE.csv / the Google Sheet),
but shows the target sentence HIGHLIGHTED inside a large paper-like block of the real text,
plus a link to the PMC article. Answers autosave to the browser and export to .xlsx with the
same columns as the worksheet (row, paper_id, section, is_gap, type, notes, annotator)."""
import sys, json, random, html
from urllib.parse import quote
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "dataset"))
from gold_label_full import load_corpus
GOLD = ROOT / "data" / "gold"
raw = {json.loads(l)["paper_id"]: json.loads(l) for l in (GOLD/"gap_gold.jsonl").read_text(encoding="utf-8").splitlines()}
corpus = {p["paper_id"]: p for p in load_corpus()}
pools = json.load(open(GOLD/"disagreement_pools.json"))
extras = [tuple(x) for x in pools["extras"]]; missed = [tuple(x) for x in pools["missed"]]
TYPES = ("limitation","future_work","research_gap"); PRI = {"limitation":0,"research_gap":1,"future_work":2}
def tmap(side):
    m={}
    for t in TYPES:
        for i in side.get(t,[]):
            if i not in m or PRI[t]<PRI[m[i]]: m[i]=t
    return m
ctrl_pos, negs = [], []
for pid,r in raw.items():
    if r["domain"]!="biomed_sal": continue
    g=set(r.get("sal_gold") or []); O=tmap(r["votes"]["opus"]); S=tmap(r["votes"]["sonnet"])
    both_lim={i for i in O if O.get(i)==S.get(i)=="limitation"}
    for i in sorted(both_lim & g): ctrl_pos.append((pid,i))
    flagged=set(O)|set(S)
    for i in range(r["n_sent"]):
        if i not in flagged and i not in g: negs.append((pid,i))
random.seed(13); random.shuffle(ctrl_pos); random.shuffle(negs)
ctrl_pos=ctrl_pos[:60]; negs=negs[:60]
pool=([("extra",p,i) for p,i in extras]+[("missed",p,i) for p,i in missed]
      +[("ctrl_pos",p,i) for p,i in ctrl_pos]+[("neg",p,i) for p,i in negs])
random.shuffle(pool)  # identical order to the worksheet/Sheet

def context_html(pid, i):
    """Large paper-like block: the whole contiguous run of the target's section (capped),
    target wrapped in <mark>. Falls back to a +/-12 window."""
    s = corpus[pid]["sents"]; secs = corpus[pid].get("secs")
    if secs:
        sec = secs[i]; lo = i
        while lo-1 >= 0 and secs[lo-1] == sec: lo -= 1
        hi = i
        while hi+1 < len(s) and secs[hi+1] == sec: hi += 1
        # cap window around target
        lo = max(lo, i-18); hi = min(hi, i+18)
    else:
        lo = max(0, i-12); hi = min(len(s)-1, i+12)
    parts = []
    for j in range(lo, hi+1):
        txt = html.escape(s[j])
        parts.append(f'<mark>{txt}</mark>' if j == i else txt)
    return " ".join(parts)

def frag_url(pid, sent):
    """Live PMC article deep-linked to the target sentence via a text fragment
    (#:~:text=start,end) so a supporting browser scrolls to + highlights it."""
    base = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pid}/"
    w = sent.split()
    if len(w) <= 10:
        return base + "#:~:text=" + quote(sent.strip(), safe="")
    start = " ".join(w[:6]); end = " ".join(w[-4:])
    return base + "#:~:text=" + quote(start, safe="") + "," + quote(end, safe="")

items = []
for n,(src,pid,i) in enumerate(pool,1):
    secs = corpus[pid].get("secs"); sect = secs[i] if secs else ""
    tgt = corpus[pid]["sents"][i]
    items.append(dict(row=n, paper_id=pid, section=sect,
                      pmc_url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pid}/",
                      frag_url=frag_url(pid, tgt),
                      target=tgt, context=context_html(pid, i)))
print(f"items: {len(items)}")

ITEMS_JS = json.dumps(items, ensure_ascii=False, separators=(",",":"))

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gap2Idea — gap annotation</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js"></script>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1a1d21;--mut:#6b7280;--line:#e5e7eb;--acc:#2563eb;
 --mark:#fde68a;--yes:#16a34a;--no:#dc2626;--chip:#eef2ff}
*{box-sizing:border-box}
body{margin:0;font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink)}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);
 padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header .sp{flex:1}
.bar{height:6px;background:var(--line);border-radius:4px;flex-basis:100%;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--acc);width:0}
.wrap{max-width:860px;margin:18px auto;padding:0 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 22px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.meta{display:flex;gap:8px;align-items:center;flex-wrap:wrap;color:var(--mut);font-size:13px;margin-bottom:14px}
.chip{background:var(--chip);color:#3730a3;border-radius:999px;padding:2px 10px;font-weight:600}
.ctx{font-size:17px;line-height:1.75;color:#23272e;max-height:46vh;overflow:auto;padding:4px 2px}
mark{background:var(--mark);padding:1px 3px;border-radius:3px;font-weight:600;box-shadow:0 0 0 1px #f59e0b inset}
.q{margin-top:18px;font-weight:600}
.opts{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px}
button.opt{cursor:pointer;border:1.5px solid var(--line);background:#fff;border-radius:10px;padding:9px 16px;font:inherit;font-weight:600}
button.opt:hover{border-color:var(--acc)}
button.opt.sel{color:#fff}
.y.sel{background:var(--yes);border-color:var(--yes)} .n.sel{background:var(--no);border-color:var(--no)}
.t.sel{background:var(--acc);border-color:var(--acc)}
#typeRow{display:none}
textarea{width:100%;margin-top:12px;border:1px solid var(--line);border-radius:10px;padding:8px 10px;font:inherit;resize:vertical;min-height:42px}
.nav{display:flex;gap:10px;align-items:center;margin-top:18px}
.nav .sp{flex:1}
button.nv{cursor:pointer;border:1px solid var(--line);background:#fff;border-radius:10px;padding:9px 16px;font:inherit;font-weight:600}
button.nv.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
a.link{color:var(--acc);text-decoration:none;font-weight:600}
a.link:hover{text-decoration:underline}
select,input[type=text]{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:8px}
.kbd{font-size:12px;color:var(--mut);margin-top:10px}
.kbd b{background:#f3f4f6;border:1px solid var(--line);border-radius:5px;padding:0 6px;font-weight:700}
.btn{cursor:pointer;border:1px solid var(--line);background:#fff;border-radius:9px;padding:7px 12px;font:inherit;font-weight:600}
.btn.go{background:#111827;color:#fff;border-color:#111827}
.done{color:var(--yes);font-weight:700}
small.warn{color:var(--no)}
</style></head>
<body>
<header>
 <strong>Gap annotation</strong>
 <label>Annotator
   <select id="annot"><option value="A1">A1</option><option value="A2">A2</option></select>
 </label>
 <span class="sp"></span>
 <span id="counter" class="chip">0 / 0</span>
 <span id="ans" class="done"></span>
 <span class="sp"></span>
 Go to <input id="jump" type="text" inputmode="numeric" style="width:64px" placeholder="row">
 <button class="btn" id="firstBlank">Next blank</button>
 <button class="btn go" id="export">Export .xlsx</button>
 <label class="btn" style="margin:0">Import .xlsx<input id="import" type="file" accept=".xlsx,.xls" hidden></label>
 <div class="bar"><i id="prog"></i></div>
</header>

<div class="wrap">
 <div class="card">
  <div class="meta">
    <span class="chip" id="rowc">row</span>
    <span id="pid"></span> · <span id="sec"></span>
    <span class="sp"></span>
    <a class="link" id="paperHi" target="_blank" rel="noopener">Open in paper · highlighted ↗</a>
    <a class="link" id="paper" target="_blank" rel="noopener" style="color:var(--mut);font-weight:500">(plain)</a>
  </div>
  <div class="ctx" id="ctx"></div>

  <div class="q">Does the <mark>highlighted</mark> sentence state a research gap of any kind?</div>
  <div class="opts" id="gapRow">
    <button class="opt y" data-v="Y">Yes &nbsp;<span style="opacity:.7">(y)</span></button>
    <button class="opt n" data-v="N">No &nbsp;<span style="opacity:.7">(n)</span></button>
  </div>

  <div id="typeRow">
   <div class="q">Type</div>
   <div class="opts">
     <button class="opt t" data-t="limitation">limitation <span style="opacity:.7">(1)</span></button>
     <button class="opt t" data-t="future_work">future_work <span style="opacity:.7">(2)</span></button>
     <button class="opt t" data-t="research_gap">research_gap <span style="opacity:.7">(3)</span></button>
   </div>
  </div>

  <textarea id="notes" placeholder="notes (optional)"></textarea>

  <div class="nav">
    <button class="nv" id="prev">← Prev</button>
    <div class="sp"></div>
    <button class="nv primary" id="next">Next →</button>
  </div>
  <div class="kbd">Shortcuts: <b>Y</b>/<b>N</b> gap · <b>1</b>/<b>2</b>/<b>3</b> type · <b>←</b>/<b>→</b> or <b>Enter</b> move.
   Autosaves in this browser. <small class="warn">Export to .xlsx before clearing site data.</small></div>
 </div>
</div>

<script>
const ITEMS = __ITEMS__;
const GUIDE = {
 limitation:"weakness/caveat/bias/constraint the authors acknowledge about THEIR OWN work",
 future_work:"an explicit direction the authors propose for FUTURE research",
 research_gap:"an unsolved problem / missing capability / open challenge in the field or prior work"};
let idx = 0;
const $ = s => document.querySelector(s);
const annotSel = $('#annot');
function key(){ return 'gap2idea_annot_' + annotSel.value; }
function load(){ try{ return JSON.parse(localStorage.getItem(key())||'{}'); }catch(e){ return {}; } }
function save(a){ try{ localStorage.setItem(key(), JSON.stringify(a)); }catch(e){} }
let ans = load();

function render(){
 const it = ITEMS[idx];
 $('#rowc').textContent = 'row ' + it.row;
 $('#pid').textContent = it.paper_id;
 $('#sec').textContent = it.section || '—';
 $('#paperHi').href = it.frag_url || it.pmc_url;
 $('#paper').href = it.pmc_url;
 $('#ctx').innerHTML = it.context;
 $('#ctx').scrollTop = 0;
 $('#counter').textContent = (idx+1) + ' / ' + ITEMS.length;
 const a = ans[it.row] || {};
 // gap buttons
 document.querySelectorAll('#gapRow .opt').forEach(b=>b.classList.toggle('sel', b.dataset.v===a.is_gap));
 // type row
 const showT = a.is_gap==='Y';
 $('#typeRow').style.display = showT ? 'block':'none';
 document.querySelectorAll('#typeRow .opt').forEach(b=>b.classList.toggle('sel', b.dataset.t===a.type));
 $('#notes').value = a.notes || '';
 // progress
 const done = ITEMS.filter(x=>{const r=ans[x.row]; return r && r.is_gap && (r.is_gap==='N'||r.type);}).length;
 $('#prog').style.width = (100*done/ITEMS.length).toFixed(1)+'%';
 $('#ans').textContent = done===ITEMS.length ? 'all '+done+' done ✓' : done+' answered';
 document.title = `[${idx+1}/${ITEMS.length}] Gap annotation`;
}
function setGap(v){
 const it=ITEMS[idx]; const a=ans[it.row]||(ans[it.row]={paper_id:it.paper_id,section:it.section});
 a.is_gap=v; if(v==='N') a.type='none'; else if(a.type==='none'||!a.type) a.type='';
 save(ans); render();
}
function setType(t){
 const it=ITEMS[idx]; const a=ans[it.row]||(ans[it.row]={paper_id:it.paper_id,section:it.section});
 a.is_gap='Y'; a.type=t; save(ans); render();
}
document.querySelectorAll('#gapRow .opt').forEach(b=>b.onclick=()=>setGap(b.dataset.v));
document.querySelectorAll('#typeRow .opt').forEach(b=>b.onclick=()=>setType(b.dataset.t));
$('#notes').oninput = e=>{const it=ITEMS[idx]; const a=ans[it.row]||(ans[it.row]={paper_id:it.paper_id,section:it.section}); a.notes=e.target.value; save(ans);};
function go(d){ idx=Math.max(0,Math.min(ITEMS.length-1, idx+d)); render(); }
$('#prev').onclick=()=>go(-1); $('#next').onclick=()=>go(1);
$('#jump').onchange=e=>{const r=parseInt(e.target.value,10); const k=ITEMS.findIndex(x=>x.row===r); if(k>=0){idx=k;render();} e.target.value='';};
$('#firstBlank').onclick=()=>{const k=ITEMS.findIndex(x=>{const a=ans[x.row]; return !(a&&a.is_gap&&(a.is_gap==='N'||a.type));}); if(k>=0){idx=k;render();}};
annotSel.onchange=()=>{ans=load(); idx=0; render();};
document.onkeydown=e=>{
 if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT'){ if(e.key==='Enter'&&e.target.id==='notes'){e.target.blur();} return; }
 const k=e.key.toLowerCase();
 if(k==='y')setGap('Y'); else if(k==='n')setGap('N');
 else if(k==='1')setType('limitation'); else if(k==='2')setType('future_work'); else if(k==='3')setType('research_gap');
 else if(e.key==='ArrowRight'||e.key==='Enter')go(1); else if(e.key==='ArrowLeft')go(-1);
};
$('#export').onclick=()=>{
 const rows=ITEMS.map(it=>{const a=ans[it.row]||{}; return {row:it.row,paper_id:it.paper_id,section:it.section,
   is_gap:a.is_gap||'',type:a.type||'',notes:a.notes||'',annotator:annotSel.value};});
 const ws=XLSX.utils.json_to_sheet(rows,{header:['row','paper_id','section','is_gap','type','notes','annotator']});
 const wb=XLSX.utils.book_new(); XLSX.utils.book_append_sheet(wb,ws,'answers');
 XLSX.writeFile(wb,'gap_annotations_'+annotSel.value+'.xlsx');
};
$('#import').onchange=e=>{
 const f=e.target.files[0]; if(!f)return; const rd=new FileReader();
 rd.onload=ev=>{const wb=XLSX.read(ev.target.result,{type:'array'});
   const rows=XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]]);
   rows.forEach(r=>{if(r.row!=null) ans[r.row]={paper_id:r.paper_id,section:r.section,is_gap:(r.is_gap||'')+'',type:(r.type||'')+'',notes:(r.notes||'')+''};});
   save(ans); render(); alert('Imported '+rows.length+' rows.');};
 rd.readAsArrayBuffer(f); e.target.value='';
};
render();
</script>
</body></html>"""

out = GOLD / "annotate.html"
out.write_text(HTML.replace("__ITEMS__", ITEMS_JS), encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
