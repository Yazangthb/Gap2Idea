# -*- coding: utf-8 -*-
"""Draw the human study sample and write blind rater pages. Design: docs/experiments/human_study_chosen_design.md

Writes to data/gold/human_study/ (gitignored):
  key_PRIVATE.json   item -> stratum and labels. Never give this to a rater.
  manifest.json      counts, seed and sample hash. Holds no labels.
  rater_R1.html, rater_R2.html   blind pages, one per rater. They hold no labels, sources or strata.
Prints counts only, so running this does not show anyone the labels.

    python scripts/dataset/human_study_sample.py
"""
import hashlib
import html
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "bench"))
OUT = ROOT / "data/gold/human_study"
SEED = 20261105
TARGETS = {"B1": 30, "B2": 30, "S1": 12, "S2": 40, "S3": 12}
POPULATIONS = {"B1": 56, "B2": 281, "S1": 174, "S2": 77, "S3": 12}
DECOYS_PER_TRACK = 6
MIN_DECOY_WORDS = 8


def norm(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def verdicts(path):
    return {r["i"]: r["keep"] for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def arxiv_items():
    from stage_c_nli_qwen import TAUS, candidates, setup
    rows, typed, _ = setup()
    pool = candidates(rows, typed, TAUS["r90"])
    exp = ROOT / "docs/experiments"
    runs = {m: verdicts(exp / f"stage_c_rubric_test_{m}.jsonl") for m in ("strict", "neutral", "keep_biased")}
    by_row = {c["row"]: n for n, c in enumerate(pool)}
    strata, decoys = {"B1": [], "B2": []}, []
    for i, r in enumerate(rows):
        if r["dom"] != "arxiv_cs":
            continue
        n = by_row.get(i)
        in_paper = r["pid"]
        before = rows[i - 1]["sentence"] if i and rows[i - 1]["pid"] == in_paper else ""
        after = rows[i + 1]["sentence"] if i + 1 < len(rows) and rows[i + 1]["pid"] == in_paper else ""
        item = {"track": "arxiv", "paper": in_paper, "idx": r["idx"], "sentence": r["sentence"], "before": before,
                "after": after, "gold_label": r["label"], "sal_label": None}
        if n is not None and runs["strict"][n]:
            strata["B1" if r["label"] == "none" else "B2"].append(item)
        elif r["label"] == "none" and not r["junk"] and len(r["sentence"].split()) >= MIN_DECOY_WORDS \
                and not (n is not None and any(runs[m][n] for m in runs)):
            decoys.append(item)
    return strata, decoys


def sal_items():
    from nli_sal_fws import KEEP_SECTIONS, OUT as NLI_OUT, SCORE_FILES, evc
    from stage_c_sal40_modes import GOLD
    rows = json.load(open(ROOT / "stage_c_package/sal_criteria/sal_test_allmodels.json", encoding="utf-8"))
    gold = {}
    for raw in GOLD.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            g = json.loads(raw)
            gold[(g["paper_id"], g["idx"])] = g["label"]
    seen, llm, idx = {}, [], []
    for r in rows:
        k = seen.get(r["pmcid"], 0)
        seen[r["pmcid"]] = k + 1
        idx.append(k)
        llm.append(gold.get((r["pmcid"], k), "none"))
    tau = json.loads(NLI_OUT.read_text(encoding="utf-8"))["thresholds"]["sal"]["tau"]
    s = evc(SCORE_FILES["sal_test"], "lim_c")
    ids = [i for i, r in enumerate(rows) if r["sec"] in KEEP_SECTIONS and s[i] >= tau]
    exp = ROOT / "docs/experiments"
    runs = {m: verdicts(exp / f"stage_c_sal40_modes_nli_sal_test_{m}.jsonl") for m in ("high_precision", "balanced", "high_recall")}
    position = {row: n for n, row in enumerate(ids)}
    strata, decoys = {"S1": [], "S2": [], "S3": []}, []
    for i, r in enumerate(rows):
        n = position.get(i)
        in_paper = r["pmcid"]
        before = rows[i - 1]["sent"] if i and rows[i - 1]["pmcid"] == in_paper else ""
        after = rows[i + 1]["sent"] if i + 1 < len(rows) and rows[i + 1]["pmcid"] == in_paper else ""
        item = {"track": "sal", "paper": in_paper, "idx": idx[i], "sentence": r["sent"], "before": before, "after": after,
                "gold_label": llm[i], "sal_label": r["label"]}
        if n is not None and runs["high_precision"][n]:
            stratum = "S1" if r["label"] == 1 else ("S2" if llm[i] != "none" else "S3")
            strata[stratum].append(item)
        elif r["sec"] in KEEP_SECTIONS and r["label"] == 0 and llm[i] == "none" and len(r["sent"].split()) >= MIN_DECOY_WORDS \
                and not (n is not None and any(runs[m][n] for m in runs)):
            decoys.append(item)
    return strata, decoys


# Tracked files that show a label next to sentence text.
LABELLED_PUBLIC = ("data/gold/human_validation_worksheet.csv", "scripts/bench/bench_stagec_opt.py")
PREFIX = 60


class PublicText:
    """Matches a sentence if its first 60 characters appear in a labelled public file, to catch small text differences."""

    def __init__(self):
        self.text = " ".join(norm((ROOT / f).read_text(encoding="utf-8", errors="ignore")) for f in LABELLED_PUBLIC)

    def __contains__(self, sentence_norm):
        return sentence_norm[:PREFIX] in self.text


def public_sentences():
    return PublicText()


def draw():
    arxiv, arxiv_decoys = arxiv_items()
    sal, sal_decoys = sal_items()
    strata = {**arxiv, **sal}
    found = {k: len(v) for k, v in strata.items()}
    if found != POPULATIONS:
        sys.exit(f"Stratum sizes changed, so the funnel output is not the frozen one: {found}")
    blocked, rng, chosen, report = public_sentences(), random.Random(SEED), [], {}
    used = set()
    # Smallest strata first, so a sentence in two strata stays where it is hardest to replace.
    for name in sorted(TARGETS, key=lambda k: POPULATIONS[k]):
        eligible = [x for x in strata[name] if norm(x["sentence"]) not in blocked and norm(x["sentence"]) not in used]
        rng.shuffle(eligible)
        take = eligible[:TARGETS[name]]
        used.update(norm(x["sentence"]) for x in take)
        chosen += [{**x, "stratum": name} for x in take]
        report[name] = {"population": POPULATIONS[name], "eligible": len(eligible), "sampled": len(take)}
    for track, pool in (("arxiv", arxiv_decoys), ("sal", sal_decoys)):
        eligible = [x for x in pool if norm(x["sentence"]) not in blocked and norm(x["sentence"]) not in used]
        rng.shuffle(eligible)
        take = eligible[:DECOYS_PER_TRACK]
        used.update(norm(x["sentence"]) for x in take)
        chosen += [{**x, "stratum": "D"} for x in take]
        report[f"D_{track}"] = {"eligible": len(eligible), "sampled": len(take)}
    rng.shuffle(chosen)
    for n, item in enumerate(chosen, 1):
        item["id"] = f"q{n:03d}"
    return chosen, report


GUIDE = [
    ("Limitation", "A weakness, caveat, bias or constraint the authors acknowledge about their own work. Examples: small sample, no blinding, single setting, short follow-up, limited generalizability, an unvalidated measure."),
    ("Future work", "An explicit direction the authors propose for future research."),
    ("Research gap", "An unsolved problem, missing capability or open challenge the authors point to in the field or in prior work."),
    ("Not a gap", "Everything else: background, methods, results, descriptions of related work, positive claims, recommendations, headers and fragments."),
]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentence rating</title>
<style>
:root{--bg:#f2f5f6;--card:#fff;--ink:#17242b;--mut:#5a6b75;--line:#d5dee3;--acc:#0f5c6e;--on:#fff;--mark:#fdeeb5;--warn:#a23b16}
@media (prefers-color-scheme:dark){:root{--bg:#10181c;--card:#18242a;--ink:#e6eef1;--mut:#9db0ba;--line:#30424b;--acc:#5ab4c8;--on:#08242c;--mark:#5c4a12;--warn:#f0a585;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 "Segoe UI",system-ui,sans-serif}
.wrap{max-width:780px;margin:0 auto;padding:16px}
header{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:12px}
header strong{font-size:17px}.sp{flex:1}
.bar{flex-basis:100%;height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;width:0;background:var(--acc)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin-bottom:12px}
details summary{cursor:pointer;font-weight:600}dl{margin:10px 0 0}dt{font-weight:600}dd{margin:0 0 8px;color:var(--mut)}
.ctx{color:var(--mut);font-size:15px;min-height:1em;margin:6px 0}
.target{background:var(--mark);color:var(--ink);padding:10px 12px;border-radius:6px;font-size:17px;margin:8px 0}
.q{font-weight:600;margin-top:16px}.row{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
button{font:inherit;font-weight:600;cursor:pointer;background:var(--card);color:var(--ink);border:1.5px solid var(--line);border-radius:8px;padding:8px 14px}
button:hover:not(:disabled){border-color:var(--acc)}button:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
button.sel{background:var(--acc);border-color:var(--acc);color:var(--on)}button:disabled{opacity:.5;cursor:not-allowed}
textarea{width:100%;margin-top:12px;font:inherit;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px;min-height:40px}
.hint{color:var(--mut);font-size:13px;margin-top:10px}.warn{color:var(--warn)}
</style></head><body><div class="wrap">
<header><strong>Sentence rating</strong><span id="who"></span><span class="sp"></span><span id="count"></span><div class="bar"><i id="prog"></i></div></header>
<div class="card"><details id="guide" open><summary>How to rate (please read first)</summary>
<p>Decide whether the highlighted sentence is a gap stated by the authors. Use the sentences around it only to understand it. Work alone and don't discuss items with anyone. Don't search for the paper. If you are unsure, choose your best judgement.</p>
<dl>__GUIDE__</dl></details></div>
<div class="card"><div class="ctx" id="before"></div><div class="target" id="target"></div><div class="ctx" id="after"></div>
<div class="q">Is the highlighted sentence a gap stated by the authors?</div>
<div class="row" id="gap"><button data-v="yes">Yes (Y)</button><button data-v="no">No (N)</button></div>
<div id="typeBox" hidden><div class="q">Which type?</div><div class="row" id="type"><button data-v="limitation">Limitation (1)</button><button data-v="future_work">Future work (2)</button><button data-v="research_gap">Research gap (3)</button></div></div>
<textarea id="notes" aria-label="Notes" placeholder="Notes (optional)"></textarea>
<div class="row"><button id="prev">Previous</button><button id="next">Next</button><button id="blank">Next unanswered</button><span class="sp"></span><button id="save" disabled>Save my answers</button></div>
<div class="hint">Answers save automatically in this browser. When every item is done, select "Save my answers" and send the downloaded file to the study organiser. <span class="warn" id="msg"></span></div>
</div></div>
<script>
const ITEMS=__ITEMS__,RATER=__RATER__,HASH=__HASH__;
const KEY="humanstudy_"+HASH.slice(0,12)+"_"+RATER;let at=0,ans={};
const $=s=>document.querySelector(s);
try{ans=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){$("#msg").textContent="This browser can't autosave. Please finish in one sitting."}
function store(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}}
function complete(a){return a&&(a.is_gap===false||(a.is_gap===true&&a.type))}
function draw(){const it=ITEMS[at],a=ans[it.id]||{};
$("#who").textContent="Rater "+RATER;$("#before").textContent=it.before;$("#target").textContent=it.sentence;$("#after").textContent=it.after;
const done=ITEMS.filter(x=>complete(ans[x.id])).length;
$("#count").textContent="Item "+(at+1)+" of "+ITEMS.length+" · "+done+" answered";$("#prog").style.width=(100*done/ITEMS.length)+"%";
document.querySelectorAll("#gap button").forEach(b=>b.classList.toggle("sel",(b.dataset.v==="yes"&&a.is_gap===true)||(b.dataset.v==="no"&&a.is_gap===false)));
$("#typeBox").hidden=a.is_gap!==true;
document.querySelectorAll("#type button").forEach(b=>b.classList.toggle("sel",b.dataset.v===a.type));
$("#notes").value=a.notes||"";$("#save").disabled=done!==ITEMS.length;$("#prev").disabled=at===0;$("#next").disabled=at===ITEMS.length-1;}
function rec(){const id=ITEMS[at].id;return ans[id]||(ans[id]={})}
function setGap(v){const a=rec();a.is_gap=v;if(!v)delete a.type;store();draw()}
function setType(t){const a=rec();a.is_gap=true;a.type=t;store();draw()}
function go(d){at=Math.max(0,Math.min(ITEMS.length-1,at+d));draw()}
document.querySelectorAll("#gap button").forEach(b=>b.onclick=()=>setGap(b.dataset.v==="yes"));
document.querySelectorAll("#type button").forEach(b=>b.onclick=()=>setType(b.dataset.v));
$("#notes").oninput=e=>{rec().notes=e.target.value;store()};$("#prev").onclick=()=>go(-1);$("#next").onclick=()=>go(1);
$("#blank").onclick=()=>{const k=ITEMS.findIndex(x=>!complete(ans[x.id]));if(k>=0){at=k;draw()}};
$("#save").onclick=()=>{const out={sample_hash:HASH,rater:RATER,finished_at:new Date().toISOString(),answers:ans};
const u=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:"application/json"}));const l=document.createElement("a");l.href=u;l.download="answers_"+RATER+".json";document.body.appendChild(l);l.click();l.remove();$("#msg").textContent="Saved answers_"+RATER+".json to your downloads."};
document.onkeydown=e=>{if(e.target.tagName==="TEXTAREA"||e.ctrlKey||e.metaKey||e.altKey)return;const k=e.key.toLowerCase();
if(k==="y")setGap(true);else if(k==="n")setGap(false);else if(k==="1")setType("limitation");else if(k==="2")setType("future_work");else if(k==="3")setType("research_gap");
else if(e.key==="ArrowRight")go(1);else if(e.key==="ArrowLeft")go(-1)};
draw();
</script></body></html>
"""


def blind(item):
    return {"id": item["id"], "sentence": item["sentence"], "before": item["before"], "after": item["after"]}


def write_page(path, items, rater, sample_hash, seed):
    order = [blind(x) for x in items]
    random.Random(f"{seed}-{rater}").shuffle(order)
    guide = "".join(f"<dt>{html.escape(a)}</dt><dd>{html.escape(b)}</dd>" for a, b in GUIDE)
    payload = json.dumps(order, ensure_ascii=False).replace("</", "<\\/")
    page = PAGE.replace("__GUIDE__", guide).replace("__ITEMS__", payload)
    page = page.replace("__RATER__", json.dumps(rater)).replace("__HASH__", json.dumps(sample_hash))
    path.write_text(page, encoding="utf-8")


def main():
    key_path = OUT / "key_PRIVATE.json"
    if key_path.exists():
        sys.exit(f"{key_path} already exists. The sample is frozen; delete it only before any rater has started.")
    OUT.mkdir(parents=True, exist_ok=True)
    items, report = draw()
    raw = json.dumps(items, ensure_ascii=False, indent=0).encode("utf-8")
    key_path.write_bytes(raw)
    sample_hash = hashlib.sha256(raw).hexdigest()
    manifest = {"seed": SEED, "sample_hash": sample_hash, "items": len(items), "strata": report,
                "populations": POPULATIONS, "funnel": "NLI + Qwen3.8-27B, high precision mode, test splits"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    for rater in ("R1", "R2"):
        write_page(OUT / f"rater_{rater}.html", items, rater, sample_hash, SEED)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
