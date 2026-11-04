# -*- coding: utf-8 -*-
"""Build the gap gold dataset: 2-judge Claude ensemble (Opus 5.5 + Sonnet 5.5) labels every
sentence of each paper as limitation / future_work / research_gap / none.

Corpus: 190 arXiv CS/ML (data/paper_texts.jsonl) + 40 SAL biomedical (sal_test_allmodels.json).
Per paper: numbered sentences -> each judge returns JSON {type:[{i,q}]} -> map by index, verify
by quote. Writes one JSON line per paper to data/gold/gap_gold.jsonl (resumable: done papers are
skipped). Agreement = confidence; disagreements are kept for later adjudication + human validation.

    python scripts/dataset/gold_label_full.py                # full run
    python scripts/dataset/gold_label_full.py --limit 5      # smoke test
"""
import os, sys, json, re, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
import anthropic
from gap2idea.pipeline.gap_funnel import split_sentences

OUT = ROOT / "data" / "gold" / "gap_gold.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)
JUDGES = [("opus", "claude-opus-5-5", 4.0, 20.0), ("sonnet", "claude-sonnet-5-5", 2.0, 10.0)]
TYPES = ("limitation", "future_work", "research_gap")
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

GUIDE = (
 "You label the sentences of a research paper for RESEARCH GAPS. Three gap types:\n"
 "- limitation: a weakness/caveat/bias/constraint the authors acknowledge about THEIR OWN work "
 "(small sample, no blinding, single setting, short follow-up, confounding, selection bias, limited "
 "generalizability, unvalidated measures, assumptions, failure modes...). Tag the sentence stating the "
 "cause; include a consequence sentence only if its cause is evident in it or its neighbors.\n"
 "- future_work: an explicit direction the authors propose for FUTURE research.\n"
 "- research_gap: an unsolved problem / missing capability / open challenge the authors point to in the "
 "field or in prior work (not necessarily their own study).\n"
 "Everything else is none (background, methods, results, related-work description, positive claims, "
 "recommendations, section headers). A sentence gets at most one type; if torn, prefer "
 "limitation > research_gap > future_work.\n"
 "Return ONLY a JSON object, no prose:\n"
 '{"limitation":[{"i":<index>,"q":"<first ~6 words of the sentence>"}],"future_work":[...],"research_gap":[...]}'
)

def trim_refs(text):
    low = text.lower(); n = len(text)
    for kw in ("\nreferences", "\nbibliography", "references\n", "\nacknowledgment", "\nacknowledgement"):
        p = low.rfind(kw)
        if p > n * 0.5: n = min(n, p)
    return text[:n]

def esc_map(i, q, sents):
    if not isinstance(i, int) or not (0 <= i < len(sents)):
        q2 = (q or "").lower().strip()
        cand = [k for k, s in enumerate(sents) if q2[:20] and q2[:20] in s.lower()]
        return cand[0] if cand else None
    q2 = (q or "").lower().strip()
    if not q2 or q2[:12] in sents[i].lower() or sents[i].lower()[:40] in (q2 + " "): return i
    cand = [k for k, s in enumerate(sents) if q2[:20] and q2[:20] in s.lower()]
    return cand[0] if cand else i

def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m: return {}
    try: return json.loads(m.group(0))
    except Exception: return {}

def judge(model, transcript, nsent, sents):
    resp = client.messages.create(model=model, max_tokens=8000, system=GUIDE,
        messages=[{"role": "user", "content":
            f"Paper ({nsent} numbered sentences). Label every gap sentence.\n\n{transcript}"}])
    txt = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    d = parse(txt); out = {t: [] for t in TYPES}
    for t in TYPES:
        seen = set()
        for e in (d.get(t) or []):
            if not isinstance(e, dict): continue
            k = esc_map(e.get("i"), e.get("q"), sents)
            if k is not None and k not in seen: seen.add(k); out[t].append(k)
    return out, resp.usage

def load_corpus():
    papers = []
    # SAL (has gold + sections)
    js = json.load(open(ROOT/"stage_c_package"/"sal_criteria"/"sal_test_allmodels.json", encoding="utf-8"))
    g = {}
    for i, r in enumerate(js): g.setdefault(r["pmcid"], []).append(i)
    for pid, idx in g.items():
        sents = [js[i]["sent"] for i in idx]; secs = [js[i]["sec"] for i in idx]
        gold = [k for k, i in enumerate(idx) if js[i]["label"] == 1]
        papers.append(dict(paper_id=pid, domain="biomed_sal", sents=sents, secs=secs, sal_gold=gold))
    # arXiv (full text -> split)
    for line in (ROOT/"data"/"paper_texts.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r = json.loads(line); sents = split_sentences(trim_refs(str(r["text"])))
        papers.append(dict(paper_id=str(r["id"]), domain="arxiv_cs", sents=sents, secs=None, sal_gold=None))
    return papers

def transcript_of(p):
    if p["secs"]:
        lines, last = [], None
        for k, (s, sec) in enumerate(zip(p["sents"], p["secs"])):
            if sec != last: lines.append(f"## {sec}"); last = sec
            lines.append(f"[{k}] {s}")
        return "\n".join(lines)
    return "\n".join(f"[{k}] {s}" for k, s in enumerate(p["sents"]))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); A = ap.parse_args()
    done = set()
    if OUT.exists():
        for ln in OUT.read_text(encoding="utf-8").splitlines():
            try: done.add(json.loads(ln)["paper_id"])
            except Exception: pass
    papers = [p for p in load_corpus() if p["paper_id"] not in done]
    if A.limit: papers = papers[:A.limit]
    print(f"to process: {len(papers)} papers (skipping {len(done)} done)", flush=True)
    total_cost = 0.0; t0 = time.time()
    with OUT.open("a", encoding="utf-8") as fh:
        for n, p in enumerate(papers, 1):
            ns = len(p["sents"]); tr = transcript_of(p)
            rec = dict(paper_id=p["paper_id"], domain=p["domain"], n_sent=ns, sal_gold=p["sal_gold"], votes={}, cost=0.0)
            ok = True
            for key, model, cin, cout in JUDGES:
                try:
                    out, u = judge(model, tr, ns, p["sents"])
                    c = (u.input_tokens/1e6)*cin + (u.output_tokens/1e6)*cout
                    rec["votes"][key] = out; rec["cost"] += c; total_cost += c
                except Exception as e:
                    rec["votes"][key] = {"error": str(e)[:200]}; ok = False
            # agreement: per type, indices both judges picked
            if ok:
                ag = {t: sorted(set(rec["votes"]["opus"][t]) & set(rec["votes"]["sonnet"][t])) for t in TYPES}
                rec["agree"] = ag
                rec["gap_any_agree"] = sorted(set().union(*[set(ag[t]) for t in TYPES]))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            print(f"[{n}/{len(papers)}] {p['paper_id']} ({p['domain']}, {ns} sents) "
                  f"{'OK' if ok else 'ERR'} cum ${total_cost:.2f} [{time.time()-t0:.0f}s]", flush=True)
    print(f"DONE. {len(papers)} papers, total ${total_cost:.2f}", flush=True)

if __name__ == "__main__":
    main()
