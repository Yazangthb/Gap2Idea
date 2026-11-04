"""Fresh gold extraction (limitation + future_work ONLY) over the 10 gold papers, full text.
Agent reads each full paper and returns verbatim limitation/future_work gap sentences,
explicitly rejecting scope / assumption / open-problem. Each sentence is verified to be
present in the source (token-containment) to drop hallucinations.

Output: data/bench_gap/gold_lf.tsv  and  data/external/gold_eval_lf.json
"""
import json, re, time, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
MODEL = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-4o"

env = dict(re.findall(r'^([A-Z_]+)=(.*)$', (ROOT/".env").read_text(), re.M))
KEY = env.get("OPENROUTER_API_KEY","").strip().strip('"')
from openai import OpenAI
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=KEY)

_W=re.compile(r"[a-z0-9]+"); _DEH=re.compile(r"(\w)-\s*(\w)")
def norm(s): return " ".join(re.sub(r"[^a-z0-9' -]"," ",(s or "").lower()).split())
def tc(n,h):
    nt=_W.findall(norm(_DEH.sub(r"\1\2",n)))
    if not nt: return 0.0
    hs=set(_W.findall(norm(_DEH.sub(r"\1\2",h)))); return sum(1 for t in nt if t in hs)/len(nt)

SYS = (
 "You extract research-gap sentences from a scientific paper. Return ONLY sentences that are, "
 "verbatim, the authors' own statement of one of these two types:\n"
 "(1) limitation — an explicit limitation, shortcoming, weakness, failure case, or restriction of "
 "THIS paper's own method/results (e.g. 'our method does not handle X', 'we did not evaluate Y', "
 "'performance degrades on Z', 'a limitation of our work is ...').\n"
 "(2) future_work — an explicit statement of future work or next steps the authors intend or "
 "recommend (e.g. 'future work will ...', 'we plan to ...', 'a promising direction is ...').\n\n"
 "Do NOT return:\n"
 "- sentences that merely describe the paper's SCOPE or focus ('this work focuses on ...') unless "
 "explicitly framed as a limitation;\n"
 "- ASSUMPTIONS stated neutrally ('we assume ...') unless explicitly framed as a limiting assumption;\n"
 "- OPEN PROBLEMS of the field in general;\n"
 "- limitations of PRIOR or related work.\n\n"
 "Copy each sentence EXACTLY as it appears in the provided text (a verbatim substring; do not fix "
 "typos or reorder words). Output strict JSON: "
 '{"gaps":[{"gap_type":"limitation"|"future_work","gap_sentence":"..."}]}'
)

def extract(pid, text):
    r = client.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=1500,
        response_format={"type":"json_object"},
        messages=[{"role":"system","content":SYS},
                  {"role":"user","content":"PAPER TEXT:\n\n"+text[:40000]}])
    try:
        gaps = json.loads(r.choices[0].message.content).get("gaps",[])
    except Exception:
        gaps = []
    out=[]
    for g in gaps:
        t=str(g.get("gap_type","")).strip(); s=str(g.get("gap_sentence","")).strip()
        if t in ("limitation","future_work") and len(s.split())>=5 and tc(s,text)>=0.85:
            out.append({"paper_id":pid,"gap_type":t,"gap_sentence":s})
    return out

def main():
    papers={}
    for l in (ROOT/"data/scibert_prep/gold_papers.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip(): r=json.loads(l); papers[str(r["id"])]=r["text"]
    rows=[]
    for pid,text in papers.items():
        t0=time.time(); g=extract(pid,text)
        # dedup within paper by normalized sentence
        seen=set(); uniq=[]
        for x in g:
            k=norm(x["gap_sentence"])[:80]
            if k and k not in seen: seen.add(k); uniq.append(x)
        rows+=uniq
        print(f"{pid}: {len(uniq)} gaps ({sum(x['gap_type']=='limitation' for x in uniq)} lim / "
              f"{sum(x['gap_type']=='future_work' for x in uniq)} fw) [{time.time()-t0:.1f}s]")
    import pandas as pd
    df=pd.DataFrame(rows)
    out=ROOT/"data/bench_gap/gold_lf.tsv"; df.to_csv(out,sep="\t",index=False)
    print(f"\nTOTAL {len(df)} gaps / {df.paper_id.nunique()} papers -> {out}")
    print("types:",df.gap_type.value_counts().to_dict())
    # build gold_eval_lf.json (reuse existing Stage-A candidates)
    ge=json.load(open(ROOT/"data/external/gold_eval.json",encoding="utf-8"))
    ge2={"candidates":ge["candidates"],"gold":rows}
    json.dump(ge2,open(ROOT/"data/external/gold_eval_lf.json","w"),ensure_ascii=False)
    print("wrote data/external/gold_eval_lf.json")

if __name__=="__main__":
    main()
