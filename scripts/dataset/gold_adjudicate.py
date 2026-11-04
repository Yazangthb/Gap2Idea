# -*- coding: utf-8 -*-
"""Recall-boost via adjudication. Candidate set = UNION of both judges (recall 0.956 on SAL vs
0.868 for the intersection). The both-agree-same-type sentences are already high-confidence gold;
this pass adjudicates the ~1,732 ONE-SIDED / type-mismatch candidates with a 3rd strong judge
(claude-opus-5-5, high effort, +/-3 context) -> confirmed ones join the gold. Resumable."""
import os, sys, json, re, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "dataset"))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
import anthropic
from gold_label_full import load_corpus

GOLD = ROOT / "data" / "gold"
OUT = GOLD / "adjudication.jsonl"
TYPES = ("limitation", "future_work", "research_gap")
PRI = {"limitation": 0, "research_gap": 1, "future_work": 2}
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

GUIDE = (
 "You are the ADJUDICATOR for research-gap labeling. For each CANDIDATE sentence (shown with +/-3 "
 "sentences of context and the «marked» target), decide its correct label:\n"
 "- limitation: a weakness/caveat/bias/constraint the authors acknowledge about THEIR OWN work.\n"
 "- future_work: an explicit direction the authors propose for FUTURE research.\n"
 "- research_gap: an unsolved problem / missing capability / open challenge in the field or prior work.\n"
 "- none: anything else (background, method, results, positive claim, recommendation, header).\n"
 "Judge accurately and inclusively: if the TARGET genuinely states a gap of one of the three kinds, "
 "label it; only use none when it clearly is not a gap. Return ONLY JSON mapping each candidate index "
 'to its label, e.g. {"142":"limitation","207":"none"}.'
)

def typemap(side):
    m = {}
    for t in TYPES:
        for i in side.get(t, []):
            if i not in m or PRI[t] < PRI[m[i]]: m[i] = t
    return m

def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    try: return json.loads(m.group(0)) if m else {}
    except Exception: return {}

def ctx(sents, i, k=3):
    return " ".join((f"«{sents[j]}»" if j == i else sents[j])
                    for j in range(i-k, i+k+1) if 0 <= j < len(sents))

def main():
    rows = [json.loads(l) for l in (GOLD/"gap_gold.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    corpus = {p["paper_id"]: p for p in load_corpus()}
    done = set()
    if OUT.exists():
        for l in OUT.read_text(encoding="utf-8").splitlines():
            try: done.add(json.loads(l)["paper_id"])
            except Exception: pass
    total = 0.0; t0 = time.time(); n = 0
    with OUT.open("a", encoding="utf-8") as fh:
        for r in rows:
            pid = r["paper_id"]
            if pid in done: continue
            O = typemap(r["votes"]["opus"]); S = typemap(r["votes"]["sonnet"])
            both_same = {i for i in O if O.get(i) == S.get(i)}
            cand = sorted((set(O) | set(S)) - both_same)     # one-sided + type-mismatch
            n += 1
            if not cand:
                fh.write(json.dumps({"paper_id": pid, "verdict": {}}) + "\n"); fh.flush(); continue
            sents = corpus[pid]["sents"]
            items = "\n".join(f"[{i}] proposed={O.get(i) or S.get(i)} :: {ctx(sents, i)}" for i in cand)
            try:
                resp = client.messages.create(model="claude-opus-5-5", max_tokens=4000,
                    extra_body={"output_config": {"effort": "high"}}, system=GUIDE,
                    messages=[{"role": "user", "content": f"Adjudicate these {len(cand)} candidates:\n\n{items}"}])
                txt = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                d = parse(txt); u = resp.usage
                c = (u.input_tokens/1e6)*4 + (u.output_tokens/1e6)*20; total += c
                verdict = {}
                for i in cand:
                    lab = str(d.get(str(i), "none")).strip().lower()
                    verdict[i] = lab if lab in TYPES else "none"
                fh.write(json.dumps({"paper_id": pid, "verdict": {str(k): v for k, v in verdict.items()}}) + "\n")
            except Exception as e:
                fh.write(json.dumps({"paper_id": pid, "error": str(e)[:200]}) + "\n")
            fh.flush()
            if n % 10 == 0:
                print(f"[{n}] {pid}: {len(cand)} cand, cum ${total:.2f} [{time.time()-t0:.0f}s]", flush=True)
    print(f"DONE adjudication. cum ${total:.2f}", flush=True)

if __name__ == "__main__":
    main()
