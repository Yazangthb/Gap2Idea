"""Audit every gold gap and every prediction across the 3 stages.

Identifies:
  - For each gold gap: was it caught by Stage A? B? C? where did it die?
  - For each Stage-C-kept "extra": is it actually a real gap gpt-4o missed?
  - Data quality: are the 19 gold gaps all real semantic limitations?

Calls gpt-4o for the semantic adjudication of borderline cases.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.pipeline.gap_funnel import slice_terminal_regions, token_containment  # noqa: E402
from gap2idea.pipeline.llm import get_llm_client  # noqa: E402


def gold_match(sent, gold_rows, tau=0.8):
    for _, g in gold_rows.iterrows():
        if max(token_containment(g["gap_sentence"], sent),
               token_containment(sent, g["gap_sentence"])) >= tau:
            return g["gap_id"]
    return None


def short(s, n=85):
    return textwrap.shorten(" ".join(str(s).split()), n)


def adjudicate_batch(client, items, model="openai/gpt-4o"):
    """For each (sentence, context_before, context_after, paper_id), ask:
    is this a real research gap? Returns list of dicts {verdict, reason}."""
    SYS = (
        "You judge whether a sentence from a scientific paper is a real research "
        "GAP — meaning the authors mention something NOT YET DONE in their own work "
        "(limitation, scope restriction, assumption, future-work direction, open "
        "problem). Use the surrounding context.\n\n"
        "Reply for each numbered input with ONE line: '<idx>. GAP | REASON: <reason>' "
        "or '<idx>. NOT_GAP | REASON: <reason>'. Keep reasons under 12 words."
    )
    user = "\n".join(
        f"{i+1}. ...{(it.get('before') or '')[-80:]} >>> {it['sentence'][:120]} <<< "
        f"{(it.get('after') or '')[:80]}..."
        for i, it in enumerate(items)
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYS},
                  {"role": "user", "content": user}],
        temperature=0.0, max_tokens=60 * len(items))
    out = [{"verdict": "?", "reason": ""} for _ in items]
    for ln in resp.choices[0].message.content.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        for tag in ("GAP", "NOT_GAP"):
            if f". {tag}" in ln.upper() or f".{tag}" in ln.upper():
                try:
                    idx = int(ln.split(".", 1)[0]) - 1
                    reason = ln.split("REASON:", 1)[-1].strip() if "REASON" in ln.upper() else ""
                    if 0 <= idx < len(items):
                        out[idx] = {"verdict": tag, "reason": reason}
                except ValueError:
                    pass
                break
    return out


def find_context(sentence, full_text, window=30):
    short_s = sentence.strip()[:80]
    idx = full_text.find(short_s)
    if idx < 0:
        return "", ""
    end = idx + len(sentence)
    before = " ".join(full_text[max(0, idx-400):idx].split()[-window:])
    after = " ".join(full_text[end:end+400].split()[:window])
    return before, after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    gold = pd.read_csv(ROOT / "data/bench_gap/gold_sentences.tsv", sep="\t", dtype={"paper_id": str})
    papers = {}
    for line in (ROOT / "data/scibert_prep/gold_papers.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            papers[str(rec["id"])] = rec

    preds = pd.read_csv(ROOT / "data/scibert_prep/scibert_gold_gaps.tsv", sep="\t", dtype=str).fillna("")
    print(f"Gold: {len(gold)} gaps   Papers: {len(papers)}   Stage B preds: {len(preds)}\n", flush=True)

    # ========== Per-gold-gap audit ==========
    print("=" * 80)
    print("PART 1: WHERE EACH GOLD GAP DIES (or survives)")
    print("=" * 80)
    audit = []
    for _, g in gold.iterrows():
        pid = g["paper_id"]
        rec = papers.get(pid)
        if not rec:
            continue
        regions = slice_terminal_regions(rec["text"], blocks=rec.get("blocks"))
        slice_text = " ".join(s for r in regions for s in r.sentences)
        in_slice_90 = token_containment(g["gap_sentence"], slice_text) >= 0.9
        in_slice_80 = token_containment(g["gap_sentence"], slice_text) >= 0.8
        sub = preds[preds["paper_id"] == pid]
        stage_b_hit = any(gold_match(p, gold[gold["paper_id"] == pid]) == g["gap_id"]
                          for p in sub["gap_sentence"])
        # find which prediction matched (if any)
        matched_pred = ""
        for p in sub["gap_sentence"]:
            if gold_match(p, pd.DataFrame([g])) == g["gap_id"]:
                matched_pred = p
                break
        audit.append({
            "gap_id": g["gap_id"], "type": g["gap_type"],
            "in_slice@0.8": "Y" if in_slice_80 else "N",
            "in_slice@0.9": "Y" if in_slice_90 else "N",
            "stage_b": "Y" if stage_b_hit else "N",
            "matched": short(matched_pred, 70) if matched_pred else "",
            "gold": short(g["gap_sentence"], 70),
        })
    adf = pd.DataFrame(audit)
    print(adf.to_string(index=False))

    # ========== Where each gold dies ==========
    print()
    n_in_slice_80 = (adf["in_slice@0.8"] == "Y").sum()
    n_hit_b = (adf["stage_b"] == "Y").sum()
    n_lost_A = len(adf) - n_in_slice_80
    n_lost_B = n_in_slice_80 - n_hit_b
    print(f"Stage A loses: {n_lost_A}/{len(adf)} gold (not in slice @ τ=0.8)")
    print(f"Stage B loses: {n_lost_B}/{n_in_slice_80} of slice-localized gold ({100*n_lost_B/max(1,n_in_slice_80):.0f}%)")

    # ========== Part 2: Are the "extras" real? ==========
    print("\n" + "=" * 80)
    print("PART 2: ARE STAGE-B 'EXTRAS' ACTUALLY REAL GAPS (gold-missed)?")
    print("=" * 80)
    extras = []
    for _, p in preds.iterrows():
        gp = gold[gold["paper_id"] == p["paper_id"]]
        m = gold_match(p["gap_sentence"], gp)
        if m is None:
            rec = papers.get(p["paper_id"], {})
            before, after = find_context(p["gap_sentence"], rec.get("text", "")) if rec else ("", "")
            extras.append({"paper_id": p["paper_id"], "sentence": p["gap_sentence"],
                            "before": before, "after": after})
    print(f"Stage B emitted {len(extras)} 'extras' (not gold-matched)")

    if not args.no_llm and extras:
        client = get_llm_client()
        # batch them
        verdicts = []
        for i in range(0, len(extras), 10):
            verdicts.extend(adjudicate_batch(client, extras[i:i+10], args.model))
        real_gaps_missed_by_gold = 0
        true_junk = 0
        for e, v in zip(extras, verdicts):
            e.update(v)
            if v["verdict"] == "GAP":
                real_gaps_missed_by_gold += 1
            elif v["verdict"] == "NOT_GAP":
                true_junk += 1
        print(f"  -> gpt-4o calls {real_gaps_missed_by_gold} real gaps gold missed")
        print(f"  -> gpt-4o calls {true_junk} true junk")
        # show some
        print("\nReal gaps gold missed (sample):")
        for e in [x for x in extras if x.get("verdict") == "GAP"][:6]:
            print(f"  - {short(e['sentence'], 100)}  | {e['reason']}")
        print("\nTrue junk Stage B emitted (sample):")
        for e in [x for x in extras if x.get("verdict") == "NOT_GAP"][:6]:
            print(f"  - {short(e['sentence'], 100)}  | {e['reason']}")

    # ========== Part 3: Gold quality check ==========
    print("\n" + "=" * 80)
    print("PART 3: ARE THE 19 GOLD GAPS ALL REAL?")
    print("=" * 80)
    if not args.no_llm:
        client = get_llm_client()
        items = []
        for _, g in gold.iterrows():
            rec = papers.get(g["paper_id"], {})
            before, after = find_context(g["gap_sentence"], rec.get("text", "")) if rec else ("", "")
            items.append({"sentence": g["gap_sentence"], "before": before, "after": after,
                          "gap_id": g["gap_id"]})
        verdicts = adjudicate_batch(client, items, args.model)
        for it, v in zip(items, verdicts):
            it.update(v)
        ok = sum(1 for x in items if x["verdict"] == "GAP")
        bad = sum(1 for x in items if x["verdict"] == "NOT_GAP")
        print(f"  gpt-4o calls {ok}/{len(gold)} gold gaps as real GAP")
        print(f"  gpt-4o calls {bad}/{len(gold)} gold gaps as NOT_GAP (contamination)")
        if bad > 0:
            print("\n  Contaminated golds (gpt-4o says NOT_GAP):")
            for x in items:
                if x["verdict"] == "NOT_GAP":
                    print(f"    {x['gap_id']}: {short(x['sentence'], 90)}  | {x['reason']}")

    # ========== Recompute corrected metrics ==========
    print("\n" + "=" * 80)
    print("PART 4: CORRECTED METRICS (gold-quality + extras-adjudicated)")
    print("=" * 80)
    if not args.no_llm:
        true_gold = ok  # = clean gold count
        true_extras_real = real_gaps_missed_by_gold  # extras gpt-4o says are gaps
        true_tp = n_hit_b + true_extras_real  # true positives (gold-hit + missed real)
        true_total_real = true_gold + true_extras_real  # all real gaps in 10 papers
        true_recall = true_tp / max(1, true_total_real)
        true_precision = true_tp / max(1, len(preds))
        true_f1 = 2 * true_precision * true_recall / max(1e-9, true_precision + true_recall)
        print(f"  Cleaned gold (after removing contamination): {true_gold}/{len(gold)}")
        print(f"  Extras gpt-4o calls real:                    {true_extras_real}")
        print(f"  Total real gaps in 10 papers (estimated):    {true_total_real}")
        print(f"  Stage B true positives:                      {true_tp}")
        print()
        print(f"  TRUE recall (adjusted): {true_recall:.3f}")
        print(f"  TRUE precision (adjusted): {true_precision:.3f}")
        print(f"  TRUE F1 (adjusted): {true_f1:.3f}")


if __name__ == "__main__":
    main()
