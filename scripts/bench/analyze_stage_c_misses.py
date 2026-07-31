"""Why is Stage C hurting recall on LimGen? Find the pattern.

Trains SciBERT on LimGen-train, predicts on test, runs the 3B V4 LLM filter on
SciBERT positives, then collects the FAILURE CASES:
  - SciBERT YES + gold YES + LLM NO  -> recall losses (the metric pain)
  - SciBERT YES + gold NO  + LLM NO  -> correct FP kills (the metric gain)
  - SciBERT YES + gold YES + LLM YES -> correct keeps (no change)

Samples ~50 of each, plus emits category-level counts via the V4 CoT category
string, so we can SEE which categories the LLM mislabels real limitations as.

    python -u scripts/bench/analyze_stage_c_misses.py --bert allenai/scibert_scivocab_uncased --llm Qwen/Qwen2.5-3B-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_limgen import build_xy, fetch  # noqa: E402
from finetune_and_chain import train_bert, predict_bert, best_thr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--cap-pos", type=int, default=8000)
    ap.add_argument("--train-papers", type=int, default=99999)
    ap.add_argument("--test-papers", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tr_s, tr_y = build_xy(fetch("train.jsonl"), args.train_papers, args.cap_pos, args.seed)
    te_s, te_y = build_xy(fetch("test.jsonl"), args.test_papers, None, args.seed + 1)
    rng = np.random.default_rng(args.seed); perm = rng.permutation(len(tr_s))
    nv = len(tr_s) // 5; vi, ti = perm[:nv], perm[nv:]
    sub_s = [tr_s[k] for k in ti]; ytr = tr_y[ti]
    val_s = [tr_s[k] for k in vi]; yval = tr_y[vi]
    print(f"train={len(sub_s)} val={len(val_s)} test={len(te_s)} "
          f"({int(te_y.sum())} lim / {int((1-te_y).sum())} other)", flush=True)

    # Train SciBERT
    model, tok, dev = train_bert(args.bert, sub_s, ytr, args.epochs, args.lr, args.bs, args.seed)
    pte = predict_bert(model, tok, te_s, dev)
    pval = predict_bert(model, tok, val_s, dev)
    thr = best_thr(pval, yval)
    bert_pred = (pte >= thr).astype(int)
    print(f"\nSciBERT (thr={thr:.3f}): "
          f"positives={int(bert_pred.sum())}  TP={int(((bert_pred==1)&(te_y==1)).sum())} "
          f"FP={int(((bert_pred==1)&(te_y==0)).sum())}", flush=True)
    del model; import torch, gc; gc.collect(); torch.cuda.empty_cache()

    # Run 3B V4 with FULL trace (category + answer)
    from gap2idea.pipeline.gap_llm_filter import (
        SYSTEM_VALIDATE_COT, SHOTS_VALIDATE_COT,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading {args.llm}...", flush=True)
    ltok = AutoTokenizer.from_pretrained(args.llm)
    lm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.float16, device_map="auto")
    lm.eval()

    def judge_with_trace(sent):
        msgs = [{"role": "system", "content": SYSTEM_VALIDATE_COT}]
        for s, a in SHOTS_VALIDATE_COT:
            msgs += [{"role": "user", "content": "Sentence: " + s},
                     {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": "Sentence: " + sent})
        text = ltok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = ltok(text, return_tensors="pt").to(lm.device)
        with torch.no_grad():
            out = lm.generate(**inp, max_new_tokens=40, do_sample=False, pad_token_id=ltok.eos_token_id)
        return ltok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

    # Collect cases
    recall_losses = []  # SciBERT YES + gold YES + LLM NO
    correct_kills = []  # SciBERT YES + gold NO + LLM NO
    correct_keeps_count = 0
    wrong_keeps = []    # SciBERT YES + gold NO + LLM YES (no help)
    n_judged = 0
    for i, s in enumerate(te_s):
        if bert_pred[i] == 0:
            continue
        n_judged += 1
        if n_judged % 200 == 0:
            print(f"  judged {n_judged}/{int(bert_pred.sum())}", flush=True)
        trace = judge_with_trace(s).strip()
        # extract category + answer
        cat = ans = ""
        for ln in trace.splitlines():
            up = ln.upper().strip()
            if up.startswith("CATEGORY:"): cat = ln.split(":", 1)[-1].strip()
            elif up.startswith("ANSWER:"): ans = up.split(":", 1)[-1].strip()
        kept = ans.startswith("YES")
        gold_pos = te_y[i] == 1
        rec = {"sentence": s, "category": cat, "answer": ans, "trace": trace}
        if gold_pos and not kept:
            recall_losses.append(rec)
        elif not gold_pos and not kept:
            correct_kills.append(rec)
        elif not gold_pos and kept:
            wrong_keeps.append(rec)
        else:
            correct_keeps_count += 1

    # Save full data
    out = ROOT / "data/scibert_prep"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(recall_losses, (out / "stage_c_recall_losses.json").open("w"), indent=2)
    json.dump(correct_kills, (out / "stage_c_correct_kills.json").open("w"), indent=2)
    json.dump(wrong_keeps, (out / "stage_c_wrong_keeps.json").open("w"), indent=2)

    # Print summary
    print(f"\n=== Stage C V4 trace ===", flush=True)
    print(f"  judged {n_judged} SciBERT positives", flush=True)
    print(f"  RECALL LOSSES (gold YES, LLM said NO): {len(recall_losses)}", flush=True)
    print(f"  CORRECT KILLS (gold NO, LLM said NO):  {len(correct_kills)}", flush=True)
    print(f"  CORRECT KEEPS (gold YES, LLM said YES):{correct_keeps_count}", flush=True)
    print(f"  WRONG KEEPS (gold NO, LLM said YES):   {len(wrong_keeps)}", flush=True)

    # Category histogram on recall losses
    from collections import Counter
    cats = Counter(r["category"][:60] for r in recall_losses)
    print(f"\n=== RECALL-LOSS category histogram (top 15) — what LLM mislabels real limitations as ===", flush=True)
    for c, n in cats.most_common(15):
        print(f"  {n:>4}  {c}", flush=True)

    print(f"\n=== 30 SAMPLE RECALL LOSSES (real limitations the LLM rejected) ===", flush=True)
    rng2 = np.random.default_rng(7)
    sampled = list(rng2.permutation(len(recall_losses))[:30])
    for k in sampled:
        r = recall_losses[k]
        s_short = r["sentence"][:120].replace("\n", " ")
        print(f"\n  [LLM-Cat: {r['category'][:50]}]", flush=True)
        print(f"  {s_short}", flush=True)

    print(f"\nsaved -> {out}/stage_c_{{recall_losses,correct_kills,wrong_keeps}}.json", flush=True)


if __name__ == "__main__":
    main()
