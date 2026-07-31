"""Fine-tune SciBERT 3-class on our self-distilled+ACL data, drop into the funnel,
run on our 10 gold papers, and compare to the current bge+logreg head.

INPUT (synced from local):
    data/scibert_prep/train.jsonl       (sentence, label) -- same as the bge head saw
    data/scibert_prep/gold_papers.jsonl (id, text, blocks)
    data/bench_gap/gold_sentences.tsv   (gold)

OUTPUT:
    data/scibert_prep/scibert_gold_gaps.tsv  (per-paper emitted gaps + gold-match flag)
    data/scibert_prep/scibert_gold_summary.md (recall/precision-floor vs gold + per-paper table)
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.pipeline.gap_funnel import (  # noqa: E402
    extract_gaps, slice_terminal_regions, token_containment,
)

LABELS = ["future_work", "limitation", "none"]
L2I = {l: i for i, l in enumerate(LABELS)}


class BertGapHead:
    """Duck-types EmbeddingGapHead.predict so it drops into extract_gaps()."""

    def __init__(self, model, tok, device):
        self.model, self.tok, self.device = model, tok, device

    def predict(self, sentences):
        import torch
        if not sentences:
            return []
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(sentences), 64):
                batch = sentences[i:i + 64]
                enc = self.tok(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=96).to(self.device)
                p = torch.softmax(self.model(**enc).logits, -1).cpu().numpy()
                for row in p:
                    j = int(row.argmax())
                    out.append((LABELS[j], float(row[j])))
        return out


def train_scibert(train_path, model_name, epochs, lr, bs, max_neg_ratio, seed):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    df = pd.read_json(train_path, lines=True)
    # balance negatives like the bge head did (max_neg_ratio * #positives)
    pos = df[df.label != "none"]
    neg = df[df.label == "none"]
    cap = int(len(pos) * max_neg_ratio)
    if len(neg) > cap:
        neg = neg.iloc[rng.permutation(len(neg))[:cap]]
    train = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    y = train.label.map(L2I).values
    print(f"train rows={len(train)}  {train.label.value_counts().to_dict()}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(dev)
    counts = np.bincount(y, minlength=3)
    w = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32).to(dev)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sents = train.sentence.tolist()
    print(f"training {model_name} x {epochs} ep on {dev}", flush=True)
    for ep in range(epochs):
        model.train(); order = rng.permutation(len(sents)); tot, nb = 0.0, 0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            enc = tok([sents[k] for k in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=96).to(dev)
            loss = lossf(model(**enc).logits,
                         torch.tensor(y[idx], dtype=torch.long).to(dev))
            loss.backward(); opt.step(); opt.zero_grad()
            tot += float(loss.detach()); nb += 1
        print(f"  epoch {ep+1}/{epochs} loss {tot/max(1,nb):.4f}", flush=True)
    return BertGapHead(model, tok, dev)


def short(s, n=85):
    return textwrap.shorten(" ".join(str(s).split()), n)


def gold_match(sent, gold_rows):
    for _, g in gold_rows.iterrows():
        if max(token_containment(g["gap_sentence"], sent),
               token_containment(sent, g["gap_sentence"])) >= 0.8:
            return g["gap_id"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--max-neg-ratio", type=float, default=4.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--llm", default=None, help="local LLM for Stage C, e.g. Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--llm-mode", default="validate", choices=["validate", "validate_cot", "validate_v5", "validate_v6", "validate_v7", "junk"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prep = ROOT / "data" / "scibert_prep"
    head = train_scibert(prep / "train.jsonl", args.bert, args.epochs, args.lr,
                         args.bs, args.max_neg_ratio, args.seed)

    gold = pd.read_csv(ROOT / "data/bench_gap/gold_sentences.tsv", sep="\t", dtype={"paper_id": str})
    papers = [json.loads(line) for line in (prep / "gold_papers.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"\ngold: {len(gold)} gaps over {gold['paper_id'].nunique()} papers; "
          f"loaded {len(papers)} paper texts", flush=True)

    # Optional Stage C judge — both SciBERT and a 3B LLM fit on a 32GB V100.
    filt = None
    if args.llm:
        from gap2idea.pipeline.gap_llm_filter import LLMGapFilter
        filt = LLMGapFilter(backend="local", model=args.llm, mode=args.llm_mode)

    rows = []
    tot_full = tot_slice = tot_gaps = tot_after_c = 0
    matched_gold = set()
    matched_after_c = set()
    R = [f"# SciBERT-FT on our 10 gold papers (3-class drop-in)", "",
         f"Model: {args.bert} fine-tuned {args.epochs}ep, lr={args.lr}, bs={args.bs}, threshold={args.threshold}.",
         f"Training data: same self-distilled+ACL set as the bge+logreg head ({prep}/train.jsonl).", ""]

    for rec in papers:
        pid = str(rec["id"])
        from gap2idea.pipeline.gap_prefilter import split_sentences
        from gap2idea.pipeline.sections import _cut_before_references
        full = split_sentences(_cut_before_references(rec["text"]))
        regions = slice_terminal_regions(rec["text"], blocks=rec.get("blocks"))
        slice_sents = [s for r in regions for s in r.sentences]
        gaps = extract_gaps(pid, rec["text"], blocks=rec.get("blocks"),
                            head=head, mode="hybrid", model_threshold=args.threshold)
        # Stage C — rule-protected LLM filter (only judges model-only gaps)
        kept = gaps
        if filt is not None:
            kept = filt.filter_gaps(gaps, protect_rules=True)
        kept_keys = {short(k["gap_sentence"], 60) for k in kept}

        tot_full += len(full); tot_slice += len(slice_sents); tot_gaps += len(gaps); tot_after_c += len(kept)
        gp = gold[gold["paper_id"] == pid]
        dropA = 100 * (1 - len(slice_sents) / max(1, len(full)))
        dropB = 100 * (1 - len(gaps) / max(1, len(slice_sents)))
        dropC = 100 * (1 - len(kept) / max(1, len(gaps))) if filt is not None else 0
        R += [f"## {pid}  ·  gold gaps: {len(gp)}", "",
              f"- Stage A: {len(full)} → {len(slice_sents)} (−{dropA:.0f}%)",
              f"- Stage B (SciBERT): {len(slice_sents)} → {len(gaps)} (−{dropB:.0f}% of slice)"]
        if filt is not None:
            R.append(f"- Stage C ({args.llm}): {len(gaps)} → {len(kept)} (−{dropC:.0f}%)")
        R += ["", "| type | source | section | C-kept | gold? | gap sentence |",
              "|---|---|---|---|---|---|"]
        for g in gaps:
            m = gold_match(g["gap_sentence"], gp)
            if m: matched_gold.add(m)
            kep = short(g["gap_sentence"], 60) in kept_keys
            if kep and m:
                matched_after_c.add(m)
            rows.append({"paper_id": pid, "gap_type": g["gap_type"], "source": g["source"],
                         "section_type": g["section_type"],
                         "kept_after_c": int(kep),
                         "gold_match": m or "",
                         "gap_sentence": g["gap_sentence"]})
            cflag = "✓" if kep else "✗"
            R.append(f"| {g['gap_type']} | {g['source']} | {g['section_type']} | {cflag} | "
                     f"{'✅ '+m if m else '— extra'} | {short(g['gap_sentence'], 95)} |")
        R += [""]

    recall = len(matched_gold) / max(1, len(gold))
    prec_floor = sum(1 for r in rows if r["gold_match"]) / max(1, len(rows))
    summary = [
        "| stage | total | per paper | drop |", "|---|---|---|---|",
        f"| full body | {tot_full} | {tot_full/len(papers):.0f} | — |",
        f"| → Stage A slice | {tot_slice} | {tot_slice/len(papers):.0f} | −{100*(1-tot_slice/tot_full):.0f}% (free) |",
        f"| → SciBERT gaps (Stage B) | {tot_gaps} | {tot_gaps/len(papers):.1f} | −{100*(1-tot_gaps/tot_slice):.0f}% of slice |",
    ]
    if filt is not None:
        recall_c = len(matched_after_c) / max(1, len(gold))
        kept_rows = [r for r in rows if r["kept_after_c"]]
        prec_c = sum(1 for r in kept_rows if r["gold_match"]) / max(1, len(kept_rows))
        summary += [
            f"| → +Stage C ({args.llm.split('/')[-1]}) | {tot_after_c} | {tot_after_c/len(papers):.1f} | "
            f"−{100*(1-tot_after_c/tot_gaps):.0f}% of Stage B |",
            "",
            f"**Vs gold (Stage B only):** recall = {len(matched_gold)}/{len(gold)} = **{recall:.3f}**; "
            f"precision_floor = **{prec_floor:.3f}**.",
            f"**Vs gold (after Stage C):** recall = {len(matched_after_c)}/{len(gold)} = **{recall_c:.3f}**; "
            f"precision_floor = **{prec_c:.3f}**.",
        ]
    else:
        summary += ["",
            f"**Vs gold:** recall = {len(matched_gold)}/{len(gold)} = **{recall:.3f}**; "
            f"precision_floor = {sum(1 for r in rows if r['gold_match'])}/{len(rows)} = "
            f"**{prec_floor:.3f}** (true precision higher — gold is partial).",
        ]
    summary += ["",
        "## bge+logreg baseline (no SciBERT) — from earlier runs",
        "Stage A+B (bge+logreg, hybrid): 62 preds, recall 0.526, precision_floor 0.194 (F1 ~0.283).",
        "",
    ]
    R = R[:4] + summary + R[4:]

    out_md = prep / "scibert_gold_summary.md"
    out_md.write_text("\n".join(R), encoding="utf-8")
    pd.DataFrame(rows).to_csv(prep / "scibert_gold_gaps.tsv", sep="\t", index=False)
    print(f"\n=== SciBERT on 10 gold papers ===")
    print(f"  full {tot_full} -> slice {tot_slice} ({100*(1-tot_slice/tot_full):.0f}% dropped, free)")
    print(f"  -> Stage B gaps {tot_gaps} ({tot_gaps/len(papers):.1f}/paper, {100*(1-tot_gaps/tot_slice):.0f}% of slice dropped)")
    print(f"     recall={recall:.3f}  precision_floor={prec_floor:.3f}  F1_floor={(2*recall*prec_floor)/max(1e-9,(recall+prec_floor)):.3f}")
    if filt is not None:
        recall_c = len(matched_after_c) / max(1, len(gold))
        kept_rows = [r for r in rows if r["kept_after_c"]]
        prec_c = sum(1 for r in kept_rows if r["gold_match"]) / max(1, len(kept_rows))
        print(f"  -> +Stage C kept {tot_after_c} ({tot_after_c/len(papers):.1f}/paper, "
              f"{100*(1-tot_after_c/tot_gaps):.0f}% of Stage B dropped)")
        print(f"     recall={recall_c:.3f}  precision_floor={prec_c:.3f}  "
              f"F1_floor={(2*recall_c*prec_c)/max(1e-9,(recall_c+prec_c)):.3f}")
    print(f"  saved: {prep}/scibert_gold_{{gaps.tsv,summary.md}}")


if __name__ == "__main__":
    main()
