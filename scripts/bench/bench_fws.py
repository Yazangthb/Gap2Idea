"""FWS (future-work-sentence) recognition bench — the other half of gap extraction.

Zhang et al. (2022), "Automatic Recognition and Classification of Future Work Sentences"
(J. Informetrics; data: github.com/xiangyi-njust/FWS). Binary recognition over 64,896
sentences from 9,013 ACL papers (13.9% positive, real negatives, section labels shipped).
Their best recognition model (Bernoulli Naive Bayes) reports Macro-F1 = 0.9073.

We fine-tune several encoders (incl. PubMedBERT, for a cross-domain contrast with the RCT
bench, vs in-domain SciBERT and a general bge-small), then add the batched Stage C filter
(mode="validate_fws"). We report BOTH:
  - positive-class P/R/F1 (the honest, RCT-comparable number)
  - Macro-F1 (comparable to their reported 0.9073; inflated by the easy 86% negative class)

Split is by paper id (no sentence leakage across train/dev/test) -> not their exact split,
so this is a faithful re-creation, not a literal reproduction.

    python scripts/bench/bench_fws.py --models microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext,allenai/scibert_scivocab_uncased,BAAI/bge-small-en-v1.5
"""
from __future__ import annotations
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse, csv, hashlib, io, sys, time, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts" / "training"))
from bench_limgen import prf  # noqa: E402
from gap2idea.pipeline.gap_funnel import cue_label, EmbeddingGapHead  # noqa: E402
from gap2idea.pipeline.gap_llm_filter import LLMGapFilter  # noqa: E402
from gap2idea.pipeline.llm import active_provider  # noqa: E402

RAW = "https://raw.githubusercontent.com/xiangyi-njust/FWS/main/Dataset/Corpus_For_FWS_Recognition.csv"
_CACHE = ROOT / "data" / "fws_recognition.csv"


def _bucket(pid):
    return int(hashlib.md5(str(pid).encode()).hexdigest(), 16) % 10


def load(split, cap=None):
    """Deterministic 80/10/10 split by paper id (buckets 0-7 train, 8 dev, 9 test)."""
    if _CACHE.exists():
        raw = _CACHE.read_text(encoding="utf-8", errors="replace")
    else:
        raw = urllib.request.urlopen(RAW, timeout=180).read().decode("utf-8", "replace")
        _CACHE.parent.mkdir(parents=True, exist_ok=True); _CACHE.write_text(raw, encoding="utf-8")
    want = {"train": set(range(8)), "dev": {8}, "test": {9}}[split]
    s, y = [], []
    for r in csv.DictReader(io.StringIO(raw)):
        txt = (r.get("text") or "").strip()
        if len(txt.split()) < 3 or _bucket(r.get("id", "")) not in want:
            continue
        s.append(txt); y.append(1 if str(r.get("label", "0")).strip() == "1" else 0)
    if cap:
        s, y = s[:cap], np.array(y[:cap])
    return s, np.array(y)


def macro_f1(y, pred):
    f_pos = prf(y, pred)[2]
    f_neg = prf(1 - y, 1 - pred)[2]
    return round((f_pos + f_neg) / 2, 4)


def stage_c(sents, sb):
    filt = LLMGapFilter(backend="api", mode="validate_fws")
    pos = [i for i, v in enumerate(sb) if v == 1]
    keep = []
    for k in range(0, len(pos), 40):
        keep.extend(filt.judge_batch([sents[i] for i in pos[k:k + 40]]))
    final = sb.copy()
    for i, kp in zip(pos, keep):
        if not kp:
            final[i] = 0
    return final, filt.n_calls, len(pos), sum(1 for x in keep if not x)


def zeroshot(args):
    """Our CURRENT shipped solution, no training: cue rules + the arXiv-trained gap
    head predict future-work sentences; Stage C (validate_fws) filters. Runs on CPU."""
    te_s, te_y = load("test", args.test_cap)
    print(f"provider={active_provider()}  test {len(te_s)} ({int(te_y.sum())} fw, {te_y.mean():.1%})")
    print(f"reference: Zhang 2022 recognition Macro-F1 0.9073 (Bernoulli NB)\n")
    print(f"{'solution':<26} {'stage':<9}  P     R     posF1  macroF1")
    head = EmbeddingGapHead.load(ROOT / "data/gap_head.joblib")
    pr = head.predict(te_s)
    zs = np.array([1 if (cue_label(s) == "future_work" or (l == "future_work" and p >= 0.5)) else 0
                   for s, (l, p) in zip(te_s, pr)])
    P, R, F = prf(te_y, zs); M = macro_f1(te_y, zs)
    print(f"{'shipped (cue+head)':<26} {'detector':<9}  {P:.3f} {R:.3f} {F:.3f}  {M:.3f}"
          f"{' >=0.907' if M >= 0.9073 else ''}")
    if args.stage_c:
        fc, nc, nj, nd = stage_c(te_s, zs)
        Pc, Rc, Fc = prf(te_y, fc); Mc = macro_f1(te_y, fc)
        print(f"{'shipped (cue+head)':<26} {'+stageC':<9}  {Pc:.3f} {Rc:.3f} {Fc:.3f}  {Mc:.3f}"
              f"{' >=0.907' if Mc >= 0.9073 else ''}  (judged {nj} in {nc} calls, dropped {nd})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="fine-tune encoders (default: zero-shot shipped solution)")
    ap.add_argument("--models", default="allenai/scibert_scivocab_uncased,"
                    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext,"
                    "BAAI/bge-small-en-v1.5")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--neg-ratio", type=int, default=3)
    ap.add_argument("--test-cap", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage-c", action="store_true", default=True)
    ap.add_argument("--no-stage-c", dest="stage_c", action="store_false")
    args = ap.parse_args()
    if not args.train:
        return zeroshot(args)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from finetune_rct import train, proba
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(1)

    tr_s, tr_y = load("train"); dv_s, dv_y = load("dev"); te_s, te_y = load("test", args.test_cap)
    pos = np.where(tr_y == 1)[0]; neg = np.where(tr_y == 0)[0]
    rng = np.random.default_rng(args.seed)
    keep = np.concatenate([pos, rng.choice(neg, size=min(len(neg), len(pos) * args.neg_ratio), replace=False)])
    rng.shuffle(keep)
    trs = [tr_s[i] for i in keep]; trly = tr_y[keep]
    print(f"provider={active_provider()} device={device}")
    print(f"FWS: train {len(trs)} ({int(trly.sum())} fw, 1:{args.neg_ratio}) / dev {len(dv_s)} / "
          f"test {len(te_s)} ({int(te_y.sum())} fw, {te_y.mean():.1%})")
    print(f"reference: Zhang 2022 recognition Macro-F1 0.9073 (Bernoulli NB)\n")
    print(f"{'model':<22} {'stage':<9}  P     R     posF1  macroF1")

    for model_name in args.models.split(","):
        model_name = model_name.strip(); short = model_name.split("/")[-1][:22]
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
        t0 = time.time()
        train(model, tok, trs, trly, args.epochs, args.seed, device)
        pd = proba(model, tok, dv_s, device)
        best_t, best_f = 0.5, -1
        for t in np.linspace(0.15, 0.9, 76):
            f = prf(dv_y, (pd >= t).astype(int))[2]
            if f > best_f: best_f, best_t = f, t
        pt = proba(model, tok, te_s, device)
        sb = (pt >= best_t).astype(int)
        P, R, F = prf(te_y, sb); M = macro_f1(te_y, sb)
        beat = " >=0.907" if M >= 0.9073 else ""
        print(f"{short:<22} {'detector':<9}  {P:.3f} {R:.3f} {F:.3f}  {M:.3f}{beat}  ({time.time()-t0:.0f}s, thr={best_t:.2f})", flush=True)
        if args.stage_c:
            fc, nc, nj, nd = stage_c(te_s, sb)
            Pc, Rc, Fc = prf(te_y, fc); Mc = macro_f1(te_y, fc)
            beatc = " >=0.907" if Mc >= 0.9073 else ""
            print(f"{short:<22} {'+stageC':<9}  {Pc:.3f} {Rc:.3f} {Fc:.3f}  {Mc:.3f}{beatc}  (judged {nj} in {nc} calls, dropped {nd})", flush=True)


if __name__ == "__main__":
    main()
