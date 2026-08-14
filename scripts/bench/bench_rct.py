"""RCT/SAL limitation-detection bench (Kilicoglu et al.; human-annotated, has TRUE
negatives -> Stage C can be measured fairly, unlike LimGen/BAGELS).

Binary: a sentence is a limitation iff it carries a category span (category_spans
!= '[0]'). We report:
  - zero-shot shipped head (cue + bge-small head, arXiv-trained)  -> cross-domain
  - in-domain bge+logreg (trained on RCT-train)                   -> fair vs PubMedBERT
  - each + batched Stage C precision filter                       -> the Stage-C lift
vs the paper's PubMedBERT detection F1 = 0.821.

    python scripts/bench/bench_rct.py --test-cap 3000   # sample
    python scripts/bench/bench_rct.py                    # full
"""
from __future__ import annotations
import argparse, csv, io, sys, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_limgen import prf  # noqa: E402
from gap2idea.pipeline.gap_funnel import cue_label, EmbeddingGapHead  # noqa: E402
from gap2idea.pipeline.gap_llm_filter import LLMGapFilter  # noqa: E402
from gap2idea.pipeline.llm import active_provider  # noqa: E402

RAW = "https://raw.githubusercontent.com/MengfeiLan/SAL_Type_Classification/main/data/"


def load(split, cap=None):
    raw = urllib.request.urlopen(urllib.request.Request(RAW + split + ".csv",
                                 headers={"User-Agent": "x"}), timeout=60).read().decode("utf-8", "replace")
    rows = list(csv.DictReader(io.StringIO(raw)))
    s, y = [], []
    for r in rows:
        sent = (r.get("sentences") or "").strip()
        if len(sent.split()) < 3:
            continue
        s.append(sent)
        y.append(0 if (r.get("category_spans", "[0]") or "[0]").strip() == "[0]" else 1)
    if cap:
        s, y = s[:cap], y[:cap]
    return s, np.array(y)


def stage_c(sents, sb):
    filt = LLMGapFilter(backend="api", mode="validate")
    pos = [i for i, v in enumerate(sb) if v == 1]
    keep = []
    for k in range(0, len(pos), 40):
        keep.extend(filt.judge_batch([sents[i] for i in pos[k:k + 40]]))
    final = sb.copy()
    for i, kp in zip(pos, keep):
        if not kp:
            final[i] = 0
    return final, filt.n_calls, len(pos), sum(1 for x in keep if not x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-cap", type=int, default=None)
    ap.add_argument("--train-cap", type=int, default=None)
    ap.add_argument("--encoder", default="BAAI/bge-small-en-v1.5",
                    help="Stage B encoder (swap in a domain model, e.g. pritamdeka/S-PubMedBert-MS-MARCO)")
    args = ap.parse_args()

    tr_s, tr_y = load("train", args.train_cap)
    te_s, te_y = load("test", args.test_cap)
    print(f"provider={active_provider()}  train {len(tr_s)} ({int(tr_y.sum())} lim) / "
          f"test {len(te_s)} ({int(te_y.sum())} lim, {te_y.mean():.1%})", flush=True)

    def row(name, pred):
        P, R, F = prf(te_y, pred)
        beat = "  >= PubMedBERT 0.821" if F >= 0.821 else ""
        print(f"{name:<34} P={P} R={R} F1={F}{beat}", flush=True)

    # 1) zero-shot shipped head (cross-domain: arXiv head on biomedical)
    head = EmbeddingGapHead.load(ROOT / "data/gap_head.joblib")
    pr = head.predict(te_s)
    zs = np.array([1 if (cue_label(s) == "limitation" or (l == "limitation" and p >= 0.5)) else 0
                   for s, (l, p) in zip(te_s, pr)])
    row("zero-shot shipped (cue+head)", zs)
    zc, nc, nj, nd = stage_c(te_s, zs)
    print(f"   Stage C: judged {nj} in {nc} calls, dropped {nd}", flush=True)
    row("  + Stage C", zc)

    # 2) in-domain bge+logreg (fair vs PubMedBERT)
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    enc = SentenceTransformer(args.encoder)
    Xtr = enc.encode(tr_s, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    Xte = enc.encode(te_s, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    clf = LogisticRegression(C=2, max_iter=2000, class_weight="balanced").fit(Xtr, tr_y)
    sb = clf.predict(Xte)
    ename = args.encoder.split("/")[-1]
    row(f"in-domain {ename}+logreg", sb)
    ic, nc2, nj2, nd2 = stage_c(te_s, sb)
    print(f"   Stage C: judged {nj2} in {nc2} calls, dropped {nd2}", flush=True)
    row("  + Stage C", ic)
    print("\nreference: PubMedBERT detection F1 0.821 (in-domain fine-tuned, their SOTA)")


if __name__ == "__main__":
    main()
