# -*- coding: utf-8 -*-
"""Full gap-extraction pipeline over a LIST OF PAPERS — the notebook funnel end to end.

  papers (PDFs or texts)
     -> Stage A : pdf_to_blocks + slice_section_aware         (stage_a_package)
     -> Stage B : cue rules + frozen bge-small+logreg head    (stage_b_package)
     -> Stage C : Yandex LLM precision filter (optional)      (stage_c_package)
     -> gaps.tsv / gaps.json

Every stage uses the exact components validated in the notebooks (imported from
gap2idea), so this is the notebook funnel, not the legacy extract_gaps path.

Examples
  # PDFs in a folder, Stage A+B only
  python scripts/run_extraction_pipeline.py --pdfs data/pdfs --out gaps.tsv
  # add Stage C (Yandex), biomedical prompt
  python scripts/run_extraction_pipeline.py --pdfs data/pdfs --out gaps.tsv --stage-c --domain biomed
  # from a texts jsonl ({"id","text"[,"blocks"]})
  python scripts/run_extraction_pipeline.py --texts data/paper_texts.jsonl --out gaps.tsv --stage-c
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gap2idea.pipeline.gap_funnel import (                       # noqa: E402
    pdf_to_blocks, slice_section_aware, cue_label, split_sentences, _context)

LABELS = ["none", "limitation", "future_work"]
TARGET = {"limitation", "future_work"}
ENCODER = "BAAI/bge-small-en-v1.5"


# ---- Stage B head: frozen bge-small + logreg (trained on the Stage-B train set) ----
def load_or_train_head(cache: Path, train_jsonl: Path):
    """Return (encoder, clf) — load the cached logreg or train it once from train_jsonl."""
    import joblib, numpy as np
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(ENCODER)
    if cache.exists():
        clf = joblib.load(cache)
        print(f"[stage B] loaded head {cache}")
        return enc, clf
    if not train_jsonl.exists():
        print(f"[stage B] no head and no train set ({train_jsonl}) -> cue-rules only")
        return enc, None
    from sklearn.linear_model import LogisticRegression
    rows = [json.loads(l) for l in train_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    s = [r["sentence"] for r in rows]; y = [r["label"] for r in rows]
    print(f"[stage B] training head on {len(s)} sentences ...", flush=True)
    t0 = time.time()
    X = enc.encode(s, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced").fit(X, y)
    joblib.dump(clf, cache)
    print(f"[stage B] trained + cached {cache} [{time.time()-t0:.0f}s]")
    return enc, clf


def stage_b(regions, enc, clf, threshold: float = 0.5):
    """cue rules win on type; frozen head fills the cue-less remainder. Returns gap rows."""
    flat = [(ri, si, s) for ri, r in enumerate(regions) for si, s in enumerate(r.sentences)]
    pred = {}
    if clf is not None and flat:
        import numpy as np
        X = enc.encode([s for _, _, s in flat], normalize_embeddings=True,
                       batch_size=256, show_progress_bar=False)
        proba = clf.predict_proba(X); classes = list(clf.classes_)
        for (ri, si, _), p in zip(flat, proba):
            j = int(np.argmax(p)); pred[(ri, si)] = (classes[j], float(p[j]))
    rows, seen = [], set()
    for ri, r in enumerate(regions):
        for si, s in enumerate(r.sentences):
            rule_t = cue_label(s)
            mt, mp = pred.get((ri, si), ("none", 0.0))
            if rule_t:
                gtype, source, conf = rule_t, "rule", 0.9
            elif mt in TARGET and mp >= threshold:
                gtype, source, conf = mt, "model", mp
            else:
                continue
            key = " ".join(s.lower().split())
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(gap_type=gtype, gap_sentence=s, source=source,
                             confidence=round(conf, 3), section_type=r.section_type,
                             paragraph_text=_context(r.sentences, si, w=1)))
    return rows


# ---- paper loading ----
def load_papers(args):
    """Yield (paper_id, text, blocks)."""
    if args.texts:
        import pandas as pd
        df = pd.read_json(args.texts, lines=True, dtype=False)
        for _, r in df.iterrows():
            blocks = r["blocks"] if isinstance(r.get("blocks"), list) else None
            yield str(r["id"]), str(r["text"]), blocks
        return
    # PDFs: a dir, a glob, or explicit paths
    paths = []
    for p in args.pdfs:
        if os.path.isdir(p):
            paths += sorted(glob.glob(os.path.join(p, "*.pdf")))
        else:
            paths += sorted(glob.glob(p))
    for path in paths:
        blocks = pdf_to_blocks(path)
        text = " ".join(b["text"] for b in blocks) if blocks else ""
        yield Path(path).stem, text, (blocks or None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdfs", nargs="+", help="PDF file(s), glob(s), or a folder")
    src.add_argument("--texts", help="paper_texts.jsonl with {id,text[,blocks]}")
    ap.add_argument("--out", default="gaps.tsv")
    ap.add_argument("--head", default=str(ROOT / "data" / "pipeline_head.joblib"))
    ap.add_argument("--train-jsonl", default=str(ROOT / "data" / "scibert_prep" / "train.jsonl"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--stage-c", action="store_true", help="run the Yandex LLM precision filter")
    ap.add_argument("--domain", choices=["general", "biomed"], default="general",
                    help="Stage C prompt: validate (general) or validate_rct (biomedical)")
    ap.add_argument("--llm-model", default="yandexgpt-5-lite", help="Yandex model variant for Stage C")
    args = ap.parse_args()

    enc, clf = load_or_train_head(Path(args.head), Path(args.train_jsonl))

    all_rows = []
    t0 = time.time()
    for pid, text, blocks in load_papers(args):
        if not text.strip():
            print(f"  {pid}: no text extracted, skipped"); continue
        regions = slice_section_aware(text, blocks=blocks)
        rows = stage_b(regions, enc, clf, threshold=args.threshold)
        for r in rows:
            r["id"] = pid
        all_rows.extend(rows)
        full = len(split_sentences(text))
        print(f"  {pid}: {full} sents -> {sum(len(r.sentences) for r in regions)} kept "
              f"-> {len(rows)} gaps (A+B)", flush=True)
    print(f"[stage A+B] {len(all_rows)} candidate gaps from the corpus [{time.time()-t0:.0f}s]")

    if args.stage_c and all_rows:
        os.environ.setdefault("LLM_PROVIDER", "yandex")
        from gap2idea.pipeline.gap_llm_filter import LLMGapFilter
        from gap2idea.pipeline.llm import get_llm_client
        folder = os.environ.get("YANDEX_FOLDER_ID", "").strip()
        model = f"gpt://{folder}/{args.llm_model}/latest" if folder else args.llm_model
        mode = "validate_rct" if args.domain == "biomed" else "validate"
        filt = LLMGapFilter(backend="api", model=model, mode=mode, client=get_llm_client("yandex"))
        n0 = len(all_rows); t1 = time.time()
        all_rows = filt.filter_gaps(all_rows, protect_rules=True, batch=True, use_context=True)
        print(f"[stage C] {mode}/{args.llm_model}: judged {filt.n_judged} in {filt.n_calls} calls, "
              f"kept {len(all_rows)}/{n0} [{time.time()-t1:.0f}s]")

    import pandas as pd
    cols = ["id", "gap_type", "gap_sentence", "section_type", "source", "confidence", "paragraph_text"]
    df = pd.DataFrame(all_rows, columns=cols)
    out = Path(args.out); df.to_csv(out, sep="\t", index=False)
    df.to_json(out.with_suffix(".json"), orient="records", force_ascii=False, indent=1)
    print(f"\nwrote {out} and {out.with_suffix('.json')}  ({len(df)} gaps, "
          f"{df['id'].nunique()} papers)")
    print(df.groupby('gap_type').size().to_dict())


if __name__ == "__main__":
    main()
