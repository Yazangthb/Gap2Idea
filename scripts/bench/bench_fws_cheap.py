"""Beat the FWS recognition SOTA with a CHEAP lexical model.

Zhang et al. (2022)'s best recognition model is itself cheap: Bernoulli Naive Bayes,
Macro-F1 0.9073 -- it beats fine-tuned BERT (our runs: ~0.853) because future-work
recognition is near-lexical ("in future work", "we plan to", "remains to be").

We push the cheap-lexical idea further than their NB: TF-IDF word+char n-grams +
logistic regression / linear SVM, trained on the FULL data (all negatives, class-
weighted -- not the 1:3 undersample the transformers used), threshold-tuned on dev for
MACRO-F1 (the metric they report). Runs in seconds on CPU, no GPU. Optionally + Stage C.

    python scripts/bench/bench_fws_cheap.py            # all cheap models
    python scripts/bench/bench_fws_cheap.py --stage-c  # + LLM filter on the best
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_limgen import prf  # noqa: E402
from bench_fws import load, macro_f1, stage_c  # noqa: E402

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer  # noqa: F401


def word_char_union():
    from sklearn.pipeline import FeatureUnion
    return FeatureUnion([
        ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])


def build(name):
    if name == "tfidf_word+logreg":
        return Pipeline([("v", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                         ("c", LogisticRegression(C=8, max_iter=2000, class_weight="balanced"))])
    if name == "tfidf_wordchar+logreg":
        return Pipeline([("v", word_char_union()),
                         ("c", LogisticRegression(C=8, max_iter=3000, class_weight="balanced"))])
    if name == "tfidf_wordchar+svm":
        return Pipeline([("v", word_char_union()),
                         ("c", CalibratedClassifierCV(LinearSVC(C=1, class_weight="balanced"), cv=3))])
    if name == "count+bernoulliNB":   # their approach
        return Pipeline([("v", CountVectorizer(ngram_range=(1, 1), min_df=2, binary=True)),
                         ("c", BernoulliNB())])
    if name == "tfidf_word+complementNB":
        return Pipeline([("v", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                         ("c", ComplementNB())])
    raise ValueError(name)


def proba(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    d = clf.decision_function(X)
    return 1 / (1 + np.exp(-d))


def tune_macro(y, p):
    best_t, best_m = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        m = macro_f1(y, (p >= t).astype(int))
        if m > best_m: best_m, best_t = m, t
    return best_t


MODELS = ["tfidf_word+logreg", "tfidf_wordchar+logreg", "tfidf_wordchar+svm",
          "count+bernoulliNB", "tfidf_word+complementNB"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--stage-c", action="store_true", help="add Stage C to the best cheap model")
    ap.add_argument("--test-cap", type=int, default=None)
    args = ap.parse_args()

    tr_s, tr_y = load("train"); dv_s, dv_y = load("dev"); te_s, te_y = load("test", args.test_cap)
    print(f"FWS: train {len(tr_s)} ({int(tr_y.sum())} fw) / dev {len(dv_s)} / "
          f"test {len(te_s)} ({int(te_y.sum())} fw, {te_y.mean():.1%})  [FULL negatives, class-weighted]")
    print("reference: Zhang 2022 recognition Macro-F1 0.9073 (Bernoulli NB)\n")
    print(f"{'cheap model':<26} {'P':>6} {'R':>6} {'posF1':>6} {'macroF1':>8}")

    best = None
    for name in args.models.split(","):
        clf = build(name.strip()).fit(tr_s, tr_y)
        t = tune_macro(dv_y, proba(clf, dv_s))
        pt = proba(clf, te_s); sb = (pt >= t).astype(int)
        P, R, F = prf(te_y, sb); M = macro_f1(te_y, sb)
        beat = " >=0.907" if M >= 0.9073 else ""
        print(f"{name.strip():<26} {P:6.3f} {R:6.3f} {F:6.3f} {M:8.3f}{beat}")
        if best is None or M > best[0]:
            best = (M, name.strip(), sb, te_s)

    print(f"\nbest cheap model: {best[1]}  macroF1={best[0]:.3f}  (vs their 0.9073)")
    if args.stage_c:
        _, name, sb, sents = best
        fc, nc, nj, nd = stage_c(sents, sb)
        Pc, Rc, Fc = prf(te_y, fc); Mc = macro_f1(te_y, fc)
        print(f"{name} + Stage C:  P={Pc:.3f} R={Rc:.3f} posF1={Fc:.3f} macroF1={Mc:.3f}"
              f"{' >=0.907' if Mc >= 0.9073 else ''}  (judged {nj} in {nc} calls, dropped {nd})")


if __name__ == "__main__":
    main()
