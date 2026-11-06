"""Secondary protocols kept separate from the unified held-out comparison."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .data import csv_rows, digest, load_prepared, read_jsonl, text_hash, write_json, write_jsonl
from .metrics import bootstrap, point


def score_external(args):
    """Accept complete sentence predictions from a frozen external model/funnel.

    The protocol JSON describes training/split/threshold provenance. Import does
    not certify the checkpoint's history and does not tune anything on test.
    """
    split, manifest = load_prepared(Path(args.data_dir) / args.dataset, args.dataset)
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    for field in ("method", "training_data", "threshold_selection", "model_or_code_revision", "limitations"):
        if field not in protocol:
            raise ValueError(f"External protocol JSON missing {field}")
        if not protocol[field] or (isinstance(protocol[field], str) and protocol[field].lstrip().startswith("REQUIRED")):
            raise ValueError(f"External protocol {field} must contain actual provenance, not an unfilled template")
    if protocol["method"] != args.method:
        raise ValueError("External protocol method differs from --method")
    source = read_jsonl(args.predictions)
    keyed = {}
    for r in source:
        key = str(r["paper_id"]), int(r["idx"])
        if key in keyed:
            raise ValueError(f"Duplicate external prediction {key}")
        keyed[key] = r
    test = split["test"]
    if set(keyed) != {(r["paper_id"], r["idx"]) for r in test}:
        raise ValueError("External predictions must cover every test row exactly, including negatives")
    records = []
    for row in test:
        r = keyed[row["paper_id"], row["idx"]]
        actual_hash = text_hash(r["sentence"]) if "sentence" in r else r.get("text_sha256")
        if actual_hash != row["text_sha256"]:
            raise ValueError(f"External sentence mismatch: {row['paper_id']}/{row['idx']}")
        if type(r["y_pred"]) is not int or r["y_pred"] not in (0, 1):
            raise ValueError("External predictions must contain integer binary y_pred")
        target = row["y"] if args.target == "any_gap" else int(row["label"] == args.target)
        if "y" in r and r["y"] != target:
            raise ValueError("External gold target differs from locked corpus")
        records.append({**row, "y": target, "method": args.method, "y_pred": r["y_pred"],
                        "external_protocol_sha256": digest(args.protocol)})
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to replace external evaluation: {output}")
    output.mkdir(parents=True)
    # Protocol and input fingerprints are saved before computing metrics.
    write_json(output / "external_protocol.json", protocol)
    write_json(output / "input_manifest.json", manifest)
    write_json(output / "input_fingerprints.json", dict(predictions=digest(args.predictions), protocol=digest(args.protocol)))
    write_jsonl(output / "test_predictions.jsonl", records)
    pred = [r["y_pred"] for r in records]
    result = dict(method=args.method, dataset=args.dataset, target=args.target, mode=manifest["mode"],
                  n_sentences=len(records), n_papers=len({r["paper_id"] for r in records}),
                  metrics=point([r["y"] for r in records], pred),
                  ci95=bootstrap(records, pred, n=args.bootstrap, seed=args.seed),
                  bootstrap=dict(unit="paper", replicates=args.bootstrap, seed=args.seed),
                  predictions_sha256=digest(output / "test_predictions.jsonl"),
                  note="Imported frozen predictions; training/protocol provenance is supplied by the experiment owner")
    write_json(output / "metrics.json", result)
    print(json.dumps(result, indent=2))


def audit_zhang(args):
    """Balanced sentence-CV diagnostic, NOT the proposed 209-paper test.

    The two pipelines differ in deliberate global fitting versus fold-local
    fitting of vocabulary/IDF/chi2. This measures that combined leakage, not an
    isolated chi2 effect. Exact reproduction still depends on original lemmas,
    sampling order, hyperparameters, and folds; never promise historical .907.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.model_selection import StratifiedKFold
    from sklearn.naive_bayes import BernoulliNB
    corpus = csv_rows(args.corpus)
    lemmas = read_jsonl(args.lemmatized_jsonl)
    if len(lemmas) != len(corpus):
        raise ValueError("Need one aligned lemmatized record for every original FWS CSV row")
    for i, (raw, lemma) in enumerate(zip(corpus, lemmas)):
        if int(lemma["raw_idx"]) != i or lemma["text_sha256"] != text_hash(raw["text"]):
            raise ValueError(f"Lemmatization alignment drift at original FWS row {i}")
        if not isinstance(lemma["lemmatized_text"], str):
            raise ValueError("Missing lemmatized_text")
    y = np.array([int(r["label"]) for r in corpus])
    pos, neg = np.flatnonzero(y == 1).tolist(), np.flatnonzero(y == 0).tolist()
    rng = random.Random(args.seed)
    rng.shuffle(neg)
    selected = pos + neg[:len(pos)]
    rng.shuffle(selected)
    if len(pos) < args.folds:
        raise ValueError("Too few positives for requested CV folds")
    texts = [lemmas[i]["lemmatized_text"] for i in selected]
    y = y[selected]
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    write_json(out / "protocol.json", dict(
        mode="diagnostic_only_NOT_unified_test", seed=args.seed, folds=args.folds,
        positive_count=len(pos), balanced_rows=len(selected),
        vectorizer="TF-IDF word (1,4), min_df=3, sublinear_tf=True",
        selector="chi2 k=14000 (or all if vocabulary smaller)", alpha=1e-5,
        clean="all preprocessing and supervised selection fitted inside folds",
        deliberately_leaky="vocabulary/IDF and chi2 fitted globally before sentence CV",
        limitations="Sentence CV has shared papers; global-vs-local contrast combines vocabulary/IDF and chi2 exposure. Not an exact reproduction certificate.",
        corpus_sha256=digest(args.corpus), lemmatized_sha256=digest(args.lemmatized_jsonl)))
    vectorizer = TfidfVectorizer(ngram_range=(1, 4), min_df=3, sublinear_tf=True)
    global_x = vectorizer.fit_transform(texts)
    selector = SelectKBest(chi2, k=min(14000, global_x.shape[1])).fit(global_x, y)
    leaky_x = selector.transform(global_x)
    clean_pred, leaky_pred = np.zeros(len(y), dtype=int), np.zeros(len(y), dtype=int)
    fold_ids, metrics = np.zeros(len(y), dtype=int), []
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train, test) in enumerate(cv.split(texts, y)):
        fold_vec = TfidfVectorizer(ngram_range=(1, 4), min_df=3, sublinear_tf=True)
        fold_x = fold_vec.fit_transform([texts[i] for i in train])
        fold_selector = SelectKBest(chi2, k=min(14000, fold_x.shape[1])).fit(fold_x, y[train])
        clean = BernoulliNB(alpha=1e-5).fit(fold_selector.transform(fold_x), y[train])
        clean_pred[test] = clean.predict(fold_selector.transform(fold_vec.transform([texts[i] for i in test])))
        leaky_pred[test] = BernoulliNB(alpha=1e-5).fit(leaky_x[train], y[train]).predict(leaky_x[test])
        fold_ids[test] = fold
        metrics.append(dict(fold=fold, clean=point(y[test], clean_pred[test]),
                            deliberately_leaky=point(y[test], leaky_pred[test])))
        print(f"diagnostic fold {fold + 1}/{args.folds} done", flush=True)
    records = [dict(raw_idx=i, paper_id=corpus[i]["id"], text_sha256=text_hash(corpus[i]["text"]),
                    y=int(y[j]), fold=int(fold_ids[j]), clean_prediction=int(clean_pred[j]),
                    deliberately_leaky_prediction=int(leaky_pred[j])) for j, i in enumerate(selected)]
    write_jsonl(out / "oof_predictions.jsonl", records)
    result = dict(folds=metrics, pooled_clean=point(y, clean_pred),
                  pooled_deliberately_leaky=point(y, leaky_pred),
                  mean_fold_macro_f1_clean=float(np.mean([r["clean"]["macro_f1"] for r in metrics])),
                  mean_fold_macro_f1_deliberately_leaky=float(np.mean([r["deliberately_leaky"]["macro_f1"] for r in metrics])))
    write_json(out / "diagnostic_results.json", result)
    print(json.dumps(result, indent=2))
