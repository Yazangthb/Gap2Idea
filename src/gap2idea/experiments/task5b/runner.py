"""Command-line protocol: prepare -> fit/lock -> evaluate once -> report."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

from .data import (EXPECTED, apply_structure, attach_latex_sections, digest, load_arxiv, load_fws, load_prepared,
                   load_sal, read_jsonl, sal_rows, text_hash, validate, write_json, write_jsonl)
from .metrics import bootstrap, point, select_threshold
from .models import (CLASSICAL, ENCODERS, NLI_MODEL, NLI_REVISION, ORIGINAL_HYPOTHESIS,
                     SUPPORTED, TYPED_HYPOTHESES, classical_scores, fit_classical, rule_scores)

ROOT = Path(__file__).resolve().parents[4]
LOADERS = {"sal": load_sal, "fws": load_fws, "arxiv": load_arxiv}
DEFAULT_METHODS = "tfidf_lr,binary_nb,kilicoglu_rules,modernbert_nli"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def names(value):
    values = value.split(",")
    if len(set(values)) != len(values) or any(not v for v in values):
        raise ValueError("Comma-separated names must be nonempty and unique")
    return values


def provenance():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, branch, dirty = None, None, None
    source = {p.relative_to(ROOT).as_posix(): digest(p)
              for p in sorted(Path(__file__).parent.glob("*.py"))}
    source["scripts/bench/task5b.py"] = digest(ROOT / "scripts/bench/task5b.py")
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "joblib", "torch", "transformers", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(git_commit=commit, branch=branch, dirty_status=dirty, source_sha256=source,
                python=sys.version, platform=platform.platform(), packages=versions)


def assert_source(saved):
    current = provenance()
    if current["source_sha256"] != saved["source_sha256"]:
        raise ValueError("Experiment source changed after fitting. Use the frozen code or a new run directory")
    for package in ("numpy", "scikit-learn", "joblib", "torch", "transformers"):
        if current["packages"][package] != saved["packages"][package]:
            raise ValueError(f"Environment changed after fitting: {package}")


def fetch_public(args):
    """Pin resolved Git commit and record byte hashes; never replace local input."""
    sources = [
        ("MengfeiLan/SAL_Type_Classification", "checkpoint", f"data/{p}.csv",
         ROOT / f"data/external/sal_rct/{p}.csv") for p in ("train", "dev", "test")]
    sources.append(("xiangyi-njust/FWS", "main", "Dataset/Corpus_For_FWS_Recognition.csv",
                    ROOT / "data/external/fws_corpus.csv"))
    sources = sources[:3] if args.datasets == "sal" else sources[3:] if args.datasets == "fws" else sources
    resolved, records = {}, []
    for repo, ref, source, target in sources:
        if target.exists():
            print(f"Keep existing {target.relative_to(ROOT)} ({digest(target)})", flush=True)
            records.append(dict(path=str(target.relative_to(ROOT)), sha256=digest(target), status="preexisting, origin not verified"))
            continue
        key = (repo, ref)
        if key not in resolved:
            request = urllib.request.Request(f"https://api.github.com/repos/{repo}/commits/{ref}",
                                             headers={"User-Agent": "Gap2Idea-Task5b"})
            resolved[key] = json.load(urllib.request.urlopen(request, timeout=60))["sha"]
        revision = resolved[key]
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{source}"
        print(f"Downloading {target.relative_to(ROOT)} at {revision}", flush=True)
        content = urllib.request.urlopen(url, timeout=120).read()
        # Detect HTML/error downloads before creating the official input path.
        if b"sentences" not in content[:2000] and b"text" not in content[:2000]:
            raise ValueError(f"Unexpected CSV response: {url}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        records.append(dict(path=str(target.relative_to(ROOT)), sha256=digest(target), url=url, revision=revision))
    log = ROOT / "data/task5b/downloads"
    log.mkdir(parents=True, exist_ok=True)
    write_json(log / (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"), records)


def preflight(args):
    paths = {
        "sal": ["data/external/sal_rct/train.csv", "data/external/sal_rct/dev.csv", "data/external/sal_rct/test.csv",
                "stage_c_package/sal_criteria/sal_test_allmodels.json"],
        "fws": ["data/external/fws_corpus.csv", "data/gold/gap_gold_final.jsonl", "data/gold/splits.json",
                "data/paper_texts.jsonl", "stage_c_package/sal_criteria/sal_test_allmodels.json"],
        "arxiv": ["data/gold/gap_gold_final.jsonl", "data/gold/splits.json", "data/paper_texts.jsonl"]}
    for dataset in names(args.datasets):
        if dataset not in paths:
            raise ValueError(f"Unknown dataset {dataset}")
        if dataset == "fws" and args.fws_protocol == "legacy":
            paths[dataset] = ["data/external/fws_corpus.csv", "data/scibert_prep/train.jsonl"]
        if dataset == "arxiv" and (ROOT / "data/gold/task5b_arxiv/train.jsonl").exists():
            paths[dataset] = [f"data/gold/task5b_arxiv/{p}.jsonl" for p in ("train", "val", "test")]
        missing = [p for p in paths[dataset] if not (ROOT / p).is_file()]
        print(f"{dataset}: {'INPUTS PRESENT (run prepare to validate)' if not missing else 'BLOCKED'}")
        for p in missing:
            print(f"  missing: {p}")
        print(f"  required test: {EXPECTED[dataset][0]} papers / {EXPECTED[dataset][1]} sentences")
        if dataset == "fws":
            print(f"  FWS protocol: {args.fws_protocol}; current-gold uses uploaded TRAIN papers, not positive-only rows")
    print("Rules additionally need aligned structure JSONL, or explicitly choose kilicoglu_section_proxy.")
    print("Lan EDA/PBA reproduction additionally needs the original training recipe / augmented train set.")


def prepare(args):
    for dataset in names(args.datasets):
        if dataset not in LOADERS:
            raise ValueError(f"Unknown dataset {dataset}")
        target = Path(args.output_dir) / dataset
        if target.exists():
            raise FileExistsError(f"Refusing to replace locked prepared data: {target}")
        split, inputs, notes = (load_fws(ROOT, protocol=args.fws_protocol) if dataset == "fws"
                                else LOADERS[dataset](ROOT))
        blocks, texts = ROOT / "data/gold/arxiv_latex_blocks.jsonl", ROOT / "data/paper_texts.jsonl"
        if dataset == "arxiv" and blocks.exists():
            notes["latex_sections"] = attach_latex_sections(split, blocks, texts)
            inputs.extend([blocks, texts])
        structure = ROOT / f"data/task5b/structure/{dataset}.jsonl"
        if structure.exists():
            apply_structure(split, structure)
            inputs.append(structure)
        stats = validate(split, dataset)
        target.mkdir(parents=True)
        for part, rows in split.items():
            write_jsonl(target / f"{part}.jsonl", rows)
        manifest = dict(dataset=dataset, mode="full", created_utc=now(), stats=stats, notes=notes,
                        input_sha256={str(p.resolve()): digest(p) for p in inputs},
                        prepared_sha256={part: digest(target / f"{part}.jsonl") for part in split},
                        provenance=provenance())
        write_json(target / "manifest.json", manifest)
        print(f"LOCKED {dataset}: {stats['test']} -> {target}", flush=True)


def model_spec(method, args, dataset):
    spec = dict(method=method, seed=args.seed, objective="macro_f1" if dataset == "fws" else "f1",
                threshold_grid="0.05..0.99 step .01; ties closest to .5, then higher",
                max_features=args.max_features)
    if method.startswith("modernbert"):
        spec.update(model=NLI_MODEL, revision=NLI_REVISION, max_length=256,
                    attention="eager", reference_compile=False,
                    inference_dtype="float16 on CUDA; float32 on CPU",
                    weight_loading="Windows: tensor-wise safetensors decode/copy; other OS: HF safetensors loader",
                    hypotheses=[ORIGINAL_HYPOTHESIS] if method == "modernbert_nli" else TYPED_HYPOTHESES[dataset],
                    score="max over hypotheses of entailment/(entailment+contradiction)",
                    batch_size=args.batch_size, device=args.device,
                    note="Original baseline uses broad hypothesis; typed is a separately registered variant; neither uses LLM/junk gating")
    elif method in ENCODERS:
        spec.update(model=ENCODERS[method], epochs=args.epochs, batch_size=args.batch_size,
                    learning_rate=2e-5, weight_decay=.01, max_length=256, device=args.device,
                    optimizer_foreach=False,
                    training_precision="CUDA: fp16 autocast + GradScaler, fp32 parameters; CPU: fp32",
                    gradient_checkpointing="enabled on CUDA, non-reentrant; disabled on CPU",
                    checkpoint_selection="fixed final epoch; validation selects threshold only",
                    revision=args.encoder_revision,
                    note="Generic in-domain binary fine-tune, not the Lan EDA/PBA recipe")
    elif method == "tfidf_lr":
        spec.update(ngram_range=[1, 2], min_df=2, sublinear_tf=True, C=4, class_weight="balanced")
    elif method in {"binary_nb", "zhang_nb"}:
        spec.update(ngram_range=[1, 4], binary=method == "binary_nb", min_df=2 if method == "binary_nb" else 3,
                    alpha=1 if method == "binary_nb" else 1e-5,
                    chi2_k=None if method == "binary_nb" else 14000,
                    note="Raw-text held-out adaptation; vocabulary and chi2 fit on train only; not published lemmatized balanced CV")
    elif method.startswith("kilicoglu"):
        spec.update(threshold_grid="none; deterministic rules at .5",
                    note="Python lexical port of Java rule logic; regex tokenization approximates original parser",
                    paragraph_policy="contiguous section as one paragraph (approximation)" if method.endswith("proxy") else "explicit sentence-aligned paragraph/section structure",
                    target="limitation only (arXiv extension, separate target from any-gap models)")
    return spec


def target_rows(rows, method):
    if not method.startswith("kilicoglu"):
        return rows
    return [{**r, "y": int(r["label"] == "limitation")} for r in rows]


def neural_worker(request, request_path):
    write_json(request_path, request)
    subprocess.run([sys.executable, "-X", "utf8", "-X", "faulthandler",
                    str(ROOT / "scripts/bench/task5b.py"), "_neural-worker",
                    "--request", str(request_path.resolve())], check=True, cwd=ROOT)


def neural_scores(method, rows_path, method_dir, part, spec):
    output = (method_dir / f"{part}_scores.npy").resolve()
    request = dict(operation="score", method=method, rows=str(rows_path.resolve()),
                   output=str(output), checkpoint=str((method_dir / "checkpoint").resolve()),
                   batch_size=spec["batch_size"], device=spec["device"], hypotheses=spec.get("hypotheses"))
    neural_worker(request, method_dir / f"{part}_worker_request.json")
    return np.load(output, allow_pickle=False), json.loads(Path(str(output) + ".metadata.json").read_text(encoding="utf-8"))


def fit(args):
    import joblib
    dataset = args.dataset
    methods = names(args.methods)
    if set(methods) - SUPPORTED:
        raise ValueError(f"Unknown methods {set(methods) - SUPPORTED}; choices: {sorted(SUPPORTED)}")
    split, manifest = load_prepared(Path(args.data_dir) / dataset, dataset)
    if manifest["mode"] != "full" and not getattr(args, "smoke", False):
        raise ValueError("Smoke data cannot enter a full experiment")
    run = Path(args.run_dir)
    if run.exists():
        raise FileExistsError(f"Run directory already exists; choose a new run ID: {run}")
    run.mkdir(parents=True)
    write_json(run / "input_manifest.json", manifest)
    write_json(run / "run.json", dict(dataset=dataset, mode=manifest["mode"], created_utc=now(),
                                      data_dir=str((Path(args.data_dir) / dataset).resolve()),
                                      provenance=provenance(), requested_methods=methods))
    registrations = {method: model_spec(method, args, dataset) for method in methods}
    write_json(run / "preregistered_methods.json", registrations)
    statuses = {}
    for method in methods:
        method_dir = run / method
        method_dir.mkdir()
        if method.startswith("kilicoglu") and dataset == "fws":
            statuses[method] = {"status": "skipped", "reason": "Limitation rules do not target FWS future-work labels"}
            continue
        if method == "zhang_nb" and dataset != "fws":
            statuses[method] = {"status": "skipped", "reason": "Zhang adaptation is registered for FWS only"}
            continue
        if method == "kilicoglu_section_proxy" and dataset == "arxiv" and not all(
                r.get("structure_source") for rows in split.values() for r in rows):
            statuses[method] = {"status": "skipped", "reason": "arXiv proxy needs sections aligned from LaTeX blocks for ALL sentences; positive-only gold section fields are not safe inputs"}
            continue
        if method == "kilicoglu_rules" and not all(
                all(f in r for f in ("paragraph_id", "top_section", "section_id")) and
                r.get("structure_quality") != "section_proxy"
                for rows in split.values() for r in rows):
            statuses[method] = {"status": "skipped", "reason": "Missing sentence-aligned paragraph/top-section structure; proxy available as separately named method"}
            continue
        print(f"Fitting / validating {dataset}/{method}", flush=True)
        train, val = target_rows(split["train"], method), target_rows(split["val"], method)
        spec = registrations[method].copy()
        if isinstance(manifest.get("notes"), dict):
            spec["input_protocol"] = manifest["notes"].get("protocol_id", "original dataset split")
            spec["text_transform"] = manifest["notes"].get("text_transform", "unchanged canonical sentence")
        model = None
        try:
            if method in CLASSICAL:
                model = fit_classical(method, train, seed=args.seed, max_features=args.max_features)
                joblib.dump(model, method_dir / "model.joblib")
                scores = classical_scores(model, val)
            elif method.startswith("modernbert"):
                scores, runtime = neural_scores(method, Path(args.data_dir) / dataset / "val.jsonl", method_dir, "validation", spec)
                spec.update(runtime)
            elif method in ENCODERS:
                cache_name = f"{args.encoder_revision or 'resolved-default'}-seed{args.seed}"
                base = ROOT / "data/task5b/hf_cache/classifier-init" / method / cache_name
                base_meta = base / "base_metadata.json"
                if not base_meta.exists():
                    if base.exists():
                        raise ValueError(f"Partial encoder base cache: {base}; preserve it and repair the cache before retrying")
                    neural_worker(dict(operation="prepare_encoder_base", method=method,
                        checkpoint=str(base), seed=args.seed, revision=args.encoder_revision),
                        method_dir / "base_worker_request.json")
                base_info = json.loads(base_meta.read_text(encoding="utf-8"))
                for name, sha in base_info["artifact_sha256"].items():
                    if digest(base / name) != sha:
                        raise ValueError(f"Encoder base cache changed: {base / name}")
                spec["base_initialization_metadata"] = base_info
                neural_worker(dict(operation="train_encoder", method=method,
                    rows=str((Path(args.data_dir) / dataset / "train.jsonl").resolve()),
                    checkpoint=str((method_dir / "checkpoint").resolve()), seed=args.seed,
                    epochs=args.epochs, batch_size=args.batch_size, device=args.device,
                    revision=args.encoder_revision, base_checkpoint=str(base),
                    output=str((method_dir / "training_metadata.json").resolve())),
                    method_dir / "training_worker_request.json")
                spec.update(json.loads((method_dir / "training_metadata.json").read_text(encoding="utf-8")))
                scores, runtime = neural_scores(method, Path(args.data_dir) / dataset / "val.jsonl", method_dir, "validation", spec)
                spec.update(runtime)
            else:
                scores = rule_scores(val, proxy=method.endswith("proxy"))
            if method.startswith("kilicoglu"):
                threshold, val_metrics = .5, point([r["y"] for r in val], scores >= .5)
            else:
                threshold, val_metrics = select_threshold([r["y"] for r in val], scores, spec["objective"])
            spec.update(threshold=threshold, validation_metrics=val_metrics,
                        target="limitation" if method.startswith("kilicoglu") else {"sal": "limitation", "fws": "future_work", "arxiv": "any_gap"}[dataset],
                        frozen_utc=now(), artifact_sha256={str(p.relative_to(method_dir)): digest(p)
                        for p in sorted(method_dir.rglob("*")) if p.is_file()})
            # Written BEFORE any test scoring. Evaluation consumes exactly these settings.
            write_json(method_dir / "frozen_settings.json", spec)
            write_jsonl(method_dir / "validation_predictions.jsonl", predictions(
                val, scores, threshold, method, digest(method_dir / "frozen_settings.json")))
            statuses[method] = {"status": "frozen", "threshold": threshold}
            print(f"  frozen threshold {threshold:.2f}; validation F1 {val_metrics['f1']:.4f}", flush=True)
        except Exception as exc:
            statuses[method] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            write_json(run / "fit_status.json", statuses)
            raise
        finally:
            if hasattr(model, "close"):
                model.close()
            del model
        write_json(run / "fit_status.json", statuses)
    write_json(run / "fit_status.json", statuses)
    write_json(run / "fit_complete.json", dict(completed_utc=now(),
               settings_sha256={m: digest(run / m / "frozen_settings.json")
                                for m, s in statuses.items() if s["status"] == "frozen"}))


def predictions(rows, scores, threshold, method, settings_sha):
    scores = np.asarray(scores)
    if len(rows) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid/misaligned prediction scores")
    return [{**r, "method": method, "score": float(score), "y_pred": int(score >= threshold),
             "threshold": threshold, "settings_sha256": settings_sha} for r, score in zip(rows, scores)]


def evaluate(args):
    import joblib
    run = Path(args.run_dir)
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert_source(meta["provenance"])
    complete = json.loads((run / "fit_complete.json").read_text(encoding="utf-8"))
    if (run / "test_started.json").exists():
        raise FileExistsError("Test scoring already started in this run; use saved predictions/report, not another test pass")
    split, manifest = load_prepared(meta["data_dir"], meta["dataset"])
    original_manifest = json.loads((run / "input_manifest.json").read_text(encoding="utf-8"))
    if manifest != original_manifest:
        raise ValueError("Prepared input manifest changed after fitting")
    if not complete["settings_sha256"]:
        raise ValueError("No methods fitted; see fit_status.json")
    # Validate ALL settings and artifacts before exposing any test predictions.
    for method, sha in complete["settings_sha256"].items():
        method_dir = run / method
        if digest(method_dir / "frozen_settings.json") != sha:
            raise ValueError(f"Frozen settings modified for {method}")
        spec = json.loads((method_dir / "frozen_settings.json").read_text(encoding="utf-8"))
        for path, expected in spec["artifact_sha256"].items():
            if digest(method_dir / path) != expected:
                raise ValueError(f"Model artifact changed: {method}/{path}")
    write_json(run / "test_started.json", dict(started_utc=now(), bootstrap=args.bootstrap,
                                               seed=args.seed, note="test evaluation cannot be repeated in this run"))
    summaries = {}
    for method, sha in complete["settings_sha256"].items():
        print(f"Scoring locked test: {meta['dataset']}/{method}", flush=True)
        method_dir = run / method
        spec = json.loads((method_dir / "frozen_settings.json").read_text(encoding="utf-8"))
        test = target_rows(split["test"], method)
        model = None
        try:
            if method in CLASSICAL:
                model = joblib.load(method_dir / "model.joblib")
                scores = classical_scores(model, test)
            elif method.startswith("modernbert"):
                scores, _ = neural_scores(method, Path(meta["data_dir"]) / "test.jsonl", method_dir, "test", spec)
            elif method in ENCODERS:
                scores, _ = neural_scores(method, Path(meta["data_dir"]) / "test.jsonl", method_dir, "test", spec)
            else:
                scores = rule_scores(test, proxy=method.endswith("proxy"))
        finally:
            if hasattr(model, "close"):
                model.close()
        records = predictions(test, scores, spec["threshold"], method, sha)
        write_jsonl(method_dir / "test_predictions.jsonl", records)
        predicted = [r["y_pred"] for r in records]
        summary = dict(dataset=meta["dataset"], method=method, target=spec["target"], mode=meta["mode"],
                       n_sentences=len(test), n_papers=len({r["paper_id"] for r in test}),
                       positives=sum(r["y"] for r in test), metrics=point([r["y"] for r in test], predicted),
                       ci95=bootstrap(test, predicted, n=args.bootstrap, seed=args.seed),
                       bootstrap=dict(unit="paper", replicates=args.bootstrap, seed=args.seed),
                       threshold=spec["threshold"], settings_sha256=sha,
                       predictions_sha256=digest(method_dir / "test_predictions.jsonl"))
        write_json(method_dir / "metrics.json", summary)
        summaries[method] = summary
        print(f"  positive F1 {summary['metrics']['f1']:.4f}; macro F1 {summary['metrics']['macro_f1']:.4f}", flush=True)
    write_json(run / "results.json", summaries)
    write_json(run / "test_complete.json", dict(completed_utc=now()))
    report(argparse.Namespace(run_dir=str(run)))


def report(args):
    run = Path(args.run_dir)
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    statuses = json.loads((run / "fit_status.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "input_manifest.json").read_text(encoding="utf-8"))
    notes = manifest.get("notes", {})
    protocol_line = (f"Protocol: **{notes['protocol_id']}**; {notes.get('text_transform', '')}. "
                     "Historical funnel pairing requires exact sentence identity, not matching counts."
                     if isinstance(notes, dict) and notes.get("protocol_id") else "")
    lines = [f"# Task 5b - {meta['dataset']} ({meta['mode']})", "",
             protocol_line, "",
             "Positive metrics use pooled sentences. CI resamples whole papers (conditional on the fitted model).", "",
             "| Method | Target | P | R | F1 [95% CI] | Macro F1 [95% CI] |", "|---|---|---:|---:|---:|---:|"]
    for method in meta["requested_methods"]:
        path = run / method / "metrics.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            m, ci = result["metrics"], result["ci95"]
            lines.append(f"| {method} | {result['target']} | {m['precision']:.3f} | {m['recall']:.3f} | "
                         f"{m['f1']:.3f} [{ci['f1'][0]:.3f}, {ci['f1'][1]:.3f}] | "
                         f"{m['macro_f1']:.3f} [{ci['macro_f1'][0]:.3f}, {ci['macro_f1'][1]:.3f}] |")
        else:
            status = statuses.get(method, {"status": "not attempted"})
            lines.append(f"| {method} | {status['status']}: {status.get('reason', 'test not scored')} | - | - | - | - |")
    lines += ["", "Funnel paired delta: **N/A until complete aligned sentence predictions are supplied**.",
              "Run `compare` with those predictions; aggregate headline F1 values cannot support pairing.", "",
              "Lan EDA/PBA: **not implemented as a claimed reproduction**; original augmentation recipe/code is missing.",
              "Generic BiomedBERT fine-tuning is available as a separate baseline.", "",
              "arXiv limitation rules have a different positive target from any-gap detectors; do not rank those rows together.",
              "Section-proxy rules and raw-text Zhang adaptation must retain their qualification in the paper.", ""]
    (run / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {run / 'report.md'}", flush=True)


def compare(args):
    left, right = read_jsonl(args.predictions), read_jsonl(args.reference)
    for path, rows in ((args.predictions, left), (args.reference, right)):
        metrics_path = Path(path).parent / "metrics.json"
        if not metrics_path.is_file():
            raise ValueError("Compare scored complete outputs only; use score-external to validate/import raw predictions first")
        evidence = json.loads(metrics_path.read_text(encoding="utf-8"))
        if evidence.get("predictions_sha256") != digest(path) or evidence["n_sentences"] != len(rows):
            raise ValueError("Prediction file differs from its completed evaluation evidence")
    def keyed(rows):
        result = {}
        for r in rows:
            key = str(r["paper_id"]), int(r["idx"])
            if key in result:
                raise ValueError(f"Duplicate prediction key {key}")
            if r["y_pred"] not in (0, 1) or r["y"] not in (0, 1):
                raise ValueError("Comparison requires explicit binary y/y_pred")
            if "sentence" in r and text_hash(r["sentence"]) != r["text_sha256"]:
                raise ValueError("Prediction sentence/hash disagreement")
            result[key] = r
        return result
    a, b = keyed(left), keyed(right)
    if set(a) != set(b):
        raise ValueError("Paired comparison requires identical full test keys; candidate-only files are insufficient")
    right_pred = []
    for r in left:
        ref = b[str(r["paper_id"]), int(r["idx"])]
        if (r["text_sha256"], r["y"], r["dataset"]) != (ref["text_sha256"], ref["y"], ref["dataset"]):
            raise ValueError("Paired comparison text/target/dataset mismatch")
        right_pred.append(ref["y_pred"])
    lp = [r["y_pred"] for r in left]
    lm, rm = point([r["y"] for r in left], lp), point([r["y"] for r in left], right_pred)
    output = dict(direction="method minus reference", n_sentences=len(left),
                  n_papers=len({r["paper_id"] for r in left}),
                  method_sha256=digest(args.predictions), reference_sha256=digest(args.reference),
                  differences={k: lm[k] - rm[k] for k in ("precision", "recall", "f1", "macro_f1")},
                  ci95=bootstrap(left, lp, right_pred, n=args.bootstrap, seed=args.seed))
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    write_json(args.output, output)
    print(json.dumps(output, indent=2))


def run_all(args):
    base = Path(args.run_dir)
    if base.exists():
        raise FileExistsError(f"Choose a new suite directory: {base}")
    # Validate every requested dataset before spending training time on any one.
    datasets = names(args.datasets)
    for dataset in datasets:
        load_prepared(Path(args.data_dir) / dataset, dataset)
    for dataset in datasets:
        options = argparse.Namespace(**vars(args))
        options.dataset, options.run_dir = dataset, str(base / dataset)
        fit(options)
        evaluate(options)


def sanity(args):
    """Real train/dev smoke; no official test sentences are scored."""
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Choose a new smoke output directory: {output}")
    dataset = args.dataset
    if dataset == "sal":
        train = sal_rows(ROOT / "data/external/sal_rct/train.csv")
        dev = sal_rows(ROOT / "data/external/sal_rct/dev.csv")
        inputs = {str(ROOT / f"data/external/sal_rct/{p}.csv"): digest(ROOT / f"data/external/sal_rct/{p}.csv") for p in ("train", "dev")}
        notes = dict(smoke="REAL SAL TRAIN/DEV ONLY; no official test scoring")
    else:
        source_dir = Path(args.data_dir) / dataset
        original = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
        # Deliberately do not open the official prepared test JSONL for smoke.
        train, dev = [read_jsonl(source_dir / f"{part}.jsonl") for part in ("train", "val")]
        for part in ("train", "val"):
            if digest(source_dir / f"{part}.jsonl") != original["prepared_sha256"][part]:
                raise ValueError(f"Prepared {part} changed before smoke")
        inputs = original["input_sha256"]
        notes = {**original.get("notes", {}), "smoke": "REAL FWS TRAIN/VAL ONLY; no official test scoring"}
    train_ids = sorted({r["paper_id"] for r in train})[:args.train_papers]
    dev_ids = sorted({r["paper_id"] for r in dev})[:args.dev_papers]
    if len(dev_ids) < 4:
        raise ValueError("Sanity check needs at least four dev papers")
    midpoint = len(dev_ids) // 2
    split = {"train": [r for r in train if r["paper_id"] in set(train_ids)],
             "val": [r for r in dev if r["paper_id"] in set(dev_ids[:midpoint])],
             "test": [r for r in dev if r["paper_id"] in set(dev_ids[midpoint:])]}
    stats = validate(split, dataset, strict=False)
    prepared = output / "prepared" / dataset
    prepared.mkdir(parents=True)
    for part, rows in split.items():
        write_jsonl(prepared / f"{part}.jsonl", rows)
    write_json(prepared / "manifest.json", dict(dataset=dataset, mode="smoke", created_utc=now(), stats=stats,
        notes=notes, input_sha256=inputs,
        prepared_sha256={p: digest(prepared / f"{p}.jsonl") for p in split}))
    options = argparse.Namespace(dataset=dataset, methods="tfidf_lr,binary_nb," +
        ("kilicoglu_section_proxy" if dataset == "sal" else "zhang_nb") +
        (",modernbert_nli" if args.with_nli else ""), seed=0, max_features=20000, batch_size=args.batch_size,
        device=args.device, epochs=1, encoder_revision=None, data_dir=str(output / "prepared"),
        run_dir=str(output / "run"), bootstrap=200, smoke=True)
    fit(options)
    evaluate(options)
    # Export a paired comparison exercising ID alignment, independently of headline metrics.
    compare(argparse.Namespace(predictions=str(output / "run/tfidf_lr/test_predictions.jsonl"),
        reference=str(output / "run/binary_nb/test_predictions.jsonl"),
        bootstrap=200, seed=0, output=str(output / "paired_smoke.json")))


def parser():
    from .secondary import audit_zhang, score_external
    from .workers import execute
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("_neural-worker", help="Internal GPU worker; invoked automatically")
    q.add_argument("--request", required=True)
    q.set_defaults(func=execute)
    q = sub.add_parser("fetch-public", help="Download public SAL/FWS at resolved commit IDs")
    q.add_argument("--datasets", choices=("sal", "fws", "sal,fws"), default="sal,fws")
    q.set_defaults(func=fetch_public)
    q = sub.add_parser("preflight", help="List required inputs without model downloads")
    q.add_argument("--datasets", default="sal,fws,arxiv")
    q.add_argument("--fws-protocol", choices=("current-gold", "legacy"), default="current-gold")
    q.set_defaults(func=preflight)
    q = sub.add_parser("prepare", help="Validate and lock full datasets; refuses count/identity drift")
    q.add_argument("--datasets", default="sal,fws,arxiv")
    q.add_argument("--output-dir", default=str(ROOT / "data/task5b/prepared"))
    q.add_argument("--fws-protocol", choices=("current-gold", "legacy"), default="current-gold")
    q.set_defaults(func=prepare)
    def fitting(q):
        q.add_argument("--data-dir", default=str(ROOT / "data/task5b/prepared"))
        q.add_argument("--methods", default=DEFAULT_METHODS)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--max-features", type=int, default=200000)
        q.add_argument("--batch-size", type=int, default=8)
        q.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        q.add_argument("--epochs", type=int, default=3)
        q.add_argument("--encoder-revision", help="Optional pinned encoder commit (run one encoder at a time)")
        q.add_argument("--run-dir", required=True)
    q = sub.add_parser("fit", help="Fit on train, choose thresholds on val, freeze settings BEFORE test")
    fitting(q)
    q.add_argument("--dataset", choices=tuple(EXPECTED), required=True)
    q.set_defaults(func=fit)
    q = sub.add_parser("evaluate", help="Score test once using frozen settings")
    q.add_argument("--run-dir", required=True)
    q.add_argument("--bootstrap", type=int, default=2000)
    q.add_argument("--seed", type=int, default=0)
    q.set_defaults(func=evaluate)
    q = sub.add_parser("run", help="Fit then evaluate each dataset; run dir must be new")
    fitting(q)
    q.add_argument("--datasets", default="sal,fws,arxiv")
    q.add_argument("--bootstrap", type=int, default=2000)
    q.set_defaults(func=run_all)
    q = sub.add_parser("report", help="Regenerate report from saved results without rescoring")
    q.add_argument("--run-dir", required=True)
    q.set_defaults(func=report)
    q = sub.add_parser("compare", help="Paired paper-bootstrap against complete aligned funnel predictions")
    q.add_argument("--predictions", required=True)
    q.add_argument("--reference", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--bootstrap", type=int, default=2000)
    q.add_argument("--seed", type=int, default=0)
    q.set_defaults(func=compare)
    q = sub.add_parser("score-external", help="Score complete frozen predictions (funnel, Lan, or cached models)")
    q.add_argument("--data-dir", default=str(ROOT / "data/task5b/prepared"))
    q.add_argument("--dataset", choices=tuple(EXPECTED), required=True)
    q.add_argument("--method", required=True)
    q.add_argument("--target", choices=("limitation", "future_work", "any_gap"), required=True)
    q.add_argument("--predictions", required=True)
    q.add_argument("--protocol", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--bootstrap", type=int, default=2000)
    q.add_argument("--seed", type=int, default=0)
    q.set_defaults(func=score_external)
    q = sub.add_parser("audit-zhang", help="Separate balanced CV leakage diagnostic; never part of unified test")
    q.add_argument("--corpus", default=str(ROOT / "data/external/fws_corpus.csv"))
    q.add_argument("--lemmatized-jsonl", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--folds", type=int, default=10)
    q.add_argument("--seed", type=int, default=0)
    q.set_defaults(func=audit_zhang)
    q = sub.add_parser("sanity", help="Small real SAL/FWS train/dev smoke; no official test scoring")
    q.add_argument("--dataset", choices=("sal", "fws"), default="sal")
    q.add_argument("--data-dir", default=str(ROOT / "data/task5b/prepared"))
    q.add_argument("--output-dir", required=True)
    q.add_argument("--with-nli", action="store_true", help="Download/use actual pinned ModernBERT (~1.6 GB)")
    q.add_argument("--train-papers", type=int, default=8)
    q.add_argument("--dev-papers", type=int, default=4)
    q.add_argument("--batch-size", type=int, default=4)
    q.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    q.set_defaults(func=sanity)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if hasattr(args, "bootstrap") and args.bootstrap < 1:
            raise ValueError("--bootstrap must be positive")
        if hasattr(args, "batch_size") and args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        if hasattr(args, "epochs") and args.epochs < 1:
            raise ValueError("--epochs must be positive")
        args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0
