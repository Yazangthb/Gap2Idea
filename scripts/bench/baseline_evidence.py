"""Run fingerprints, attempt accounting, coverage and the V100 Ollama transport.

No labels or credentials are sent to the server except the numbered sentence inputs.
"""
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


def file_hash(path):
    # Text normalization makes fingerprints identical on Windows and Linux checkouts.
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()


def code_hashes():
    paths = git("ls-files", "*.py").splitlines()
    paths += [str(p.relative_to(ROOT).as_posix()) for p in (ROOT / "scripts/bench").glob("baseline*.py")]
    paths += ["scripts/bench/whole_paper_other_llm.py"]
    return {p: file_hash(ROOT / p) for p in sorted(set(paths))}


def committed(path):
    try:
        rel = Path(path).resolve().relative_to(ROOT).as_posix()
        return bool(git("log", "-1", "--format=%H", "--", rel)) and not git("status", "--porcelain", "--", rel)
    except (ValueError, subprocess.CalledProcessError):
        return False


def fingerprint(args, rows, todo, system):
    from baseline_qwen import EXTRA, SETUPS
    if args.dataset in {"sal", "fws"}:
        sources = [f"data/splits/{args.dataset}_{args.split}.jsonl"]
    else:
        sources = ["data/gold/splits.json", "data/gold/gap_gold_final.jsonl", "data/paper_texts.jsonl",
                   "stage_c_package/sal_criteria/sal_test_allmodels.json"]
    return dict(dataset=args.dataset, split=args.split, setup=args.setup, prompt=args.prompt,
                model=getattr(args, "model", "qwen3.8-27b"),
                backend=getattr(args, "backend", "openai"),
                num_ctx=getattr(args, "num_ctx", 131072),
                limit_papers=args.limit_papers, temperature=0, extra=EXTRA,
                max_tokens=SETUPS[args.setup]["max_tokens"],
                data_sha256=digest(rows), input_sha256=digest(todo), prompt_sha256=digest(system),
                data_file_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
                code_sha256=digest(code_hashes()))


def prepare_run(args, rows, todo, system, path):
    spec = fingerprint(args, rows, todo, system)
    plan_path = getattr(args, "freeze_plan", None)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8")) if plan_path else None
    if args.split == "test":
        if not plan_path or not committed(plan_path):
            raise SystemExit("Commit --freeze-plan (with hashes and model revisions) before any test call")
        if plan.get("code_hashes") != code_hashes():
            raise SystemExit("Code differs from frozen plan; freeze and commit a new plan in a fresh folder")
        if spec not in plan["runs"]:
            raise SystemExit("Configuration, prompt, data or inputs differ from the frozen plan")
    model_info = plan.get("models", {}).get(spec["model"]) if plan else None
    if args.split == "test" and not model_info:
        raise SystemExit("Model identity missing from frozen plan")
    if spec["backend"] == "ollama" and plan:
        current = OllamaClient(getattr(args, "base_url", None), spec["num_ctx"]).identity(spec["model"])
        if current != model_info:
            raise SystemExit("Served model digest/template differs from frozen model")
    manifest = dict(spec, model_identity=model_info,
                    plan_sha256=digest(plan) if plan else None, code_commit=git("rev-parse", "HEAD"))
    sidecar = path.with_suffix(".manifest.json")
    # Commit may move after freezing the plan, but the actual code content must match.
    if sidecar.exists():
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if {k: v for k, v in previous.items() if k != "code_commit"} != {
                k: v for k, v in manifest.items() if k != "code_commit"}:
            raise SystemExit(f"Refusing to mix configurations in {path}; use a fresh output directory")
        return previous
    if path.exists():
        raise SystemExit(f"Refusing to append to legacy verdicts without a manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("x", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    path.with_suffix(".prompt.txt").write_text(system, encoding="utf-8")
    return manifest


def validate_records(records, todo):
    expected = {u["unit"]: u["keys"] for u in todo}
    seen = set()
    for r in records:
        if r["unit"] in seen or expected.get(r["unit"]) != r["keys"]:
            raise ValueError(f"Duplicate, foreign or mismatched saved unit: {r['unit']}")
        if r["status"] not in {"ok", "failed"}:
            raise ValueError(f"Unknown status: {r['status']}")
        if r["status"] == "ok" and any(k not in r["keys"] for k in r["kept"]):
            raise ValueError("Saved prediction outside its input")
        seen.add(r["unit"])


def coverage(rows, todo, records):
    expected = {tuple(k) for u in todo for k in u["keys"]}
    answered = Counter(tuple(k) for r in records if r["status"] == "ok" for k in r["keys"])
    attempted = {tuple(k) for r in records for k in r["keys"]}
    return dict(total_sentences=len(rows), requested_sentences=len(expected),
                excluded_by_stage_a=len(rows) - len(expected), answered_sentences=len(answered),
                failed_sentences=len(attempted - set(answered)),
                unattempted_sentences=len(expected - attempted),
                duplicate_answers=sum(n - 1 for n in answered.values()),
                expected_calls=len(todo), saved_calls=len(records),
                failed_calls=sum(r["status"] != "ok" for r in records),
                all_calls_recorded=len(records) == len(todo),
                every_requested_sentence_answered=set(answered) == expected and all(n == 1 for n in answered.values()))


def costs(records):
    attempts = []
    for r in records:
        if "attempt_log" in r:
            attempts.extend(r["attempt_log"])
        else:
            # Historical verdicts kept only the final attempt; retry costs are unknown.
            attempts.extend(dict(status="failed") for _ in range(r.get("attempts", 1) - 1))
            attempts.append(r)
    result = dict(attempts=len(attempts), failed_attempts=sum(a["status"] != "ok" for a in attempts),
                  seconds=round(sum(r["seconds"] for r in records), 4))
    for field in ("prompt_tokens", "completion_tokens"):
        values = [a.get(field) for a in attempts]
        result[field + "_known"] = sum(v for v in values if v is not None)
        result[field + "_missing_attempts"] = sum(v is None for v in values)
        result[field] = sum(values) if all(v is not None for v in values) else None
    return result


class OllamaClient:
    """OpenAI-shaped transport; native think=false and schema are explicit, not ignored extras."""

    def __init__(self, base_url=None, num_ctx=131072):
        self.base_url = (base_url or os.environ.get("QWEN_OLLAMA_URL", "http://127.0.0.1:11445")).rstrip("/")
        self.num_ctx = num_ctx
        self.chat = NS(completions=self)

    def request(self, route, body=None):
        req = Request(self.base_url + route, data=json.dumps(body).encode() if body is not None else None,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=3600) as response:
            return json.load(response)

    def identity(self, model):
        tags = self.request("/api/tags")["models"]
        tag = next((t for t in tags if t["name"] == model), None)
        if not tag:
            raise ValueError(f"Model not installed: {model}")
        show = self.request("/api/show", {"model": model})
        return dict(digest=tag["digest"], details=show.get("details"),
                    model_info=show.get("model_info"), template_sha256=digest(show.get("template")),
                    parameters=show.get("parameters"), system=show.get("system"),
                    server_version=self.request("/api/version")["version"])

    def create(self, **kw):
        # Disable all default sampling penalties and reject implicit input truncation.
        r = self.request("/api/chat", dict(model=kw["model"], messages=kw["messages"], stream=False,
                         think=False, format=kw["response_format"]["json_schema"]["schema"],
                         truncate=False, keep_alive="1h",
                         options=dict(temperature=0, seed=0, num_predict=kw["max_tokens"],
                                      num_ctx=self.num_ctx, repeat_penalty=1.0)))
        if r.get("error"):
            raise ValueError(r["error"])
        return NS(choices=[NS(finish_reason=r.get("done_reason"), message=NS(content=r["message"]["content"]))],
                  usage=NS(prompt_tokens=r.get("prompt_eval_count"), completion_tokens=r.get("eval_count")),
                  model=r.get("model"), id=r.get("created_at"),
                  server_timings={k: v for k, v in r.items() if k.endswith("duration") or "cached" in k})
