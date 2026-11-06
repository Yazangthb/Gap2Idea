"""Task 5b model implementations; no LLM/API inference."""
from __future__ import annotations

import gc
import re
import random
import sys
from pathlib import Path
from contextlib import nullcontext
from collections import defaultdict

import numpy as np

NLI_MODEL = "tasksource/ModernBERT-large-nli"
NLI_REVISION = "ca476cb923a8637073d4ceb0f19f7fc236e260d4"
ORIGINAL_HYPOTHESIS = (
    "This sentence points out a limitation, weakness, shortcoming, open research problem, "
    "or future-work direction of the research described.")
TYPED_HYPOTHESES = {"sal": ["This is a limitation."],
                    "fws": ["This is future work."],
                    "arxiv": ["This is a limitation.", "This is future work.", "This is a research gap."]}
ENCODERS = {"scibert_ft": "allenai/scibert_scivocab_uncased",
            "biomedbert_ft": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"}
CLASSICAL = {"tfidf_lr", "binary_nb", "zhang_nb"}
SUPPORTED = CLASSICAL | set(ENCODERS) | {
    "modernbert_nli", "modernbert_nli_typed", "kilicoglu_rules", "kilicoglu_section_proxy"}


def fit_classical(method, train, seed=0, max_features=200000):
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.pipeline import Pipeline
    if method == "tfidf_lr":
        pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                          max_features=max_features)),
            ("classifier", LogisticRegression(C=4, max_iter=2000, class_weight="balanced", random_state=seed))])
    else:
        vectorizer = (CountVectorizer(ngram_range=(1, 4), binary=True, min_df=2, max_features=max_features)
                      if method == "binary_nb" else TfidfVectorizer(
                          ngram_range=(1, 4), min_df=3, sublinear_tf=True, max_features=max_features))
        steps = [("vectorizer", vectorizer)]
        # Fit vocabulary AND chi2 using TRAIN labels only. This is the unified
        # held-out adaptation, not a reproduction of the published balanced CV.
        if method == "zhang_nb":
            train_x = vectorizer.fit_transform([r["sentence"] for r in train])
            train_y = [r["y"] for r in train]
            selector = SelectKBest(chi2, k=min(14000, train_x.shape[1])).fit(train_x, train_y)
            classifier = BernoulliNB(alpha=1e-5).fit(selector.transform(train_x), train_y)
            return Pipeline(steps + [("chi2", selector), ("classifier", classifier)])
        steps.append(("classifier", BernoulliNB(alpha=1 if method == "binary_nb" else 1e-5)))
        pipeline = Pipeline(steps)
    pipeline.fit([r["sentence"] for r in train], [r["y"] for r in train])
    return pipeline


def classical_scores(pipeline, rows):
    positive = list(pipeline.classes_).index(1)
    return pipeline.predict_proba([r["sentence"] for r in rows])[:, positive]


def nli_evc(logits, id2label):
    """Resolve semantic labels, never assume model output order."""
    label_to_index = {str(v).lower(): int(k) for k, v in id2label.items()}
    def index(prefix):
        matches = [v for k, v in label_to_index.items() if k.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Cannot resolve {prefix!r} in {id2label}")
        return matches[0]
    e, c = index("entail"), index("contrad")
    # softmax(e,c) equals three-way entailment/(entailment+contradiction).
    # Neutral is intentionally excluded, as in nli_score_remote.py.
    difference = np.clip(np.asarray(logits)[:, e] - np.asarray(logits)[:, c], -700, 700)
    return 1 / (1 + np.exp(-difference))


class ModernBERT:
    def __init__(self, batch_size=8, device="auto"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
        self.torch = torch
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else (
            "cpu" if device == "auto" else device)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL, revision=NLI_REVISION)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        if sys.platform == "win32":
            from huggingface_hub import hf_hub_download
            from .safeweights import load_windows_model
            path = hf_hub_download(NLI_MODEL, "model.safetensors", revision=NLI_REVISION)
            config = AutoConfig.from_pretrained(NLI_MODEL, revision=NLI_REVISION, reference_compile=False)
            self.model = load_windows_model(path, config, self.device, dtype)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                NLI_MODEL, revision=NLI_REVISION, use_safetensors=True, torch_dtype=dtype,
                attn_implementation="eager", reference_compile=False).eval().to(self.device)

    def scores(self, rows, hypotheses):
        torch = self.torch
        out = np.empty((len(rows), len(hypotheses)), dtype=float)
        # Stable length sorting is an efficiency measure only. Original row order
        # is restored before predictions are serialized.
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["sentence"]))
        for h, hypothesis in enumerate(hypotheses):
            for start in range(0, len(order), self.batch_size):
                ix = order[start:start + self.batch_size]
                tokens = self.tokenizer([rows[i]["sentence"] for i in ix], [hypothesis] * len(ix),
                                        padding=True, truncation=True, max_length=256, return_tensors="pt")
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                with torch.inference_mode():
                    logits = self.model(**tokens).logits.float().cpu().numpy()
                out[ix, h] = nli_evc(logits, self.model.config.id2label)
                if start and start % (self.batch_size * 100) == 0:
                    print(f"  NLI hypothesis {h + 1}/{len(hypotheses)}: {start}/{len(order)} sentences", flush=True)
        return out.max(axis=1)

    def close(self):
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def introduction(sentence):
    text = sentence.lower()
    # Java's original uses tokenized word count. Regex tokenization here is an
    # explicit approximation, not the original Stanford biomedical preprocessing.
    return bool(re.search(r" (limitations|weaknesses)[^\w\s]$", text) or
                (len(re.findall(r"\w+|[^\w\s]", text)) <= 10 and
                 re.search(r" (limitations|weaknesses)", text)))


def negative_filter(sentence):
    first = re.search(r"[a-z]+", sentence.lower())
    return bool((first and first.group() in {"however", "nonetheless", "nevertheless"}) or
                re.search(r"\[([0-9,\-]+)\]", sentence) or
                re.search(r"\.([0-9,\-]+)$", sentence) or " et al" in sentence)


def rule_scores(rows, proxy=False):
    """Section gate + original Java span/filter logic, with declared tokenizer port.

    Faithful structural mode requires explicit section and paragraph boundaries.
    Proxy mode treats each contiguous section as a paragraph; names/results remain
    distinct to avoid presenting the proxy as the original published method.
    """
    groups = defaultdict(list)
    contiguous = defaultdict(lambda: [-1, None])
    for i, r in enumerate(rows):
        if not proxy:
            for field in ("section", "top_section", "paragraph_id", "section_id"):
                if field not in r:
                    raise ValueError("kilicoglu_rules needs sentence-aligned structure; see README")
            sid = r["section_id"]
        else:
            tracker = contiguous[r["paper_id"]]
            if tracker[1] != r.get("section", ""):
                tracker[:] = [tracker[0] + 1, r.get("section", "")]
            sid = tracker[0]
        groups[r["paper_id"], sid].append((i, r))
    out = np.zeros(len(rows), dtype=float)
    for group in groups.values():
        group = sorted(group, key=lambda pair: pair[1]["idx"])
        heading = group[0][1].get("section", "").lower()
        top = group[0][1].get("top_section", heading).lower()
        if not any(word in top for word in ("discussion", "conclusion", "limitation", "weakness")):
            continue
        limited = "limitation" in heading or "weakness" in heading
        both = limited and "strength" in heading
        active = set(range(len(group))) if limited and not both else set()
        if both:
            begin = next((j for j, (_, r) in enumerate(group)
                          if re.search(r"limitation|weakness", r["sentence"], re.I)), None)
            if begin is not None:
                end = next((j for j in range(begin + 1, len(group))
                            if "strength" in group[j][1]["sentence"].lower()), len(group) - 1)
                active = set(range(begin, end + 1))
            else:
                # Follow Java's heading-order half-section fallback. There is
                # no data-driven lexical tuning or additional future-work filter.
                lpos = min([heading.find(w) for w in ("limitation", "weakness") if w in heading])
                middle = len(group) // 2
                active = set(range(middle) if lpos < heading.find("strength") else range(middle, len(group)))
        introduced = set()
        for j, (i, r) in enumerate(group):
            paragraph = r.get("paragraph_id", "section_proxy")
            intro = introduction(r["sentence"])
            if intro:
                introduced.add(paragraph)
            eligible = j in active if limited else paragraph in introduced
            out[i] = float(eligible and not intro and not negative_filter(r["sentence"]))
    return out


def prepare_encoder_base(method, output, seed=0, revision=None):
    """Convert the pinned pretrained encoder + seeded binary head on CPU once.

    A separate process exits after saving, so original .bin storage/mappings and
    conversion threads are gone before GPU training allocates optimizer states.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .data import digest, write_json
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cache_dir = Path(__file__).resolve().parents[4] / "data/task5b/hf_cache"
    model_id = ENCODERS[method]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision,
        num_labels=2, weights_only=True, cache_dir=cache_dir, attn_implementation="eager",
        use_safetensors=None if Path(model_id).is_dir() else False)
    resolved = getattr(model.config, "_commit_hash", None)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    write_json(Path(output) / "base_metadata.json", dict(model=model_id, requested_revision=revision,
        resolved_model_revision=resolved, initialization_seed=seed,
        artifact_sha256={p.name: digest(p) for p in Path(output).iterdir() if p.is_file()}))


def train_encoder(method, train, output, seed=0, epochs=3, batch_size=8, device="auto", revision=None, base_checkpoint=None):
    """Generic in-domain binary fine-tune; not Lan's unreleased EDA/PBA recipe.

    Fixed epochs and optimizer, no test-based checkpoint selection. Validation
    chooses the decision threshold afterwards. Always saves safetensors.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_id = ENCODERS[method]
    # Keep new encoder weights on the project drive rather than a potentially
    # full Windows system-drive cache. Existing NLI cache/results are untouched.
    cache_dir = Path(__file__).resolve().parents[4] / "data/task5b/hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else (
        "cpu" if device == "auto" else device)
    if base_checkpoint:
        import json
        meta = json.loads((Path(base_checkpoint) / "base_metadata.json").read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(base_checkpoint, local_files_only=True)
        if sys.platform == "win32":
            from .safeweights import load_windows_model
            config = AutoConfig.from_pretrained(base_checkpoint, local_files_only=True)
            model = load_windows_model(Path(base_checkpoint) / "model.safetensors", config, selected_device, torch.float32)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(base_checkpoint,
                local_files_only=True, use_safetensors=True, attn_implementation="eager")
        resolved_revision = meta["resolved_model_revision"]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision, num_labels=2, weights_only=True, cache_dir=cache_dir,
            attn_implementation="eager")
        resolved_revision = getattr(model.config, "_commit_hash", None)
    model.to(selected_device).train()
    use_amp = selected_device.startswith("cuda")
    if use_amp:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    def collate(batch):
        tokens = tokenizer([r["sentence"] for r in batch], padding=True, truncation=True,
                           max_length=256, return_tensors="pt")
        tokens["labels"] = torch.tensor([r["y"] for r in batch])
        return tokens
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"  {method}: {len(train)} training sentences, device={selected_device}, "
          f"revision={resolved_revision}, {len(loader)} batches/epoch", flush=True)
    for epoch in range(epochs):
        loss_sum = 0.
        for step, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with (torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()):
                loss = model(**{k: v.to(selected_device) for k, v in batch.items()}).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            if step % 100 == 0:
                print(f"  {method}: epoch {epoch + 1}/{epochs}, batch {step}/{len(loader)}, "
                      f"mean loss {loss_sum / step:.5f}", flush=True)
        print(f"  {method}: epoch {epoch + 1}/{epochs}, mean batch loss {loss_sum / len(loader):.5f}", flush=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return resolved_revision


class Encoder:
    def __init__(self, path, batch_size=8, device="auto"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
        self.torch, self.batch_size = torch, batch_size
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else (
            "cpu" if device == "auto" else device)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if sys.platform == "win32":
            from pathlib import Path
            from .safeweights import load_windows_model
            config = AutoConfig.from_pretrained(path, local_files_only=True)
            self.model = load_windows_model(Path(path) / "model.safetensors", config, self.device, torch.float32)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path, use_safetensors=True, local_files_only=True).to(self.device).eval()

    def scores(self, rows):
        values = []
        for start in range(0, len(rows), self.batch_size):
            tokens = self.tokenizer([r["sentence"] for r in rows[start:start + self.batch_size]],
                                    padding=True, truncation=True, max_length=256, return_tensors="pt")
            with self.torch.inference_mode():
                logits = self.model(**{k: v.to(self.device) for k, v in tokens.items()}).logits
            values.extend(logits.softmax(-1)[:, 1].cpu().tolist())
        return np.array(values)

    def close(self):
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
