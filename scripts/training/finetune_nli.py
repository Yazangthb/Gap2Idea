# -*- coding: utf-8 -*-
"""Fine-tune ModernBERT-large-nli on gold sentence labels as NLI pairs, then score sentence files.

Runs on the GPU box with the files from export_nli_ft_data.py.

  python finetune_nli.py DATA_DIR OUT_DIR [MODEL_DIR]

Pairs: each training sentence with each Stage B hypothesis ("This is a limitation." / "This is future work." /
"This is a research gap."). Label entailment when the gold type matches the hypothesis, neutral when the
sentence is a gap of another type, contradiction when it is not a gap.

Settings, fixed before training: lr 1e-5, batch 32, 2 epochs, linear decay with 6% warmup, weight decay 0.01,
fp16 autocast, max length 256, seed 0. After each epoch the model scores gold validation; the epoch with the
highest validation precision at the Stage B rule (highest tau with recall >= 90%, counting Stage A losses) is kept.
The zero-shot model is scored the same way first, as epoch 0. Test files are scored only with the kept model.
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

HYPOTHESES = {"lim_c": "This is a limitation.", "fw_c": "This is future work.", "gap_c": "This is a research gap."}
TYPE_OF = {"lim_c": "limitation", "fw_c": "future_work", "gap_c": "research_gap"}
LR, BATCH, EPOCHS, WARMUP, WD, MAX_LEN, SEED = 1e-5, 32, 2, 0.06, 0.01, 256, 0
SCORE_BATCH = 192
RECALL_RULE = 0.90


def jl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def class_ids(model):
    id2label = {i: l.lower() for i, l in model.config.id2label.items()}
    return [next(i for i, l in id2label.items() if l.startswith(p)) for p in ("entail", "neutral", "contra")]


@torch.inference_mode()
def score(model, tok, sentences, order):
    """{hyp: (n, 3) probs in entail, neutral, contra order}, the layout nli_score_remote.py saves."""
    model.eval()
    by_len = np.argsort([len(s) for s in sentences])
    out = {}
    for h, hyp in HYPOTHESES.items():
        probs = np.zeros((len(sentences), 3), dtype=np.float32)
        for s in range(0, len(sentences), SCORE_BATCH):
            idx = by_len[s:s + SCORE_BATCH]
            enc = tok([sentences[i] for i in idx], [hyp] * len(idx), return_tensors="pt", padding=True,
                      truncation="only_first", max_length=MAX_LEN).to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**enc).logits.float()
            probs[idx] = torch.softmax(logits, -1)[:, order].cpu().numpy()
        out[h] = probs
    return out


def val_metric(val, probs):
    evc = np.stack([p[:, 0] / (p[:, 0] + p[:, 2] + 1e-9) for p in probs.values()], 1).max(1)
    y = np.array([r["label"] != "none" for r in val])
    passed = np.array([not r["junk"] for r in val])
    ranked = np.sort(evc[passed & y])[::-1]
    need = int(math.ceil(RECALL_RULE * y.sum()))
    tau = float(ranked[need - 1])
    kept = passed & (evc >= tau)
    tp = int((kept & y).sum())
    return {"tau": round(tau, 4), "candidates": int(kept.sum()), "precision": round(tp / kept.sum(), 4),
            "recall": round(tp / y.sum(), 4)}


def main():
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    model_dir = sys.argv[3] if len(sys.argv) > 3 else str(Path.home() / "models/modernbert-large-nli")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda()
    order = class_ids(model)
    ent, neu, con = order

    train, val = jl(data / "train.jsonl"), jl(data / "val.jsonl")
    pairs = []
    for r in train:
        for h, hyp in HYPOTHESES.items():
            lab = ent if r["label"] == TYPE_OF[h] else (con if r["label"] == "none" else neu)
            pairs.append((r["sentence"], hyp, lab))
    print(f"train sentences {len(train)}, pairs {len(pairs)}, entail {sum(p[2] == ent for p in pairs)}, "
          f"neutral {sum(p[2] == neu for p in pairs)}, contra {sum(p[2] == con for p in pairs)}", flush=True)

    log = {"settings": {"lr": LR, "batch": BATCH, "epochs": EPOCHS, "warmup": WARMUP, "weight_decay": WD,
                        "max_len": MAX_LEN, "seed": SEED, "model": model_dir}, "epochs": []}
    vs = [r["sentence"] for r in val]
    m0 = val_metric(val, score(model, tok, vs, order))
    log["epochs"].append({"epoch": 0, **m0})
    print("epoch 0 (zero-shot) val", m0, flush=True)
    best = (m0["precision"], 0)
    torch.save(model.state_dict(), out / "best.pt")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * math.ceil(len(pairs) / BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        random.shuffle(pairs)
        total = 0.0
        for k in range(0, len(pairs), BATCH):
            b = pairs[k:k + BATCH]
            enc = tok([p[0] for p in b], [p[1] for p in b], return_tensors="pt", padding=True,
                      truncation="only_first", max_length=MAX_LEN).to("cuda")
            labels = torch.tensor([p[2] for p in b], device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            total += loss.item()
            step = k // BATCH + 1
            if step % 500 == 0:
                print(f"epoch {epoch} step {step}/{math.ceil(len(pairs) / BATCH)} loss {total / step:.4f} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
        m = val_metric(val, score(model, tok, vs, order))
        log["epochs"].append({"epoch": epoch, "train_loss": round(total / step, 4), **m})
        print(f"epoch {epoch} val", m, flush=True)
        if m["precision"] > best[0]:
            best = (m["precision"], epoch)
            torch.save(model.state_dict(), out / "best.pt")
        (out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")

    log["kept_epoch"] = best[1]
    (out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print("kept epoch", best[1], flush=True)
    model.load_state_dict(torch.load(out / "best.pt"))
    for f in sorted(data.glob("score_*.json")):
        sents = [r["sentence"] for r in json.loads(f.read_text(encoding="utf-8"))]
        probs = score(model, tok, sents, order)
        np.savez_compressed(out / f"nli_ft_{f.stem[len('score_'):]}_scores.npz",
                            hypotheses=np.array(list(HYPOTHESES)), class_order=np.array(["entail", "neutral", "contra"]),
                            **{f"sent__{h}": p for h, p in probs.items()})
        print("scored", f.name, len(sents), flush=True)
    (out / "DONE").write_text("ok")


if __name__ == "__main__":
    main()
