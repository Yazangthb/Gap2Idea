# -*- coding: utf-8 -*-
"""Score every (premise, hypothesis) pair with an NLI cross-encoder and save raw class probabilities.

Runs on the GPU box and needs only the sentence file, which holds no labels. Evaluation happens
elsewhere, so nothing here can be tuned against the gold.

  python nli_score_remote.py sentences.json out.npz
"""
import json
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "tasksource/ModernBERT-large-nli"
REVISION = "ca476cb923a8637073d4ceb0f19f7fc236e260d4"
MAX_LENGTH = 256
BATCH = 192

HYPOTHESES = {
    # Original single hypothesis from the Oct 27 run.
    "orig": "This sentence points out a limitation, weakness, shortcoming, open research problem, or future-work direction of the research described.",
    # Type-specific, using the earlier zero-shot labels.
    "lim_a": "This sentence describes a limitation, weakness, or shortcoming of the research.",
    "fw_a": "This sentence describes a future-work direction or something left for future research.",
    "gap_a": "This sentence describes an open research problem or unresolved gap in knowledge.",
    # Type-specific, closer to the gold guideline's definitions.
    "lim_b": "The authors acknowledge a weakness, caveat, bias, or constraint of their own study.",
    "fw_b": "The authors propose a direction for future research.",
    "gap_b": "This sentence states an unsolved problem, missing capability, or open challenge in the field.",
    # Short forms.
    "lim_c": "This is a limitation.",
    "fw_c": "This is future work.",
    "gap_c": "This is a research gap.",
    # Competing non-gap hypotheses.
    "neg_result": "This sentence reports a result or finding.",
    "neg_method": "This sentence describes a method, procedure, or experimental setup.",
    "neg_background": "This sentence gives background information or describes related work.",
    "neg_claim": "This sentence claims a positive contribution or achievement.",
}


def main():
    src, out = sys.argv[1], sys.argv[2]
    rows = json.load(open(src, encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION).eval().cuda()
    id2label = {i: l.lower() for i, l in model.config.id2label.items()}
    print("labels:", id2label, "| rows:", len(rows), flush=True)
    order = [next(i for i, l in id2label.items() if l.startswith(p)) for p in ("entail", "neutral", "contra")]

    premises = {"sent": [r["sentence"] for r in rows], "ctx": [r.get("context") for r in rows]}
    hypotheses = HYPOTHESES
    if len(sys.argv) > 3:  # optional: premise names and hypothesis names, each comma-separated
        premises = {k: premises[k] for k in sys.argv[3].split(",")}
        hypotheses = {k: HYPOTHESES[k] for k in sys.argv[4].split(",")}
    saved = {"hypotheses": np.array(list(hypotheses)), "class_order": np.array(["entail", "neutral", "contra"])}
    t0 = time.time()
    for pname, plist in premises.items():
        by_length = np.argsort([len(p) for p in plist])
        for hname, hyp in hypotheses.items():
            probs = np.zeros((len(plist), 3), dtype=np.float32)
            for start in range(0, len(plist), BATCH):
                idx = by_length[start:start + BATCH]
                enc = tok([plist[i] for i in idx], [hyp] * len(idx), return_tensors="pt", padding=True,
                          truncation="only_first", max_length=MAX_LENGTH).to("cuda")
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**enc).logits.float()
                probs[idx] = torch.softmax(logits, -1)[:, order].cpu().numpy()
            saved[f"{pname}__{hname}"] = probs
            print(f"{pname}/{hname} done [{time.time() - t0:.0f}s]", flush=True)
    np.savez_compressed(out, **saved)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
