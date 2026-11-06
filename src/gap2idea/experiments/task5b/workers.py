"""Isolate each GPU operation, releasing CUDA state and Windows weight mappings.

Fresh workers bound GPU memory when the suite switches between NLI and trainable
encoders. The separate safeweights loader avoids Windows memory-mapping issues;
process isolation by itself did not resolve the observed native loading crash.
"""
from pathlib import Path
import json
import numpy as np

from .data import read_jsonl, write_json
from .models import ModernBERT, Encoder, train_encoder, prepare_encoder_base


def execute(args):
    spec = json.loads(Path(args.request).read_text(encoding="utf-8"))
    if spec["operation"] == "prepare_encoder_base":
        prepare_encoder_base(spec["method"], spec["checkpoint"], spec["seed"], spec["revision"])
        return
    rows = read_jsonl(spec["rows"])
    if spec["operation"] == "train_encoder":
        revision = train_encoder(spec["method"], rows, spec["checkpoint"], spec["seed"],
                                 spec["epochs"], spec["batch_size"], spec["device"], spec["revision"], spec.get("base_checkpoint"))
        write_json(spec["output"], dict(resolved_model_revision=revision))
        return
    model = ModernBERT(spec["batch_size"], spec["device"]) if spec["method"].startswith("modernbert") else Encoder(
        spec["checkpoint"], spec["batch_size"], spec["device"])
    try:
        scores = model.scores(rows, spec["hypotheses"]) if spec["method"].startswith("modernbert") else model.scores(rows)
        if len(scores) != len(rows) or not np.isfinite(scores).all():
            raise ValueError("Neural inference produced invalid scores")
        np.save(spec["output"], scores, allow_pickle=False)
        write_json(spec["output"] + ".metadata.json", dict(actual_device=model.device, n_sentences=len(rows)))
    finally:
        model.close()
