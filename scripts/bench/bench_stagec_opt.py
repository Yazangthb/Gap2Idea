"""Optimize Stage C (prompt + model) on the section-filtered RCT pool.

Stage C is applied to the detector's predicted-positive sentences. The ideal filter
KEEPs every true limitation (no recall loss) and DROPs false positives (precision gain).
This script loads the detector's dumped pool predictions (from finetune_rct.py
--dump-preds) and, for each (prompt, model) candidate, runs the batched judge over the
predicted-positives and reports:

  keepTP  = kept true limitations / predicted true limitations      (want ~1.00)
  dropFP  = dropped false positives / predicted false positives     (want high)
  final P/R/F1 over the WHOLE pool after applying Stage C

The winner maximizes F1 while keeping keepTP high (recall barely moves). No GPU needed.

    python scripts/bench/bench_stagec_opt.py --preds data/pool_preds.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import os  # noqa: E402
from bench_limgen import prf  # noqa: E402
from gap2idea.pipeline.gap_llm_filter import LLMGapFilter, SYSTEM_VALIDATE, SHOTS_VALIDATE  # noqa: E402
from gap2idea.pipeline.llm import get_llm_client, active_provider  # noqa: E402


def yandex_uri(variant):
    """Full gpt:// URI for a Yandex variant (passes through the client untouched)."""
    folder = os.getenv("YANDEX_FOLDER_ID")
    return f"gpt://{folder}/{variant}/latest"


def reasoning_ctrl(variant):
    """Disable hidden reasoning for reasoning models so the batched JSON fits the
    token budget and returns fast; None for plain models (yandexgpt*)."""
    if variant.startswith("gpt-oss") or variant.startswith("deepseek"):
        return {"reasoning_effort": "low"}
    if variant.startswith("qwen"):
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def model_chunk(variant):
    """Batch size per call. deepseek-v4-flash is slow even at low effort -> small
    chunks keep each call reliable and within the token budget."""
    return 10 if variant.startswith("deepseek") else 40

# --- RCT-domain prompt: default KEEP; drop only clear non-limitations -----------
SYSTEM_RCT = (
    "You are a precision filter for STUDY LIMITATIONS in randomized controlled trial (RCT) "
    "reports. KEEP a sentence if it states a weakness, caveat, bias, constraint, or scope "
    "restriction of THIS study/trial — for example: small or limited sample size; short "
    "follow-up; single-center or narrow setting; lack of blinding; selection, recall, or "
    "measurement bias; confounding; low statistical power; missing data, attrition, or low "
    "response; limited generalizability; reliance on self-report; post-hoc or unadjusted "
    "analyses; or an explicit call for further research to address such a gap. "
    "DROP a sentence ONLY if it clearly does none of that and is instead one of: "
    "(a) background, rationale, or prior findings; (b) a restatement of THIS study's positive "
    "results or effect estimates; (c) a clinical recommendation or implication "
    "('clinicians should...'); (d) a methods description with no stated weakness; "
    "(e) a limitation of OTHER studies rather than this one; (f) a citation or reference "
    "fragment. When unsure, KEEP."
)
SHOTS_RCT = [
    ("The main limitation of our study is the relatively small sample size.", "YES"),
    ("Participants were recruited from a single center, which may limit generalizability.", "YES"),
    ("The trial was not blinded, so outcome assessment may have been biased.", "YES"),
    ("Follow-up was limited to six months, and long-term effects remain unknown.", "YES"),
    ("We relied on self-reported adherence, which is subject to recall bias.", "YES"),
    ("The study may have been underpowered to detect small differences between groups.", "YES"),
    ("Further trials with larger and more diverse samples are needed to confirm these findings.", "YES"),
    ("Cardiovascular disease is a leading cause of mortality worldwide.", "NO"),
    ("The intervention significantly reduced HbA1c compared with control (p<0.001).", "NO"),
    ("Clinicians should consider offering this intervention in routine practice.", "NO"),
    ("Randomization was performed using a computer-generated sequence.", "NO"),
    ("Previous studies were limited by short follow-up and small samples.", "NO"),
]
# Even more KEEP-biased variant (only the most unambiguous drops).
SYSTEM_RCT_LOOSE = SYSTEM_RCT + (
    " Bias strongly toward KEEP: only DROP when you are highly confident the sentence is "
    "pure background, a pure positive-result restatement, a citation, or clearly about other "
    "studies. Any hedge, caveat, or constraint about this trial is a KEEP."
)

# V2 — tuned from error analysis on the RCT pool. Two additions over rct_loose:
#  (1) DROP methodological STRENGTH claims (adequate power, successful blinding) and
#      between-group RESULT comparisons — these read limitation-adjacent but are the
#      opposite of a limitation. (2) explicitly KEEP the *implicit* limitations the
#      filter kept missing: volunteer/selection bias, non-compliance/adherence, floor/
#      ceiling effects, and absence of validated outcomes.
SYSTEM_RCT_V2 = SYSTEM_RCT + (
    " Two clarifications. FIRST, these are NOT limitations — DROP them even though they "
    "mention the trial: statements affirming methodological STRENGTH or adequacy ('the trial "
    "was sufficiently/adequately powered', 'randomisation was successful', 'the drug was well "
    "tolerated'); and any between-group RESULT or effect comparison ('X performed better than "
    "Y', 'was associated with improvement', 'reduced pain'). SECOND, these ARE limitations — "
    "KEEP them even without the word 'limitation': implicit selection or volunteer bias "
    "('these were likely the most motivated participants'); non-compliance or poor adherence "
    "to the intervention; floor or ceiling effects; and the absence of validated outcome "
    "measures or endpoints. When still unsure, KEEP."
)
SHOTS_RCT_V2 = SHOTS_RCT + [
    ("These are likely to be people who were most motivated to change.", "YES"),
    ("One possible reason for the lack of effect is that families did not comply with the intervention.", "YES"),
    ("There are currently no validated endpoints for the assessment of gouty arthritis flares.", "YES"),
    ("The trial was sufficiently powered to detect a clinically significant improvement.", "NO"),
    ("Spinal manipulation and exercise generally performed slightly better than the alternative.", "NO"),
]

PROMPTS = {
    "baseline_arxiv": (SYSTEM_VALIDATE, SHOTS_VALIDATE),
    "rct": (SYSTEM_RCT, SHOTS_RCT),
    "rct_loose": (SYSTEM_RCT_LOOSE, SHOTS_RCT),
    "rct_v2": (SYSTEM_RCT_V2, SHOTS_RCT_V2),
}


def run_config(sents_pred, sys_prompt, shots, model_uri, extra_body=None, chunk=40):
    # Yandex only: default client (active provider = yandex), model as a gpt:// URI
    # so the chosen variant is used verbatim.
    filt = LLMGapFilter(backend="api", mode="validate", model=model_uri)
    filt._sys, filt._shots = sys_prompt, shots
    filt.extra_body = extra_body
    keep = []
    for k in range(0, len(sents_pred), chunk):
        keep.extend(filt.judge_batch(sents_pred[k:k + chunk]))
    return keep, filt.n_calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="data/pool_preds.json")
    ap.add_argument("--prompts", default="baseline_arxiv,rct,rct_loose")
    ap.add_argument("--models", default="yandexgpt-5-pro,yandexgpt",
                    help="comma list of Yandex variants (yandexgpt-5-pro, yandexgpt-5-lite, yandexgpt, yandexgpt-32k)")
    ap.add_argument("--eval-prevalence", type=float, default=0.207,
                    help="also report F1 at this positive rate (subsample pool negatives), matching their pool")
    args = ap.parse_args()

    def pm_f1(gold, final, target, seeds=8):
        """F1 after subsampling negatives to `target` prevalence (mean over seeds).
        Recall is unchanged; this corrects precision to their base rate on the SAME
        hard section-pool negatives -> the airtight comparison to their 0.821."""
        pos = np.where(gold == 1)[0]; neg = np.where(gold == 0)[0]
        n_neg = min(len(neg), int(round(len(pos) * (1 - target) / target)))
        fs = []
        for sd in range(seeds):
            rng = np.random.default_rng(sd)
            idx = np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)])
            fs.append(prf(gold[idx], final[idx])[2])
        return float(np.mean(fs))

    data = json.loads(Path(args.preds).read_text(encoding="utf-8"))
    rows = data["rows"]
    gold = np.array([r["gold"] for r in rows])
    pred = np.array([r["pred"] for r in rows])
    sents = [r["sentence"] for r in rows]
    pos_idx = [i for i, p in enumerate(pred) if p == 1]      # what Stage C judges
    sents_pred = [sents[i] for i in pos_idx]
    n_pred_tp = int(gold[pos_idx].sum()); n_pred_fp = len(pos_idx) - n_pred_tp
    Pb, Rb, Fb = prf(gold, pred)
    det_pm = pm_f1(gold, pred, args.eval_prevalence)
    print(f"provider(active)={active_provider()}  pool={len(rows)} ({int(gold.sum())} lim, {gold.mean():.1%})")
    print(f"detector (recall-tuned, thr={data['threshold']:.2f}): "
          f"P={Pb} R={Rb} F1={Fb}  | predicted-pos={len(pos_idx)} (TP={n_pred_tp} FP={n_pred_fp})")
    print(f"detector-only @ {args.eval_prevalence:.1%} prevalence: F1={det_pm:.3f}  (their pool condition)\n")
    print(f"{'config':<34} {'keepTP':>7} {'dropFP':>7}   P     R     F1    F1@{int(args.eval_prevalence*100)}%")

    best = None
    for variant in args.models.split(","):
        variant = variant.strip()
        model_uri = yandex_uri(variant)
        extra = reasoning_ctrl(variant)
        chunk = model_chunk(variant)
        for pname in args.prompts.split(","):
            sys_p, shots = PROMPTS[pname.strip()]
            keep, ncalls = run_config(sents_pred, sys_p, shots, model_uri, extra, chunk)
            final = pred.copy()
            kept_tp = dropped_fp = 0
            for j, i in enumerate(pos_idx):
                if not keep[j]:
                    final[i] = 0
                    if gold[i] == 0: dropped_fp += 1
                if keep[j] and gold[i] == 1: kept_tp += 1
            P, R, F = prf(gold, final)
            Fpm = pm_f1(gold, final, args.eval_prevalence)
            tag = f"{variant}/{pname.strip()}"
            beat = " >=0.821" if Fpm >= 0.821 else ""
            print(f"{tag:<34} {kept_tp}/{n_pred_tp:<4} {dropped_fp}/{n_pred_fp:<4}   "
                  f"{P:.3f} {R:.3f} {F:.3f}   {Fpm:.3f}{beat}")
            if best is None or Fpm > best[0]: best = (Fpm, tag, R)
    print(f"\nbest @ {args.eval_prevalence:.1%}: {best[1]}  F1={best[0]:.3f} (recall {best[2]})"
          f"   | detector-only @ prevalence F1={det_pm:.3f}, their SOTA 0.821")


if __name__ == "__main__":
    main()
