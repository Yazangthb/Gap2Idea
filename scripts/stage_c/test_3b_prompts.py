"""Test a 3B LLM on a curated 10-sample set with 4 prompt variants.

The samples are real outputs from our SciBERT-FT funnel on the gold papers
(see docs/experiments/scibert_gold_notes.md). Each is hand-labeled GAP / NOT_GAP
based on the actual paper context.

Models four prompt variants:
  V1  current "validate" prompt (strict gap detector, our shipped Stage C)
  V2  reframed "is this junk?" (drop iff YES is junk)
  V3  V1 + explicit FP-type list (acknowledgment, citation, contribution, fragment)
  V4  V3 with chain-of-thought ("think step by step before answering")

Reports per-variant accuracy + per-sample correctness.

    python -u scripts/stage_c/test_3b_prompts.py --model Qwen/Qwen2.5-3B-Instruct
"""
from __future__ import annotations

import argparse

# (sentence, expected_label, FP_type or 'gap')
SAMPLES = [
    ("To address these limitations, we propose F DAN, a evaluation approach.",
     "NOT_GAP", "contribution_claim"),
    ("We would like to express our gratitude to Dr. Yitian Qian and Dr. Lianghai Xiao for kindly providing the code.",
     "NOT_GAP", "acknowledgment"),
    ("See Appendix L.3 for further details.",
     "NOT_GAP", "cross_ref"),
    ("Lemma 4.4 in the next step t + 1.",
     "NOT_GAP", "math_fragment"),
    ("We thank the anonymous reviewers for alerting us to a mistake in an earlier version of this paper.",
     "NOT_GAP", "acknowledgment"),
    ("Societal impact and Limitations: For some of our experi-",
     "NOT_GAP", "section_header_fragment"),
    ("We did not study risks that may or may not arise when our fine-tuned large language models are used for other application scenarios than ours.",
     "GAP", "limitation"),
    ("This work focuses on English-language datasets.",
     "GAP", "limitation_scope"),
    ("For future work, it would be interesting to understand the tightness of GEC by establishing a regret lower bound.",
     "GAP", "future_work"),
    ("We leave other tasks such as detection and segmentation for future work.",
     "GAP", "future_work"),
]

# ---- Prompt variants ----
SYS_V1 = (
    "You judge whether a sentence states a research GAP. Answer YES if it expresses, "
    "even IMPLICITLY, a LIMITATION of the authors' own work — including a scope "
    "restriction, an ASSUMPTION or DEPENDENCY it relies on, a weakness, or "
    "something left undone — OR a concrete FUTURE-WORK direction. Answer NO if it "
    "is clearly NOT a gap: a contribution/result, a method or equation description, "
    "an acknowledgment, a citation, an affiliation, or a caption. Reply one word: "
    "YES or NO."
)
SHOTS_V1 = [
    ("We leave multilingual evaluation for future work.", "YES"),
    ("This work focuses on English-language datasets.", "YES"),
    ("Our method assumes the availability of a knowledge base.", "YES"),
    ("We show via Theorem 3.3 that the estimators are consistent.", "NO"),
    ("We would like to express our gratitude to Dr. Qian.", "NO"),
    ("Our method achieves 95% accuracy on the benchmark.", "NO"),
]

SYS_V2 = (
    "You decide if a sentence is OBVIOUSLY NOT a research gap — an acknowledgment, "
    "citation, equation/formula, caption, affiliation, or pure numeric result. "
    "Answer NO_GAP if clearly one of those. If it could be a limitation or "
    "future-work, answer GAP. Reply one word."
)
SHOTS_V2 = [
    ("We would like to express our gratitude to Dr. Qian.", "NO_GAP"),
    ("This is shown in Equation (3): g(x) = f(x) + h(x).", "NO_GAP"),
    ("We leave multilingual evaluation for future work.", "GAP"),
    ("Evaluation so far is restricted to single charts.", "GAP"),
]

SYS_V3 = (
    "You are a precision filter for research-gap extraction. Reject (NO) a sentence "
    "if it is ONE OF: (a) an acknowledgment or thanks; (b) a citation, cross-reference, "
    "or 'See Appendix/Figure/Section' line; (c) a CONTRIBUTION claim — sentences like "
    "'we propose', 'to address these limitations we...', 'our method achieves'; (d) a "
    "math equation, formula reference, or lemma statement; (e) a scramble or fragment "
    "(broken hyphenation, mid-clause start, or section-header fragment). Accept (YES) "
    "if it is the authors' own LIMITATION, ASSUMPTION, SCOPE restriction, or "
    "FUTURE-WORK direction. Reply one word: YES or NO."
)
SHOTS_V3 = [
    ("We leave multilingual evaluation for future work.", "YES"),
    ("A limitation of our approach is that it assumes English input.", "YES"),
    ("This work focuses on English-language datasets.", "YES"),
    ("Future work will explore co-evolutionary settings.", "YES"),
    ("To address these limitations, we propose a new framework.", "NO"),
    ("We would like to express our gratitude to Dr. Qian.", "NO"),
    ("See Appendix L.3 for further details.", "NO"),
    ("Lemma 4.4 in the next step t + 1.", "NO"),
    ("We thank the anonymous reviewers.", "NO"),
    ("Our method achieves 95% accuracy.", "NO"),
]

SYS_V4 = SYS_V3 + (
    " First, in 1 short sentence, identify which category the sentence falls into. "
    "Then answer YES or NO on a new line. Final answer format:\n"
    "Category: <one phrase>\nAnswer: YES|NO"
)
SHOTS_V4 = [
    ("We leave multilingual evaluation for future work.",
     "Category: future-work direction\nAnswer: YES"),
    ("To address these limitations, we propose a new framework.",
     "Category: contribution claim\nAnswer: NO"),
    ("See Appendix L.3 for further details.",
     "Category: cross-reference\nAnswer: NO"),
    ("We thank the anonymous reviewers.",
     "Category: acknowledgment\nAnswer: NO"),
    ("This work focuses on English-language datasets.",
     "Category: scope limitation\nAnswer: YES"),
]


def build_messages(system, shots, sentence):
    msgs = [{"role": "system", "content": system}]
    for s, a in shots:
        msgs += [{"role": "user", "content": "Sentence: " + s},
                 {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": "Sentence: " + sentence})
    return msgs


def parse(text, mode):
    t = text.strip().upper()
    if mode == "yn":
        if "YES" in t.split() or t.startswith("YES"): return "GAP"
        if "NO" in t.split() or t.startswith("NO"): return "NOT_GAP"
        return "?"
    if mode == "gap_nogap":
        if t.startswith("GAP"): return "GAP"
        if "NO_GAP" in t or t.startswith("NO"): return "NOT_GAP"
        return "?"
    if mode == "cot":
        # find "Answer: ..." line
        for line in text.splitlines():
            if "ANSWER" in line.upper():
                return parse(line.split(":", 1)[-1], "yn")
        return parse(text, "yn")
    return "?"


VARIANTS = [
    ("V1 current strict", SYS_V1, SHOTS_V1, "yn", 3),
    ("V2 junk-detector", SYS_V2, SHOTS_V2, "gap_nogap", 3),
    ("V3 explicit FP types", SYS_V3, SHOTS_V3, "yn", 3),
    ("V4 V3 + CoT", SYS_V4, SHOTS_V4, "cot", 30),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {dev} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if dev == "cuda" else torch.float32,
        device_map="auto" if dev == "cuda" else None)
    model.eval()

    def call(msgs, max_new):
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)

    print()
    results = {}
    for name, sysp, shots, mode, max_new in VARIANTS:
        print(f"=== {name} ===", flush=True)
        correct = 0
        per_sample = []
        for sent, expected, ftype in SAMPLES:
            ans = call(build_messages(sysp, shots, sent), max_new)
            pred = parse(ans, mode)
            ok = pred == expected
            correct += int(ok)
            per_sample.append((sent, expected, pred, ok, ftype, ans.strip()[:60]))
        results[name] = (correct, per_sample)
        print(f"  accuracy: {correct}/{len(SAMPLES)} = {correct/len(SAMPLES):.2f}", flush=True)
        for sent, exp, pred, ok, ftype, raw in per_sample:
            flag = "OK" if ok else "X "
            print(f"  [{flag}] exp={exp:<7} got={pred:<7} ({ftype:<25}) {sent[:55]}", flush=True)
        print()

    # summary
    print("=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print(f"{'variant':<25} {'accuracy':>10} {'FP-caught':>12} {'gap-kept':>10}", flush=True)
    for name, sysp, shots, mode, max_new in VARIANTS:
        correct, per = results[name]
        fp = [(p, e) for _, e, p, _, _, _ in per if e == "NOT_GAP"]
        gp = [(p, e) for _, e, p, _, _, _ in per if e == "GAP"]
        fp_caught = sum(1 for p, e in fp if p == "NOT_GAP")
        gp_kept = sum(1 for p, e in gp if p == "GAP")
        print(f"{name:<25} {correct}/{len(SAMPLES):>2}={correct/len(SAMPLES):>4.2f}  "
              f"{fp_caught}/{len(fp):>2}        {gp_kept}/{len(gp):>2}", flush=True)


if __name__ == "__main__":
    main()
