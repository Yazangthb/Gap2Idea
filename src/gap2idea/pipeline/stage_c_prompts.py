# -*- coding: utf-8 -*-
"""Single source of truth for the experiment Stage C prompts.

Experiment scripts import these instead of inlining their own copy, so a prompt tweak here
propagates to every experiment on the next rerun. (The *shipped* funnel prompts live in
`gap_llm_filter.py` as SYSTEM_VALIDATE / SYSTEM_VALIDATE_RCT / SYSTEM_VALIDATE_FWS; these are
the research-experiment variants.)

Each prompt is a (system_str, shots_list) pair; shots are (sentence, "YES"/"NO").
Use with: f = LLMGapFilter(..., mode="validate_rct"); f._sys, f._shots = SAL_SYS, SAL_SHOTS
"""

# --- SAL keep-biased (cause of a self-acknowledged limitation; recovers cause from context) ---
SAL_SYS = (
    "You judge whether the «marked» TARGET sentence is a SELF-ACKNOWLEDGED LIMITATION of the "
    "AUTHORS' OWN study, using the SAL annotation criteria. A limitation is a weakness/caveat/bias/"
    "constraint the authors admit about THIS trial, fitting a known category: study design; loose or "
    "narrow population / convenience sampling; single-center or narrow setting; intervention "
    "composite/non-standardized; control (no placebo / active placebo / care-as-usual); outcome measure "
    "excluded/imprecise/unvalidated; missing data or drop-out; small sample / underpowered; randomization "
    "unbalanced/poor/open; blinding (patient or team not blinded); short duration or follow-up; "
    "multiple/post-hoc testing or unadjusted confounding; funding; limited generalizability.\n"
    "DECISION RULE (keep-biased):\n"
    "- keep=true if the TARGET states such a limitation, OR states a consequence ('could introduce bias', "
    "'cannot be assessed', 'results may be confounded', 'risk of type 1 error') whose underlying CAUSE "
    "(no blinding, small sample, single center, multiple testing, short follow-up...) is evident in the "
    "TARGET or its SURROUNDING context. When in doubt and a real weakness of this study is present, keep=true.\n"
    "- keep=false ONLY for CLEAR non-limitations: restated positive results; clinical recommendations; "
    "critique of prior/other work; plain methods description; background; or a bare header like "
    "'This study has several limitations.' with no specific weakness."
)
SAL_SHOTS = [
    ("the lack of a no treatment arm in this study", "YES"),
    ("our study included only a relatively small sample of participants", "YES"),
    ("The absence of masking is most relevant for the assessment of events.", "YES"),
    ("the crossover design and the relatively short treatment periods are limitations", "YES"),
    ("the trial was conducted at a single center", "YES"),
    ("Because outcomes were exploratory, the risk of type 1 error was considerable.", "YES"),
    ("This could have introduced a bias in the results.", "YES"),
    ("This study has several limitations.", "NO"),
    ("The intervention significantly reduced symptoms at 12 weeks.", "NO"),
    ("Clinicians should consider these findings when counseling patients.", "NO"),
    ("Existing methods cannot capture long-range dependencies.", "NO"),
    ("Participants were randomized 1:1 using a computer-generated sequence.", "NO"),
]

# --- GAP-matched (limitations/research-gaps in prior/existing work; e.g. Sci-Sentence) ---
GAP_SYS = (
    "You are a precision filter for RESEARCH-GAP and LIMITATION statements in academic text, "
    "including related-work sections. KEEP the «marked» sentence if it identifies a limitation, weakness, "
    "shortcoming, unsolved problem, open challenge, or missing capability — in prior/existing work, a method, "
    "or the field (it need NOT be the authors' own study). DROP it if it merely describes what a work does, "
    "gives background, states results/findings, or makes a neutral or positive general statement with no gap "
    "or weakness. When a sentence clearly names a shortcoming or an unaddressed problem, KEEP."
)
GAP_SHOTS = [
    ("Existing methods cannot capture long-range dependencies.", "YES"),
    ("However, these approaches do not scale to large graphs.", "YES"),
    ("No prior work has addressed the multilingual setting.", "YES"),
    ("This remains an open problem in the field.", "YES"),
    ("Current benchmarks lack annotations for rare classes.", "YES"),
    ("archjava is an architecture description language on top of java.", "NO"),
    ("The model achieves 92% accuracy on the benchmark.", "NO"),
    ("Smith et al. proposed a transformer-based model for this task.", "NO"),
    ("These tools are widely used in industry.", "NO"),
]

PROMPTS = {"sal_keepbiased": (SAL_SYS, SAL_SHOTS), "gap_matched": (GAP_SYS, GAP_SHOTS)}
