"""Stage C — LLM precision filter over the funnel's survivors.

The funnel (Stage A slice + Stage B classify) is recall-first and cheap, but
leaks false positives on scrambled / math papers (acknowledgments, formulas,
citations, contribution claims — see docs/experiments/funnel_demo_output.md).
Stage C asks an LLM, per surviving candidate, "is this really a gap?" and drops
the No's. Because Stage A+B already cut ~98% of sentences, the LLM only sees
~6 sentences/paper — so LLM-grade precision costs ~cents/1000s of papers, not the
~$4000/1M of a per-paper LLM.

Backends (swappable — `backend=`):
  - "local"  a small instruct model via transformers (default, no API/credits).
             Generation is forward-only, so it is stable on CPU (unlike training).
  - "api"    any OpenAI-compatible client (OpenRouter/gpt-4o) for max quality.

The few-shot prompt is tuned to REJECT the observed false-positive classes while
keeping genuine own-work limitations and concrete future-work directions.
"""
from __future__ import annotations

from gap2idea.utils import get_logger

log = get_logger(__name__)

DEFAULT_LOCAL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_API = "openai/gpt-4o-mini"

# Batched-judge output schema: one verdict per input sentence. Sent with
# response_format=json_schema strict (YandexGPT / OpenAI honour it), so we get a
# clean array back instead of parsing prose. See LLMGapFilter.judge_batch.
BATCH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["results"],
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "keep"],
                "properties": {
                    "id": {"type": "integer"},
                    "keep": {"type": "boolean"},
                },
            },
        }
    },
}
#: sentences per batched LLM call (keeps prompts well within context limits;
#: one call handles a normal paper's candidates, chunking only huge inputs).
BATCH_CHUNK = 40

# Two prompt modes (set via mode=):
#  "validate" — strict "is this a gap?"; keep iff YES. Catches method-description
#               / contribution false positives too. Needs a capable model (>=1.5B).
#  "junk"     — conservative "is this obvious junk?"; drop iff YES. Recall-safe but
#               only removes acknowledgments/formulas/citations (use with tiny models).
SYSTEM_VALIDATE = (
    "You are a precision filter for research-gap extraction. A gap is a limitation, "
    "assumption, scope restriction, or future-work direction of THE AUTHORS' OWN work. "
    "Reject (NO) a sentence if it is ONE OF: (a) an acknowledgment or thanks; (b) a "
    "citation, cross-reference, or 'See Appendix/Figure/Section' line; (c) a CONTRIBUTION "
    "or SELF-PROMOTIONAL claim — 'we propose', 'to address these limitations we...', 'our "
    "method achieves', or a vague boast with no concrete gap ('we believe our work/approach "
    "is promising', 'opens a wide array of topics', 'is a promising alternative'); (d) a "
    "math equation, formula reference, or lemma statement; (e) a "
    "scramble or fragment (broken hyphenation, mid-clause start, or section-header "
    "fragment); (f) a limitation, weakness, or gap of PRIOR or OTHER work rather than the "
    "authors' own — motivating criticism such as 'existing methods cannot...', "
    "'traditional approaches fail to...', 'previous work is limited to...', 'X et al. do "
    "not handle...', 'current models struggle with...'. Accept (YES) ONLY if the sentence "
    "states the AUTHORS' OWN limitation, assumption, scope restriction, or future-work "
    "direction. Reply one word: YES or NO."
)
SHOTS_VALIDATE = [
    ("We leave multilingual evaluation for future work.", "YES"),
    ("A limitation of our approach is that it assumes English input.", "YES"),
    ("This work focuses on English-language datasets.", "YES"),
    ("Future work will explore co-evolutionary settings.", "YES"),
    ("Another limitation of our theory is that it only applies to i.i.d. sequences.", "YES"),
    ("To address these limitations, we propose a new framework.", "NO"),
    ("We would like to express our gratitude to Dr. Qian.", "NO"),
    ("See Appendix L.3 for further details.", "NO"),
    ("Lemma 4.4 in the next step t + 1.", "NO"),
    ("We thank the anonymous reviewers.", "NO"),
    ("Our method achieves 95% accuracy.", "NO"),
    # (f) limitations of PRIOR / OTHER work — motivation, not the authors' own gap
    ("Existing methods cannot easily capture long-range dependencies.", "NO"),
    ("Traditional approaches fail to scale beyond a few thousand nodes.", "NO"),
    ("SIFT-like methods cannot produce meaningful matches across spectral contents.", "NO"),
    # (c) vague self-promotion — the authors' own sentence, but no concrete gap
    ("We believe our approach is a promising alternative to current methods.", "NO"),
    ("We believe our work introduces a wide array of topics for future research.", "NO"),
]
SYSTEM_JUNK = (
    "You decide if a sentence is OBVIOUSLY NOT a research gap — an acknowledgment, "
    "citation, equation/formula, caption, affiliation, or pure numeric result. "
    "Answer YES only if clearly one of those; if it could be a limitation or "
    "future-work, or you are unsure, answer NO. Reply with one word: YES or NO."
)
SHOTS_JUNK = [
    ("We would like to express our gratitude to Dr. Qian.", "YES"),
    ("This is shown in Equation (3): g(x) = f(x) + h(x).", "YES"),
    ("We leave multilingual evaluation for future work.", "NO"),
    ("Evaluation so far is restricted to single charts.", "NO"),
]

# RCT / biomedical mode — a domain-tuned, KEEP-biased validate prompt. Tuned and
# validated on the Lan et al. (2024) RCT/SAL limitation-detection pool: it matches
# their PubMedBERT precision (~0.75) while dropping only ~2/180 true limitations.
# Use mode="validate_rct" when filtering biomedical / clinical-trial gaps, where the
# arXiv/ACL-flavoured default over-rejects study limitations (small sample, short
# follow-up, single-centre, no blinding, attrition, generalisability).
SYSTEM_VALIDATE_RCT = (
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
    "fragment. When unsure, KEEP. "
    "Bias strongly toward KEEP: only DROP when highly confident the sentence is pure "
    "background, a pure positive-result restatement, a citation, or clearly about other "
    "studies. Any hedge, caveat, or constraint about this trial is a KEEP."
)
SHOTS_VALIDATE_RCT = [
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

# FWS / future-work mode — for the other half of gap extraction (Zhang et al. 2022,
# future-work-sentence recognition). KEEP forward-looking author intentions/suggestions;
# DROP contributions, results, background, and non-forward-looking conclusions. The
# lexical decoys ("this suggests...", "results confirm...") are the hard negatives.
SYSTEM_VALIDATE_FWS = (
    "You are a precision filter for FUTURE-WORK sentences in academic papers. KEEP a sentence "
    "if it describes what the AUTHORS plan, intend, or suggest to do in the FUTURE, or a "
    "direction future research should take — e.g. 'in future work we will...', 'we plan/intend "
    "to...', 'we leave X for future work', 'it would be interesting to...', 'future research "
    "should explore...', 'X remains to be investigated', 'we aim to extend...'. "
    "DROP a sentence if it is NOT forward-looking: (a) background or motivation; (b) a "
    "description of what was DONE in this paper (contributions, methods, results); (c) a "
    "restatement of findings or conclusions with no future direction; (d) a limitation stated "
    "with no future action; (e) a citation or reference. When unsure whether it is "
    "forward-looking, KEEP. Bias toward KEEP for any explicit plan, intention, or suggestion "
    "about the future."
)
SHOTS_VALIDATE_FWS = [
    ("In future work, we plan to extend our approach to multilingual settings.", "YES"),
    ("It would be interesting to investigate the effect of larger training corpora.", "YES"),
    ("We leave the exploration of unsupervised variants for future work.", "YES"),
    ("Future research should examine how these findings generalize to other domains.", "YES"),
    ("We intend to incorporate syntactic features in subsequent versions of the model.", "YES"),
    ("These questions remain to be addressed in follow-up studies.", "YES"),
    ("Our model achieves state-of-the-art results on three benchmarks.", "NO"),
    ("In this paper, we proposed a novel attention mechanism.", "NO"),
    ("The dataset consists of 10,000 manually annotated sentences.", "NO"),
    ("These results confirm our hypothesis that context improves accuracy.", "NO"),
    ("Prior work has largely ignored this phenomenon.", "NO"),
]

# V4 — V3 prompt + chain-of-thought (category first, then YES/NO). Best on the
# 10-sample test (100% gap recall vs V3's 75%).
SYSTEM_VALIDATE_COT = SYSTEM_VALIDATE + (
    " First, in 1 short phrase, identify which category the sentence falls into. "
    "Then on a new line write 'Answer: YES' or 'Answer: NO'.\n"
    "Format:\nCategory: <one phrase>\nAnswer: YES|NO"
)

# V5 — diagnosed from LimGen: V4's "contribution claim" rejection over-fires on
# 64% of real limitations. V5 narrows rejections to a STRICT LIST of 6
# categories with explicit ACL checklist coverage, and DEFAULTS TO ACCEPT when
# uncertain. Adds shots for the borderline limitation/future-work patterns that
# V4 mislabeled (we-plan-to, we-might, we-discuss-limitations).
SYSTEM_VALIDATE_V5 = (
    "You are a strict rejection filter for research-gap sentences. Output NO only "
    "when you are HIGHLY CONFIDENT the sentence belongs to one of these six reject "
    "categories:\n"
    "  1. ACK_THANKS - explicit thanks/gratitude (e.g., 'we thank', 'we are grateful')\n"
    "  2. CITATION_REF - pure cross-reference with no content (e.g., 'See Appendix X', "
    "'as shown in Table Y', 'Smith et al., 2024')\n"
    "  3. CHECKLIST - ACL/Responsible-Research checklist questions or labels (e.g., "
    "'Did you describe X?', 'C1. ...', 'Have you used Y?')\n"
    "  4. FORMULA - pure equation/lemma reference (e.g., 'Lemma 4.4 in step t+1', "
    "'See Equation (3)')\n"
    "  5. ACHIEVEMENT - clear performance result with numbers or comparisons "
    "(e.g., 'our method achieves 95%', 'outperforms baselines by 3 points', "
    "'we are the first to achieve X'). NOT every 'we'/'our' sentence!\n"
    "  6. BROKEN - clearly truncated sentence (mid-clause start, abandoned at hyphen)\n"
    "\nACCEPT (YES) everything else. When in doubt, output YES. This explicitly "
    "INCLUDES: limitation statements ('we might', 'we did not', 'our method is exact "
    "but...', 'we have not'), scope restrictions ('focuses on', 'restricted to'), "
    "future-work plans ('we plan to', 'we leave for', 'we aim to', 'future work could'), "
    "limitation introducers ('we discuss the limitations'), method descriptions inside "
    "limitation discussions, and any sentence that starts with 'We' but is not an "
    "explicit achievement claim.\n"
    "\nFormat:\nReject_category: <one of the 6 categories or NONE>\nAnswer: YES|NO\n"
    "Output NO only if Reject_category is one of the 6. Otherwise YES."
)
SHOTS_VALIDATE_V5 = [
    # === ACCEPT (subtle limitations / future-work the LLM kept failing on) ===
    ("We plan to improve computational efficiency in future work.",
     "Reject_category: NONE\nAnswer: YES"),
    ("We might require a different linking hypothesis for each such effect.",
     "Reject_category: NONE\nAnswer: YES"),
    ("We discuss the limitations of our framework as follows.",
     "Reject_category: NONE\nAnswer: YES"),
    ("Our closed-form equations are exact, but they become intractable for large N.",
     "Reject_category: NONE\nAnswer: YES"),
    ("This work focuses on English-language datasets.",
     "Reject_category: NONE\nAnswer: YES"),
    ("We did not study risks that may arise in other application scenarios.",
     "Reject_category: NONE\nAnswer: YES"),
    ("We leave multilingual evaluation for future work.",
     "Reject_category: NONE\nAnswer: YES"),
    ("Our method assumes the availability of a knowledge base.",
     "Reject_category: NONE\nAnswer: YES"),
    # === REJECT (the 6 strict categories) ===
    ("We thank the anonymous reviewers for their helpful feedback.",
     "Reject_category: ACK_THANKS\nAnswer: NO"),
    ("See Appendix L.3 for further details.",
     "Reject_category: CITATION_REF\nAnswer: NO"),
    ("Did you describe the limitations of your work?",
     "Reject_category: CHECKLIST\nAnswer: NO"),
    ("Lemma 4.4 in the next step t + 1.",
     "Reject_category: FORMULA\nAnswer: NO"),
    ("Our method achieves 95.2% accuracy, outperforming baselines by 3.1 points.",
     "Reject_category: ACHIEVEMENT\nAnswer: NO"),
    ("of the proposed F DAN framework with the additional",
     "Reject_category: BROKEN\nAnswer: NO"),
]

# V6 — V5 + sharper rejection shots based on what V5 misses on real PDF output.
# V5 was too permissive on gold (kept all of V4's correct kills). V6 adds shots
# covering the specific patterns V5 missed: hedged gratitude, "we propose X"
# contribution framing, advisory statements, recommendations, "we abide/use".
SYSTEM_VALIDATE_V6 = (
    "You are a strict rejection filter for research-gap sentences. Output NO when "
    "the sentence belongs to one of these seven reject categories:\n"
    "  1. ACK_THANKS - any thanks, gratitude, or 'would like to express thanks/gratitude'\n"
    "  2. CITATION_REF - cross-reference with no content (e.g., 'See Appendix X', 'Table Y shows')\n"
    "  3. CHECKLIST - ACL Responsible-Research checklist Q&A (e.g., 'Did you describe...?', 'C1.')\n"
    "  4. FORMULA - pure equation/lemma reference (e.g., 'Lemma 4.4', 'Equation (3)')\n"
    "  5. ACHIEVEMENT - performance result OR 'we propose/introduce/develop X' contribution claim\n"
    "  6. ADVISORY - recommendations, warnings, terms-of-use (e.g., 'we advise against', "
    "'we recommend', 'we abide by')\n"
    "  7. BROKEN - clearly truncated sentence (mid-clause start, abandoned at hyphen)\n"
    "\nACCEPT (YES) everything else. Limitation statements, scope restrictions, future-work "
    "plans, and limitation-discussion setup ALL get YES. When the sentence COULD be a real "
    "limitation acknowledgment or future-work plan, output YES.\n"
    "\nFormat:\nReject_category: <one of the 7 categories or NONE>\nAnswer: YES|NO"
)
# V10 — V9 + 1 line of speculation/encourage hints. ~50% shorter than V7.
SYSTEM_VALIDATE_V10 = (
    "A classifier picked this sentence as a paper limitation. Reply NO if it is "
    "clearly one of: gratitude; numbered result; hyperparam list; method recipe "
    "('we use X to do Y'); prior-work citation; encouragement to reader; speculation "
    "about benefits ('could help', 'might improve'); paper-intro ('in this work we "
    "propose/aim/hypothesize'); truncated fragment. Otherwise YES.\n"
    "Format:\nReject: <category or NONE>\nAnswer: YES|NO"
)
SHOTS_VALIDATE_V10 = [
    ("We leave multilingual evaluation for future work.",
     "Reject: NONE\nAnswer: YES"),
    ("This work focuses on English-language datasets.",
     "Reject: NONE\nAnswer: YES"),
    ("Did you describe the limitations of your work?",
     "Reject: NONE\nAnswer: YES"),
    ("We thank the anonymous reviewers.",
     "Reject: gratitude\nAnswer: NO"),
    ("Our method achieves 95.2% accuracy.",
     "Reject: numbered result\nAnswer: NO"),
    ("Batch size b, learning rate lr.",
     "Reject: hyperparam list\nAnswer: NO"),
    ("We use commonly-used PaddleOCR for OCR.",
     "Reject: method recipe\nAnswer: NO"),
    ("Hopkins (2022) observe that PCFGs are too simple.",
     "Reject: prior-work citation\nAnswer: NO"),
    ("We encourage readers to consider annotated roles.",
     "Reject: encouragement\nAnswer: NO"),
    ("Diverse outputs could help users find solutions faster.",
     "Reject: speculation\nAnswer: NO"),
    ("In this work, we propose a new hybrid framework.",
     "Reject: paper-intro\nAnswer: NO"),
]


# V9 — minimal version of V7. Same targeting, ~70% shorter.
SYSTEM_VALIDATE_V9 = (
    "A classifier picked this sentence as a paper limitation. Reply NO if it is "
    "clearly one of: gratitude, numbered result, hyperparam list, method recipe, "
    "prior-work citation, benefit speculation, or truncated fragment. Otherwise YES.\n"
    "Format:\nReject: <category or NONE>\nAnswer: YES|NO"
)
SHOTS_VALIDATE_V9 = [
    ("We leave multilingual evaluation for future work.",
     "Reject: NONE\nAnswer: YES"),
    ("This work focuses on English-language datasets.",
     "Reject: NONE\nAnswer: YES"),
    ("Did you report the number of parameters?",
     "Reject: NONE\nAnswer: YES"),
    ("We thank the anonymous reviewers.",
     "Reject: gratitude\nAnswer: NO"),
    ("Our method achieves 95.2% accuracy, outperforming baselines by 3.1 points.",
     "Reject: numbered result\nAnswer: NO"),
    ("Batch size b, learning rate lr, regularization alpha.",
     "Reject: hyperparam list\nAnswer: NO"),
    ("We use commonly-used PaddleOCR to handle our dataset.",
     "Reject: method recipe\nAnswer: NO"),
    ("Hopkins (2022) observe that PCFGs may not reflect real grammar.",
     "Reject: prior-work citation\nAnswer: NO"),
    ("Diverse outputs could help users find solutions faster.",
     "Reject: benefit speculation\nAnswer: NO"),
]


# V7 — diagnosed from 20-sample iteration: LimGen Limitations sections often
# include ACL checklist Q&A and inline contribution mentions. The optimization
# is to identify SciBERT FPs (sentences that are NOT in Limitations sections
# despite looking limitation-like): hyperparameter descriptions, method recipes,
# encouragements, speculative-benefit statements. V7 keeps V5's default-accept
# but adds METHODOLOGY/HYPERPARAM/ENCOURAGE reject categories, and EXPLICITLY
# ACCEPTS ACL checklist Q&A (LimGen labels them as gold positives).
SYSTEM_VALIDATE_V7 = (
    "You filter sentences from a paper's LIMITATIONS section (not other sections). "
    "ACCEPT (YES) ANY sentence that plausibly comes from a limitations/future-work "
    "section, including: limitation acknowledgments, scope restrictions, future-work "
    "plans, ethical concerns, ACL responsible-research checklist questions, and "
    "subtle hedged statements about the work.\n"
    "\nREJECT (NO) only if the sentence is one of these — and clearly does NOT belong "
    "in a Limitations section:\n"
    "  1. ACK_THANKS - explicit thanks/gratitude to people\n"
    "  2. ACHIEVEMENT - performance result with NUMBERS (e.g., '95% accuracy', 'outperforms by 3 pts')\n"
    "  3. HYPERPARAM - hyperparameter list (e.g., 'batch size b, learning rate lr, regularization alpha')\n"
    "  4. METHOD_RECIPE - step-by-step procedural description (e.g., 'we exhaustively include all', "
    "'we make use of knowledge of the exact rules')\n"
    "  5. ENCOURAGE - encouragements or suggestions to readers (e.g., 'we encourage them to consider')\n"
    "  6. SPECULATIVE_BENEFIT - speculation about benefits (e.g., 'X could help users follow')\n"
    "  7. BROKEN - truncated fragment (mid-clause start, abandoned at hyphen)\n"
    "\nWhen in doubt, output YES. The default is to ACCEPT.\n"
    "\nFormat:\nReject_category: <one of the 7 categories or NONE>\nAnswer: YES|NO"
)
SHOTS_VALIDATE_V7 = [
    # ACCEPT: classic limitation/future-work patterns
    ("We leave multilingual evaluation for future work.",
     "Reject_category: NONE\nAnswer: YES"),
    ("This work focuses on English-language datasets.",
     "Reject_category: NONE\nAnswer: YES"),
    ("We did not study risks that may arise in other scenarios.",
     "Reject_category: NONE\nAnswer: YES"),
    # ACCEPT: ACL checklist Q&A (LimGen labels these as gold positives)
    ("C Did you run computational experiments? section 5 and 6 C1.",
     "Reject_category: NONE\nAnswer: YES"),
    ("Did you report the number of parameters in the models used?",
     "Reject_category: NONE\nAnswer: YES"),
    # ACCEPT: contribution claims that appear in Limitations sections (subtle)
    ("This work demonstrates promising zero-shot detection using prompt engineering.",
     "Reject_category: NONE\nAnswer: YES"),
    ("First, our data augmentation strategy relies on the reconstruction ability of cycle adversarial nets.",
     "Reject_category: NONE\nAnswer: YES"),
    # ACCEPT: limitation-discussion methodology mentions
    ("In our interpretability analyses, where we make use of knowledge of the exact rules.",
     "Reject_category: NONE\nAnswer: YES"),
    # REJECT: clear non-Limitations patterns
    ("We thank the anonymous reviewers for their helpful feedback.",
     "Reject_category: ACK_THANKS\nAnswer: NO"),
    ("Our method achieves 95.2% accuracy, outperforming baselines by 3.1 points.",
     "Reject_category: ACHIEVEMENT\nAnswer: NO"),
    ("Batch size b, learning rate lr, and regularization coefficient alpha are among the hyperparameters.",
     "Reject_category: HYPERPARAM\nAnswer: NO"),
    ("We exhaustively include all linked passages for each table in the dataset.",
     "Reject_category: METHOD_RECIPE\nAnswer: NO"),
    ("We encourage readers to consider the annotated roles for better reasoning.",
     "Reject_category: ENCOURAGE\nAnswer: NO"),
    ("NL explanations could help users follow the flow of complex data transformations.",
     "Reject_category: SPECULATIVE_BENEFIT\nAnswer: NO"),
    ("of the proposed framework with the additional",
     "Reject_category: BROKEN\nAnswer: NO"),
]


# V8 — SURGICAL FP detector. Targets only the SPECIFIC patterns SciBERT FPs
# fall into (analyzed from 112 real SciBERT FPs on LimGen). Default = ACCEPT;
# reject ONLY when the sentence matches one of the explicit FP patterns below.
SYSTEM_VALIDATE_V8 = (
    "You remove ONLY sentences that match one of these specific patterns. "
    "Reply NO if and only if the sentence is clearly one of these. Reply YES for "
    "EVERYTHING ELSE — including limitation acknowledgments, future-work plans, "
    "scope restrictions, ACL checklist questions, and any sentence whose category "
    "is unclear.\n"
    "\nReject (NO) categories:\n"
    "  1. PAPER_INTRO - paper/method summary describing what the work does: "
    "'In this work/paper, we propose/present/hypothesize/introduce X' as a "
    "contribution claim, NOT a limitation.\n"
    "  2. METHOD_DESC - description of the method's components, equations, "
    "datasets used: 'The proposed method regards X as Y', 'We use the commonly-used X', "
    "'We modify Equation Y', 'For singleton cases, we...'\n"
    "  3. PRIOR_WORK - statement about prior work's findings: 'Hopkins (2022) "
    "observe X', 'Prior work has shown Y', 'Recent studies find Z'.\n"
    "  4. DATASET_DESC - dataset/benchmark descriptions: 'X is a dataset of Y', "
    "'X (Author, Year) is a benchmark for Z'.\n"
    "  5. HYPERPARAM - hyperparameter listings: 'batch size b, learning rate lr, "
    "regularization alpha'.\n"
    "  6. SPECULATIVE - speculation about benefits, beliefs, possibilities: "
    "'we believe X', 'could further improve Y', 'it should be feasible to Z'.\n"
    "  7. ACK_THANKS - explicit thanks/gratitude to people.\n"
    "  8. OFF_TOPIC - clearly off-topic noise (product reviews, irrelevant text).\n"
    "  9. BROKEN - truncated fragment.\n"
    "\nIMPORTANT — these should be ACCEPTED (YES), even though they may look "
    "borderline:\n"
    "  - 'We did not X', 'We leave X for future work', 'We plan to X' (limitation/future-work)\n"
    "  - 'This work focuses on X', 'Our method assumes Y' (scope/assumption)\n"
    "  - ACL Responsible-Research checklist Q&A ('Did you describe X?', 'C1.')\n"
    "  - Any sentence with explicit 'limitation', 'shortcoming', 'open problem'\n"
    "\nFormat:\nReject_category: <one of the 9 categories or NONE>\nAnswer: YES|NO"
)
SHOTS_VALIDATE_V8 = [
    # === REJECT examples (the SciBERT FPs we want to remove) ===
    ("In this work, we hypothesize that end-to-end neural models for topic classification rely on shortcuts.",
     "Reject_category: PAPER_INTRO\nAnswer: NO"),
    ("In this paper, we present a hybrid approach for dialogical data collection.",
     "Reject_category: PAPER_INTRO\nAnswer: NO"),
    ("The proposed method regards word segmentation and tagging as a joint, multiclass classification problem.",
     "Reject_category: METHOD_DESC\nAnswer: NO"),
    ("We use the commonly-used PaddleOCR to handle our dataset.",
     "Reject_category: METHOD_DESC\nAnswer: NO"),
    ("For singleton cases, we modify the last two terms of Equation 5.",
     "Reject_category: METHOD_DESC\nAnswer: NO"),
    ("Hopkins (2022) observe that the simplistic nature of the PCFGs may not be reflective of real grammar.",
     "Reject_category: PRIOR_WORK\nAnswer: NO"),
    ("Furthermore, prior work has shown that LLMs follow the path of least resistance.",
     "Reject_category: PRIOR_WORK\nAnswer: NO"),
    ("Yelp-2 (Zhang et al., 2015) is a sentiment analysis dataset on Yelp reviews.",
     "Reject_category: DATASET_DESC\nAnswer: NO"),
    ("Batch size b, learning rate lr, and regularization coefficient alpha are among the hyperparameters.",
     "Reject_category: HYPERPARAM\nAnswer: NO"),
    ("We believe that this work will lead to a greater understanding of lexical semantics.",
     "Reject_category: SPECULATIVE\nAnswer: NO"),
    ("Combining these orthogonal optimizations could further accelerate the inference.",
     "Reject_category: SPECULATIVE\nAnswer: NO"),
    ("We thank the anonymous reviewers for their feedback.",
     "Reject_category: ACK_THANKS\nAnswer: NO"),
    ("I have had it for a year and it still looks just as good as the day I bought it.",
     "Reject_category: OFF_TOPIC\nAnswer: NO"),
    ("of the proposed framework with the additional",
     "Reject_category: BROKEN\nAnswer: NO"),
    # === ACCEPT examples (real limitations the LLM kept misreading) ===
    ("We leave multilingual evaluation for future work.",
     "Reject_category: NONE\nAnswer: YES"),
    ("This work focuses on English-language datasets.",
     "Reject_category: NONE\nAnswer: YES"),
    ("We did not study risks that may arise in other scenarios.",
     "Reject_category: NONE\nAnswer: YES"),
    ("Our method assumes the availability of a knowledge base.",
     "Reject_category: NONE\nAnswer: YES"),
    ("C Did you run computational experiments? section 5 and 6 C1.",
     "Reject_category: NONE\nAnswer: YES"),
    ("Did you report the number of parameters in the models used?",
     "Reject_category: NONE\nAnswer: YES"),
    ("First, our data augmentation strategy relies on the reconstruction ability of cycle adversarial nets.",
     "Reject_category: NONE\nAnswer: YES"),
    ("A limitation of our approach is that it assumes English input.",
     "Reject_category: NONE\nAnswer: YES"),
]


SHOTS_VALIDATE_V6 = SHOTS_VALIDATE_V5 + [
    # Hedged gratitude (V5 missed: "We would like to express our gratitude...")
    ("We would like to express our gratitude to Dr. Smith for sharing the data.",
     "Reject_category: ACK_THANKS\nAnswer: NO"),
    # "We propose X" contribution claim (V5 missed: "To address these limitations, we propose...")
    ("To address these limitations, we propose a new evolutionary framework.",
     "Reject_category: ACHIEVEMENT\nAnswer: NO"),
    ("We demonstrated that fine-tuned Llama 2 outperforms baselines.",
     "Reject_category: ACHIEVEMENT\nAnswer: NO"),
    # Advisory/recommendation (V5 missed: "we advise against", "we recommend")
    ("We advise against using this framework for production deployments.",
     "Reject_category: ADVISORY\nAnswer: NO"),
    ("We recommend security testing for any downstream use.",
     "Reject_category: ADVISORY\nAnswer: NO"),
    ("We abide by their terms of use.",
     "Reject_category: ADVISORY\nAnswer: NO"),
]
SHOTS_VALIDATE_COT = [
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
    ("Lemma 4.4 in the next step t + 1.",
     "Category: math fragment\nAnswer: NO"),
]


class LLMGapFilter:
    def __init__(self, backend: str = "local", model: str | None = None,
                 mode: str = "validate", client=None, context_style: str = "fields"):
        self.backend = backend
        self.mode = mode
        # How ``use_context`` renders a candidate's surrounding text in the batch
        # prompt: "fields" = separate TARGET + SURROUNDING TEXT lines (target
        # appears twice); "inline" = one passage with the target wrapped in
        # «guillemets», judged in place (target appears once).
        self.context_style = context_style
        self.model = model or (DEFAULT_LOCAL if backend == "local" else DEFAULT_API)
        if mode == "validate":
            self._sys, self._shots = SYSTEM_VALIDATE, SHOTS_VALIDATE
        elif mode == "validate_cot":
            self._sys, self._shots = SYSTEM_VALIDATE_COT, SHOTS_VALIDATE_COT
        elif mode == "validate_v5":
            self._sys, self._shots = SYSTEM_VALIDATE_V5, SHOTS_VALIDATE_V5
        elif mode == "validate_v6":
            self._sys, self._shots = SYSTEM_VALIDATE_V6, SHOTS_VALIDATE_V6
        elif mode == "validate_v7":
            self._sys, self._shots = SYSTEM_VALIDATE_V7, SHOTS_VALIDATE_V7
        elif mode == "validate_v8":
            self._sys, self._shots = SYSTEM_VALIDATE_V8, SHOTS_VALIDATE_V8
        elif mode == "validate_v9":
            self._sys, self._shots = SYSTEM_VALIDATE_V9, SHOTS_VALIDATE_V9
        elif mode == "validate_v10":
            self._sys, self._shots = SYSTEM_VALIDATE_V10, SHOTS_VALIDATE_V10
        elif mode == "validate_rct":
            self._sys, self._shots = SYSTEM_VALIDATE_RCT, SHOTS_VALIDATE_RCT
        elif mode == "validate_fws":
            self._sys, self._shots = SYSTEM_VALIDATE_FWS, SHOTS_VALIDATE_FWS
        else:
            self._sys, self._shots = SYSTEM_JUNK, SHOTS_JUNK
        self._cot = mode in ("validate_cot", "validate_v5", "validate_v6", "validate_v7", "validate_v8", "validate_v9", "validate_v10")
        #: junk mode inverts (YES = obvious junk = drop); every validate* mode keeps
        #: on YES. Flag off the actual prompt so it is robust to the mode string.
        self._junk = (self._sys is SYSTEM_JUNK)
        self._tok = self._lm = None        # local, lazy
        self._client = client              # api
        #: extra provider params merged into chat.completions.create (e.g. Yandex
        #: reasoning-model controls: {"reasoning_effort": "low"} for gpt-oss,
        #: {"chat_template_kwargs": {"enable_thinking": False}} for qwen3).
        self.extra_body: dict | None = None
        self.n_judged = 0                  # sentences judged
        self.n_calls = 0                   # LLM calls made (batched << judged)

    # -- prompt -----------------------------------------------------------
    def _messages(self, sentence: str) -> list[dict]:
        msgs = [{"role": "system", "content": self._sys}]
        for s, a in self._shots:
            msgs += [{"role": "user", "content": "Sentence: " + s},
                     {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": "Sentence: " + sentence})
        return msgs

    def _keep(self, text: str) -> bool:
        # Empty/None content (e.g. a reasoning model that spent its budget before
        # emitting an answer) is treated as KEEP for validate* modes — never silently
        # drop a candidate; junk mode's empty means "not obvious junk" -> also keep.
        t = (text or "").strip().upper()
        if not t:
            return not self._junk
        if self._cot:
            for ln in t.splitlines():
                if "ANSWER" in ln:
                    return "YES" in ln.split(":", 1)[-1]
            return t.startswith("Y")
        yes = t.startswith("Y")
        return (not yes) if self._junk else yes   # validate*: YES=gap=keep; junk: YES=junk=drop

    # -- backends ---------------------------------------------------------
    def _ensure_local(self):
        if self._lm is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            log.info("Loading local judge %s ...", self.model)
            self._tok = AutoTokenizer.from_pretrained(self.model)
            use_cuda = torch.cuda.is_available()
            kw = {"dtype": torch.float16 if use_cuda else torch.float32}
            if use_cuda:
                kw["device_map"] = "auto"
            self._lm = AutoModelForCausalLM.from_pretrained(self.model, **kw)
            self._lm.eval()
            self._device = next(self._lm.parameters()).device

    def judge(self, sentence: str) -> bool:
        self.n_judged += 1
        self.n_calls += 1
        if self.backend == "local":
            import torch
            self._ensure_local()
            text = self._tok.apply_chat_template(self._messages(sentence), tokenize=False,
                                                 add_generation_prompt=True)
            inp = self._tok(text, return_tensors="pt").to(self._device)
            max_new = 30 if self._cot else 2
            with torch.no_grad():
                out = self._lm.generate(**inp, max_new_tokens=max_new, do_sample=False,
                                        pad_token_id=self._tok.eos_token_id)
            ans = self._tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            return self._keep(ans)
        # api
        if self._client is None:
            from gap2idea.pipeline.llm import get_llm_client
            self._client = get_llm_client()
        resp = self._client.chat.completions.create(
            model=self.model, messages=self._messages(sentence),
            temperature=0.0, max_tokens=(30 if self._cot else 5))
        return self._keep(resp.choices[0].message.content)

    # -- batched judge ----------------------------------------------------
    @staticmethod
    def _item_text(it) -> str:
        """Target sentence of a batch item (a bare str or a {sentence,...} dict)."""
        return str(it.get("sentence", "") if isinstance(it, dict) else it)

    def _batch_messages(self, items: list) -> list[dict]:
        """One prompt that judges MANY items. The system prompt + the mode's
        few-shots (rendered inline as KEEP/DROP calibration) are sent once; the
        candidates follow as a numbered list. Output is a per-id verdict array.

        Each item is a bare sentence, or a dict with optional metadata to
        disambiguate own-vs-prior work:
          {"sentence": str, "title": str|None, "context": str|None}
        where ``context`` is the surrounding text (±1 sentence, i.e. the row's
        ``paragraph_text``). Metadata is context only — the model judges the
        TARGET sentence, using title/surrounding text just to resolve references
        like "this approach" / "such methods".
        """
        cal = "\n".join(f'- {"KEEP" if self._keep(a) else "DROP"}: "{s}"'
                        for s, a in self._shots)
        has_title = any(isinstance(it, dict) and it.get("title") for it in items)
        has_ctx = any(isinstance(it, dict) and it.get("context") for it in items)
        inline = has_ctx and self.context_style == "inline"
        note = ""
        if has_title or has_ctx:
            if inline:
                bits = (["the PAPER TITLE"] if has_title else [])
                note = ("\n\nEach item is a short PASSAGE with exactly ONE sentence wrapped in "
                        "«guillemets» — that marked sentence is the TARGET. Judge ONLY the "
                        "«marked» TARGET; use the rest of the passage"
                        + (" and " + " and ".join(bits) if bits else "")
                        + " solely to tell whether the TARGET refers to the AUTHORS' OWN work "
                        "or to prior/other work — never judge the surrounding text itself.")
            else:
                bits = (["the PAPER TITLE"] if has_title else []) + \
                       (["the SURROUNDING TEXT (sentences around it)"] if has_ctx else [])
                note = ("\n\nEach item has a TARGET sentence plus " + " and ".join(bits) +
                        ". Judge ONLY the TARGET; use " + " / ".join(
                            (["title"] if has_title else []) + (["surrounding text"] if has_ctx else []))
                        + " solely to tell whether the TARGET refers to the AUTHORS' OWN work or "
                        "to prior/other work — never judge the context itself.")
        sys = (
            self._sys + note
            + "\n\nYou will receive a NUMBERED list. For EACH item, decide keep=true if the "
            "TARGET states the AUTHORS' OWN limitation, assumption, scope restriction, or "
            "future-work direction, and keep=false otherwise (including limitations of prior "
            "or other work). Return JSON of the form "
            '{"results":[{"id":<n>,"keep":<bool>}, ...]} with EXACTLY one entry per input '
            "item, using the same ids, in order.\n\nCalibration examples:\n" + cal
        )
        lines = []
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                lines.append(f"{i + 1}. {it}")
                continue
            s = str(it.get("sentence", ""))
            ctx = str(it.get("context", "") or "").strip()
            title = str(it.get("title", "") or "")
            if inline and ctx:
                # mark the target once, in place; fall back to prepending if the
                # exact sentence isn't a substring of the paragraph
                marked = ctx.replace(s, f"«{s}»", 1) if s and s in ctx \
                    else f"«{s}»  [context: {ctx}]"
                block = f"{i + 1}. {marked}"
                if title:
                    block += f"\n   PAPER TITLE: {title}"
            else:
                block = f"{i + 1}. TARGET: {s}"
                if title:
                    block += f"\n   PAPER TITLE: {title}"
                if ctx and ctx != s.strip():
                    block += f"\n   SURROUNDING TEXT: {ctx}"
            lines.append(block)
        return [{"role": "system", "content": sys},
                {"role": "user", "content": "Items:\n" + "\n".join(lines)}]

    def judge_batch(self, items: list) -> list[bool]:
        """Judge every item in ONE api call; returns keep-flags in input order.

        ``items`` are bare sentences or ``{sentence,title,context}`` dicts (see
        ``_batch_messages``). Falls back to per-sentence judging for the local
        backend and for any verdict the batch response omits or malforms (so a bad
        row never silently drops a candidate). ``n_calls`` counts the single batch
        call plus any fallbacks — normally 1 per chunk of ``BATCH_CHUNK``.
        """
        if not items:
            return []
        if self.backend != "api":
            return [self.judge(self._item_text(it)) for it in items]
        if self._client is None:
            from gap2idea.pipeline.llm import get_llm_client
            self._client = get_llm_client()
        from gap2idea.pipeline.llm import parse_json_response

        self.n_calls += 1
        vmap: dict[int, bool] = {}
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=self._batch_messages(items),
                # generous budget: reasoning models (gpt-oss, qwen3) burn tokens on
                # hidden analysis before the JSON — a terse yandexgpt reply stops early.
                temperature=0.0, max_tokens=min(8192, 256 + 400 * len(items)),
                response_format={"type": "json_schema", "json_schema": {
                    "name": "gap_verdicts", "schema": BATCH_SCHEMA, "strict": True}},
                **({"extra_body": self.extra_body} if self.extra_body else {}))
            data = parse_json_response(resp.choices[0].message.content)
            for r in data.get("results", []):
                if isinstance(r, dict) and "id" in r and "keep" in r:
                    vmap[int(r["id"])] = bool(r["keep"])
        except Exception as e:  # network / JSON / schema — degrade, don't crash
            log.warning("Stage C batch judge failed (%s) — per-sentence fallback", e)

        out: list[bool] = []
        resolved = 0
        for i, it in enumerate(items):
            if (i + 1) in vmap:
                out.append(vmap[i + 1])
                resolved += 1
            else:
                out.append(self.judge(self._item_text(it)))   # counts its own call
        self.n_judged += resolved
        return out

    # -- apply ------------------------------------------------------------
    #: cue-rule hits inside these sections are trusted (see ``protect_rules``);
    #: rule hits in every other section (discussion, GROBID-introduction, tail,
    #: midpaper) are judged, because that is where prior-work critiques and
    #: conclusion-summary false positives concentrate.
    PROTECT_SECTIONS = ("limitations", "future_work")

    def filter_gaps(self, gaps: list[dict], protect_rules: bool = True,
                    protect_sections: "tuple[str, ...] | None" = None,
                    batch: bool = True, chunk_size: "int | None" = None,
                    use_context: bool = False, use_title: bool = False) -> list[dict]:
        """Keep gaps the LLM confirms; tag survivors with source '+llm'.

        protect_rules: cue-rule hits are already high-precision and hold most real
        gaps, so don't risk the LLM rejecting them — but ONLY inside explicit gap
        sections (``protect_sections``: Limitations / Future-Work). A cue rule that
        fires in a discussion, introduction, tail, or mid-paper region is far more
        likely to be a prior-work critique ("existing methods cannot...") or a
        conclusion-summary claim, so we still judge those. Pure-model predictions
        are always judged. Set ``protect_rules=False`` to judge everything.

        batch: judge all candidates in ~one call per ``chunk_size`` (default
        ``BATCH_CHUNK``) instead of one call per sentence — the system prompt and
        few-shots are sent once per chunk, so tokens and latency collapse. Order
        is preserved. Set ``batch=False`` for the legacy one-call-per-sentence path.

        use_context / use_title: feed each candidate's surrounding text (the row's
        ``paragraph_text``, ±1 sentence) and/or the paper ``title`` to the judge as
        disambiguation context (helps resolve "this approach" own-vs-prior). Only
        the target sentence is judged. Requires ``batch`` + the api backend.
        """
        protect_sections = self.PROTECT_SECTIONS if protect_sections is None else protect_sections
        chunk = chunk_size or BATCH_CHUNK

        # Decide which candidates to judge (rest are protected rule hits, kept as-is).
        judge_idx = [
            i for i, g in enumerate(gaps)
            if not (protect_rules and "rule" in g.get("source", "")
                    and str(g.get("section_type", "")).lower() in protect_sections)
        ]
        verdict: dict[int, bool] = {}
        if judge_idx:
            def _item(g):
                if not (use_context or use_title):
                    return str(g.get("gap_sentence", ""))
                it = {"sentence": str(g.get("gap_sentence", ""))}
                if use_title:
                    it["title"] = str(g.get("title", "") or "")
                if use_context:
                    it["context"] = str(g.get("paragraph_text", "") or "")
                return it
            items = [_item(gaps[i]) for i in judge_idx]
            if batch and self.backend == "api":
                flags: list[bool] = []
                for k in range(0, len(items), chunk):
                    flags.extend(self.judge_batch(items[k:k + chunk]))
            else:
                flags = [self.judge(self._item_text(it)) for it in items]
            verdict = dict(zip(judge_idx, flags))

        kept = []
        for i, g in enumerate(gaps):
            if i not in verdict:
                kept.append(g)                                  # protected
            elif verdict[i]:
                kept.append({**g, "source": g.get("source", "") + "+llm"})
        return kept
