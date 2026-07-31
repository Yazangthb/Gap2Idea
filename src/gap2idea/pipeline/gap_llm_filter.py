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

# Two prompt modes (set via mode=):
#  "validate" — strict "is this a gap?"; keep iff YES. Catches method-description
#               / contribution false positives too. Needs a capable model (>=1.5B).
#  "junk"     — conservative "is this obvious junk?"; drop iff YES. Recall-safe but
#               only removes acknowledgments/formulas/citations (use with tiny models).
SYSTEM_VALIDATE = (
    "You are a precision filter for research-gap extraction. Reject (NO) a sentence "
    "if it is ONE OF: (a) an acknowledgment or thanks; (b) a citation, cross-reference, "
    "or 'See Appendix/Figure/Section' line; (c) a CONTRIBUTION claim — sentences like "
    "'we propose', 'to address these limitations we...', 'our method achieves'; (d) a "
    "math equation, formula reference, or lemma statement; (e) a scramble or fragment "
    "(broken hyphenation, mid-clause start, or section-header fragment). Accept (YES) "
    "if it is the authors' own LIMITATION, ASSUMPTION, SCOPE restriction, or "
    "FUTURE-WORK direction. Reply one word: YES or NO."
)
SHOTS_VALIDATE = [
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
                 mode: str = "validate", client=None):
        self.backend = backend
        self.mode = mode
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
        else:
            self._sys, self._shots = SYSTEM_JUNK, SHOTS_JUNK
        self._cot = mode in ("validate_cot", "validate_v5", "validate_v6", "validate_v7", "validate_v8", "validate_v9", "validate_v10")
        self._tok = self._lm = None        # local, lazy
        self._client = client              # api
        self.n_judged = 0

    # -- prompt -----------------------------------------------------------
    def _messages(self, sentence: str) -> list[dict]:
        msgs = [{"role": "system", "content": self._sys}]
        for s, a in self._shots:
            msgs += [{"role": "user", "content": "Sentence: " + s},
                     {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": "Sentence: " + sentence})
        return msgs

    def _keep(self, text: str) -> bool:
        t = text.strip().upper()
        if self._cot:
            for ln in t.splitlines():
                if "ANSWER" in ln:
                    return "YES" in ln.split(":", 1)[-1]
            return t.startswith("Y")
        yes = t.startswith("Y")
        return yes if self.mode == "validate" else (not yes)   # validate: YES=gap=keep

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
            temperature=0.0, max_tokens=2)
        return self._keep(resp.choices[0].message.content)

    # -- apply ------------------------------------------------------------
    def filter_gaps(self, gaps: list[dict], protect_rules: bool = True) -> list[dict]:
        """Keep gaps the LLM confirms; tag survivors with source '+llm'.

        protect_rules: cue-rule hits are already high-precision and hold most real
        gaps, so don't risk the LLM rejecting them — only judge the pure-model
        predictions, where the false positives (math exposition, fragments)
        concentrate. This keeps recall while still filtering the noisy candidates.
        """
        kept = []
        for g in gaps:
            src = g.get("source", "")
            if protect_rules and "rule" in src:
                kept.append(g)
                continue
            if self.judge(str(g.get("gap_sentence", ""))):
                kept.append({**g, "source": src + "+llm"})
        return kept
