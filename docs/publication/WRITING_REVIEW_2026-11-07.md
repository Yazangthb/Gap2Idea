# Focused writing review - 7 November 2026

Scope: `manuscript_mdpi/main.tex` and its included sections, with edits restricted
to Materials and Methods, Results, Discussion and Conclusions. The abstract,
introduction, Related Work, main/preprint wrappers, appendix prompt, bibliography,
figure designs, captions, data tables and experiment code are unchanged.
No result was added, removed or recomputed. No pending Task 5b result was imported.
No Git commit or push was made for this review.

## Review judgement

The section order is already logical: task -> data -> pipeline -> configuration
choices -> comparisons -> evaluation; results follow the RQs and discussion
interprets them. I kept that structure. The principal problems were ambiguous
referents and denominators, broad statements that obscured a narrower comparison,
and using cost as shorthand for estimated LLM input tokens. I edited those places
rather than rewriting paragraphs that already explain the method adequately.

## Changes and reasons

| Location | Change | Reason |
| --- | --- | --- |
| Methods: task | Describe a paper as a sequence of sentences. | Clarifies the representation without implying every benchmark contains full text. |
| Methods: data | Qualify paper-disjoint splits as within each dataset. | SAL papers also occur in the gold corpus; the original could imply cross-dataset disjointness. |
| Methods: SAL | State that full-paper evaluation follows the disclosed short-sentence exclusion. | Removes the apparent conflict between scoring every sentence and removing some sentences first. |
| Methods: gold labels | Separate kappa from its interpretation. | Makes clear that it measures inter-model consistency, not agreement with humans. |
| Methods: Stage A | Replace cannot be a gap with exclusion rules. | The filter removes 20 labelled gold gaps; the previous wording implied perfect rejection of negatives. |
| Methods: Stage B | Explain normalisation of entailment and contradiction probabilities. | Expresses the same score calculation more directly. |
| Methods: Stage C | Describe the mode as a paragraph rather than the final instruction. | The appendix places a JSON-output instruction after the mode paragraph. |
| Methods: Stage C context | Specify that context includes the candidate and its immediate neighbours, truncated to 600 characters. | Verified in the context-window builders and request construction; the original sounded like only neighbours were sent. |
| Methods: threshold | State explicitly that positives lost in either A or B count as missed. | Clarifies the recall denominator and the end-to-end threshold target. |
| Methods: input cost | Call input-token volume a cost proxy and describe the counted request text. | The code does not measure money, latency, output tokens or all serving overhead. |
| Results: stage contribution | Use percentage points for absolute precision gains. | Distinguishes them from relative improvements. |
| Results: validation checks | Name the precision-increase/recall-decrease ordering being checked. | The previous this ordering followed an F1-ranking paragraph and could refer to the wrong quantity. |
| Results: validation percentages | Group gap retention separately from non-gap rejection and state that both concern Stage B candidates. | Prevents confusion with end-to-end recall; all values are preserved. |
| Results: gap types | Explain overall/pooled scores and the influence of research gaps. | Replaces an unclear dominate statement with its mechanism. |
| Results: no-research-gap analysis | Explain that reference-labelled research-gap rows are excluded while negatives remain. | The code filters evaluation rows; it does not relabel these gaps as negatives or rerun extraction. |
| Results: supervised comparison | Restrict similar F1 to high-precision mode. | The next paragraph reports larger F1 differences for the other modes. |
| Results: NLI and SAL labels | Distinguish fitting model weights from selecting a threshold. | No SAL labels at all contradicted the stated use of SAL dev labels. |
| Results: token savings | Express ratios as alternative tokens divided by funnel tokens. | Avoids the ambiguous times fewer formulation without changing any ratio. |
| Results: pending comparison | State what token counts alone cannot establish. | Does not imply that a future experiment already demonstrates a preferred approach. |
| Results: label-source table | Correct the navigation to the table's right side. | Those columns contain LLM-label precision, not both label sources. |
| Discussion: RQ1 | Define the recall ceiling as recall after Stage B. | Explains why Stage C cannot exceed it. |
| Discussion: RQ2 | Replace costs no F1 and level with observed ranking and similar F1 values. | Avoids implying exact equality or an established equivalence result. |
| Discussion: RQ3 | Focus on the measured supervised precision advantage rather than recommending all projects use available labels. | The experiment does not establish a universal model-selection rule. |
| Discussion: whole-paper comparison | Replace worthwhile only if with the unresolved accuracy trade-off. | Input counts alone do not establish an overall decision criterion. |
| Conclusions | Attribute the gold result to the test split; describe savings in input tokens. | Prevents readers inferring evaluation on all 230 papers or measured monetary/runtime savings. |

## Details to verify with the team - left unchanged

These are open evidence or interpretation questions, not claims that the values
are wrong. I did not fill gaps by guessing, and I did not rerun experiments.

1. **Historical localisation target and matching protocol.** Discussion's
   section-localisation paragraph calls 55.6% and 55.2% retention of arXiv gold
   gaps. `docs/experiments/stage_a_latex_sections.md` specifies limitation +
   future_work targets and token-containment matching >= 0.7, rather than the
   current three-type exact-sentence evaluation. Confirm the intended target,
   denominator, run artefact and paper set before revising that paragraph.
   The historical observations are present in documentation; their equivalence
   to the current benchmark is not established. Also confirm whether the 97%
   junk-filter retention there refers to arXiv alone or the combined gold test.
2. **True precision between two label sets.** Discussion says true precision is
   likely between 66.2% and 93.2%. Two disagreeing reference sets do not themselves
   establish a numerical bound. `stage_c_sal40_modes.py` can score each set, but
   cannot establish which set is correct. Keep this as an unverified hypothesis
   until human adjudication; the statement was left unchanged as requested.
3. **ArXiv collection and preprocessing.** Methods names 190 arXiv papers and
   PyMuPDF, but does not explain the search/selection criteria, collection dates,
   inclusion rules or exact sentence-splitter/configuration used for this frozen
   corpus. Multiple historical builders exist. Ask for the acquisition manifest
   and the specific builder revision; do not infer these details from a script
   that may have been used for a different corpus version.
4. **Stage C server identity and environment.** The request code uses the alias
   `qwen3.8-27b`; a client alias is not an immutable model identifier. Ask for the
   exact served checkpoint/revision, quantisation and server environment export.
   `uv.lock` exists but the inspected lock does not contain a vLLM package entry;
   the blanket claim that exact experimental versions are pinned there needs
   confirmation for the separately hosted server. Client-side scripts alone do
   not establish the serving configuration.
5. **Supervised model-selection procedure.** Methods says PubMedBERT was strongest
   of eight and was selected as Stage B, and Results presents test F1 values for
   that ranking. Confirm which split/metric actually determined that choice and
   provide the training configuration/run manifest. Current prose does not
   distinguish development-set selection from a retrospective test ranking.
6. **Development/test chronology.** Code and its docstrings describe frozen
   rubrics and validation checks, but cannot independently prove when each prompt
   was written or viewed. Provide dated commits/run logs for the 196-candidate
   setup and later mode selection if this chronology is to support a held-out
   claim. SAL dev is prompt-unseen, but it is used for threshold calibration;
   it is not unused development data.
7. **FWS historical split dependency and file identity.** The manuscript's FWS
   loader `scripts/bench/nli_sal_fws.py` still reads the historical
   `data/scibert_prep/train.jsonl` to exclude overlapping text before shuffling.
   Keep a hash/manifest or frozen export of that exact input; do not substitute
   the newer Task 5b split. Equal byte lengths of FWS copies also do not prove
   identity: the manuscript correctly discloses that no direct comparison was
   made. A checksum comparison would resolve that remaining uncertainty.
8. **Human-study status and frozen sample.** The design and 135-item sample counts
   are documented in `docs/experiments/human_study_freeze.md`; this supports the
   design description, not the current completion status or any human result.
   Ask for the protected frozen key/hash and final report when available. I did
   not open rater answers/private keys or alter the pending-study statements.

## Code checks used for the wording edits

- `scripts/bench/nli_variants.py`: context contains the candidate and immediate neighbours.
- `scripts/bench/nli_sal_fws.py`: benchmark context windows, development thresholds, historical FWS split dependency.
- `scripts/bench/stage_c_rubric_val.py`: mode paragraph replacement, 400/600-character truncation, conditional validation retention/rejection.
- `scripts/bench/stage_c_sal40_modes.py`: scope adaptation and benchmark context construction.
- `scripts/bench/stage_c_modes_report.py`: exclusion of rows whose reference label is research_gap.
- `scripts/bench/stage_c_cost.py`: prompt/item text token counting and alternative/funnel ratios.
- `docs/experiments/stage_a_latex_sections.md`, `stage_c_nli_qwen.md`, and `human_study_freeze.md`: historical evidence and human-study design.

## Verification

- SHA-256 comparison against the pre-edit snapshot confirms that the abstract,
  introduction, Related Work, main/preprint wrappers, appendix,
  figures and generated table files are unchanged.
- Git diff confirms the bibliography and experiment source files are unchanged.
- All numeric-value occurrences in the edited sections are preserved (metric
  name F1 is excluded from this check); numerical results were not recomputed.
- Figure and table environments, including captions, are byte-for-byte unchanged
  after newline normalisation.
- Both manuscript variants were rebuilt and checked for unresolved citations,
  undefined references, oversized floats and overfull boxes; rendered pages
  were reviewed before delivery.

## Exact source changes

The patch below records every source edit relative to the files at the start
of this review. Lines beginning with `-` are the original; `+` is the replacement.

```diff
--- before/manuscript_mdpi/sections/03_materials_methods.tex
+++ after/manuscript_mdpi/sections/03_materials_methods.tex
@@ -15,3 +15,3 @@
 
-Given all sentences of a paper, the system returns the sentences in which the authors state a research gap. We distinguish three types of gap:
+Given a paper represented as a sequence of sentences, the system selects sentences in which the authors state a research gap. We distinguish three types of gap:
 \begin{itemize}
@@ -25,3 +25,3 @@
 
-We use three datasets (Table~\ref{tab:data}): two public benchmarks with human labels, each covering one gap type, and a corpus of 230 papers that we labelled for all three types with LLMs. All splits are by paper, so the sentences of one paper never appear in two splits.
+We use three datasets (Table~\ref{tab:data}): two public benchmarks with human labels, each covering one gap type, and a corpus of 230 papers that we labelled for all three types with LLMs. Within each dataset, splits are made by paper: all sentences from a paper belong to the same split.
 
@@ -46,3 +46,3 @@
 
-\paragraph{SAL.} SAL contains 200 RCT reports in which annotators marked the sentences that state a limitation; it is split into 120 training, 40 development and 40 test papers~\cite{lan2024automatic,salrepo}. We call the test split SAL-40 and the development split SAL dev. We score every sentence of each paper, not a section-filtered subset. To apply section rules in Stage~A, we recovered the section type of each sentence by matching it to the PubMed Central BioC version of its article. Sentences of fewer than three words were removed; none of the removed test sentences is a limitation. The training split is used only to train the supervised classifiers we compare against (Section~\ref{sec:comparison}).
+\paragraph{SAL.} SAL contains 200 RCT reports in which annotators marked the sentences that state a limitation; it is split into 120 training, 40 development and 40 test papers~\cite{lan2024automatic,salrepo}. We call the test split SAL-40 and the development split SAL dev. After the short-sentence removal described below, evaluation covers the full paper rather than only the sections selected by Stage~A. To apply section rules in Stage~A, we recovered the section type of each sentence by matching it to the PubMed Central BioC version of its article. Sentences of fewer than three words were removed; none of the removed test sentences is a limitation. The training split is used only to train the supervised classifiers we compare against (Section~\ref{sec:comparison}).
 
@@ -55,3 +55,3 @@
 \end{enumerate}
-The final corpus has 4462 gap sentences: 2234 research gaps, 1543 limitations and 685 future-work statements. The two labelling models agreed on whether a sentence is a gap with $\kappa = 0.778$~\cite{cohen1960kappa}; this measures agreement between models, not agreement with people. On the 40 SAL papers, the ensemble marked all but nine of SAL's 205 human-labelled limitations as a gap. Labelling cost about US\$36. We call this corpus the \emph{gold corpus} because its labels are our reference for three-type evaluation; since no person has checked them yet, all results on it are provisional. Papers were split into 162 training, 34 validation and 34 test papers. The test split has 664 gaps (220 limitations, 97 future work and 347 research gaps); 28 of its papers are from arXiv (527 gaps) and 6 are SAL papers (137 gaps).
+The final corpus has 4462 gap sentences: 2234 research gaps, 1543 limitations and 685 future-work statements. Agreement between the two labelling models on whether a sentence is a gap was $\kappa = 0.778$~\cite{cohen1960kappa}. This measures consistency between the models, not agreement with human annotations. On the 40 SAL papers, the ensemble marked all but nine of SAL's 205 human-labelled limitations as a gap. Labelling cost about US\$36. We call this corpus the \emph{gold corpus} because its labels are our reference for three-type evaluation; since no person has checked them yet, all results on it are provisional. Papers were split into 162 training, 34 validation and 34 test papers. The test split has 664 gaps (220 limitations, 97 future work and 347 research gaps); 28 of its papers are from arXiv (527 gaps) and 6 are SAL papers (137 gaps).
 
@@ -133,3 +133,3 @@
   \item \textbf{SAL:} only sentences in Discussion and Conclusion sections are kept. This removes 76.3\% of the sentences but keeps 203 of the 205 test limitations.
-  \item \textbf{Gold corpus:} no section rule is applied, because in arXiv papers many gaps are stated in the methods and experiments sections (Section~\ref{sec:localisation}). Stage~A only removes text that cannot be a gap statement, and we call it the \emph{junk filter}. It drops titles, tables, figure captions and abbreviation lists; competing-interest, author-contribution, funding and supplementary sections, where section types are known; bibliography-like strings; and malformed sentences (under 25 characters or six tokens, starting mid-word, containing dot leaders, or consisting mostly of non-letters).
+  \item \textbf{Gold corpus:} no section rule is applied, because in arXiv papers many gaps are stated in the methods and experiments sections (Section~\ref{sec:localisation}). Stage~A instead applies a set of exclusion rules, which we call the \emph{junk filter}. It drops titles, tables, figure captions and abbreviation lists; competing-interest, author-contribution, funding and supplementary sections, where section types are known; bibliography-like strings; and malformed sentences (under 25 characters or six tokens, starting mid-word, containing dot leaders, or consisting mostly of non-letters).
   \item \textbf{FWS:} there is no Stage~A. The corpus already consists of conclusion sections, and the junk filter, which rejects sentences that start in lower case, would remove almost all FWS text once it is lowercased.
@@ -137,7 +137,7 @@
 
-\paragraph{Stage B: zero-shot NLI classifier.} Stage~B selects candidate sentences with a classifier that needs no gap-labelled training data. We use an NLI cross-encoder, ModernBERT-large fine-tuned on NLI data~\cite{warner2024modernbert,modernbertnli}, as a zero-shot classifier~\cite{yin2019zeroshot}: the model reads a sentence (the premise) together with a short statement of the label (the hypothesis) and estimates whether the sentence entails the statement. There is one hypothesis per gap type: ``This is a limitation.'', ``This is future work.'' and ``This is a research gap.'' A sentence's score for a hypothesis is the entailment probability divided by the sum of the entailment and contradiction probabilities. On the gold corpus, a sentence becomes a candidate if its highest score over the three hypotheses reaches a threshold $\tau$, and its predicted type is the hypothesis with that highest score. On SAL only the limitation hypothesis is used, and on FWS only the future-work hypothesis. Inputs are truncated to 256 tokens. Section~\ref{sec:choices} describes how $\tau$ is set.
+\paragraph{Stage B: zero-shot NLI classifier.} Stage~B selects candidate sentences with a classifier that needs no gap-labelled training data. We use an NLI cross-encoder, ModernBERT-large fine-tuned on NLI data~\cite{warner2024modernbert,modernbertnli}, as a zero-shot classifier~\cite{yin2019zeroshot}: the model reads a sentence (the premise) together with a short statement of the label (the hypothesis) and estimates whether the sentence entails the statement. There is one hypothesis per gap type: ``This is a limitation.'', ``This is future work.'' and ``This is a research gap.'' For each hypothesis, the model's entailment and contradiction probabilities are normalised to sum to one; the normalised entailment probability is the candidate score. On the gold corpus, a sentence becomes a candidate if its highest score over the three hypotheses reaches a threshold $\tau$, and its predicted type is the hypothesis with that highest score. On SAL only the limitation hypothesis is used, and on FWS only the future-work hypothesis. Inputs are truncated to 256 tokens. Section~\ref{sec:choices} describes how $\tau$ is set.
 
-\paragraph{Stage C: LLM filter.} Stage~C asks an LLM to keep or drop each candidate. We use Qwen3.8-27B, self-hosted with vLLM~\cite{kwon2023vllm}; this model was not used to create any labels. Candidates are sent in reading order, 20 per request. For each candidate, the model receives its predicted type, the sentence (up to 400 characters) and, as context, the previous and next sentences of the same paper (up to 600 characters). It returns a keep or drop verdict for every candidate as JSON constrained by a schema. Decoding uses temperature 0, seed 0 and no reasoning mode.
+\paragraph{Stage C: LLM filter.} Stage~C asks an LLM to keep or drop each candidate. We use Qwen3.8-27B, self-hosted with vLLM~\cite{kwon2023vllm}; this model was not used to create any labels. Candidates are sent in reading order, 20 per request. For each candidate, the model receives its predicted type, the candidate sentence (up to 400 characters) and a context window containing that sentence and its immediate neighbours in the same paper. The context window is truncated to 600 characters. It returns a keep or drop verdict for every candidate as JSON constrained by a schema. Decoding uses temperature 0, seed 0 and no reasoning mode.
 
-The system prompt (Appendix~\ref{app:prompts}) defines the three gap types, lists the kinds of sentences to drop (results, method descriptions, contribution claims, background, citations, captions and similar), and ends with one instruction that sets the mode. Only this final instruction differs between the modes:
+The system prompt (Appendix~\ref{app:prompts}) defines the three gap types, lists the kinds of sentences to drop (results, method descriptions, contribution claims, background, citations, captions and similar), and includes one paragraph that sets the mode. Within a dataset, only this paragraph changes between the three modes:
 \begin{itemize}
@@ -155,3 +155,3 @@
   \item \textbf{NLI hypotheses.} We compared hypothesis variants (long definitions, short type names, a single combined hypothesis, adding the neighbouring sentences to the premise, and subtracting competing ``not a gap'' hypotheses) on one half of the gold validation papers and confirmed the choice on the other half. Short type names worked best; adding context or competing hypotheses made the results worse.
-  \item \textbf{Thresholds.} $\tau$ is set on development data only. It is the highest value at which recall, counting the positives already removed by Stage~A, is still at least 90\%. This gives $\tau = 0.354$ on the gold validation split, $\tau = 0.7286$ on SAL dev and $\tau = 0.1922$ on FWS dev. Fixing recall in this way lets Stage~B keep most gaps and leaves the removal of false positives to Stage~C. The stored values are rounded; at the rounded SAL value one development limitation falls below $\tau$, so SAL dev recall before Stage~C is 89.4\% instead of 90.0\%.
+  \item \textbf{Thresholds.} $\tau$ is set on development data only. We choose the highest threshold for which end-to-end recall after Stage~B is at least 90\%: positives removed by either Stage~A or Stage~B count as missed. This gives $\tau = 0.354$ on the gold validation split, $\tau = 0.7286$ on SAL dev and $\tau = 0.1922$ on FWS dev. Fixing recall in this way lets Stage~B keep most gaps and leaves the removal of false positives to Stage~C. The stored values are rounded; at the rounded SAL value one development limitation falls below $\tau$, so SAL dev recall before Stage~C is 89.4\% instead of 90.0\%.
   \item \textbf{Stage C model and prompt.} The LLM, the batch size, the use of context and the high-recall instruction were chosen during development on 196 candidates from gold test papers. We therefore also report gold results without these 196 sentences (Section~\ref{sec:rq2}). The balanced and high-precision instructions were written after reading the false positives of a high-recall run on gold test papers, and were then checked on the gold validation split before they were run on any test set. All three modes were run on every test set and all are reported, so no mode was selected by its test score.
@@ -173,3 +173,3 @@
 
-\paragraph{LLM input cost.} We estimate the number of LLM input tokens with the \texttt{cl100k\_base} tokenizer~\cite{tiktoken} on the prompts and items exactly as sent. Output tokens and the computation of Stages~A and~B are not counted. We compare the funnel with two alternatives that skip Stages~A and~B: one whole-paper call (the high-precision prompt followed by the numbered sentences of the paper) and judging every sentence in batches of 20 in the funnel's format.
+\paragraph{LLM input cost.} We use LLM input-token count as a proxy for cost. Counts are estimated with the \texttt{cl100k\_base} tokenizer~\cite{tiktoken}, applied to the prompt text and candidate items in the request format. Output tokens and the computation of Stages~A and~B are not counted. We compare the funnel with two alternatives that skip Stages~A and~B: one whole-paper call (the high-precision prompt followed by the numbered sentences of the paper) and judging every sentence in batches of 20 in the funnel's format.
 
--- before/manuscript_mdpi/sections/04_results.tex
+++ after/manuscript_mdpi/sections/04_results.tex
@@ -12,3 +12,3 @@
   \item \textbf{Stage B.} The NLI classifier raises precision by a factor of 2.7 to 3.6 and keeps 88--91\% of all positives, close to the 90\% recall that its threshold rule targets on development data.
-  \item \textbf{Stage C.} The LLM filter raises precision by a further 37 points on SAL-40, 28 on FWS and 63 on the gold corpus, and lowers recall by 6, 12 and 34 points, respectively. On the gold corpus, the lost recall falls mostly on research gaps and limitations (Section~\ref{sec:types}).
+  \item \textbf{Stage C.} The LLM filter raises precision by a further 37 percentage points on SAL-40, 28 on FWS and 63 on the gold corpus, and lowers recall by 6, 12 and 34 points, respectively. On the gold corpus, the lost recall falls mostly on research gaps and limitations (Section~\ref{sec:types}).
 \end{itemize}
@@ -72,3 +72,3 @@
 
-Two checks indicate that this ordering of the modes is not specific to the test data. First, on SAL dev, whose papers influenced no prompt, the modes behave as on SAL-40 with slightly lower values: high-precision mode reaches 64.3\% precision, 80.0\% recall and $F_1$ 0.713. Second, on the gold validation split, where the modes were checked before the balanced and high-precision test runs, high-precision mode kept 60.3\% of the gaps it received and removed 97.7\% of the non-gaps, compared with 94.1\% and 82.9\% in high-recall mode; precision was 90.4\% and 66.8\%, respectively.
+Two additional checks examine whether the increase in precision and decrease in recall across modes also occur outside the test sets. First, on SAL dev, whose papers influenced no prompt, the modes behave as on SAL-40 with slightly lower values: high-precision mode reaches 64.3\% precision, 80.0\% recall and $F_1$ 0.713. Second, on the gold validation split, where the modes were checked before the balanced and high-precision test runs, of the labelled gaps among the Stage~B candidates, high-precision mode retained 60.3\%, compared with 94.1\% in high-recall mode. Of the non-gap candidates, the two modes rejected 97.7\% and 82.9\%, respectively. Their output precision was 90.4\% and 66.8\%.
 
@@ -79,3 +79,3 @@
 
-On the gold corpus, the recall that high-precision mode gives up is lost mainly on research gaps and limitations (Figure~\ref{fig:types} and Table~\ref{tab:types}). Future-work recall barely changes across modes (82.5--86.6\%), whereas research-gap recall falls from 89.3\% in high-recall mode to 51.3\% in high-precision mode, and limitation recall falls from 79.5\% to 55.9\%. This matters for the pooled scores: research gaps make up about half of the gold labels and are the type the earlier stages retain best, so they dominate the scores that pool all types. When research-gap sentences are excluded from scoring, high-precision mode has the highest $F_1$ of the three modes (0.701, against 0.595 for balanced and 0.509 for high recall).
+On the gold corpus, the recall that high-precision mode gives up is lost mainly on research gaps and limitations (Figure~\ref{fig:types} and Table~\ref{tab:types}). Future-work recall barely changes across modes (82.5--86.6\%), whereas research-gap recall falls from 89.3\% in high-recall mode to 51.3\% in high-precision mode, and limitation recall falls from 79.5\% to 55.9\%. The overall scores combine all three types. Because research gaps account for about half of the gold labels and are retained most often by the earlier stages, their treatment strongly affects the overall result. When sentences labelled as research gaps are removed from the evaluation set, while limitation, future-work and negative sentences are retained, high-precision mode has the highest $F_1$ of the three modes (0.701, against 0.595 for balanced and 0.509 for high recall).
 
@@ -104,5 +104,5 @@
 
-The training-free NLI classifier is a competitive Stage~B. On SAL-40, the funnel built on it reaches almost the same $F_1$ as one built on a supervised PubMedBERT trained on SAL's own labels, although the supervised version is more precise. On the gold corpus, it performs better than or comparably to the alternatives we tested.
+The training-free NLI classifier is a competitive Stage~B. On SAL-40, the two funnels have similar $F_1$ in high-precision mode, although the funnel with supervised PubMedBERT as Stage~B is more precise. On the gold corpus, it performs better than or comparably to the alternatives we tested.
 
-\paragraph{Against a supervised classifier on SAL.} We ran the funnel in all three modes with PubMedBERT as Stage~B. Before Stage~C, PubMedBERT selects 391 candidates with 46.0\% precision and 87.8\% recall ($F_1$ 0.604); it was the best of the eight SAL-trained classifiers (SciBERT 0.553, DistilBERT 0.523, frozen encoders 0.371--0.457). After Stage~C in high-precision mode, it reaches 74.5\% precision (95\% CI 68.4--80.7), 78.5\% recall and $F_1$ 0.765. Table~\ref{tab:salcmp} compares the two classifiers mode by mode. With PubMedBERT, the funnel is 6.9 to 9.0 points more precise in every mode. In high-precision mode, however, the $F_1$ difference is small and not significant ($-0.021$, CI $-0.053$ to $+0.014$); in the other two modes, PubMedBERT's $F_1$ advantage is larger (0.035 and 0.059). The NLI classifier uses no SAL labels at all: its only SAL-specific setting is the threshold~$\tau$.
+\paragraph{Against a supervised classifier on SAL.} We ran the funnel in all three modes with PubMedBERT as Stage~B. Before Stage~C, PubMedBERT selects 391 candidates with 46.0\% precision and 87.8\% recall ($F_1$ 0.604); it was the best of the eight SAL-trained classifiers (SciBERT 0.553, DistilBERT 0.523, frozen encoders 0.371--0.457). After Stage~C in high-precision mode, it reaches 74.5\% precision (95\% CI 68.4--80.7), 78.5\% recall and $F_1$ 0.765. Table~\ref{tab:salcmp} compares the two classifiers mode by mode. With PubMedBERT, the funnel is 6.9 to 9.0 points more precise in every mode. In high-precision mode, however, the $F_1$ difference is small and not significant ($-0.021$, CI $-0.053$ to $+0.014$); in the other two modes, PubMedBERT's $F_1$ advantage is larger (0.035 and 0.059). The NLI model is not fitted on SAL labels. Its decision threshold~$\tau$ is selected using the SAL development labels.
 
@@ -127,3 +127,3 @@
 
-The funnel sends far fewer tokens to the LLM than judging every sentence, but only moderately fewer than a single whole-paper call (Table~\ref{tab:cost} and Figure~\ref{fig:cost}). Compared with judging every sentence, it sends 3.7 (FWS) to 13.1 (SAL-40) times fewer input tokens. Compared with one whole-paper call, the saving is 3.1 times on SAL-40, 2.0 times on FWS and 1.25 times on the gold corpus. The saving is small on arXiv papers for two reasons: Stage~B passes about one sentence in five, and each candidate is sent with its neighbouring sentences, so the funnel sends almost as much text as the whole paper. Whether the funnel is still preferable there depends on whether whole-paper prompting finds the same gaps, which the comparison in progress will show. The token counts of the other two modes differ from those of high-precision mode by less than 2\%, because only the final instruction changes.
+The funnel sends far fewer tokens to the LLM than judging every sentence, but only moderately fewer than a single whole-paper call (Table~\ref{tab:cost} and Figure~\ref{fig:cost}). Judging every sentence requires 3.7 times as many input tokens as the funnel on FWS and 13.1 times as many on SAL-40. A single whole-paper call requires 3.1 times as many as the funnel on SAL-40, 2.0 times as many on FWS and 1.25 times as many on the gold corpus. The saving is small on arXiv papers for two reasons: Stage~B passes about one sentence in five, and each candidate is sent with its neighbouring sentences, so the funnel sends almost as much text as the whole paper. These token counts alone do not establish which approach is preferable: a comparison of their extraction accuracy is still in progress. The token counts of the other two modes differ from those of high-precision mode by less than 2\%, because only the final instruction changes.
 
@@ -152,3 +152,3 @@
 
-The precision of the same output changes greatly with the reference labels. The right side of Table~\ref{tab:salcmp} scores the SAL-40 output against SAL's human labels and against our LLM labels. Of the 263 sentences returned in high-precision mode, SAL's annotators marked 174 as limitations, whereas the LLM ensemble marked 245 as limitations and 250 as a gap of some type. The two label sets disagree mainly on 77 sentences that are gaps according to the LLM labels but not according to SAL; only 12 returned sentences are negative under both. Precision is therefore 66.2\% against SAL and 93.2--95.1\% against the LLM labels. The true precision depends on which label set is right for these 77 sentences, and the human study described in Section~\ref{sec:quality} is designed to settle this.
+The precision of the same output changes greatly with the reference labels. The right side of Table~\ref{tab:salcmp} reports precision against the LLM labels; we compare those values with precision against the human labels for SAL-40 below. Of the 263 sentences returned in high-precision mode, SAL's annotators marked 174 as limitations, whereas the LLM ensemble marked 245 as limitations and 250 as a gap of some type. The two label sets disagree mainly on 77 sentences that are gaps according to the LLM labels but not according to SAL; only 12 returned sentences are negative under both. Precision is therefore 66.2\% against SAL and 93.2--95.1\% against the LLM labels. The true precision depends on which label set is right for these 77 sentences, and the human study described in Section~\ref{sec:quality} is designed to settle this.
 
--- before/manuscript_mdpi/sections/05_discussion.tex
+++ after/manuscript_mdpi/sections/05_discussion.tex
@@ -7,7 +7,7 @@
 
-\paragraph{RQ1: stage contribution.} The three stages have distinct roles. Stage~A is cheap: when section headings are informative, as in RCT reports, it removes three quarters of the text at almost no cost in recall. Stage~B largely determines the final recall, because Stage~C only removes candidates and cannot recover a gap that Stage~B missed; with our threshold rule, this ceiling was 88--91\%. Stage~C provides most of the gain in precision, between 28 and 63 points. Throughout, every reported gap remains a sentence the authors wrote.
+\paragraph{RQ1: stage contribution.} The three stages have distinct roles. Stage~A is cheap: when section headings are informative, as in RCT reports, it removes three quarters of the text at almost no cost in recall. Stage~B largely determines the final recall, because Stage~C only removes candidates and cannot recover a gap that Stage~B missed; with our threshold rule, recall after Stage~B was 88--91\%, setting the maximum recall that Stage~C could retain. Stage~C provides most of the gain in precision, between 28 and 63 points. Throughout, every reported gap remains a sentence the authors wrote.
 
-\paragraph{RQ2: precision control.} Changing a single instruction moves the filter along the precision--recall trade-off in the expected order on every evaluation set: SAL-40, SAL dev, FWS, and the gold test split as a whole and in its arXiv and biomedical parts. The same order appeared on the gold validation split before the balanced and high-precision modes were run on test papers. The funnel can therefore be adapted to its use. High precision suits idea generation, where every false gap costs a downstream model or a person time; high recall suits a reviewer who wants to see every candidate. On human labels, high precision costs no $F_1$: it is the best mode on SAL-40 and level with the others on FWS. On the LLM-labelled corpus, it does lose recall, mostly on research gaps. A plausible explanation is that the high-precision instruction asks the model to drop background and motivation (Appendix~\ref{app:prompts}), and authors often state open problems in exactly that role, when they motivate their own work.
+\paragraph{RQ2: precision control.} Changing a single instruction moves the filter along the precision--recall trade-off in the expected order on every evaluation set: SAL-40, SAL dev, FWS, and the gold test split as a whole and in its arXiv and biomedical parts. The same order appeared on the gold validation split before the balanced and high-precision modes were run on test papers. The funnel can therefore be adapted to its use. High precision suits idea generation, where every false gap costs a downstream model or a person time; high recall suits a reviewer who wants to see every candidate. On the human-labelled benchmarks, high-precision mode has the highest observed $F_1$ on SAL-40, while the FWS modes have similar $F_1$ values. On the LLM-labelled corpus, it does lose recall, mostly on research gaps. A plausible explanation is that the high-precision instruction asks the model to drop background and motivation (Appendix~\ref{app:prompts}), and authors often state open problems in exactly that role, when they motivate their own work.
 
-\paragraph{RQ3: classifier and cost.} A zero-shot NLI classifier with short hypotheses that name the gap type is a workable Stage~B. It needs no gap-labelled training data, works on both biomedical and computer-science text, and on SAL comes within 0.021 $F_1$ of a PubMedBERT trained on SAL's own labels. The supervised classifier is, however, 8.4 points more precise in high-precision mode, so a project that has in-domain labels should still use them. On cost, the funnel sends 3.7 to 13 times fewer tokens than judging every sentence, but its saving over a single whole-paper call is modest and almost disappears on arXiv papers. A whole-paper prompt could also include the mode instruction and return sentence numbers. On arXiv papers, the funnel is therefore worthwhile only if it extracts gaps more accurately than whole-paper prompting; the comparison in progress will show whether it does.
+\paragraph{RQ3: classifier and cost.} A zero-shot NLI classifier with short hypotheses that name the gap type is a workable Stage~B. It needs no gap-labelled training data, works on both biomedical and computer-science text, and on SAL comes within 0.021 $F_1$ of a PubMedBERT trained on SAL's own labels. The supervised classifier is, however, 8.4 points more precise in high-precision mode, so the small $F_1$ difference should not obscure the supervised funnel's precision advantage. On cost, the funnel sends 3.7 to 13 times fewer tokens than judging every sentence, but its saving over a single whole-paper call is modest and almost disappears on arXiv papers. A whole-paper prompt could also include the mode instruction and return sentence numbers. For arXiv papers, these input-token estimates alone provide little basis for preferring the funnel. The planned whole-paper comparison is needed to assess the accuracy trade-off.
 
--- before/manuscript_mdpi/sections/06_conclusions.tex
+++ after/manuscript_mdpi/sections/06_conclusions.tex
@@ -5,3 +5,3 @@
 
-On the human-labelled benchmarks, high-precision mode reached 66.2\% precision at 84.9\% recall for limitations in RCT reports (SAL-40) and 76.5\% precision at 75.5\% recall for future work in NLP conclusion sections (FWS), and held-out SAL papers gave similar results. On a 230-paper corpus with provisional LLM labels for three gap types, it reached 86.6\% precision at 57.4\% recall. The modes shifted precision and recall in the intended direction on every dataset. The training-free classifier came close to a supervised one on SAL in $F_1$, though not in precision. The funnel is clearly cheaper than judging every sentence, but only moderately cheaper than a single whole-paper call.
+On the human-labelled benchmarks, high-precision mode reached 66.2\% precision at 84.9\% recall for limitations in RCT reports (SAL-40) and 76.5\% precision at 75.5\% recall for future work in NLP conclusion sections (FWS), and held-out SAL papers gave similar results. On the test split of the 230-paper corpus, using provisional LLM labels for three gap types, it reached 86.6\% precision at 57.4\% recall. The modes shifted precision and recall in the intended direction on every dataset. The training-free classifier came close to a supervised one on SAL in $F_1$, though not in precision. The funnel requires substantially fewer estimated LLM input tokens than judging every sentence, with a smaller reduction relative to a single whole-paper call.
 
```
