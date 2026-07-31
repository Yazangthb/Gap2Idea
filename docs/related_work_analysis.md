# Related work: gap / future-work / limitation extraction — analysis

> Deep-research sweep (5 search angles → 18 primary sources → 90 claims → 25
> adversarially verified). The automated synthesis step failed on an API outage,
> so this report is synthesised by hand from the verified claim set.
> **Confidence tags:** `✓✓` confirmed 3-0 by independent verifiers · `✓` 2-0 ·
> `~` from a primary source but verification aborted on the outage (treat as
> *likely, unverified*).

---

## 1. The landscape (closest work first)

| System (venue, year) | Task | Method | Data | Headline number | Per-paper LLM? |
|---|---|---|---|---|---|
| **RCT self-acknowledged-limitations** ([PMC11807350](https://pmc.ncbi.nlm.nih.gov/articles/PMC11807350/), ~2024) | Limitation **extract+filter** | keyword slice of candidate sentences → **fine-tuned PubMedBERT** binary filter | RCT/biomed | **F1 0.821** (P .75 / R .91) `~` | **No** — scaled to ~12k articles, no per-doc LLM `~` |
| **Zhang et al.** ([arXiv 2212.13860](https://arxiv.org/abs/2212.13860) / [J. Informetrics 2022](https://www.sciencedirect.com/science/article/abs/pii/S1751157722001262)) | Future-work **recognise + classify** | 2 stages: Naive Bayes (binary) → **SciBERT** (6-way) | human-labelled NLP/ACL, 6 types | recognise **Macro-F1 90.7%** `✓✓`; classify **wF1 72.6%** `✓✓` | No |
| **Hu & Wan** ([arXiv 1507.02140](https://arxiv.org/abs/1507.02140), 2015) | Future-work extract + classify | **regex** extractor → 4-way classifier | scientific articles | "high precision & recall" `✓✓` | No |
| **FutureGen** ([arXiv 2503.16561](https://arxiv.org/abs/2503.16561), 2025) | Future-work **generation** (RAG) | regex high-recall → **LLM filter** → RAG generate | ACL/NeurIPS | — | **Yes** (extract+gen+judge per paper) `✓✓` |
| **LimGen** ([arXiv 2403.15529](https://arxiv.org/abs/2403.15529), 2024) | Limitation **generation** | harvest **mandated "Limitations" section** as gold → BART/PEGASUS/T5/Llama-2 + DPR (MiniLM) | **4068 ACL papers** `~` | — | Yes (generative) `~` |
| **BAGELS** ([arXiv 2505.18207](https://arxiv.org/abs/2505.18207), 2025) | Limitation dataset | regex "limitation" + ScienceParse + zero-shot LLM, + OpenReview/PeerJ reviews | ACL/NeurIPS/PeerJ `~` | extract ROUGE-1 ~.86, ~95% faithful `~` | Yes (build-time) |
| **LimTopic** (Al Azher, [JCDL 2024](https://dl.acm.org/doi/pdf/10.1145/3677389.3702612)) | Limitation **generation** | generate Limitations section from other sections | ACL | — | Yes `~` |
| **FWS location study** ([arXiv 2405.20785](https://arxiv.org/abs/2405.20785), 2024) | Manual analysis | — | manual, 129 FWS | — | — |
| **GAPMAP** ([arXiv 2510.25055](https://arxiv.org/abs/2510.25055) / [code](https://github.com/UCDenver-ccp/GAPMAP), 2025) | Knowledge-gap mapping | LLM | biomedical | — | Yes |

---

## 2. The methodological consensus — and it is exactly our funnel

Every extraction-style system (as opposed to pure-generation) uses the **same
two-stage shape Gap2Idea uses**:

> **high-recall structural/lexical retrieval → high-precision classifier filter.**

- Hu & Wan: regex extract → classify `✓✓`.
- Zhang et al.: recognise (binary) → classify (type) `✓✓`.
- FutureGen states it outright: *"Regex-based string matching … shows a high
  recall, so we further filtered these sentences using LLMs to improve [precision]"* `✓✓`.
- RCT/PubMedBERT: keyword slice → BERT filter `~`.

**So our architecture is the field standard, not a one-off.** Where systems
differ is only in **what plays the Stage-B filter**: a classical model
(Naive Bayes), a fine-tuned encoder (SciBERT / PubMedBERT), or an **LLM**
(FutureGen, GAPMAP). FutureGen explicitly pays the per-paper-LLM cost and even
trims input to "abstract + top-3 sections" *to reduce API cost* `✓✓` — which is
precisely the cost wall Gap2Idea's cheap Stage B is designed to avoid.

---

## 3. The closest analog validates us — and gives a target number

The **RCT self-acknowledged-limitations** system ([PMC11807350](https://pmc.ncbi.nlm.nih.gov/articles/PMC11807350/)) is almost
identical to Gap2Idea: it **slices candidate limitation sentences from the
abstract/discussion/limitation sections by keyword cues, then filters with a
fine-tuned PubMedBERT**, reaching **F1 0.821 (P 0.75 / R 0.91)** and beating a
rule-only baseline (F1 0.800) `~`. It then **scaled to ~12,000 articles with no
per-paper LLM** `~`.

Take-aways for us:
1. Our Stage A → Stage B design is a published, working recipe; **F1 ≈ 0.82 is a
   realistic target** for the in-slice gap-vs-not filter.
2. It used a **domain-pretrained BERT** (PubMedBERT) as the filter — direct
   support for trying **SciBERT** on our scientific text.
3. Rule-only (0.800) was only marginally below the BERT filter (0.821) — i.e.
   the classifier earns its keep but **rules alone are a strong baseline**,
   matching what we see (rules carry most of our precision).

---

## 4. The hard problems — the literature hit every one we did

Our independent findings are corroborated:

- **Future-work / limitation are concentrated in terminal sections** — Discussion
  (63), dedicated Future-Work sections (36), Conclusion (7), Limitations (6) —
  **but some sit in non-obvious places (Results, Intro) "at higher risk of being
  overlooked"** `✓`. This is exactly our Stage-A terminal prior *and* our
  documented miss of mid-paper limitations.
- **Authors conflate future work with limitations**, and in a 60-paper manual
  study **>50% of stated limitations cannot be discerned directly from the text**
  `~`. This is our gold-contamination / "is it actually a gap" problem, named in
  the literature as a genuine discourse-level difficulty.
- **Fine-grained typing is intrinsically hard.** Binary detection is easy
  (Naive Bayes Macro-F1 **90.7%** `✓✓`), but multi-class typing tops out low even
  with a domain encoder: **SciBERT wF1 72.6%** `✓✓` on 6 future-work types;
  **PubMedBERT F1 0.49** on 24 limitation sub-types / ~0.70 on 15 `~`, with low
  annotator agreement (α 0.30–0.45) `~`. **This explains our results precisely:**
  future-work (cue-rich, near-binary) is easy; limitation discrimination is the
  hard tail the whole field struggles with.
- **FWS are often vague** — 5/129 too ambiguous to categorise `✓` — limiting
  downstream usefulness regardless of extractor.

---

## 5. Datasets: mostly domain-locked

- **Future-work, NLP/ACL:** Zhang et al. (human, 6 types) `✓✓`; Hu & Wan.
- **Limitations, ACL (mandated section):** LimGen (4068 papers, section-as-gold)
  `~`; BAGELS (+ peer-review commentary) `~`.
- **Limitations, biomed/RCT:** PMC11807350 (PubMedBERT) `~`.
- **Knowledge gaps:** GAPMAP (biomed); IPBES (biodiversity); Sci-Challenges (COVID).

**None target arXiv AI/ML/math**, and all are domain-locked. This confirms our
decision to **build our own small gold** — there was nothing off-the-shelf to
reuse. (Cross-domain transfer of the *future-work* classifier from ACL data is
plausible; limitation patterns transfer less well across domains.)

---

## 6. Where Gap2Idea sits

| | Gap2Idea | Field |
|---|---|---|
| Architecture | regex terminal slice → rules + small classifier, LLM audit only | **same two-stage pattern** `✓✓` |
| Cost | no per-paper LLM | FutureGen/GAPMAP pay per paper; classifier systems don't |
| Stage-A recall | 0.84 (0.91 future-work) | location studies confirm terminal prior + mid-paper misses `✓` |
| Future-work | strong | "easy" task across the field (NB 90.7%) `✓✓` |
| Limitations | weak (data-limited) | **hard for everyone** (PubMedBERT .49 fine; α .30) `~` |

We are **methodologically on the SOTA path**; our weak spot (limitations) is the
field's weak spot, and is data- rather than architecture-bound — consistent with
our own BERT vs logreg experiment (both ≈ equal, both stuck on limitations).

---

## 7. The fix for limitations — literature-grounded recipe

Our distant-supervision attempt over-fired because it labelled *every* section
sentence a gap. The literature points to a cleaner version:

1. **Harvest the MANDATED "Limitations" sections of post-2023 *ACL / NeurIPS
   papers as distant-supervision positives** — LimGen did exactly this to get
   **4068 papers** of reliable, *authors'-own* limitations `~`, and BAGELS adds
   reviewer-stated ones `~`. These are far cleaner than arbitrary
   conclusion sentences because a mandated "Limitations" heading is, by
   construction, the authors' self-critique. **This is the single highest-value
   next step** and needs no LLM credits (the sections are free text in the ACL
   Anthology).
2. **Use a domain-pretrained encoder** (SciBERT) for the filter, per PubMedBERT's
   precedent `~`.
3. **Scale the (weak-)labelled set to thousands** + validate a sample, as RCT did
   (12k articles, 250-sample check) `~` — augmentation/scale, not a bigger model,
   is what moved the needle.
4. **Target F1 ≈ 0.82** for in-slice gap-vs-not `~`; accept that **fine-grained
   limitation *typing* will stay ~0.5–0.7** — that's the field ceiling, so keep
   the binary "is-it-a-gap + coarse type" framing rather than fine sub-types.

---

## 8. Sources
Primary (verified): [2212.13860](https://arxiv.org/abs/2212.13860),
[1507.02140](https://arxiv.org/abs/1507.02140),
[2503.16561](https://arxiv.org/abs/2503.16561),
[J.Informetrics 2022](https://www.sciencedirect.com/science/article/abs/pii/S1751157722001262),
[2405.20785](https://arxiv.org/abs/2405.20785).
Primary (sourced, verification aborted on outage):
[PMC11807350](https://pmc.ncbi.nlm.nih.gov/articles/PMC11807350/),
[LimGen 2403.15529](https://arxiv.org/abs/2403.15529),
[BAGELS 2505.18207](https://arxiv.org/abs/2505.18207),
[LimTopic JCDL'24](https://dl.acm.org/doi/pdf/10.1145/3677389.3702612),
[GAPMAP 2510.25055](https://arxiv.org/abs/2510.25055).
