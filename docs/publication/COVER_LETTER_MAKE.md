# Cover letter for MAKE (draft)

Fill in every item in [square brackets] before sending. MDPI requires the two statements marked
**(required)** word for word. Names of suggested reviewers go into the submission system, not into this letter.

---

[Date]

Dear Editors of *Machine Learning and Knowledge Extraction*,

We submit the manuscript **"Scalable Extraction of Author-Stated Research Gaps: Towards Automated Idea
Generation"** by [final author list] for consideration as an Article.

**What the paper is about.** Research papers state their own limitations, future work and open problems.
Collected across many papers, these statements are a natural input for automated idea generation, but they
are rare and scattered through long texts. We ask how to extract them with a large language model (LLM)
without asking the model to read every paper in full. Our answer is a three-stage funnel: rules remove
unlikely text, a zero-shot natural-language-inference classifier proposes candidate sentences, and the LLM
only verifies these candidates. Every output is a sentence the authors wrote, and one instruction sets the
trade-off between precision and recall.

**Why it matters.** The paper gives a controlled comparison between this funnel and whole-paper prompting
with the same LLM at three model sizes, in accuracy and in measured cost. The funnel helps most where LLMs
are weakest: with small models on long papers. It also compares the funnel with published sentence-level
methods on identical test sentences, and it shows that precision measured against LLM-made labels cannot be
read as precision against human judgement. [After the human study: add one sentence with its main result.]
We report the limits of the evidence openly, including settings that were developed on test papers and
labels that are not yet validated by people.

**Why MAKE.** The work combines machine learning (zero-shot classification, LLM verification) with
knowledge extraction from scientific text, and it puts weight on reproducibility: the code, the prompts,
every LLM verdict behind the reported numbers and scripts that rebuild the evaluation data are public at
[repository URL and tag, e.g. https://github.com/Yazangthb/Gap2Idea/tree/submission-v1].

**Statements.**

- **(required)** We confirm that neither the manuscript nor any parts of its content are currently under
  consideration for publication with or published in another journal.
- **(required)** All authors have approved the manuscript and agree with its submission to MAKE.
- A preprint of an earlier version is available on Zenodo: [DOI]. [Delete this item if no preprint was posted.]
- This manuscript has not been submitted to an MDPI journal before. [Confirm, or give the earlier manuscript ID.]
- Generative AI was used in the study and in preparing the manuscript. Its use is described in Section 3.7
  and in the Acknowledgments, as the journal's instructions require.

Thank you for considering our work.

Sincerely,

[Corresponding author name]
[Affiliation and address]
[E-mail]
