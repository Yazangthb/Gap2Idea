# Human study preparation and proposed analysis amendment

Status: preparation only; no participant has been recruited or rated through this work. Institutional approval or an explicit exemption and consent requirements must be determined by the institution before recruitment.

## Blinding and the published sample

On 16 November 2026, the originally frozen key and its labels were made publicly accessible on GitHub branch `human-study-materials` at commit `528e319`. The original protocol assumed the key and labels were not public. The old sample, pages, and SHA-256 are retained as evidence; do not silently treat the publication as if it had not happened. The organizer should document who had access and whether any potential raters saw it. If any rater could have seen the key or labels, or the team cannot defend blinding, design and freeze a new sample before recruitment. A new sample needs a separate private key, a new seed and sample hash, updated eligibility exclusions, and a dated protocol amendment. It must not be presented as the original frozen sample. A private local directory alone is not a secure distribution method: send raters only their blind page.

The current `HUMAN_STUDY_INSERT.md` describes the original sample and cannot be applied unchanged to a replacement. Determine which sample is approved first. Neither the original nor this amendment establishes that any human ratings have been completed.

Using the original manifest's eligibility counts, excluding every previously sampled item leaves at most B1: 24 eligible for a target of 30, S2: 29 for a target of 40, and S3: zero for a target of 11. This arithmetic assumes no other eligibility changes; a fresh draw excluding those public items cannot preserve the old targets. Agree and document an amended design and statistical analysis if excluding them. Alternatively, the team and institution must decide whether the original sample remains defensibly blinded for raters who have not seen the public information. Do not claim that publishing the key automatically proves actual rater exposure, or that the original blinding assumptions still hold without a documented assessment.

## Proposed supplementary analysis (freeze before ratings)

Keep the existing sample, primary tests and gate unchanged if the original study remains appropriate. Original-population weights assume that excluded items behave like eligible items within each stratum. Report a supplementary point range by extrapolating the observed sample share to the eligible population and assigning every excluded item first negative and then positive. Report supplementary 95% outer bounds by inverting exact hypergeometric tests for the eligible finite populations, applying a Bonferroni adjustment across strata, then assigning excluded items to the extreme outcomes. These bounds remain non-degenerate for unanimous sample results unless a whole eligible stratum has been observed. They are not a correction for clustering within papers or nonresponse. They do not replace the prespecified Wald-based primary test or the exact S2 test. If a new design is used, record any changes and verify these calculations apply to the new sampling method before use.

## Draft participant information and consent: institution review required

Purpose: independent assessment of sentences extracted from published research papers. Participants will judge whether a highlighted sentence states an author-stated research gap and, if so, its type, with adjacent context shown. The original design estimates 65–75 minutes per rater; revise this if the sample changes. Participation is voluntary; a participant may decline without penalty. The study organizer must fill in the contact person, institution, approval or exemption reference, any compensation, handling and retention of identifiable information, withdrawal window, and whether and how anonymized ratings may be shared. Do not promise anonymity or a right to withdraw data after anonymization unless confirmed by the institution. The rater should confirm they have read the approved information and consent before receiving a study page.

## Required decisions before recruitment

1. Obtain the institution's written decision on review or exemption and its required consent process.
2. Decide whether the public-key exposure requires a replacement sample; record the reasoning and the actual exposure.
3. Freeze the chosen sample, protocol, analysis amendment and applicable deviations before collecting answers.
4. Recruit independent raters who have not seen the key, existing labels or system outputs, subject to the approved protocol.
