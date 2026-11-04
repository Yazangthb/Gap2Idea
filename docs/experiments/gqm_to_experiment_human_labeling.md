# From GQM to a Fully-Fledged Experiment — Methodology Notes for the Human Labeling of Research Gaps

_Study-notes file. The left column of every section is the **course concept** (Innopolis
"Empirical Methods / Applied statistics and experiments" — EM Lectures 1–5 and ASE Labs 1–7);
the right column is **how it instantiates in our human gap-labeling validation study**
(`human_study_design.md` + the frozen `human_study_chosen_design.md`)._

> **One-line thesis of this file:** the human study is the *measurement / interpretation* phase of
> a GQM whose Goal is "are our gold's SAL-extras real gaps?". It is **not a treatment experiment**
> — it is a **blind, pre-registered diagnostic measurement study with hidden controls**. These
> notes walk the full course pipeline (GQM → hypotheses → measurement → scales → experiment design
> → validity) and, at each step, record the decision we actually made and *why the experiment
> taxonomy says that decision is the right one*.

---

## 0. The pipeline in one picture

```
GQM              →  Hypotheses        →  Measurement        →  Scales/goodness    →  Design           →  Validity
(Lec 1 / Lab 1)     (Lec 2 / Lab 2)      (Lec 3 / Lab 3)       (Lec 4 / Lab 4)       (Lec 5 / Lab 6-7)    (Lec 5)
What to study?      What would prove     How do we assign     Is the number        What arrangement     Could the number
Define Goal→Q→M     / disprove it?       numbers to the       type legal for       of raters/items/     be right for the
                    H0 / H1, α           labels? valid?       the stats we run?    controls?            wrong reason?
```

Course mantra (Lec 1, GQM phases): **Planning → Definition → Data collection → Interpretation.**
Three *levels*: **Goal (conceptual) · Question (operational) · Metric (quantitative)** — the gold
rule is *every metric must trace up to a question, every question up to the goal; collect the
smallest set of measures that still answers every question* (Quiz 1.3).

Our study sits in **Data collection + Interpretation**; the Planning/Definition were frozen on
2026-10-27 (file + sample key + SHA-256 hashed before any rater sees an item).

---

## 1. GOAL — the conceptual level (EM Lec 1, ASE Lab 1)

### 1.1 Course content
GQM defines a measurement program top-down on three levels (Lec 1 slide 31). A **Goal** is written
with the **goal template** (Lec 1 slides 33–34), five slots:

| Slot | Template prompt | Legal fillers |
|---|---|---|
| **Object / Purpose** | Analyze `<object>` | process, product, resource |
| **Purpose (why)** | for the purpose of `<…>` | characterize, evaluate, predict, motivate, improve |
| **Focus / Quality** | with respect to `<aspect>` | cost, correctness, reliability, … |
| **Perspective (who)** | from the point of view of `<stakeholder>` | user, customer, manager, developer, researcher |
| **Environment (where)** | in the context of `<…>` | the context factors that bound the measurement |

GQM+Strategies (Lec 1 slides 35–36) adds the *business* framing: **Object, Focus, Magnitude, Time
frame, Organizational scope, Constraints, Relations**. SMART goals (ASE Lab 1) are the same
discipline: Specific, Measurable, Achievable, Relevant, Time-bound.

### 1.2 Our instantiation
> **Goal (filled template):**
> **Analyze** our gap gold's *extra* `limitation` labels (the 227 sentences we label `limitation`
> on the 40 SAL papers that SAL's human gold does **not**),
> **for the purpose of** evaluating (validating),
> **with respect to** whether they are *genuine research gaps*,
> **from the point of view of** independent human raters (the researcher/evaluator),
> **in the context of** biomed/SAL papers judged from the sentence + its ±2-sentence context.

GQM+Strategies view of the same goal:

| Element | Value in our study |
|---|---|
| Object | The SAL-extras population (gold `limitation`, absent from SAL) |
| Focus | "Is it a real gap?" (truth of the label) |
| Magnitude | Show the true-gap share is **> 50%** (planning target 70%) |
| Time frame | One pooled blind labelling round |
| Organizational scope | 2 raters + 1 external tie-breaker, none an author |
| Constraints | Blind; no source/tier/SAL-membership shown; frozen before labelling |
| Relations | Feeds RQ2 (text-based gap mining); re-certifies the LLM-ensemble gold |

**Why this is the goal and not "measure the pipeline":** measuring funnel precision is a *second,
clearly-separated stratum* (Stratum B), deliberately pooled into the same labelling effort so the
test stays clean (TREC-style pooling). One Goal per test; the pipeline-precision estimate is a
secondary descriptive output, underpowered by design.

---

## 2. QUESTION → HYPOTHESIS — the operational level (EM Lec 2, ASE Lab 2)

### 2.1 Course content
Questions operationalize the goal (Lec 1). A testable study turns the question into a **statistical
hypothesis pair** (Lec 2 / Quiz 2):

- **Null hypothesis H₀** — the default "no effect / no improvement" claim, *includes equality*
  (e.g. `H₀: μ_choc ≤ μ_control`). A test **rejects H₀ or fails to reject** it; failing to reject
  does **not** prove H₀ true (Quiz 2.1).
- **Alternative H₁** — the research claim.
- **Type I error** = rejecting a *true* H₀ (false alarm); its probability is the significance level
  **α**, *chosen before looking at the data* (Quiz 2.2).
- **Type II error** = failing to reject a *false* H₀; **power = 1 − β** is the chance of detecting a
  real effect of a stated size.
- One-sided vs two-sided is decided a priori from the direction of the claim.

### 2.2 Our instantiation
> **Question:** What share of the SAL-extras do independent humans judge to be real research gaps,
> from the sentence + its context?
>
> **Hypotheses (frozen before any label exists):**
> **H₀: p ≤ 0.50  vs  H₁: p > 0.50**, one-sided, **α = 0.05**.
> **Power:** 80% at a true share of **p = 0.70** (effect 0.20) → exact binomial test needs **n ≈ 37**
> SAL-extras; we draw **40** for margin.

Mapping to the course error table:

| | H₀ true (extras ≤50% real) | H₀ false (extras >50% real) |
|---|---|---|
| **Reject H₀** (we claim "SAL under-labels") | **Type I** — prob. ≤ α = 0.05 (false claim) | ✅ correct |
| **Fail to reject** | ✅ correct | **Type II** — prob. β = 0.20 at p=0.70 |

**Course discipline we obey:** α and n fixed *before* labelling; the decision rule ("reject H₀ if
the weighted confirmed-gap share clears the power-simulated threshold **and** the validity gate
passes") is pre-specified; if we do **not** reject we report *"data do not show > 50%"*, never
*"H₀ accepted"* (Quiz 2.1 — failure to reject ≠ proof).

---

## 3. MEASUREMENT — assigning the numbers (EM Lec 3, ASE Lab 3)

### 3.1 Course content
**Measurement** = "the process by which numbers or symbols are assigned to attributes of entities
in the real world by clearly defined rules" (Fenton & Pfleeger; Lec 3 slide 6). A **measure** is the
resulting number/symbol.

**Representational theory of measurement** (Lec 3 slides 12–20): a measure is a *mapping* `M` from an
empirical relational system to a numerical one that **preserves the relevant relations** in both
directions — the **representation condition**:

```
empirical:  A  (taller than)  B     ⇔     numerical:  M(A) > M(B)
```

A **valid measure** is one that *satisfies the representation condition* (Lec 3 slide 17). Key stages
of formal measurement (Lec 3 slide 25): identify attribute → identify empirical relations → identify
corresponding numerical relations → define the mapping → **check the representation condition holds.**

### 3.2 Our instantiation
- **Entity:** a candidate sentence (in its ±2-sentence context).
- **Attribute:** "is a genuine research gap" (and, if yes, its *type*).
- **Rule (the clearly-defined mapping M):** a blind rater reads the sentence + context + the one-line
  paper topic + the three type definitions, and answers **gap: yes/no**, then **type ∈ {limitation,
  future_work, research_gap}**. The *label of record* is **majority-of-three** (2 raters + external
  tie-breaker on disagreements).
- **Empirical relation we must preserve:** "sentence s₁ is more clearly a gap than s₂." Our mapping is
  only **nominal** (gap / not-gap) + **nominal type**, so the representation condition we must honour
  is the weak one: *equivalence* (two sentences that are empirically the same kind of gap must get the
  same label) — see §4.
- **Representation-condition check = our validity gate + controls:** positive controls (high-confidence
  gold gaps) must map to "gap"; decoys (true negatives) must map to "not a gap." If the mapping fails
  to preserve these known empirical facts, the measure is invalid and the claim is void (≤ 2/12 decoys
  may be called "gap").

**Circularity note (Lec 3 "define the attribute carefully"):** the gold was itself labelled by
Opus + Sonnet. Human raters *replace* that machine mapping with an independent one — that is exactly
"improving the measure by improving understanding of the attribute" (Lec 3 slide 13). Raters must not
be the papers' authors.

---

## 4. SCALES & GOODNESS OF MEASUREMENT (EM Lec 4, ASE Lab 4)

### 4.1 Course content
A **scale** is the class of mapping linking empirical and numerical relations with specific
properties (Lec 4). The five scales, their admissible transformations, and the statistics each
*legally* supports (Lec 4 slides 9–11, Siegel–Castellan table):

| Scale | Defining relations | Admissible transform | Legal statistics |
|---|---|---|---|
| **Nominal** | equivalence (=, ≠) | any 1-to-1 relabel | mode, frequency, proportion; counts; χ², Krippendorff/κ |
| **Ordinal** | + order (<, >) | any strictly increasing | + median, percentile, Spearman/Kendall |
| **Interval** | + equal intervals | `Y = aX + b, a>0` | + mean, SD, Pearson r |
| **Ratio** | + true zero | `Y = aX, a>0` | + geometric mean, coeff. of variation, ratios ("twice as") |
| **Absolute** | counts (the count *is* the value) | `Y = X` only | everything |

**Valid measure (scale sense):** a statement whose **truth value does not change under an admissible
transformation** of the scale (Lec 4 slides 13–14). "Twice as many atoms" is valid (ratio); "twice
as black" is not (colour is not ratio).

Two orthogonal metric classifications (Lec 4 slides 15–17; Quiz 4.2):
- **Objective** (fixed rule, automatable, e.g. LOC) vs **Subjective** (needs human judgment, e.g.
  comfort rating — captures perception but risks rater bias/inconsistency).
- **Direct** (measured straight, one calculation) vs **Indirect** (derived/computed from other
  measures, or an *indicator* of a construct — inherits the component errors).

### 4.2 Our instantiation

| Our measure | Scale | Objective/Subjective | Direct/Indirect | Consequence for the stats we may run |
|---|---|---|---|---|
| gap: yes/no | **Nominal** (2 categories) | **Subjective** (human judgment) | **Direct** indicator of "is a gap" | only proportions/counts; the primary test is an **exact binomial on a proportion** — legal on nominal |
| gap type (lim/fw/rg) | **Nominal** (3 categories) | Subjective | Direct | mode/frequency per type; **no mean of type codes** |
| majority-of-three label | **Nominal** | aggregated subjective | Indirect (derived from 3 raters) | legal input to the binomial; inherits rater error → that is why we report agreement |
| inter-rater agreement | — (κ/α on nominal) | — | — | **Krippendorff's α** (nominal variant) + % agreement |

**Why this matters (Lec 4 slide 18 — "scales limit the operations"):** because the gap label is
**nominal + subjective**, we are *forbidden* from treating it as interval/ratio. That is the formal
reason the analysis is a **proportion test with a binomial CI**, agreement is **Krippendorff's α**
(not Pearson r), and the Rogan–Gladen correction is applied to a *share*, not a mean. Treating a
nominal subjective label as if it were ratio would be a **construct-validity failure** (Lec 5 §7:
"applying the mean to ordinal/nominal data").

**Goodness of the subjective measure:** because it is subjective, the known risk is rater
bias/inconsistency (Lec 4 slide 15). The course remedy is exactly our design: multiple independent
raters + a blind tie-breaker (reduce inconsistency), anchored definitions and shown context (reduce
bias), and decoys (detect a "label-everything-gap" rater).

---

## 5. THE STUDY AS AN EXPERIMENT — characteristics of a fully-fledged experiment (EM Lec 5)

### 5.1 Course content
**Experiment** = "a test under controlled conditions made to demonstrate a known truth or examine the
validity of a hypothesis" (Lec 5 slide 4). Core definitions (Lec 5 slides 6–8): **Treatment** (the
exposure whose effect is measured), **Subjects** (experimental units), **Observation** (the act of
measuring the DV), **Experimental design** (systematic manipulation of ≥1 variable holding others
constant).

**Population vs sample** (Lec 5 slides 9–10): population = all objects of interest; sample = the finite
subset studied. **Random sample** (equal selection probability → generalizable) vs **convenience
sample** (whoever is available). Note **random selection ≠ random assignment** (Lec 5 slide 12):
selection buys external validity; assignment buys internal validity.

**The six characteristics of a formal, fully-fledged experiment** (Lec 5 slide 11; Quiz 5.2):
1. **Random assignment** of subjects to groups,
2. **Control over extraneous factors**,
3. **Manipulation of the treatment conditions**,
4. **Outcome measures**,
5. **Group comparisons**,
6. **Assessment of threats to validity** (internal + external [+ construct]).

**Control toolkit for extraneous factors** (Lec 5 slides 13–18): pretests/posttests, **covariates**
(relate to DV not IV; removed by analysis of covariance), **matching** (equalize an outcome-driving
characteristic across groups), **homogeneous sampling** (pick units that vary little), **blocking**
(divide into subgroups, then randomize within each).

### 5.2 Our instantiation — honest mapping

Our study is a **validation / measurement study**, *not* a treatment experiment: there is **no
treatment applied to subjects** and no causal claim. The "subjects" are **sentences**, and the
"raters" are the **instrument**. Mapping the six characteristics:

| Characteristic | In a true experiment | In our study |
|---|---|---|
| Random assignment | subjects → treatment groups | **n/a** — no treatment; instead *items are drawn by a frozen systematic/balanced sample* and shown to raters in each rater's own random order |
| Control of extraneous factors | pretest/covariate/matching/blocking | **blinding** (hide source, tier, SAL-membership, control/decoy role); **stratification** of the draw by gap type & paper (= blocking on item covariates); balance rule uses covariates only, never the label |
| Manipulation of treatment | change IV levels | **none** — this is the key difference; see §6 |
| Outcome measure | the DV | **majority-of-three gap label** (§3–4) |
| Group comparison | treatment vs control | **extras vs hidden positive controls vs hidden decoys** (a *measurement* comparison, not a treatment contrast) |
| Threats to validity | internal/external/construct | fully enumerated in §7 |

> **Design-family verdict:** by the course taxonomy our study is **observational/descriptive with a
> validity-gate**, closest in spirit to a **one-group measurement with known-answer controls**. The
> "controls" and "decoys" play the role the course gives to a **control group** — they are the
> known-truth baseline against which the instrument (the raters) is checked. We borrow experimental
> *rigor devices* (randomized order, blinding, pre-registration, hidden controls, a priori α/power)
> without a treatment.

**Sampling (Lec 5 slides 9–10):** population = the **227 SAL-extras**; sample = a **frozen balanced
subset of 40** (chosen-design) drawn systematically, stratified by type & paper, hash-frozen. This is
a **probability sample of a bounded population**, so inference generalizes to *all 227 extras* — but
**not** beyond biomed/SAL (external-validity limit, §7).

---

## 6. EXPERIMENTAL DESIGN TAXONOMY — where our study sits, and the treatment-experiment we are *not* running (ASE Labs 6–7)

### 6.1 Course content — the design families and their notation
Notation (ASE Lab 6–7): **R** = randomization, **X** = treatment, **O** = observation/measurement;
a dashed line between rows = non-equivalent (non-randomized) groups.

Three design elements (Lab 6 slide 3): **Manipulation**, **Control**, **Randomization**.

| Family | Has manipulation? | Control group? | Randomization? | Validity |
|---|---|---|---|---|
| **Pre-experimental** | partial | maybe not | **no** | lowest |
| **Quasi-experimental** | yes | yes | **no** (non-equivalent groups) | medium |
| **True experimental** | yes | yes | **yes** | high |
| **Factorial** | ≥2 factors crossed | yes | yes | high + interactions |

Canonical diagrams (Lab 6–7):
- **One-shot case study** (pre): `A: X — O` — effect cannot be determined.
- **One-group pretest–posttest** (pre): `A: O₁ — X — O₂` — looks like a true experiment but has
  **no control group**; threatened by history, maturation, testing, instrumentation.
- **Static group comparison** (quasi): two groups, treatment vs control, **no randomization** → groups
  not guaranteed equivalent (threats: selection, mortality).
- **Pretest–Posttest Control Group** (true): `R O₁ X O₂ / R O₃ — O₄`; effect `E = (O₂−O₁) − (O₄−O₃)`.
- **Posttest-Only Control Group** (true): `R X O₁ / R — O₂`; adequate only when groups are large
  enough that the pretest is unnecessary (pretest-impossible/too-costly cases).
- **Factorial**: ≥2 independent variables (factors) crossed, to read **main effects + interactions**.

### 6.2 Our instantiation
Our validation study is **not** in the X/O causal families — there is no X. Its closest analogue is a
**one-group measurement with known-answer anchors**:

```
Items drawn (frozen, stratified)  —  blind rating by 2 raters (random order)  —  majority-of-three label
with hidden positive controls (must → "gap")  and  hidden decoys (must → "not a gap")   [the validity gate]
```

**Where the experiment taxonomy becomes useful for us** is in designing the *next* study — the point
where we would turn validation into a genuine **treatment experiment** about the labelling process
itself. Candidate designs we could pre-register later:

- **True experiment on context size** — IV `X` = amount of context shown (sentence-only vs
  ±2-sentence vs full-paragraph vs full-paper). `R` raters/items to context conditions, DV = agreement
  with an expert panel. Reads whether "sentence + local context" (our current limitation, §7) biases
  judgments.
- **Factorial design** — Factor A = context size (2 levels) × Factor B = rater expertise
  (biomed-literate vs general). A **2×2 factorial** reads the **main effect** of each and, crucially,
  the **interaction** (does expertise matter *more* when context is thin?). This directly tests two of
  our stated limitations at once.
- **Posttest-only control group** would fit a between-rater treatment (e.g. "guideline + examples" vs
  "guideline only") when rater groups are large enough that a calibration pretest is unnecessary —
  mirroring our deliberate **"no calibration / no practice round"** choice.

> **Takeaway:** the *current* human study is a rigor-maximized **descriptive measurement study**
> (validate the labels). The experiment taxonomy is the roadmap for the *follow-up* studies that would
> establish *why* raters agree or disagree — and it names the exact design (2×2 factorial) that would
> retire the "context" and "expertise" limitations in one shot.

---

## 7. THREATS TO VALIDITY (EM Lec 5 slides 20–36) — applied

Course recap (Lec 5 slide 36): **Internal** = is the cause→effect real, or some other factor?
**External** = does it generalize to other settings/people/measures? **Construct** = are we measuring
the construct we claim (right analysis for the scale)?

### 7.1 Internal validity (is the "share of real gaps" number caused by the extras actually being real, or by an artefact of the instrument?)

| Course threat (Lec 5 slides 23–28) | Risk in our study | Control in the design |
|---|---|---|
| **Selection** | the drawn 40 extras unrepresentative of the 227 | probability + **balanced** draw, covariates < 0.1 std diff; hash-frozen |
| **Instrumentation** | raters drift / score differently | anchored type definitions; same blind page; majority-of-three |
| **Testing / order effects** | earlier items shift later judgments | **per-rater random order** |
| **Mortality** | a rater quits mid-set | full overlap (2 raters all items) + tie-breaker; hard-stop on incomplete files |
| **Experimenter bias** | an author nudges contested items | **no author decides**; external blind tie-breaker; leftovers → "not a gap" |
| **"Yes"-leaning rater** | labelling everything "gap" fabricates the claim | **12 hidden decoys + gate** (≤ 2/12 may be "gap") makes "yes" and "no" count equally |

### 7.2 External validity (how far does the result generalize? — Lec 5 slides 29–32)

- **Interaction of selection & domain:** result generalizes to the **227 biomed/SAL extras**, *not* to
  arXiv/CS gold (no SAL reference there) — an explicit, accepted limit; arXiv is a later direct-labelling round.
- **Reactive setting:** judgments use **sentence + local context, not the full paper** — a deliberate
  scope limit; a full-paper round is deferred (and is exactly the context-size experiment of §6.2).
- **Rater/stratum confounding:** who rates which item is not fully crossed with stratum — an accepted
  limitation of the 2+1 design (the pooled binomial is unaffected by *who* rates *which* item).

### 7.3 Construct validity (are we measuring "is a real research gap"? — Lec 5 slides 33–35)

- The construct is "genuine research gap," operationalized as the majority-of-three yes/no on nominal
  scale. **Construct risk:** a rater may conflate "limitation" with "prior-work gap" or with a mere
  hedge. **Control:** three anchored **type definitions** shown with every item; **convergent** check =
  positive controls (known gaps) should read "gap"; **divergent** check = decoys (known non-gaps)
  should read "not a gap." These are the course's convergent/divergent validity tests (Lec 5 slide 34).
- **Scale-construct check (Lec 4 link):** we only ever run proportion/agreement statistics on the
  nominal label — we never average type codes — so we avoid the textbook construct error of "applying
  the mean to ordinal/nominal data" (Lec 5 slide 33).
- **GQM as construct-validity device (Lec 5 slide 35):** the course explicitly says *"one approach to
  assure construct validity is the GQM model."* Our whole chain (§1→§4) is that device: the goal
  template forces the construct ("real gap"), the question fixes the measure, the scale fixes the legal
  statistics — so construct validity is built in, not bolted on.

---

## 8. THE FULL EXPERIMENTATION PLAN (the operational protocol, each step tagged to its course concept)

**Planning & Definition (frozen 2026-10-27 — Lec 1 phases; Lec 2 pre-registration):**
1. Goal written with the template (§1); Question + **H₀/H₁, α=0.05, power 80% @ p=0.70** (§2). ✔ frozen.
2. Measure + rule defined, scale fixed as nominal (§3–4). ✔
3. Sample frame = 227 extras; draw = **40** balanced/stratified; **+ controls/decoys** (below). Freeze
   the sample key + **SHA-256** before any rater sees an item (Lec 5 internal validity: selection). ✔ designed, **not yet drawn**.

**Pooled strata (the "groups" being compared — Lec 5 group comparison):**

| Stratum | n (chosen design) | Role (course mapping) |
|---|---|---|
| **A. SAL-extras** | 40 | the sample under test (the "treatment of interest") |
| **B. pipeline positives** (funnel+Stage C, gold `none`) | 30 | secondary: funnel true precision (separate, clean) |
| **C. positive controls** (high-conf gold gaps) | 12 | known-"gap" anchor → convergent/instrument check |
| **D. decoys** (true negatives) | 12 | known-"not-gap" anchor → divergent check + **validity gate** |
| (E. SAL-missed, optional) | 0–10 | gold recall hole |
| **Total distinct** | **~94** | ≈ 210 ratings (2× + ~20 tie-breaks) |

**Data collection (Lec 1 phase 3; Lec 5 instrument + order controls):**
4. Blind per-rater page (extend `data/gold/annotate.html`): rater ID + sample hash, decoy role hidden,
   per-rater random order, autosave, hard-stop on incomplete, XLSX export.
5. 2 raters rate all ~94 items (~45–55 min each); external tie-breaker rates ~20 disagreements.

**Interpretation / Analysis (Lec 2 test; Lec 4 legal statistics; pre-specified before labels exist):**
6. **Gate first:** decoys labelled "gap" ≤ 2/12, else the claim is void (representation-condition
   check, §3; divergent validity, §7.3).
7. **Primary test:** weighted proportion of extras labelled "gap" by majority-of-three, with its
   one-sided **95% lower bound**; **reject H₀ iff** it clears the power-simulated threshold **and** the
   gate passes (nominal-legal, §4).
8. **Robustness:** Rogan–Gladen correction from control sensitivity + decoy specificity — *reported,
   never used to rescale the primary estimate.*
9. **Agreement:** **Krippendorff's α** (gap yes/no and type) + % agreement, pre-adjudication.
10. **Report discipline (Lec 2):** if not rejected → "data do not show > 50%", never "H₀ accepted".
    Secondary per-type/per-paper shares are descriptive and underpowered.

**Scripts to build (mirrors the lab/E5 scaffolding):**
- `scripts/dataset/human_study_sample.py` — balanced draw of A–D(+E) + private key + sample SHA-256.
- blind per-rater `annotate.html` variant.
- `scripts/dataset/human_study_analyze.py` — gate + one-sided binomial + Rogan–Gladen + Krippendorff α;
  rejects incomplete/foreign files.

---

## 9. Concept → study crosswalk (quick reference)

| Course concept | Lecture/Lab | Our study's instance |
|---|---|---|
| GQM goal template | Lec 1 s33–34 / Lab 1 | "Analyze SAL-extras … to evaluate … are they real gaps … from raters' view … in biomed context" |
| GQM "reduce the measures" | Quiz 1.3 | one primary label (gap y/n) + type; no redundant metrics |
| H₀/H₁, α, power, Type I/II | Lec 2 / Quiz 2 | H₀ p≤0.5 vs H₁ p>0.5, α=0.05, 80%@0.70, n≈37→40 |
| Representation condition / valid measure | Lec 3 s17,25 | controls+decoys = the preservation check on the label mapping |
| Scales → legal statistics | Lec 4 s9–11,18 | nominal subjective → binomial proportion + Krippendorff α; no means |
| Objective/subjective, direct/indirect | Lec 4 s15–17 | gap label = subjective/direct; majority = indirect (derived) |
| 6 experiment characteristics | Lec 5 s11 | we have outcome+comparison+validity; no treatment/assignment → measurement study |
| Random selection ≠ assignment | Lec 5 s12 | probability sample of 227 (ext. validity); no assignment (no treatment) |
| Control toolkit (blocking/matching) | Lec 5 s13–18 | stratified-by-type/paper draw = blocking; blinding = extraneous-factor control |
| Design families + O/X/R | Lab 6–7 | ours ≈ one-group + known-answer controls; 2×2 factorial = the planned follow-up |
| Internal/External/Construct threats | Lec 5 s20–36 | §7 tables (selection, instrumentation, order, mortality; domain; construct via GQM) |

---

## Sources
- **EM Lecture 1 / ASE Lab 1** — Introduction & the GQM model (goal template, GQM+Strategies, SMART, SWOT).
- **EM Lecture 2 / ASE Lab 2** — Hypothesis testing (H₀/H₁, Type I/II, α) [via Quiz 2 solutions].
- **EM Lecture 3 / ASE Lab 3** — Measurement: definition, representational theory, representation condition.
- **EM Lecture 4 / ASE Lab 4** — Measurement scales, admissible transforms, valid measures, metric types, goodness.
- **EM Lecture 5 / ASE Labs 6–7** — Experimental design: definitions, population/sample, 6 characteristics,
  control toolkit, design families (pre/quasi/true/factorial, O/X/R), threats to validity.
- **Lecture_Quizzes1-6_Short_Solutions.pdf** — worked GQM→hypothesis→measurement→experiment chain.
- **Project:** `docs/experiments/human_study_design.md`, `human_study_chosen_design.md`;
  `data/gold/disagreement_pools.json`, `data/gold/funnel_v2_candidates*.json`, `data/gold/annotate.html`.

_Notes compiled 2026-11-02 from the EM/ASE course decks + quiz solutions, applied to the frozen
human gap-labeling validation design._
