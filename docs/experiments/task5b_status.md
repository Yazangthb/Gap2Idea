# Task 5b completion audit - 2026-11-06

**SAL and updated FWS runs are complete and verified. Task 5b as a whole is not
finished.** arXiv runs, full encoder comparisons, faithful published-method
reproductions and paired funnel comparisons remain incomplete.

The data-ready remainder now has a tested launcher:
`python scripts/bench/run_task5b_remaining.py --run`. See the
[remaining-run instructions](task5b_remaining_runs.md) for individual stages,
actual encoder smoke validation and the external files still needed. Nothing in
that launcher pushes Git or reruns the completed SAL/FWS baselines.

## Verified completed runs

- SAL: `runs/task5b/sal-full-01/sal/`; 4 requested methods; 40 papers / 7968
  sentences / 205 positives; completion marker present.
- FWS: `runs/task5b/fws-current-gold-full-01/fws/`; 5 requested methods; 209
  papers / 1506 sentences / 220 positives; completion marker present.
- The current implementation passed all 17 offline tests again during this audit.

For all 9 method/dataset results, the audit checked:

1. Every requested method has frozen settings, predictions and final metrics.
2. Input and prepared dataset hashes, frozen-settings hashes, saved model/score
   artifact hashes and full-test prediction hashes match their manifests.
3. Every validation/test paper ID, sentence index, exact sentence/hash and gold
   target matches the frozen data, with no missing or duplicate prediction rows.
4. Saved binary decisions equal score >= frozen threshold. The selected threshold
   and validation metrics reproduce from saved validation predictions; fixed
   rules remain deterministic. Every setting was frozen before test scoring.
5. Confusion counts, P/R/F1/macro F1 and all 2000-resample paper bootstrap CIs
   reproduce from the saved test predictions. NLI score arrays match the exported
   prediction scores. No models were rerun on the test set.
6. Saved source hashes match an exact committed source snapshot: SAL `c0543af`,
   FWS `a7dfd3e`. The SAL run legitimately predates the FWS-only source revision.

Machine-readable evidence: [task5b_audit_20261106.json](task5b_audit_20261106.json).
The consolidated result table is in [baseline_methods.md](baseline_methods.md).

## What the current results show

On updated FWS, TF-IDF/LR has the highest point estimate among the five completed
methods: positive F1 .748, macro F1 .854. Typed NLI reaches .708 positive F1;
the original broad NLI hypothesis reaches .434. Both attempted NLI variants
remain in the table. Overlapping marginal CIs do not by themselves establish a
paired significance result, and no claim of a statistically significant winner
is made here.

On SAL, the section-rule proxy has positive F1 .472, TF-IDF/LR .422, and broad
NLI .423. Binary NB has zero true positives (two false positives), so F1=0 is a
completed negative result, not a missing run.

FWS thresholds .05 for binary NB and .99 for the Zhang adaptation lie at the
registered validation-grid boundaries. Their selection reproduced correctly;
do not widen the grid or tune settings in response to the test scores. Any later
calibration/threshold study needs a separately declared validation-only protocol.

## Proposal coverage and remaining tasks

| Proposal requirement | Implemented / available | Completed evidence | Remaining action |
|---|---|---|---|
| 1. TF-IDF/LR and binary NB on SAL/FWS/arXiv | Runnable adapters and models | SAL + FWS complete | Run arXiv |
| 2. Kilicoglu limitation rules | Structural lexical port and separately named section proxy | SAL proxy complete | Obtain independent paragraph structure for structural mode; obtain arXiv LaTeX sections and run limitation-only extension. No FWS rule run is required |
| 3. ModernBERT NLI alone | Pinned model; broad and typed hypotheses | SAL broad complete; FWS broad + typed complete | Run arXiv-only evaluation; do not borrow the earlier mixed-gold .614 headline |
| 4. SciBERT / BiomedBERT fine-tuning | Generic binary fine-tune code, safetensors save/load; tiny-model training test passed | No full new encoder runs. Legacy SAL encoder decisions exist in the cached fixture | Run both encoders on arXiv and SciBERT on FWS; import cached SAL vectors with their actual training/threshold provenance and CIs, or run separately registered SAL fine-tunes |
| 5a. Lan PubMedBERT + EDA/PBA | External frozen-prediction adapter only | No faithful reproduction | Obtain original recipe/augmentation implementation (`h100_allmodels.py` or equivalent) and training-only augmentation provenance; implement/reproduce it or explicitly report inability and a qualified closest baseline |
| 5b. Zhang ngram/chi2/NB | Train-only held-out raw-text adaptation; separate CV diagnostic code | FWS adaptation complete | Obtain aligned original lemmas and reproduce clean/global-fit CV diagnostic variants. Current adaptation is not the published balanced lemmatized protocol; diagnostic code combines vocabulary/IDF and chi2 exposure |
| Paired F1 differences against funnel | Import/identity-check and paired paper bootstrap implemented | N/A for SAL, FWS and arXiv | Obtain complete sentence-level funnel predictions with provenance on matching test inputs; import and compare. Aggregate F1 values are insufficient |
| Unified results and skipped reasons | Consolidated table + this checklist | Nine completed rows, all with P/R/F1 and CIs | Fill missing rows/differences as their evidence arrives |

## Data readiness

The new gold upload unblocks arXiv inputs. During this audit its dataset was
validated and locked at `data/task5b/prepared-current-gold-v1/arxiv/`:

| Split | Papers | Sentences | Any-gap positives |
|---|---:|---:|---:|
| Train | 134 | 40276 | 2624 |
| Validation | 28 | 10087 | 600 |
| Test | 28 | 10388 | 527 |

ArXiv test positives comprise 152 limitations, 73 future-work sentences and 302
research-gap sentences. Limitation-only rules must use the 152-positive target,
not compare their F1 directly with any-gap detector F1.

The next baseline run is prepared but was **not started by the audit**:

```powershell
python scripts/bench/task5b.py run --datasets arxiv --data-dir data/task5b/prepared-current-gold-v1 --methods tfidf_lr,binary_nb,modernbert_nli,modernbert_nli_typed --batch-size 4 --bootstrap 2000 --run-dir runs/task5b/arxiv-current-gold-full-01
```

FWS uses the user-approved `fws_current_gold_v1` protocol (current-gold TRAIN text
exclusions and lowercase inputs). Matching historical test counts do not certify
the old test membership. Use exactly matching full predictions for funnel pairing;
the old Yandex .707 headline cannot fill the new difference column.

## Publication / Git state

The completed SAL evidence is already tracked on `Task_5b` at `c0543af`. The
current-gold inputs and FWS preparation changes are committed locally as
`a030d0d` and `a7dfd3e`; they have not been pushed in this session. Full FWS output
is currently under the ignored run directory and is not yet committed/pushed.
Preserve the full FWS run and explicitly stage it when publishing its results;
do not force-add all failed smoke directories. This audit changes documentation
only and leaves the completed SAL/FWS experiment files unchanged.
