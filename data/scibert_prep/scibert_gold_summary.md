# SciBERT-FT on our 10 gold papers (3-class drop-in)

Model: allenai/scibert_scivocab_uncased fine-tuned 2ep, lr=3e-05, bs=24, threshold=0.5.
Training data: same self-distilled+ACL set as the bge+logreg head (/home/yazan/gap2idea/data/scibert_prep/train.jsonl).
| stage | total | per paper | drop |
|---|---|---|---|
| full body | 3407 | 341 | — |
| → Stage A slice | 623 | 62 | −82% (free) |
| → SciBERT gaps (Stage B) | 49 | 4.9 | −92% of slice |
| → +Stage C (Qwen2.5-3B-Instruct) | 22 | 2.2 | −55% of Stage B |

**Vs gold (Stage B only):** recall = 10/19 = **0.526**; precision_floor = **0.245**.
**Vs gold (after Stage C):** recall = 6/19 = **0.316**; precision_floor = **0.273**.

## bge+logreg baseline (no SciBERT) — from earlier runs
Stage A+B (bge+logreg, hybrid): 62 preds, recall 0.526, precision_floor 0.194 (F1 ~0.283).


## 2309.09902  ·  gold gaps: 1

- Stage A: 198 → 70 (−65%)
- Stage B (SciBERT): 70 → 7 (−90% of slice)
- Stage C (Qwen/Qwen2.5-3B-Instruct): 7 → 1 (−86%)

| type | source | section | C-kept | gold? | gap sentence |
|---|---|---|---|---|---|
| limitation | model | limitations | ✗ | — extra | We demonstrated that fine-tuned Llama 2 language |
| limitation | model | limitations | ✗ | — extra | Task without relying on traditional linguistic fea- |
| limitation | rule+model | limitations | ✓ | — extra | as the cues were provided in Subtask 2, the roles We did not study risks that may or may [...] |
| limitation | model | limitations | ✗ | ✅ 2309.09902::g1 | It leveraged when our fine-tuned large language models are used |
| limitation | model | limitations | ✗ | — extra | Instead, the generated outputs are processed |
| limitation | model | limitations | ✗ | — extra | Table 3 shows the final results of our submissions ited. |
| limitation | model | limitations | ✗ | — extra | We recommend security testing if our trained |

## 2507.03922  ·  gold gaps: 2

- Stage A: 109 → 33 (−70%)
- Stage B (SciBERT): 33 → 4 (−88% of slice)
- Stage C (Qwen/Qwen2.5-3B-Instruct): 4 → 1 (−75%)

| type | source | section | C-kept | gold? | gap sentence |
|---|---|---|---|---|---|
| limitation | model | limitations | ✗ | — extra | See Appendix F for further analysis. |
| limitation | model | limitations | ✗ | ✅ 2507.03922::g1 | This work focuses on English-language datasets |
| limitation | rule | limitations | ✓ | ✅ 2507.03922::g1 | and assumes the availability of a KB, namely |
| limitation | model | limitations | ✗ | ✅ 2507.03922::g2 | English or in domains lacking a comprehensive KB |

## 2510.04514  ·  gold gaps: 5

- Stage A: 183 → 114 (−38%)
- Stage B (SciBERT): 114 → 11 (−90% of slice)
- Stage C (Qwen/Qwen2.5-3B-Instruct): 11 → 5 (−55%)

| type | source | section | C-kept | gold? | gap sentence |
|---|---|---|---|---|---|
| limitation | model | limitations | ✗ | — extra | See Appendix L.3 for further details. |
| future_work | rule+model | limitations | ✓ | — extra | remaining challenges and areas for future improve- |
| future_work | rule | limitations | ✓ | — extra | restricted to single charts; future work will driven visual self-verification mechanism fur- |
| limitation | model | limitations | ✗ | — extra | Our ICL examples are textual rather than mul- model to internally evaluate tool sufficiency |
| limitation | model | limitations | ✗ | — extra | While ChartAgent performs strongly on the |
| limitation | rule | limitations | ✓ | — extra | etc.) adds latency and cost due to the agen- radar plots, which are affected by depth dis- |
| limitation | model | limitations | ✗ | — extra | We advise against using this framework or |
| future_work | rule+model | limitations | ✓ | — extra | Future work includes on-the-fly responsibly released for research purposes only. |
| limitation | model | limitations | ✗ | — extra | Finally, since ChartAgent is designed to research. |
| limitation | model | limitations | ✗ | — extra | We abide by their terms of use. |
| limitation | rule | limitations | ✓ | — extra | is currently no standard method for quan- |

## 2511.13548  ·  gold gaps: 1

- Stage A: 333 → 108 (−68%)
- Stage B (SciBERT): 108 → 12 (−89% of slice)
- Stage C (Qwen/Qwen2.5-3B-Instruct): 12 → 7 (−42%)

| type | source | section | C-kept | gold? | gap sentence |
|---|---|---|---|---|---|
| limitation | model | limitations | ✗ | — extra | To address these limitations, we generating outputs that deviate from ethical or legal [...] |
| limitation | model | limitations | ✗ | — extra | To address these limitations, we propose F DAN, a evaluation approach. |
| limitation | model | limitations | ✗ | — extra | In this setting, human experts actively |
| future_work | model | limitations | ✗ | — extra | Together, these works form the foundation for lies in mutation diversity and shallow [...] |
| limitation | rule+model | limitations | ✓ | — extra | AutoDAN-HGA and GCG suffer from three core limitations: |
| limitation | rule | limitations | ✓ | — extra | can bypass the safety alignment mechanisms of and in- prone to false positives or negatives. |
| limitation | rule+model | limitations | ✓ | — extra | process by evaluating the quality and relevance of mutated A critical limitation of prior [...] |
| future_work | rule | future_work | ✓ | — extra | jailbreak effectiveness that inform future research on jailbreak also varies—models with [...] |
| future_work | rule | future_work | ✓ | — extra | vulnerabilities in aligned LLMs requires a multi-layered de- filtering details remain [...] |
| limitation | rule | future_work | ✓ | — extra | fense that integrates training-time hardening, runtime safe- models, the causes of cross- [...] |
| limitation | rule+model | limitations | ✓ | — extra | (i.e., whether the model attempts to comply with an unsafe limitations of prior [...] |
| limitation | model | limitations | ✗ | — extra | Third, enhancing the Extensive experiments on benchmark datasets and real- |

## 2002.09564  ·  gold gaps: 2

- Stage A: 365 → 105 (−71%)
- Stage B (SciBERT): 105 → 4 (−96% of slice)
- Stage C (Qwen/Qwen2.5-3B-Instruct): 4 → 1 (−75%)

| type | source | section | C-kept | gold? | gap sentence |
|---|---|---|---|---|---|
| limitation | model | limitations | ✗ | — extra | Societal impact and Limitations: For some of our experi- |
| limitation | model | future_work | ✗ | ✅ 2002.09564::g2 | We leave other tasks such as detec- |
| future_work | rule | future_work | ✓ | — extra | have insufﬁciently regularized the models and/or did not tion and segmentation for future work. |
| limitation | model | future_work | ✗ | — extra | This observations suggests that either transferability |

## 2003.01908  ·  gold gaps: 2

- Stage A: 199 → 27 (−86%)
- Stage B (SciBERT): 27 → 3 (−89% of slice)
- Stage C (Qwen/Qwen2.5-3B-Instruct): 3 → 1 (−67%)

| type | source | section | C-kept | gold? | gap sentence |
|---|---|---|---|---|---|
| limitation | model | future_work | ✗ | — extra | To the best of our knowledge, all existing input transformation based |
| future_work | model | future_work | ✗ | ✅ 2003.01908::g2 | Finding methods that can train denoisers to close |
| future_work | rule+model | future_work | ✓ | ✅ 2003.01908::g2 | the gap between our method and Cohen et al. ( 2019 ) remains a valuable future direction. |

## 2102.04998  ·  gold gaps: 2

- Stage A: 794 → 37 (−95%)
- Stage B (SciBERT): 37 → 3 (−92% of slice)
- Stage C (Qwen/Qwen2.5-3B-Instruct): 3 → 1 (−67%)

| type | source | section | C-kept | gold? | gap sentence |
|---|---|---|---|---|---|
| future_work | rule | future_work | ✓ | — extra | Lemma 4.4 in the next step t + 1 . |
| future_work | model | future_work | ✗ | ✅ 2102.04998::g2 | Analysis of architectures such as Residual Networks and Transformers would be a potentially |
| limitation | model | future_work | ✗ | — extra | We thank the anonymous reviewers for alerting us to a mistake in an earlier version of this |

## 2208.03805  ·  gold gaps: 0

- Stage A: 306 → 47 (−85%)
- Stage B (SciBERT): 47 → 0 (−100% of slice)
- Stage C (Qwen/Qwen2.5-3B-Instruct): 0 → 0 (−100%)

| type | source | section | C-kept | gold? | gap sentence |
|---|---|---|---|---|---|

## 2211.01962  ·  gold gaps: 3

- Stage A: 533 → 56 (−89%)
- Stage B (SciBERT): 56 → 3 (−95% of slice)
- Stage C (Qwen/Qwen2.5-3B-Instruct): 3 → 3 (−0%)

| type | source | section | C-kept | gold? | gap sentence |
|---|---|---|---|---|---|
| future_work | rule+model | future_work | ✓ | ✅ 2211.01962::g1 | the scope of this paper, and we leave this as future work. |
| future_work | rule+model | future_work | ✓ | ✅ 2211.01962::g2 | For future work, it would be interesting to understand the tightness of GEC by establishing |
| future_work | rule+model | future_work | ✓ | ✅ 2211.01962::g3 | In addition, for algorithm design, it would be interesting to develop poste- |

## 2511.03443  ·  gold gaps: 1

- Stage A: 387 → 26 (−93%)
- Stage B (SciBERT): 26 → 2 (−92% of slice)
- Stage C (Qwen/Qwen2.5-3B-Instruct): 2 → 2 (−0%)

| type | source | section | C-kept | gold? | gap sentence |
|---|---|---|---|---|---|
| future_work | rule | tail | ✓ | ✅ 2511.03443::g1 | For future studies, we are interested in developing |
| future_work | rule | tail | ✓ | — extra | We would like to express our gratitude to Dr. Yitian Qian and Dr. Lianghai Xiao for [...] |
