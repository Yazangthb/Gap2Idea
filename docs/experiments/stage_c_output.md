# Stage C — LLM precision filter (results)

Backend: local (Qwen/Qwen2.5-1.5B-Instruct).  40 LLM judgments over 10 papers (~4.0/paper).

| | predictions | gold matched | recall | precision (floor) |
|---|---|---|---|---|
| **before** (Stage A+B) | 62 | 10 | 0.526 | 0.194 |
| **after** (+Stage C) | 30 | 8 | 0.421 | 0.333 |

Stage C dropped **32** predictions; recall 0.526→0.421, precision-floor 0.194→0.333.

## What Stage C dropped (✗ = real gold gap lost; rest = false positives removed)
- [✓ FP removed] (limi/limi) We demonstrated that fine-tuned Llama 2 language
- [✓ FP removed] (limi/limi) Task without relying on traditional linguistic fea-
- [✗ LOST gold 2309.09902::g1] (limi/limi) It leveraged when our fine-tuned large language models are used
- [✓ FP removed] (limi/limi) By using nor read the generated texts produced by our mod-
- [✓ FP removed] (limi/limi) We recommend security testing if our trained
- [✓ FP removed] (limi/limi) How do entity embeddings affect perfor- ( Xiao et al. , 2024 ) and train KPR using it as
- [✓ FP removed] (limi/limi) We evaluate two baselines for comput- the base model.
- [✓ FP removed] (limi/limi) We also test embeddings ex- tailed settings are provided in Appendix C .
- [✓ FP removed] (limi/limi) To address these shortcomings, we draw inspira-
- [✓ FP removed] (limi/limi) See Appendix L.3 for further details.
- [✓ FP removed] (limi/limi) VideoAgent2 ( Zhi et al. , 2025 )), we report
- [✓ FP removed] (limi/limi) We advise against using this framework or
- [✓ FP removed] (limi/limi) (e.g., stronger OCR or segmentation models). et al. , 2024 ).
- [✓ FP removed] (limi/limi) We abide by their terms of use.
- [✓ FP removed] (limi/limi) Together, these works form the foundation for lies in mutation diversity and shallow [...]
- [✓ FP removed] (limi/limi) This threat model reflects real-world scenarios where ma- of F ORGE DAN, including [...]
- [✓ FP removed] (limi/limi) To address these limitations, F ORGE DAN introduces a
- [✓ FP removed] (futu/limi) Third, enhancing the Extensive experiments on benchmark datasets and real-
- [✓ FP removed] (limi/futu) When evaluating automated jailbreak generation methods, it is for their valuable [...]
- [✓ FP removed] (limi/limi) Finn, “Direct preference optimization: Your language model is
- [✓ FP removed] (limi/limi) The models learned with this data could
- [✓ FP removed] (futu/futu) Conclusion and Proposed Guidelines [4] Ekin D Cubuk, Barret Zoph, Dandelion Mane, [...]
- [✓ FP removed] (futu/futu) Our extensive experiments suggest a strong need for a policies from data. arXiv [...]
- [✓ FP removed] (futu/futu) We are the ﬁrst, to the best of our knowledge, to
- [✓ FP removed] (limi/futu) This is only a ﬁrst stab at provably robustifying pretrained classiﬁers, and we [...]
- [✗ LOST gold 2102.04998::g2] (futu/futu) Analysis of architectures such as Residual Networks and Transformers would be a [...]
- [✓ FP removed] (limi/futu) We thank the anonymous reviewers for alerting us to a mistake in an earlier version [...]
- [✓ FP removed] (futu/futu) (extended Fatou’s lemma for weakly convergent probabilities; cp.
- [✓ FP removed] (futu/futu) 2019 ) with a Hellinger-distance-based loss function, which is different from the [...]
- [✓ FP removed] (limi/futu) We then adopt the conditional posterior sampling from Dann et al. ( 2021 ) but
- [✓ FP removed] (futu/futu) When specialized to particular examples of MDP, POMDP, PSR models, GEC
- [✓ FP removed] (futu/futu) Furthermore, in addition to a new complexity

Dropped 32: 30 false positives removed, 2 real gold gaps lost.