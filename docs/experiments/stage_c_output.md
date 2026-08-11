# Stage C — LLM precision filter (results)

Backend: api (openai/gpt-4o-mini).  85 LLM judgments over 10 papers (~8.5/paper).

| | predictions | gold matched | recall | precision (floor) |
|---|---|---|---|---|
| **before** (Stage A+B) | 111 | 12 | 0.632 | 0.108 |
| **after** (+Stage C) | 42 | 12 | 0.632 | 0.286 |

Stage C dropped **69** predictions; recall 0.632→0.632, precision-floor 0.108→0.286.

## What Stage C dropped (✗ = real gold gap lost; rest = false positives removed)
- [✓ FP removed] (futu/futu) An extension to separable reﬂexive Banach spaces, with a focus on [...]
- [✓ FP removed] (futu/futu) For separable Banach spaces, [3] establishes epiconvergence for empirical [...]
- [✓ FP removed] (futu/futu) (extended Fatou’s lemma for weakly convergent probabilities; cp.
- [✓ FP removed] (futu/futu) Most existing works such as Russo and Van Roy 8 ( 2014 ) study the Bayesian regret, [...]
- [✓ FP removed] (futu/futu) These extensions allow us to capture much more general interactive decision making [...]
- [✓ FP removed] (futu/futu) We then adopt the conditional posterior sampling from Dann et al. ( 2021 ) but also [...]
- [✓ FP removed] (futu/futu) When specialized to particular examples of MDP, POMDP, PSR models, GEC can be [...]
- [✓ FP removed] (futu/futu) When restricted to special GEC GEC MDP, POMDP, and PSR examples, such a regret bound [...]
- [✓ FP removed] (limi/limi) We ones reproduced in this study.
- [✓ FP removed] (limi/limi) With these observations, we recommend that AL etc).
- [✓ FP removed] (limi/limi) We techniques such as RA and SWA, though it is likely that provide the index sets [...]
- [✓ FP removed] (limi/limi) We do believe that with and test sets.
- [✓ FP removed] (limi/limi) This observations suggests that either transferability Ohta, and Masanori Koyama.
- [✓ FP removed] (futu/limi) Optuna: A nextgeneration experiments should be conducted to assess if the active [...]
- [✓ FP removed] (futu/limi) Conclusion and Proposed Guidelines [4] Ekin D Cubuk, Barret Zoph, Dandelion Mane, [...]
- [✓ FP removed] (limi/limi) To this end, we [5] Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V recommend [...]
- [✓ FP removed] (futu/limi) Alternatively, experiments should be re- [8] Xavier Gastaldi.
- [✓ FP removed] (limi/limi) Goodfellow, Jean PougetAbadie, Mehdi Mirza, Bing performed using a common evaluation [...]
- [✓ FP removed] (limi/futu) However, these defenses are, in general, not scalable to large models (e.g. [...]
- [✓ FP removed] (limi/futu) A few works afterwards were able to provide formal guarantees for randomized [...]
- [✓ FP removed] (futu/futu) Although, in theory, randomized smoothing does not require any training of the [...]
- [✓ FP removed] (limi/futu) However, there are two key differences between this work and ours: 1) Lecuyer et al. [...]
- [✓ FP removed] (limi/futu) Indeed, the denoising autoencoder in this prior work was largely intended as a [...]
- [✓ FP removed] (futu/futu) Many such defenses have been proposed (but later broken) in previous works ( Guo et [...]
- [✓ FP removed] (futu/futu) We are the ﬁrst, to the best of our knowledge, to show that we can provably defend [...]
- [✓ FP removed] (futu/futu) Finally we ensure that the third part of the inductive hypothesis holds.
- [✓ FP removed] (limi/limi) If a puncPrompting our finetuned models was a twostep process.
- [✓ FP removed] (limi/limi) To do score as proposed for opinion role labeling ( Johansthis, we prepended the [...]
- [✓ FP removed] (limi/limi) Several postprocessing steps were necessary to We used the same finetuned Llama 2 [...]
- [✓ FP removed] (limi/limi) If the models’ out2023 Shared Task 1 – a cues model to identify cues put did not [...]
- [✓ FP removed] (limi/limi) Instead, the generated outputs are processed we were able to analyze the impact of [...]
- [✓ FP removed] (futu/futu) KPR detects entities in the input text using a simIn this paper, we address this [...]
- [✓ FP removed] (futu/futu) Since both the proposing a simple extension to dense retrievers entity embeddings [...]
- [✓ FP removed] (limi/futu) KPR is intentionally designed cludes many queries with lessfrequent entities, as [...]
- [✓ FP removed] (futu/futu) R N +1 × D : without KPR extensions and consistently improves
- [✓ FP removed] (limi/limi) This is feasible because every EQ notably larger on EQ in all settings, as it [...]
- [✓ FP removed] (limi/limi) BERT and DPR PELT , especially on KPR with the dictionarybased linker outperqueries [...]
- [✓ FP removed] (limi/limi) Compared to BERT forms its ReFinEDbased variant on average.
- [✓ FP removed] (limi/limi) (§ 2 ) and may detect incorrect or noisy entities.
- [✓ FP removed] (limi/limi) We observe that KPR tends as it extracts entities only when confident.
- [✓ FP removed] (limi/limi) See Appendix F for further analysis. theshelf retrievers, we select the bgebase [...]
- [✓ FP removed] (limi/limi) Since bgebase is already trained ing entity embeddings: random initialization and on [...]
- [✓ FP removed] (limi/limi) This iteragraphical elements), a setting where even stateoftive reasoning process [...]
- [✓ FP removed] (limi/limi) We advise against using this framework or • Vision Tools and Query Handling.
- [✓ FP removed] (limi/limi) While MLLM agents to automate critical chartor imagemanually designed, our vision [...]
- [✓ FP removed] (limi/limi) We abide by their terms of use.
- [✓ FP removed] (limi/limi) Model card addendum: Claude 3.5 This document is not intended as investment rehaiku [...]
- [✓ FP removed] (limi/limi) To address these limitations, we generating outputs that deviate from ethical or [...]
- [✓ FP removed] (limi/limi) To mitigate LLMbased classifier to jointly assess model compliance and these risks [ [...]
- [✓ FP removed] (limi/limi) Our evaluation demonstrates techniques such as Supervised FineTuning (SFT) [ 11 ] [...]
- [✓ FP removed] (limi/limi) Nevertheless, alignment safeguards are not unbreakable.
- [✓ FP removed] (limi/limi) Other studies introduced manipulations such as semantic insensitivity in fitness [...]
- [✓ FP removed] (limi/limi) Such strategy textual perturbations—including character, word and practices are [...]
- [✓ FP removed] (limi/limi) Existing studies relied on manual prompts, later evolvrithms to evolve prompts from [...]
- [✓ FP removed] (limi/limi) Together, these works form the foundation for lies in mutation diversity and shallow [...]
- [✓ FP removed] (futu/limi) Moreover, Liu et al. extended this line of work with Adversary AutoDANTurbo [ 16 ], [...]
- [✓ FP removed] (limi/limi) Other research has investigated - no training data accessible universal or [...]
- [✓ FP removed] (limi/limi) The search process is realized To overcome these challenges, F DAN introduces a ORGE [...]
- [✓ FP removed] (limi/limi) For instance, as mentioned earlier, homophone semantic embeddings provide both [...]
- [✓ FP removed] (limi/limi) To address these limitations, F ORGE DAN introduces a semanticaware fitness function [...]
- [✓ FP removed] (limi/limi) In practice, different encoders can be adopted, such is randomly sampled from the [...]
- [✓ FP removed] (limi/limi) Next we turn to the semantic fitness measurement mechaE.
- [✓ FP removed] (limi/limi) Then we describe the datasets, encompassing both widely adopted benchmarks and a [...]
- [✓ FP removed] (limi/limi) By integrating multistrategy text mutafences is essential.
- [✓ FP removed] (limi/limi) Third, enhancing the Extensive experiments on benchmark datasets and realrobustness [...]
- [✓ FP removed] (limi/limi) Rather than depending success rates with greater naturalness and stealth compared on [...]
- [✓ FP removed] (limi/limi) When evaluating automated jailbreak generation methods, it is for their valuable [...]
- [✓ FP removed] (limi/limi) The research infrastructure and dataset of this work are supported by National [17] D.
- [✓ FP removed] (futu/limi) Vajravelu, “Assessing the current limitations of large C.

Dropped 69: 69 false positives removed, 0 real gold gaps lost.