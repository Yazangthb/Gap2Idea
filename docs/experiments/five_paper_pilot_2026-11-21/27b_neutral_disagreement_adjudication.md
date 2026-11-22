# Review of 27B neutral disagreements

This is one Codex model’s review, not a human adjudication or independent ground truth. Only the 40 disagreements were re-reviewed, using each sentence and its immediate neighbors. All 1,673 agreements were left at their existing label; thus the following recomputed metrics are conditional and not validated corpus performance.

Rule used: Sentence-level: explicit author-work constraint or assumption, proposed future work, or specific unsolved problem in this work or prior work; use neighbouring text for references, but do not label ordinary experimental results or vague motivation alone.

| Measure | Original dataset labels | After this disagreement-only review |
| --- | ---: | ---: |
| True positives | 75 | 80 |
| False positives | 10 | 5 |
| False negatives | 30 | 18 |
| Precision | 88.2% | 94.1% |
| Recall | 71.4% | 81.6% |
| F1 | 78.9% | 87.4% |

Five of the original 10 false positives are judged positive; 12 of the original 30 false negatives are judged negative. The other 23 disagreements remain errors under this review. The binary decisions below include borderline cases; a different interpretation of author-stated gaps, assumptions, or experimental outcomes may change results.

## Previously false positives

- **2201.02967 / 9 — gap:** States a specific weakness of prior methods when their distributional assumption fails.  
  However, when the underlying assumption fails to hold, the impact on the result can be signiﬁcant.

- **2207.09511 / 14 — not a gap:** A broad survey question; it does not establish a specific unsolved problem on its own.  
  Learning Process: Given a neural network architecture and a data set, how can we efficiently train generalizable networks, and when do optimization techniques like stochastic gradient descent work?

- **2207.09511 / 15 — not a gap:** Asks about alternatives but does not establish that they are lacking or propose a specific future investigation.  
  Are there alternatives to gradient-based optimization methods?

- **2207.09511 / 24 — not a gap:** States the survey’s aim; the limitations or exclusions occur in the adjacent sentence.  
  Instead, our goal is to present a clear and focused narrative that illuminates the fundamental mechanisms driving the success of deep networks, while staying within the constraints of scope and length.

- **2207.09511 / 646 — gap:** Explicitly restricts the analysis to noiseless data and networks without a regularization parameter.  
  Specifically, here we only consider the scenario of noiseless data (ε = 0) and Fourier networks that do not include the regularization parameter (δ = 0).

- **2207.09511 / 817 — not a gap:** A general comparison recommendation followed by references to existing comparisons; not clearly a proposed new research direction.  
  Neural network approximation is a nonlinear method and should be compared with other nonlinear techniques.

- **2207.09511 / 833 — gap:** In context explicitly calls for further exploration of the listed directions’ relevance and impact.  
  While these directions are intriguing, their broader relevance and impact warrant further exploration. 62

- **2301.03441 / 119 — gap:** Explicitly identifies inefficient handling of long sequences in an existing approach.  
  However, we have shown earlier that this approach is inefﬁcient to handle long sequences.

- **2507.19233 / 38 — gap:** In context identifies an expertise requirement that constrains use of the CFD method.  
  Furthermore, the CFD model is based on Navier-Stokes equations and numerical schemes, which are require a high degree of expertise in fluid mechanics and their numerical solutions.

- **2507.19233 / 210 — not a gap:** Describes observed locations of model error, an experimental result.  
  Most errors are occurred to thin regions near ceiling and wall boundaries, where temperature gradients are sharp due to boundary heat flux and convective cooling from inlet jets.

## Previously false negatives

- **2401.17544 / 22 — not a gap:** Separate training and inference models are described, but no limitation is stated in this sentence.  
  Even- tually, there are two different models for training and FPGA-based inference.

- **2401.17544 / 121 — gap:** Substantial use of a scarce DSP resource is an explicit existing-method constraint.  
  Even for highly compressed DNN models, a nontrivial number of DSPs are typically still required for operations using fixed-point arithmetic.

- **2401.17544 / 122 — not a gap:** Reports a measured DSP utilization result; the limitation is articulated in adjacent sentences.  
  By profiling a state-of-the-art BNN accelerator, FracBNN, we find that more than 60% of the DSPs on its Ultra96v2 FPGA deployment are still in use, even if all the matrix multiplications in the model are binarized.

- **2401.17544 / 123 — gap:** In context explicitly describes the preceding resource use as suboptimal for edge devices.  
  This presents a suboptimal scenario for edge devices.

- **2401.17544 / 149 — gap:** Limited fine-tuning due to long training time is a constraint on the authors’ experiment.  
  For the other two BNN models, we only fine-tune QFX quantization for 30 epochs from the pretrained checkpoint due to the long training time.

- **2401.17544 / 150 — gap:** Explicitly states limited GPU memory constrained their experimental setup.  
  The hyperparameters are the same as defined in their original papers except that the learning rate of FracBNN-ImageNet is ad- justed linearly with batch size of 128 due to limited GPU memory.

- **2401.17544 / 152 — not a gap:** Speculates about an accuracy–cost tradeoff rather than explicitly proposing future research or stating an unsolved limitation.  
  Further fine-tuning may lead to a higher accuracy but at a higher training cost.

- **2201.02967 / 23 — not a gap:** Vague transition; it does not itself specify a constraint.  
  The counterpart to such ﬂexibility relies in the characteristics of the clusters.

- **2201.02967 / 24 — gap:** Explicit assumption imposed on clusters in the authors’ work.  
  Indeed, we assume that points in the same cluster are drawn from distributions that share the same mean and scatter matrix.

- **2201.02967 / 25 — not a gap:** States a mathematical implication under an assumption, rather than articulating the assumption or a limitation.  
  However, under assumption of existence, points in the same cluster have only proportional covariances matrices.

- **2201.02967 / 86 — gap:** Explains that the datasets have unusually well-separated clusters and mild contamination, constraints on the evidence.  
  This can be explained by the two follow- ing reasons: • Most dataset used have well separated clusters : even a linear classiﬁer (LDA) achieves very good perfor- mance. • Contamination is mild, it is a random noise with no structure that could lead the classiﬁer to consider all the noisy data as another cluster.

- **2207.09511 / 221 — gap:** Explicitly excludes further architectural considerations from the review’s scope.  
  However, we will not address further architectural considerations here, as they can be viewed as part of optimal experimental design.

- **2207.09511 / 250 — gap:** Explicitly states a simplifying assumption of abundant high-quality data.  
  For simplicity, throughout this review paper, we assume the availability of abundant high-quality data.

- **2207.09511 / 256 — gap:** Identifies computational constraints and challenges not typically addressed in approximation theory.  
  In numerical computation, target functions are often implicitly defined through differential, integral, or integro-differential equations, presenting additional constraints and challenges not typically addressed in approximation theory. 11 Density in the space of continuous functions Here, we briefly review the concept of uniform convergence, essential for exploring density in the space of continuous functions.

- **2207.09511 / 400 — gap:** Explicitly limits this review section to one-dimensional input domains.  
  Throughout this section, we limit our discussion to one-dimensional input domains where X = [0, 1].

- **2207.09511 / 448 — gap:** Explicitly says the one-dimensional result does not address the effect of composition on dimension; extraction is corrupted.  
  We note that the input dimension is one here, and hence this result does not address the effect of composition on dimension. 31 1 0 -1 0 -1/2 1 1 -1/2 -1/2 1 1 1 x 1 0 -1 0 -1/2 1 1 -1/2 -1/2 1 1 1 0 -1/4 0 -1/8 1/2 -1 1/2 0 1/2 1/2 1/2 1/2 -1/2 -1/2 1 1 -1 -1 0 0 x Figure 13: ReLU networks that realize g1(x) (left) and g2(x) (right). 14.2 Approximating product of two numbers by ReLU networks We next consider two real numbers x ∈R and y ∈R, where max{|x|, |y|} ≤D, with D > 0.

- **2207.09511 / 704 — gap:** States a specific approximation challenge for deep networks, despite an attached figure caption.  
  Figure 23: Original image (left) and rFN reconstruction (right) The presence of high-frequency features at multiple scales presents a significant approximation challenge for deep neural networks.

- **2207.09511 / 716 — gap:** Explicitly assumes bounded frequencies in the analysis.  
  We also assume that all frequencies (ω, ω′) in the residual Fourier network fΨ of Theorem 3 are uniformly bounded.

- **2207.09511 / 812 — not a gap:** Generic statement that much remains to explore, without a specific gap.  
  From this and (60), the desired complexity (46) follows, and this completes the proof. 61 Chapter 4: Further reading and concluding remarks The mathematics of deep learning is in its early stages, with foundational results in place but much yet to be explored.

- **2207.09511 / 830 — gap:** Explicitly proposes exploring connections between named network families.  
  One avenue involves exploring the connections between residual Fourier networks and other sinusoidal activation networks, such as sinusoidal representation networks (SIRENs) [44] and Fourier Neural Operators (FNOs) [45].

- **2207.09511 / 831 — not a gap:** Describes the potential benefit of the preceding proposal rather than itself proposing a direction.  
  Understanding these relationships may provide fur- ther insights into the applicability of sinusoidal activations in diverse settings.

- **2301.03441 / 186 — gap:** States prior systems struggle with particular recordings; the extracted sentence is corrupted by a table heading.  
  It was found in [52] that even though the performance of automatic sleep staging systems has been deemed sufﬁcient for clinical use these systems may generally struggle with particular recordings, leading to exceptionally high individual THIS ARTICLE HAS BEEN PUBLISHED IN IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS 7 TABLE III: Performance obtained by L-SeqSleepNet (L = 200), the SeqSleepNet baseline (L = 20) in comparison with previous works on the experimental databases.

- **2301.03441 / 216 — gap:** States the authors’ work is limited in scope to single-EEG input and an RNN backbone.  
  While the work here focuses on single-EEG input and uses RNN as the backbone for sequential modelling, the presented long sequential modelling method can be considered as a generic method and can be used in replacement for the “ﬂat” sequential modelling method in the current state-of- the-art sequence-to-sequence sleep staging framework [1].

- **2301.03441 / 220 — gap:** The surviving fragment proposes investigating multimodal and multi-view settings or faster architectures; extraction is corrupted.  
  System L (B × K) Acc. k MF1 Training time (s) L-SeqSleepNet 100 (10×10) 88.1 0.833 80.7 354.1 L-SeqSleepNet 200 (10×20) 88.2 0.835 81.4 454.4 L-SeqSleepNet 200 (20×10) 88.4 0.837 81.6 449.0 L-SeqSleepNet 400 (20×20) 88.4 0.838 81.4 605.2 SeqSleepNet 20 87.2 0.820 80.2 322.1 SeqSleepNet 100 87.4 0.823 80.1 842.0 SeqSleepNet 200 87.5 0.824 80.4 1465.6 framework to investigate the effects of whole-cycle sequential modelling in multimodal fusion [32] and multi-view learning [9] or to speed up the computation with recurrent-free archi- tectures, such as Transformer [7].

- **2507.19233 / 105 — not a gap:** Describes a data-selection procedure without explicitly identifying a limitation.  
  The selection of inlet velocities, presented in Table 1 and Table 2, was carried out manually to ensure sufficient diversity in flow patterns while keeping the dataset size manageable.

- **2507.19233 / 187 — not a gap:** Reports observed minor deviations as an experimental result rather than an explicit general limitation.  
  Minor deviations between the predicted and CFD results are evident in certain regions, particularly in areas with strong velocity gradients and shear layers.

- **2507.19233 / 189 — not a gap:** Reports deviations in particular experimental cases rather than explicitly identifying a general limitation.  
  For cases with small right inlet velocities (e.g., L = 0.5 m/s, R = 0.05 m/s and L = 0.5 m/s, R = 0.25 m/s), the surrogate model exhibits noticeable deviations near the jet boundary and recirculation zones.

- **2507.19233 / 226 — gap:** Explicitly states that the model struggles to capture sharp momentum differences.  
  Besides, the spatial error map explains that largest deviations occur in cases with significant inlet velocity asymmetry (e.g., L = 0.1 m/s, R = 0.9 22 m/s and L = 0.9 m/s, R = 0.1 m/s), where the model struggles to fully capture the sharp momentum differences between the two inlet streams.

- **2507.19233 / 227 — not a gap:** Reports quantitative test error and visualization results rather than explicitly asserting a limitation.  
  To further compare the error trend and statistics under different inlet conditions, violin plots are employed in Figure 10, which shows that median error of all testing cases are lower than 0.03 m/s, which is around 3.5 times higher than median error in training dataset at 0.007m/s, but the trend of errors is similar to training set, which has low errors under relative symmetry cases.

- **2507.19233 / 234 — not a gap:** Reports the observed R-squared and scatter for one case rather than explicitly stating a limitation.  
  In contrast, the case with L = 0.1 m/s and R = 0.9 m/s, which exhibits a strong inlet imbalance, yields a slightly lower R² of 0.9410 and larger scatter around the diagonal.

