# 27B neutral prompt: errors on five development papers

The fixed dataset labels are LLM-generated and not human verified. "False positive" and "false negative" mean disagreement with those labels, not an independently confirmed error. This is the fresh neutral run with matched five-paper batching.

All 1,713 sentences are scored: 105 are labeled positive; 75 were selected, 10 unlabeled sentences were selected, and 30 labeled positives were missed. Stage C received only candidates surviving A and B.

## False positives (10)

1. **2201.02967 / sentence 9** — However, when the underlying assumption fails to hold, the impact on the result can be signiﬁcant.

2. **2207.09511 / sentence 14** — Learning Process: Given a neural network architecture and a data set, how can we efficiently train generalizable networks, and when do optimization techniques like stochastic gradient descent work?

3. **2207.09511 / sentence 15** — Are there alternatives to gradient-based optimization methods?

4. **2207.09511 / sentence 24** — Instead, our goal is to present a clear and focused narrative that illuminates the fundamental mechanisms driving the success of deep networks, while staying within the constraints of scope and length.

5. **2207.09511 / sentence 646** — Specifically, here we only consider the scenario of noiseless data (ε = 0) and Fourier networks that do not include the regularization parameter (δ = 0).

6. **2207.09511 / sentence 817** — Neural network approximation is a nonlinear method and should be compared with other nonlinear techniques.

7. **2207.09511 / sentence 833** — While these directions are intriguing, their broader relevance and impact warrant further exploration. 62

8. **2301.03441 / sentence 119** — However, we have shown earlier that this approach is inefﬁcient to handle long sequences.

9. **2507.19233 / sentence 38** — Furthermore, the CFD model is based on Navier-Stokes equations and numerical schemes, which are require a high degree of expertise in fluid mechanics and their numerical solutions.

10. **2507.19233 / sentence 210** — Most errors are occurred to thin regions near ceiling and wall boundaries, where temperature gradients are sharp due to boundary heat flux and convective cooling from inlet jets.

## False negatives — Stage A: filtered as junk (6)

1. **2207.09511 / sentence 448** — label: limitation; filter: reference/caption rule, sentence-quality rule  
We note that the input dimension is one here, and hence this result does not address the effect of composition on dimension. 31 1 0 -1 0 -1/2 1 1 -1/2 -1/2 1 1 1 x 1 0 -1 0 -1/2 1 1 -1/2 -1/2 1 1 1 0 -1/4 0 -1/8 1/2 -1 1/2 0 1/2 1/2 1/2 1/2 -1/2 -1/2 1 1 -1 -1 0 0 x Figure 13: ReLU networks that realize g1(x) (left) and g2(x) (right). 14.2 Approximating product of two numbers by ReLU networks We next consider two real numbers x ∈R and y ∈R, where max{|x|, |y|} ≤D, with D > 0.

2. **2207.09511 / sentence 704** — label: research_gap; filter: reference/caption rule  
Figure 23: Original image (left) and rFN reconstruction (right) The presence of high-frequency features at multiple scales presents a significant approximation challenge for deep neural networks.

3. **2301.03441 / sentence 186** — label: research_gap; filter: reference/caption rule  
It was found in [52] that even though the performance of automatic sleep staging systems has been deemed sufﬁcient for clinical use these systems may generally struggle with particular recordings, leading to exceptionally high individual THIS ARTICLE HAS BEEN PUBLISHED IN IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS 7 TABLE III: Performance obtained by L-SeqSleepNet (L = 200), the SeqSleepNet baseline (L = 20) in comparison with previous works on the experimental databases.

4. **2301.03441 / sentence 220** — label: future_work; filter: sentence-quality rule  
System L (B × K) Acc. k MF1 Training time (s) L-SeqSleepNet 100 (10×10) 88.1 0.833 80.7 354.1 L-SeqSleepNet 200 (10×20) 88.2 0.835 81.4 454.4 L-SeqSleepNet 200 (20×10) 88.4 0.837 81.6 449.0 L-SeqSleepNet 400 (20×20) 88.4 0.838 81.4 605.2 SeqSleepNet 20 87.2 0.820 80.2 322.1 SeqSleepNet 100 87.4 0.823 80.1 842.0 SeqSleepNet 200 87.5 0.824 80.4 1465.6 framework to investigate the effects of whole-cycle sequential modelling in multimodal fusion [32] and multi-view learning [9] or to speed up the computation with recurrent-free archi- tectures, such as Transformer [7].

5. **2507.19233 / sentence 189** — label: limitation; filter: sentence-quality rule  
For cases with small right inlet velocities (e.g., L = 0.5 m/s, R = 0.05 m/s and L = 0.5 m/s, R = 0.25 m/s), the surrogate model exhibits noticeable deviations near the jet boundary and recirculation zones.

6. **2507.19233 / sentence 226** — label: limitation; filter: sentence-quality rule  
Besides, the spatial error map explains that largest deviations occur in cases with significant inlet velocity asymmetry (e.g., L = 0.1 m/s, R = 0.9 22 m/s and L = 0.9 m/s, R = 0.1 m/s), where the model struggles to fully capture the sharp momentum differences between the two inlet streams.

## False negatives — Stage B: NLI did not select (7)

1. **2201.02967 / sentence 23** — label: limitation  
The counterpart to such ﬂexibility relies in the characteristics of the clusters.

2. **2201.02967 / sentence 24** — label: limitation  
Indeed, we assume that points in the same cluster are drawn from distributions that share the same mean and scatter matrix.

3. **2207.09511 / sentence 250** — label: limitation  
For simplicity, throughout this review paper, we assume the availability of abundant high-quality data.

4. **2207.09511 / sentence 400** — label: limitation  
Throughout this section, we limit our discussion to one-dimensional input domains where X = [0, 1].

5. **2207.09511 / sentence 716** — label: limitation  
We also assume that all frequencies (ω, ω′) in the residual Fourier network fΨ of Theorem 3 are uniformly bounded.

6. **2207.09511 / sentence 830** — label: future_work  
One avenue involves exploring the connections between residual Fourier networks and other sinusoidal activation networks, such as sinusoidal representation networks (SIRENs) [44] and Fourier Neural Operators (FNOs) [45].

7. **2301.03441 / sentence 216** — label: future_work  
While the work here focuses on single-EEG input and uses RNN as the backbone for sequential modelling, the presented long sequential modelling method can be considered as a generic method and can be used in replacement for the “ﬂat” sequential modelling method in the current state-of- the-art sequence-to-sequence sleep staging framework [1].

## False negatives — Stage C: 27B rejected (17)

1. **2401.17544 / sentence 22** — label: research_gap  
Even- tually, there are two different models for training and FPGA-based inference.

2. **2401.17544 / sentence 121** — label: research_gap  
Even for highly compressed DNN models, a nontrivial number of DSPs are typically still required for operations using fixed-point arithmetic.

3. **2401.17544 / sentence 122** — label: research_gap  
By profiling a state-of-the-art BNN accelerator, FracBNN, we find that more than 60% of the DSPs on its Ultra96v2 FPGA deployment are still in use, even if all the matrix multiplications in the model are binarized.

4. **2401.17544 / sentence 123** — label: research_gap  
This presents a suboptimal scenario for edge devices.

5. **2401.17544 / sentence 149** — label: limitation  
For the other two BNN models, we only fine-tune QFX quantization for 30 epochs from the pretrained checkpoint due to the long training time.

6. **2401.17544 / sentence 150** — label: limitation  
The hyperparameters are the same as defined in their original papers except that the learning rate of FracBNN-ImageNet is ad- justed linearly with batch size of 128 due to limited GPU memory.

7. **2401.17544 / sentence 152** — label: limitation  
Further fine-tuning may lead to a higher accuracy but at a higher training cost.

8. **2201.02967 / sentence 25** — label: limitation  
However, under assumption of existence, points in the same cluster have only proportional covariances matrices.

9. **2201.02967 / sentence 86** — label: limitation  
This can be explained by the two follow- ing reasons: • Most dataset used have well separated clusters : even a linear classiﬁer (LDA) achieves very good perfor- mance. • Contamination is mild, it is a random noise with no structure that could lead the classiﬁer to consider all the noisy data as another cluster.

10. **2207.09511 / sentence 221** — label: limitation  
However, we will not address further architectural considerations here, as they can be viewed as part of optimal experimental design.

11. **2207.09511 / sentence 256** — label: research_gap; candidate text was truncated to 400 characters in the 27B call  
In numerical computation, target functions are often implicitly defined through differential, integral, or integro-differential equations, presenting additional constraints and challenges not typically addressed in approximation theory. 11 Density in the space of continuous functions Here, we briefly review the concept of uniform convergence, essential for exploring density in the space of continuous functions.

12. **2207.09511 / sentence 812** — label: research_gap  
From this and (60), the desired complexity (46) follows, and this completes the proof. 61 Chapter 4: Further reading and concluding remarks The mathematics of deep learning is in its early stages, with foundational results in place but much yet to be explored.

13. **2207.09511 / sentence 831** — label: future_work  
Understanding these relationships may provide fur- ther insights into the applicability of sinusoidal activations in diverse settings.

14. **2507.19233 / sentence 105** — label: limitation  
The selection of inlet velocities, presented in Table 1 and Table 2, was carried out manually to ensure sufficient diversity in flow patterns while keeping the dataset size manageable.

15. **2507.19233 / sentence 187** — label: limitation  
Minor deviations between the predicted and CFD results are evident in certain regions, particularly in areas with strong velocity gradients and shear layers.

16. **2507.19233 / sentence 227** — label: limitation  
To further compare the error trend and statistics under different inlet conditions, violin plots are employed in Figure 10, which shows that median error of all testing cases are lower than 0.03 m/s, which is around 3.5 times higher than median error in training dataset at 0.007m/s, but the trend of errors is similar to training set, which has low errors under relative symmetry cases.

17. **2507.19233 / sentence 234** — label: limitation  
In contrast, the case with L = 0.1 m/s and R = 0.9 m/s, which exhibits a strong inlet imbalance, yields a slightly lower R² of 0.9410 and larger scatter around the diagonal.

## Interpretation limits

Neither the dataset label nor the model output includes a human adjudication; some disagreements may reflect labeling omissions or different interpretations of what counts as a gap. Candidate context was truncated to 600 characters; candidate sentences to 400 characters. The model did not receive the labeled answers.

One concrete cleanup issue: sentences 189 and 226 from paper 2507.19233 were rejected by the sentence-quality rule because each has seven periods, including decimal points in numerical values. The rule rejects strings with more than six periods as likely table-of-contents debris. This is an identifiable rule behavior, not a human judgment of whether the two labels are correct.
