# Report template — replace all bracketed fields with measured results

**Title:** Bagging, Boosting and Bias–Variance on Face Recognition  
**Name:** Abhishek Kapar  
**Roll number:** M26ecm511  
**Repository:** [GitHub URL]  
**Final commit:** [actual commit hash]

## Abstract
We compare a decision-tree baseline, bootstrap aggregation, and multiclass SAMME AdaBoost on the Olivetti Faces dataset using raw 4096-dimensional pixels. A fixed identity-stratified split is reused across experiments. We evaluate clean and central-block-masked test images and estimate empirical bias and variance using 30 identity-wise bootstrap refits. The best validation-selected tree used [depth] and [leaf size]; bagging used [B] trees; boosting used [depth] and [rounds]. The measured results show [summary based on outputs].

## 1. Experimental protocol
Dataset: 40 identities × 10 images, 64 × 64 pixels. Seed: 26511. Per identity: six training, two validation, two clean test images. The split is saved in `outputs/split.json`. Inputs are raw flattened pixels; no PCA or dimensionality reduction is used. For robustness, pixels in rows 24–39 and columns 24–39 are set to zero only in a copy of the test set. All models are trained on clean images.

## 2. Decision tree and bagging
### 2.1 Single tree
We evaluated max_depth ∈ {2, 4, 8, 16, None} and min_samples_leaf ∈ {1, 2, 4}. The configuration was selected using validation accuracy only, then frozen. The selected model achieved train [ ], validation [ ], clean-test [ ] accuracy. As depth increases, the tree can represent more complex partitions and usually reduces training error, but may create small leaves and increase sensitivity to the training sample. Validation performance need not improve monotonically because deeper trees can overfit.

### 2.2 Bagging from scratch
For each B ∈ {1, 5, 10, 25, 50, 100, 200}, we sampled n training examples with replacement and trained one tree on each bootstrap sample. Predictions use majority vote. The measured train/validation/test accuracies and OOB accuracy are reported in Table [ ]. For a bootstrap sample of size n, the probability that a particular training example is omitted is (1 − 1/n)^n ≈ e^−1 ≈ 0.368; therefore the expected number of OOB trees per example is approximately 0.368B. The empirical mean is [ ].

### 2.3 Disagreement and vote margin
Pairwise disagreement is the average fraction of examples on which two ensemble members differ. Vote margin is the winning class's vote fraction minus the runner-up class's vote fraction. The curves are in [figure names]. Higher disagreement indicates diversity, but diversity alone does not guarantee better accuracy. A larger margin indicates a more decisive ensemble prediction, not necessarily a correct one.

### 2.4 Clean versus masked faces
On clean images, the tree and bagging accuracies were [ ] and [ ]; on masked images they were [ ] and [ ]. Examples where the models disagree are listed in `outputs/q1d_clean_masked.json`. Bagging can reduce variance by averaging unstable tree decisions. It cannot recover visual information removed from every masked image, and correlated errors shared by most trees can remain.

## 3. Multiclass SAMME boosting
### 3.1 SAMME update
For K classes, learner error ε_t is the current weighted fraction misclassified. If ε_t is not better than random multiclass prediction (ε_t ≥ 1 − 1/K), the weak learner is rejected/stopping is applied. Otherwise:
α_t = log((1 − ε_t)/ε_t) + log(K − 1).
Weights are updated as w_i ← w_i exp(α_t I[h_t(x_i) ≠ y_i]) and normalized to sum to one. Thus misclassified observations gain relative weight. The three-class, ten-example update is documented in `outputs/q2a_samme_toy.json`.

### 3.2 Selection and convergence
We evaluated depths 1–4 and requested rounds 1, 2, 3, 4, 10, 25, 50 and 100, selecting depth and rounds using clean validation accuracy only. The selected configuration was [ ]. Training accuracy [increased/decreased/plateaued] with rounds, while validation accuracy [pattern]. Figures should show weighted training error, learner weights, maximum sample weight, and total weight of the ten heaviest examples across rounds; discuss any late-round overemphasis and diminishing validation returns.

### 3.3 Bagging versus boosting
For 25, 50 and 100 learners, train/validation/test accuracies and training time are in `outputs/q2c_comparison.json`. Bagging reduces variance through averaging independently bootstrapped learners; boosting sequentially emphasizes observations previously misclassified. Repeatedly high weights may indicate difficult, ambiguous, noisy or atypical samples. Report the identified sample's true class and prediction, without assuming it is mislabeled.

### 3.4 Missing-region robustness
The clean and masked accuracies and normalized-vote confidence for all three frozen models are in `outputs/q2d_clean_masked.json`. Bagging uses vote proportions. SAMME uses class scores normalized by the sum of learner weights as specified in the assignment; these normalized scores are not calibrated probabilities. Compare the changes under masking and give one case where model responses differ.

## 4. Bias and variance
For each of 30 refits, six training entries per identity were sampled with replacement from that identity's original six training entries. The clean test set was fixed and no tuning was repeated. For each test image i, p_r(x_i) is the class-vector prediction for refit r and p̄(x_i) is its mean over refits. The assignment's empirical quantities are:
Bias² = (1/N) Σ_i ||p̄(x_i) − e_i||²
Variance = (1/(NR)) Σ_i,r ||p_r(x_i) − p̄(x_i)||²
where e_i is the one-hot true class vector. Results and test-accuracy mean ± standard deviation are in `outputs/q3_bias_variance.json`.

A deeper tree often has lower training bias but can be more sample-sensitive; bagging often reduces prediction variance by averaging trees. These are hypotheses to compare with measured values, not guaranteed outcomes. The deepest tree is not necessarily the depth at which bagging gives the largest test-accuracy improvement: leaf occupancy, tree disagreement, bias and variance all matter.

## 5. Summary
| Model | Clean test accuracy | Masked test accuracy | Bias² | Variance |
|---|---:|---:|---:|---:|
| Selected single tree | [ ] | [ ] | [ ] | [ ] |
| Selected bagging | [ ] | [ ] | [ ] | [ ] |
| Selected SAMME boosting | [ ] | [ ] | [ ] | [ ] |

The most stable model was [ ], judged from refit accuracy standard deviation and measured variance. The most mask-sensitive model was [ ]. The lowest-bias model was [ ], but it [did/did not] achieve the highest test accuracy, illustrating that predictive performance depends on the bias–variance trade-off and robustness, not one metric alone.

## References
[1] AT&T Laboratories Cambridge / Olivetti Research Laboratory, The Olivetti Faces Dataset.
[2] F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.
[3] Y. Freund and R. E. Schapire, “A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting,” Journal of Computer and System Sciences, 1997.
