# Bagging, Boosting and Bias–Variance on Face Recognition

> **Report template.** Replace every `[bracketed field]` with measured results from `outputs/`. Delete this note before submitting.

| | |
|---|---|
| **Student** | Abhishek Kapar |
| **Roll number** | M26ECM511 |
| **Random seed** | 26511 |
| **Repository** | [github.com/m26ecm511-cpu/ML-Assignment-2-Bagging-Boosting](https://github.com/m26ecm511-cpu/ML-Assignment-2-Bagging-Boosting) |
| **Final commit** | `[paste full commit hash]` |
| **Commit link** | `https://github.com/m26ecm511-cpu/ML-Assignment-2-Bagging-Boosting/commit/[commit hash]` |

> **Getting the commit hash:** run `git rev-parse HEAD` for the full hash, or `git rev-parse --short HEAD` for the short one. On GitHub, it is shown next to the latest commit on the repository page. Take it **after** your last push so the report matches the final code.

---

## Contents

1. [Abstract](#abstract)
2. [Experimental protocol](#1-experimental-protocol)
3. [Decision tree and bagging](#2-decision-tree-and-bagging)
4. [Multiclass SAMME boosting](#3-multiclass-samme-boosting)
5. [Bias and variance](#4-bias-and-variance)
6. [Summary](#5-summary)
7. [References](#references)

---

## Abstract

We compare a decision-tree baseline, bootstrap aggregation, and multiclass SAMME AdaBoost on the Olivetti Faces dataset using raw 4096-dimensional pixels. A fixed identity-stratified split is reused across experiments. We evaluate clean and central-block-masked test images and estimate empirical bias and variance using 30 identity-wise bootstrap refits.

The best validation-selected tree used depth **[depth]** and leaf size **[leaf size]**; bagging used **[B]** trees; boosting used depth **[depth]** and **[rounds]** rounds. The measured results show **[summary based on outputs]**.

---

## 1. Experimental protocol

| Item | Setting |
|---|---|
| Dataset | Olivetti Faces: 40 identities × 10 images, 64 × 64 pixels |
| Features | Raw flattened pixels (4096-dimensional); no PCA or dimensionality reduction |
| Seed | 26511 |
| Split (per identity) | 6 training, 2 validation, 2 clean test images |
| Split file | `outputs/split.json` |
| Masking | Pixels in rows 24–39 and columns 24–39 set to zero, **only** in a copy of the test set |
| Training data | All models are trained on clean images |

---

## 2. Decision tree and bagging

### 2.1 Single tree

We evaluated `max_depth` ∈ {2, 4, 8, 16, None} and `min_samples_leaf` ∈ {1, 2, 4}. The configuration was selected using validation accuracy only, then frozen.

| Selected setting | Train acc. | Validation acc. | Clean-test acc. |
|---|---:|---:|---:|
| depth = [ ], leaf = [ ] | [ ] | [ ] | [ ] |

As depth increases, the tree can represent more complex partitions and usually reduces training error, but may create small leaves and increase sensitivity to the training sample. Validation performance need not improve monotonically because deeper trees can overfit.

*Source:* `outputs/q1a_tree_grid.json`

### 2.2 Bagging from scratch

For each B ∈ {1, 5, 10, 25, 50, 100, 200}, we sampled *n* training examples with replacement and trained one tree on each bootstrap sample. Predictions use majority vote.

| B | Train acc. | Validation acc. | Test acc. | OOB acc. |
|---:|---:|---:|---:|---:|
| 1 | [ ] | [ ] | [ ] | [ ] |
| 5 | [ ] | [ ] | [ ] | [ ] |
| 10 | [ ] | [ ] | [ ] | [ ] |
| 25 | [ ] | [ ] | [ ] | [ ] |
| 50 | [ ] | [ ] | [ ] | [ ] |
| 100 | [ ] | [ ] | [ ] | [ ] |
| 200 | [ ] | [ ] | [ ] | [ ] |

For a bootstrap sample of size *n*, the probability that a particular training example is omitted is

$$\left(1 - \frac{1}{n}\right)^n \approx e^{-1} \approx 0.368,$$

so the expected number of OOB trees per example is approximately $0.368B$. The empirical mean is **[ ]**.

*Source:* `outputs/q1b_bagging.json`, `outputs/figures/q1_oob_accuracy.png`

### 2.3 Disagreement and vote margin

Pairwise disagreement is the average fraction of examples on which two ensemble members differ. Vote margin is the winning class's vote fraction minus the runner-up class's vote fraction. The curves are in **[figure names]**.

Higher disagreement indicates diversity, but diversity alone does not guarantee better accuracy. A larger margin indicates a more decisive ensemble prediction, not necessarily a correct one.

| Figure | File |
|---|---|
| Disagreement | `outputs/figures/q1_disagreement.png` |
| Vote margin | `outputs/figures/q1_vote_margin.png` |

*Source:* `outputs/q1c_ensemble_diagnostics.json`

### 2.4 Clean versus masked faces

| Model | Clean test acc. | Masked test acc. |
|---|---:|---:|
| Single tree | [ ] | [ ] |
| Bagging | [ ] | [ ] |

Examples where the models disagree are listed in `outputs/q1d_clean_masked.json`. Bagging can reduce variance by averaging unstable tree decisions. It cannot recover visual information removed from every masked image, and correlated errors shared by most trees can remain.

---

## 3. Multiclass SAMME boosting

### 3.1 SAMME update

For *K* classes, learner error $\varepsilon_t$ is the current weighted fraction misclassified. If $\varepsilon_t$ is not better than random multiclass prediction ($\varepsilon_t \ge 1 - 1/K$), the weak learner is rejected / stopping is applied. Otherwise:

$$\alpha_t = \log\frac{1-\varepsilon_t}{\varepsilon_t} + \log(K-1)$$

Weights are updated and then normalized to sum to one:

$$w_i \leftarrow w_i \exp\!\big(\alpha_t \, \mathbb{I}[h_t(x_i) \ne y_i]\big)$$

Thus misclassified observations gain relative weight. The three-class, ten-example update is documented in `outputs/q2a_samme_toy.json`.

### 3.2 Selection and convergence

We evaluated depths 1–4 and requested rounds 1, 2, 3, 4, 10, 25, 50 and 100, selecting depth and rounds using clean validation accuracy only.

| Selected depth | Selected rounds | Train acc. | Validation acc. | Clean-test acc. |
|---:|---:|---:|---:|---:|
| [ ] | [ ] | [ ] | [ ] | [ ] |

Training accuracy **[increased / decreased / plateaued]** with rounds, while validation accuracy **[pattern]**. Discuss any late-round overemphasis and diminishing validation returns.

| Figure | File |
|---|---|
| Weighted training error | `outputs/figures/q2_weighted_training_error.png` |
| Learner weights | `outputs/figures/q2_learner_weights.png` |
| Maximum sample weight | `outputs/figures/q2_max_sample_weight.png` |
| Top-10 weight mass | `outputs/figures/q2_top10_weight_mass.png` |
| Training accuracy | `outputs/figures/q2_training_accuracy.png` |

*Source:* `outputs/q2b_boosting_grid.json`

### 3.3 Bagging versus boosting

For 25, 50 and 100 learners, train/validation/test accuracies and training time are in `outputs/q2c_comparison.json`.

| Learners | Method | Train acc. | Validation acc. | Test acc. | Training time (s) |
|---:|---|---:|---:|---:|---:|
| 25 | Bagging | [ ] | [ ] | [ ] | [ ] |
| 25 | Boosting | [ ] | [ ] | [ ] | [ ] |
| 50 | Bagging | [ ] | [ ] | [ ] | [ ] |
| 50 | Boosting | [ ] | [ ] | [ ] | [ ] |
| 100 | Bagging | [ ] | [ ] | [ ] | [ ] |
| 100 | Boosting | [ ] | [ ] | [ ] | [ ] |

Bagging reduces variance through averaging independently bootstrapped learners; boosting sequentially emphasizes observations previously misclassified. Repeatedly high weights may indicate difficult, ambiguous, noisy or atypical samples.

**Heaviest-weight sample:** true class **[ ]**, predicted class **[ ]**. Report this without assuming the sample is mislabeled.

### 3.4 Missing-region robustness

The clean and masked accuracies and normalized-vote confidence for all three frozen models are in `outputs/q2d_clean_masked.json`.

| Model | Clean acc. | Masked acc. | Change | Mean confidence (clean) | Mean confidence (masked) |
|---|---:|---:|---:|---:|---:|
| Single tree | [ ] | [ ] | [ ] | [ ] | [ ] |
| Bagging | [ ] | [ ] | [ ] | [ ] | [ ] |
| SAMME boosting | [ ] | [ ] | [ ] | [ ] | [ ] |

Bagging uses vote proportions. SAMME uses class scores normalized by the sum of learner weights, as specified in the assignment; these normalized scores are **not** calibrated probabilities.

**Case where model responses differ:** **[describe one example]**

---

## 4. Bias and variance

For each of 30 refits, six training entries per identity were sampled with replacement from that identity's original six training entries. The clean test set was fixed and no tuning was repeated.

For each test image $i$, $p_r(x_i)$ is the class-vector prediction for refit $r$, and $\bar p(x_i)$ is its mean over refits. The assignment's empirical quantities are:

$$\text{Bias}^2 = \frac{1}{N}\sum_{i} \left\lVert \bar p(x_i) - e_i \right\rVert^2$$

$$\text{Variance} = \frac{1}{NR}\sum_{i,r} \left\lVert p_r(x_i) - \bar p(x_i) \right\rVert^2$$

where $e_i$ is the one-hot true class vector.

| Model | Bias² | Variance | Test acc. (mean ± std) |
|---|---:|---:|---:|
| Single tree | [ ] | [ ] | [ ] ± [ ] |
| Bagging | [ ] | [ ] | [ ] ± [ ] |
| SAMME boosting | [ ] | [ ] | [ ] ± [ ] |

| Figure | File |
|---|---|
| Bias squared | `outputs/figures/q3_bias_squared.png` |
| Variance | `outputs/figures/q3_variance.png` |

A deeper tree often has lower training bias but can be more sample-sensitive; bagging often reduces prediction variance by averaging trees. These are hypotheses to compare with measured values, not guaranteed outcomes. The deepest tree is not necessarily the depth at which bagging gives the largest test-accuracy improvement: leaf occupancy, tree disagreement, bias and variance all matter.

*Source:* `outputs/q3_bias_variance.json`

---

## 5. Summary

| Model | Clean test accuracy | Masked test accuracy | Bias² | Variance |
|---|---:|---:|---:|---:|
| Selected single tree | [ ] | [ ] | [ ] | [ ] |
| Selected bagging | [ ] | [ ] | [ ] | [ ] |
| Selected SAMME boosting | [ ] | [ ] | [ ] | [ ] |

- The most stable model was **[ ]**, judged from refit accuracy standard deviation and measured variance.
- The most mask-sensitive model was **[ ]**.
- The lowest-bias model was **[ ]**, but it **[did / did not]** achieve the highest test accuracy, illustrating that predictive performance depends on the bias–variance trade-off and robustness, not one metric alone.

---

## Before you submit

- [ ] All `[ ]` and `[bracketed]` fields are replaced with measured values
- [ ] Commit hash is filled in and matches the final push
- [ ] Every figure and table is referenced in the text
- [ ] Final PDF is saved as `outputs/report.pdf`

---

## References

1. AT&T Laboratories Cambridge / Olivetti Research Laboratory, *The Olivetti Faces Dataset*.
2. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
3. Y. Freund and R. E. Schapire, "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting," *Journal of Computer and System Sciences*, 1997.
