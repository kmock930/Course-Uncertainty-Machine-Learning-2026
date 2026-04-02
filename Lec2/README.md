# Lecture 2 – Bayesian Inference for Gaussians and Bayesian Linear Regression

**Course:** ELG 5218 / CSI 5218 / EACJ 5600 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
**Date:** January 24, 2026  
**Instructor:** Miodrag Bolic, University of Ottawa

---

## Overview

This lecture develops the Bayesian treatment of **Gaussian models**, from scalar to multivariate
settings, and extends the framework to **Bayesian linear regression** and **Bayesian logistic
regression** (introduction). Emphasis is placed on precision additivity (the engine of conjugate
Gaussian updates), sequential/online learning, the Normal–Gamma conjugate family for unknown
variance, and multivariate sensor fusion via the joint Gaussian.

---

## Learning Goals

By the end of this lecture you should be able to:

- Explain why Gaussian distributions dominate Bayesian inference (CLT, conjugacy, tractability).
- Compute the **conjugate Bayesian update** for a Gaussian likelihood (known and unknown variance).
- Derive **precision additivity** and the precision-weighted posterior mean.
- Apply **sequential (online) Bayesian updates** with streaming observations.
- Work with the **Normal–Gamma** conjugate family for jointly unknown mean and variance.
- Manipulate **multivariate Gaussians**: marginals, conditionals, affine transforms.
- Set up and solve **Bayesian linear regression**, deriving the predictive distribution.

---

## Topics Covered

### Gaussian Models (`GaussianModels.pdf`)
- Why Gaussians? Universality (CLT), analytical tractability, conjugacy, convex optimization.
- Univariate and multivariate Gaussian density; precision notation.
- **Known variance case**: precision additivity; posterior as precision-weighted average of prior mean and sample mean.
- **Online (sequential) learning**: recursive Bayesian update; decreasing learning rate as evidence accumulates.
- **Unknown variance**: Normal–Gamma conjugate prior; Student-t marginal; robustness interpretation.
- **Monte Carlo** posterior sampling: functions of parameters, practical inference.
- **Multivariate case**: covariance and precision matrices; sensor fusion; conditional independence.

### Gaussian Identities Reference Sheet (`GaussianFormulas.pdf`)
- Marginal and conditional distributions for partitioned Gaussian vectors.
- Affine transformations of Gaussians.
- Key scalar identities (completing the square, conjugate update formula).
- Truncated Gaussian distributions.
- Applications to Bayesian linear regression derivations.

### Bayesian Linear Regression (`tpmi_w19_lec5_slides_print.pdf` and `tpmi_w19_lec6_slides_print.pdf`)
- Linear regression as a probabilistic model; Gaussian likelihood.
- Conjugate Gaussian prior on weights; closed-form posterior.
- Posterior predictive distribution for new inputs (predictive intervals).
- Ridge regression as MAP with a Gaussian prior.
- Extension to Bayesian logistic regression (introduction).

---

## Materials

| File | Description |
|------|-------------|
| `GaussianModels.pdf` | Main lecture slides: Bayesian Gaussian models (33 slides) |
| `GaussianFormulas.pdf` | Reference sheet: Gaussian identities and worked examples |
| `tpmi_w19_lec5_slides_print.pdf` | Guest slides: Bayesian linear regression (from slide 19 by Prof. Rao) |
| `tpmi_w19_lec6_slides_print.pdf` | Guest slides: Bayesian logistic regression introduction |
| `Gaussian_Models_Final.ipynb` | Jupyter notebook: Gaussian model inference and sensor fusion |

---

## Recommended Reading

- **Villani, Mattias (2025). *Bayesian Learning*.** – Chapters 2, 3.3, and 5.
  [PDF](https://github.com/mattiasvillani/BayesianLearningBook/raw/main/pdf/BayesBook.pdf)
- **NumPyro Bayesian Regression Tutorial:**
  [Bayesian Regression Using NumPyro](https://num.pyro.ai/en/stable/tutorials/bayesian_regression.html)

---

## Practice Questions

Relevant practice sets from the `Practice Questions/` folder:

- `Problems_Gaussian_Models_Questions.pdf` / `Problems_Gaussian_Models_Solutions.pdf` – Precision additivity, precision-weighted averaging, sequential updates, Normal–Gamma hyperparameters, multivariate sensor fusion.
- `Problems_Linear_Regression_Questions.pdf` / `Problems_Linear_Regression_Solutions.pdf` – Bayesian linear regression derivations, posterior and predictive distributions, MAP vs. MLE, model comparison.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| Precision (λ) | Reciprocal of variance; precisions add in Bayesian updates |
| Conjugate Gaussian update | Normal prior × Normal likelihood → Normal posterior |
| Precision-weighted mean | Posterior mean is a weighted combination of prior mean and MLE |
| Online/sequential learning | Recursive update: add one observation at a time |
| Normal–Gamma | Conjugate prior for jointly unknown Gaussian mean and precision |
| Student-t marginal | Marginalizing over precision yields a heavy-tailed distribution |
| Multivariate Gaussian | Generalization: covariance matrix, Schur complement for conditioning |
| Bayesian linear regression | Gaussian prior on weights yields closed-form posterior and predictive |
| Predictive distribution | Posterior predictive integrates weight uncertainty into forecasts |
