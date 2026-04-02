# Practice Questions

**Course:** ELG 5218 / CSI 5218 / EACJ 5600 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
**Instructor:** Miodrag Bolic, University of Ottawa

---

## Overview

This folder contains **practice problem sets with full solutions** covering all major topics
of the course. Each problem set is provided in two files: a **Questions** file and a
**Solutions** file. The questions are designed to reinforce conceptual understanding,
mathematical derivations, and practical implementation skills aligned with the course
assignments, midterms, and final exam.

---

## File Index

| Questions | Solutions | Corresponding Lecture(s) |
|-----------|-----------|--------------------------|
| `Problems_Bayesian_Questions.pdf` | `Problems_Bayesian_Solutions.pdf` | Lec 1 – Bayesian Modeling Foundations |
| `Problems_Gaussian_Models_Questions.pdf` | `Problems_Gaussian_Models_Solutions.pdf` | Lec 2 – Bayesian Inference for Gaussians |
| `Problems_Linear_Regression_Questions.pdf` | `Problems_Linear_Regression_Solutions.pdf` | Lec 2 – Bayesian Linear Regression |
| `Problems_Logistic_Regression_Questions.pdf` | `Problems_Logistic_Regression_Solutions.pdf` | Lec 3 – Bayesian Logistic Regression |
| `Problems_MCMC_Questions.pdf` | `Problems_MCMC_Solutions.pdf` | Lec 3 – Markov Chain Monte Carlo |
| `Problems_Langevin_Questions.pdf` | `Problems_Langevin_Solutions.pdf` | Lec 3 – Langevin Monte Carlo / MALA |
| `Problems_Kalman_Filter_Questions.pdf` | `Problems_Kalman_Filter_Solutions.pdf` | Lec 4 – Kalman Filter |
| `Problems_Particle_Filter_Questions.pdf` | `Problems_Particle_Filter_Solutions.pdf` | Lec 4 – Particle Filters |
| `Problems_VI_Questions.pdf` | `Problems_VI_Solutions.pdf` | Lec 5 – Variational Inference |
| `SVI_Problems.pdf` | `SVI_solutions.pdf` | Lec 5 – Stochastic Variational Inference |
| `Problems_NN_Parameterization_Questions.pdf` | `Problems_NN_Parameterization_Solutions.pdf` | Lec 6 – NN as Distribution Parameterizers |
| `Problems_VAE_Questions.pdf` | `Problems_VAE_Solutions.pdf` | Lec 6 – Variational Autoencoders |
| `Problems_Calibration_Conformal_Questions.pdf` | `Problems_Calibration_Conformal_Solutions.pdf` | Lec 7 – Calibration and Conformal Prediction |
| `Problems_Fusion_Questions.pdf` | `Problems_Fusion_Solution.pdf` | Lec 8 – Sensor Fusion |

---

## Problem Set Summaries

### Bayesian Modeling (`Problems_Bayesian_*`)
- **Part A (Conceptual):** Confidence intervals vs. credible intervals; aleatoric vs. epistemic uncertainty; conjugate priors; MLE vs. MAP.
- **Part B (Derivations):** Beta-Binomial conjugate update; MAP and MLE estimation from the derived posterior.
- **Part C (Parametric Analysis):** Comparing weak (Beta(1,1)) vs. strong (Beta(10,10)) priors; effect on posterior variance and predictive probabilities; appropriate priors for rare events.
- **Part D (Implementation):** Computing the posterior predictive distribution (PPD); implementing Bayesian updates in code.

### Gaussian Models (`Problems_Gaussian_Models_*`)
- **Part A:** Deriving precision additivity in the univariate conjugate Gaussian update; limiting cases of the precision-weighted mean (n→∞, σ₀²→0).
- **Part B:** Sequential Gaussian updates; learning rate dynamics; why the step size shrinks as more data arrive.
- **Part C:** Normal–Gamma conjugate family; interpreting hyperparameters κ₀, α₀, β₀; effect of prior–data discrepancy.
- **Part D (Multivariate):** Multivariate Gaussian conditionals; Schur complement; applying to sensor fusion.

### Linear Regression (`Problems_Linear_Regression_*`)
- Bayesian linear regression setup; Gaussian likelihood and conjugate prior.
- Deriving the posterior over weights; posterior mean and covariance.
- Predictive distribution for new inputs (mean and variance).
- Connection between MAP and ridge regression.
- Model comparison and marginal likelihood.

### Logistic Regression (`Problems_Logistic_Regression_*`)
- Logistic likelihood and Gaussian prior; why the posterior is intractable.
- Laplace approximation: MAP estimation and Hessian computation.
- Approximate predictive distribution for binary classification.
- Comparing Laplace approximation with MCMC on a classification task.

### MCMC (`Problems_MCMC_*`)
- Metropolis–Hastings: deriving the acceptance ratio; detailed balance proof.
- Designing proposal distributions for different target shapes.
- Convergence diagnostics: computing ESS, ACF, Gelman–Rubin R̂.
- Comparing random-walk MH with HMC in terms of mixing.

### Langevin Monte Carlo (`Problems_Langevin_*`)
- Drift–diffusion stochastic differential equations; Euler–Maruyama discretization.
- Deriving the stationary distribution of Langevin dynamics.
- ULA vs. MALA: bias of ULA and how Metropolis correction removes it.
- Applying MALA to Bayesian logistic regression.

### Kalman Filter (`Problems_Kalman_Filter_*`)
- Setting up a linear Gaussian SSM; predict and update equations.
- Deriving the Kalman gain; interpreting its limiting cases.
- 1D and 2D filtering examples with worked numerical solutions.
- Comparison of prediction uncertainty before and after measurement incorporation.

### Particle Filters (`Problems_Particle_Filter_*`)
- Sequential importance sampling; deriving unnormalized weights.
- SIR algorithm: propagation, weighting, and resampling steps.
- Diagnosing weight degeneracy; effective sample size (ESS) for particles.
- Comparing particle filter with Kalman filter on a nonlinear model.

### Variational Inference (`Problems_VI_*` and `SVI_*`)
- ELBO derivation: showing log p(x) = ELBO + KL(q||p).
- Mean-field approximation: factored variational family; CAVI update equations.
- Reparameterization trick: deriving low-variance pathwise gradients.
- Stochastic VI: mini-batch ELBO estimation; convergence analysis.

### NN Parameterization (`Problems_NN_Parameterization_*`)
- Designing Gaussian output heads for heteroscedastic regression.
- Mixture Density Networks: loss function, mode collapse issues.
- Distinguishing aleatoric and epistemic uncertainty in NN outputs.

### Variational Autoencoders (`Problems_VAE_*`)
- VAE generative model; encoder and decoder design.
- ELBO for VAEs: reconstruction term + KL to prior.
- Applying the reparameterization trick through the encoder.
- VAE for anomaly detection: using ELBO as anomaly score.

### Calibration and Conformal Prediction (`Problems_Calibration_Conformal_*`)
- Computing reliability diagrams and ECE from model outputs.
- Temperature scaling: finding the optimal T via validation set.
- Split conformal prediction: computing nonconformity scores and the (1−α)-quantile.
- Verifying marginal coverage; analyzing prediction set size (efficiency).

### Sensor Fusion (`Problems_Fusion_*`)
- Precision-weighted fusion for two Gaussian sensors; deriving the optimal fused estimate.
- Multivariate fusion with N sensors.
- Covariance Intersection: when and why to use it.
- Data association: applying gating and PDAF in a multi-target scenario.

---

## How to Use These Practice Questions

1. **Attempt the questions independently** before looking at the solutions.
2. **Align with lectures**: work through each problem set after attending the corresponding lecture and reviewing the lecture notebooks.
3. **Exam preparation**: the midterms and final exam draw heavily on the types of questions in these problem sets. Pay special attention to derivation questions (Part B) and conceptual questions (Part A).
4. **Cross-reference with assignments**: many assignment questions are extensions or variations of the practice problems.

---

## Course Assessment Alignment

| Assessment | Weight | Topics Covered |
|------------|--------|---------------|
| Assignments / Projects | 30% | All topics (practical implementation focus) |
| Midterm 1 (Feb 11) | 15% | Lec 1–3: Bayesian modeling, Gaussian models, linear/logistic regression |
| Midterm 2 (Mar 18) | 15% | Lec 4–6: State space models, filters, VI, VAEs, BNNs |
| Final Exam | 40% | All topics (comprehensive) |
