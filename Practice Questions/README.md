# Practice Questions

> ELG 5218 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
> Each topic has a **Questions** PDF and a **Solutions** PDF. Work through questions independently before checking solutions.

---

## File Index

| Questions | Solutions | Lecture |
|-----------|-----------|---------|
| `Problems_Bayesian_Questions.pdf` | `Problems_Bayesian_Solutions.pdf` | [Lec 1 – Bayesian Modeling](../Lec1) |
| `Problems_Gaussian_Models_Questions.pdf` | `Problems_Gaussian_Models_Solutions.pdf` | [Lec 2 – Gaussian Models](../Lec2) |
| `Problems_Linear_Regression_Questions.pdf` | `Problems_Linear_Regression_Solutions.pdf` | [Lec 2 – Bayesian Linear Regression](../Lec2) |
| `Problems_Logistic_Regression_Questions.pdf` | `Problems_Logistic_Regression_Solutions.pdf` | [Lec 3 – Bayesian Logistic Regression](../Lec3) |
| `Problems_MCMC_Questions.pdf` | `Problems_MCMC_Solutions.pdf` | [Lec 3 – MCMC](../Lec3) |
| `Problems_Langevin_Questions.pdf` | `Problems_Langevin_Solutions.pdf` | [Lec 3 – Langevin / MALA](../Lec3) |
| `Problems_Kalman_Filter_Questions.pdf` | `Problems_Kalman_Filter_Solutions.pdf` | [Lec 4 – Kalman Filter](../Lec4) |
| `Problems_Particle_Filter_Questions.pdf` | `Problems_Particle_Filter_Solutions.pdf` | [Lec 4 – Particle Filters](../Lec4) |
| `Problems_VI_Questions.pdf` | `Problems_VI_Solutions.pdf` | [Lec 5 – Variational Inference](../Lec5) |
| `SVI_Problems.pdf` | `SVI_solutions.pdf` | [Lec 5 – Stochastic VI](../Lec5) |
| `Problems_NN_Parameterization_Questions.pdf` | `Problems_NN_Parameterization_Solutions.pdf` | [Lec 6 – NN Parameterization](../Lec6) |
| `Problems_VAE_Questions.pdf` | `Problems_VAE_Solutions.pdf` | [Lec 6 – VAEs](../Lec6) |
| `Problems_Calibration_Conformal_Questions.pdf` | `Problems_Calibration_Conformal_Solutions.pdf` | [Lec 7 – Calibration & Conformal Prediction](../Lec7) |
| `Problems_Fusion_Questions.pdf` | `Problems_Fusion_Solution.pdf` | [Lec 8 – Sensor Fusion](../Lec8) |

---

## What Each Problem Set Tests

**Bayesian Modeling** — Beta-Binomial conjugate update; MAP vs MLE; credible intervals; posterior predictive distribution.

**Gaussian Models** — Precision additivity derivation; precision-weighted mean (limiting cases); sequential/online updates; Normal–Gamma hyperparameter interpretation; multivariate conditionals.

**Linear Regression** — Bayesian regression posterior; predictive distribution; MAP = ridge regression; marginal likelihood for model comparison.

**Logistic Regression** — Sigmoid likelihood; Laplace approximation (MAP + Hessian); approximate predictive distribution.

**MCMC** — Metropolis–Hastings acceptance ratio; detailed balance; convergence diagnostics (ESS, R̂); HMC vs random-walk comparison.

**Langevin / MALA** — Langevin SDE stationary distribution; Euler–Maruyama discretisation; ULA bias; MALA correction; comparison with HMC.

**Kalman Filter** — Predict and update derivations; Kalman gain interpretation; 1D/2D numerical examples; EKF linearisation.

**Particle Filters** — Importance sampling weights; SIR algorithm steps; weight degeneracy; effective sample size (ESS); comparison with Kalman filter on nonlinear model.

**Variational Inference** — ELBO derivation; mean-field / CAVI updates; reparameterization trick; MCMC vs VI comparison.

**Stochastic VI** — Mini-batch ELBO estimation; score-function vs reparameterization gradients; variance analysis.

**NN Parameterization** — Gaussian output heads; MDN loss function; heteroscedastic regression; aleatoric vs epistemic uncertainty.

**VAEs** — VAE ELBO (reconstruction + KL); reparameterization; latent space properties; anomaly scoring.

**Calibration & Conformal Prediction** — ECE computation; reliability diagrams; temperature scaling; split conformal algorithm; marginal coverage proof.

**Sensor Fusion** — Precision-weighted fusion derivation; information form; covariance intersection; gating; PDAF soft association.

---

## Assessment Reference

| Component | Weight | Topics Tested |
|-----------|--------|---------------|
| Assignments / Projects | 30% | Practical implementation across all topics |
| Midterm 1 (Feb 11) | 15% | Lec 1–3 (Bayesian modeling, Gaussians, MCMC) |
| Midterm 2 (Mar 18) | 15% | Lec 4–7 (Filters, VI, VAEs, Calibration) |
| Final Exam | 40% | Comprehensive — all lectures |
