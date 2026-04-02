# Lecture 3 – Bayesian Logistic Regression, MCMC, and Langevin Monte Carlo

**Course:** ELG 5218 / CSI 5218 / EACJ 5600 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
**Date:** February 1, 2026  
**Instructor:** Miodrag Bolic, University of Ottawa

---

## Overview

This lecture tackles **Bayesian classification** via logistic regression, where the
non-conjugate likelihood makes exact posterior inference intractable. Three approximate
inference techniques are presented in increasing sophistication:
the **Laplace approximation** (local Gaussian fit at the MAP),
**Markov Chain Monte Carlo (MCMC)** sampling, and
**Langevin Monte Carlo / MALA** (gradient-informed MCMC via diffusion processes).

---

## Learning Goals

By the end of this lecture you should be able to:

- Set up the Bayesian logistic regression model and explain why the posterior is intractable.
- Apply the **Laplace approximation**: find the MAP, compute the Hessian, and use the resulting Gaussian approximation for prediction.
- Describe the fundamental MCMC problem: computing the evidence integral.
- Implement **Metropolis–Hastings** sampling and diagnose convergence (burn-in, ACF, ESS, Gelman–Rubin R̂).
- Explain **Hamiltonian Monte Carlo (HMC)** and **NUTS** and why they outperform random-walk MH.
- Derive **Langevin dynamics** from drift–diffusion stochastic differential equations.
- Implement the **Unadjusted Langevin Algorithm (ULA)** and the **Metropolis-Adjusted Langevin Algorithm (MALA)**.
- Compare MALA, random-walk MH, and HMC on a Bayesian classification task.

---

## Topics Covered

### Bayesian Logistic Regression (`LogisticRegression.pdf`)
- Logistic regression as a discriminative model; sigmoid likelihood.
- Gaussian prior on weights; non-conjugacy makes posterior intractable.
- **Laplace Approximation**: MAP via gradient ascent + Hessian-based Gaussian approximation.
- Approximate predictive distribution for classification.
- Application: failure prediction in engineering systems.
- Implementation: Laplace approximation + MCMC with NumPyro.

### Markov Chain Monte Carlo (`MCMC.pdf`)
- The fundamental challenge: high-dimensional evidence integral.
- Monte Carlo estimation and why sampling beats numerical integration.
- Markov chains: stationarity, detailed balance, ergodicity.
- **Metropolis–Hastings**: proposal, acceptance ratio, evidence cancellation.
- **Hamiltonian Monte Carlo (HMC)**: leapfrog integrator, momentum variables.
- **NUTS** (No-U-Turn Sampler): adaptive path length.
- Convergence diagnostics: burn-in, autocorrelation function (ACF), effective sample size (ESS), Gelman–Rubin R̂.

### Langevin Monte Carlo (`LangevinMC.pdf`)
- Drift–diffusion processes in one dimension: pure diffusion → pure drift → combined.
- **Langevin diffusion**: drift derived from a potential U(θ) = −log π(θ).
- Why Langevin dynamics samples from the target distribution π(θ).
- Discretization: Euler–Maruyama scheme; discretization bias in ULA.
- **MALA**: adds Metropolis correction to ULA to remove bias.
- Practical comparison: MALA vs. random-walk MH vs. HMC on Bayesian logistic regression.
- Convergence verification with standard diagnostics.

---

## Materials

| File | Description |
|------|-------------|
| `LogisticRegression.pdf` | Lecture slides: Bayesian logistic regression and Laplace approximation (28 slides) |
| `MCMC.pdf` | Lecture slides: Markov Chain Monte Carlo sampling (40 slides) |
| `LangevinMC.pdf` | Lecture slides: Drift–diffusion processes and Langevin Monte Carlo (30 slides) |
| `Laplace_Approximation_Logistic_Regression.ipynb` | Notebook: Laplace approximation + logistic regression |
| `Lec3a_python_numpyro.ipynb` | Notebook: Markov chains and MCMC with NumPyro |
| `langevin_MALA.ipynb` | Notebook: Drift–diffusion processes and MALA algorithm |
| `slide_data_plot.png` | Supporting figure |
| `slide_posterior_weights.png` | Supporting figure |
| `slide_predictive_surface.png` | Supporting figure |

---

## Recommended Reading

- **Villani, Mattias (2025). *Bayesian Learning*.** – Chapters 8.2 and 10.
  [PDF](https://github.com/mattiasvillani/BayesianLearningBook/raw/main/pdf/BayesBook.pdf)
- **Friedman, Roy (2022). *A Simplified Overview of Langevin Dynamics*.**
  [Blog post](https://friedmanroy.github.io/blog/2022/Langevin/)

---

## Practice Questions

Relevant practice sets from the `Practice Questions/` folder:

- `Problems_Logistic_Regression_Questions.pdf` / `Problems_Logistic_Regression_Solutions.pdf` – Logistic likelihood, Laplace approximation derivations, predictive distributions.
- `Problems_MCMC_Questions.pdf` / `Problems_MCMC_Solutions.pdf` – Metropolis–Hastings, detailed balance, convergence diagnostics.
- `Problems_Langevin_Questions.pdf` / `Problems_Langevin_Solutions.pdf` – Drift–diffusion SDEs, ULA vs. MALA, Langevin dynamics derivations.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| Logistic sigmoid | Maps linear scores to probabilities for binary classification |
| Intractable posterior | No closed-form when likelihood and prior are not conjugate |
| Laplace approximation | Gaussian fit to the posterior at the MAP point using the Hessian |
| MAP estimate | Mode of the posterior; found via gradient-based optimization |
| Metropolis–Hastings | Accept/reject MCMC sampler; evidence term cancels in acceptance ratio |
| HMC | Gradient-informed MCMC using Hamiltonian dynamics; low correlation between samples |
| NUTS | Adaptive HMC variant; automatically sets path length |
| Gelman–Rubin R̂ | Multi-chain convergence diagnostic; R̂ ≈ 1 indicates convergence |
| Langevin SDE | Stochastic differential equation whose stationary distribution is the target |
| MALA | Metropolis-corrected Langevin sampler; removes discretization bias |
