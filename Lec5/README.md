# Lecture 5 – Variational Inference

**Course:** ELG 5218 / CSI 5218 / EACJ 5600 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
**Date:** February 25, 2026  
**Instructor:** Miodrag Bolic, University of Ottawa

---

## Overview

This lecture introduces **Variational Inference (VI)** as a scalable alternative to MCMC for
approximate Bayesian inference. Rather than sampling from the posterior, VI casts inference as
an **optimization problem**: find a tractable distribution `q(θ)` that is as close as possible
to the true posterior `p(θ|x)`, measured by the KL divergence. The lecture builds from
gradient descent and automatic differentiation up to **Stochastic (Black-box) VI** and the
**reparameterization trick** used in modern deep generative models.

---

## Learning Goals

By the end of this lecture you should be able to:

- Explain why variational inference is often preferred over MCMC for large-scale problems.
- Define the **Evidence Lower BOund (ELBO)** and show how maximizing it minimizes `KL(q||p)`.
- Apply the **mean-field approximation** to factorize the variational family.
- Derive coordinate ascent VI (CAVI) updates for simple models.
- Use **stochastic (black-box) VI** with the REINFORCE / score-function gradient estimator.
- Apply the **reparameterization trick** for low-variance pathwise gradient estimation.
- Implement variational inference using NumPyro / Pyro (SVI API).

---

## Topics Covered

### Variational Inference (`VariationalInference.pdf`)

**Part 1 – Gradient Machinery**
- Gradient descent and stochastic gradient descent (SGD).
- Stochastic optimization: mini-batches, learning rate schedules.
- **Automatic differentiation (autodiff)**: forward and reverse mode; JAX/PyTorch.

**Part 2 – KL Divergence and the ELBO**
- KL divergence `KL(q||p)`: definition, non-symmetry, non-negativity.
- The ELBO as a lower bound on the log marginal likelihood log p(x).
- ELBO decomposition: expected log-likelihood minus KL to prior.
- Tightening the bound: better `q` → higher ELBO → closer approximation.

**Part 3 – Variational Families**
- Mean-field approximation: fully factored `q(θ) = ∏ q_i(θ_i)`.
- Coordinate Ascent Variational Inference (CAVI) updates.
- Structured / normalizing flow families for richer approximations.

**Part 4 – Stochastic (Black-box) VI**
- Replacing exact expectations with Monte Carlo estimates.
- Score-function (REINFORCE) gradient estimator; high variance issue.
- Control variates and baselines for variance reduction.

**Part 5 – Reparameterization and Pathwise Gradients**
- Reparameterization trick: sample `ε ~ p(ε)`, set `θ = g(ε, φ)` for differentiable sampling.
- Low-variance pathwise gradient of the ELBO.
- Why this is the key enabler for Variational Autoencoders (Lec 6).

---

## Materials

| File | Description |
|------|-------------|
| `VariationalInference.pdf` | Lecture slides: Variational inference end-to-end (64 slides) |
| `From_entropy_to_SVI.ipynb` | Notebook: From entropy and KL divergence to Stochastic VI |

---

## Recommended Reading

- **Villani, Mattias (2025). *Bayesian Learning*.** – Relevant variational inference chapters.
  [PDF](https://github.com/mattiasvillani/BayesianLearningBook/raw/main/pdf/BayesBook.pdf)
- **Stochastic Variational Inference with NumPyro:**
  [SVI Tutorial](https://phuijse.github.io/BLNNbook/chapters/variational/svi.html)

---

## Practice Questions

Relevant practice sets from the `Practice Questions/` folder:

- `Problems_VI_Questions.pdf` / `Problems_VI_Solutions.pdf` – ELBO derivation, mean-field approximation, CAVI updates, reparameterization trick.
- `SVI_Problems.pdf` / `SVI_solutions.pdf` – Stochastic variational inference implementation and analysis problems.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| Variational inference | Approximate posterior inference via optimization instead of sampling |
| ELBO | Evidence Lower BOund; surrogate objective for log p(x); ELBO = log p(x) − KL(q\|\|p) |
| KL divergence | Measure of difference between two distributions; KL(q\|\|p) ≥ 0 |
| Mean-field | Variational family where parameters are mutually independent |
| CAVI | Coordinate Ascent VI: iteratively optimize each variational factor |
| Score-function gradient | Unbiased but high-variance gradient estimator for discrete/non-reparameterizable models |
| Reparameterization trick | Express samples as deterministic function of noise; enables low-variance gradients |
| Autodiff | Automatic differentiation; computes exact gradients of any differentiable computation graph |
| SVI | Stochastic VI using mini-batch data and Monte Carlo gradient estimates |
