# Lecture 3 – Bayesian Logistic Regression, MCMC, and Langevin Monte Carlo

> **Revision summary** | ELG 5218 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
> Slides: `LogisticRegression.pdf`, `MCMC.pdf`, `LangevinMC.pdf` · Notebooks: `Laplace_Approximation_Logistic_Regression.ipynb`, `Lec3a_python_numpyro.ipynb`, `langevin_MALA.ipynb`  
> Reading: Villani Ch. 8.2, 10; Friedman (2022) Langevin blog

---

## Core Ideas

### Bayesian Logistic Regression
- Model: `p(y=1|x,w) = σ(wᵀx)` where `σ` is the sigmoid; prior `w ~ N(0, τ²I)`
- **Problem:** the posterior is non-conjugate → no closed-form solution.
- **Laplace approximation** — two steps:
  1. Find MAP: `ŵ = argmax log p(w|y)` via gradient ascent.
  2. Approximate posterior as Gaussian: `q(w) = N(ŵ, H⁻¹)` where `H = −∇²log p(w|y)|_{ŵ}`
- Predictive: `p(y*=1|x*) ≈ ∫ σ(wᵀx*) q(w) dw` (approximated via probit).

### Markov Chain Monte Carlo (MCMC)
- Goal: draw samples from `p(θ|x)` without computing the intractable evidence `p(x)`.
- **Metropolis–Hastings:**
  1. Propose `θ' ~ q(θ'|θ)`
  2. Accept with probability `min(1, p(θ'|x)q(θ|θ') / p(θ|x)q(θ'|θ))` — evidence cancels.
- **HMC:** introduces momentum; uses gradient `∇ log p(θ|x)` to propose distant moves with high acceptance.
- **NUTS:** adaptive HMC — no manual tuning of path length.
- **Convergence diagnostics:**
  - Burn-in: discard initial samples.
  - ACF (autocorrelation function): check mixing.
  - ESS (effective sample size): accounts for autocorrelation.
  - **Gelman–Rubin R̂:** run multiple chains; R̂ ≈ 1.0 indicates convergence.

### Langevin Monte Carlo
- Continuous Langevin SDE: `dθ = ½ ∇ log π(θ) dt + dW` — stationary distribution is `π(θ)`.
- Potential: `U(θ) = −log π(θ)` (negative log-posterior); drift pushes toward high probability.
- **ULA (Unadjusted Langevin):** Euler–Maruyama discretisation; **biased** due to step-size error.
- **MALA:** adds Metropolis–Hastings correction step to ULA → removes bias.
- MALA vs HMC: MALA simpler to implement; HMC usually mixes faster for high-dimensional posteriors.

---

## Things to Remember

- Logistic regression likelihood is **not conjugate** with any standard prior → approximation needed.
- In M-H, the evidence `p(x)` cancels in the acceptance ratio — this is the key insight.
- HMC/NUTS should be preferred over random-walk M-H for continuous posteriors in practice.
- MALA = gradient-informed random walk + correction; more efficient than plain random-walk M-H.

---

## Practice Questions

- [`../Practice Questions/Problems_Logistic_Regression_Questions.pdf`](../Practice%20Questions/Problems_Logistic_Regression_Questions.pdf) — Laplace approximation derivation, predictive distribution
- [`../Practice Questions/Problems_MCMC_Questions.pdf`](../Practice%20Questions/Problems_MCMC_Questions.pdf) — Metropolis–Hastings, detailed balance, diagnostics
- [`../Practice Questions/Problems_Langevin_Questions.pdf`](../Practice%20Questions/Problems_Langevin_Questions.pdf) — Langevin SDE, ULA vs MALA, discretisation bias
