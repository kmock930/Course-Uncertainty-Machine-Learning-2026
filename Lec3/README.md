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

---

## Key Derivations

### 1. Laplace Approximation

Goal: approximate the intractable posterior `p(θ|x)` near the MAP estimate `θ̂`.

**Step 1 — Taylor expand** log-posterior around `θ̂`:
```
log p(θ|x) ≈ log p(θ̂|x) + (1/2)(θ−θ̂)ᵀ H (θ−θ̂)
```
where `H = ∇²_θ log p(θ|x)|_{θ̂}` (Hessian; negative definite at a mode).

**Step 2 — Exponentiate:**
```
p(θ|x) ≈ N(θ̂, −H⁻¹)
```

The covariance is `Σ = −H⁻¹` — curvature of the log-posterior (flatter ↔ more uncertainty).

**Step 3 — Predictive for logistic regression** (probit approximation):
```
p(y*=1|x*) ≈ σ(x*ᵀθ̂ / √(1 + π/8 · x*ᵀΣx*))
```

### 2. Metropolis–Hastings: Detailed Balance

For the chain to have stationary distribution `π(θ) = p(θ|x)`, we need detailed balance:
```
π(θ) T(θ→θ') = π(θ') T(θ'→θ)
```

where `T(θ→θ') = q(θ'|θ) · A(θ, θ')` is the transition kernel.

Setting `A(θ, θ') = min(1, r)` with:
```
r = π(θ')q(θ|θ') / (π(θ)q(θ'|θ))
  = p(θ'|x) q(θ|θ') / (p(θ|x) q(θ'|θ))
```

**Key cancellation:** `p(x)` cancels in the ratio `p(θ'|x)/p(θ|x)` → only unnormalised posteriors needed.

Verification of detailed balance:
```
π(θ) q(θ'|θ) min(1, r) = min(π(θ)q(θ'|θ), π(θ')q(θ|θ'))
                        = π(θ') q(θ|θ') min(1, 1/r)
```
Both sides equal `min(π(θ)q(θ'|θ), π(θ')q(θ|θ'))`. ✓

### 3. Langevin SDE → Euler–Maruyama Discretisation

**Continuous SDE** with stationary distribution `π(θ) ∝ exp(−U(θ))`:
```
dθ = −(1/2)∇U(θ) dt + dW_t = (1/2)∇ log π(θ) dt + dW_t
```

**Euler–Maruyama** (step size h):
```
θ_{k+1} = θ_k + (h/2)∇ log π(θ_k) + √h · ε_k,   ε_k ~ N(0, I)
```

This is **ULA** — discretisation error makes it biased (does not leave π exactly).

**MALA** adds a Metropolis–Hastings correction with proposal:
```
q(θ'|θ) = N(θ + (h/2)∇ log π(θ), h I)
```

Acceptance ratio:
```
A = min(1,  π(θ') q(θ|θ') / (π(θ) q(θ'|θ)))
```

MALA is unbiased — the correction removes the discretisation error at the cost of potential rejections.
