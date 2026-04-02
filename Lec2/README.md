# Lecture 2 – Bayesian Inference for Gaussians and Bayesian Linear Regression

> **Revision summary** | ELG 5218 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
> Slides: `GaussianModels.pdf`, `GaussianFormulas.pdf`, `tpmi_w19_lec5_slides_print.pdf` · Notebook: `Gaussian_Models_Final.ipynb`  
> Reading: Villani Ch. 2, 3.3, 5

---

## Core Ideas

### Gaussian Conjugate Update (known variance σ²)

- Prior: `μ ~ N(μ₀, σ₀²)`, Likelihood: `xᵢ ~ N(μ, σ²)`
- **Precisions add:** `λₙ = λ₀ + n·λ` where `λ = 1/σ²`
- **Posterior mean** = precision-weighted average:

  `μₙ = (λ₀μ₀ + nλx̄) / (λ₀ + nλ)`

- As `n → ∞`: posterior concentrates on MLE `x̄` (data dominates).
- As `σ₀² → 0` (very strong prior): posterior stays near `μ₀`.

### Sequential (Online) Updates
- Process one observation at a time; previous posterior becomes the new prior.
- Learning rate `wₙ = λ / (λₙ₋₁ + λ)` decreases as `n` grows — model becomes more certain.

### Unknown Variance — Normal–Gamma Family
- Joint conjugate prior: `(μ, λ) ~ NormalGamma(μ₀, κ₀, α₀, β₀)`
- After `n` observations: `κₙ = κ₀ + n`, `αₙ = α₀ + n/2`
- Marginal for `μ` is **Student-t** (heavier tails → robust to outliers)
- `κ₀` acts as an effective prior sample size for the mean.

### Multivariate Gaussian – Key Identities
- **Marginal:** `p(x) = N(μₓ, Σₓₓ)` — just slice the mean/covariance block.
- **Conditional:** `p(x|y) = N(μₓ + Σₓᵧ Σᵧᵧ⁻¹(y−μᵧ), Σₓₓ − Σₓᵧ Σᵧᵧ⁻¹ Σᵧₓ)` — Schur complement shrinks variance.

### Bayesian Linear Regression
- Model: `y = Xw + ε`, `ε ~ N(0, σ²I)`, prior `w ~ N(0, τ²I)`
- **Posterior:** `w|y ~ N(μ_w, Σ_w)`:
  - `Σ_w⁻¹ = (1/σ²) XᵀX + (1/τ²) I`
  - `μ_w = (1/σ²) Σ_w Xᵀy`
- **MAP = Ridge regression** with `λ = σ²/τ²`
- **Predictive distribution:** `p(y*|x*, y) = N(x*ᵀμ_w, x*ᵀΣ_w x* + σ²)` — captures both weight uncertainty and noise.

---

## Things to Remember

- **Precision** (not variance) is additive in Bayesian Gaussian updates.
- Posterior mean always lies between the prior mean and the MLE.
- Bayesian linear regression yields **prediction intervals**, not just a point estimate.
- Marginalising out unknown precision gives heavier-tailed Student-t distributions — more robust.

---

## Practice Questions

- [`../Practice Questions/Problems_Gaussian_Models_Questions.pdf`](../Practice%20Questions/Problems_Gaussian_Models_Questions.pdf) — Precision additivity, online updates, Normal–Gamma
- [`../Practice Questions/Problems_Linear_Regression_Questions.pdf`](../Practice%20Questions/Problems_Linear_Regression_Questions.pdf) — Posterior derivation, predictive distribution, MAP vs MLE
