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

---

## Key Derivations

### 1. Precision Additivity via Completing the Square

Prior: `p(μ) ∝ exp(−λ₀(μ−μ₀)²/2)`, Likelihood: `p(x|μ) ∝ exp(−λΣ(xᵢ−μ)²/2)`

Log-posterior (ignoring constants):
```
log p(μ|x) = −(λ₀/2)(μ−μ₀)² − (λ/2)Σ(xᵢ−μ)²
```

Expanding and collecting μ² and μ terms:
```
= −(1/2)(λ₀ + nλ)μ² + (λ₀μ₀ + nλx̄)μ + const
```

Completing the square:
```
= −(λₙ/2)(μ − μₙ)² + const
```

where **λₙ = λ₀ + nλ** (precisions add) and **μₙ = (λ₀μ₀ + nλx̄) / λₙ** (precision-weighted mean).

### 2. Bayesian Linear Regression Posterior

Model: `y = Xw + ε`, `p(y|X,w) = N(Xw, σ²I)`, prior `p(w) = N(0, τ²I)`

Log-posterior:
```
log p(w|y) = −(1/2σ²)||y − Xw||² − (1/2τ²)||w||² + const
```

Completing the square in w:
```
= −(1/2)(w − μ_w)ᵀ Σ_w⁻¹ (w − μ_w) + const
```

where:
- `Σ_w⁻¹ = (1/σ²)XᵀX + (1/τ²)I`
- `μ_w = (1/σ²) Σ_w Xᵀy`

Setting `λ = σ²/τ²`: `μ_w = (XᵀX + λI)⁻¹ Xᵀy` = **Ridge regression** solution.

### 3. Bayesian Predictive Distribution

For new input `x*`:
```
p(y*|x*, y) = ∫ p(y*|x*, w) p(w|y) dw
```

Since both factors are Gaussian, the integral is also Gaussian:
```
p(y*|x*, y) = N(x*ᵀ μ_w,  x*ᵀ Σ_w x* + σ²)
```

- `x*ᵀ μ_w` — prediction from posterior mean weights
- `x*ᵀ Σ_w x*` — **epistemic** variance (weight uncertainty)
- `σ²` — **aleatoric** variance (irreducible noise)

### 4. Conditional Gaussian (Schur Complement)

Given joint: `[x; y] ~ N([μₓ; μᵧ], [[Σₓₓ, Σₓᵧ]; [Σᵧₓ, Σᵧᵧ]])`

Condition on y by completing the square in x:
```
p(x|y) = N(μₓ + Σₓᵧ Σᵧᵧ⁻¹(y−μᵧ),  Σₓₓ − Σₓᵧ Σᵧᵧ⁻¹ Σᵧₓ)
```

The term `Σₓₓ − Σₓᵧ Σᵧᵧ⁻¹ Σᵧₓ` is the **Schur complement** — always ≤ Σₓₓ (conditioning reduces uncertainty).
