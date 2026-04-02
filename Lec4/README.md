# Lecture 4 – State Space Models, Kalman Filter, and Particle Filters

> **Revision summary** | ELG 5218 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
> Slides: `ssm_slides.pdf`, `Kalman_filter_slides.pdf`, `Particle_filter_slides.pdf` · Notebooks: `ssm_examples.ipynb`, `bearings_only_particle_filter.ipynb`  
> Reading: Villani Ch. 17; Sarkka Ch. 4, 7; Labbe (Kalman Filters in Python)

---

## Core Ideas

### State Space Models (SSMs)
- Model systems with **hidden state** `zₜ` and **noisy observations** `yₜ`:
  - **Transition model:** `p(zₜ | zₜ₋₁)` — how the state evolves
  - **Measurement model:** `p(yₜ | zₜ)` — how the state generates observations
- For linear Gaussian SSMs: `zₜ = Azₜ₋₁ + Bwₜ`, `yₜ = Czₜ + vₜ`, where `w,v` are zero-mean Gaussian noise.

### Kalman Filter (exact for linear Gaussian SSMs)
Two recursive steps per time step:

**Predict:**
- `z̃ₜ = A μₜ₋₁`
- `P̃ₜ = A Pₜ₋₁ Aᵀ + Q` (Q = process noise covariance)

**Update:**
- Innovation: `νₜ = yₜ − C z̃ₜ`
- **Kalman gain:** `Kₜ = P̃ₜ Cᵀ (C P̃ₜ Cᵀ + R)⁻¹` (R = measurement noise covariance)
- `μₜ = z̃ₜ + Kₜ νₜ`
- `Pₜ = (I − Kₜ C) P̃ₜ`

Interpretation of K:
- K → I: trust measurement completely (R ≈ 0)
- K → 0: trust prediction completely (R → ∞)

Extensions: **EKF** (linearise via Jacobians), **UKF** (sigma-point transform), **RTS smoother** (backward pass for offline smoothing).

### Particle Filter (nonlinear / non-Gaussian SSMs)
- Represent `p(zₜ | y₁:ₜ)` with `N` weighted particles `{zₜ⁽ⁱ⁾, wₜ⁽ⁱ⁾}`.
- **Sequential Importance Resampling (SIR) — three steps each time step:**
  1. **Propagate:** `zₜ⁽ⁱ⁾ ~ p(zₜ | zₜ₋₁⁽ⁱ⁾)`
  2. **Weight:** `wₜ⁽ⁱ⁾ ∝ p(yₜ | zₜ⁽ⁱ⁾)` (update by likelihood)
  3. **Resample:** draw N particles with replacement according to weights (avoids degeneracy)
- Effective sample size (ESS) = `1 / Σᵢ (wᵢ)²` — resample when ESS falls below threshold.

---

## Things to Remember

- Kalman filter is **optimal** (minimum MSE) only for linear Gaussian SSMs.
- Particle filter is approximate but works for any SSM; accuracy improves with more particles.
- Without resampling, particles **degenerate** — all weight concentrates on a single particle.
- The Kalman filter is a special case of Bayesian filtering where all distributions are Gaussian.

---

## Practice Questions

- [`../Practice Questions/Problems_Kalman_Filter_Questions.pdf`](../Practice%20Questions/Problems_Kalman_Filter_Questions.pdf) — Predict/update derivations, Kalman gain interpretation, 1D/2D examples
- [`../Practice Questions/Problems_Particle_Filter_Questions.pdf`](../Practice%20Questions/Problems_Particle_Filter_Questions.pdf) — SIR steps, importance weights, degeneracy, ESS

---

## Key Derivations

### 1. Kalman Filter as Bayesian Gaussian Fusion

Prior on state: `p(zₜ) = N(z̃ₜ, P̃ₜ)` (prediction step output)  
Likelihood: `p(yₜ|zₜ) = N(Czₜ, R)`

Log-posterior (complete the square in zₜ):
```
log p(zₜ|yₜ) = −(1/2)(zₜ−z̃ₜ)ᵀ P̃ₜ⁻¹ (zₜ−z̃ₜ) − (1/2)(yₜ−Czₜ)ᵀ R⁻¹ (yₜ−Czₜ) + const
```

Collecting zₜ² terms (precision matrix):
```
Pₜ⁻¹ = P̃ₜ⁻¹ + CᵀR⁻¹C
```

Collecting zₜ terms (information vector):
```
Pₜ⁻¹ μₜ = P̃ₜ⁻¹ z̃ₜ + CᵀR⁻¹yₜ
```

Solving for mean gives the standard Kalman update, and applying the **Woodbury identity** to the posterior covariance yields the Kalman gain form:
```
Kₜ = P̃ₜ Cᵀ (C P̃ₜ Cᵀ + R)⁻¹
μₜ = z̃ₜ + Kₜ(yₜ − Czₜ)
Pₜ = (I − KₜC) P̃ₜ
```

### 2. Kalman Gain Minimises MSE

Posterior mean is the **MMSE** estimator. The gain `K` is derived by minimising `tr(Pₜ)`:

```
∂/∂K tr(Pₜ) = 0
```

where `Pₜ = (I−KC)P̃ₜ(I−KC)ᵀ + KRKᵀ` (Joseph form for numerical stability).

Setting the derivative to zero:
```
−P̃ₜCᵀ + K(CP̃ₜCᵀ + R) = 0
K* = P̃ₜCᵀ (CP̃ₜCᵀ + R)⁻¹
```

### 3. Particle Filter Importance Weights Derivation

We want to approximate `p(z₁:ₜ | y₁:ₜ)` using samples from a proposal `q(z₁:ₜ | y₁:ₜ)`.

**Importance weight:**
```
wₜ⁽ⁱ⁾ ∝ p(z₁:ₜ⁽ⁱ⁾ | y₁:ₜ) / q(z₁:ₜ⁽ⁱ⁾ | y₁:ₜ)
```

With the **bootstrap filter** (proposal = transition prior):
```
q(zₜ|z₁:ₜ₋₁, y₁:ₜ) = p(zₜ|zₜ₋₁)
```

Sequential weight update:
```
wₜ⁽ⁱ⁾ ∝ wₜ₋₁⁽ⁱ⁾ · p(yₜ | zₜ⁽ⁱ⁾)
```

Only the likelihood term `p(yₜ|zₜ)` is needed — the transition probability cancels with the proposal. This is why only the measurement model is needed for weighting.

### 4. Prediction Step Covariance

From `zₜ = Azₜ₋₁ + wₜ`, `wₜ ~ N(0, Q)`:
```
P̃ₜ = Var[zₜ] = A Var[zₜ₋₁] Aᵀ + Var[wₜ] = A Pₜ₋₁ Aᵀ + Q
```

The noise term Q adds uncertainty every time step — uncertainty grows during prediction and shrinks during updates.
