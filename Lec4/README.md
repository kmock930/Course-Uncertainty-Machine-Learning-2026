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
