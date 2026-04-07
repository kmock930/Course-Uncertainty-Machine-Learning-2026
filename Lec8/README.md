# Lecture 8 – Sensor Fusion and Data Association

> **Revision summary** | ELG 5218 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
> Slides: `Sensor_Fusion.pdf` · Notebooks: `notebook_traditional_fusion.ipynb`, `ci_vs_kf.ipynb`  
> Reading: Li & Jilkov (2001), Survey of maneuvering target tracking III

---

## Core Ideas

### Optimal Gaussian Fusion (MLE)
- Two sensors with measurements `y₁ ~ N(z, σ₁²)` and `y₂ ~ N(z, σ₂²)`.
- **Fused estimate** = precision-weighted mean:

  `ẑ = (y₁/σ₁² + y₂/σ₂²) / (1/σ₁² + 1/σ₂²)`
  `1/σ²_fused = 1/σ₁² + 1/σ₂²`

- Fused precision = sum of individual precisions → more sensors = more information = smaller uncertainty.
- **Entropy view:** fusion reduces entropy; each independent sensor adds information.

### Bayesian Fusion (Kalman Filter as Fusion)
- Each Kalman filter update step is exactly one measurement fusion step.
- **Information form:** `Λₙ = Λₙ₋₁ + CᵀR⁻¹C`, `ηₙ = ηₙ₋₁ + CᵀR⁻¹y` — precision and information vector add.

### Covariance Intersection (CI)
- Used when **cross-covariances between sensors are unknown**.
- Fused covariance: `P_fused⁻¹ = ω P₁⁻¹ + (1−ω) P₂⁻¹`, `ω ∈ [0,1]`
- Optimise `ω` to minimise `tr(P_fused)` or `det(P_fused)`.
- CI is **conservative** — never overestimates precision even with correlations.

### Data Association
The problem of assigning measurements to tracks when multiple targets exist.

| Algorithm | Key idea | Limitation |
|---|---|---|
| **GNN** | Hard 1-to-1 assignment; nearest Mahalanobis | Fragile with clutter/close targets |
| **PDAF** | Soft weights over gated measurements; covariance spread term | Single target only |
| **JPDA** | Joint association events over all targets | Exponential complexity |
| **MHT** | Maintain hypothesis tree; defer decisions | High memory; pruning needed |

- **Gating:** discard measurement–track pairs where Mahalanobis distance exceeds chi-squared threshold.

---

## Things to Remember

- Ignoring correlated sensor noise leads to **overconfident** (too small) fused covariance → use CI when correlations are unknown.
- PDAF adds an extra term to the covariance to account for measurement origin uncertainty.
- GNN is fast but fails when targets are close; JPDA and MHT handle ambiguous scenarios better.
- Kalman filter's update step is exactly Bayesian Gaussian fusion (same precision-additivity formula).

---

## Practice Questions

[`../Practice Questions/Problems_Fusion_Questions.pdf`](../Practice%20Questions/Problems_Fusion_Questions.pdf) — Precision-weighted fusion derivation, multivariate fusion, CI algorithm, PDAF scenario

---

## Key Derivations

### 1. Optimal Gaussian Fusion (MLE/MAP Derivation)

Two sensors: `y₁ ~ N(z, σ₁²)`, `y₂ ~ N(z, σ₂²)`, independent.

Log-likelihood:
```
log p(y₁, y₂|z) = −(y₁−z)²/(2σ₁²) − (y₂−z)²/(2σ₂²) + const
```

Setting `∂/∂z = 0`:
```
(y₁−z)/σ₁² + (y₂−z)/σ₂² = 0
z(1/σ₁² + 1/σ₂²) = y₁/σ₁² + y₂/σ₂²
ẑ = (y₁/σ₁² + y₂/σ₂²) / (1/σ₁² + 1/σ₂²)
```

Fused precision: `1/σ²_fused = 1/σ₁² + 1/σ₂²` — always greater than either individual precision. The fused estimate always lies between `y₁` and `y₂`.

### 2. Covariance Intersection Derivation

When cross-covariances between estimates are unknown but bounded, we need a conservative (safe) fusion.

CI fused covariance: `P_f⁻¹ = ω P₁⁻¹ + (1−ω) P₂⁻¹`

**Why conservative?** For any true cross-covariance `P₁₂`:
```
[P₁  P₁₂]     must be PSD
[P₁₂ᵀ P₂]
```

The CI estimate satisfies `P_f ≥ P_true` for all valid `P₁₂` (where ≥ means "more uncertain"). This prevents false confidence.

**Optimal ω:** minimise `tr(P_f)` or `det(P_f)` over `ω ∈ [0,1]`.

### 3. Kalman Filter = Optimal Gaussian Fusion

The Kalman update is exactly the two-sensor fusion above with:
- Sensor 1: the predicted state `z̃ₜ ~ N(z̃ₜ, P̃ₜ)`
- Sensor 2: the measurement `yₜ ~ N(Czₜ, R)` (after transforming through C)

Information form update:
```
Λₙ = Λₙ₋₁ + CᵀR⁻¹C       (precision matrix update)
ηₙ = ηₙ₋₁ + CᵀR⁻¹yₜ      (information vector update)
μₙ = Λₙ⁻¹ ηₙ               (mean estimate)
```

This is equivalent to the standard Kalman gain form via the Woodbury matrix identity.

### 4. PDAF Measurement-Origin Uncertainty

When the measurement source is uncertain (track + clutter), the PDAF weights:
```
βᵢ = P(mᵢ is from target | measurements)    (association probability)
β₀ = P(no measurement from target)
```

Combined measurement: `ȳ = Σᵢ βᵢ mᵢ`  
**Spread-of-innovations covariance** (extra term for association uncertainty):
```
Pₜ = β₀ P̃ₜ + (1−β₀)P̃KF + P̃_spread
P̃_spread = KₜΣᵢβᵢ(νᵢνᵢᵀ) − ȳȳᵀ)Kₜᵀ
```

This term inflates the posterior covariance when association is uncertain — prevents overconfidence in noisy environments.
