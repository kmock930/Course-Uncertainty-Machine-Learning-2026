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
