# Lecture 8 – Sensor Fusion and Data Association

**Course:** ELG 5218 / CSI 5218 / EACJ 5600 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
**Date:** March 11, 2026  
**Instructor:** Miodrag Bolic, University of Ottawa

---

## Overview

This lecture addresses the challenge of combining information from **multiple sensors**
in a principled, uncertainty-aware way. Starting from the optimal Gaussian fusion rule
derived via maximum likelihood, the lecture builds up to Bayesian fusion (the Kalman
filter as fusion), covariance intersection for correlated noise, and a range of
**data association** algorithms for multi-target scenarios where it is unclear which
measurement belongs to which target.

---

## Learning Goals

By the end of this lecture you should be able to:

- Derive the **precision-weighted (MLE) fusion** rule for combining multiple Gaussian sensors.
- Interpret sensor fusion from an **information / entropy perspective**.
- Apply fusion for N sensors and multivariate measurements.
- Explain why ignoring **correlated noise** leads to overconfident fusion estimates.
- Apply **Covariance Intersection (CI)** when cross-covariances are unknown.
- Recognize the Kalman filter as an instance of Bayesian sequential fusion.
- Describe the data association problem and apply **Nearest Neighbor (NN) gating**.
- Explain **PDAF** (Probabilistic Data Association Filter), **JPDA** (Joint PDA), and **MHT** (Multiple Hypothesis Tracker).
- Outline sensor registration, track-to-track fusion, and the PHD filter.

---

## Topics Covered

### Block 1 – Optimal Gaussian Multi-Sensor Fusion
- **MLE derivation**: maximize log-likelihood over multiple independent Gaussian sensors.
- Precision-weighted fusion formula: fused mean and variance.
- **Entropy and information view**: fusion reduces entropy; information additivity.
- Multivariate fusion: N sensors with covariance matrices.
- **Correlated sensor noise**: naive fusion is no longer optimal; cross-covariance must be tracked.

### Block 2 – Bayesian Fusion
- **Kalman Filter as Bayesian fusion**: sequential measurement incorporation.
- Sequential vs. stacked (batch) updates: equivalence for independent measurements.
- **Information form of the Kalman filter**: precision (information) matrices; additivity of information.
- **Covariance Intersection (CI)**: conservative fusion when cross-covariances are unavailable.
- When CI is appropriate vs. when full cross-covariance tracking is needed.

### Block 3 – Data Association
- The data association problem: multiple targets, multiple measurements, unknown origin.
- **Gating**: discard clearly inconsistent measurement–track pairs using chi-squared test.
- **Global Nearest Neighbour (GNN)**: assign each measurement to the nearest track.
- **PDAF** (Probabilistic Data Association Filter): soft association; weights over all gated measurements; covariance spread term accounts for uncertainty.
- **JPDA** (Joint PDA): computes joint association probabilities over all targets simultaneously.
- **MHT** (Multiple Hypothesis Tracker): deferred decisions; maintains a tree of association hypotheses.

### Block 4 – Registration and System Integration
- **Sensor registration**: spatial alignment (coordinate transforms) and temporal alignment (synchronization).
- Track-to-track fusion: fusing tracks from distributed trackers.
- **PHD filter**: probability hypothesis density; represents multi-target distributions without explicit association.
- Complete multi-sensor pipeline: detection → registration → gating → association → fusion → tracking.

---

## Materials

| File | Description |
|------|-------------|
| `Sensor_Fusion.pdf` | Lecture slides: Sensor fusion and association end-to-end (78 slides) |
| `notebook_traditional_fusion.ipynb` | Notebook: Traditional sensor fusion and data association |
| `ci_vs_kf.ipynb` | Notebook: Covariance Intersection vs. Kalman Filter comparison |

---

## Recommended Reading

- **Li, X. R., & Jilkov, V. P. (2001). *Survey of maneuvering target tracking: III. Measurement models.*** *Signal and Data Processing of Small Targets, SPIE.*

---

## Practice Questions

Relevant practice sets from the `Practice Questions/` folder:

- `Problems_Fusion_Questions.pdf` / `Problems_Fusion_Solution.pdf` – Precision-weighted fusion derivations, multivariate fusion, CI algorithm, data association scenarios (PDAF, GNN).

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| Precision-weighted fusion | Fused estimate weights each sensor by its precision (inverse variance) |
| Information form | Express Kalman filter in terms of information matrix (Λ = P⁻¹) and information vector |
| Covariance Intersection (CI) | Conservative fusion algorithm valid for any unknown cross-correlation |
| Data association | Problem of assigning measurements to tracks when targets are unknown |
| Gating | Pre-filter: discard measurement–track pairs with too large a Mahalanobis distance |
| GNN | Global Nearest Neighbour: hard one-to-one measurement–track assignment |
| PDAF | Soft association: weighted average over gated measurements; covariance inflated for missed detection |
| JPDA | Joint PDA: accounts for multiple-target associations simultaneously |
| MHT | Multiple Hypothesis Tracker: maintains tree of hypotheses; deferred decision |
| PHD filter | Random finite set approach; no explicit data association; propagates target intensity |
