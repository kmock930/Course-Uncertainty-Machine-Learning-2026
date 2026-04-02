# Lecture 4 – State Space Models, Kalman Filter, and Particle Filters

**Course:** ELG 5218 / CSI 5218 / EACJ 5600 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
**Date:** February 4, 2026  
**Instructor:** Miodrag Bolic, University of Ottawa

---

## Overview

This lecture moves from static Bayesian inference to **sequential (temporal) probabilistic models**.
It introduces **State Space Models (SSMs)** as a principled way to represent hidden dynamics
with noisy observations, derives the **Kalman Filter** as the exact Bayesian filtering solution
for linear Gaussian SSMs, and then extends to **Particle Filters** (Sequential Monte Carlo)
for nonlinear and non-Gaussian settings.

---

## Learning Goals

By the end of this lecture you should be able to:

- Explain the hidden-state problem and why State Space Models are useful.
- Write down the transition and measurement models for a linear SSM.
- Discretize a continuous-time differential equation to obtain a discrete SSM.
- Derive and implement the **Kalman Filter** predict–update recursion.
- Interpret the **Kalman gain** and explain how it balances model predictions and measurements.
- Describe extensions: Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), RTS smoother.
- Explain why particle filters are needed for nonlinear/non-Gaussian problems.
- Implement the **Sequential Importance Resampling (SIR)** particle filter.
- Apply both filters to real-world tracking and state estimation problems.

---

## Topics Covered

### State Space Models (`ssm_slides.pdf`)
- **Motivation**: systems with hidden structure; noisy observations; domain knowledge about dynamics.
- SSM definition: latent state `z_t`, transition model `p(z_t | z_{t-1})`, measurement model `p(y_t | z_t)`.
- Examples: aircraft tracking (radar → position + velocity), physiological signals, econometric models.
- From continuous-time ODEs to discrete-time SSMs: Euler discretization, exact matrix exponential.
- Latent variable and probabilistic graphical model view.

### Kalman Filter (`Kalman_filter_slides.pdf`)
- **The filtering problem**: estimate current state from all past observations.
- Bayesian filtering framework: predict step (Chapman–Kolmogorov) + update step (Bayes).
- **Kalman Filter derivation** for linear Gaussian SSMs:
  - Predict: propagate mean and covariance through the transition model.
  - Update: condition on new measurement using the joint Gaussian.
- **Kalman gain**: weight matrix balancing prediction uncertainty vs. measurement noise.
- Properties: optimal (minimum MSE) for linear Gaussian systems; equivalent to Wiener filter.
- Extensions: Extended KF (linearization via Jacobians), Unscented KF (sigma points), RTS smoother (backward pass).
- Connection to modern ML: Kalman filter as a precursor to state-space deep learning models.

### Particle Filters (`Particle_filter_slides.pdf`)
- **Why particle filters**: nonlinear dynamics and non-Gaussian noise break the Kalman filter.
- Sequential Monte Carlo (SMC) / particle filtering paradigm.
- **Importance sampling**: representing a distribution with weighted particles.
- **Sequential Importance Resampling (SIR)**:
  1. Propagate particles through transition model.
  2. Weight by likelihood of new measurement.
  3. Resample to avoid weight degeneracy.
- **Bearings-only tracking** example: estimating position from angle-only radar measurements.
- Advanced topics: auxiliary particle filter, Rao-Blackwellized particle filter, degeneracy diagnostics.
- Computational complexity vs. accuracy trade-offs.

---

## Materials

| File | Description |
|------|-------------|
| `ssm_slides.pdf` | Lecture slides: State Space Models (24 slides) |
| `Kalman_filter_slides.pdf` | Lecture slides: Kalman Filter derivation and examples (36 slides) |
| `Particle_filter_slides.pdf` | Lecture slides: Particle Filters / Sequential Monte Carlo (42 slides) |
| `ssm_examples.ipynb` | Notebook: Kalman filter implementations and SSM examples |
| `bearings_only_particle_filter.ipynb` | Notebook: Particle filter for bearings-only tracking |

---

## Recommended Reading

- **Villani, Mattias (2025). *Bayesian Learning*.** – Chapter 17.
  [PDF](https://github.com/mattiasvillani/BayesianLearningBook/raw/main/pdf/BayesBook.pdf)
- **Sarkka, Simo. *Bayesian Filtering and Smoothing*.** – Chapters 4 and 7.
  [PDF](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf)
- **Labbe, Roger. *Kalman and Bayesian Filters in Python*.**
  [Online book](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)

---

## Practice Questions

Relevant practice sets from the `Practice Questions/` folder:

- `Problems_Kalman_Filter_Questions.pdf` / `Problems_Kalman_Filter_Solutions.pdf` – Kalman predict–update derivations, Kalman gain interpretation, 1D and 2D filter examples.
- `Problems_Particle_Filter_Questions.pdf` / `Problems_Particle_Filter_Solutions.pdf` – Importance sampling, SIR algorithm steps, weight degeneracy and resampling.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| State Space Model (SSM) | Probabilistic model of hidden state evolving over time with noisy observations |
| Transition model `p(z_t\|z_{t-1})` | Describes how the hidden state evolves |
| Measurement model `p(y_t\|z_t)` | Describes how the state generates observations |
| Kalman gain K | Optimal weighting matrix; K → I when measurement is very accurate |
| Predict step | Propagate belief forward in time using the dynamics model |
| Update step | Condition the predicted belief on the new measurement |
| EKF / UKF | Nonlinear extensions of the Kalman filter |
| RTS smoother | Backward pass to refine all state estimates given full data |
| Particle filter | Monte Carlo approximation to the filtering distribution; handles nonlinear/non-Gaussian SSMs |
| Resampling | Duplicate high-weight particles and discard low-weight ones to maintain filter quality |
