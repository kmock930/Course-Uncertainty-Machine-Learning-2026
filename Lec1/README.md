# Lecture 1 – Probabilistic Reasoning and Bayesian Modeling Foundations

**Course:** ELG 5218 / CSI 5218 / EACJ 5600 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
**Date:** January 14, 2026  
**Instructor:** Miodrag Bolic, University of Ottawa

---

## Overview

This lecture introduces the foundational concepts of **Uncertainty Quantification (UQ)** and
establishes the **Bayesian modeling framework** that underpins the rest of the course.
It distinguishes between the two fundamental sources of uncertainty (aleatoric and epistemic),
contrasts frequentist and Bayesian paradigms, and walks through conjugate inference
with the Beta–Binomial example.

---

## Learning Goals

By the end of this lecture you should be able to:

- Define uncertainty quantification and explain why it matters in engineering and ML.
- Distinguish **aleatoric** (irreducible) uncertainty from **epistemic** (reducible) uncertainty.
- Specify a Bayesian model using **prior**, **likelihood**, and **posterior**.
- Explain and compute **Maximum Likelihood Estimation (MLE)** and **Maximum A Posteriori (MAP)** estimation.
- Derive and interpret a simple **conjugate update** (Beta–Binomial).
- Compute and interpret **credible intervals** (central and HPD/HDI).
- Form the **posterior predictive distribution (PPD)** and use it for prediction.

---

## Topics Covered

### Intro Slides (`Lec_intro.pdf`)
- Course overview: scope, objectives, and philosophy.
- Taxonomy of UQ methods (Bayesian approach, deep learning with uncertainty, engineering motivation).
- Types of uncertainty:
  - *Aleatoric*: sensor noise, label ambiguity, occlusion.
  - *Epistemic*: model uncertainty, limited data, distributional shift.
- Major UQ method categories (calibration, conformal prediction, Bayesian inference, generative models, etc.).

### Main Lecture (`Lec1.pdf`)
- **Roadmap:** Introduction → CI vs. CrI → Bayesian framework → Conjugate priors → MLE/MAP → Posterior summaries → PPD.
- Representing uncertainty: standard error, confidence intervals, credible intervals, probability distributions.
- Bayesian inference framework: prior, likelihood, posterior, evidence.
- **Beta–Binomial conjugate example**: closed-form posterior update; MAP vs. MLE comparison.
- Posterior summaries: mean, median, mode; central and HPD credible intervals.
- **Posterior Predictive Distribution (PPD)**: marginalizing over parameters to generate predictions.

### Notebook (`Lec1_Bayesian_Modeling.ipynb`)
- Hands-on implementation of Beta–Binomial Bayesian inference.
- Visualization of prior, likelihood, and posterior distributions.
- Computing credible intervals and the posterior predictive distribution.
- Introduction to NumPyro for probabilistic programming.

---

## Materials

| File | Description |
|------|-------------|
| `Lec_intro.pdf` | Introductory slides: course overview and UQ taxonomy |
| `Lec1.pdf` | Main lecture slides: Bayesian modeling foundations (42 slides) |
| `Lec1_Bayesian_Modeling.ipynb` | Jupyter notebook with implementations and exercises |
| `Images/` | Supporting figures for the lecture slides |

---

## Recommended Reading

- **Villani, Mattias (2025). *Bayesian Learning*.** – Chapters 1 and 2.
  [PDF](https://github.com/mattiasvillani/BayesianLearningBook/raw/main/pdf/BayesBook.pdf)
- **Prince, Simon J.D. (2025). *Understanding Deep Learning*.** – Probability refresher ("Math for Machine Learning").
  [Book website](https://udlbook.github.io/udlbook/)

---

## Practice Questions

Relevant practice sets from the `Practice Questions/` folder:

- `Problems_Bayesian_Questions.pdf` / `Problems_Bayesian_Solutions.pdf` – Conceptual, derivation, and implementation questions on Bayesian modeling (Beta–Binomial, MLE vs. MAP, credible intervals, PPD).

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| Aleatoric uncertainty | Irreducible randomness in data/measurements |
| Epistemic uncertainty | Reducible uncertainty due to limited knowledge or data |
| Prior `p(θ)` | Belief about parameters before observing data |
| Likelihood `p(x\|θ)` | Probability of data given parameters |
| Posterior `p(θ\|x)` | Updated belief after observing data |
| Conjugate prior | Prior that yields a posterior in the same family as the prior |
| MLE | Estimate that maximizes the likelihood |
| MAP | Estimate that maximizes the posterior (= MLE + prior regularization) |
| Credible interval | Bayesian analog of confidence interval: interval containing θ with specified posterior probability |
| PPD | Posterior predictive distribution: prediction integrated over parameter uncertainty |
