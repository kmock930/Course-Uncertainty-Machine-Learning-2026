# Lecture 7 – Calibration and Conformal Prediction

**Course:** ELG 5218 / CSI 5218 / EACJ 5600 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
**Date:** March 4, 2026  
**Instructor:** Miodrag Bolic, University of Ottawa

---

## Overview

This lecture addresses a practical question that arises once a probabilistic model is trained:
*are the predicted probabilities actually reliable?* The first half covers **Proper Scoring
Rules** and **Calibration** — tools for evaluating and correcting the reliability of
probabilistic predictions. The second half introduces **Conformal Prediction**, a
distribution-free framework that provides rigorous finite-sample coverage guarantees for
any trained predictor without retraining.

---

## Learning Goals

By the end of this lecture you should be able to:

- Define **calibration** and explain why modern deep networks are typically miscalibrated.
- Compute and interpret **proper scoring rules**: Negative Log-Likelihood (NLL), Brier Score, and CRPS.
- Construct and read a **reliability diagram** and compute the **Expected Calibration Error (ECE)**.
- Apply post-hoc calibration methods: temperature scaling, Platt scaling, histogram binning, isotonic regression.
- Explain the **split conformal prediction** algorithm and its finite-sample coverage guarantee.
- Construct conformal prediction sets for classification and intervals for regression.
- Apply conformal prediction to object detection and language model abstention.

---

## Topics Covered

### Proper Scoring Rules and Calibration (`Proper_Scoring_Rules_and_Calibration.pdf`)

**Why calibration matters**
- Many decisions require risk-aware predictions (medicine, autonomy, finance).
- Accuracy alone is insufficient; two models with the same accuracy can differ dramatically in reliability.

**Proper Scoring Rules**
- A scoring rule S(p, y) is **proper** if the optimal strategy is to report true beliefs.
- **NLL** (log loss): penalizes overconfident wrong predictions heavily.
- **Brier Score**: mean squared error between predicted probabilities and one-hot labels.
- **CRPS** (Continuous Ranked Probability Score): generalizes Brier score to continuous distributions.

**Calibration Diagnostics**
- **Reliability diagram**: plot of mean predicted probability vs. fraction of positives per bin.
- **Expected Calibration Error (ECE)**: weighted average of calibration gaps across bins.
- Empirical finding: modern deep networks are overconfident (Guo et al., 2017).

**Post-hoc Calibration Methods**
- **Temperature scaling**: divide logits by a learned scalar T.
- **Platt scaling**: logistic regression on model outputs.
- **Histogram binning** and **isotonic regression**: non-parametric recalibration.
- Applications: image classification, object detection, LLMs.

### Conformal Prediction (`Conformal_Prediction.pdf`)

**Motivation**
- We want valid uncertainty even when the model is wrong or misspecified.
- Goal: prediction set/interval that contains the true label/value with guaranteed probability ≥ 1 − α.

**Split Conformal Prediction Algorithm**
1. Split data: training set + calibration set.
2. Define a **nonconformity score** s(x, y) (e.g., 1 − softmax score for true class).
3. Compute scores on the calibration set; take the (1−α)-quantile as threshold q̂.
4. At test time: include all labels y for which s(x, y) ≤ q̂.

**Theoretical Guarantee**
- **Marginal coverage**: P(Y ∈ C(X)) ≥ 1 − α for any model, any distribution (exchangeability only).
- No distributional assumptions; no retraining required.

**Efficiency**
- Set size / interval length measures efficiency; shorter sets = more informative predictions.
- Adaptive conformal scores improve efficiency while maintaining coverage.

**Applications**
- Image classification prediction sets.
- Object detection: bounding-box regions with coverage guarantees.
- LLM question answering: abstention when uncertainty is too high.

---

## Materials

| File | Description |
|------|-------------|
| `Proper_Scoring_Rules_and_Calibration.pdf` | Lecture slides: Scoring rules and calibration (34 slides) |
| `Conformal_Prediction.pdf` | Lecture slides: Conformal prediction (23 slides) |
| `calibration_lab_notebook.ipynb` | Notebook: Calibration diagnostics and post-hoc methods |
| `conformal_prediction_lab_notebook.ipynb` | Notebook: Split conformal prediction implementation |

---

## Recommended Reading

- **Murphy, Kevin P. *Probabilistic Machine Learning: Advanced Topics*.** – Calibration and proper scoring rules (Sections 14.2 and 14.3).
- **Calibration Tutorial** by Ferrer et al.: [GitHub](https://github.com/luferrer/CalibrationTutorial)
- **Conformal Prediction** by Angelopoulos et al.: [GitHub](https://github.com/aangelopoulos/conformal-prediction)

---

## Practice Questions

Relevant practice sets from the `Practice Questions/` folder:

- `Problems_Calibration_Conformal_Questions.pdf` / `Problems_Calibration_Conformal_Solutions.pdf` – Reliability diagrams, ECE computation, temperature scaling, conformal prediction set construction, coverage guarantee proofs.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| Calibration | A model is calibrated if predicted probability p̂ matches empirical frequency p(y=1\|p̂) |
| Reliability diagram | Visual calibration check: predicted probability vs. observed frequency |
| ECE | Expected Calibration Error: mean absolute calibration gap across bins |
| Proper scoring rule | Score that incentivizes honest probability reporting |
| NLL | Negative log-likelihood; strictly proper scoring rule |
| Brier score | Mean squared probability error; strictly proper scoring rule |
| Temperature scaling | Divide logits by T > 1 to soften overconfident predictions |
| Conformal prediction | Distribution-free uncertainty sets with finite-sample coverage guarantee |
| Nonconformity score | Measure of how unusual a (input, label) pair is; used to size conformal sets |
| Marginal coverage | P(Y ∈ C(X)) ≥ 1 − α guaranteed without distributional assumptions |
| Coverage–efficiency trade-off | Larger sets always achieve coverage; smaller sets are more useful |
