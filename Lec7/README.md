# Lecture 7 – Calibration and Conformal Prediction

> **Revision summary** | ELG 5218 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
> Slides: `Proper_Scoring_Rules_and_Calibration.pdf`, `Conformal_Prediction.pdf` · Notebooks: `calibration_lab_notebook.ipynb`, `conformal_prediction_lab_notebook.ipynb`  
> Reading: Murphy PML Advanced Ch. 14.2–14.3; [Calibration Tutorial](https://github.com/luferrer/CalibrationTutorial); [Conformal Prediction](https://github.com/aangelopoulos/conformal-prediction)

---

## Core Ideas

### Proper Scoring Rules
A scoring rule `S(p̂, y)` is **proper** if the forecaster minimises expected score by reporting true probabilities.

| Score | Formula | Notes |
|---|---|---|
| NLL (log loss) | `−log p̂(y)` | Strictly proper; penalises overconfident errors heavily |
| Brier Score | `(p̂ − y)²` | Strictly proper; MSE in probability space |
| CRPS | `∫(F̂(z) − 1{y≤z})² dz` | Generalises Brier to continuous distributions |

### Calibration
- **Definition:** a model is calibrated if `P(Y=1 | p̂=p) = p` for all p.
- **Reliability diagram:** plot observed fraction vs. predicted probability per bin. Perfect calibration = diagonal.
- **ECE (Expected Calibration Error):** `Σ_b (|Bᵦ|/n) |acc(Bᵦ) − conf(Bᵦ)|`
- **Finding:** modern deep nets are typically **overconfident** (Guo et al., 2017).

**Post-hoc calibration methods:**
- **Temperature scaling:** divide logits by `T > 1`; single parameter; very effective.
- **Platt scaling:** logistic regression on model outputs.
- **Isotonic regression:** non-parametric; more flexible but needs more calibration data.

### Conformal Prediction
Distribution-free uncertainty sets with **guaranteed marginal coverage**.

**Split conformal algorithm:**
1. Split data: training set + calibration set (size n).
2. Define nonconformity score `s(x, y)` (e.g. `1 − softmax[true class]`).
3. Compute scores `s₁, …, sₙ` on calibration set; set threshold `q̂ = ⌈(n+1)(1−α)⌉/n` quantile.
4. Test prediction set: `C(xₜₑₛₜ) = {y : s(xₜₑₛₜ, y) ≤ q̂}`

**Coverage guarantee:** `P(Y ∈ C(X)) ≥ 1 − α` — holds for **any** model, **any** distribution (only requires exchangeability).

**Efficiency** = average set size; smaller sets are more informative. Adaptive scores improve efficiency.

---

## Things to Remember

- Calibration ≠ accuracy: a model can be accurate but poorly calibrated (overconfident).
- Temperature scaling is the go-to post-hoc recalibration method — simple and effective.
- Conformal prediction gives **valid coverage by construction**, regardless of model quality.
- Conformal prediction does not require retraining — it wraps any existing predictor.
- Larger calibration set → tighter, more useful prediction sets.

---

## Practice Questions

[`../Practice Questions/Problems_Calibration_Conformal_Questions.pdf`](../Practice%20Questions/Problems_Calibration_Conformal_Questions.pdf) — ECE computation, temperature scaling, conformal set construction, coverage guarantee derivation
