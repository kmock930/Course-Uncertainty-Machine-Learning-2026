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

---

## Key Derivations

### 1. Proper Scoring Rule — Log Loss (NLL) is Strictly Proper

A scoring rule `S(p̂, y)` is **proper** if `E_p[S(p, y)] ≤ E_p[S(q, y)]` for any `q ≠ p` — i.e. the forecaster is incentivised to report the true distribution.

For NLL: `S(p̂, y) = −log p̂(y)`.

Expected score when reporting `q` but truth is `p`:
```
E_p[−log q(Y)] = −Σ_y p(y) log q(y) = H(p, q) = H(p) + KL(p||q)
```

Since `KL(p||q) ≥ 0` and = 0 only when `p = q`, the minimum is achieved by reporting the true distribution → NLL is strictly proper.

### 2. ECE (Expected Calibration Error) Derivation

Partition predictions into M equal-width bins `B₁, …, B_M` on [0,1].

Per-bin calibration error = |average confidence − average accuracy|:
```
ECE = Σ_m (|B_m|/n) · |conf(B_m) − acc(B_m)|
```

where:
- `conf(B_m) = (1/|B_m|) Σ_{i∈B_m} p̂ᵢ`
- `acc(B_m) = (1/|B_m|) Σ_{i∈B_m} 1[ŷᵢ = yᵢ]`

ECE = 0 means perfect calibration. ECE > 0 with `conf > acc` means overconfidence.

### 3. Temperature Scaling

After training, apply a single scalar T > 0 to the logits `f(x)`:
```
p̂_T(y=k|x) = exp(f_k(x)/T) / Σ_j exp(f_j(x)/T)
```

- T > 1: softens the distribution → reduces overconfidence
- T < 1: sharpens the distribution → increases confidence
- T = 1: no change (original model)

Optimal T is found by minimising NLL on a held-out calibration set (1D optimisation).

**Why it works:** NLL is a proper scoring rule — minimising it on calibration data forces the model to report better-calibrated probabilities.

### 4. Conformal Prediction Coverage Guarantee

**Claim:** Split conformal guarantees `P(Y ∈ C(X)) ≥ 1 − α`.

**Setup:** calibration scores `s₁, …, s_n` are i.i.d. exchangeable with test score `s_{n+1}`. The threshold:
```
q̂ = ⌈(n+1)(1−α)⌉-th smallest value of {s₁, …, s_n}
```

**Proof sketch:**
```
P(s_{n+1} ≤ q̂) = P(rank(s_{n+1}) ≤ ⌈(n+1)(1−α)⌉ in {s₁,…,s_{n+1}})
```

By exchangeability, all `n+1` orderings are equally likely. The probability that `s_{n+1}` is at or below the `⌈(n+1)(1−α)⌉`-th position is at least `1 − α`.

Since `C(x) = {y : s(x,y) ≤ q̂}`, this gives `P(Y ∈ C(X)) ≥ 1 − α`. ✓

The guarantee holds **for any model and any distribution** — only exchangeability is required.
