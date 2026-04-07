# Lecture 1 – Probabilistic Reasoning and Bayesian Modeling Foundations

> **Revision summary** | ELG 5218 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
> Slides: `Lec_intro.pdf`, `Lec1.pdf` · Notebook: `Lec1_Bayesian_Modeling.ipynb`  
> Reading: Villani Ch. 1–2

---

## Core Ideas

- **Two types of uncertainty:**
  - *Aleatoric* – irreducible; inherent noise in data (e.g. sensor errors, label ambiguity). Cannot be reduced with more data.
  - *Epistemic* – reducible; due to limited data or model misspecification. Shrinks as data grows.

- **Bayesian rule:**
  $$p(\theta \mid x) = \frac{p(x \mid \theta)\, p(\theta)}{p(x)}$$
  Prior × Likelihood ∝ Posterior. Evidence `p(x)` is a normalising constant (hard to compute).

- **MLE vs MAP:**
  - MLE: `θ̂ = argmax p(x|θ)` — ignores prior; equivalent to MAP with a uniform prior.
  - MAP: `θ̂ = argmax p(θ|x)` — adds prior regularisation. Ridge regression = MAP with Gaussian prior.

- **Beta–Binomial conjugate update** (key example):
  - Prior: `θ ~ Beta(α, β)`, Likelihood: `k ~ Binomial(N, θ)`
  - Posterior: `θ | k ~ Beta(α + k, β + N − k)` — same family, parameters simply add.
  - MAP estimate: `(α + k − 1) / (α + β + N − 2)`; MLE: `k/N`

- **Posterior summaries:**
  - Point: mean, median (L1-optimal), mode (MAP, L0-optimal)
  - Interval: central CrI (equal-tailed), HPD (highest posterior density — shortest interval containing probability mass)

- **Posterior Predictive Distribution (PPD):**
  $$p(\tilde{x} \mid x) = \int p(\tilde{x} \mid \theta)\, p(\theta \mid x)\, d\theta$$
  Averages predictions over all plausible parameters — accounts for parameter uncertainty.

---

## Confidence Interval vs Credible Interval

| | Frequentist CI | Bayesian CrI |
|--|--|--|
| Interpretation | 95% of such intervals cover the true θ | θ lies in this interval with 95% posterior probability |
| Requires prior? | No | Yes |
| Answers "where is θ?" | Not directly | Yes |

---

## Practice Questions

[`../Practice Questions/Problems_Bayesian_Questions.pdf`](../Practice%20Questions/Problems_Bayesian_Questions.pdf) — Beta-Binomial update, MLE vs MAP, credible intervals, PPD computation

---

## Key Derivations

### 1. Bayes' Rule from the Product Rule
Starting from the joint distribution:

```
p(θ, x) = p(θ|x) p(x) = p(x|θ) p(θ)
```

Rearranging: `p(θ|x) = p(x|θ) p(θ) / p(x)`

The evidence `p(x) = ∫ p(x|θ) p(θ) dθ` is a normalising constant, so:

`p(θ|x) ∝ p(x|θ) p(θ)`

### 2. Beta–Binomial Conjugate Update (completing the algebra)

Prior: `p(θ) = Beta(α, β) ∝ θ^{α−1} (1−θ)^{β−1}`  
Likelihood for k successes in N trials: `p(k|θ) ∝ θ^k (1−θ)^{N−k}`

Posterior:
```
p(θ|k) ∝ p(k|θ) p(θ)
        ∝ θ^k (1−θ)^{N−k} · θ^{α−1} (1−θ)^{β−1}
        = θ^{(α+k)−1} (1−θ)^{(β+N−k)−1}
        = Beta(α+k, β+N−k)
```

This is the same Beta family with updated parameters — the definition of conjugacy.

**MAP** = mode of `Beta(α+k, β+N−k)` = `(α+k−1) / (α+β+N−2)` (set derivative to zero)  
**MLE** = `k/N` (corresponds to α=β=1, i.e. uniform prior)

### 3. Posterior Predictive Distribution (PPD)

For a new trial `x̃ ∈ {0,1}`:
```
p(x̃=1 | k) = ∫₀¹ p(x̃=1|θ) p(θ|k) dθ
             = ∫₀¹ θ · Beta(θ; α+k, β+N−k) dθ
             = E[θ | k]        (mean of Beta posterior)
             = (α+k) / (α+β+N)
```

Intuition: the PPD mean equals the posterior mean — parameter uncertainty is "averaged out".

### 4. MAP as Regularised MLE

Log-posterior = log-likelihood + log-prior:
```
log p(θ|x) = log p(x|θ) + log p(θ) + const
```

With a Gaussian prior `p(θ) = N(0, τ²)`:
```
log p(θ|x) = log p(x|θ) − θ²/(2τ²) + const
```

Maximising this is MLE with L2 penalty `λ = 1/τ²` → Ridge regression / weight decay.
