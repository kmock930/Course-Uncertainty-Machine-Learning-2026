# Lecture 5 – Variational Inference

> **Revision summary** | ELG 5218 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
> Slides: `VariationalInference.pdf` · Notebook: `From_entropy_to_SVI.ipynb`  
> Reading: Villani (VI chapters); [SVI with NumPyro](https://phuijse.github.io/BLNNbook/chapters/variational/svi.html)

---

## Core Ideas

### The Central Idea
- Posterior `p(θ|x)` is intractable → find the closest tractable approximation `q(θ; φ)`.
- Minimise `KL(q || p)` ↔ maximise the **ELBO**:

  `ELBO(φ) = E_q[log p(x,θ)] − E_q[log q(θ)] = log p(x) − KL(q||p)`

- Because `KL ≥ 0`, ELBO ≤ log p(x) always. Higher ELBO = closer approximation.

### Mean-Field Approximation
- Assume fully factored: `q(θ) = ∏ᵢ qᵢ(θᵢ)` — each parameter independent.
- **CAVI update** for factor `qⱼ`: `log qⱼ*(θⱼ) = E_{q_{-j}}[log p(x, θ)] + const`
- Iterate over factors until ELBO converges.

### Stochastic (Black-box) VI
- Replace exact ELBO expectations with Monte Carlo estimates over mini-batches.
- **Score-function gradient (REINFORCE):** `∇_φ ELBO = E_q[(log p(x,θ) − log q(θ)) ∇_φ log q(θ)]`
  - Unbiased but **high variance** — needs many samples or control variates.

### Reparameterization Trick
- Write `θ = g(ε, φ)` with `ε ~ p(ε)` (noise-free base distribution, e.g. `N(0,1)`).
- Example: `θ ~ N(μ, σ²)` → `θ = μ + σε`, `ε ~ N(0,1)`
- Gradient flows through `g` → **low-variance** pathwise gradient of ELBO.
- This is the key operation inside VAEs (Lec 6).

---

## MCMC vs VI — Quick Comparison

| | MCMC | VI |
|--|--|--|
| Approach | Sampling | Optimisation |
| Asymptotic | Exact (given enough samples) | Approximate (biased by q family) |
| Scalability | Slow for large datasets | Scales with SGD |
| Uncertainty | Full posterior | Restricted to q family |

---

## Things to Remember

- Maximising ELBO ≡ minimising `KL(q||p)` (they differ by the constant `log p(x)`).
- Mean-field VI underestimates posterior variance (independence assumption squeezes correlations).
- Reparameterization only works for **continuous** latent variables; use score-function for discrete.

---

## Practice Questions

- [`../Practice Questions/Problems_VI_Questions.pdf`](../Practice%20Questions/Problems_VI_Questions.pdf) — ELBO derivation, CAVI updates, reparameterization trick
- [`../Practice Questions/SVI_Problems.pdf`](../Practice%20Questions/SVI_Problems.pdf) — Stochastic VI implementation and analysis

---

## Key Derivations

### 1. Jensen's Inequality → ELBO

**Jensen's Inequality:** For a concave function f and random variable X: `f(E[X]) ≥ E[f(X)]`  
(`log` is concave, so `log E[X] ≥ E[log X]`)

Apply to the log marginal likelihood:
```
log p(x) = log ∫ p(x, θ) dθ
          = log ∫ p(x, θ) · q(θ)/q(θ) dθ
          = log E_{q(θ)}[p(x,θ)/q(θ)]
          ≥ E_{q(θ)}[log(p(x,θ)/q(θ))]    ← Jensen's inequality
          = E_q[log p(x,θ)] − E_q[log q(θ)]
          = ELBO(q)
```

Therefore: `log p(x) ≥ ELBO(q)`, with equality iff `q(θ) = p(θ|x)`.

### 2. ELBO = log p(x) − KL(q||p)

```
ELBO(q) = E_q[log p(x,θ)] − E_q[log q(θ)]
        = E_q[log p(x|θ)] + E_q[log p(θ)] − E_q[log q(θ)]
        = E_q[log p(x|θ)] − KL(q(θ) || p(θ))         ← reconstruction − prior KL
```

Alternatively:
```
ELBO(q) = E_q[log(p(x,θ)/q(θ))]
        = E_q[log(p(θ|x)p(x)/q(θ))]
        = log p(x) + E_q[log(p(θ|x)/q(θ))]
        = log p(x) − KL(q(θ) || p(θ|x))
```

So **maximising ELBO ≡ minimising KL(q||p)** (since log p(x) is a constant w.r.t. q).

### 3. CAVI Update Derivation

For mean-field `q(θ) = ∏_j q_j(θ_j)`, fix all factors except `q_j` and maximise ELBO:

```
ELBO = E_q[log p(x,θ)] − Σ_j E_{q_j}[log q_j(θ_j)] + const
```

Taking functional derivative w.r.t. `q_j` and setting to zero:
```
log q_j*(θ_j) = E_{q_{-j}}[log p(x, θ)] + const
```

This means `q_j*` is proportional to the exponentiated expected log joint, averaged over all other factors — it is the "conditional" under the mean-field approximation.

### 4. Reparameterization Gradient

**Score-function estimator** (high variance):
```
∇_φ E_{q_φ}[f(θ)] = E_{q_φ}[f(θ) ∇_φ log q_φ(θ)]
```

**Reparameterization** (low variance): write `θ = g(ε, φ)` with `ε ~ p(ε)` independent of φ:
```
E_{q_φ}[f(θ)] = E_{p(ε)}[f(g(ε, φ))]
∇_φ E_{q_φ}[f(θ)] = E_{p(ε)}[∇_φ f(g(ε, φ))]    ← move gradient inside
                   ≈ (1/S) Σ_s ∇_φ f(g(εˢ, φ))  ← Monte Carlo
```

Variance is lower because the gradient is computed via the deterministic path `g`, not by multiplying by a noisy score.
