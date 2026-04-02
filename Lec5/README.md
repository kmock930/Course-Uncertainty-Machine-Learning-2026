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
