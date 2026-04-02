# Lecture 6 – Variational Autoencoders and Bayesian Neural Networks

> **Revision summary** | ELG 5218 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
> Slides: `NN_Parameterization.pdf`, `VAE.pdf`, `BNN.pdf` · Notebooks: `NN_Parameterization_Notebook.ipynb`, `VAE_Breathing_Anomaly.ipynb`, `Bnn_notebook.ipynb`  
> Reading: Kingma & Welling (VAE); Huijse (VAE/BNN chapters); Gal & Ghahramani (2016); Blundell et al. (2015)

---

## Core Ideas

### Neural Networks as Distribution Parameterizers
- A network `fθ: x → (μ(x), σ(x))` outputs distribution parameters, not just a point.
- **Heteroscedastic regression:** learn input-dependent noise `σ(x)` — captures aleatoric uncertainty.
- **Mixture Density Network (MDN):** output is a mixture of Gaussians; handles multi-modal outputs.
- Loss: negative log-likelihood of the output distribution (not MSE).

### Variational Autoencoders (VAEs)
- **Generative model:** `z ~ p(z) = N(0,I)`, `x|z ~ p_θ(x|z)` (decoder network)
- **Inference model (encoder):** `q_φ(z|x) = N(μ_φ(x), σ_φ²(x))`
- **ELBO objective** (maximise):

  `ELBO = E_{q_φ}[log p_θ(x|z)] − KL(q_φ(z|x) || p(z))`

  = **Reconstruction term** − **KL regularisation to prior**

- **Reparameterization:** `z = μ_φ(x) + σ_φ(x) · ε`, `ε ~ N(0,I)` — differentiable sampling.
- After training: sample `z ~ N(0,I)`, decode to generate new `x`; or encode `x` to get latent code.
- Application: anomaly detection — low ELBO score signals anomalous input.

### Bayesian Neural Networks (BNNs)
Place a prior over weights `w ~ p(w)` instead of learning a point estimate.

| Method | How | Trade-off |
|---|---|---|
| **MCMC (SGLD, HMC)** | Sample weight posterior | Accurate but slow; scales poorly |
| **Bayes-by-Backprop** | Variational: `q(w) = ∏ N(μᵢ, σᵢ²)` | Scalable; mean-field underestimates uncertainty |
| **MC Dropout** | Apply dropout at test time; average T forward passes | Simple; implicit BNN; may not be well-calibrated |

- **Epistemic uncertainty** ↔ variance across BNN predictions (reducible; decreases with more data).
- **Aleatoric uncertainty** ↔ heteroscedastic output variance (irreducible).

---

## Things to Remember

- KL term in VAE ELBO acts as a regulariser, preventing the encoder from collapsing to a point.
- MC Dropout is the simplest BNN approximation: `Var[ŷ] = (1/T) Σ ŷₜ² − (mean ŷ)²`
- VAE latent space is structured (smooth interpolation); standard AE latent space is not.
- BNNs capture **epistemic** uncertainty; heteroscedastic NNs capture **aleatoric** uncertainty; full BNN with heteroscedastic head captures both.

---

## Practice Questions

- [`../Practice Questions/Problems_VAE_Questions.pdf`](../Practice%20Questions/Problems_VAE_Questions.pdf) — ELBO for VAEs, reparameterization, anomaly detection
- [`../Practice Questions/Problems_NN_Parameterization_Questions.pdf`](../Practice%20Questions/Problems_NN_Parameterization_Questions.pdf) — MDN, heteroscedastic loss, output heads

---

## Key Derivations

### 1. VAE ELBO (Jensen's Inequality Applied)

Marginal log-likelihood:
```
log p_θ(x) = log ∫ p_θ(x|z) p(z) dz
           = log E_{q_φ(z|x)}[p_θ(x,z) / q_φ(z|x)]
           ≥ E_{q_φ}[log p_θ(x|z)] − KL(q_φ(z|x) || p(z))     ← ELBO
```

The two terms have clear roles:
- **Reconstruction:** `E_{q_φ}[log p_θ(x|z)]` — decoder should reconstruct x well
- **KL regularisation:** `KL(q_φ(z|x) || p(z))` — encoder posterior should stay close to prior `N(0,I)`

### 2. KL Divergence Between Two Gaussians (closed form)

For `q = N(μ, σ²)` and `p = N(0, I)`:
```
KL(q || p) = (1/2)(σ² + μ² − 1 − log σ²)
```

**Derivation:**
```
KL(q||p) = E_q[log q(z)] − E_q[log p(z)]
         = E_q[−(z−μ)²/(2σ²) − (1/2)log(2πσ²)]
           − E_q[−z²/2 − (1/2)log(2π)]
         = −(1/2) − (1/2)log σ² + (1/2)E_q[z²]
         = −(1/2) − (1/2)log σ² + (1/2)(σ² + μ²)
         = (1/2)(σ² + μ² − 1 − log σ²)
```

This is differentiable w.r.t. μ and σ — enables direct backpropagation through the KL term.

### 3. Reparameterization in VAEs

Without reparameterization, `z ~ q_φ(z|x) = N(μ_φ(x), σ_φ²(x))` is a stochastic node — gradients cannot flow.

With reparameterization:
```
z = μ_φ(x) + σ_φ(x) ⊙ ε,   ε ~ N(0, I)
```

Now z is a **deterministic function** of x and ε, so:
```
∂z/∂μ_φ = 1,   ∂z/∂σ_φ = ε
```

Gradients flow through μ_φ and σ_φ (encoder) during backpropagation. ε is a fixed sample.

### 4. MC Dropout Uncertainty Decomposition

For T stochastic forward passes with predictions `ŷ₁, …, ŷ_T`:

**Predictive mean:** `ȳ = (1/T) Σ_t ŷ_t`

**Total predictive variance:**
```
Var[y*|x*] = (1/T) Σ_t ŷ_t² − ȳ²        (for regression)
```

This decomposes as:
```
Var[y*|x*] = Epistemic (variance in ŷ_t) + Aleatoric (output noise σ²)
```

MC Dropout approximates the posterior `p(w|D)` by treating dropout masks as a variational distribution, so the variance across forward passes approximates epistemic uncertainty.
