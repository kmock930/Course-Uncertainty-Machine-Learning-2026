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
