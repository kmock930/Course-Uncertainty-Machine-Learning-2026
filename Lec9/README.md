# Lecture 9 – Diffusion Models

> **Revision summary** | ELG 5218 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
> Slides: `Diffusion_Models_Uncertainty_in_ML_course_2026.pdf` · Notebooks: `Diffusion_Session_Notebook_v2.ipynb`, `DiffusionDet_real_inference_demo.ipynb`, `SANA_diffusion_trajectory_notebook.ipynb`  
> Reading: Ho et al. (2020) DDPM; Song et al. (2021) Score-SDEs; Chen et al. (2023) DiffusionDet

---

## Core Ideas

### Why Diffusion Models Fit This Course
- Diffusion models learn a **full distribution** — not a point estimate.
- Uncertainty is represented by the diversity of generated samples.
- Builds on Langevin dynamics (Lec 3): score `∇_x log p(x)` drives denoising.

### DDPM – Forward Process
- Gradually corrupt `x₀` over `T` steps by adding Gaussian noise:

  `q(xₜ | xₜ₋₁) = N(xₜ; √(1−βₜ) xₜ₋₁, βₜ I)`

- **Closed-form marginal:** `q(xₜ | x₀) = N(xₜ; √ᾱₜ x₀, (1−ᾱₜ)I)` where `ᾱₜ = ∏ᵢ₌₁ᵗ (1−βᵢ)`
- At `t = T`: `xₜ ≈ N(0, I)` — pure noise.

### DDPM – Training Objective
- A neural network `ε_θ(xₜ, t)` predicts the noise `ε` added to `x₀` at step `t`.
- **Loss:** `L = E_{x₀, ε, t} [||ε − ε_θ(√ᾱₜ x₀ + √(1−ᾱₜ) ε, t)||²]`
- Equivalent to denoising score matching: the network learns the score `∇_xₜ log q(xₜ)`.

### DDPM – Reverse Sampling
- Start from `xₜ ~ N(0, I)`, iteratively denoise:

  `xₜ₋₁ = (1/√αₜ)(xₜ − βₜ/√(1−ᾱₜ) · ε_θ(xₜ, t)) + √βₜ z`, `z ~ N(0,I)`

### Extensions

| Extension | Key idea |
|---|---|
| **Conditional diffusion** | Condition on class label, text, or partial observation during reverse process |
| **Latent diffusion** | Run diffusion in a compressed VAE latent space; faster and more memory-efficient |
| **DDIM** | Deterministic non-Markovian sampler; same model weights; fewer steps (~50 vs 1000) |
| **DPM-Solver** | ODE-based fast sampler; ~20 function evaluations |
| **CSDI** | Conditional score-based diffusion for time-series imputation |
| **DiffusionDet** | Diffusion over bounding boxes for object detection |

---

## Things to Remember

- The forward process is **fixed** (not learned); only the reverse network `ε_θ` is trained.
- `ᾱₜ` controls the signal-to-noise ratio: `ᾱₜ → 0` as `t → T`.
- Latent diffusion (Stable Diffusion) is more efficient but requires a pre-trained VAE encoder/decoder.
- The ε-prediction loss and denoising score matching are mathematically equivalent.

---

## Practice Questions

Implementation exercises are in [`Diffusion_Session_Notebook_v2.ipynb`](Diffusion_Session_Notebook_v2.ipynb). See also the reading papers for theoretical questions on DDPM and score-based models.

---

## Key Derivations

### 1. Forward Process Closed-Form Marginal

From the one-step formula `q(xₜ|xₜ₋₁) = N(√(1−βₜ) xₜ₋₁, βₜI)`, we can show by induction that:

```
q(xₜ|x₀) = N(√ᾱₜ x₀, (1−ᾱₜ)I)
```

where `αₜ = 1−βₜ` and `ᾱₜ = ∏ᵢ₌₁ᵗ αᵢ`.

**Proof (two-step induction):**  
Base: `q(x₁|x₀) = N(√α₁ x₀, β₁I)` = `N(√ᾱ₁ x₀, (1−ᾱ₁)I)`. ✓

Inductive step — apply the Gaussian marginalisation `z₁ ~ N(μ₁, σ₁²)`, `z₂|z₁ ~ N(√a z₁, b²)`:
```
z₂ ~ N(√a μ₁, a σ₁² + b²)
```

At step t: `q(xₜ|x₀) = N(√αₜ · √ᾱₜ₋₁ x₀, αₜ(1−ᾱₜ₋₁) + βₜ) = N(√ᾱₜ x₀, (1−ᾱₜ)I)` ✓

**Implication:** we can sample any noisy `xₜ` directly from `x₀` in one step: `xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε`, `ε ~ N(0,I)`. No need to simulate T steps during training.

### 2. DDPM Training Objective (ε-prediction)

Full ELBO for the reverse process contains a sum of KL divergences. The simplified Ho et al. (2020) objective discards a weighting term and reduces to:

```
L_simple = E_{t, x₀, ε}[||ε − ε_θ(xₜ, t)||²]
```

**Derivation sketch:**

The KL term for timestep t is:
```
L_{t−1} = E_q[KL(q(xₜ₋₁|xₜ,x₀) || p_θ(xₜ₋₁|xₜ))]
```

Both distributions are Gaussian (given `xₜ` and `x₀`). The posterior mean:
```
μ̃ₜ(xₜ, x₀) = (√ᾱₜ₋₁βₜ x₀ + √αₜ(1−ᾱₜ₋₁) xₜ) / (1−ᾱₜ)
```

Express `x₀ = (xₜ − √(1−ᾱₜ) ε) / √ᾱₜ` and substitute:
```
μ̃ₜ = (1/√αₜ)(xₜ − βₜ/√(1−ᾱₜ) · ε)
```

Train a network `ε_θ` to predict `ε`, giving the simple MSE loss. KL minimisation reduces to noise prediction error.

### 3. Score Function Connection

The **score function** is `∇_x log p(x)`. For the noised distribution:
```
∇_{xₜ} log q(xₜ|x₀) = −(xₜ − √ᾱₜ x₀) / (1−ᾱₜ) = −ε / √(1−ᾱₜ)
```

So learning `ε_θ(xₜ, t)` is equivalent to learning the **score** `s_θ(xₜ, t) = −ε_θ / √(1−ᾱₜ)`.

This connects DDPM to **score-based generative models** (Song et al. 2021), where the reverse SDE uses:
```
dx = [f(x,t) − g(t)² ∇_x log pₜ(x)] dt + g(t) dW̄
```
