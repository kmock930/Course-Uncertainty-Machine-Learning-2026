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
