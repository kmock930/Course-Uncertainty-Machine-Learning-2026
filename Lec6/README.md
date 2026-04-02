# Lecture 6 – Variational Autoencoders and Bayesian Neural Networks

**Course:** ELG 5218 / CSI 5218 / EACJ 5600 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
**Date:** February 25, 2026  
**Instructor:** Miodrag Bolic, University of Ottawa

---

## Overview

This lecture brings together deep learning and probabilistic modeling.
It first shows how **neural networks can parameterize probability distributions**
(mixture density networks, autoregressive models, latent-variable models),
then develops **Variational Autoencoders (VAEs)** as a deep generative model trained via
the ELBO and the reparameterization trick, and finally covers **Bayesian Neural Networks (BNNs)**,
where uncertainty is placed over network weights rather than over a latent space.

---

## Learning Goals

By the end of this lecture you should be able to:

- Explain how neural networks act as flexible parameterizers of probability distributions.
- Describe Mixture Density Networks (MDNs) and autoregressive factorizations.
- Derive the **VAE objective (ELBO)** and explain the role of the encoder and decoder.
- Implement a VAE using the reparameterization trick.
- Apply VAEs to anomaly detection and generative modeling tasks.
- Explain the differences between **aleatoric** and **epistemic** uncertainty in neural networks.
- Compare inference methods for BNNs: MCMC (SGLD, HMC), Variational (Bayes-by-Backprop), and MC Dropout.
- Choose the appropriate BNN inference method for a given application.

---

## Topics Covered

### Neural Networks as Distribution Parameterizers (`NN_Parameterization.pdf`)
- Universal approximation → networks can learn any map into a parameter space.
- Distribution output heads:
  - **Mixture Density Networks (MDNs)**: mixture weights, means, variances output by the network.
  - **Autoregressive factorizations**: chain rule decomposition for sequential data.
  - **Latent-variable models**: VAE family.
- **Gaussian likelihood heads**: mean–variance networks (MVE), learning heteroscedastic σ(x).
- Failure modes: variance collapse, negative variances, calibration issues.

### Variational Autoencoders (`VAE.pdf`)
- Motivation: combining neural network expressiveness with principled Bayesian uncertainty.
- Deterministic autoencoders vs. VAEs: representation learning vs. generative modeling.
- **VAE generative model**: p(x|z) decoder + p(z) prior.
- **VAE inference model**: q(z|x) encoder (amortized inference).
- **ELBO for VAEs**: reconstruction term + KL regularization to prior.
- **Reparameterization trick**: differentiable sampling through the encoder.
- Discrete VAEs (Gumbel-softmax) for categorical latent variables.
- Application: **VAE for Breathing Anomaly Detection** (biomedical time-series).

### Bayesian Neural Networks and MC Dropout (`BNN.pdf`)
- From point estimates to weight distributions: placing priors on network weights.
- **MCMC for BNNs**: Stochastic Gradient Langevin Dynamics (SGLD), HMC.
- **Variational inference: Bayes-by-Backprop** (Blundell et al., 2015):
  - Factored Gaussian variational family over weights.
  - Reparameterization through weights; local reparameterization trick.
- **MC Dropout** (Gal & Ghahramani, 2016):
  - Dropout at test time as approximate Bayesian inference.
  - Simple implementation; uncertainty via prediction variance across forward passes.
- When to use each method: VI vs. MC Dropout decision guide.
- Diagnostics and calibration of BNN uncertainty estimates.

---

## Materials

| File | Description |
|------|-------------|
| `NN_Parameterization.pdf` | Lecture slides: Neural networks as probability distribution parameterizers (27 slides) |
| `VAE.pdf` | Lecture slides: Variational Autoencoders (36 slides) |
| `BNN.pdf` | Lecture slides: Bayesian Neural Networks and MC Dropout (32 slides) |
| `VAE_Breathing_Anomaly.pdf` | Application slides: VAE for breathing anomaly detection |
| `NN_Parameterization_Notebook.ipynb` | Notebook: NN distribution parameterization exercises |
| `VAE_Breathing_Anomaly.ipynb` | Notebook: VAE for biomedical anomaly detection |
| `Bnn_notebook.ipynb` | Notebook: Bayesian Neural Network implementations |

---

## Recommended Reading

- **Kingma, D.P. & Welling, M. *An Introduction to Variational Autoencoders*.**
- **Huijse, Pablo. *Variational Autoencoder***, chapter from *Bayesian Learning and Neural Networks*.
  [Online](https://phuijse.github.io/BLNNbook/chapters/neural_networks/vae.html)
- **Huijse, Pablo. *Bayesian Neural Networks with NumPyro***, chapter from *Bayesian Learning and Neural Networks*.
  [Online](https://phuijse.github.io/BLNNbook/chapters/neural_networks/bayesian.html)
- **Aziz, Wilker. *DPM 2 – Variational Inference for Deep Continuous LVMs*.**
  [UvA DLC Notebooks](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/deep_probabilistic_models_II/tutorial_2b.html)

---

## Practice Questions

Relevant practice sets from the `Practice Questions/` folder:

- `Problems_VAE_Questions.pdf` / `Problems_VAE_Solutions.pdf` – ELBO derivation for VAEs, encoder/decoder design, reparameterization trick, anomaly detection with VAEs.
- `Problems_NN_Parameterization_Questions.pdf` / `Problems_NN_Parameterization_Solutions.pdf` – MDN design, heteroscedastic loss, Gaussian output heads.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| Mixture Density Network (MDN) | NN outputting parameters of a mixture distribution |
| Heteroscedastic uncertainty | Input-dependent output variance learned by the network |
| VAE encoder | Amortized inference network: q(z\|x) → approximate posterior |
| VAE decoder | Generative network: p(x\|z) → likelihood of data given latent code |
| ELBO (VAE) | Reconstruction loss − KL(q(z\|x)\|\|p(z)); trained by maximizing |
| Reparameterization | z = μ + σ·ε, ε~N(0,1); makes sampling differentiable |
| Bayes-by-Backprop | VI over NN weights: Gaussian variational family, trained with ELBO |
| MC Dropout | Dropout at test time provides approximate posterior predictive samples |
| SGLD | Stochastic Gradient Langevin Dynamics: MCMC for BNNs using noisy gradients |
| Epistemic uncertainty | Reducible weight uncertainty; captured by BNNs; decreases with more data |
