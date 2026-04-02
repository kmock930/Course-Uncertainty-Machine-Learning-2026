# Lecture 10 – Multimodal Learning and Attention

**Course:** ELG 5218 / CSI 5218 / EACJ 5600 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
**Date:** March 25, 2026  
**Instructor:** Miodrag Bolic, University of Ottawa

---

## Overview

This lecture covers **deep learning approaches to multimodal sensor fusion**, where
information from different modalities (camera, LiDAR, radar, language, etc.) must be
combined in a coherent, uncertainty-aware way. It introduces **attention mechanisms**
and **Transformers** as the dominant architectural building blocks, and then
discusses how these tools can be extended to propagate and represent uncertainty
in multi-sensor systems.

---

## Learning Goals

By the end of this lecture you should be able to:

- Explain why multimodal learning improves robustness and coverage over single-modality models.
- Define **embeddings** and describe how distance metrics in embedding space enable cross-modal matching.
- Derive and compute **scaled dot-product attention** (queries, keys, values).
- Explain **multi-head attention** and its role in capturing multiple relationship patterns.
- Describe the full **Transformer block** (attention + FFN + layer norm + residual).
- Explain positional encoding and Vision Transformers (ViT) for image data.
- Apply Transformers to **camera–LiDAR–radar fusion** for detection and tracking.
- Connect probabilistic/Bayesian attention to uncertainty-aware fusion.
- Describe the **DeepSORT** tracking framework and its deep association metric.

---

## Topics Covered

### Deep Learning for Multimodal Sensor Fusion (`MultimodalLearning.pdf`)

**Introduction**
- Why multimodal? Complementary signals; robustness to sensor failure; reduced ambiguity.
- Applications: autonomous driving (camera–LiDAR–radar), robotics, air/ground surveillance, healthcare.

**Distance Metrics and Embeddings**
- Embedding functions map raw inputs (images, point clouds, text) to a shared metric space.
- Distance metrics in embedding space: Euclidean, cosine, Mahalanobis.
- Contrastive learning and cross-modal retrieval.

**Multimodal Embeddings**
- Early fusion: concatenate raw features before any processing.
- Late fusion: process each modality independently, combine predictions.
- Intermediate (deep) fusion: merge at intermediate feature levels.
- Cross-modal attention: dynamically weight information across modalities.

**Deep Fusion Operators**
- Hadamard product, gated fusion, FiLM (Feature-wise Linear Modulation).
- Graph-based fusion for multi-target scenarios.

**Modern Tracking Stack Summary**
- Full pipeline: detection → feature extraction → embedding → association → fusion → tracking.

### Attention, Transformers, and Fusion (`AttentionTransformerFusion.pdf`)

**Attention Mechanism**
- Intuition: a self-driving car attends more to a nearby pedestrian than to background clouds.
- **Scaled dot-product attention**: Attention(Q, K, V) = softmax(QK^T / √d_k) · V.
- Queries, keys, values: Q from the query token, K and V from the context.
- **Multi-head attention**: run h attention heads in parallel to capture diverse patterns.

**Transformers**
- Why attention alone is insufficient: no positional information, no nonlinearity.
- **Transformer block**: multi-head attention → add & layer norm → FFN → add & layer norm.
- **Positional encoding**: sinusoidal or learned; enables the model to use sequence order.
- **Vision Transformer (ViT)**: patch embedding of images; global attention over patches.
- **Transformers for radar + camera fusion**: cross-attention over modality tokens.

**Fusion and Uncertainty**
- **Probabilistic/Bayesian attention**: attention weights as uncertain soft assignments.
- Connecting attention-based fusion to JPDA (Lecture 8) data association.
- Uncertainty propagation through attention layers.

---

## Materials

| File | Description |
|------|-------------|
| `MultimodalLearning.pdf` | Lecture slides: Deep learning for multimodal sensor fusion (41 slides) |
| `AttentionTransformerFusion.pdf` | Lecture slides: Attention, Transformers, and probabilistic fusion (59 slides) |
| `AttentionFusion.ipynb` | Notebook: Attention mechanism and multimodal fusion implementations |

---

## Recommended Videos

- **Attention and Transformers** – Lecture 8, Stanford CS231N, Spring 2025:
  [YouTube](https://www.youtube.com/watch?v=RQowiOF_FvQ)
- **Multimodal Transformers** – CMU Multimodal Machine Learning, Fall 2023 (Parts 1 & 2):
  [YouTube](https://www.youtube.com/watch?v=90AGqQYYZc0&list=PL-Fhd_vrvisMYs8A5j7sj8YW1wHhoJSmW&index=9)

---

## Recommended Reading

- **Wojke, N., Bewley, A., & Paulus, D. (2017). *Simple Online and Realtime Tracking with a Deep Association Metric (DeepSORT).*** 2017 IEEE ICIP.
  [arXiv:1703.07402](https://doi.org/10.48550/arXiv.1703.07402)

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| Multimodal learning | Combining information from multiple sensor types or data modalities |
| Embedding | Dense vector representation of a raw input; enables cross-modal comparison |
| Query / Key / Value | Three roles in attention: Q queries against K to compute attention, V is the retrieved content |
| Scaled dot-product attention | Attention(Q,K,V) = softmax(QKᵀ/√d_k)V; efficiently computes soft content retrieval |
| Multi-head attention | h parallel attention heads; each learns different relationship patterns |
| Positional encoding | Injects token position into embeddings; needed because attention is permutation-invariant |
| Vision Transformer (ViT) | Applies Transformer to image patches; replaces convolutional feature extraction |
| Cross-modal attention | Q from one modality (e.g., radar), K/V from another (e.g., camera); enables cross-sensor fusion |
| Probabilistic attention | Attention weights interpreted as soft association probabilities; connects to PDAF/JPDA |
| DeepSORT | Multi-object tracker combining Kalman filter with deep appearance embeddings |
