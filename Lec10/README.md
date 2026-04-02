# Lecture 10 – Multimodal Learning and Attention

> **Revision summary** | ELG 5218 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
> Slides: `MultimodalLearning.pdf`, `AttentionTransformerFusion.pdf` · Notebook: `AttentionFusion.ipynb`  
> Reading: Wojke et al. (2017) DeepSORT; Stanford CS231N Lec 8; CMU Multimodal ML (videos)

---

## Core Ideas

### Why Multimodal Fusion?
- Each sensor modality is noisy or ambiguous alone; combining complementary signals improves robustness.
- Fusion strategies:
  - **Early fusion:** concatenate raw features → single model.
  - **Late fusion:** independent models → combine predictions.
  - **Intermediate (deep) fusion:** merge at hidden feature levels; most flexible.

### Attention Mechanism
- **Scaled dot-product attention:**

  `Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V`

  - **Q** (query): what we are looking for.
  - **K** (key): what each token offers.
  - **V** (value): what we actually retrieve.
  - Dividing by `√d_k` prevents softmax saturation for large `d_k`.

- **Multi-head attention:** run `h` attention heads in parallel, concatenate and project:

  `MultiHead(Q,K,V) = Concat(head₁, …, headₕ) Wᴼ`

  Each head learns different relationship patterns.

- **Cross-modal attention:** Q from one modality (e.g. radar), K/V from another (e.g. camera) — enables directed information retrieval across sensors.

### Transformer Block
1. Multi-head self-attention
2. Add & LayerNorm (residual connection)
3. Feed-forward network (2-layer MLP)
4. Add & LayerNorm

- **Positional encoding:** sinusoidal or learned; needed because attention is permutation-invariant.
- **Vision Transformer (ViT):** split image into patches → treat each patch as a token → standard Transformer.

### Fusion and Uncertainty
- Attention weights ∈ [0,1] can be interpreted as **soft association probabilities** — analogous to PDAF (Lec 8).
- **Probabilistic/Bayesian attention:** propagate uncertainty through attention weights; connects to JPDA.
- **DeepSORT:** Kalman filter (Lec 4) for motion + deep appearance embedding for re-identification.

---

## Things to Remember

- Attention is `O(n²)` in sequence length — becomes expensive for long sequences.
- Cross-attention is the key operation for multimodal fusion in Transformers.
- Without positional encoding, Transformers are bag-of-tokens models (order-invariant).
- ViT outperforms CNNs at large scale; CNNs are better with limited data (inductive biases).

---

## Practice Questions

Implementation exercises are in [`AttentionFusion.ipynb`](AttentionFusion.ipynb). Key topics to review: attention derivation, multi-head attention parameter count, cross-modal fusion design, and probabilistic interpretation of attention weights.
