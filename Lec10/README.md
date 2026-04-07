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

---

## Key Derivations

### 1. Scaled Dot-Product Attention — Why Divide by √d_k

For query `q` and key `k` both drawn from `N(0,I)` with dimension `d_k`:
```
q · k = Σⱼ qⱼkⱼ
E[q·k] = 0,   Var[q·k] = d_k
```

The dot product has standard deviation `√d_k`. Without scaling, the softmax input has large magnitude, pushing softmax into the **saturation region** where gradients vanish:
```
softmax(x_i) → 1{i = argmax},   ∂softmax/∂x → 0
```

Dividing by `√d_k` normalises variance to 1, keeping softmax in a useful gradient regime.

### 2. Attention as Soft Lookup

Attention can be seen as a **differentiable database lookup**:

Given a query `q`, keys `K = [k₁|…|kN]ᵀ`, values `V = [v₁|…|vN]ᵀ`:

1. **Score:** `eᵢ = q·kᵢ / √d_k` — how well the query matches key i
2. **Weight:** `αᵢ = exp(eᵢ) / Σⱼ exp(eⱼ)` — soft selection probabilities, `Σᵢ αᵢ = 1`
3. **Retrieve:** `o = Σᵢ αᵢ vᵢ` — weighted sum of values

For hard (argmax) lookup: one αᵢ = 1, rest = 0. Softmax approximates this differentiably.

### 3. Multi-Head Attention Parameter Count

For `h` heads, each with key/query/value dimension `d_k = d_v = d_model/h`:

- Per head: 3 projection matrices `Wᵢ^Q, Wᵢ^K ∈ R^{d_{model}×d_k}`, `Wᵢ^V ∈ R^{d_{model}×d_v}`
- Per head parameters: `3 · d_model · (d_model/h)`
- Total for h heads: `3 · d_model²`
- Output projection `W^O ∈ R^{d_model×d_model}`: `d_model²`

**Total attention layer:** `4 d_model²` parameters — **independent of number of heads h**.

Multi-head costs the same as single-head but learns `h` different representation subspaces.

### 4. Cross-Modal Attention for Sensor Fusion

Given modality A (e.g. radar features `F_A`) and modality B (e.g. camera features `F_B`):

```
Q = F_A W^Q,   K = F_B W^K,   V = F_B W^V
Fused = softmax(QKᵀ/√d_k) V
```

**Interpretation:** each radar feature (query) attends over camera features (keys) to retrieve the most relevant camera information (values). The attention weight `α_{ij}` represents how much radar element i should borrow from camera element j.

**Connection to probabilistic fusion:** the attention weights are analogous to the PDAF association probabilities `βᵢ` (Lec 8) — both compute a soft assignment between elements across two sources.
