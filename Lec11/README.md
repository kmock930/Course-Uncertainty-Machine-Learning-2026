# Lecture 11 – Sequential Decision Making and Reinforcement Learning

> **Revision summary** | ELG 5218 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
> Slides: `Ch34_Decision_Making_Under_Uncertainty.pdf`, `ch_35_Reinforcement_Learning.pdf`, `UAV_SAC_example.pptx`  
> Reading: Murphy PML Advanced Topics, Ch. 34–35

---

## Core Ideas

### Bayesian Decision Theory
- An agent observes state, takes action, incurs loss. Goal: minimise expected loss (risk).
- **Bayes-optimal action:** minimise posterior expected loss `ρ(a|x) = E_θ[L(a, θ) | x]`
  - 0-1 loss → MAP estimate (most probable class)
  - L2 loss → posterior mean (MMSE)
  - L1 loss → posterior median

### Markov Decision Process (MDP)
- Tuple `(S, A, p(s'|s,a), R(s,a), γ)`: states, actions, transition model, reward, discount.
- **Value function:** `V^π(s) = E_π[Σₜ γᵗ Rₜ | s₀=s]`
- **Q-function:** `Q^π(s,a) = E_π[Σₜ γᵗ Rₜ | s₀=s, a₀=a]`
- **Bellman optimality equations:**

  `V*(s) = max_a [R(s,a) + γ Σ_{s'} p(s'|s,a) V*(s')]`

  `Q*(s,a) = R(s,a) + γ Σ_{s'} p(s'|s,a) max_{a'} Q*(s',a')`

- **Value iteration:** repeatedly apply Bellman operator until convergence.
- **Policy iteration:** alternate policy evaluation (solve for V^π) and greedy improvement.
- **POMDP:** agent cannot observe full state; maintains a **belief state** (distribution over states) — links back to Kalman/particle filters (Lec 4).

### Contextual Bandits (one-step sequential decision)
- **Exploration–exploitation dilemma:** balance gathering information vs. maximising reward.
- **UCB:** select action with highest upper confidence bound on expected reward.
- **Thompson sampling:** sample θ from posterior, act greedily on the sample.

### Reinforcement Learning (model-unknown)
- Agent learns from `(s, a, r, s')` tuples by interacting with the environment.

**Value-based RL:**
- **TD learning:** `V(s) ← V(s) + α(r + γV(s') − V(s))`
- **Q-learning (off-policy):** `Q(s,a) ← Q(s,a) + α(r + γ max_{a'} Q(s',a') − Q(s,a))`
- **DQN:** neural Q-function + experience replay + target network.

**Policy-based RL:**
- **REINFORCE:** `∇_θ J ≈ (1/T) Σₜ ∇_θ log π_θ(aₜ|sₜ) · Gₜ`
- **Actor-Critic:** actor = policy gradient; critic = value function baseline → lower variance.
- **PPO:** clipped surrogate objective for stable updates.
- **SAC (Soft Actor-Critic):** maximum-entropy RL; optimises `J = E[Σₜ (Rₜ + α H(π(·|sₜ)))]` — built-in exploration via entropy bonus. Used in UAV control case study.

---

## Things to Remember

- MDP requires a **known model**; RL does not — it learns from interaction.
- POMDP = MDP over belief states; computationally intractable in general → approximate solvers.
- Q-learning converges to optimal Q* regardless of the behaviour policy (off-policy).
- SAC's entropy term `α H(π)` prevents premature convergence to suboptimal deterministic policies.
- Value iteration and policy iteration both converge to `V*` and `π*` for finite MDPs.

---

## Practice Questions

Key revision topics: Bellman equations, value/policy iteration convergence, Q-learning update derivation, Thompson sampling, POMDP belief-state update, SAC entropy objective. No dedicated problem set — see Murphy PML Advanced Topics Ch. 34–35 exercises.

---

## Key Derivations

### 1. Bellman Expectation Equation

By definition of the value function:
```
V^π(s) = E_π[Σₜ γᵗ Rₜ | s₀=s]
        = E_π[R₀ + γ Σₜ₌₁^∞ γᵗ⁻¹ Rₜ | s₀=s]
        = E_π[R(s,a)] + γ E_π[V^π(s₁) | s₀=s]
        = Σ_a π(a|s)[R(s,a) + γ Σ_{s'} p(s'|s,a) V^π(s')]
```

For the **optimal** policy (Bellman optimality):
```
V*(s) = max_a [R(s,a) + γ Σ_{s'} p(s'|s,a) V*(s')]
```

Value iteration repeatedly applies this operator: `Vₖ₊₁ = T* Vₖ`. It converges because the Bellman operator `T*` is a γ-contraction: `||T*V − T*U||_∞ ≤ γ||V−U||_∞`.

### 2. Q-Learning Update Derivation

Q-function: `Q*(s,a) = R(s,a) + γ Σ_{s'} p(s'|s,a) max_{a'} Q*(s',a')`

TD error (temporal difference):
```
δₜ = Rₜ + γ max_{a'} Q(sₜ₊₁, a') − Q(sₜ, aₜ)
```

Online update rule:
```
Q(sₜ,aₜ) ← Q(sₜ,aₜ) + α δₜ
```

This is stochastic approximation of the fixed-point equation. Converges to `Q*` under standard conditions (all state-action pairs visited infinitely often, decaying learning rate).

**Why off-policy?** The update uses `max_{a'} Q(sₜ₊₁, a')` regardless of the action actually taken — the behaviour policy (exploration) decouples from the target policy (greedy).

### 3. Policy Gradient Theorem

**Goal:** maximise `J(θ) = E_{τ~π_θ}[Σₜ γᵗ Rₜ]`.

**REINFORCE gradient:**
```
∇_θ J(θ) = E_{τ~π_θ}[Σₜ ∇_θ log π_θ(aₜ|sₜ) · Gₜ]
```

where `Gₜ = Σₜ'≥ₜ γᵗ'⁻ᵗ Rₜ'` (return from time t).

**Derivation:**
```
∇_θ E_{τ}[G] = ∇_θ ∫ p_θ(τ) G(τ) dτ
             = ∫ G(τ) ∇_θ p_θ(τ) dτ
             = ∫ G(τ) p_θ(τ) ∇_θ log p_θ(τ) dτ    ← log-derivative trick
             = E_{τ}[G(τ) ∇_θ log p_θ(τ)]
```

Since `∇_θ log p_θ(τ) = Σₜ ∇_θ log π_θ(aₜ|sₜ)` (transition and reward terms don't depend on θ), we recover REINFORCE.

**Baseline variance reduction:** adding a baseline `b(sₜ)` does not bias the gradient (since `E[∇_θ log π b] = 0`) but reduces variance. The optimal baseline is `b(sₜ) = V^π(sₜ)` — leading to the **advantage function** `Aₜ = Gₜ − V^π(sₜ)`.

### 4. SAC Entropy-Augmented Objective

SAC maximises the **soft value function**:
```
J(π) = Σₜ E_{(sₜ,aₜ)~π}[R(sₜ,aₜ) + α H(π(·|sₜ))]
```

The entropy term `H(π(·|s)) = −E_{a~π}[log π(a|s)]` with temperature `α`:
- Encourages exploration: prevents collapse to a deterministic policy too early.
- Leads to the **soft Bellman equation:**
  ```
  Q*(s,a) = R(s,a) + γ E_{s'}[V*(s')]
  V*(s) = E_{a~π*}[Q*(s,a) − α log π*(a|s)]
  ```
- Optimal policy: `π*(a|s) ∝ exp(Q*(s,a)/α)` — a softmax (Boltzmann) distribution over Q-values.
