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
