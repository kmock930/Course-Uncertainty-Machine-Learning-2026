# Lecture 11 – Sequential Decision Making and Reinforcement Learning

**Course:** ELG 5218 / CSI 5218 / EACJ 5600 – Uncertainty Evaluation in Engineering Measurements and Machine Learning  
**Date:** April 1, 2026  
**Instructor:** Bingze Xia, University of Ottawa (guest lecturer)

---

## Overview

This final lecture closes the course by moving from *estimating uncertainty* to *acting under
uncertainty*. It covers **Statistical Decision Theory** (how to map beliefs to optimal actions),
**Markov Decision Processes (MDPs)** (sequential planning with known models), and
**Reinforcement Learning (RL)** (sequential decision-making when the model is unknown and
must be learned from interaction). The lecture also includes a practical case study on **Deep
RL for UAV control** using Soft Actor–Critic (SAC).

---

## Learning Goals

By the end of this lecture you should be able to:

- Apply **Bayesian decision theory**: specify agents, actions, loss functions, and risk.
- Explain the difference between frequentist and Bayesian decision frameworks.
- Compute optimal decisions for one-shot classification (0-1 loss → MAP) and regression (L2 loss → MMSE).
- Model multi-stage decisions with **influence diagrams** and calculate the **value of information**.
- Formulate **contextual bandits** and implement UCB and Thompson sampling.
- Define an MDP (states, actions, transition model, reward function, discount factor).
- Derive and apply the **Bellman equations** for value functions V and Q.
- Implement **value iteration** and **policy iteration** to find the optimal policy.
- Describe **POMDPs** (partially observable MDPs) and belief-state planning.
- Explain **temporal difference (TD) learning**, SARSA, and Q-learning.
- Describe **Deep Q-Network (DQN)** and policy gradient methods (REINFORCE, Actor-Critic, PPO, SAC).
- Discuss the exploration–exploitation trade-off and off-policy learning.

---

## Topics Covered

### Decision Making Under Uncertainty (`Ch34_Decision_Making_Under_Uncertainty.pdf`)

**Statistical Decision Theory**
- Agents, states, actions, utility/loss functions, and risk.
- **Frequentist risk**: expected loss under data distribution.
- **Bayesian risk**: expected loss under posterior; minimized by the Bayes-optimal action.
- Connection to Bayesian inference from earlier lectures.

**One-Shot Decision Examples**
- **Classification** (0-1 loss): optimal decision is the MAP estimate.
- **Regression** (L2 loss): optimal decision is the posterior mean (MMSE).
- **Regression** (L1 loss): optimal decision is the posterior median.

**Influence Diagrams and Value of Information (VOI)**
- Oil wildcatter example: decision tree, expected utility computation.
- **Value of perfect information (VPI)**: how much is it worth to know the true state?
- Sequential decisions via dynamic programming.

**Contextual Bandits**
- Exploration–exploitation dilemma: balancing information gathering and reward maximization.
- **UCB** (Upper Confidence Bound): optimism in the face of uncertainty.
- **Thompson Sampling**: sample from posterior, act greedily on the sample.

**Markov Decision Processes**
- MDP tuple: (S, A, p(s'|s,a), R(s,a), γ).
- **Value function** V^π(s): expected discounted return under policy π.
- **Q-function** Q^π(s,a): expected discounted return for action a in state s.
- **Bellman equations**: recursive characterization of value functions.
- **POMDP**: partially observable MDP; belief state replaces state; connects to Kalman/particle filters (Lec 4).

**Planning Algorithms**
- **Value Iteration**: iteratively apply the Bellman optimality operator until convergence.
- **Policy Iteration**: alternate between policy evaluation and policy improvement.

### Reinforcement Learning (`ch_35_Reinforcement_Learning.pdf`)

**RL Overview**
- Key difference from planning: model p(s'|s,a) and R(s,a) are **unknown**.
- Agent must learn from observed (s, a, r, s') tuples via interaction.

**Value-Based RL**
- **TD learning**: bootstrap value estimates from the next-step value.
- **SARSA** (on-policy): update Q toward the next action actually taken.
- **Q-learning** (off-policy): update Q toward the greedy next action.
- **DQN** (Deep Q-Network): neural network approximates Q; experience replay; target network.

**Policy-Based RL**
- **Policy gradient (REINFORCE)**: directly optimize the policy by gradient ascent on expected return.
- **Actor-Critic**: actor updates policy; critic estimates value; reduces variance.
- **PPO** (Proximal Policy Optimization): clipped objective for stable updates.
- **DDPG** (Deep Deterministic Policy Gradient): off-policy actor-critic for continuous actions.

**Exploration and Practical Issues**
- ε-greedy vs. Boltzmann exploration vs. entropy regularization.
- **SAC** (Soft Actor–Critic): off-policy actor-critic with maximum entropy framework; built-in exploration.
- Application: **Deep RL for UAV control** using SAC (`UAV_SAC_example.pptx`).

---

## Materials

| File | Description |
|------|-------------|
| `Ch34_Decision_Making_Under_Uncertainty.pdf` | Lecture slides: Statistical decision theory and MDPs (40 slides) |
| `ch_35_Reinforcement_Learning.pdf` | Lecture slides: Reinforcement learning (35 slides) |
| `UAV_SAC_example.pptx` | Case study slides: Deep RL for UAV control using Soft Actor–Critic |

---

## Recommended Reading

- **Murphy, Kevin P. *Probabilistic Machine Learning: Advanced Topics*.** – Chapters 34 and 35.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| Bayes-optimal action | Action that minimizes Bayesian risk (expected loss under posterior) |
| MAP decision rule | Optimal under 0-1 loss: classify to the most probable class |
| MMSE decision rule | Optimal under L2 loss: predict the posterior mean |
| Value of information (VPI) | Expected utility gain from observing the true state before acting |
| MDP | Tuple (S, A, T, R, γ): formalism for sequential decision-making under uncertainty |
| Bellman equation | Recursive relation defining value functions; foundation of dynamic programming |
| Value iteration | Iteratively apply Bellman operator; converges to optimal value function |
| Policy iteration | Alternate policy evaluation and greedy policy improvement; also converges to optimal |
| POMDP | Partially Observable MDP; agent maintains a belief state; planning in belief space |
| TD learning | Temporal difference: update value estimates using one-step bootstrapped targets |
| Q-learning | Off-policy TD that converges to the optimal Q-function |
| DQN | Deep Q-Network: neural Q-function with experience replay and target network |
| PPO | Proximal Policy Optimization: stable policy gradient with clipped update |
| SAC | Soft Actor–Critic: maximum-entropy off-policy RL; natural uncertainty/exploration balance |
