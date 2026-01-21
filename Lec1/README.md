# Lecture 1: Probabilistic Reasoning

## Summary

This lecture provides a foundational introduction to Bayesian modeling and probabilistic reasoning for uncertainty quantification in machine learning and engineering measurements.

### Key Topics

#### 1. Uncertainty Quantification
- Understanding and characterizing the impact of limited knowledge on quantities of interest
- Representing uncertainty through confidence intervals, credible intervals, and probability distributions
- Difference between frequentist confidence intervals and Bayesian credible intervals

#### 2. Bayesian Inference Framework
The Bayesian approach combines three key components:
- **Prior Distribution** p(θ): Encodes initial beliefs about model parameters before observing data
- **Likelihood Function** p(D|θ): Describes the probabilistic model of how data is generated
- **Posterior Distribution** p(θ|D): Updated beliefs after observing data, computed via Bayes' theorem

**Bayes' Theorem**: Posterior ∝ Likelihood × Prior

#### 3. Inference Methods
- **Maximum Likelihood Estimation (MLE)**: Point estimate that maximizes likelihood
- **Maximum A Posteriori (MAP)**: Point estimate that maximizes posterior distribution
- **Highest Probability Density (HPD)**: Identifies most credible parameter regions
- Posterior predictive distributions for making predictions

#### 4. Practical Application: Binomial Model
The lecture demonstrates Bayesian inference using a binomial model for estimating success rates:
- Binomial likelihood for modeling binary outcomes
- Beta distribution as a conjugate prior
- Analytical and Monte Carlo methods for posterior computation
- Posterior predictive inference

### Learning Objectives
By the end of this lecture, students should understand:
- The principles of Bayesian modeling and inference
- How to construct and interpret probabilistic models
- The role of prior information in statistical inference
- Methods for summarizing and visualizing posterior distributions
- Practical implementation of Bayesian analysis for simple models

### Materials
- [Lecture Slides (PDF)](/Lec1/Lec1.pdf)
- [Introduction Slides (PDF)](/Lec1/Lec_intro.pdf)
- [Jupyter Notebook: Bayesian Modeling Examples](/Lec1/Lec1_Bayesian_Modeling.ipynb)

### Recommended Reading
- Chapters 1 and 2 from Villani, Mattias, (2025), Bayesian Learning
- Probability refresher from Prince, Simon J.D. (2025), Understanding Deep Learning
