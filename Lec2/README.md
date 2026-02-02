# Lecture 2: Bayesian Inference for Gaussians, Bayesian Linear and Logistic Regression

## Summary

This lecture extends Bayesian inference methods to Gaussian models and explores Bayesian approaches to linear and logistic regression, which are fundamental techniques in machine learning and statistical modeling.

### Key Topics

#### 1. Gaussian Models and Inference
- **Gaussian distributions**: Properties and formulas for univariate and multivariate Gaussians
- **Bayesian estimation** for Gaussian parameters (mean and variance)
- **Conjugate priors** for Gaussian distributions:
  - Normal prior for the mean (when variance is known)
  - Inverse-Gamma prior for the variance (when mean is known)
  - Normal-Inverse-Gamma for joint inference of mean and variance
- Posterior computations and credible intervals

#### 2. Bayesian Linear Regression
Linear regression models the relationship between input features and continuous outputs:
- **Model formulation**: y = Xβ + ε, where ε ~ N(0, σ²)
- **Prior distributions** on regression coefficients β and noise variance σ²
- **Posterior inference** for regression parameters
- **Predictive distributions** for new observations
- Advantages over frequentist approaches:
  - Natural uncertainty quantification
  - Regularization through priors
  - Principled handling of small sample sizes

#### 3. Bayesian Logistic Regression
Logistic regression for binary classification problems:
- **Model**: p(y=1|x) = σ(xᵀβ), where x is the input feature vector and σ is the sigmoid function
- **Likelihood**: Bernoulli distribution for binary outcomes
- **Prior specification** for regression coefficients
- **Posterior computation** (typically requires approximation methods)
- **Classification and prediction** with uncertainty estimates
- Applications in probabilistic classification tasks

#### 4. Practical Considerations
- Choosing appropriate priors based on domain knowledge
- Computational methods for posterior inference
- Model comparison and validation
- Interpreting posterior distributions and making decisions under uncertainty

### Learning Objectives
By the end of this lecture, students should be able to:
- Apply Bayesian inference to Gaussian models
- Implement Bayesian linear regression for prediction tasks
- Use Bayesian logistic regression for classification problems
- Choose appropriate prior distributions for regression parameters
- Interpret and communicate uncertainty in regression models
- Compare Bayesian and frequentist approaches to regression

### Materials
- [Gaussian Formulas Reference (PDF)](/Lec2/GaussianFormulas.pdf)
- [Gaussian Models Lecture (PDF)](/Lec2/GaussianModels.pdf)
- [Bayesian Linear Regression Slides (PDF)](/Lec2/tpmi_w19_lec5_slides_print.pdf) - from slide 19
- [Bayesian Logistic Regression Slides (PDF)](/Lec2/tpmi_w19_lec6_slides_print.pdf) - from slide 15

### Recommended Reading
- Chapters 2, 3.3, 5, and 8.2 from Villani, Mattias, (2025), Bayesian Learning
