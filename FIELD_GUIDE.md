# Field guide: key concepts and good practices

This course teaches you **key concepts** (grouped by theme) and **good practices** when tackling a new data-driven problem (grouped by the stages of the project). Each item ends with the lectures where it is discussed, so you can consult it for details.

This should help you fulfil the last two objectives of the course: *identify appropriate methods based on the application of interest* and *develop data-driven strategies for new problems*.

The list grows with the course. Each lecture notebook ends with a notes cell, "Field guide: what this lecture adds", generated from this file by `Lectures/sync_field_guide.py`: edit this file, not those cells, then run the script.

## Key concepts

### Probability

- **An unknown is treated as a random variable (rv).** Continuous rv's are described by a pdf, i.e. a probability density, from which probability results by integrating the pdf in an interval. Discrete rv's are described by a pmf, i.e. a probability value. (L1, L22)
- **Joint, marginal and conditional distributions.** Marginalizing a variable means integrating it out. (L1, L3, L4)
- **Moments**: the expected value is linear; the variance of a sum of independent variables is the sum of their variances; mean and mode differ for asymmetric distributions. (L3)
- **Change of variables**: how a pdf transforms when its variable is transformed. (L3)
- **Gaussians are closed under the operations of Bayesian inference**: if variables are jointly Gaussian, their marginals and conditionals are Gaussian; the product of two Gaussian pdfs is proportional to a Gaussian pdf; and a linear transformation of a Gaussian is Gaussian. So if the observation distribution is Gaussian with a mean that is linear in the unknowns (and a known noise), a Gaussian prior gives a Gaussian posterior and a Gaussian PPD. Careful: the product of two Gaussian random variables, or a nonlinear transformation of one, is not Gaussian. Covariance and correlation measure only linear dependence. (L4, L6, L7, L13)

### Bayesian inference

- **Bayes' rule**: posterior ∝ likelihood × prior, normalized by the marginal likelihood. The likelihood is not a distribution over the unknowns. (L1, L4, L5)
- **The likelihood of i.i.d. data** is the product of the observation distributions at the data points. (L5, L6)
- **Predictions come from the posterior predictive distribution (PPD)**, which integrates the unknowns out against the posterior. Its variance has the noise of the data (aleatoric uncertainty) plus the uncertainty about the unknowns (epistemic uncertainty), which shrinks as $1/N$. (L5, L6, L7)
- **The prior matters with few data points, and fades away with many.** (L7)
- **A point estimate (MLE, MAP, posterior mean) replaces the posterior by a Dirac delta**: its plug-in PPD drops the epistemic uncertainty, and is overconfident with few data. MLE is MAP with a Uniform prior. (L8)
- **Credible, confidence and prediction intervals** answer different questions. (L3, L6, L7)

### Models

- **Supervised regression learns $p(\mathbf{y}|\mathbf{x})$ from data**, and when calculated by Bayesian inference also tells how confident the prediction is. (L9)
- **Linear models are linear in the parameters, not in the inputs**: the basis functions can be any fixed nonlinear transformation of the inputs. (L9)
- **Least Squares is the MLE with Gaussian noise**: it has a closed form (the normal equations), its estimate of the noise is biased low, and it is sensitive to outliers. (L9)
- **Most machine learning models are defined by the observation distribution and respective likelihood, prior distribution and the choice between calculating the point estimate or conducting Bayesian inference**: For example, Ridge regression assumed a Gaussian observation distribution, a choice of basis functions to describe the distribution parameters, a Gaussian prior on the weights, and the MAP point estimate. Lasso is the same but considering a Laplace prior. Robust regression considers a different observation distribution (e.g., leading to a Laplace or Student-$t$ likelihood). (L12)
- **The shape of the prior matters**: a Gaussian prior shrinks every weight smoothly; the cusp of a Laplace prior makes the MAP of weakly supported weights exactly zero (a sparse model), although the posterior mean is never zero. (L12)
- **Closed form or numerical optimization**: Few ML models have closed forms for their point estimates. For example, Least Squares and Ridge have closed forms (Ridge is invertible for any $\alpha > 0$), but even Lasso must be solved numerically. Most ML models need numerical optimization to find their point estimates. (L9, L12)
- **The kernel trick**: the PPD of Bayesian linear regression uses the basis functions only through the inner products $k(\mathbf{x}, \mathbf{x}') = \boldsymbol{\phi}(\mathbf{x})^T\overset{\scriptscriptstyle <}{\boldsymbol{\Sigma}}_w\boldsymbol{\phi}(\mathbf{x}')$. Replacing them by a kernel allows infinitely many basis functions (e.g. the RBF kernel), at the cost of solving an $N\times N$ linear system instead of an $M\times M$ one. (L14)
- **A prior on the weights is a prior on functions, and the kernel is its covariance**: with random weights, $\mu_{y|z}(\mathbf{x}) = \boldsymbol{\phi}(\mathbf{x})^T\mathbf{w}$ (the mean of the observation distribution) is a random function, and its values at any set of inputs are jointly Gaussian with covariance $k(\mathbf{x}, \mathbf{x}') = \boldsymbol{\phi}(\mathbf{x})^T\overset{\scriptscriptstyle <}{\boldsymbol{\Sigma}}_w\boldsymbol{\phi}(\mathbf{x}')$ (the same prior seen from the weights or from the functions). A kernel gives this covariance directly, gathering the basis functions and the prior on their weights into one object, possibly with infinitely many basis functions; with a zero mean it fully specifies the prior (a Gaussian process). Its hyperparameters are fixed numbers that set the prior variances of the weights: $s$ the amplitude of the functions, $l$ the length scale over which they change. (L14, L15)
- **Non-parametric does not mean without coefficients**: the mean prediction of a Gaussian process is a weighted sum of kernels centered at the training points, $\sum_n \alpha_n k(\mathbf{x}^*, \mathbf{x}_n)$, with one coefficient per training point, $\boldsymbol{\alpha} = (\mathbf{K}+\boldsymbol{\Lambda})^{-1}\mathbf{y}$. The model grows with the data, instead of having a fixed number of weights. (L14)
- **Not every function is a kernel**: a kernel must be positive semi-definite (Mercer's theorem). Sums and products of valid kernels are valid kernels: a sum gives the properties of either kernel, a product of both. (L14, L15)

### Generalization

- **The expected error on unseen data is noise + bias$^2$ + variance**: noise is the aleatoric uncertainty that cannot be reduced, bias results from the model misspecification (if any), and the variance term is the epistemic uncertainty that can be reduced when considering more data. (L11)
- **Underfitting and overfitting; interpolation and extrapolation.** (L11)
- **Curse of dimensionality**: Training a model with more inputs requires many more points. (L12)
- **Raw polynomial features become ill-conditioned at high degree**: a numerical failure, not only overfitting. (L11)
- **Gaussian processes are expensive** (review): fitting one solves an $N\times N$ linear system, $O(N^3)$ operations and $O(N^2)$ memory, repeated at every step of the hyperparameter optimization. They are excellent with few data, and impractical with many. (L16)

## Good practices

### 1. Frame the problem

- **Write the model down explicitly**: the observation distribution and corresponding likelihood, the prior, point estimate or Bayesian inference of the posterior, and the PPD. (L5, L9)
- **If you know nothing about an unknown, start with a Uniform prior**, then check how much the prior changes the prediction. (L5, L7)
- **What do you know about the noise?**: is the problem noiseless or noisy? Is the noise the same everywhere (homoscedastic) or does it change with the input (heteroscedastic)? Is it given, or must it be estimated? (L12, L13, L14)
- **Interpolation is much easier than extrapolation**: overparameterized models can behave well between the training points (interpolation) but poorly outside their range (extrapolation). Keep the simple picture of the divergence that occurs in Linear Regression for a high-degree polynomial (it is surprisingly descriptive even for the most sophisticated models). (L11)

### 2. Prepare the data

- **Judge a model on data it has not seen**: split into training and test sets before anything else. (L10)
- **Rescale the inputs (before building the basis) and, when useful, the outputs**, with scalers fitted on the training points only. It also puts the weights on comparable scales, which a penalty on the weights needs, and keeps the data near the zero mean that most priors assume (a Gaussian process far from its prior mean fails). (L11, L12, L16)

### 3. Choose the model

- **The model is your decision**: the likelihood, the prior and the basis functions, as well as the decision to use a point estimate or Bayesian inference.  (L10, L12)
- **Choose model structures such as basis functions, kernels or model architecture from what you know about the problem**: for example, a smooth problem benefits from differentiable basis functions or kernels. Image data benefits from convolutions or attention. History-dependent data benefits from recurrent units or attention. (L9, L14)
- **Look at draws from the prior before fitting**: do functions sampled from the prior, through the basis functions or the kernel, look like the functions you expect (smoothness, periodicity, amplitude)? (L14)
- **Many weights for few points**: get more data, or constrain the weights with a prior. (L12)
- **With outliers, change the likelihood, not the prior.** (L9, L12)

### 4. Fit the model

- **When two unknowns depend on each other** (e.g. the mean and the noise), finding the point estimate may require a staggered scheme (alternating optimization): fix one, solve for the other, and repeat. Solving for both at once can fail. (L13)

### 5. Validate and select

- **Never choose a model on its training error**: it keeps falling as the model grows, and it is biased below the noise. (L11)
- **Overfitting shows as a training error that keeps falling while the error on unseen data grows.** (L11)
- **Choose models and hyperparameters on validation data**, with cross-validation on the training set, reading the means together with their spread. Use the test set once, at the end, to report the chosen model. (L11)
- **Report error metrics on the test set, and know how to read them**: for example, in supervised regression use MSE and $R^2$ (a negative $R^2$ is worse than simply predicting the mean of the data). (L10, L11)

### 6. Quantify the uncertainty

- **Disentangling the aleatoric and epistemic uncertainties is difficult but possible**. (L8, L12, L13)
- **The Bayesian PPD is useful for data scarce cases and when you need to estimate the epistemic uncertainty**: a point estimate misses the epistemic uncertainty (we don't know what we don't know). (L8)
- **Check the uncertainty locally, not only on average**: a single noise level can be right on average and wrong everywhere, too wide where the data are clean and too narrow where they are noisy. (L13)
- **A residual at a training point underestimates the noise there**, by the epistemic variance at that point: the model was fitted to that point's own noise. (L11, L13)

### 7. Work with an AI coding agent

- **What is the same for every task goes in the instructions file** (packages, where files go, how to report); **what is specific to the task goes in the prompt** (the model, the data, what to plot). (L10)
- **Read the code the agent writes**: the documentation, and the agent itself, can explain it. (L10)
- **Check against something you know**, not against "the code runs": e.g. the normal equations, or "the training error cannot increase with the degree". (L10, L11)
- **Pin every choice that can change a number you report**, including the randomness of the split. Read the agent's **Assumptions** and **open its files**: they may contain things that your prompt should have defined. (L10)
- **The agent's own explanation, or its flag, is not a check.** The math is. (L10)

<!-- BEGIN INDEX (generated by Lectures/sync_field_guide.py: do not edit by hand) -->

## Index by lecture

- **Lecture 1**
    - An unknown is treated as a random variable (rv) *(concept: Probability)*
    - Joint, marginal and conditional distributions *(concept: Probability)*
    - Bayes' rule *(concept: Bayesian inference)*
- **Lecture 3**
    - Joint, marginal and conditional distributions *(concept: Probability)*
    - Moments *(concept: Probability)*
    - Change of variables *(concept: Probability)*
    - Credible, confidence and prediction intervals *(concept: Bayesian inference)*
- **Lecture 4**
    - Joint, marginal and conditional distributions *(concept: Probability)*
    - Gaussians are closed under the operations of Bayesian inference *(concept: Probability)*
    - Bayes' rule *(concept: Bayesian inference)*
- **Lecture 5**
    - Bayes' rule *(concept: Bayesian inference)*
    - The likelihood of i.i.d. data *(concept: Bayesian inference)*
    - Predictions come from the posterior predictive distribution (PPD) *(concept: Bayesian inference)*
    - Write the model down explicitly *(practice: Frame the problem)*
    - If you know nothing about an unknown, start with a Uniform prior *(practice: Frame the problem)*
- **Lecture 6**
    - Gaussians are closed under the operations of Bayesian inference *(concept: Probability)*
    - The likelihood of i.i.d. data *(concept: Bayesian inference)*
    - Predictions come from the posterior predictive distribution (PPD) *(concept: Bayesian inference)*
    - Credible, confidence and prediction intervals *(concept: Bayesian inference)*
- **Lecture 7**
    - Gaussians are closed under the operations of Bayesian inference *(concept: Probability)*
    - Predictions come from the posterior predictive distribution (PPD) *(concept: Bayesian inference)*
    - The prior matters with few data points, and fades away with many *(concept: Bayesian inference)*
    - Credible, confidence and prediction intervals *(concept: Bayesian inference)*
    - If you know nothing about an unknown, start with a Uniform prior *(practice: Frame the problem)*
- **Lecture 8**
    - A point estimate (MLE, MAP, posterior mean) replaces the posterior by a Dirac delta *(concept: Bayesian inference)*
    - Disentangling the aleatoric and epistemic uncertainties is difficult but possible *(practice: Quantify the uncertainty)*
    - The Bayesian PPD is useful for data scarce cases and when you need to estimate the epistemic uncertainty *(practice: Quantify the uncertainty)*
- **Lecture 9**
    - Supervised regression learns $p(\mathbf{y}|\mathbf{x})$ from data *(concept: Models)*
    - Linear models are linear in the parameters, not in the inputs *(concept: Models)*
    - Least Squares is the MLE with Gaussian noise *(concept: Models)*
    - Closed form or numerical optimization *(concept: Models)*
    - Write the model down explicitly *(practice: Frame the problem)*
    - Choose model structures such as basis functions, kernels or model architecture from what you know about the problem *(practice: Choose the model)*
    - With outliers, change the likelihood, not the prior *(practice: Choose the model)*
- **Lecture 10**
    - Judge a model on data it has not seen *(practice: Prepare the data)*
    - The model is your decision *(practice: Choose the model)*
    - Report error metrics on the test set, and know how to read them *(practice: Validate and select)*
    - What is the same for every task goes in the instructions file *(practice: Work with an AI coding agent)*
    - Read the code the agent writes *(practice: Work with an AI coding agent)*
    - Check against something you know *(practice: Work with an AI coding agent)*
    - Pin every choice that can change a number you report *(practice: Work with an AI coding agent)*
    - The agent's own explanation, or its flag, is not a check *(practice: Work with an AI coding agent)*
- **Lecture 11**
    - The expected error on unseen data is noise + bias$^2$ + variance *(concept: Generalization)*
    - Underfitting and overfitting; interpolation and extrapolation *(concept: Generalization)*
    - Raw polynomial features become ill-conditioned at high degree *(concept: Generalization)*
    - Interpolation is much easier than extrapolation *(practice: Frame the problem)*
    - Rescale the inputs (before building the basis) and, when useful, the outputs *(practice: Prepare the data)*
    - Never choose a model on its training error *(practice: Validate and select)*
    - Overfitting shows as a training error that keeps falling while the error on unseen data grows *(practice: Validate and select)*
    - Choose models and hyperparameters on validation data *(practice: Validate and select)*
    - Report error metrics on the test set, and know how to read them *(practice: Validate and select)*
    - A residual at a training point underestimates the noise there *(practice: Quantify the uncertainty)*
    - Check against something you know *(practice: Work with an AI coding agent)*
- **Lecture 12**
    - Most machine learning models are defined by the observation distribution and respective likelihood, prior distribution and the choice between calculating the point estimate or conducting Bayesian inference *(concept: Models)*
    - The shape of the prior matters *(concept: Models)*
    - Closed form or numerical optimization *(concept: Models)*
    - Curse of dimensionality *(concept: Generalization)*
    - What do you know about the noise? *(practice: Frame the problem)*
    - Rescale the inputs (before building the basis) and, when useful, the outputs *(practice: Prepare the data)*
    - The model is your decision *(practice: Choose the model)*
    - Many weights for few points *(practice: Choose the model)*
    - With outliers, change the likelihood, not the prior *(practice: Choose the model)*
    - Disentangling the aleatoric and epistemic uncertainties is difficult but possible *(practice: Quantify the uncertainty)*
- **Lecture 13**
    - Gaussians are closed under the operations of Bayesian inference *(concept: Probability)*
    - What do you know about the noise? *(practice: Frame the problem)*
    - When two unknowns depend on each other *(practice: Fit the model)*
    - Disentangling the aleatoric and epistemic uncertainties is difficult but possible *(practice: Quantify the uncertainty)*
    - Check the uncertainty locally, not only on average *(practice: Quantify the uncertainty)*
    - A residual at a training point underestimates the noise there *(practice: Quantify the uncertainty)*
- **Lecture 14**
    - The kernel trick *(concept: Models)*
    - A prior on the weights is a prior on functions, and the kernel is its covariance *(concept: Models)*
    - Non-parametric does not mean without coefficients *(concept: Models)*
    - Not every function is a kernel *(concept: Models)*
    - What do you know about the noise? *(practice: Frame the problem)*
    - Choose model structures such as basis functions, kernels or model architecture from what you know about the problem *(practice: Choose the model)*
    - Look at draws from the prior before fitting *(practice: Choose the model)*
- **Lecture 15**
    - A prior on the weights is a prior on functions, and the kernel is its covariance *(concept: Models)*
    - Not every function is a kernel *(concept: Models)*
- **Lecture 16**
    - Gaussian processes are expensive *(concept: Generalization)*
    - Rescale the inputs (before building the basis) and, when useful, the outputs *(practice: Prepare the data)*
- **Lecture 22**
    - An unknown is treated as a random variable (rv) *(concept: Probability)*

<!-- END INDEX -->
