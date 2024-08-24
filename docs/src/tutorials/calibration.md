

``` @meta
CurrentModule = LaplaceRedux
```

# Uncertainty Calibration

## The issue of calibrated uncertainty distributions

Bayesian methods offer a general framework for quantifying uncertainty. However, due to model misspecification and the use of approximate inference techniques,uncertainty estimates are often inaccurate: for example, a 90% credible interval may not contain the true outcome 90% of the time, in such cases the model is said to be miscalibrated. This problem arises because of model bias: a predictor may not be sufficiently expressive to assign the right probability to every credible interval, just as it may not be able to always assign the right label to a datapoint. Miscalibrated credible intervals reduce the trustworthyness of the forecaster because they lead to a false sense of precision and either overconfidence or underconfidence in the results.

A forecaster is said to be perfectly calibrated if a 90% credible interval contains the true outcome approximately 90% of the time. Perfect calibration however cannot be achieved with limited data, because with limited data comes inherent statistical fluctuations that can cause the estimated credible intervals to deviate from the ideal coverage probability. Furthermore, a finite sample of collected data points cannot eliminate completely the influence of the possible misjudged priors probabilities. On top of these issues, which stem directly from Bayes’ theorem, with Bayesian neural network there is also the problems introduced by the approximate inference method adopted to compute the posterior distribution of the weights. To introduce the concept of Average Calibration and the Calibration Plots, we will follow closely the paper [Accurate Uncertainties for Deep Learning Using Calibrated Regression](https://arxiv.org/abs/1807.00263), written by Volodymyr Kuleshov, Nathan Fenner and Stefano Ermon.

## Notation

We are given a labeled dataset $x_t, y_t \in X \times Y$ for $t = 1, 2, ..., T$ of i.i.d. realizations of random variables $X, Y \sim P$, where $P$ is the data distribution.
Given $x_t$, a forecaster $H : X \rightarrow (Y \rightarrow [0, 1])$ outputs a probability distribution $F_t(y)$ targeting the label $y_t$. When $Y$ is continuous, $F_t$ is a cumulative probability distribution (CDF). We will use $F^{−1}_t: [0, 1] → Y$ to denote the quantile function $F^{−1}_t (p) = inf\{y : p ≤ F_t(y)\}$.

\## Calibration in the Regression case
In the regression case, we say that the forecaster H is on average calibrated if
$$ \frac{\sum_{t=1}^T \mathbb{1} \{ y_t \leq F_t^{-1}(p)  \} }{T} \rightarrow p \quad \text{for all}\quad  p \in [0,1]$$

as $T \rightarrow \infty$. In other words, the empirical and the predicted CDFs should match as the dataset size goes to infinity.
Perfect Calibration is a sufficient condition for average calibration, the opposite however is not necessarily true.

### Sharpness

Average calibration by itself is not sufficient to produce a useful forecast. For example, it is easy to see that if we use for the forecast the marginal distribution $F(y) = \mathbb{P}(Y ≤ y)$, without considering the input feature $X$, the forecast will be calibrated but still not accurate. In order to be useful, forecasts must also be sharp, which,in a regression context, means that the confidence intervals should all be as tight as possible around a single value. ore formally, we want the variance $var(F_t)$ of the random variable whose CDF is $F_t$ to be small.

### Calibration Plots

\## Calibration in the Classification case

### Sharpness

## Calibration Plots

yadda yadda
