

``` @meta
CurrentModule = LaplaceRedux
```

# Uncertainty Calibration

# The issue of calibrated uncertainty distributions

Bayesian methods offer a general framework for quantifying uncertainty. However, due to model misspecification and the use of approximate inference techniques,uncertainty estimates are often inaccurate: for example, a 90% credible interval may not contain the true outcome 90% of the time, in such cases the model is said to be miscalibrated.

This problem arises due to the limitations of the model itself: a predictor may not be sufficiently expressive to assign the right probability to every credible interval, just as it may not be able to always assign the right label to a datapoint. Miscalibrated credible intervals reduce the trustworthyness of the forecaster because they lead to a false sense of precision and either overconfidence or underconfidence in the results.

A forecaster is said to be perfectly calibrated if a 90% credible interval contains the true outcome approximately 90% of the time. Perfect calibration however cannot be achieved with limited data, because with limited data comes inherent statistical fluctuations that can cause the estimated credible intervals to deviate from the ideal coverage probability. Furthermore, a finite sample of collected data points cannot eliminate completely the influence of the possible misjudged priors probabilities.

On top of these issues, which stem directly from Bayes’ theorem, with Bayesian neural network there is also the problems introduced by the approximate inference method adopted to compute the posterior distribution of the weights.  To introduce the concept of Average Calibration and the Calibration Plots, we will follow closely the paper [Accurate Uncertainties for Deep Learning Using Calibrated Regression](https://arxiv.org/abs/1807.00263), written by Volodymyr Kuleshov, Nathan Fenner and Stefano Ermon, although with some small differences. We will hightlight these differences in the following paragraphs whenever they appear. We present here the theoretical basis necessary to understand the issue of calibration and we refer to the tutorials for the coding examples.

## Notation

We are given a labeled dataset $x_t, y_t \in X \times Y$ for $t = 1, 2, ..., T$ of i.i.d. realizations of random variables $X, Y \sim P$, where $P$ is the data distribution.
Given $x_t$, a forecaster $H : X \rightarrow (Y \rightarrow [0, 1])$ outputs at each step $t$ a CDF $F_t(y)$ targeting the label $y_t$. When $Y$ is continuous, $F_t$ is a cumulative probability distribution (CDF). We will use $F^{−1}_t: [0, 1] → Y$ to denote the quantile function $F^{−1}_t (p) = inf\{y : p ≤ F_t(y)\}$.

## Calibration in the Regression case

In the regression case, we say that the forecaster H is (on average) calibrated if

$$\frac{\sum_{t=1}^T \mathbb{1} \{ y_t \leq F_t^{-1}(p)  \} }{T} \rightarrow p \quad \text{for all}\quad  p \in [0,1]$$

as $T \rightarrow \infty$.  In other words, the empirical and the predicted CDFs should match as the dataset size goes to infinity.
Perfect Calibration is a sufficient condition for average calibration, the opposite however is not necessarily true: a model can be average calibrated but not perfectly calibrated.  
From now on when we talk about calibration we will implicitly talk about average calibration rather than perfect calibration.

### Sharpness

Calibration by itself is not sufficient to produce a useful forecast. For example, it is easy to see that if we use for the forecast the marginal distribution $F(y) = \mathbb{P}(Y ≤ y)$, without considering the input feature $X$, the forecast will be calibrated but still not accurate. In order to be useful, forecasts must also be sharp, which,in a regression context, means that the confidence intervals should all be as tight as possible around a single value. More formally, we want the variance $var(F_t)$ of the random variable whose CDF is $F_t$ to be small.  
As a sharpness score of the forecaster, Kuleshov et al. proposed the average predicted variance

$$sharpness(F_{1} ,\dots, F_T) = \frac{1}{T} \sum_{t=1}^T var{F_t}$$  
the smaller the sharpness, the tighter will be the confidence intervals on average.

### Calibration Plots

To check the level of calibration, Kuleshov et al. proposed a calibration plot that display the true frequency of points in each confidence interval relative to the predicted fraction of points in that interval.  
More formally, we choose $m$ confidence levels $0 ≤ p_1 < p_2 < . . . < p_m ≤ 1$; for each threshold $p_j$ , and compute the empirical frequency

$$\hat{p}_j = \frac{|\{ y_t|F_t(y_t) \leq p_t, t= 1,2,\dots,T    \} |}{T}.$$

To visualize the level of average calibration, we plot $\{(p_j,\hat{p_j}) \}_{j=1}^M$; A forecaster that is calibrated will correspond a straight line on the plot that goes from $\{0,0\}$ to $\{1,1\}$ .  
As a measure of the level of miscalibration of the forecaster, differently from what it is suggested in the original paper, we a measure the area between the bisector of the plot and the line produced by the forecaster.

### Post-training calibration

As we have said previously, uncertainty estimates obtained by deep BNNs tends to be miscalibrated. We introduced the support to a post-training technique for regression problems presented in [Recalibration of Aleatoric and Epistemic Regression Uncertainty in Medical Imaging](https://arxiv.org/abs/2104.12376)
by Max-Heinrich Laves, Sontje Ihler, Jacob F. Fast, Lüder A. Kahrs and Tobias Ortmaier and usually referred to as sigma-scaling. Using a Gaussian model, the technique consist in scaling the predicted standard deviation $\sigma$ with a scalar value $s$ to recalibrate the probability density function

$$p(y|x; \hat{y}(x), \hat{σ}^2(x)) = \mathbb{N}( y; \hat{y}(x),(s  \cdot  \hat{σ}(x))^2 ).$$

This results in the following minimization objective:

$$L_G(s) = m \log(s) + \frac{1}{2}s^{−2} \sum_{i=1}^m (\hat{σ}^{(i)}_θ)^{−2} || y^{(i)} − \hat{y}_{\theta}^{(i)}||^2.$$

In general, this equation can be optimized respect to s with fixed values for the parameters $\theta$ using gradient descent in a second phase over a separate calibration set. However, for the case of a gaussian distribution, the analytical solution is known and takes the closed form

$$s = \pm \sqrt{\frac{1}{m} \sum_{i=1}^m (\hat{σ}^{(i)}_θ)^{−2} || y^{(i)} − \hat{y}_{\theta}^{(i)}||^2}.$$

Once the scalar $s$ is computed, all we have to do to obtain better calibrated predictions is to multiply the predicted standard deviation with the scalar.

## Calibration in the Binary Classification case

In binary classification, we have $Y = {0, 1}$, and we say that H is calibrated if

$$\frac{\sum_{t=1}^{T} y_t\mathbb{1}\{H(x_t)=p\}}{\sum_{t=1}^{T}\mathbb{1}\{H(x_t)=p\}} \rightarrow p \quad \text{for all} \quad p\in [0,1]$$

as $T \rightarrow \infty$. For simplicity, we have denoted $H(x_t)$ as the probability of the event $y_t=1$. Once again, perfect calibration is a sufficient condition for calibration.

### Sharpness

We can assess sharpness by looking at the distribution of model predictions. When forecasts are sharp, most
predicted probabilities for the correct class are close to $1$; unsharp forecasters make predictions closer to $0.5$.

## Calibration Plots

Given a dataset ${(x_t, y_t)}^T_t=1$, let $p_t = H(x_t) ∈ [0, 1]$ be the forecasted probability. We group the $p_t$ into intervals $I_j$ for
$j = 1, 2, ..., m$ that form a partition of $[0, 1]$.  
A calibration curve plots the predicted average \$ p_j = T^{−1}*j *{t:p_t∈ I_j} p_t \$ in each interval $I_j$ against the observed empirical average
$$p_j = T^{−1}_j \sum_{t:p_t ∈ I_j}y_t,$$ where $T_j = |{t : p_t ∈ I_j}|$. Perfect calibration corresponds once again to a straight line.

### multiclass case

For multiclass classification tasks the above technique can be extended by plotting each class versus all the remaining classes considered as one.
