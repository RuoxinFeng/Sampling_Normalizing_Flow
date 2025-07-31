# Advanced residual distributions

## Truncated distribution

First we want to take a look at the truncated distribution. For this we define a lower bound $a \in \mathbb{R} \cup \{-\infty\}$ and an upper bound $b \in \mathbb{R} \cup \{\infty\}$ with $a < b$. Now we want to define the probability density function (pdf), cumulative distribution function (cdf) and the percent point function (ppf), which is the inverse of the cdf.

Let $X$ be a random variable with density $f : \mathbb{R} \rightarrow [0,\infty)$.

### pdf

We get
$$f(x | a \leq X < b) = \frac{g(x)}{F(b) - F(a)} = \frac{f(x) \cdot I(\{a \leq x < b\})}{F(b) - F(a)},$$
where $g(x) = f(x)$ for all $a\leq x < b$ and $g(x) = 0$ everywhere else. This is in fact a density since
$$\int_a^b f(x | a\leq X < b) \mathrm{d} x = \frac{1}{F(b) - F(a)} \int_a^b g(x) \mathrm{d} x = 1.$$

### cdf

To get the cdf we compute
$$\int_{-\infty}^{x} \frac{g(t)}{F(b) - F(a)} \mathrm{d} t = \int_{a}^{x} \frac{g(t)}{F(b) - F(a)} \mathrm{d} t = \frac{F(x) - F(a)}{F(b) - F(a)}.$$
Then the cdf of the truncated distribution is defined as

$$
F_t(x) = \begin{cases}
	0 & x < a\\
	\frac{F(x) - F(a)}{F(b) - F(a)} & x \in [a, b)\\
	1 & x \geq b
\end{cases}
$$

### ppf

To get the ppf we have to invert $F_t$. Let $x \in [a, b)$.

$$
\begin{align*}
	q &= F_t(x) = \frac{F(x) - F(a)}{F(b) - F(a)}\\
	\Leftrightarrow F(x) &= q \cdot (F(b) - F(a)) + F(a)\\
	\Leftrightarrow x &= F^{-1}(q \cdot (F(b) - F(a)) + F(a))
\end{align*}
$$

## Piecewise distribution

Now we want to look at the piecewise distribution. For that we define a threshold $t \in \mathbb{R}$ and a lower and upper distribution with pdf and cdf $f_l$, $F_l$ and $f_u$, $F_u$ respectively. These distributions are truncated at $t$. Also we define a probability mass for the lower tail $m \in (0,1)$ for which $m = F(t)$, where $F$ is the cdf of the piecewise distribution, should hold.

Like for the truncated distribution we want to determine the pdf, cdf and ppf.

### pdf

The pdf $f$ is simply defined as
$$f(x) = m \cdot f_l(x) + (1 - m) \cdot f_u(x)$$
This is in fact a density since
$$\int_{-\infty}^\infty f(x) \mathrm{d} x = m\int_{-\infty}^t f_l(x) \mathrm{d} x + (1-m)\int_t^\infty f_u(x) \mathrm{d} x = m + (1-m) = 1.$$

### cdf

Likewise the cdf $F$ is defined by

$$
\begin{align*}
	F(x) &= \int_{-\infty}^x f(z) \mathrm{d} z \\
	&= m\int_{-\infty}^x f_l(z)\mathrm{d} z + (1-m) \int_{-\infty}^x f_u(z) \mathrm{d} z \\
	&= m \cdot F_l(x) + (1-m) \cdot F_u(x).
\end{align*}
$$

### ppf

To get the ppf we have to invert $F$ again. Let $x \in \mathbb{R}$. Then we get
$$q = F(x) = m \cdot F_l(x) + (1-m) \cdot F_u(x).$$
We have two cases.

Case 1, $x \in (-\infty, t)$: We now can easily calculate the ppf.

$$
\begin{align*}
q &= m \cdot F_l(x) + (1-m) \cdot F_u(x)\\
\Leftrightarrow q &= m \cdot F_l(x) + (1-m) \cdot 0\\
\Leftrightarrow x &= F_l^{-1}\left(\frac{q}{m}\right)
\end{align*}
$$

with $q \in (0, m)$.

Case 2, $x \in [t, \infty)$:

$$
\begin{align*}
q &= m \cdot F_l(x) + (1-m) \cdot F_u(x)\\
\Leftrightarrow q &= m \cdot 1 + (1-m) \cdot F_u(x)\\
\Leftrightarrow x &= F_u^{-1}\left(\frac{q-m}{1-m}\right)
\end{align*}
$$

with $q \in [m, 1)$.

Therefore the ppf is defined as

$$
F^{-1}(q) = \begin{cases}
F_l^{-1}\left(\frac{q}{m}\right) & q \in (0, m)\\
F_u^{-1}\left(\frac{q-m}{1-m}\right) & q \in [m, 1)
\end{cases}
$$

# Backward stepwise selection

Algorithm:

1. Let $\mathcal{M}_p$ denote the full model, which contains all $p$ predictors.
2. For $k = p, p-1, ..., 1$:
    1. Consider all $k$ models that contain all but one of the predictors in $\mathcal{M}_k$, for a total of $k-1$ predictors. Note that we only remove predictors, if they are not hierarchically dependent on other predictors.
    2. Choose the best among these $k$ models and call it $\mathcal{M}_{k-1}$ by checking for the highest $R^2$.
3. Select a single best model from among $\mathcal{M}_0, ..., \mathcal{M}_p$ using the cross-validation method, i.e. we want to minimize the following formula.

$$
CV_{(n)} = \frac{1}{n} \sum_{i=1}^n \left(\frac{y_i - \hat y_i}{1 - h_i}\right)^2
$$

Here $n$ is the number of samples, $\hat y_i$ is the $i$-th fitted value and $h_i$ is the leverage, which is defined for multiple linear regression as the diagonal elements of the ortho-projection matrix $H = X(X^TX)^{-1}X^T$, where $X$ is the design matrix of the regression model.

**Remarks:**
1. Backward stepwise selection requires that the number of samples $n$ is larger than the number of variables $p$.
2. The matrix inversion $(X^TX)^{-1}$ would not work if $X$ does not have full column rank; meaning that there is a set of linearly dependent predictors. A check is introduced in order to circumvent this problem, which calculates the null space of X. The null space of X consists of all vectors $v$, such that $Xv = 0$. Suppose $X$ is of size $m \times n$, we calculate a matrix $V$ denoting orthonormal basis of the null space which will be of size $n \times k$, where $k$ is the number of linear relations among the predictors. Using the columns of $V$, we print $k$ linear combinations of columns of $X$. The user can then pick the predictors from the printed linear combinations to remove in order to create a design matrix $X$ with linearly independent predictors.
The Null space of a matrix with full column rank will be empty.

For further information see:

-   G. James, "An Introduction to Statistical Learning with Applications in Python", 2023, Springer (for backward stepwise selection)
-   D. C. Montgomery, "Introduction to Linear Regression Analysis", 2012, Wiley (for leverage for multiple linear regression)
