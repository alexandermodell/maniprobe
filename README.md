# maniprobe

This package implements the Manifold Probe introduced in the paper
[*Probing for Representation Manifolds in Superposition*](https://arxiv.org/abs/2605.18537)
(Modell, 2026). The Manifold Probe is a supervised method for discovering representation
manifolds in superposition. It generalizes linear regression probes by learning the
space of features of a concept which can be linearly predicted from the representations,
and then learning the directions used to encode them.

Given a probing dataset of representations $x_1, \ldots, x_n \in \mathbb R^p$ and
corresponding concept values $z_1, \ldots, z_n \in \mathcal Z$, the probe is fitted in
two stages. The first stage learns a sequence of mean-zero, orthonormal features
$\hat f_1, \ldots, \hat f_d$ of the concept values, each of which is well predicted by a
linear function of the representations. For $k = 1, \ldots, d$, the feature $\hat f_k$
and its linear prediction $g_k(x) = \hat w_k^\top x + \hat b_k$ solve

$$
\underset{f \in \mathcal H,\; w \in \mathbb R^p,\; b \in \mathbb R}{\mathrm{minimize}}
\quad \sum_{i=1}^n \left( f(z_i) - w^\top x_i - b \right)^2 + \lambda_w \lVert w \rVert_2^2 + \lambda_f J(f)
$$

$$
\text{subject to} \quad \sum_{i=1}^n f(z_i) = 0, \qquad \frac{1}{n} \sum_{i=1}^n \left[ f(z_i) \right]^2 = 1, \qquad f \perp \hat f_{k-1}, \ldots, \hat f_1,
$$

where $\mathcal H$ is a space of smooth functions and $J$ is a penalty on their
roughness, both of which are described below. The scale constraint rules out the trivial
solution $f = 0$, $w = 0$, and the orthogonality constraint ensures that each feature is
new. The second stage learns the direction used to encode each feature by linear
regression of the representations on it, which gives

$$
\hat u_k = \frac{1}{n} \sum_{i=1}^n \hat f_k(z_i) \left( x_i - \bar x \right).
$$

Together, the features and directions define two maps:

$$
\hat \phi(z) = \hat u_1 \hat f_1(z) + \cdots + \hat u_d \hat f_d(z), \qquad
\Psi(x) = \hat u_1 g_1(x) + \cdots + \hat u_d g_d(x).
$$

The first, $\hat \phi : \mathcal Z \to \mathbb R^p$, maps the concept values to a
$d$-dimensional representation manifold $\hat{\mathcal M} = \hat \phi(\mathcal Z)$. The
second, $\Psi : \mathbb R^p \to \mathbb R^p$, is a linear (affine) map which estimates
where a representation $x$ lies on the manifold from $x$ alone, without access to its
concept value.

The package is organized around two classes. `Probe` fits the sequence of features, and
`Manifold` holds the maps they define. Since the features are only defined up to a
change of basis, `Manifold` also provides tools to reorder, rescale and rotate them (for
example, using a Varimax rotation), none of which change the maps themselves.

## Parametrizing the features

In order to fit a feature $f$, we parametrize it in a basis $h_1, \ldots, h_m$, so that

$$
f(z) = \beta^\top h(z) \equiv \beta_1 h_1(z) + \cdots + \beta_m h_m(z)
$$

for some coefficients $\beta = (\beta_1, \ldots, \beta_m)$, and $\mathcal H$ is the space
of functions of this form. The standard approach is to choose an overly flexible basis,
and then to control the complexity of $f$ with a quadratic penalty
$J(f) = \beta^\top S \beta$, whose weight $\lambda_f$ is chosen from the data. For the
spline bases below, $J$ is the integrated squared derivative of a chosen order, and the
default is the second derivative,

$$
J(f) = \int_{\mathcal Z} \left[ f''(z) \right]^2 \, \mathrm d z.
$$

The basis must be centered, which is how the constraint $\sum_i f(z_i) = 0$ is imposed.
Without it, the constant function $f = 1$ would satisfy the scale constraint, be matched
exactly by the intercept $b$, and have $J(f) = 0$, so the objective would be zero at a
fit which says nothing. Centering removes one dimension from the basis, and each
orthogonality constraint removes another, so a basis of $m$ functions can carry at most
$m - 1$ features.

The package provides two helpers, `bs` and `tps`, which between them construct three
kinds of basis. Both return a centered basis with a penalty, ready to be passed to a
`Probe`. The concept values $z$ are passed to the basis unchanged, as an array of shape
`(n,)` for a one-dimensional concept or `(n, d)` for a $d$-dimensional one.

**B-splines**, for a one-dimensional concept such as time. By default, `bs` returns
cubic B-splines with a second-derivative penalty:

```python
from maniprobe import bs

bs(k=20)                              # 20 cubic B-splines
bs(k=20, degree=2, penalty=1)         # quadratic, with a first-derivative penalty
bs(k=20, knots="uniform", limits=(1950, 2020))
```

The knots are placed at quantiles of the data by default, or uniformly, or can be given
explicitly as a knot vector. The boundary of $\mathcal Z$ is taken from the data unless
`limits` is given.

**Tensor products of B-splines**, for a concept on a rectangle such as latitude and
longitude. Passing a tuple to `bs` gives one axis per entry, and every other argument
can be given per axis:

```python
bs(k=(40, 80))                        # 40 x 80 functions on two axes
bs(k=(20, 10), penalty=(2, 1))        # a different penalty on each axis
```

Each axis is penalized in its own units, so the axes need not be on the same scale.

**Thin plate splines**, for an isotropic concept in any number of dimensions, such as a
position in space. The basis is knot-free, and its penalty is invariant to rotations of
$\mathcal Z$:

```python
from maniprobe import tps

tps(k=50, d=2)                        # a thin plate spline on a plane
tps(k=20)                             # on a line, the cubic smoothing spline
```

The penalty is of order $m = 2$ by default, which gives the cubic smoothing spline when
$d = 1$ and the classical thin plate spline when $d = 2$. For $d \geq 4$ the order must
be raised, since a thin plate spline requires $2m > d$.

The basis classes themselves are available from `maniprobe.basis`, but they are neither
centered nor penalized by default, so we recommend using the helpers.

## Selecting the regularization parameters

Each feature has two regularization parameters, $\lambda_f$ on the roughness of $f$ and
$\lambda_w$ on the ridge penalty for $w$, which are selected from the data separately
for every feature. The package provides two approaches, chosen with the `criterion`
argument of `Probe`.

**Alternating least squares.** The objective can be minimized by alternating between
two ridge regressions, one for $w$ and one for $\beta$, which converges to the global
minimizer under mild conditions. This allows $\lambda_w$ and $\lambda_f$ to be
reselected at every iteration using a closed-form criterion for linear smoothers, which
is cheap. The available
criteria are

| criterion | |
|---|---|
| `gcv()` | Generalized Cross-Validation (the default) |
| `reml()` | Restricted Maximum Likelihood |
| `aic()` | Akaike Information Criterion |
| `bic()` | Bayesian Information Criterion |

The settings of the alternation itself can be changed by wrapping a criterion in `als`,
as in `als(reml(), tol=1e-4, max_iter=200, accel=3)`. Passing `reml()` is the same as
passing `als(reml())`.

**Cross-validation.** Alternatively, the pair $(\lambda_w, \lambda_f)$ can be chosen by
scoring the joint objective on held-out data. The available criteria are

| criterion | |
|---|---|
| `kf(k=5)` | $k$-fold cross-validation |
| `ho(train=0.8)` | a single train/validation split |

Since there is no closed form for the cross-validated score as a function of
$(\lambda_w, \lambda_f)$, it is maximized by searching over candidate values, using one
of two search methods:

| search method | |
|---|---|
| `bayes(n_eval=40)` | Bayesian optimization with `n_eval` evaluations (the default) |
| `grid(grid_size=15)` | a `grid_size` $\times$ `grid_size` grid |

The search method is passed to the criterion, as in
`kf(k=5, method=grid(15), seed=0)`, where `seed` fixes both the split into folds and the
random draws of the Bayesian search, so that a fit can be reproduced.

> [!WARNING]
> Cross-validation is considerably more expensive than alternating least squares: every
> evaluation of the search refits the probe on every fold, and the folds are built when
> the `Probe` is constructed rather than when it is fitted. For general use, we
> recommend alternating least squares.

## Fitting the probe

A probe is constructed from the representations `X`, an array of shape `(n, p)`, the
concept values `z`, and a basis. Features are then fitted with `fit`:

```python
import numpy as np
from maniprobe import Probe, bs

# 30-dimensional representations of a one-dimensional concept, built from two features
rng = np.random.default_rng(0)
z = rng.uniform(0, 1, size=400)
F = np.column_stack([np.sin(2 * np.pi * z), np.cos(4 * np.pi * z)])
X = F @ rng.normal(size=(2, 30)) + 0.3 * rng.normal(size=(400, 30))

probe = Probe(X, z, bs(k=15), verbose=True)
probe.fit(4)
probe.inspect()
```

```
Components fitted: 4
  Component 0   train R²  0.994113
  Component 1   train R²  0.994539
  Component 2   train R²  0.000044
  Component 3   train R²  0.000033
```

For each feature, `inspect` reports the $R^2$ coefficient of its linear prediction,

$$
R_k^2 = 1 - \frac{\sum_i \left( \hat f_k(z_i) - g_k(x_i) \right)^2}{\sum_i \left( \hat f_k(z_i) - \bar f_k \right)^2},
$$

which measures the extent to which the feature is linearly represented. Here two
features are linearly represented and two are not, which is correct, since the data were
constructed from two features.

The argument to `fit` is the total number of features rather than the number to add, so
`probe.fit(6, continue_fit=True)` fits two more, and a total at or below the number
already fitted does nothing. Without `continue_fit=True`, `fit` starts again from the
first feature.

The $R^2$ coefficients on the training data are optimistic, and the number of features
to keep is better decided on held-out data. A test set can be passed to `Probe`, which
does not enter any fit, and its $R^2$ coefficients are reported alongside:

```python
from maniprobe import reml

probe = Probe(X, z, bs(k=15), criterion=reml(), test_set=(X_test, z_test))
```

## Working with the manifold

The fitted features define a `Manifold`, constructed from either the first few features,
or a chosen subset of them:

```python
manifold = probe.manifold(n_components=2)
manifold = probe.manifold(component_idx=[0, 1])
```

Its four methods evaluate the features, their linear predictions, and the two maps. Each
defaults to the training data when called without an argument:

```python
manifold.f(z_new)        # (n_new, d)   the features f_k(z)
manifold.g(X_new)        # (n_new, d)   their linear predictions g_k(x)
manifold.embed(z_new)    # (n_new, p)   the manifold, phi(z)
manifold.project(X_new)  # (n_new, p)   the readout, Psi(x)
```

Both $\hat \phi$ and $\Psi$ map into the centered representation space, so $\bar x$ is
not added back.

The features are produced in the order in which they were fitted, which is not
necessarily the order of how well they are represented, and they are only defined up to
a change of basis of the directions $\hat u_1, \ldots, \hat u_d$. `Manifold` provides
methods to choose that basis:

```python
manifold.reorder(by="test_r2")  # order by R^2 coefficient, "r2" or "test_r2"
manifold.rescale(unit="u")      # unit-norm directions, or unit="f" for unit-norm features
manifold.rotate()               # Varimax rotation of the features
manifold.reset()                # return to the basis as fitted
manifold.inspect()              # the R^2 coefficients in the current basis
```

These operations compose, and each returns the `Manifold`, so they can be chained. None
of them change $\hat \phi$ or $\Psi$. Each is a change of basis by a $d \times d$ matrix
$T$, applied to the features and the directions together,

$$
\left[ \hat f_1, \ldots, \hat f_d \right] \longmapsto \left[ \hat f_1, \ldots, \hat f_d \right] T,
\qquad
\left[ \hat u_1, \ldots, \hat u_d \right] \longmapsto \left[ \hat u_1, \ldots, \hat u_d \right] T^{-\top},
$$

with the $\hat w_k$ and $\hat b_k$ transformed in the same way as the features, so that
$T$ and $T^{-\top}$ cancel in both maps.

## Installing

The package can be installed with

```
pip install git+https://github.com/alexandermodell/maniprobe.git
```

or, from a clone of the repository, with `pip install -e .`. It requires only `numpy`
and `scipy`.

## Understanding the codebase

The vignette [vignettes/the_manifold_probe.ipynb](vignettes/the_manifold_probe.ipynb)
demonstrates the package on a synthetic probing dataset in which a known
three-dimensional manifold is represented in superposition with other semantics. It
covers how many features to keep, what the fit does and does not identify, a comparison
with the truth after Procrustes alignment and how its error decreases with $n$, what
$\hat \phi$ and $\Psi$ look like, how to change the basis of the features, a
two-dimensional concept, and a comparison of the criteria for selecting the
regularization parameters. It is saved with its outputs, so it can be read without
running it. To run it, install the additional dependencies with `pip install -e ".[vignettes]"`.

The package is organized as follows.

| module | contents |
|---|---|
| [probe.py](maniprobe/probe.py) | `Probe`, the sequence of fitted features |
| [manifold.py](maniprobe/manifold.py) | `Manifold`, the two maps and the basis they are written in |
| [bilinear.py](maniprobe/bilinear.py) | a single feature, fitted in closed form by one eigendecomposition |
| [als.py](maniprobe/als.py) | the same fit by alternating least squares, reselecting $\lambda_w$ and $\lambda_f$ at each iteration |
| [cv.py](maniprobe/cv.py) | cross-validation of the joint objective |
| [search.py](maniprobe/search.py) | grid search and Bayesian optimization over $\lambda_w$ and $\lambda_f$ |
| [linear_model.py](maniprobe/linear_model.py) | generalized ridge regression, with GCV, REML, AIC, BIC and linear constraints |
| [linear_smoother.py](maniprobe/linear_smoother.py) | the same, with a basis in front of it |
| [basis/](maniprobe/basis/) | B-splines, tensor products and thin plate splines |

The names used above are imported from `maniprobe` directly, and everything else is
imported from the module which defines it. The modules below `probe.py` do not depend on
the probe, so that `LinearModel`, for example, can be used as a penalized regression in
its own right. Each module begins with a docstring describing what it implements and why.

The tests are in [tests/](tests/), and can be run with `pytest` from the root of the
repository after `pip install -e ".[test]"`. They check results against dense, independent
reference implementations rather than against the package's own arithmetic.
