"""Maximising a scalar function of two variables over a box.

`GridSearch` and `BayesSearch` are handed a callable and a box and know nothing else:
no model, no `lmbda`, no notion of what a good answer looks like. Cross-validating a
pair of penalised models is a call site,

    bounds = np.log([lm1.lmbda_grid(2, edf_tol), lm2.lmbda_grid(2, edf_tol)])
    s = BayesSearch(lambda v: cv.score(*np.exp(v)), bounds).search(n_eval=40)
    lmbda1, lmbda2 = np.exp(s.points[s.scores.argmax()])

and both transformations in it belong to the caller. The log is a statement about
`lmbda` -- the useful range spans decades and a criterion moves smoothly in the
logarithm, neither of which is true of a general objective -- and the box comes from
`LinearModel.lmbda_grid`, whose ends each give up exactly `edf_tol` degrees of freedom,
so an optimum on the boundary is an answer rather than a truncation.

Which one
---------

The grid costs `grid_size**2` evaluations and reports the whole surface; the Bayesian
search costs `n_eval` and reports the path it took to one point. At their defaults they
are equally good answers -- against a 121x121 reference on the eight cross-validation
surfaces described below, the grid's worst regret is 0.030 of a fold standard error and
the Bayesian search's 0.041, which is no difference at all -- so what separates them is
225 evaluations against 40, and what the grid gives back for them.

Prefer the grid wherever `grid_size**2` evaluations are affordable: it is
deterministic, has no seed and nothing to fit, and a flat surface -- which a
cross-validation score routinely is -- is better read than optimised. What decides
affordability is the objective, not this module: a five-fold score of a pair at
`r = 2000`, `n = 50000` is 2.6 seconds, so a 15x15 grid is ten minutes against under
two for 40 Bayesian evaluations, while at a few milliseconds an evaluation the
comparison inverts and the grid is both cheaper and more informative. The surrogate's
own arithmetic is about a millisecond an iteration, or 0.04 seconds over a default run,
and never decides this.

Stating one before there is an objective
----------------------------------------

`grid(grid_size=20)` and `bayes(n_eval=60)` are the same choice made early: a budget and
nothing else, with `spec.search(func, bounds, seed)` the place the budget and the
problem meet. A caller whose objective is expensive to build -- `cv.EigenCVFit`, whose
folds are the whole of its construction cost -- can then settle what the search costs
before building it, and a budget belonging to the other search is a `TypeError` from
`bayes(grid_size=20)` rather than from a `search()` call much later. The two classes
differ in their constructors and in their budget arguments, so a caller holding one
string and a bag of keywords could have it checked at neither end.

`as_spec` takes `"grid"` and `"bayes"` as shorthand for the two at their default
budgets, which is what a caller with no view on the budget writes. It is the same
bargain `linear_model`'s `gcv()` makes against `als(gcv())`: two spellings of one thing,
paid for because the short one is what gets written and the long one is the one that
says what it is.

Why this is written here
------------------------

Nothing outside numpy and scipy is imported by this project, and no available library
survives that. `skopt` is effectively unmaintained and pulls in scikit-learn; `optuna`'s
default sampler is a tree-structured estimator built for high-dimensional and
categorical spaces rather than for two smooth continuous ones, and its GP sampler needs
torch; `botorch` is a research framework for batch acquisition and Monte Carlo
acquisition functions; `bayes_opt` pulls scikit-learn too. After deleting high
dimensions, noise estimation, categorical variables and batching -- none of which
applies -- what is left is the hundred lines below.

The surrogate
-------------

A Gaussian process with a Matern 5/2 correlation, on the box rescaled to the unit
square. Matern rather than the usual squared exponential because the objective here is
not analytic: `EigenFit.fit` takes the *smallest* eigenpair, so where the two
smallest eigenvalues approach each other the fitted direction rotates quickly and the
surface has a sharp feature. A squared exponential asserts infinitely differentiable
sample paths, which is a strong claim to make about a surface with ridges in it, and it
pays for the claim by smoothing them away and then being surprised.

One length-scale for both axes, chosen by maximising the log marginal likelihood over
`ELL`. Isotropy is what the box earns: both ends are derived from `edf_tol`, so a step
in either normalised coordinate is the same amount of fit given up, and the axes are
already commensurate. `ELL`'s ends span the whole meaningful range -- below 0.05 no two
points in a unit square are correlated and the posterior mean is flat between them,
above 2.0 the correlation across the box is so near one that the surface is a plane --
so the length-scale is chosen by the data rather than tuned, and a maximiser landing on
an end is an answer in the same sense the box's ends are. The amplitude is removed by
standardising the scores rather than fitted, and the prior mean is zero for the same
reason.

Why the noise is not fitted
---------------------------

`JITTER` is conditioning and not a noise term. The objective is deterministic --
`KFoldCV.score` repeats to the bit, since each fold is refitted from a cache fixed at
construction -- so there is no observation noise to estimate, and the only reason the
correlation matrix needs anything on its diagonal is that expected improvement drives
the search toward points it has already evaluated. Fitting a noise term instead, which
is what a library would do, would let the surrogate explain the surface's sharp
features as measurement error and flatten them, which is precisely backwards. The
fold-to-fold spread is not that noise either: it describes uncertainty about the
population score, whereas the surrogate models the number the callable returns.

The acquisition
---------------

Expected improvement in closed form, maximised over `N_CAND` fresh Sobol points each
iteration rather than by an inner optimiser. At two dimensions 512 points sit 0.044 of
the box apart, which is finer than the surrogate can resolve, and this removes an
optimiser, its tolerances and its restarts from a module whose whole content would
otherwise be linear algebra. That trade is dimension-dependent -- the candidate count
would have to grow exponentially to hold the spacing -- and nothing here is claimed
beyond the two dimensions it is written for. The candidates are drawn from the same
generator that supplied the initial design, so a run is one Sobol sequence and repeats
exactly given `seed`.

What the constants are worth
----------------------------

None of the four below is a knob, and all four were set by measurement. The benchmark
is eight `KFoldCV` surfaces over B-spline pairs of varying `n`, basis size and signal
strength, each against its own 121x121 reference grid, with regret reported in units of
the score's own fold standard error over thirty Sobol scramblings apiece; plus one
analytic bump narrow enough that a search has to localise it rather than cover it.

The lesson of the cross-validation surfaces is that nothing matters. Between 23% and
78% of each box lies within one standard error of its maximum, and every setting tried
of every constant -- `N_CAND` from 64 to 4096, `JITTER` across eight decades, `ELL`
from one fixed value to a 32-point grid -- lands within a tenth of a standard error of
the reference. That is the argument for having four constants and exposing none of
them. The bump is what separates them, and the shortfalls quoted below are its.

`N_INIT`
    At a budget of 24 the worst shortfall over forty scramblings is 0.024 at 8, against
    0.45 at 4 and 0.43 at 16 -- an order of magnitude either side. Too few and the
    likelihood cannot yet see a length-scale, so it takes the flattest one offered and
    the acquisition extrapolates confidently to a corner; too many and the budget went
    on covering a box that is mostly flat.
`N_CAND`
    The worst shortfall falls from 0.032 at 64 to 0.008 at 512, and then to 0.0079 at
    1024 for 1.7 ms an iteration against 1.0.
`ELL`
    Instrumented over 3840 fits on the eight surfaces, the chosen length-scale ran from
    0.155 to 0.739, so both ends sit a factor of nearly three outside anything observed
    and one being chosen would be worth noticing. Eight points is as good as sixteen or
    thirty-two and six is not, at 1.2 ms an iteration against 1.8. Fitting rather than
    fixing is worth a little and buys more: the best fixed value found, 0.4, beats the
    fit over the eight surfaces at 0.021 worst regret against 0.041, but 0.3 loses to
    it at 0.075 -- and a constant that has to be right to within a third is one this
    module has no way to choose.
`JITTER`
    Regret is identical from 1e-12 to 1e-4 and degrades only at 1e-2, so this is doing
    a conditioning job and nothing else. Unjittered, `cond(K)` reached 4.9e11 over those
    3840 fits; 1e-8 caps it at 1e8, so it binds, and floors the posterior standard
    deviation at 1e-4 of a standardised score, too small to act as an exploration bonus
    -- at 1e-2 the floor is 0.1 and the regret shows it.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm, qmc

# Points evaluated before expected improvement means anything. A precondition rather
# than a budget: below this the posterior is mostly prior and its argmax is arbitrary.
N_INIT = 8

# Candidates the acquisition is maximised over, per iteration. Both this and `N_INIT`
# are powers of two, which is what Sobol' points need for their balance property and
# what keeps scipy from warning at a caller who cannot act on it.
N_CAND = 512

# Length-scales offered to the marginal likelihood, on the unit square; see the module
# docstring for both ends and for why there are only eight.
ELL = np.geomspace(0.05, 2.0, 8)

# Added to the correlation matrix's diagonal so that near-duplicate points -- which the
# acquisition actively seeks -- leave it invertible. Not a noise variance.
JITTER = 1e-8


def _matern52(A, B, ell):
    """Matern 5/2 correlation between two sets of points.

    `(1 + d + d**2 / 3) * exp(-d)` with `d = sqrt(5) * r / ell`, `r` the Euclidean
    distance. The distances are reformed on every call rather than cached and rescaled:
    at the sizes here that is a few hundred thousand flops against a Cholesky
    factorisation, and one argument list is worth more than the saving.

    Parameters
    ----------
    A : ndarray, shape (m, d)
    B : ndarray, shape (q, d)
    ell : float
        Length-scale, in the units of `A` and `B`.

    Returns
    -------
    ndarray, shape (m, q)
    """
    d = np.sqrt(5.0) / ell * np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    return (1.0 + d + d**2 / 3.0) * np.exp(-d)


def _log_marginal(U, y, ell):
    """Log marginal likelihood of `y` under the GP with this length-scale.

    `-0.5 * y.T @ K_inv @ y - 0.5 * log|K| - 0.5 * m * log(2 pi)`, with the log
    determinant read off the Cholesky factor as twice the sum of the log diagonal.

    Parameters
    ----------
    U : ndarray, shape (m, d)
        Evaluated points, on the unit box.
    y : ndarray, shape (m,)
        Standardised scores, which is what makes the unit amplitude assumed here right.
    ell : float

    Returns
    -------
    float
        The `2 pi` term is constant in `ell` and so changes no choice made from this.
        It is kept because with it the return value is a log density, which
        `scipy.stats.multivariate_normal` can check independently; dropping it would
        leave a quantity only this file knows the meaning of.
    """
    K = _matern52(U, U, ell) + JITTER * np.eye(len(U))
    L = cho_factor(K, lower=True)
    return (
        -0.5 * y @ cho_solve(L, y)
        - np.log(np.diag(L[0])).sum()
        - 0.5 * len(y) * np.log(2.0 * np.pi)
    )


def _posterior(U, y, V):
    """Posterior mean and standard deviation at `V`, at the best length-scale.

    The length-scale is chosen here rather than passed in because no caller has a use
    for it: it is refitted from scratch on every iteration, since the point of the last
    evaluation was to change what the surface looks like. Choosing it by a search over
    `ELL` rather than by an optimiser is the same trade `LmbdaCriterion.fit_lmbda`
    makes -- a grid wide enough to contain the answer costs less than a bracketing
    optimiser and has no tolerance to state.

    The winning length-scale is factorised twice, once inside `_log_marginal` and once
    here. That is sixteen Cholesky factorisations of an `m < 100` matrix per iteration
    against a callable that costs seconds, bought in exchange for `_log_marginal` being
    a function of `(U, y, ell)` alone and so testable against a density.

    Parameters
    ----------
    U : ndarray, shape (m, d)
        Evaluated points, on the unit box.
    y : ndarray, shape (m,)
        Standardised scores.
    V : ndarray, shape (q, d)
        Points to predict at.

    Returns
    -------
    mu : ndarray, shape (q,)
    sd : ndarray, shape (q,)
        Floored at `sqrt(JITTER)`: the posterior variance vanishes at an evaluated
        point and goes slightly negative there in floating point, and below the jitter
        the model does not resolve it anyway.
    """
    ell = ELL[np.argmax([_log_marginal(U, y, e) for e in ELL])]
    K = _matern52(U, U, ell) + JITTER * np.eye(len(U))
    L = cho_factor(K, lower=True)
    Ks = _matern52(V, U, ell)
    mu = Ks @ cho_solve(L, y)
    # The prior variance is 1 because the kernel is a correlation and `y` is
    # standardised, so this is 1 - k*.T @ K_inv @ k* row by row.
    var = 1.0 - (Ks * cho_solve(L, Ks.T).T).sum(1)
    return mu, np.sqrt(np.maximum(var, JITTER))


def _expected_improvement(mu, sd, best):
    """`E[max(Y - best, 0)]` for `Y ~ N(mu, sd**2)`, in closed form.

    `gain * Phi(z) + sd * phi(z)` with `z = gain / sd`, `gain = mu - best`. Separate
    from `_posterior` because it is the only thing in the module that is not linear
    algebra, and because stated this way it can be checked against the expectation it
    is the closed form of rather than against itself.

    Invariant to a positive affine map of the scores, which is what lets `search`
    standardise them: it scales with `sd` and `gain`, and the shift cancels in both.

    Parameters
    ----------
    mu, sd : ndarray, shape (q,)
        Posterior mean and standard deviation. `sd` must be positive, which
        `_posterior`'s floor guarantees.
    best : float
        The best score so far, in the same units.

    Returns
    -------
    ndarray, shape (q,)
    """
    gain = mu - best
    z = gain / sd
    return gain * norm.cdf(z) + sd * norm.pdf(z)


class GridSearch:
    """Every point of a regular grid over the box, evaluated.

    Attributes
    ----------
    func : callable
        `func(v) -> float`, `v` of shape (2,). Maximised.
    bounds : ndarray, shape (2, 2)
        Rows are dimensions, columns `(lo, hi)`.
    points : ndarray, shape (m, 2)
        Evaluated points, in the caller's coordinates. None before `search`.
    scores : ndarray, shape (m,)
        `func` at each, in the same order.

    The answer is `points[scores.argmax()]`. Nothing is stored beyond the two arrays
    because nothing else was measured: what a searcher may report is what it evaluated.
    """

    def __init__(self, func, bounds):
        """
        Parameters
        ----------
        func : callable
        bounds : array_like, shape (2, 2)
            See the class attributes.
        """
        self.func = func
        self.bounds = np.asarray(bounds, dtype=float)
        self.points = self.scores = None

    def search(self, grid_size=15, report=None):
        """Evaluate `func` on a `grid_size` by `grid_size` grid.

        The grid is regular in the coordinates it is given, so a caller searching in
        `log(lmbda)` gets a geometric grid in `lmbda` without this having to know it.

        Parameters
        ----------
        grid_size : int
            Points per dimension, so `grid_size**2` evaluations. There is no other
            knob: refinement, early exit and a stopping rule would all make this
            something other than the reference `BayesSearch` is checked against.

            15 puts the points 1/14 of the box apart, half the shortest correlation
            length the surrogate ever fitted on the benchmark surfaces, so the grid
            resolves what is there rather than sampling it. Measured worst-case regret
            over those eight surfaces is 0.030 of a fold standard error, against 0.026
            at 20x20 for nearly twice the evaluations and 0.071 at 8x8.
        report : callable, optional
            Called after each evaluation as `report(i, total, {"cv R²": best})`, with
            `i` counting evaluations completed and `total` being `grid_size**2`. The
            best so far rather than the last value, since that is the number a caller
            watches. A callback is what keeps this module knowing nothing about its
            caller: what is displayed, and whether anything is, stays outside.

        Returns
        -------
        self
            `scores.reshape(grid_size, grid_size)` is the surface, first axis the first
            dimension of `bounds`.
        """
        axes = [np.linspace(lo, hi, grid_size) for lo, hi in self.bounds]
        mesh = np.meshgrid(*axes, indexing="ij")
        self.points = np.stack(mesh, axis=-1).reshape(-1, len(axes))

        scores = []
        for v in self.points:
            scores.append(self.func(v))
            if report is not None:
                report(len(scores), grid_size**2, {"cv R²": max(scores)})
        self.scores = np.array(scores)
        return self


class BayesSearch:
    """Expected improvement over a Gaussian process fitted to what has been evaluated.

    Attributes
    ----------
    func : callable
        `func(v) -> float`, `v` of shape (2,). Maximised.
    bounds : ndarray, shape (2, 2)
        Rows are dimensions, columns `(lo, hi)`.
    seed : int or Generator, optional
        Seeds the scrambling of the Sobol' sequence, which supplies both the initial
        design and every candidate set, so a run repeats exactly.
    points : ndarray, shape (n_eval, 2)
        Evaluated points, in the caller's coordinates and in the order evaluated -- the
        first `N_INIT` the initial design, the rest chosen by the acquisition. None
        before `search`.
    scores : ndarray, shape (n_eval,)
        `func` at each.

    The answer is `points[scores.argmax()]`, read off the measurements rather than off
    the surrogate: the posterior mean's maximiser is a prediction, and it is largest
    where the search has been least, which is where it should be trusted least.
    """

    def __init__(self, func, bounds, seed=None):
        """
        Parameters
        ----------
        func, bounds, seed
            See the class attributes.
        """
        self.func = func
        self.bounds = np.asarray(bounds, dtype=float)
        self.seed = seed
        self.points = self.scores = None

    def search(self, n_eval=40, report=None):
        """Evaluate `func` `n_eval` times, choosing where after the first `N_INIT`.

        Parameters
        ----------
        n_eval : int
            Total evaluations, the initial design included, and the only budget: there
            is no convergence test, because on a surface flat enough for one to fire
            the points it would have stopped short of are the informative ones.

            40 is where the benchmark curve flattens: worst-case regret over the eight
            surfaces is 0.041 of a fold standard error at 40, against 0.123 at 30 and
            0.012 at 50, and the surrogate's own cost grows faster than linearly
            afterwards -- 1.5 ms an iteration at 40 against 4.6 at 120.
        report : callable, optional
            As `GridSearch.search`, with `total` being `n_eval`. The initial design is
            counted like any other evaluation: it costs the objective the same, which
            is what a progress count is about.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If `n_eval` is below `N_INIT`. Truncating the initial design instead would
            silently give a smaller search than the one asked for; running it in full
            would silently give a larger one.
        """
        if n_eval < N_INIT:
            raise ValueError(f"n_eval must be at least N_INIT, {N_INIT}, got {n_eval}")

        lo, hi = self.bounds.T
        sobol = qmc.Sobol(len(self.bounds), seed=self.seed)
        points = list(lo + sobol.random(N_INIT) * (hi - lo))
        scores = []
        for v in points:
            scores.append(self.func(v))
            if report is not None:
                report(len(scores), n_eval, {"cv R²": max(scores)})

        for _ in range(n_eval - N_INIT):
            y = np.array(scores)
            # Standardising removes the amplitude and the prior mean from the GP. A
            # constant objective would divide by zero here; leaving the scale at 1
            # makes the acquisition uniform and the run degenerate to its Sobol'
            # sequence, which is the honest response to a surface with no information.
            spread = y.std()
            y = (y - y.mean()) / (spread if spread > 0 else 1.0)

            U = (np.array(points) - lo) / (hi - lo)
            cand = sobol.random(N_CAND)
            mu, sd = _posterior(U, y, cand)
            ei = _expected_improvement(mu, sd, y.max())

            v = lo + cand[ei.argmax()] * (hi - lo)
            points.append(v)
            scores.append(self.func(v))
            if report is not None:
                report(len(scores), n_eval, {"cv R²": max(scores)})

        self.points, self.scores = np.array(points), np.array(scores)
        return self


# ---------------------------------------------------------------------------------
# the specification
# ---------------------------------------------------------------------------------


class GridSpec:
    """A grid search stated before it has an objective: the budget alone.

    `GridSearch` takes its objective and its box at construction, and a caller that
    wants to say *which* search to run, and how much of it, before either exists needs
    somewhere to put the budget in the meantime. That is the whole of this class.

    Attributes
    ----------
    grid_size : int
        As `GridSearch.search` reads it.
    """

    def __init__(self, grid_size):
        """
        Parameters
        ----------
        grid_size : int
            See the class attributes, and `grid` for the default.
        """
        self.grid_size = grid_size

    def search(self, func, bounds, seed=None, report=None):
        """Run the search this specifies.

        Parameters
        ----------
        func, bounds
            As `GridSearch`.
        seed : optional
            Accepted and ignored: the grid is deterministic and has no seed, which is
            half of what it is kept for. The signature matches `BayesSpec.search` so
            that a caller holding one of the two need not know which.
        report : callable, optional
            As `GridSearch.search`.

        Returns
        -------
        GridSearch
            Already run, so `points` and `scores` are the surface it measured. A fresh
            one per call: this holds a budget, and the results belong to one run.
        """
        return GridSearch(func, bounds).search(self.grid_size, report=report)


class BayesSpec:
    """A Bayesian search stated before it has an objective: the budget alone.

    As `GridSpec`, whose docstring gives the argument for both.

    Attributes
    ----------
    n_eval : int
        As `BayesSearch.search` reads it.
    """

    def __init__(self, n_eval):
        """
        Parameters
        ----------
        n_eval : int
            See the class attributes, and `bayes` for the default.
        """
        self.n_eval = n_eval

    def search(self, func, bounds, seed=None, report=None):
        """Run the search this specifies.

        Parameters
        ----------
        func, bounds, seed
            As `BayesSearch`. The seed arrives here rather than being held, so that one
            number given to a caller can reach both this and whatever else that caller
            makes random -- `cv.KFoldCV`'s split, for the one this was written for.
        report : callable, optional
            As `BayesSearch.search`.

        Returns
        -------
        BayesSearch
            Already run; as `GridSpec.search`.
        """
        return BayesSearch(func, bounds, seed).search(self.n_eval, report=report)


def grid(grid_size=15):
    """Construct a `GridSpec`.

    Parameters
    ----------
    grid_size : int
        Points per dimension, so `grid_size**2` evaluations; see `GridSearch.search`
        for what 15 buys and what it does not.

    Returns
    -------
    GridSpec
    """
    return GridSpec(grid_size)


def bayes(n_eval=40):
    """Construct a `BayesSpec`.

    Parameters
    ----------
    n_eval : int
        Total evaluations, the initial design included; see `BayesSearch.search`.

    Returns
    -------
    BayesSpec
    """
    return BayesSpec(n_eval)


def as_spec(method):
    """The specification `method` names: a `GridSpec` or `BayesSpec` as given, a name at
    its default budget, or `bayes()` for None.

    Here rather than at the call site because the names belong to this module -- a
    caller that had to know them would be the second place the strategies are
    enumerated. `isinstance` is the whole of the shorthand rather than a type gate on a
    duck-typed argument: being a string is what makes something a name.

    Parameters
    ----------
    method : GridSpec, BayesSpec, {"grid", "bayes"} or None
        A specification is returned unchanged, whatever it is, so a caller may supply
        one of its own. None is `bayes()`; see the module docstring for that choice.

    Returns
    -------
    GridSpec or BayesSpec

    Raises
    ------
    ValueError
        If `method` names neither. A misspelling would otherwise reach the caller as an
        `AttributeError` from wherever it first tried to search -- for `cv.EigenCVFit`,
        after every fold has been built.
    """
    specs = {"grid": grid, "bayes": bayes}
    if isinstance(method, str):
        if method not in specs:
            raise ValueError(f"method must be 'grid' or 'bayes', got {method!r}")
        return specs[method]()
    return bayes() if method is None else method
