"""A sequence of joint fits, each orthogonal to the ones before it.

`Probe` fits `bilinear` joint fits of a smoother against a linear model one after
another, deflating between them, and the components together define two maps into
`R^p`:

    phi(z) = u_1 f_1(z) + ... + u_d f_d(z)
    psi(x) = u_1 g_1(x) + ... + u_d g_d(x),    g_k(x) = w_k.T x + b_k

`phi` maps the covariate space onto a manifold in `R^p`; `psi` estimates a point of that
manifold from `x`. The `f_k` come from the smoother, the `g_k` from the linear model,
and `u_k` from neither -- it is `Xc.T @ f_k / n`, the covariance of the data columns
with the fitted index.

`Probe` fits, and nothing here reorders, rescales or rotates what it produced: after the
last component there is nothing left to decide about the fit and everything left to
decide about the frame it is written in, and the frame is `manifold.Manifold`'s.

The deflation
-------------

`BilinearModel.add_constraints(c)` imposes `(X1 c).T @ (X1 beta1) == 0`, so with `c` the
last component's smoother coefficients it reads `f_prev . f_new == 0` on the training
rows. That is the whole of the orthogonalisation: nothing is subtracted by hand, the
constraint lives in the smoother's cache, and every fit after it is orthogonal to every
one before. Hence

    F.T @ F / n == I

-- the columns of `F` are centred by the basis, unit mean square by the scale
constraint, and mutually orthogonal by the deflation. That is what makes `f_1 ... f_d`
an orthonormal frame under the empirical measure on the training rows, and so makes the
components a frame for one map rather than a list of unrelated fits.

It also bounds how many there can be. Each constraint costs the smoother a rank, so a
centred basis of `k` functions carries `k - 1` components. Running out warns and stops
with whatever has been fitted: asking for more components than the basis can carry is a
statement about the basis, and the ones already fitted are not thereby wrong.

The fitter is built once
------------------------

Only the smoother changes between components, and only by one constraint row, so one
fitter is built and `add_constraints` brings it up to date. Under `cv.CVCriterion` that
reaches the fold models a rebuild would have reached -- `cv.KFoldCV` copies
`constraints` off its parents verbatim, and the two routes agree to roundoff -- at the
cost of re-forming each fold's `K`, `O(n r1 r2)`, where a rebuild would re-decompose the
linear model's design, which never changes, `d * k` times. At `p = 1000`, `n = 50000`,
`k = 5`, `d = 20` that difference is the dominant cost of the whole fit.

The criterion
-------------

`Probe` dispatches on nothing. Both `als.ALSCriterion` and `cv.CVCriterion` carry

    fitter(sm, lm, test) -> BilinearModel

and construction is one call to it. A bare `LmbdaCriterion` -- `gcv()`, `reml()` -- is
shorthand, wrapped in `als()` on arrival: choosing GCV over REML is a statement about
the fit, whereas alternating, its tolerance and its history depth are how the library
happens to reach the answer, and a caller with no view on those should not have to name
them, or learn what ALS is. The price is two spellings of one thing, `gcv()` and
`als(gcv())`, and it is paid because the short one is what gets written and the long one
is the one that says what it is.
"""

import warnings

from .als import als
from .linear_model import LinearModel, LmbdaCriterion
from .linear_smoother import LinearSmoother
from .manifold import Manifold


class Component:
    """One fitted joint component, and no methods.

    Attributes
    ----------
    c : ndarray, shape (p1,)
        The smoother's coefficients, so `f = basis.model_matrix(z) @ c`. Also the row
        the next component is deflated against.
    w : ndarray, shape (p2,)
        The linear model's coefficients.
    b : float
        Its intercept, so `g(x) = w.T @ x + b`.
    u : ndarray, shape (p2,)
        `Xc.T @ f / n`, the loading of each of the linear model's predictors on this
        component's index; see `bilinear.BilinearModel.u` for the four readings of it.
    r2 : float
        `1 - mean((f - g)**2) / var(f)` on the training rows -- how well the linear
        readout recovers this coordinate, which is what the fit maximises.
    test_r2 : float or None
        The same on the test set, where there is one. The number that says when to stop
        adding components.

    No `n`-sized array is kept: `F == basis.model_matrix(z) @ C` is one matmul off a
    matrix the smoother already holds. Neither `lmbda` is kept either -- the smoother
    renormalises `omega` at every constraint, so the scale each is measured against
    moves from one component to the next by design, and the numbers are comparable to
    nothing.
    """

    def __init__(self, c, w, b, u, r2, test_r2):
        self.c = c
        self.w = w
        self.b = b
        self.u = u
        self.r2 = r2
        self.test_r2 = test_r2


class Probe:
    """A sequence of deflated joint fits of a smoother against a linear model.

    Attributes
    ----------
    X : array-like, shape (n, p)
        The columns being explained, as supplied.
    z : array-like
        The covariates, as supplied; interpreting them is the basis's job.
    basis : Basis
        Centred, and set up on `z` by the smoother if it was not set up already.
    criterion : ALSCriterion or CVCriterion
        The specification the fitter is built from, after a bare `LmbdaCriterion` has
        been wrapped in `als()`. It is what a restart rebuilds from.
    test_set : tuple or None
        `(X_test, z_test)`, in the order of `(X, z)`.
    verbose : bool
        Whether each fit is given a `Progress` to report to.
    sm : LinearSmoother
        `lm1` of the pair: the constrained side, carrying one row per component fitted.
        `self.sm is self.fitter.lm1`, one shared reference.
    lm : LinearModel
        `lm2`, with an intercept. Never constrained, which is the way round `u` needs.
    fitter : BilinearModel
        Built once; see the module docstring.
    components : list of Component
        In the order fitted, which nothing sorts; `Manifold.reorder` is what does.
    """

    def __init__(self, X, z, basis, criterion=None, test_set=None, verbose=False):
        """Build the pair and the one fitter the whole sequence will use.

        Under a cross-validated criterion this builds the folds, so construction is the
        expensive part, as `cv.EigenCVFit`'s own always is.

        Parameters
        ----------
        X : array-like, shape (n, p)
            The columns to be explained. Centred by the linear model's intercept.
        z : array-like
            The covariates, of whatever shape the basis reads -- `(n,)` for a curve,
            `(n, d)` for a surface.
        basis : Basis
            Centred. Set up on `z` if it is not set up already; one that is already set
            up is taken exactly as it stands, as `LinearSmoother` takes it.
        criterion : optional
            How each fit chooses its two `lmbda`s. One of `gcv()`, `reml()`, `aic()` or
            `bic()`, selected on each model separately as the fit proceeds; or a
            cross-validated `kf()` or `ho()`, which scores the pair's two together on
            held-out rows. None is `gcv()`. `als()` is the same four with the settings
            of the alternation exposed -- `als(gcv(), accel=5)` -- and is the only
            reason to name it.
        test_set : tuple, optional
            `(X_test, z_test)`, in the order of `(X, z)`. Scored per component and
            reported, and it enters no fit. Under a cross-validated criterion it is
            scored here and ignored by the fitter, which has a held-out score already.
        verbose : bool
            Print one rewritten line per component; see `Progress`.

        Raises
        ------
        ValueError
            If `basis` is uncentred. The constant is then reachable, so `f = 1`
            satisfies the scale constraint, is matched exactly by `lm`'s intercept, and
            lies in the null space of any derivative penalty -- the objective is zero at
            a fit that says nothing. The `bs` and `tps` helpers always centre, so this
            is a basis built by hand.
        """
        if not basis.centre:
            raise ValueError(
                "basis must be centred: otherwise f = 1 satisfies the scale "
                "constraint, is matched exactly by the linear model's intercept, and "
                "is unpenalised, so the objective is zero at an empty fit"
            )
        self.X = X
        self.z = z
        self.basis = basis
        # A bare criterion is shorthand for alternating on it, and `als` reads None as
        # `gcv()`. Built here rather than as a default argument, so that two probes
        # never share one object.
        self.criterion = (
            als(criterion)
            if criterion is None or isinstance(criterion, LmbdaCriterion)
            else criterion
        )
        self.test_set = test_set
        self.verbose = verbose
        # `ALSFit.test` and `BilinearModel.r2` take the arguments to `lm1.predict` and
        # `lm2.predict` in that order, and the smoother is `lm1` -- the reverse of
        # `test_set`, which is in the order of `(X, z)`.
        self._test = None if test_set is None else (test_set[1], test_set[0])

        self.lm = LinearModel(X, fit_intercept=True)
        self.sm = LinearSmoother(z, basis)
        self.fitter = self.criterion.fitter(self.sm, self.lm, self._test)
        self.components = []

    def fit(self, n_components, continue_fit=False):
        """Fit components until there are `n_components` of them.

        A total rather than a count, so `fit(20, continue_fit=True)` after `fit(10)`
        fits ten more and a total at or below what is already fitted does nothing:
        keeping fewer components is a question about the map, not a reason to refit.

        Parameters
        ----------
        n_components : int
            How many components there should be when this returns.
        continue_fit : bool
            Keep what is fitted and add to it. False rebuilds the smoother and the
            fitter and starts over, constraints not being removable -- and under a
            cross-validated criterion that rebuilds the folds, which is the honest cost
            of starting over. `lm` is kept either way: it is never constrained and every
            field the next fit touches is overwritten, so rebuilding it would cost a
            full SVD of `X` for nothing.

        Returns
        -------
        self

        Warns
        -----
        UserWarning
            If the smoother runs out of rank before `n_components` is reached; see the
            module docstring for why that stops rather than raises.
        """
        if not continue_fit:
            self.sm = LinearSmoother(self.z, self.basis)
            self.fitter = self.criterion.fitter(self.sm, self.lm, self._test)
            self.components = []

        for k in range(len(self.components), n_components):
            if len(self.sm.omega) == 0:
                warnings.warn(
                    f"stopping at {k} components: every direction the smoother had has "
                    "been spent on a constraint, so there is none left to fit. A "
                    "larger basis carries more."
                )
                break

            self.fitter.report = Progress(k) if self.verbose else None
            self.fitter.fit()

            r2 = self.fitter.r2()
            test_r2 = None if self._test is None else self.fitter.r2(*self._test)
            self.components.append(
                Component(
                    c=self.sm.beta,
                    w=self.lm.beta,
                    b=self.lm.b,
                    u=self.fitter.u(),
                    r2=r2,
                    test_r2=test_r2,
                )
            )
            if self.verbose:
                metrics = {"train R²": r2}
                if test_r2 is not None:
                    metrics["test R²"] = test_r2
                self.fitter.report.finish(metrics, self.fitter.converged)

            # After the component is recorded, never before: `add_constraints` clears
            # `sm.beta`, which is the `c` being recorded.
            self.fitter.add_constraints(self.sm.beta)
        return self

    def inspect(self):
        """Print how many components were fitted, then the two `R^2`s of each.

        Its own printer rather than one shared with `Progress`: the marks and the
        running counter are the whole of what that one is for.
        """
        print(f"Components fitted: {len(self.components)}")
        for i, component in enumerate(self.components):
            metrics = f"train R² {component.r2:9.6f}"
            if component.test_r2 is not None:
                metrics += f"   test R² {component.test_r2:9.6f}"
            print(f"  Component {i}   {metrics}")

    def manifold(self, n_components=None, component_idx=None):
        """The map some or all of the fitted components define.

        Parameters
        ----------
        n_components : int, optional
            Keep the first this many, in the order fitted.
        component_idx : sequence of int, optional
            Keep these, 0-based, in the order given -- which is a frame, not a
            selection: `Manifold.reorder` is how a frame's order is changed afterwards.

        Returns
        -------
        Manifold

        Raises
        ------
        ValueError
            If both are given, the two being different answers to one question. If
            nothing has been fitted, in the words `LinearModel.predict` uses for the
            same situation: an empty map is not a map.
        """
        if n_components is not None and component_idx is not None:
            raise ValueError("give n_components or component_idx, not both")
        if not self.components:
            raise ValueError("no components fitted; call fit first")
        components = (
            self.components[:n_components]
            if component_idx is None
            else [self.components[i] for i in component_idx]
        )
        return Manifold(self.basis, components, self.z, self.X, self.test_set)


# ---------------------------------------------------------------------------------
# the display
# ---------------------------------------------------------------------------------


class Progress:
    """One rewritten line per component: the `report` callable a fitter is handed.

        •  Component 3     17/200   train R²  0.498120   test R²  0.451002
        ✓  Component 0     41/200   train R²  0.841212   test R²  0.820704
        ⚠  Component 2    200/200   train R²  0.524011   test R²  0.481133
        ✓  Component 1     40/40   cv R²  0.560100   train R²  0.713301

    `•` while the fit runs, `✓` when it finishes, `⚠` when it finishes having not
    converged. Which metrics appear is the *fitter's* statement and not a dispatch
    here: this formats whatever dict it is handed, in the order the dict gives it, and
    the two layouts above are `als.ALSFit` reporting a training `R^2` per sweep against
    `cv.EigenCVFit` reporting the best cross-validated score so far, whose two `R^2`s
    exist only once the winner has been refitted.

    Attributes
    ----------
    index : int
        The component being fitted, 0-based, matching `Probe.manifold`'s
        `component_idx`.
    metrics : dict
        What was last reported, so that `finish` can add to it rather than replace it.
    count : str
        The counter last reported, kept for the same reason: `finish` is handed no
        `i` and no `total`, the fit having stopped where it stopped.
    width : int
        The longest line written for this component. Every line is padded to it, since
        a `\\r` leaves whatever the previous one wrote past its end.
    """

    def __init__(self, index):
        """
        Parameters
        ----------
        index : int
            The component this line is for.
        """
        self.index = index
        self.metrics = {}
        self.count = ""
        self.width = 0

    def __call__(self, i, total, metrics):
        """Rewrite the line: `i` of `total` done, and these metrics.

        Parameters
        ----------
        i, total : int
            Units of the fitter's own budget completed, and that budget -- sweeps under
            `als.ALSFit`, evaluations under a search. `i` is right-aligned to the width
            of `total`, so the counter does not jog as it fills.
        metrics : dict
            Label to value.
        """
        self.count = f"{i:>{len(str(total))}}/{total}"
        self.metrics = metrics
        self._write("•")

    def finish(self, metrics, converged=True):
        """Close the line with `metrics` added to what was last reported, and a newline.

        Added rather than substituted, so that a search's cross-validated score stays
        on the line beside the two `R^2`s of the fit it chose. Under alternation the
        same two labels are already there and are overwritten in place, which is why
        one layout gains two entries at this point and the other does not.

        Parameters
        ----------
        metrics : dict
            Label to value, as `__call__` takes them.
        converged : bool
            The fitter's own. False marks the line rather than raising: `als.ALSFit`
            has already warned, and the fit it stopped on is still a fit.
        """
        self.metrics = {**self.metrics, **metrics}
        self._write("✓" if converged else "⚠")
        print(flush=True)

    def _write(self, mark):
        """Rewrite the line in place, padded to the longest yet written for it."""
        metrics = "   ".join(f"{k} {v:9.6f}" for k, v in self.metrics.items())
        line = f"  {mark}  Component {self.index}    {self.count}   {metrics}"
        self.width = max(self.width, len(line))
        print(f"\r{line:<{self.width}}", end="", flush=True)
