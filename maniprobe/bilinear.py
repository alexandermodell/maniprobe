"""Two penalised models fitted to each other in one eigendecomposition.

`BilinearModel` is a pair of `LinearModel`s and the problem of minimising, over both
their coefficients at once,

    ||X1 beta1 - X2 beta2 - b2||^2 + lmbda1 * beta1.T @ S1 @ beta1
                                   + lmbda2 * beta2.T @ S2 @ beta2

subject to `sum(f) == 0` and `||f||^2 / n == 1`, where `f = X1 @ beta1`. `X1` is assumed
centered, which is what makes the first constraint hold for every `beta1` -- it is not
checked. Bilinear in the sense that names it: linear in each model's coefficients given
the other's, which is what lets `als.ALSFit` reach it by ordinary penalised regressions.

`EigenFit` solves it in one eigendecomposition, at `lmbda`s given rather than chosen.
`lm2` is profiled out: for a fixed `f` the inner problem is exactly `lm2.set_y(f)` then
`lm2.fit()`, and its optimal value telescopes to `||f||^2 - sum(a2 * (U2.T @ f)**2)`.
Since `||f||^2 == n` is fixed, what is left is a Rayleigh quotient,

    n + nu.T @ M @ nu,    M = lmbda1 * diag(omega1) - K @ diag(a2) @ K.T

with `K = U1.T @ U2`, so `nu` is the eigenvector of the smallest eigenvalue `theta`,
scaled to `sqrt(n)`. One eigenproblem, no alternation.

The base class
--------------

`BilinearModel` is that problem without an algorithm: the two models, the asymmetry
between them, the `R^2` of one against the other, and the loadings `u` of `lm2`'s
predictors on `lm1`'s index. Subclasses supply a `fit` and differ in how it is computed
and where the two `lmbda`s come from, never in what is being minimised -- one
eigendecomposition at given `lmbda`s (`EigenFit`), alternation with each `lmbda`
re-selected from its own model's criterion as the fit proceeds (`als.ALSFit`), or the
eigendecomposition again with both cross-validated (`cv.EigenCVFit`).

The base is written here rather than in a module of its own because a thirty-line class
is not a module, and because this is where the objective all three minimise is stated.

The cache
---------

`K` is why this is a class and not a method. It depends on the two designs and on
neither `lmbda`, so a search over `(lmbda1, lmbda2)` pays `O(n r1 r2)` for it once and
`O(r1^2 r2 + r1^3)` per fit thereafter. At `r1 = r2 = 2000` and `n = 40000` that is 1.21
seconds to form against 0.52 for everything a fit then does -- the difference between
8.7 and 2.6 seconds per five-fold cross-validation score. It is the bargain
`LinearModel` strikes one level down, fix the design and diagonalise once, applied to
the pair; and it is the reason `LinearModel` has no `fit_jointly`, which had nowhere to
put a cache belonging to neither model.

The cache is also what keeps `fit` out of `n`-sized arrays entirely. `lm2`'s half would
be `lm2.set_y(X1 @ beta1)` then `lm2.fit()`, but `X1 @ beta1 == U1 @ nu`, so

    U2.T @ (X1 @ beta1) == K.T @ nu

and the response `lm2` is fitted to is never formed. That identity is `_ybar2`, which
`u` rides on too. `lm2.y` is therefore cleared rather than set, so `lm2.rss` and any
criterion attached to it raise after a joint fit. That is the honest state: a criterion
on `lm2` is meaningless mid-joint-fit, and a `y` left over from some earlier use would
silently outlive the fit it belonged to.

Relation to `LinearModel.fit`
-----------------------------

`-theta` is the Lagrange multiplier of the scale constraint, and it stands exactly where
`fit`'s unit weight on the data term stands:

    here:          nu_i = ybar_i / (lmbda1 * omega_i - theta)
    LinearModel:   nu_i = ybar_i / (lmbda1 * omega_i + 1)

so fitting `lm1` alone and then normalising gives a different *direction*, not merely a
different scale, and iterating those two converges somewhere else. `lmbda1` also no
longer sets the size of `f`, only its shape, so one calibrated for `lm1` alone does not
carry over. The accuracy here is set by the eigenvalue gap rather than by the usual
tolerance. `als.ALSFit` reaches the same solution path by alternation instead, at a
different indexing of `lmbda1`; its docstring gives the correspondence, and the reason
to prefer it is that alternation lets both `lmbda`s be re-selected as the fit proceeds,
which one eigendecomposition cannot.
"""

import numpy as np
from scipy.linalg import eigh


class BilinearModel:
    """A pair of `LinearModel`s fitted to each other.

    Attributes
    ----------
    lm1 : LinearModel
        The constrained side: no intercept, and `||X1 beta1||^2 / n == 1` after every
        fit. Its fitted values are the response `lm2` is fitted to, which is what makes
        the pair asymmetric and `lm1` the one `r2` scores against.
    lm2 : LinearModel
        Fitted to `lm1`'s fitted values. May carry an intercept.
    n : int
        Training rows. Held here rather than by any subclass because it is part of the
        statement of the problem -- the `n` in `||f||^2 / n == 1` and in `u` -- and not
        of an algorithm for solving it.
    converged : bool
        True unless the last `fit` stopped short of its own convergence test. A class
        attribute, since neither `EigenFit` nor `cv.EigenCVFit` has such a test to stop
        short of -- one eigendecomposition, and a search that always spends its whole
        budget -- so `als.ALSFit` is the only subclass that writes it.

    Subclasses differ in how the fit is computed and where the two `lmbda`s come from,
    never in what is being minimised, so `fit()` takes no arguments and leaves both
    models holding the joint fit. Anything a subclass accepts beyond that is optional
    and says only how.
    """

    converged = True

    def __init__(self, lm1, lm2):
        """
        Parameters
        ----------
        lm1, lm2 : LinearModel
            Fitted in place. `lm1`'s design must be centred, or the constant is
            reachable: `f = 1` then satisfies the scale constraint, is matched exactly
            by `lm2`'s intercept, and lies in the null space of any derivative penalty,
            so the objective is zero at a fit that says nothing. Not checked, being a
            property of a design rather than of anything this sees.
        """
        self.lm1 = lm1
        self.lm2 = lm2
        self.n = len(lm1.X)

    def fit(self):
        """Fit both models to each other. Implemented by subclasses.

        Returns
        -------
        self
        """
        raise NotImplementedError

    def add_constraints(self, c):
        """Constrain the pair so that `(X1 c).T @ (X1 beta1) == 0`.

        `LinearModel.add_constraints` on `lm1`, and nothing else. With `c` a previous
        fit's `beta1` that reads `f_prev . f_new == 0` on the training rows, so a
        sequence of joint fits is deflated by this call alone: the constraint lives in
        `lm1`'s cache, and every fit after it is orthogonal to every one before without
        anything being subtracted by hand.

        Here rather than on a subclass because the base owns the pair and the asymmetry
        that makes `lm1` the constrained side, and because it is what lets that sequence
        be one call whichever fitter is driving it. `lm2` is never constrained by this,
        which is the way round `u` needs: it tolerates a constrained `lm1` and refuses a
        constrained `lm2`.

        Parameters
        ----------
        c : ndarray, shape (p1,) or (k, p1), or list of ndarray
            Constraint rows in `lm1`'s coefficient space, as `LinearModel` reads them.

        Returns
        -------
        self
        """
        self.lm1.add_constraints(c)
        return self

    def r2(self, t1=None, t2=None):
        """`R^2` of `lm2`'s fit to `lm1`'s, defaulting to the training data.

        `lm1`'s fitted values are the response and `lm2`'s the prediction, matching the
        objective. On the training data the scale constraint makes the total sum of
        squares exactly `n`, but computing it from `f` costs nothing and is what makes
        this one expression rather than two: off the training data there is no such
        constraint and the spread of `f` has to be measured.

        Parameters
        ----------
        t1, t2 : optional
            Arguments to `lm1.predict` and `lm2.predict`; None on both scores the
            training data.

        Returns
        -------
        float
        """
        f, g = self.lm1.predict(t1), self.lm2.predict(t2)
        return 1.0 - np.mean((f - g) ** 2) / f.var()

    def _ybar2(self):
        """`lm1`'s fitted values in `lm2`'s basis: `U2.T @ f`.

        The response `lm2` is fitted to, in the coordinates its cache is written in --
        `LinearModel._ybar` for a `y` that is never set. No mean is subtracted:
        `sum(f) == 0` by the scale constraint on a centred `X1`, which is the assumption
        the whole pair rests on and the same one `EigenFit.fit` leans on for `b2`.

        `O(n r2)` here, alternation having `f` in hand and no `K` to form. `EigenFit`
        overrides it and forms neither.

        Returns
        -------
        ndarray, shape (r2,)
        """
        return self.lm2.U.T @ self.lm1.predict()

    def u(self):
        """Covariances of `lm2`'s predictors with `lm1`'s fitted index: `Xc2.T @ f / n`.

        Entry `j` is `sum_i (x_ij - xbar_j) f_i / n`. On the training data `f` is
        centred with unit mean square, so that is `cov(x_j, f)`, and equally
        `corr(x_j, f) * sd(x_j)`: the marginal loading of each of `lm2`'s predictors on
        the index, in the units of `x`. Three other readings of the same vector:

        - the right-hand side of `lm2`'s normal equations,
          `(Sigma_hat + lmbda2 * S2 / n) @ beta2 == u`, so `beta2` is `u` decorrelated
          and shrunk -- and `u` itself does not depend on `beta2` at all;
        - `-2u` is the gradient of the objective's data term at `beta2 == 0`;
        - `u / ||u||` maximises `cov(X2 @ w, f)` over unit `w`, which makes this the
          one-component PLS weight vector: the *unadjusted* loadings, where `beta2` is
          adjusted for the correlation among the `x`s.

        Taken from the cache rather than from `X2`, since `A_inv2 == U2.T @ Xc2` gives

            n * u == Xc2.T @ f == Xc2.T @ P2 @ f == A_inv2.T @ (U2.T @ f)

        for `P2 = U2 @ U2.T`, exact because `U2` spans `col(Xc2)`. Two consequences.
        `lm2.fit_intercept` does not matter: without one the cache holds `X2` uncentred
        and this returns `u + xbar * mean(f)`, whose second term is zero by the scale
        constraint -- so the centring is never formed and `u` is the same vector either
        way. And on `EigenFit` the whole of it is `O(r1 r2 + r2 p2)`, touching no
        n-sized array.

        Returns
        -------
        ndarray, shape (p2,)

        Raises
        ------
        ValueError
            If `lm2` carries constraints. `U2` then spans a proper subspace of
            `col(Xc2)`, so `Xc2.T @ P2 != Xc2.T` and this would quietly return the
            covariance with the *projection* of `f` onto the constrained span. Nothing
            in the cache can repair it: the directions the constraint removed are gone
            from `A_inv` and from `K` alike. A constrained `lm1` is no obstacle, `f`
            being `X1 @ beta1` whatever `beta1` was constrained to.
        """
        if self.lm2.constraints is not None:
            raise ValueError(
                "u is unavailable for a constrained lm2: its cache no longer spans the "
                "directions the constraint removed"
            )
        return self.lm2.A_inv.T @ self._ybar2() / self.n


# ---------------------------------------------------------------------------------
# the eigendecomposition
# ---------------------------------------------------------------------------------


class EigenFit(BilinearModel):
    """A pair of `LinearModel`s fitted to each other, at given `lmbda`s.

    Attributes
    ----------
    lm1, lm2 : LinearModel
        The pair, as `BilinearModel` describes them.
    K : ndarray, shape (r1, r2)
        `U1.T @ U2`, `lm2`'s design seen from `lm1`'s; see the module docstring.

    Both models are fitted in place. Constrain through `add_constraints` here rather
    than on a model directly: a constraint rebuilds that model's `U`, and `K` would
    otherwise no longer be formed from the designs the pair now carries. `lm2` cannot be
    constrained at all once this is built -- there is no route for it, since `K` is the
    only thing this could repair and `u` refuses such a model anyway. `fit` says so if
    either happens.
    """

    def __init__(self, lm1, lm2):
        """Form the cache `K` for a fixed pair of designs.

        Parameters
        ----------
        lm1, lm2 : LinearModel
            As `BilinearModel`, and fitted in place by `fit`. An intercept on `lm2` is
            optional: without one, `b2` is held at zero by `x_mean` being zeros.

        Raises
        ------
        ValueError
            If `lm1` fits an intercept -- the scale constraint would invalidate `b`.
        """
        if lm1.fit_intercept:
            raise ValueError("EigenFit requires lm1.fit_intercept=False")
        super().__init__(lm1, lm2)
        self.K = lm1.U.T @ lm2.U

    def fit(self, lmbda1=None, lmbda2=None):
        """Fit both models, at these `lmbda`s or at the ones they already hold.

        Parameters
        ----------
        lmbda1, lmbda2 : float, optional
            Set on the respective model before fitting, so the models' state always
            matches the fit they carry. None leaves a model's `lmbda` alone, which is
            what makes `fit()` the pair's analogue of `LinearModel.fit`.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If either `lmbda` is left None on a model that has none set. If a constraint
            has been added to either model since `K` was formed, which is checked on the
            one thing it changes for certain -- the rank -- rather than trusted to a
            docstring. Cheap enough to sit in a function a search calls thousands of
            times, and it is the whole of the guard: a reparametrisation that left the
            rank alone would leave `K` stale and undetected, but `add_constraints` only
            does that for a constraint that constrains nothing.
        """
        if (self.lm1.U.shape[1], self.lm2.U.shape[1]) != self.K.shape:
            raise ValueError(
                "K was formed from designs these models no longer carry; build a new "
                "EigenFit rather than constraining one of them in place"
            )
        if lmbda1 is not None:
            self.lm1.set_lmbda(lmbda1)
        if lmbda2 is not None:
            self.lm2.set_lmbda(lmbda2)
        if self.lm1.lmbda is None:
            raise ValueError("lmbda is not set; use set_lmbda, set_edf or fit_lmbda")
        a2 = self.lm2._shrinkage()  # raises the same way for lm2

        M = np.diag(self.lm1.lmbda * self.lm1.omega) - (self.K * a2) @ self.K.T
        # The smallest eigenpair *by design*, not a rank decision, so this stays on
        # `eigh` rather than `linalg.eig`: `M` is indefinite, and dropping its
        # noise-level eigenvalues would answer a question nobody asked. Nor is
        # `subset_by_index` reliably faster than a full `eigh` -- it is the honest
        # statement of what is wanted, and that is the whole of its justification.
        _, v = eigh(M, subset_by_index=[0, 0])
        nu = v[:, 0] * np.sqrt(self.n)
        # Both models' signs flip together without changing the objective or either
        # constraint, so the sign is a convention, not something the data decides.
        nu = nu * np.sign(nu[np.abs(nu).argmax()])

        self.lm1.beta = self.lm1.pullback(nu)
        self.lm1.b = 0.0
        # `lm2`'s half in the cache's coordinates: `U2.T @ (X1 @ beta1) == K.T @ nu`,
        # and its mean is zero because a centred `X1` gives `1.T @ U1 == 0`, so `b2`
        # keeps only the `x_mean` term -- zero in turn without an intercept.
        self.lm2.beta = self.lm2.pullback(a2 * (self.K.T @ nu))
        self.lm2.b = -self.lm2.x_mean @ self.lm2.beta
        self.lm2.y = self.lm2.y_mean = None
        return self

    def add_constraints(self, c):
        """Constrain `lm1`, then re-form the `K` its rebuilt `U1` has invalidated.

        The one supported route past `fit`'s stale-`K` raise, which stays as it is for
        everything else -- a constraint reaching a model by any other path still leaves
        `K` describing designs the pair no longer carries. `O(n r1 r2)`, a fraction of
        this object's own construction, and the reason a sequence of deflated fits
        reuses one `EigenFit` rather than rebuilding it: nothing about `lm2` changes,
        and rebuilding would re-decompose its design for nothing.

        Returns
        -------
        self
        """
        super().add_constraints(c)
        self.K = self.lm1.U.T @ self.lm2.U
        return self

    def _ybar2(self):
        """`U2.T @ f` from the cache: `K.T @ nu`, forming neither `f` nor a response.

        `nu` is recovered as `lm1.pushforward(lm1.beta)` rather than kept: `A_inv @ A`
        is the identity and `beta1` came from a `pullback`, so that is exact, and it
        holds for `als.ALSFit`'s Anderson-mixed `beta1` too, every term of which is a
        `pullback`. `O(r1 p1 + r1 r2)` against the base class's `O(n r2)`.

        Returns
        -------
        ndarray, shape (r2,)
        """
        return self.K.T @ self.lm1.pushforward(self.lm1.beta)
