"""Generalised ridge regression with a reusable design cache.

Solves

    min_beta  ||y - X beta - b||^2 + lmbda * beta.T @ S @ beta

where `X` and `S` are fixed but many different `y`s and `lmbda`s are tried. The
constructor precomputes a change of basis in which the problem is *diagonal*, so each
new (y, lmbda) pair costs O(r) rather than a fresh solve.

The cache is three arrays (see `LinearModel._build_cache`):

    U      (n, r)   orthonormal design,  U.T @ U == I
    omega  (r,)     diagonal penalty weights, normalised to unit median
    A      (p, r)   pullback map,   beta = A @ nu
    A_inv  (r, p)   pushforward map, its left inverse:  A_inv @ A == I

related by the invariant `Xc @ A == U`, where `Xc` is the (centered) design. In that
basis the solution is one line:

    nu = ybar / (1 + lmbda * omega),      ybar = U.T @ y

`lmbda` enters only as `lmbda * omega`, so the scale of `omega` is free. It is fixed
here at unit median over the penalised coordinates, which makes a `lmbda` comparable
between models and gives `set_lmbda` by hand something to mean. `lmbda` is therefore
relative to `omega` and is not the multiplier on the `S` you passed. Nothing needs it
to be: `lmbda` is chosen by a criterion or by `set_edf`, and both search the range
`lmbda_grid` derives from `omega` itself, which is invariant to this scale.

Everything else in this module -- fitting, EDF, the selection criteria, constraints --
is a thin function of those three arrays.
"""

import numpy as np
from scipy.optimize import brentq

from .linalg import EPS, eig, svd


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------


def _pinv_solve(S0, S01, atol):
    """Return `pinv(S0) @ S01`, dropping the noise-level eigenvalues of `S0`.

    A pseudoinverse rather than a solve because `S0` (the penalty restricted to the
    null space of the design) is routinely singular. Where the penalty misses that null
    space entirely `S0` is roundoff throughout, and measured against its own spectrum
    nothing is zeroed -- so the floor is what stops this dividing by roundoff and
    handing the caller a Schur complement with a large negative eigenvalue in place of
    the flat direction it actually has.

    Dropping such a direction is the right answer and not merely the stable one. For a
    PSD `S`, `S0 @ v == 0` gives `S @ V0 @ v == 0` and hence `S01.T @ v == 0`, so the
    penalty along `v` is flat rather than sloped: every `xi` there minimises it, and
    zero is as good a choice as any.

    Parameters
    ----------
    S0 : ndarray, shape (m, m)
        Symmetric, possibly singular. Definiteness is not required: `S` is checked
        once the Schur complement is formed, which is after this runs.
    S01 : ndarray, shape (m, k)
    atol : float
        Absolute floor on the cut; see `linalg`'s module docstring. Required rather
        than defaulted, there being no sensible default: the plain relative rule zeroes
        nothing at all when every eigenvalue is roundoff.

    Returns
    -------
    ndarray, shape (m, k)
    """
    lam, Q = eig(S0, atol=atol)
    return Q @ ((Q.T @ S01) / lam[:, None])


def _null_space(K, dim):
    """The last `dim` right singular vectors of `K`: a basis of its null space whose
    dimension was decided elsewhere.

    The rank of a constraint is decided where its scale means something (on `ctilde`,
    against the size of the map), and `K` is that constraint written in coordinates
    whose columns are scaled by anything from `EPS` to `1 / EPS`. Measured against its
    own spectrum it would be ranked wrongly, so the count is carried in instead -- the
    same transfer `linalg.eig`'s `n_zero` makes for a congruence.

    Parameters
    ----------
    K : ndarray, shape (k, m)
    dim : int
        `m` less the number of independent constraints.

    Returns
    -------
    ndarray, shape (m, dim)
        Orthonormal columns.
    """
    _, _, Vt = np.linalg.svd(K, full_matrices=True)
    return Vt[len(Vt) - dim :].T


def _diagonalise(M, Y):
    """Diagonalise a penalty against an orthonormal design, never forming its matrix.

    Coordinates `t` reach the fitted values through `M` -- the column `M @ t` is the fit
    in an orthonormal basis of the column space -- and are penalised by
    `||Y @ t||^2`. This returns an orthonormal basis `W` of `range(M)` and weights
    `omega` with `t`'s penalty equal to `nu.T @ diag(omega) @ nu` at `M @ t == W @ nu`.

    The obvious route forms the penalty in the orthonormal basis and eigendecomposes it,
    and that route loses exactly the eigenvalues that matter. A basis function the data
    barely reaches gets an enormous weight there, an eigensolver's absolute error is
    machine epsilon times the largest weight, and so every *small* weight -- the smooth,
    lightly penalised directions a fit is made of -- is roundoff, sign included. On a
    40 x 80 tensor-product spline over the mainland U.S. the weights spanned 1e12,
    two came out negative, and the relative cut of a constrained re-diagonalisation
    zeroed 2,433 of 2,580 of them.

    So the penalty is whitened first and the *data* decomposed instead: in coordinates
    `y` with penalty `||y||^2`, the fitted columns have singular values `sigma`, and
    `omega = 1 / sigma**2`. A small weight is now a large singular value, which an SVD
    resolves to full relative accuracy; the roundoff lands on the large weights, whose
    directions are shrunk to nothing at any `lmbda` a criterion would choose. And
    `omega >= 0` holds by construction rather than by luck.

    The penalty's null space is split off first -- it carries no `y` to whiten -- and
    the penalised directions are made orthogonal to its fits, which is the profiling
    `_build_cache` does in step 2, here on the design side.

    Parameters
    ----------
    M : ndarray, shape (r, q)
        Full column rank.
    Y : ndarray, shape (m, q)
        Any scale; only its row space and singular values enter.

    Returns
    -------
    W : ndarray, shape (r, q')
        Orthonormal; `q' == q` unless `M` was rank-deficient in a direction `Y`
        leaves unpenalised.
    omega : ndarray, shape (q',)
        Ascending, the unpenalised entries exactly zero. Not normalised.
    """
    r, q = M.shape
    if len(Y):
        _, sy, Vyt, V0t = svd(Y, return_null=True)
    else:
        sy, Vyt, V0t = np.zeros(0), np.zeros((0, q)), np.eye(q)

    # The penalty's null space, carried as an orthonormal basis of its fits.
    Qn = svd(M @ V0t.T)[0] if len(V0t) else np.zeros((r, 0))

    # The penalised directions, whitened (`t = Vy diag(1 / sy) y`, so the penalty is
    # `||y||^2`) and profiled against the null space, whose fits are free.
    P = (M @ Vyt.T) / sy
    P -= Qn @ (Qn.T @ P)
    if not P.shape[1]:
        return Qn, np.zeros(Qn.shape[1])
    Ub, sigma, _ = np.linalg.svd(P, full_matrices=False)

    # A floor rather than a cut: a direction the data cannot resolve against the
    # penalty is kept, shrunk to nothing, so that `U` keeps its rank. It also bounds
    # `omega`'s spread at `1 / EPS**2`, which keeps `lmbda_grid` finite and every
    # criterion free of `inf - inf`.
    sigma = np.maximum(sigma, EPS * sigma[0])
    W = np.hstack([Qn, Ub])
    return W, np.concatenate([np.zeros(Qn.shape[1]), 1.0 / sigma**2])


# ---------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------


class LinearModel:
    """Generalised ridge regression over a fixed design and penalty.

    Attributes
    ----------
    X : ndarray, shape (n, p)
        The design as supplied, uncentered.
    S : ndarray, shape (p, p), or None
        The penalty as supplied -- neither coerced nor symmetrised, since
        `_build_cache` works on its own copy. Nothing here reads it after
        construction: it is retained so that a caller can rebuild this model over a
        subset of the rows, which is what `cv.KFoldCV` does. `add_constraints`
        reparametrises the cache and does not touch it, so on a constrained model
        this is the unconstrained penalty.
    constraints : ndarray, shape (k, p), or None
        Every constraint row imposed on this model, whether at construction or by
        `add_constraints`, and the third field of its specification. Retained for the
        same reason as `X` and `S`: it is what lets a caller rebuild this model over a
        subset of the rows. A fold rebuilt from `X` and `S` alone is a differently
        constrained model, and says nothing about this one.
    x_mean : ndarray, shape (p,)
        Column means of `X`; zeros if not fitting an intercept.
    U, omega, A, A_inv
        The design cache; see the module docstring. The centered design itself is
        not retained -- nothing after construction needs it.
    y, y_mean, lmbda, beta, b
        Fitted state. `y` and `lmbda` start as None and must both be set (or `lmbda`
        chosen by a criterion) before `fit`.
    """

    def __init__(
        self, X, S=None, fit_intercept=False, constraints=None, lmbda_criterion=None
    ):
        """Build the design cache for a fixed `X`, `S` and set of constraints.

        Parameters
        ----------
        X : ndarray, shape (n, p)
        S : ndarray, shape (p, p), optional
            Symmetric positive semi-definite penalty; see `_build_cache` for how far
            that is checked. None means the identity, which is ordinary ridge and
            takes a much cheaper path.
        fit_intercept : bool
            If True, `X` is centered here and the intercept is recovered in `fit`.
        constraints : ndarray, shape (p,) or (k, p), or list of ndarray, optional
            Constraint rows, meaning exactly what they mean in `add_constraints`.
            Imposing them here costs one eigendecomposition where imposing them
            afterwards costs two, so this is the route for a constraint that is known
            this early -- and the route a rebuild takes, since `constraints` is part of
            the specification a caller reads off a model.
        lmbda_criterion : LmbdaCriterion, optional
            Attached for later use by `fit_lmbda`; see `set_lmbda_criterion`.
            Attaching alone selects nothing.

        Raises
        ------
        ValueError
            As `_coerce_constraints` raises: `constraints` with an intercept, or a row
            of the wrong length.
        """
        self.X = np.asarray(X, dtype=float)
        self.S = S
        self.fit_intercept = fit_intercept
        self.constraints = (
            None if constraints is None else self._coerce_constraints(constraints)
        )
        # Zero means with no intercept, so that `fit` and `predict` need no branch:
        # b = y_mean - x_mean @ beta is then exactly 0.
        self.x_mean = self.X.mean(0) if fit_intercept else np.zeros(self.X.shape[1])
        self._build_cache(
            self.X - self.x_mean if fit_intercept else self.X, S, self.constraints
        )

        self.y = self.y_mean = self.lmbda = self.beta = self.b = None
        self.set_lmbda_criterion(lmbda_criterion)

    # -- cache ---------------------------------------------------------------------

    def _build_cache(self, Xc, S, C=None):
        """Set `self.U`, `self.omega`, `self.A` and `self.A_inv` from `Xc`, `S` and `C`.

        `Xc` is passed rather than stored: it is the only n-sized array the cache
        needs, and only here.

        Step 1 -- compact SVD `Xc = U1 diag(D1) V1.T`, with `V1` spanning the row space
        and `V0` the null space of `Xc`.

        Step 2 -- profile the null space out of the penalty (a Schur complement).
        Writing `beta = V1 @ eta + V0 @ xi`, the data fit never sees `xi`, so
        minimising the penalty over it gives `xi = -Z @ eta` with `Z = pinv(S0) @ S01`
        and collapses the penalty to `eta.T @ Stil @ eta`, where
        `Stil = S1 - S01.T @ Z`. The lift back to the full coefficient vector is
        `T = V1 - V0 @ Z`. A pseudoinverse and not a solve because the penalty need not
        reach every null direction, and where it reaches none of them `S0` is roundoff
        throughout -- which is what `_pinv_solve`'s floor is for.

        Step 3 -- whiten the penalty, restrict to the constraints, and diagonalise.
        With `Stil = E diag(s) E.T`, write `eta = E0 @ z + E+ @ (y / sqrt(s+))`, so the
        penalty is `||y||^2` and the fitted values are `U1 @ mu` with `mu = D1 * eta`.
        Under constraints `(z, y)` is restricted to the null space of the constraints
        written in those coordinates. `_diagonalise` then returns an orthonormal `W`
        with `U = U1 @ W` and the penalty diagonal. `omega` are the eigenvalues of the
        pencil `(Stil, D1**2)`: penalty per unit of the curvature the data supplies --
        obtained as `1 / sigma**2` from the data's singular values rather than from an
        eigendecomposition of `Stil / D1 / D1.T`, which resolves the small ones to
        roundoff; see `_diagonalise` for why.

        Step 4 -- normalise `omega` to unit median over its non-zero entries, which
        fixes what `lmbda` means; see `_normalize_omega` and the module docstring.

        `S=None` short-circuits steps 2 and 3: `omega = 1 / D1**2`, `U = U1`,
        `A = V1 / D1`. A constraint mixes the coordinates that penalty is diagonal in,
        so a constrained ridge model is rediagonalised like any other.

        Where a constraint enters
        -------------------------

        `(Xc c).T @ (Xc beta) == 0` is `ctilde.T @ mu == 0` for
        `ctilde = diag(D1) @ V1.T @ c`, since `U1.T @ Xc == diag(D1) @ V1.T` exactly.
        The whole of a constraint is therefore a null space taken in step 3's
        coordinates, at `O(k p r)` and without touching an n-sized array. Imposing it
        here rather than on the finished cache -- which is what `add_constraints` must
        do, its constraint not being known until it is called -- saves one `r x r`
        decomposition and one `O(n r^2)` update of `U`. The two routes agree: both
        restrict the same whitened coordinates to the same subspace and diagonalise
        what is left, differing only by the singular-vector conventions of the SVDs.
        Earlier than here is worse, not better: in `p` the null space is that of
        `C @ Xc.T @ Xc`, and the design would then have to be decomposed a second time.

        Two ranks are decided where their scale means something and carried forward
        rather than measured again: the penalty's on `Stil`, at the scale of `S`, and
        the constraints' on `ctilde`, against the size of the map. The coordinates
        `(z, y)` between them are scaled by anything from `EPS` to `1 / EPS`, so no
        tolerance applied there would find either.
        """
        U1, D1, V1t, V0t = svd(Xc, return_null=True)

        # The constraints, in step 3's coordinates. The cut is floored by the size of
        # the map and of the constraints rather than judged against ctilde's own
        # largest singular value: a constraint with Xc @ c == 0 constrains nothing, and
        # its ctilde is roundoff whose spectrum must not be measured against itself.
        # The floor is `add_constraints`'s to the digit: `norm(A_inv, 2) == D1.max()`,
        # W being orthogonal and V1.T's rows orthonormal.
        Q = None
        if C is not None:
            ctilde = D1[:, None] * (V1t @ C.T)
            atol = EPS * D1.max() * np.linalg.norm(C, axis=1).max()
            *_, ctilde_null = svd(ctilde.T, return_null=True, atol=atol)
            Q = ctilde_null.T

        if S is None and Q is None:
            # Ordinary ridge. The penalty is already diagonal in this basis, so
            # steps 2 and 3 have nothing to do: omega_i = 1 / D1_i^2 gives the
            # familiar filter factor a_i = D1_i^2 / (D1_i^2 + lmbda), up to the scale
            # step 4 goes on to fix.
            self.U, self.omega, self.A = U1, 1.0 / D1**2, V1t.T / D1
            self.A_inv = D1[:, None] * V1t
            self._normalize_omega()  # step 4
            return

        if S is None:
            # Constrained ridge: the constraint mixes the coordinates 1 / D1^2 is
            # diagonal in, so the short-circuit above no longer applies. `Stil` is the
            # identity -- `xi` is penalised and unseen, so it is zero -- and so is its
            # whitening: `mu = D1 * y` with penalty `||y||^2`.
            T, M, Y = V1t.T, np.diag(D1), np.eye(len(D1))
        else:
            S = (S + S.T) / 2
            V1, V0 = V1t.T, V0t.T

            # Step 2: profile the null space out of the penalty. EPS**2 is machine
            # epsilon, so atol is the rounding error in forming a congruence of S --
            # the level below which neither S0 nor the Schur complement carries any
            # information. One level, stated once, used by both.
            atol = EPS**2 * np.abs(S).max()
            S1, S0, S01 = V1.T @ S @ V1, V0.T @ S @ V0, V0.T @ S @ V1
            Z = _pinv_solve(S0, S01, atol=atol)
            Stil = S1 - S01.T @ Z  # Schur complement
            T = V1 - V0 @ Z  # lift from reduced eta back to full beta

            # Step 3: whiten the penalty. The rank is decided here, on Stil, where the
            # scale is still that of S.
            stil_w, E = eig(Stil, keep_zeros=True, atol=atol)
            if (stil_w < 0).any():
                # Step 2 is a minimisation over xi only when S is PSD; otherwise Stil
                # is the value at a saddle point. The cover is partial by construction:
                # this sees the reduced problem, where beta lives, not a negative
                # direction confined to null(Xc).
                raise ValueError("S must be positive semi-definite")
            pen = stil_w > 0
            # eta = E0 @ z + E+ @ (y / sqrt(s)), so the penalty is ||y||^2, and the
            # fitted values are U1 @ mu with mu = D1 * eta.
            M = D1[:, None] * np.hstack([E[:, ~pen], E[:, pen] / np.sqrt(stil_w[pen])])
            Y = np.eye(len(D1))[np.count_nonzero(~pen) :]

        if Q is not None:
            # Restrict the coordinates (z, y) to those whose fit satisfies the
            # constraints, with the count decided on `ctilde` above.
            N = _null_space(ctilde.T @ M, Q.shape[1])
            M, Y = M @ N, Y @ N

        W, self.omega = _diagonalise(M, Y)
        self.U = U1 @ W
        self.A = (T / D1) @ W
        # U.T @ Xc, but U1.T @ Xc == diag(D1) @ V1.T exactly, so this costs O(r^2 p)
        # instead of O(n r p) -- and no later call has to touch an n-sized array.
        self.A_inv = W.T @ (D1[:, None] * V1t)
        self._normalize_omega()  # step 4

    def _normalize_omega(self):
        """Build step 4: scale `omega` to unit median over its penalised entries.

        `lmbda` enters only as `lmbda * omega`, so the scale of `omega` is free and
        fixing it is what gives `lmbda` a meaning; see the module docstring. A penalty
        with no reach into the row space leaves nothing to normalise against, and
        nothing to normalise: `lmbda` does not act at all.

        Called wherever `omega` is set, `add_constraints` included, so that a model
        constrained at construction and the same model constrained afterwards carry the
        same `omega` -- and with it the same `lmbda`. Two models differing only in how
        they were told about a constraint would otherwise be searched over two different
        scales, which is what `cv.KFoldCV` rebuilding a constrained parent's folds would
        do.

        It is also the one place every `omega` passes through, so it is where a negative
        weight is refused. `_diagonalise` cannot produce one; a negative weight is a
        shrinkage factor above 1 or below 0, and every criterion evaluated on it is
        meaningless or NaN.
        """
        if (self.omega < 0).any():
            raise FloatingPointError(
                f"negative penalty weight {self.omega.min():.3g}; the cache is corrupt"
            )
        penalised = self.omega[self.omega > 0]
        self.omega = self.omega / (np.median(penalised) if len(penalised) else 1.0)

    def pushforward(self, c):
        """Map a coefficient vector into the orthonormal-design basis: `A_inv @ c`.

        `A_inv` is `U.T @ Xc`, the left inverse of `A`, so this inverts `pullback` on
        the model's parameter space -- and it is also the map that turns the
        constraint `(Xc).T @ (X beta) == 0` into plain orthogonality `ctilde.T @ nu`,
        which is what makes `add_constraints` cheap.

        Parameters
        ----------
        c : ndarray, shape (p,) or (p, k)

        Returns
        -------
        ndarray, shape (r,) or (r, k)
        """
        return self.A_inv @ c

    def pullback(self, nu):
        """Map back from the orthonormal-design basis to coefficients: `A @ nu`.

        Parameters
        ----------
        nu : ndarray, shape (r,)

        Returns
        -------
        ndarray, shape (p,)
        """
        return self.A @ nu

    # -- setting -------------------------------------------------------------------

    def set_y(self, y):
        """Set the response. Leaves `lmbda` alone; discards any existing fit.

        Parameters
        ----------
        y : ndarray, shape (n,)
        """
        y = np.asarray(y, dtype=float)
        if y.shape != (len(self.X),):
            raise ValueError(f"y must have shape ({len(self.X)},), got {y.shape}")
        self.y = y
        self.y_mean = y.mean() if self.fit_intercept else 0.0
        self.beta = self.b = None

    def set_lmbda(self, lmbda):
        """Set the penalty weight.

        Parameters
        ----------
        lmbda : float
            Non-negative, and relative to the normalised `omega` rather than to the
            units of the `S` you passed; see the module docstring. Zero is allowed and
            gives the unpenalised least-squares fit -- well defined here, since the
            shrinkage factors are simply 1.
        """
        if lmbda < 0:
            raise ValueError("lmbda must be non-negative")
        self.lmbda = float(lmbda)
        self.beta = self.b = None

    def set_lmbda_criterion(self, criterion):
        """Attach, replace or detach the criterion used to select `lmbda`.

        Attaching does not select anything -- call `fit_lmbda`.

        Parameters
        ----------
        criterion : LmbdaCriterion or None
            None detaches, leaving `lmbda` under manual control.
        """
        self.lmbda_criterion = criterion
        if criterion is not None:
            criterion.attach(self)

    def fit_lmbda(self):
        """Set `lmbda` by minimising the attached criterion.

        The explicit trigger for criterion-based selection, alongside `set_lmbda`
        (by hand) and `set_edf` (by degrees of freedom). `fit` never selects on your
        behalf; it raises if `lmbda` is unset.

        Returns
        -------
        self
        """
        if self.lmbda_criterion is None:
            raise ValueError("no lmbda_criterion attached; use set_lmbda_criterion")
        self.set_lmbda(self.lmbda_criterion.fit_lmbda())
        return self

    # -- core quantities -----------------------------------------------------------

    def _shrinkage(self, lmbda=None):
        """Per-coordinate shrinkage factors `a_i = 1 / (1 + lmbda * omega_i)`.

        The one quantity underlying `fit`, `edf`, `rss` and every criterion. Needs no
        `y`.

        Parameters
        ----------
        lmbda : float or ndarray, shape (G,), optional
            Defaults to the current `lmbda`.

        Returns
        -------
        ndarray, shape (r,) or (G, r)
        """
        lmbda = self.lmbda if lmbda is None else lmbda
        if lmbda is None:
            raise ValueError("lmbda is not set; use set_lmbda, set_edf or fit_lmbda")
        return 1.0 / (1.0 + np.asarray(lmbda, dtype=float)[..., None] * self.omega)

    def _ybar(self):
        """The response in the orthonormal-design basis: `U.T @ (y - y_mean)`.

        Recomputed on demand rather than cached, per the spec.

        Returns
        -------
        ndarray, shape (r,)
        """
        if self.y is None:
            raise ValueError("y is not set; use set_y")
        return self.U.T @ (self.y - self.y_mean)

    # -- fitting -------------------------------------------------------------------

    def fit(self):
        """Fit `beta` (and `b`, if fitting an intercept).

        Pushes forward, applies `nu = a * ybar` coordinate-wise, pulls back.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If `y` or `lmbda` is unset. Selection is never implicit: use
            `set_lmbda`, `set_edf` or `fit_lmbda` first.
        """
        self.beta = self.pullback(self._shrinkage() * self._ybar())
        self.b = self.y_mean - self.x_mean @ self.beta  # 0 without an intercept
        return self

    def predict(self, X_test=None):
        """Predict from the fitted model, defaulting to the training design.

        Parameters
        ----------
        X_test : ndarray, shape (m, p), optional
            Uncentered, like the `X` given to the constructor.

        Returns
        -------
        ndarray, shape (m,)
        """
        if self.beta is None:
            raise ValueError("model is not fitted; call fit first")
        return (self.X if X_test is None else X_test) @ self.beta + self.b

    def rss(self, lmbda=None):
        """Residual sum of squares, without refitting.

        `r_perp + sum((1 - a_i)^2 * ybar_i^2)`, where `r_perp = ||y||^2 - ||ybar||^2`
        is the part of `y` no coefficient can reach. Requires `y`.

        Parameters
        ----------
        lmbda : float or ndarray, shape (G,), optional
            Defaults to the current `lmbda`.

        Returns
        -------
        float or ndarray, shape (G,)
        """
        ybar = self._ybar()
        yc = self.y - self.y_mean
        # Clamped because the subtraction cancels to zero for a saturated design,
        # where roundoff could otherwise make it negative and send log(rss) to nan.
        r_perp = max(yc @ yc - ybar @ ybar, 0.0)
        return r_perp + ((1.0 - self._shrinkage(lmbda)) ** 2 * ybar**2).sum(-1)

    def minimize_penalty(self):
        """Set `beta` to the least-penalised fit the scale constraint allows.

        Minimises `beta.T @ S @ beta` subject to `||X beta||^2 / n == 1`, which in the
        orthonormal-design basis is `min nu.T @ diag(omega) @ nu` at `||nu||^2 == n` --
        so the answer is `sqrt(n)` on the smallest-`omega` coordinate and zero on the
        rest. It is the one fit here that needs no `y`: `omega` is a property of the
        design and the penalty alone.

        This is `bilinear.EigenFit`'s `lmbda -> inf` limit, whose objective is a
        *difference* at fixed scale, so a large enough `lmbda` leaves the penalty
        deciding the direction alone. It is not `fit`'s: there the scale is free, every
        coordinate shrinks by the same `1 / lmbda` to first order, and the direction
        that survives renormalisation stays `ybar / omega` -- a function of the data.

        For a penalised smoother that is the smoothest admissible function -- with a
        second-derivative penalty on a centred basis, the identity transform. For
        ordinary ridge (`S=None`) `omega` is `1 / D**2` up to step 4's scale, which
        `argmin` does not see, so it is the design's leading principal component.
        `ALSFit` starts its sweep from it.

        The minimiser is not unique when the smallest `omega` is repeated, which is any
        penalty whose null space has dimension above one. An arbitrary member of that
        space is returned; they are equally penalised by construction.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If `fit_intercept` is True -- the scale constraint would invalidate `b`,
            exactly as in `normalize_beta`.
        """
        if self.fit_intercept:
            raise ValueError("minimize_penalty requires fit_intercept=False")
        # `omega` comes out ascending from every branch that builds it, so this is
        # coordinate 0; `argmin` says which one is wanted without resting on that.
        nu = np.zeros(len(self.omega))
        nu[np.argmin(self.omega)] = np.sqrt(len(self.X))
        self.beta = self.pullback(nu)
        self.b = 0.0
        return self

    # -- degrees of freedom --------------------------------------------------------

    def edf(self, lmbda=None):
        """Effective degrees of freedom: `sum(a_i) + fit_intercept`.

        This is the model EDF -- the trace of the influence matrix -- and so it
        *includes* the intercept, matching mgcv and matching what GCV/AIC/BIC need.
        Requires no `y`, which is the point of `set_edf`.

        Parameters
        ----------
        lmbda : float or ndarray, shape (G,), optional
            Defaults to the current `lmbda`.

        Returns
        -------
        float or ndarray, shape (G,)
        """
        return self._shrinkage(lmbda).sum(-1) + self.fit_intercept

    def lmbda_grid(self, grid_size=300, edf_tol=0.1):
        """`lmbda`s spanning every fit this model can produce, to within `edf_tol`.

        `edf` falls from `r` at `lmbda = 0` to `#{omega == 0}` as `lmbda -> inf`, and
        both ends are first order in `omega`:

            lmbda -> 0     edf ~ r - lmbda * sum(omega)
            lmbda -> inf   edf ~ #{omega == 0} + sum(1 / omega) / lmbda

        so the ends losing exactly `edf_tol` degrees of freedom are

            lmbda_min = edf_tol / sum(omega),   lmbda_max = sum(1 / omega) / edf_tol

        both sums over penalised coordinates. `fit_intercept` sits in both limits and
        so cancels. This replaces a fixed `[lmbda_min, lmbda_max]`, which cannot be
        chosen without knowing `omega`: eight decades is right for `bs(k=10)` and
        twenty for a three-margin tensor product.

        Every `lmbda` outside the range gives a fit within `edf_tol` of one inside, so
        a search bounded this way reports a minimum at an endpoint as an answer rather
        than as a truncation -- which is why nothing here warns about one.

        Parameters
        ----------
        grid_size : int
            Number of points, spaced evenly in `log(lmbda)`. Costs `O(grid_size * r)`
            in time and in a temporary of that shape, so lower it for a very large
            `r`. The default errs high: a criterion flat enough for the spacing to
            matter is one whose exact minimiser does not.
        edf_tol : float
            Degrees of freedom given up at each end. Must be below the number of
            penalised coordinates, or the range inverts.

        Returns
        -------
        ndarray, shape (grid_size,), or (1,)
            A single zero if no coordinate is penalised: `lmbda` then does not act at
            all and every value gives the same fit.
        """
        omega = self.omega[self.omega > 0]
        if not len(omega):
            return np.zeros(1)
        return np.geomspace(
            edf_tol / omega.sum(), (1.0 / omega).sum() / edf_tol, grid_size
        )

    def set_edf(self, target, proportion=False, edf_tol=0.1):
        """Choose `lmbda` to hit a target EDF.

        `edf` is strictly decreasing in `lmbda`, so this is a root-find (Brent) on
        `rho = log(lmbda)`, bracketed by the ends of `lmbda_grid`.

        Parameters
        ----------
        target : float
            Target EDF, or a proportion of the total if `proportion`.
        proportion : bool
            If True, `target` is multiplied by the total degrees of freedom,
            `r + fit_intercept`.
        edf_tol : float
            As for `lmbda_grid`, and the reason the attainable range is open at both
            ends: a target within `edf_tol` of `r` or of the floor is refused rather
            than chased to a `lmbda` of 0 or infinity.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the target is not attainable. The range never reaches 0: unpenalised
            directions (`omega == 0`) and the intercept do not shrink however large
            `lmbda` grows.
        """
        if proportion:
            target *= len(self.omega) + self.fit_intercept
        grid = self.lmbda_grid(2, edf_tol)
        # edf decreases in lmbda, so these are the ends of the attainable range --
        # and they are the two evaluations brentq would make anyway.
        lo, hi = self.edf(grid[-1]), self.edf(grid[0])
        if not lo <= target <= hi:
            raise ValueError(
                f"edf {target:g} unattainable on lmbda in [{grid[0]:g}, "
                f"{grid[-1]:g}], which spans edf [{lo:g}, {hi:g}]"
            )
        if grid[0] == grid[-1]:  # nothing penalised, so lo == hi == target
            self.set_lmbda(0.0)
            return self
        rho = brentq(
            lambda rho: self.edf(np.exp(rho)) - target,
            np.log(grid[0]),
            np.log(grid[-1]),
        )
        self.set_lmbda(np.exp(rho))
        return self

    # -- reparametrisation ---------------------------------------------------------

    def _coerce_constraints(self, c):
        """Constraint rows as a `(k, p)` array, checked.

        Shared by the constructor and `add_constraints`, which differ in when a
        constraint is imposed but not in what one is.

        Parameters
        ----------
        c : ndarray, shape (p,) or (k, p), or list of ndarray
            Rows are constraint vectors.

        Returns
        -------
        ndarray, shape (k, p)

        Raises
        ------
        ValueError
            If `fit_intercept` is True, or a row is not `p` long.
        """
        if self.fit_intercept:
            raise ValueError("constraints require fit_intercept=False")
        C = np.atleast_2d(np.asarray(c, dtype=float))
        if C.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"constraint rows must have length {self.X.shape[1]}, got {C.shape[1]}"
            )
        return C

    def add_constraints(self, c):
        """Constrain the fit so that `(X c).T @ (X beta) == 0`.

        In the orthonormal-design basis the constraint is just `ctilde.T @ nu == 0`
        with `ctilde = pushforward(c)`, so it is imposed by restricting to
        `null(ctilde.T)`. The restricted penalty is no longer diagonal, so the cache is
        re-diagonalised -- by `_diagonalise`, as in build step 3 -- and then step 4 runs
        again, the constrained spectrum being a new spectrum. A `lmbda`
        already set therefore buys a slightly different amount of smoothing afterwards;
        `edf` is the coordinate that carries across a constraint, so `set_edf` is how to
        hold the complexity of the fit fixed over one.

        Passing the same rows to the constructor reaches the same cache for one
        eigendecomposition rather than two, and is what a rebuild does; see
        `_build_cache`. This is the route for a constraint that is not known that early.

        Constraints accumulate: each call constrains the current basis, and every row
        imposed is recorded in `constraints`. Clears any fitted state.

        Parameters
        ----------
        c : ndarray, shape (p,) or (k, p), or list of ndarray
            Rows are constraint vectors.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            As `_coerce_constraints` raises.
        """
        C = self._coerce_constraints(c)

        # Orthonormal basis Q of null(Ctilde.T). The cut is floored by the size of the
        # map and of the constraints rather than judged against Ctilde's own largest
        # singular value: a constraint with X @ c == 0 constrains nothing, and its
        # Ctilde is roundoff whose spectrum must not be measured against itself.
        Ctilde = self.pushforward(C.T)
        atol = EPS * np.linalg.norm(self.A_inv, 2) * np.linalg.norm(C, axis=1).max()
        *_, ctilde_null = svd(Ctilde.T, return_null=True, atol=atol)

        # Not `eig(Q.T @ diag(omega) @ Q)`: `omega` spans many decades, and that
        # eigensolver's error is EPS**2 times the largest, so the small weights -- the
        # ones a fit is made of -- come back as roundoff and the relative cut zeroes
        # most of them. Instead the cache is read in its whitened form,
        # `nu = E0 @ z + E+ @ (y / sqrt(omega+))` with penalty `||y||^2`, and
        # re-diagonalised exactly as build step 3 does; see `_diagonalise`.
        pen = self.omega > 0
        scale = 1.0 / np.sqrt(np.where(pen, self.omega, 1.0))
        N = _null_space(Ctilde.T * scale, ctilde_null.shape[0])
        W, omega = _diagonalise(scale[:, None] * N, N[pen])
        self.U = self.U @ W
        self.A = self.A @ W
        self.A_inv = W.T @ self.A_inv
        self.omega = omega
        self._normalize_omega()
        self.constraints = (
            C if self.constraints is None else np.vstack((self.constraints, C))
        )
        self.beta = self.b = None
        return self

    def normalize_beta(self):
        """Rescale `beta` so that `||X beta||^2 / n == 1`.

        In the orthonormal-design basis this is just `||nu||^2 / n == 1`, so the
        factor is `sqrt(n) / ||nu||`.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If `fit_intercept` is True -- rescaling `beta` would invalidate `b` --
            or if the model is not fitted.
        """
        if self.fit_intercept:
            raise ValueError("normalize_beta requires fit_intercept=False")
        if self.beta is None:
            raise ValueError("model is not fitted; call fit first")
        self.beta = self.beta * (
            np.sqrt(len(self.X)) / np.linalg.norm(self.pushforward(self.beta))
        )
        return self


# ---------------------------------------------------------------------------------
# lmbda selection
# ---------------------------------------------------------------------------------


class LmbdaCriterion:
    """Base class for automatic `lmbda` selection.

    Holds a reference to the model plus the two numbers that shape its search grid.
    Subclasses supply a vectorised `criterion`, which is always *minimised*. Where the
    grid runs is not among them: `LinearModel.lmbda_grid` derives that from `omega`.
    """

    def __init__(self, grid_size=300, edf_tol=0.1):
        """
        Parameters
        ----------
        grid_size, edf_tol
            Passed to `LinearModel.lmbda_grid`; see it for both.
        """
        self.grid_size = grid_size
        self.edf_tol = edf_tol
        self.model = None

    def attach(self, model):
        """Bind this criterion to a model.

        Parameters
        ----------
        model : LinearModel
        """
        self.model = model

    def criterion(self, lmbda):
        """Criterion values to be minimised. Implemented by subclasses.

        Parameters
        ----------
        lmbda : ndarray, shape (G,)

        Returns
        -------
        ndarray, shape (G,)
        """
        raise NotImplementedError

    def fit_lmbda(self):
        """Select `lmbda` by minimising `criterion` over the grid.

        The grid is rebuilt on every call rather than cached: `add_constraints`
        changes `omega`, and with it where the grid belongs.

        Returns
        -------
        float
        """
        if self.model is None:
            raise ValueError("criterion is not attached; use set_lmbda_criterion")
        # One call with the whole grid: every criterion here is vectorised over lmbda.
        grid = self.model.lmbda_grid(self.grid_size, self.edf_tol)
        scores = self.criterion(grid)
        # `np.argmin` returns the first NaN, so a criterion undefined anywhere on the
        # grid would otherwise be "minimised" there.
        if np.isnan(scores).all():
            raise FloatingPointError(
                f"{type(self).__name__} is NaN at every lmbda in the grid"
            )
        return grid[np.nanargmin(scores)]


class GCV(LmbdaCriterion):
    """Generalised cross-validation: `n * rss / (n - gamma * edf)**2`."""

    def __init__(self, gamma=1, grid_size=300, edf_tol=0.1):
        """
        Parameters
        ----------
        gamma : float
            Inflation factor on the EDF. Values above 1 penalise complexity harder,
            a common remedy for GCV's tendency to undersmooth.
        """
        super().__init__(grid_size, edf_tol)
        self.gamma = gamma

    def criterion(self, lmbda):
        m, n = self.model, len(self.model.X)
        return n * m.rss(lmbda) / (n - self.gamma * m.edf(lmbda)) ** 2


class REML(LmbdaCriterion):
    """Restricted maximum likelihood, returned as `-2 log L_R` so it is minimised.

    `n_eff * log(D_p) + sum(log(1 + lmbda * omega)) - sum(log(lmbda * omega))`, the
    last sum over penalised coordinates only. `D_p` is the penalised deviance, which
    telescopes to `r_perp + sum((1 - a_i) * ybar_i**2)`, and
    `n_eff = n - #{omega == 0} - fit_intercept`, since unpenalised coordinates are
    fixed effects rather than random ones.
    """

    def criterion(self, lmbda):
        m = self.model
        a = m._shrinkage(lmbda)
        # D_p = rss + lmbda * sum(omega * nu^2), and the penalty term telescopes:
        # (1 - a) - (1 - a)^2 == a * (1 - a).
        D_p = m.rss(lmbda) + (a * (1 - a) * m._ybar() ** 2).sum(-1)

        # log|marginal covariance|, in terms of lmbda * omega rather than of `a`:
        # for a tiny product, 1 - a cancels to zero in floating point and its log
        # would be -inf, whereas log1p / log of the product stay accurate.
        lw = np.asarray(lmbda, dtype=float)[..., None] * m.omega
        penalised = m.omega > 0
        n_eff = len(m.X) - (~penalised).sum() - m.fit_intercept
        return (
            n_eff * np.log(D_p)
            + np.log1p(lw).sum(-1)
            - np.log(lw[..., penalised]).sum(-1)
        )


class AIC(LmbdaCriterion):
    """Akaike information criterion: `n * log(rss / n) + 2 * edf`."""

    def criterion(self, lmbda):
        m, n = self.model, len(self.model.X)
        return n * np.log(m.rss(lmbda) / n) + 2 * m.edf(lmbda)


class BIC(LmbdaCriterion):
    """Bayesian information criterion: `n * log(rss / n) + log(n) * edf`."""

    def criterion(self, lmbda):
        m, n = self.model, len(self.model.X)
        return n * np.log(m.rss(lmbda) / n) + np.log(n) * m.edf(lmbda)


def gcv(gamma=1, grid_size=300, edf_tol=0.1):
    """Construct a `GCV` criterion."""
    return GCV(gamma, grid_size, edf_tol)


def reml(grid_size=300, edf_tol=0.1):
    """Construct a `REML` criterion."""
    return REML(grid_size, edf_tol)


def aic(grid_size=300, edf_tol=0.1):
    """Construct an `AIC` criterion."""
    return AIC(grid_size, edf_tol)


def bic(grid_size=300, edf_tol=0.1):
    """Construct a `BIC` criterion."""
    return BIC(grid_size, edf_tol)
