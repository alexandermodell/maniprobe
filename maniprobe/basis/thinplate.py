"""Thin plate regression splines: Wood (2003), JRSS-B 65(1), 95-114.

A thin plate spline fits n data with n parameters. Wood's construction truncates it to
rank k in the way that perturbs the original smoothing problem as little as possible:
eigendecompose the matrix E of radial basis functions, keep the k eigenvectors of
largest `|eigenvalue|`, and absorb the polynomial side conditions `T'delta = 0` into the
parameterisation. That single choice of basis simultaneously minimises the worst-case
change in fitted values and the worst-case change in the penalty -- a coincidence Wood
notes is rather special to splines (section 2.1).

What the construction buys
--------------------------
There are no knots to place. The basis functions sit on the covariate points themselves,
so the arbitrariness that dogs multidimensional regression splines disappears, and bases
of different rank are *nested*, which is what makes F-ratio model selection between them
legitimate. The smoother is isotropic: it depends on the covariates only through
Euclidean distance, which is right for spatial coordinates and wrong for covariates in
unrelated units.

What it costs
-------------
The basis functions are radial and globally supported, so there is no mass matrix -- the
integral of `B^T B` over R^d diverges -- and `gram_matrix` raises. A tensor product with
a thin plate margin therefore needs `fan_out='identity'`, which fans the marginal
penalties out against `I` rather than against a mass matrix nobody can supply.

Only the basis itself is available, not its derivatives, so there is no
`derivative_matrix` here as there is on `BSplineBasis`. The radial kernel's multivariate
derivatives are tractable but intricate, and nothing here called for them.

Setup costs a truncated eigendecomposition of an n x n matrix: O(n^3) dense, or O(k n^2)
by Lanczos iteration (Wood, appendix B), which is used when it pays. That truncation is
*by design* rather than by conditioning -- it is Wood's rank reduction, keeping the k
largest eigenpairs of a full-rank matrix -- which is why it does not go through
`linalg.eig`, whose job is to drop eigenvalues that are noise. The side conditions do go
through `linalg.svd`, so their rank is decided by the module-wide `EPS` rule rather than
by a second tolerance of scipy's.

Unlike the other bases here, setup *retains* an n x d array of knots and an n x (k - M)
transformation, because those are the basis -- the analogue of a B-spline's knot vector,
not a copy of the training data. Both costs are why `max_knots` caps n at 2000 by
default, as mgcv does; raise it, set it to None, or pass `knots` to choose the subsample
yourself.

The penalty option
------------------
For every other basis here the penalty is a modelling choice made after the basis
exists. Here it is not: J_md builds E, so the wiggliness order m is constructor state
and the only penalty the basis can express is the one it was built from. `penalty=m` is
accepted (so that `penalty=2` still selects comparable smoothness across bases) and
`penalty != m` is rejected rather than quietly reinterpreted. `penalty=None` leaves the
same basis unpenalised -- Wood's pure regression spline, where flexibility is chosen by
k rather than by lambda.
"""

from itertools import product
from math import factorial

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from scipy.spatial.distance import cdist
from scipy.special import gamma

from ..linalg import svd

from .base import Basis

#: mgcv's ``max.knots`` and ``seed``.
_DEFAULT_MAX_KNOTS = 2000
_DEFAULT_SEED = 1
#: Below this many knots a full eigendecomposition beats Lanczos iteration.
_DENSE_MAX = 800


class ThinPlateBasis(Basis):
    """Rank-k thin plate regression spline in d dimensions.

    Setup places the basis functions at the distinct rows of `z`; ties are collapsed,
    which is Wood's prescription in section 2.1 and is also what keeps E non-singular.
    Note that subsampling thins the *basis*, not the fit: every row of `z` still gets a
    row of the model matrix.

    Attributes
    ----------
    k, d, m : int
        Basis dimension, number of covariate axes (`ndim` is the same number), and the
        order of the wiggliness penalty.
    penalty : int or None
        Equal to `m`, or None for an unpenalised basis.
    knot_points : ndarray, shape (n_knots, d)
        The centres of the radial basis functions. Set at setup.
    """

    def __init__(
        self,
        k,
        d=None,
        m=None,
        knots=None,
        penalty=None,
        centre=False,
        max_knots=_DEFAULT_MAX_KNOTS,
        seed=_DEFAULT_SEED,
    ):
        """
        Parameters
        ----------
        k : int
            Basis dimension: the number of retained eigenvectors, and the number of
            columns of the model matrix. Must exceed the null space dimension M.
        d : int, optional
            Number of covariates the smoother is a function of -- `ndim` in the Basis
            interface, but named for Wood's notation because it appears in every
            constraint here (`2m > d`, `M = C(m+d-1, d)`). Inferred from `knots` when
            those are given, otherwise 1.
        m : int, optional
            Order of the wiggliness penalty J_md. `2m > d` is required, and the default
            goes one better: the smallest m with `2m > d + 1`, which is what mgcv uses.
            The stricter rule is about smoothness rather than well-posedness -- eta
            grows like `r^(2m-d)`, so the merely well-posed `2m - d = 1` puts a cone at
            every knot. It gives the familiar cubic smoothing spline at `d = 1` and the
            familiar thin plate spline at `d = 2` without a special case.
        knots : array-like, shape (n_knots, d), optional
            Centres for the radial basis functions. Given explicitly the basis is fully
            data-independent, so `setup()` may be called with no argument; this is also
            how to work from a subsample of a large data set, which Wood suggests in
            section 2.2. Duplicated rows are dropped.
        penalty : int, optional
            Order of the penalised derivative. Must equal `m`, since that is the only
            penalty this basis can express; supplies `m` when `m` is not given. None
            leaves the basis unpenalised.
        centre : bool, optional
            Constrain the basis functions to sum to zero over the setup data; see
            `Basis`. Requires `z` at setup, even where explicit `knots` would otherwise
            make the basis data-independent.
        max_knots : int or None, optional
            Cap on the number of knots taken from data, 2000 as in mgcv. Above it, that
            many distinct points are sampled without replacement. None keeps every
            distinct point, at O(n^3) or O(k n^2). Ignored when `knots` are given, which
            is the deliberate way to choose the subsample yourself.
        seed : optional
            Seeds the knot subsample and the Lanczos starting vector, 1 as in mgcv, so
            that a basis is reproducible across refits.

        Raises
        ------
        ValueError
            If `d` contradicts an explicit knot array; if `2m <= d`; if `penalty != m`;
            or if `k <= M`.
        """
        self._params = dict(
            k=k, d=d, m=m, knots=knots, penalty=penalty, centre=centre,
            max_knots=max_knots, seed=seed,
        )
        self.centre = centre

        knot_array = None
        if knots is not None:
            knot_array = np.asarray(knots, dtype=float)
            if knot_array.ndim == 1:
                knot_array = knot_array[:, None]
            knot_array = _unique_rows(knot_array)
            if d is not None and d != knot_array.shape[1]:
                raise ValueError(
                    f"d={d} contradicts the knot array, which has "
                    f"{knot_array.shape[1]} columns and so fixes d."
                )
            d = knot_array.shape[1]
        self.d = 1 if d is None else d
        self.ndim = self.d
        self.knots = knots
        self._knot_array = knot_array

        if m is None:
            m = penalty if penalty is not None else (self.d + 1) // 2 + 1
        self.m = m
        if 2 * self.m <= self.d:
            raise ValueError(
                f"m={self.m} does not satisfy 2m > d = {self.d}. Below "
                f"m={self.d // 2 + 1} the wiggliness penalty J_md has no "
                "finite-dimensional minimiser, so there is no spline to approximate."
            )
        if penalty is not None and penalty != self.m:
            raise ValueError(
                f"penalty={penalty} contradicts m={self.m}. A thin plate regression "
                "spline can express exactly one penalty -- the J_md that built its "
                f"basis -- so pass m={penalty} to penalise that order instead."
            )
        self.penalty = penalty

        self._powers = _monomial_powers(self.d, self.m)
        self._M = len(self._powers)
        self.k = k
        if k <= self._M:
            raise ValueError(
                f"k={k} does not exceed the null space dimension M={self._M} "
                f"(polynomials in {self.d} variables of degree < {self.m}). A rank-M "
                "basis is the unpenalised polynomial alone, with no wiggly directions "
                "left to smooth."
            )

        self.max_knots = max_knots
        self.seed = seed
        self._coefficient = _eta_coefficient(self.m, self.d)
        self._exponent = (2 * self.m - self.d) / 2.0
        self._log_form = self.d % 2 == 0

    # -- setup ---------------------------------------------------------------------

    def _setup(self, z):
        """Place the basis functions, truncate the eigenbasis, then build the penalty.

        Wood's appendix A, with the null space of the side conditions obtained from
        `linalg.svd` rather than by QR, so the rank is decided by the same `EPS` rule as
        everything else in the project.
        """
        if self._knot_array is not None:
            # The user fixed the centres; z is deliberately ignored.
            knots = np.array(self._knot_array, dtype=float)
        elif z is not None:
            knots = _unique_rows(self._as_points(z))
            if self.max_knots is not None and len(knots) > self.max_knots:
                # Silent, as in mgcv, and consistent with ignoring z when knots were
                # given: this thins the basis, not the fit, and the caller passing z may
                # not be the user.
                rng = np.random.default_rng(self.seed)
                taken = rng.choice(len(knots), self.max_knots, replace=False)
                knots = knots[np.sort(taken)]
        else:
            raise ValueError(
                "setup() needs data: a thin plate regression spline places its basis "
                "functions at the covariate points themselves. Pass z, or give knots "
                "explicitly to fix the basis in advance."
            )
        if self.k > len(knots):
            raise ValueError(
                f"k={self.k} exceeds the {len(knots)} knots available. The basis "
                "truncates a spline with one parameter per knot, so it cannot have "
                "higher rank than that spline."
            )

        self.knot_points = knots
        self._centre = knots.mean(axis=0)
        E = self._radial(knots, knots)
        T = _polynomial(knots - self._centre, self._powers)

        eigenvalues, U = _truncated_eig(E, self.k, self.seed)
        # T' U_k delta_k = 0: M side conditions on the k retained coordinates.
        *_, side_null = svd((U.T @ T).T, return_null=True)
        Z = side_null.T
        if Z.shape[1] != self.k - self._M:
            raise ValueError(
                "The polynomial part of the basis is not identifiable: the knots are "
                f"degenerate (collinear, say) in {self.d} dimensions, so the "
                f"{self._M} side conditions have rank {self.k - Z.shape[1]}. Supply "
                "knots that span the domain."
            )

        self._transform = U @ Z
        # Z' D_k Z padded with zeros, Wood appendix A(d). Not an approximation to J_md:
        # it is J_md exactly, for every function the truncated basis can represent.
        S = np.zeros((self.k, self.k))
        S[: self.k - self._M, : self.k - self._M] = Z.T @ (eigenvalues[:, None] * Z)
        self._penalty_matrix = S

    # -- matrices ------------------------------------------------------------------

    def _model_matrix(self, z):
        """Design matrix `(E_z U_k Z_k, T_z)`, Wood equation (7).

        At the knots themselves this is `(U_k D_k Z_k, T)`, since `E U_k = U_k D_k`.

        Parameters
        ----------
        z : array-like, shape (n, d)

        Returns
        -------
        ndarray, shape (n, k)
        """
        z = self._as_points(z)
        return np.concatenate(
            [
                self._radial(z, self.knot_points) @ self._transform,
                _polynomial(z - self._centre, self._powers),
            ],
            axis=1,
        )

    def gram_matrix(self, derivative=0):
        raise NotImplementedError(
            "ThinPlateBasis has no mass matrix: its basis functions are radial and "
            "globally supported, so the integral of B^T B over R^d diverges. In a "
            "tensor product use fan_out='identity', which fans the marginal penalties "
            "out against I."
        )

    def penalty_matrix(self):
        """`Z_k' D_k Z_k` padded with zeros, built at setup.

        Returns
        -------
        ndarray, shape (k, k)
        """
        self._require_penalty()
        self._require_setup()
        return self._penalty_matrix

    @property
    def null_space_dim(self):
        """Polynomials of degree < m are unpenalised: M = C(m + d - 1, d)."""
        self._require_penalty()
        return self._M

    # -- kernel --------------------------------------------------------------------

    def _as_points(self, z):
        z = np.asarray(z, dtype=float)
        if self.d == 1 and z.ndim != 2:
            z = z.reshape(-1, 1)
        if z.ndim != 2 or z.shape[1] != self.d:
            raise ValueError(f"z must have shape (n, {self.d}); got {np.shape(z)}.")
        return z

    def _radial(self, z, knots):
        """`eta(||z_i - knot_j||)`, Wood section 2. Shape (len(z), len(knots)).

        Written as a function of `s = r^2` rather than of `r`: `f(s) = s^a`, or
        `s^a log(s) / 2` in even dimensions, since `log r = log s / 2`.

        `s` is clamped at zero because for odd `d` the exponent `(2m - d) / 2` is a
        half-integer, so a roundoff-negative squared distance would give `nan` rather
        than a merely inaccurate number. At `s == 0` the kernel is 0, which is the
        limit: `2m > d`, so `r^(2m-d) log r -> 0`.
        """
        s = np.maximum(cdist(z, knots, "sqeuclidean"), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = s**self._exponent
            if self._log_form:
                f = f * np.log(s) / 2
        return self._coefficient * np.where(s == 0.0, 0.0, f)


# ---------------------------------------------------------------------------------
# the ready-to-smooth basis
# ---------------------------------------------------------------------------------


def tps(
    k, d=None, penalty=2, knots=None, max_knots=_DEFAULT_MAX_KNOTS, seed=_DEFAULT_SEED
):
    """A centred, penalised thin plate regression spline basis: `bs`'s counterpart.

    The class takes no view on either, and a caller fitting a smoother wants both: an
    uncentred basis reaches the constant, and one without a penalty is Wood's pure
    regression spline, which `lmbda` then cannot touch. `m` is not an argument, since
    the only penalty this basis can express is the one it was built from, so `penalty`
    is the order and the two cannot disagree.

    Parameters
    ----------
    k : int
        Basis dimension, as for `ThinPlateBasis`.
    d : int, optional
        Number of covariates, as for `ThinPlateBasis`: inferred from `knots` when those
        are given, otherwise 1.
    penalty : int
        Order m of the penalty J_md. 2 rather than the class's `2m > d + 1` rule: it is
        the usual choice at `d == 1`, where the result is the cubic smoothing spline,
        and at `d == 2`, where it is the thin plate spline proper -- and it is what `bs`
        defaults to, so `penalty=2` means comparable smoothness across the two. From
        `d == 4` it fails `2m > d` and must be raised.
    knots, max_knots, seed
        As for `ThinPlateBasis`.

    Returns
    -------
    ThinPlateBasis
        Centred, so `setup` requires data even where explicit `knots` would otherwise
        leave the basis data-independent.
    """
    return ThinPlateBasis(
        k,
        d=d,
        knots=knots,
        penalty=penalty,
        centre=True,
        max_knots=max_knots,
        seed=seed,
    )


def _eta_coefficient(m, d):
    """Wood's eta_md, section 2, without its `r` factor."""
    if d % 2 == 0:
        sign = (-1.0) ** (m + 1 + d // 2)
        return sign / (
            2.0 ** (2 * m - 1)
            * np.pi ** (d / 2)
            * factorial(m - 1)
            * factorial(m - d // 2)
        )
    return gamma(d / 2 - m) / (2.0 ** (2 * m) * np.pi ** (d / 2) * factorial(m - 1))


def _monomial_powers(d, m):
    """Exponents of the monomials spanning polynomials of degree < m."""
    powers = [e for e in product(range(m), repeat=d) if sum(e) < m]
    return sorted(powers, key=lambda e: (sum(e), e))


def _polynomial(centred, powers):
    """Each monomial evaluated at centred coordinates. Shape (len(centred),
    len(powers)).

    `0 ** 0 == 1`, so the constant column falls out of the same expression as the rest.
    """
    return np.stack(
        [np.prod(centred ** np.array(e), axis=1) for e in powers], axis=1
    )


def _unique_rows(points):
    """Distinct rows, in the order they first appear."""
    _, index = np.unique(points, axis=0, return_index=True)
    return points[np.sort(index)]


def _truncated_eig(matrix, k, seed):
    """The k eigenpairs of largest `|eigenvalue|`, largest first.

    Lanczos iteration (Wood, appendix B) gets these in O(k n^2) rather than the O(n^3)
    of
    a full decomposition, which is most of what makes the method usable on large data.
    It
    only pays when k is a small fraction of n, and it needs a starting vector unlikely
    to be orthogonal to an eigenvector -- random, but from the basis's own seed, so that
    a basis is reproducible across refits.
    """
    n = len(matrix)
    if n <= _DENSE_MAX or 2 * k > n:
        eigenvalues, vectors = np.linalg.eigh(matrix)
    else:
        start = np.random.default_rng(seed).standard_normal(n)
        try:
            eigenvalues, vectors = eigsh(matrix, k=k, which="LM", v0=start)
        except ArpackNoConvergence:
            eigenvalues, vectors = np.linalg.eigh(matrix)
    order = np.argsort(np.abs(eigenvalues))[::-1][:k]
    return eigenvalues[order], vectors[:, order]
