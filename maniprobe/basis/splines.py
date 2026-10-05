"""B-spline basis on an interval, and a factory for products of them.

A `BSplineBasis` spans one axis, so a derivative here is a plain integer -- the
multi-index form and its ambiguities belong to `TensorProductBasis`, which is where more
than one axis exists.

Knots are placed either by a rule ('uniform', 'quantiles') or given outright as a full
knot vector. Precedence at setup is `knots` > `limits` > `z`. An explicit knot array
makes the basis fully data-independent, so `setup()` may then be called with no argument
at all, which is what prediction-only work or a model space fixed in advance needs.

`bs()` at the bottom returns a `BSplineBasis` for one axis and a
`TensorProductBasis` of B-spline margins for several. It is a factory rather than a
merged class on purpose: a one-dimensional basis and a tensor product are genuinely
different objects. They expose different attributes (`knot_vector` against `bases`),
they read `penalty` differently (a derivative order against a fan-out scheme), and going
from one axis to several introduces a modelling choice -- the fan-out, and in due course
one lambda per axis -- absent in one dimension. Returning the right class keeps
that visible; a single class with a mode flag would hide it behind a branch in every
method.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss

from .base import Basis
from .tensor import TensorProductBasis


class BSplineBasis(Basis):
    """B-splines of the given degree, penalised by a derivative order.

    Attributes
    ----------
    k : int
        Number of basis functions.
    degree : int
    knots
        As supplied: a placement rule, or an explicit knot vector.
    limits : (low, high) or None
        As supplied.
    penalty : int or None
        Order of the penalised derivative.
    knot_vector : ndarray, shape (k + degree + 1,)
        The full knot vector, including repeated boundary knots. Set at setup.
    domain : (low, high)
        Set at setup.
    """

    ndim = 1

    def __init__(
        self,
        k=None,
        degree=3,
        knots="quantiles",
        limits=None,
        penalty=None,
        centre=False,
    ):
        """
        Parameters
        ----------
        k : int, optional
            Number of basis functions. Required, *except* with an explicit knot vector,
            which fixes it at `len(knots) - degree - 1`.
        degree : int
            Spline polynomial degree; 3 is cubic.
        knots : {'uniform', 'quantiles'} or array-like
            A placement rule, or a full knot vector of length `k + degree + 1` including
            the repeated boundary knots. The full vector is required rather than
            interior knots alone because it is the only form that can express a
            non-clamped or repeated-interior knot vector -- and someone placing knots by
            hand is exactly the person who may want a repeated interior knot to drop a
            derivative at a breakpoint.
        limits : (low, high), optional
            Domain boundary. A hyperparameter -- an assertion about the domain -- not
            data. Inferred from `z` at setup if not given.
        penalty : int, optional
            Order of the penalised derivative. None means no penalty.
        centre : bool, optional
            Constrain the basis functions to sum to zero over the setup data; see
            `Basis`. Requires `z` at setup.

        Raises
        ------
        ValueError
            If `k` is neither given nor implied; if `k` or `limits` is given alongside
            an explicit knot vector, which already fixes both; if `k < degree + 1`; or
            if `penalty > degree`.
        """
        self._params = dict(
            k=k, degree=degree, knots=knots, limits=limits, penalty=penalty,
            centre=centre,
        )
        self.degree = degree
        self.knots = knots
        self.limits = limits
        self.penalty = penalty
        self.centre = centre

        if isinstance(knots, str):
            if k is None:
                raise ValueError("k is required unless knots is an explicit vector.")
            self._knot_array = None
        else:
            if k is not None or limits is not None:
                raise ValueError(
                    "an explicit knot vector already fixes k and the domain; "
                    "drop k and limits."
                )
            self._knot_array = np.asarray(knots, dtype=float)
            k = len(self._knot_array) - degree - 1
        self.k = k

        if k < degree + 1:
            raise ValueError(f"k must be at least degree + 1 = {degree + 1}; got {k}.")
        if penalty is not None and penalty > degree:
            raise ValueError(
                f"penalty={penalty} exceeds degree={degree}. The {penalty}th "
                f"derivative of a degree-{degree} spline is identically zero, so the "
                "penalty matrix would be all zeros and lmbda would have no effect."
            )

    # -- setup ---------------------------------------------------------------------

    def _setup(self, z):
        """Place the knots, then integrate the penalty.

        The penalty is built here rather than on demand, so it is fixed state of a
        configured basis. It goes through `_quadrature_gram` rather than `gram_matrix`
        because the latter's `_require_setup` guard is for callers and does not hold
        yet: this basis is not marked set up until `_setup` returns.
        """
        if self._knot_array is not None:
            self.knot_vector = np.array(self._knot_array, dtype=float)
        else:
            self.knot_vector = self._place_knots(z)
        self.domain = (float(self.knot_vector[0]), float(self.knot_vector[-1]))
        if self.penalty is not None:
            self._penalty_matrix = self._quadrature_gram(self.penalty)

    def _place_knots(self, z):
        """The full knot vector implied by `limits`, `z` and the placement rule."""
        if self.limits is not None:
            lo, hi = self.limits
        elif z is not None:
            z = np.asarray(z, dtype=float)
            lo, hi = float(z.min()), float(z.max())
        else:
            raise ValueError(
                "setup() needs data or limits: neither z nor limits was given, and "
                "knots were not specified explicitly."
            )

        # At k == degree + 1 there are no interior knots, and both rules return an
        # empty array that concatenates cleanly -- so this needs no branch of its own.
        n_interior = self.k - self.degree - 1
        if self.knots == "quantiles":
            if z is None:
                raise ValueError(
                    "knots='quantiles' requires data. Pass z to setup(), or use "
                    "knots='uniform' for uniform spacing over limits."
                )
            interior = np.quantile(
                np.asarray(z, dtype=float), np.linspace(0, 1, n_interior + 2)[1:-1]
            )
        elif self.knots == "uniform":
            interior = np.linspace(lo, hi, n_interior + 2)[1:-1]
        else:
            raise ValueError(
                "knots must be 'uniform', 'quantiles' or a knot vector; "
                f"got {self.knots!r}."
            )
        return np.concatenate(
            [np.full(self.degree + 1, lo), interior, np.full(self.degree + 1, hi)]
        )

    # -- matrices ------------------------------------------------------------------

    def _evaluate(self, z, degree, derivative=0):
        """Cox-de Boor, carrying the derivative down with it. No setup guard, because
        `_setup` builds the penalty before this basis is marked configured.

        Two recursions in one, sharing a base case. Writing `B[i,q]` for the i-th basis
        function of degree q on this knot vector,

            B[i,q]   = (z - t[i])/(t[i+q] - t[i]) B[i,q-1]
                     + (t[i+q+1] - z)/(t[i+q+1] - t[i+1]) B[i+1,q-1]
            d B[i,q] = q/(t[i+q] - t[i]) B[i,q-1] - q/(t[i+q+1] - t[i+1]) B[i+1,q-1]

        so the `derivative`-th derivative of the degree-`degree` basis is the
        `derivative - 1`-th of the degree-1-lower one, differenced once. Each level
        makes one recursive call, so this costs `degree` passes over an (n, k) array.

        A term whose denominator is zero is dropped, which is the whole reason this
        exists rather than `scipy.interpolate.BSpline.derivative`: scipy refuses a
        spline with repeated interior knots outright, judging the whole spline rather
        than the span evaluated, though the derivative exists on every span's interior.

        Parameters
        ----------
        z : ndarray, shape (n,)
            Already coerced; `model_matrix` does that once at the entry.
        degree : int
            Recursion variable, not a property of the basis.
        derivative : int, optional

        Returns
        -------
        ndarray, shape (n, len(knot_vector) - degree - 1)
        """
        t = self.knot_vector
        m = degree
        n = len(t) - m - 1

        if derivative > degree:
            # Identically zero: the spline is a polynomial of degree `degree` on each
            # span, so any higher derivative vanishes.
            return np.zeros((len(z), n))

        if degree == 0:
            # Which span each point falls in. The end spans own everything beyond them,
            # which both closes the right endpoint -- the half-open rule would leave
            # z == t[-1] in no span at all -- and makes the recursion extrapolate
            # polynomially rather than return zero, since with the span fixed every
            # formula above is a polynomial in z.
            spans = np.flatnonzero(np.diff(t) > 0)
            B = (z[:, None] >= t[:-1]) & (z[:, None] < t[1:])
            B[:, spans[0]] |= z < t[spans[0]]
            B[:, spans[-1]] |= z >= t[spans[-1] + 1]
            return B.astype(float)

        denom1 = t[m : m + n] - t[0 : n]
        denom2 = t[m + 1 : m + n + 1] - t[1 : n + 1]

        if derivative == 0:
            lower = self._evaluate(z, degree - 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                left = np.where(
                    denom1 > 0,
                    ((z[:, None] - t[0:n]) / denom1) * lower[:, :-1], 0.0
                )
                right = np.where(
                    denom2 > 0,
                    ((t[m + 1 : m + n + 1] - z[:, None]) / denom2) * lower[:, 1:], 0.0
                )
            return left + right

        lower = self._evaluate(z, degree - 1, derivative - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            left = np.where(denom1 > 0, (m / denom1) * lower[:, :-1], 0.0)
            right = np.where(denom2 > 0, (m / denom2) * lower[:, 1:], 0.0)
        return left - right

    def _model_matrix(self, z):
        """Basis functions evaluated at `z`.

        Parameters
        ----------
        z : array-like, shape (n,)
            Points outside the domain are extrapolated: the boundary spans' polynomials
            are continued, rather than the basis dropping to zero.

        Returns
        -------
        ndarray, shape (n, k)
        """
        return self._evaluate(np.asarray(z, dtype=float), self.degree, 0)

    def derivative_matrix(self, z, order=1):
        """The `order`-th derivative of each basis function, evaluated at `z`.

        A method of its own rather than an option on `model_matrix`, because only some
        bases differentiate. Centring does not reach it: the shift is a constant, so the
        derivative of a centred basis function is the derivative of the original.

        Parameters
        ----------
        z : array-like, shape (n,)
        order : int, optional

        Returns
        -------
        ndarray, shape (n, k)
        """
        self._require_setup()
        return self._evaluate(np.asarray(z, dtype=float), self.degree, order)

    def gram_matrix(self, derivative=0):
        """`G(m) = integral of (d^m B).T @ (d^m B)` over the domain.

        Composite Gauss-Legendre, and *exact* rather than merely accurate. Splitting at
        the distinct knots makes the integrand a polynomial on each piece: a product of
        two polynomials of degree `degree - m`, hence of degree at most `2 * degree`. A
        `degree + 2`-node Gauss rule is exact through degree `2 * degree + 3`, so the
        result is the integral, to rounding.

        Parameters
        ----------
        derivative : int, optional

        Returns
        -------
        ndarray, shape (k, k)
            Computed afresh on each call, unlike `penalty_matrix`, which is fixed state:
            a Gram matrix is a query at an order, and no basis can precompute every one.
        """
        self._require_setup()
        return self._quadrature_gram(derivative)

    def _quadrature_gram(self, m):
        breaks = np.unique(self.knot_vector)
        nodes, weights = leggauss(self.degree + 2)
        G = np.zeros((self.k, self.k))
        for a, b in zip(breaks[:-1], breaks[1:]):
            mid, half = (a + b) / 2, (b - a) / 2
            B = self._evaluate(mid + half * nodes, self.degree, m)
            G += (B * (half * weights)[:, None]).T @ B
        return G

    def penalty_matrix(self):
        """The integrated squared `penalty`-th derivative, built at setup.

        `penalty=0` is legitimate but unusual: it makes this the mass matrix, shrinking
        the function toward zero rather than toward smoothness, and is the one case
        where the penalty and the Gram matrix coincide.

        Returns
        -------
        ndarray, shape (k, k)
        """
        self._require_penalty()
        self._require_setup()
        return self._penalty_matrix

    @property
    def null_space_dim(self):
        """How much of the basis the penalty leaves untouched.

        The null space is the functions that are polynomials of degree < `penalty` on
        each span *and* still lie in the spline space, which forces `C^(degree - q)`
        continuity at an interior knot of multiplicity `q`. So each span contributes
        `penalty` dimensions and each interior knot takes back the continuity it
        imposes, at most `penalty` of them and never fewer than none:

            m * n_spans - sum_j clip(min(degree - q_j, m - 1) + 1, 0, m)

        With every interior knot simple this collapses to `penalty`, the familiar
        answer -- polynomials of degree < m are unpenalised and nothing else is. It
        stops collapsing as knots repeat, and the interesting case is intermediate
        rather than extreme: a cubic with a knot tripled and `penalty=2` has null space
        3, not 2 and not 4 -- linear on each side, joined continuously, slope free.

        Needs the knot vector, so this is a property of the configured basis rather
        than of the hyperparameters alone.
        """
        self._require_penalty()
        self._require_setup()
        m, values = self.penalty, np.unique(self.knot_vector)
        _, counts = np.unique(self.knot_vector, return_counts=True)
        interior = counts[(values > values[0]) & (values < values[-1])]
        joins = np.clip(np.minimum(self.degree - interior, m - 1) + 1, 0, m).sum()
        return m * (len(values) - 1) - joins


# ---------------------------------------------------------------------------------
# products of B-spline margins
# ---------------------------------------------------------------------------------


def bs(
    k=None,
    degree=3,
    knots="quantiles",
    limits=None,
    penalty=2,
    fan_out="integral",
):
    """B-splines on one axis or several: one basis, or a tensor product of them.

    `k` says how many axes there are as well as how large each is -- an int is one axis
    and an n-tuple is n of them -- and every other per-axis argument follows it, a
    scalar broadcast to every axis and a list or tuple giving one value per axis. So
    `bs(20, penalty=2)` is a curve and `bs((20, 10), penalty=(2, 1))` is a surface.
    There is no isotropic shorthand: two axes of twenty functions is `k=(20, 20)`, since
    `k` is the argument carrying the number of axes and so cannot also be broadcast.

    `knots` and `limits` are themselves sequences, so for those per-axis means a
    sequence of sequences: `limits=(0, 1)` is one interval on every axis and
    `limits=[(0, 1), (0, 5)]` is one per axis. With `k=None` -- explicit knot vectors,
    each fixing its own `k` -- the number of axes comes from `knots` under that same
    rule, so one vector is one axis and a list of them is one axis each.

    The number of axes is read from `k`, or from `knots` when `k` is None, and never
    from whichever argument happens to arrive as a sequence: naming one argument as the
    authority is what keeps the rule to a sentence.

    Parameters
    ----------
    k : int or tuple of int, optional
        Number of basis functions on each axis, and how many axes there are. None only
        with explicit knot vectors, which fix it themselves.
    degree, knots, limits, penalty
        As for `BSplineBasis`, per axis or broadcast. `penalty` defaults to 2 rather
        than to None: a factory that hands back a ready-to-smooth basis wants the
        second-derivative penalty, where the class itself takes no view.
    fan_out : {'integral', 'identity'}
        As for `TensorProductBasis`. Inert on one axis.

    Returns
    -------
    BSplineBasis on one axis, else TensorProductBasis
        Centred either way, so `setup` requires data even where explicit knots would
        otherwise leave the basis data-independent. `BSplineBasis` still takes `centre`
        for anyone who wants the uncentred basis.
    """
    if k is not None:
        ndim = len(k) if isinstance(k, (list, tuple)) else 1
    else:
        ndim = 1 if _is_single(knots) else len(knots)
    # Centring goes to whichever object is returned and never to a margin inside one:
    # centring each margin does not centre their product, since the column means of a
    # row-wise Kronecker product are covariances rather than zeros.
    margins = [
        BSplineBasis(k=a, degree=b, knots=c, limits=d, penalty=e, centre=ndim == 1)
        for a, b, c, d, e in zip(
            _per_axis(k, ndim, "k"),
            _per_axis(degree, ndim, "degree"),
            _per_axis(knots, ndim, "knots", atom=_is_single),
            _per_axis(limits, ndim, "limits", atom=_is_single),
            _per_axis(penalty, ndim, "penalty"),
        )
    ]
    if ndim == 1:
        return margins[0]
    return TensorProductBasis(*margins, fan_out=fan_out, centre=True)


def _per_axis(value, ndim, name, atom=None):
    """One value per axis: a list or tuple is per-axis, anything else is broadcast.

    `atom` names an exception to that rule, for an argument whose own value is a
    sequence.
    """
    if (atom is not None and atom(value)) or not isinstance(value, (list, tuple)):
        return [value] * ndim
    if len(value) != ndim:
        raise ValueError(
            f"{name} has {len(value)} entries but the basis spans {ndim} axes."
        )
    return list(value)


def _is_single(value):
    """True for one value that is itself a sequence -- a knot vector, an interval, a
    placement rule -- as against one such per axis."""
    return value is None or np.isscalar(value[0])
