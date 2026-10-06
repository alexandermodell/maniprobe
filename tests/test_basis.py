"""Characterisation tests for the `basis` package.

These pin the behaviour of the package as it stands, ahead of a rewrite. Where a value
can be checked against something derived independently of the code, it is: scipy's
`BSpline` used directly, the Cox-de Boor difference formula for derivatives, QUADPACK
and a far finer Gauss rule for the Gram matrices, explicit Kronecker products for the
tensor product, and hand-written radial kernels for the thin plate spline. Only the
end-to-end fits are pinned as literals, and those are the values a rewrite most has to
reproduce.

The Gram tests are the sharpest instrument here. `BSplineBasis` claims its composite
Gauss-Legendre rule is *exact*, not merely accurate, so those are compared at 1e-12
relative rather than at some quadrature tolerance -- a rewrite that quietly drops to an
approximate rule fails them.

Two conventions exist so the file survives the rewrite:

- every constructor argument is passed by keyword, since some signatures are being
  reordered;
- `TensorProductBasis(penalty=...)` is being renamed to `fan_out=`, so it appears in
  exactly one place, `tensor_identity` below.
"""

from math import comb

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.interpolate import BSpline, PPoly
from scipy.linalg import null_space
from scipy.optimize import brentq
from scipy.spatial.distance import cdist

from maniprobe import linear_model as lm
from maniprobe.basis import (
    BSplineBasis,
    TensorProductBasis,
    ThinPlateBasis,
    bs,
    tps,
)
from maniprobe.basis.thinplate import _eta_coefficient
from maniprobe.linear_smoother import LinearSmoother

N = 40


# ---------------------------------------------------------------------------------
# fixtures and constructors
# ---------------------------------------------------------------------------------


def bspline(degree=3, k=8, penalty=2, knots="uniform", limits=(0.0, 1.0)):
    """A fresh unconfigured basis on [0, 1]. setup() is once-only, so no test may
    share one; every case builds its own."""
    return BSplineBasis(degree=degree, k=k, knots=knots, limits=limits, penalty=penalty)


def tensor_identity(*bases):
    """The identity fan-out, isolated so the rename is one edit."""
    return TensorProductBasis(*bases, fan_out="identity")


def curve_data(seed=0, n=N):
    rng = np.random.default_rng(seed)
    z = np.sort(rng.uniform(0.0, 1.0, n))
    return z, np.sin(6 * z) + 0.1 * rng.normal(size=n)


def surface_data(seed=1, n=N):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.0, 1.0, (n, 2))
    return z, np.sin(4 * z[:, 0]) * np.cos(3 * z[:, 1]) + 0.1 * rng.normal(size=n)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def z2():
    return surface_data()[0]


def close(got, ref, tol=1e-12):
    """Agreement scaled by the reference's own magnitude, so a near-zero entry is
    judged against the matrix it sits in rather than against itself."""
    assert_allclose(got, ref, rtol=tol, atol=tol * np.max(np.abs(ref), initial=1.0))


def held_arrays(basis):
    """Every ndarray a basis retains, following sub-bases: what `_freeze` covers."""
    out = [v for v in vars(basis).values() if isinstance(v, np.ndarray)]
    for sub in getattr(basis, "bases", ()):
        out += held_arrays(sub)
    return out


# ---------------------------------------------------------------------------------
# independent references
# ---------------------------------------------------------------------------------


def scipy_columns(t, degree, m=0):
    """Each basis function as its own scipy spline: the library used directly, one
    unit coefficient vector at a time, rather than through a coefficient matrix."""
    k = len(t) - degree - 1
    out = []
    for i in range(k):
        s = BSpline(t, np.eye(k)[i], degree, extrapolate=True)
        out.append(s.derivative(m) if m else s)
    return out


def scipy_matrix(t, degree, z, m=0):
    return np.column_stack([s(z) for s in scipy_columns(t, degree, m)])


def ppoly_matrix(basis, z, m=0):
    """The same thing through `PPoly`, which holds per-interval polynomial coefficients
    and so differentiates without scipy's global continuity check.

    This is the only independent reference available for a repeated interior knot:
    `BSpline.derivative` refuses such a spline outright, judging the whole spline rather
    than the span being evaluated."""
    t, degree, k = basis.knot_vector, basis.degree, basis.k
    out = np.empty((len(z), k))
    for i in range(k):
        pp = PPoly.from_spline((t, np.eye(k)[i], degree), extrapolate=True)
        out[:, i] = (pp.derivative(m) if m else pp)(z)
    return out


def difference_derivative(t, degree, m, z):
    """d^m/dz^m of the degree-`degree` B-splines on `t`, by the Cox-de Boor difference
    formula B'_{i,p} = p (B_{i,p-1}/(t_{i+p} - t_i) - B_{i+1,p-1}/(t_{i+p+1} - t_{i+1})).

    Independent of scipy's `.derivative()`: it reads the lower-degree basis on the same
    knot vector and differences it, which is where the recursion is stated in the
    literature rather than where scipy implements it.

    NOTE: this is now the same recursion `BSplineBasis._evaluate` uses, so it is no
    longer independent of the code under test -- it agrees by construction, and would
    keep agreeing if both were wrong the same way. It survives as a cross-check on the
    *arrangement* (its base case calls `model_matrix` at derivative 0, so a mistake in
    the descent still shows). The independent references for derivatives are
    `scipy_matrix` wherever scipy will differentiate, `ppoly_matrix` where it will not,
    and the Gram tests, which reach scipy through `fine_gram` and `quad_gram`.
    """
    if m == 0:
        return BSplineBasis(degree=degree, knots=t, penalty=None).setup().model_matrix(z)
    lower = difference_derivative(t, degree - 1, m - 1, z)
    out = np.zeros((np.size(z), len(t) - degree - 1))
    for i in range(out.shape[1]):
        for j, sign in ((i, 1.0), (i + 1, -1.0)):
            width = t[j + degree] - t[j]
            if width > 0:
                out[:, i] += sign * degree * lower[:, j] / width
    return out


def composite_nodes(basis, n_nodes):
    """Gauss-Legendre nodes and weights on every knot span of `basis`."""
    t = np.unique(basis.knot_vector)
    gp, gw = leggauss(n_nodes)
    mid, half = 0.5 * (t[:-1] + t[1:]), 0.5 * (t[1:] - t[:-1])
    return (mid[:, None] + np.outer(half, gp)).ravel(), np.outer(half, gw).ravel()


def fine_gram(basis, m, n_nodes=40):
    """The Gram matrix by a Gauss rule an order of magnitude finer than the code's."""
    pts, wts = composite_nodes(basis, n_nodes)
    B = basis.derivative_matrix(pts, m)
    return (B * wts[:, None]).T @ B


def quad_gram(basis, m):
    """The Gram matrix entry by entry through QUADPACK, breaking at the interior
    knots. Shares no line of code with the package: a different integrator over
    splines built by scipy directly."""
    t = basis.knot_vector
    cols = scipy_columns(t, basis.degree, m)
    interior = np.unique(t[1:-1])
    G = np.zeros((basis.k, basis.k))
    for i in range(basis.k):
        for j in range(i, basis.k):
            f = lambda x, i=i, j=j: cols[i](x) * cols[j](x)
            G[i, j] = G[j, i] = quad(f, t[0], t[-1], points=interior, limit=200)[0]
    return G


def product_quadrature(basis, n_nodes=20):
    """Tensor Gauss-Legendre nodes and weights over the product domain of a tensor
    product of B-spline margins. Exact on the integrands here."""
    axes = [composite_nodes(margin, n_nodes) for margin in basis.bases]
    grids = np.meshgrid(*[p for p, _ in axes], indexing="ij")
    weights = axes[0][1]
    for _, w in axes[1:]:
        weights = np.outer(weights, w).ravel()
    return np.column_stack([g.ravel() for g in grids]), weights


def eta(r, d):
    """Wood's radial kernel at m = 2, written out for the two standard cases:
    r^3 / 12 on the line, and the r^2 log r of the thin plate spline in the plane."""
    with np.errstate(divide="ignore", invalid="ignore"):
        value = r**3 / 12.0 if d == 1 else r**2 * np.log(r) / (8.0 * np.pi)
    return np.where(r == 0.0, 0.0, value)


def radial_coefficients(basis, knots):
    """delta = E^-1 (radial block at the knots), i.e. U_k Z_k, recovered through the
    hand-written kernel. The model matrix is (E_z delta, T_z) at every z, so this is
    the one unknown; everything else about the kernel is then testable."""
    wiggly = basis.k - basis.null_space_dim
    E = eta(cdist(knots, knots), basis.d)
    return np.linalg.solve(E, basis.model_matrix(knots)[:, :wiggly])


# ---------------------------------------------------------------------------------
# BSplineBasis -- the model matrix
# ---------------------------------------------------------------------------------


class TestBSplineModelMatrix:
    @pytest.mark.parametrize("degree,k", [(0, 4), (1, 5), (2, 6), (3, 8), (5, 9)])
    def test_matches_scipy_used_directly(self, degree, k):
        b = bspline(degree=degree, k=k, penalty=None).setup()
        z = np.linspace(0.0, 1.0, 37)
        close(b.model_matrix(z), scipy_matrix(b.knot_vector, degree, z), tol=1e-14)

    def test_extrapolates_outside_the_domain(self):
        """`model_matrix` builds its spline with extrapolate=True, so points beyond
        the boundary continue the end polynomials rather than returning zero."""
        b = bspline(degree=3, k=8, penalty=None).setup()
        z = np.array([-0.3, -0.05, 1.05, 1.4])
        X = b.model_matrix(z)
        close(X, scipy_matrix(b.knot_vector, 3, z), tol=1e-13)
        assert np.abs(X).max() > 1.0  # genuinely extrapolating, not clamped

    @pytest.mark.parametrize("degree,k", [(1, 5), (2, 6), (3, 8), (4, 10)])
    def test_partition_of_unity(self, degree, k):
        """A full clamped basis sums to 1 across the domain, endpoints included."""
        b = bspline(degree=degree, k=k, penalty=None).setup()
        X = b.model_matrix(np.linspace(0.0, 1.0, 101))
        close(X.sum(axis=1), np.ones(101), tol=1e-13)
        assert (X >= -1e-15).all()

    def test_first_derivative_sums_to_zero(self):
        b = bspline(degree=3, k=9, penalty=None).setup()
        D = b.derivative_matrix(np.linspace(0.0, 1.0, 51), 1)
        close(D.sum(axis=1), np.zeros(51), tol=1e-12)

    @pytest.mark.parametrize("degree,k", [(1, 5), (2, 6), (3, 8)])
    def test_derivatives_match_the_difference_formula(self, degree, k):
        b = bspline(degree=degree, k=k, penalty=None, limits=(-1.0, 2.0)).setup()
        z = np.linspace(-0.97, 1.97, 29)  # away from the knots: degree 0 is half-open
        for m in range(degree + 1):
            ref = difference_derivative(b.knot_vector, degree, m, z)
            close(b.derivative_matrix(z, m), ref, tol=1e-9)

    def test_derivative_at_the_degree_is_piecewise_constant(self):
        """order == degree leaves a step function, so it is constant strictly inside
        each span and generally discontinuous across knots."""
        b = bspline(degree=3, k=7, penalty=None).setup()
        spans = np.unique(b.knot_vector)
        inside = np.column_stack([spans[:-1] + f * np.diff(spans) for f in (0.25, 0.75)])
        D = b.derivative_matrix(inside.ravel(), 3).reshape(len(spans) - 1, 2, 7)
        close(D[:, 0], D[:, 1], tol=1e-10)
        assert np.abs(np.diff(D[:, 0], axis=0)).max() > 1.0

    @pytest.mark.parametrize("m", [4, 5, 9])
    def test_derivative_above_the_degree_is_exactly_zero(self, m):
        b = bspline(degree=3, k=7, penalty=None).setup()
        D = b.derivative_matrix(np.linspace(0.0, 1.0, 11), m)
        assert D.shape == (11, 7)
        assert (D == 0.0).all()

    def test_eval_and_function_agree_with_the_matrix(self, rng):
        b = bspline(degree=3, k=8, penalty=2).setup()
        c = rng.normal(size=8)
        z = np.linspace(0.0, 1.0, 13)
        close(b.eval(z, c), b.model_matrix(z) @ c)
        close(b.function(c)(z), b.model_matrix(z) @ c)


# ---------------------------------------------------------------------------------
# BSplineBasis -- Gram and penalty
# ---------------------------------------------------------------------------------


class TestBSplineGram:
    @pytest.mark.parametrize("degree,k", [(1, 5), (2, 6), (3, 8), (4, 10)])
    def test_quadrature_is_exact_against_a_finer_rule(self, degree, k):
        """The rule is degree + 2 Gauss points per span, which integrates the product
        of two degree-`degree` polynomials exactly. A 40-point rule must therefore
        agree to rounding, not merely to quadrature tolerance."""
        b = bspline(degree=degree, k=k, penalty=None, limits=(0.0, 2.0)).setup()
        for m in range(degree + 1):
            close(b.gram_matrix(m), fine_gram(b, m), tol=1e-12)

    @pytest.mark.parametrize("m", [0, 1, 2])
    def test_matches_quadpack(self, m):
        b = bspline(degree=3, k=7, penalty=None, limits=(0.0, 2.0)).setup()
        close(b.gram_matrix(m), quad_gram(b, m), tol=1e-10)

    def test_matches_quadpack_on_uneven_spans(self):
        """Quantile knots leave spans of very different widths, which is where a rule
        keyed to the wrong interval would show up."""
        z = curve_data()[0]
        b = BSplineBasis(degree=3, k=7, knots="quantiles", penalty=2).setup(z)
        close(b.gram_matrix(2), quad_gram(b, 2), tol=1e-10)

    def test_gram_at_order_zero_is_the_mass_matrix(self):
        b = bspline(degree=3, k=8, penalty=None).setup()
        G = b.gram_matrix()
        close(G, b.gram_matrix(0))
        close(G.sum(), 1.0)  # partition of unity integrated over [0, 1]
        assert np.linalg.matrix_rank(G) == 8

    def test_gram_above_the_degree_is_zero(self):
        b = bspline(degree=2, k=6, penalty=None).setup()
        assert (b.gram_matrix(3) == 0.0).all()

    def test_default_derivative_is_order_zero(self):
        b = bspline(degree=3, k=6, penalty=None).setup()
        close(b.gram_matrix(), b.gram_matrix(0))

    @pytest.mark.parametrize("penalty", [0, 1, 2, 3])
    def test_penalty_is_the_gram_at_the_penalty_order(self, penalty):
        b = bspline(degree=3, k=10, penalty=penalty).setup()
        assert np.array_equal(b.penalty_matrix(), b.gram_matrix(penalty))

    @pytest.mark.parametrize(
        "degree,k,penalty", [(3, 10, 0), (3, 10, 1), (3, 10, 2), (3, 10, 3), (5, 12, 4)]
    )
    def test_penalty_rank_leaves_the_polynomials_unpenalised(self, degree, k, penalty):
        b = bspline(degree=degree, k=k, penalty=penalty).setup()
        assert b.null_space_dim == penalty
        assert np.linalg.matrix_rank(b.penalty_matrix()) == k - penalty

    def test_penalty_annihilates_exactly_the_low_polynomials(self):
        """The null space is spanned by the coefficient vectors of 1, z, ..., z^(m-1),
        which is the statement `null_space_dim == penalty` is making."""
        b = bspline(degree=3, k=9, penalty=2).setup()
        z = np.linspace(0.0, 1.0, 60)
        X = b.model_matrix(z)
        P = b.penalty_matrix()
        for power in (0, 1):
            c = np.linalg.lstsq(X, z**power, rcond=None)[0]
            close(X @ c, z**power, tol=1e-10)
            assert c @ P @ c < 1e-20
        c = np.linalg.lstsq(X, z**2, rcond=None)[0]
        assert c @ P @ c > 1.0

    def test_penalty_without_one_raises(self):
        b = bspline(degree=3, k=8, penalty=None).setup()
        with pytest.raises(ValueError, match="without a penalty"):
            b.penalty_matrix()
        with pytest.raises(ValueError, match="without a penalty"):
            b.null_space_dim

    def test_returned_matrices_are_the_callers_to_keep(self):
        """The cached Gram matrix is handed out as a copy, so a caller writing to one
        cannot corrupt the frozen basis."""
        b = bspline(degree=3, k=6, penalty=2).setup()
        before = b.gram_matrix(2)
        before[0, 0] = 99.0
        assert b.gram_matrix(2)[0, 0] != 99.0
        assert b.penalty_matrix()[0, 0] != 99.0


# ---------------------------------------------------------------------------------
# BSplineBasis -- knot placement
# ---------------------------------------------------------------------------------


class TestBSplineKnots:
    def test_uniform_spacing_over_limits(self):
        b = BSplineBasis(
            degree=2, k=6, knots="uniform", limits=(-1.0, 3.0), penalty=1
        ).setup()
        close(b.knot_vector, [-1, -1, -1, 0, 1, 2, 3, 3, 3], tol=1e-15)
        assert b.domain == (-1.0, 3.0)

    def test_uniform_spacing_over_the_range_of_z(self):
        b = BSplineBasis(degree=3, k=6, knots="uniform", penalty=2).setup(
            np.array([0.4, 2.0, 1.1, 4.4])
        )
        interior = [1.7333333333333334, 3.0666666666666669]
        close(b.knot_vector, [0.4] * 4 + interior + [4.4] * 4)
        assert b.domain == (0.4, 4.4)

    def test_quantile_knots_are_the_quantiles_of_z(self):
        z = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0])
        b = BSplineBasis(degree=3, k=6, knots="quantiles", penalty=2).setup(z)
        interior = np.quantile(z, np.linspace(0, 1, 4)[1:-1])
        close(b.knot_vector, np.concatenate([[0.0] * 4, interior, [1.0] * 4]))
        close(interior, [0.25, 0.6333333333333333])

    def test_quantile_knots_follow_the_data_not_the_range(self):
        """Half the mass in the left tenth: the interior knots must crowd there."""
        z = np.concatenate([np.linspace(0.0, 0.1, 50), np.linspace(0.9, 1.0, 50)])
        b = BSplineBasis(degree=3, k=8, knots="quantiles", penalty=2).setup(z)
        interior = b.knot_vector[4:-4]
        assert (interior[:2] < 0.11).all() and (interior[-2:] > 0.89).all()

    def test_an_explicit_knot_vector_is_taken_verbatim(self):
        t = [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]
        b = BSplineBasis(degree=1, knots=t, penalty=1).setup()
        assert b.k == 4
        close(b.knot_vector, t, tol=1e-15)
        assert b.domain == (0.0, 1.0)

    def test_an_explicit_knot_vector_ignores_z(self):
        t = np.concatenate([[0.0] * 4, [0.3, 0.7], [1.0] * 4])
        a = BSplineBasis(degree=3, knots=t, penalty=2).setup()
        b = BSplineBasis(degree=3, knots=t, penalty=2).setup(np.linspace(-5.0, 5.0, 20))
        assert np.array_equal(a.knot_vector, b.knot_vector)

    def test_a_repeated_interior_knot_drops_a_derivative(self):
        """The reason the full vector is accepted rather than interior knots alone:
        multiplicity `degree` leaves the curve continuous but kinks its slope."""
        t = np.array([0.0] * 4 + [0.5, 0.5, 0.5] + [1.0] * 4)
        b = BSplineBasis(degree=3, knots=t, penalty=2).setup()
        below, above = np.array([0.5 - 1e-7]), np.array([0.5 + 1e-7])
        slope = b.derivative_matrix(below, 1) - b.derivative_matrix(above, 1)
        assert np.abs(slope).max() > 1.0
        close(b.model_matrix(below), b.model_matrix(above), tol=1e-5)

    @pytest.mark.parametrize("multiplicity", [1, 2, 3, 4])
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_repeated_interior_knots_still_differentiate(self, multiplicity, order):
        """What the bespoke recursion buys over scipy: a repeated interior knot drops a
        term rather than refusing the whole spline. scipy's `.derivative()` cannot
        supply a reference here at all, so this goes through `PPoly`, which converts to
        per-interval polynomials and so carries no global continuity check."""
        t = np.array([0.0] * 4 + [0.5] * multiplicity + [1.0] * 4)
        b = BSplineBasis(degree=3, knots=t, penalty=order).setup()
        z = np.linspace(0.02, 0.98, 21)
        close(b.derivative_matrix(z, order), ppoly_matrix(b, z, order), tol=1e-11)

    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_null_space_dim_matches_the_penalty_rank(self, degree):
        """`null_space_dim` is derived from the knot multiplicities; the rank is
        measured from the matrix. The two never see each other.

        A knot of multiplicity q leaves the spline C^(degree - q), so the penalty's
        null space grows as knots repeat: it is `penalty` only while the basis stays
        smooth enough to join its spans seamlessly."""
        for multiplicity in range(1, degree + 2):
            for order in range(1, degree + 1):
                t = np.array(
                    [0.0] * (degree + 1) + [0.4] * multiplicity + [0.7]
                    + [1.0] * (degree + 1)
                )
                b = BSplineBasis(degree=degree, knots=t, penalty=order).setup()
                rank = np.linalg.matrix_rank(b.penalty_matrix(), tol=1e-8)
                assert b.k - rank == b.null_space_dim, (multiplicity, order)

    def test_null_space_dim_is_the_penalty_order_when_knots_are_simple(self):
        """The familiar answer, and the one the rule has to keep giving."""
        for order in (1, 2, 3):
            b = bspline(degree=3, k=9, penalty=order).setup()
            assert b.null_space_dim == order

    def test_limits_override_the_range_of_z(self):
        b = BSplineBasis(
            degree=3, k=5, knots="uniform", limits=(-2.0, 5.0), penalty=2
        ).setup(np.linspace(0.0, 1.0, 10))
        assert b.domain == (-2.0, 5.0)
        close(b.knot_vector, [-2.0] * 4 + [1.5] + [5.0] * 4, tol=1e-15)

    def test_no_interior_knots_when_k_is_minimal(self):
        b = bspline(degree=3, k=4, penalty=2).setup()
        close(b.knot_vector, [0.0] * 4 + [1.0] * 4, tol=1e-15)

    def test_setup_without_data_or_limits_raises(self):
        with pytest.raises(ValueError, match="needs data or limits"):
            BSplineBasis(degree=3, k=6, knots="uniform", penalty=2).setup()

    def test_quantiles_without_data_raises(self):
        with pytest.raises(ValueError, match="requires data"):
            BSplineBasis(
                degree=3, k=6, knots="quantiles", limits=(0.0, 1.0), penalty=2
            ).setup()

    def test_knot_vector_before_setup_does_not_exist(self):
        """It is a plain attribute set at setup, not a guarded property: absent before
        configuration rather than present and wrong."""
        with pytest.raises(AttributeError, match="knot_vector"):
            bspline(degree=3, k=6, penalty=2).knot_vector


# ---------------------------------------------------------------------------------
# TensorProductBasis
# ---------------------------------------------------------------------------------


def marginal_rows(basis, z, derivative=None):
    """Row i of the model matrix, as the explicit Kronecker product of the margins'
    row i, for one-dimensional margins. `row_kron` is what the class computes; this
    is what it means.

    At a nonzero multi-index this is also the *only* route to a differentiated product
    design: the class differentiates its Gram matrix and not its model matrix, since
    that would need every margin to differentiate and a thin plate margin does not."""
    alpha = (0,) * len(basis.bases) if derivative is None else derivative
    mats = [
        b.derivative_matrix(z[:, i], a) if a else b.model_matrix(z[:, i])
        for i, (b, a) in enumerate(zip(basis.bases, alpha))
    ]
    rows = [mats[0][i] for i in range(len(z))]
    for M in mats[1:]:
        rows = [np.kron(rows[i], M[i]) for i in range(len(z))]
    return np.array(rows)


class TestTensorProduct:
    @pytest.fixture
    def tp(self, z2):
        return TensorProductBasis(
            bspline(degree=3, k=6, penalty=2), bspline(degree=2, k=5, penalty=1)
        ).setup(z2)

    def test_shape_and_defaults(self, tp):
        assert (tp.ndim, tp.k) == (2, 30)
        assert tp.penalty == "integral"
        assert tp.null_space_dim == 2 * 1

    def test_model_matrix_is_the_kronecker_product_of_the_rows(self, tp, z2):
        close(tp.model_matrix(z2), marginal_rows(tp, z2), tol=1e-14)

    def test_a_product_function_is_separable(self, tp, rng):
        """The whole point of the row-wise Kronecker layout: separable coefficients
        give a separable function."""
        cx, cy = rng.normal(size=6), rng.normal(size=5)
        c = np.kron(cx, cy)
        z = np.column_stack([np.linspace(0.05, 0.95, 11), np.linspace(0.9, 0.1, 11)])
        fx = tp.bases[0].model_matrix(z[:, 0]) @ cx
        fy = tp.bases[1].model_matrix(z[:, 1]) @ cy
        close(tp.model_matrix(z) @ c, fx * fy, tol=1e-11)

    @pytest.mark.parametrize("alpha", [(0, 0), (1, 0), (0, 1), (2, 2)])
    def test_gram_is_the_kronecker_product_of_the_marginal_grams(self, tp, alpha):
        ref = np.kron(
            tp.bases[0].gram_matrix(alpha[0]), tp.bases[1].gram_matrix(alpha[1])
        )
        assert np.array_equal(tp.gram_matrix(alpha), ref)

    def test_gram_default_is_the_product_mass_matrix(self, tp):
        assert np.array_equal(
            tp.gram_matrix(), np.kron(tp.bases[0].gram_matrix(), tp.bases[1].gram_matrix())
        )

    def test_a_scalar_derivative_is_rejected(self, tp):
        with pytest.raises(TypeError, match="multi-index"):
            tp.gram_matrix(2)

    def test_integral_penalty_is_the_fanned_out_sum(self, tp):
        Sx, Sy = tp.bases[0].penalty_matrix(), tp.bases[1].penalty_matrix()
        Gx, Gy = tp.bases[0].gram_matrix(), tp.bases[1].gram_matrix()
        close(tp.penalty_matrix(), np.kron(Sx, Gy) + np.kron(Gx, Sy), tol=1e-14)

    def test_integral_penalty_is_the_integrated_squared_partials(self, tp, rng):
        """The independent statement: c' P c is the integral over the product domain
        of the squared partial derivative in each axis, summed. Nothing the class does
        is reused -- the differentiated design is assembled row by row through
        `marginal_rows`, and the function is then differentiated and squared."""
        c = rng.normal(size=tp.k)
        points, weights = product_quadrature(tp)
        total = 0.0
        for alpha in ((2, 0), (0, 1)):  # the margins' own penalty orders
            v = marginal_rows(tp, points, alpha) @ c
            total += weights @ v**2
        close(c @ tp.penalty_matrix() @ c, total, tol=1e-12)

    def test_identity_penalty_fans_out_against_the_identity(self, z2):
        tp = tensor_identity(
            bspline(degree=3, k=6, penalty=2), bspline(degree=2, k=5, penalty=1)
        ).setup(z2)
        assert tp.penalty == "identity"
        Sx, Sy = tp.bases[0].penalty_matrix(), tp.bases[1].penalty_matrix()
        ref = np.kron(Sx, np.eye(5)) + np.kron(np.eye(6), Sy)
        close(tp.penalty_matrix(), ref, tol=1e-14)

    def test_identity_penalty_is_the_marginal_roughness_of_the_coefficient_array(
        self, z2, rng
    ):
        """What 'not the Gram matrix of anything' means concretely: c' P c sums the
        marginal penalty over the *columns* of the coefficient array, ignoring the
        other axis's geometry entirely."""
        tp = tensor_identity(
            bspline(degree=3, k=6, penalty=2), bspline(degree=2, k=5, penalty=1)
        ).setup(z2)
        C = rng.normal(size=(6, 5))
        Sx, Sy = tp.bases[0].penalty_matrix(), tp.bases[1].penalty_matrix()
        ref = np.trace(C.T @ Sx @ C) + np.trace(C @ Sy @ C.T)
        close(C.ravel() @ tp.penalty_matrix() @ C.ravel(), ref, tol=1e-12)

    def test_the_two_forms_differ(self, z2):
        def margins():
            return bspline(degree=3, k=6, penalty=2), bspline(degree=2, k=5, penalty=1)

        a = TensorProductBasis(*margins()).setup(z2)
        b = tensor_identity(*margins()).setup(z2)
        assert not np.allclose(a.penalty_matrix(), b.penalty_matrix())

    @pytest.mark.parametrize("form", ["integral", "identity"])
    @pytest.mark.parametrize("px,py", [(2, 1), (2, 2), (1, 1)])
    def test_penalty_rank_matches_null_space_dim(self, z2, form, px, py):
        """Both forms leave exactly null(S_x) tensor null(S_y) unpenalised, which is
        the product `null_space_dim` reports."""
        margins = (bspline(degree=3, k=6, penalty=px), bspline(degree=3, k=5, penalty=py))
        build = TensorProductBasis if form == "integral" else tensor_identity
        tp = build(*margins).setup(z2)
        assert tp.null_space_dim == px * py
        assert np.linalg.matrix_rank(tp.penalty_matrix()) == tp.k - px * py

    def test_penalty_matrices_are_the_unfanned_margins(self, tp):
        mats = tp.penalty_matrices()
        assert [M.shape for M in mats] == [(6, 6), (5, 5)]
        for M, b in zip(mats, tp.bases):
            assert np.array_equal(M, b.penalty_matrix())

    def test_unpenalised_margins_give_an_unpenalised_product(self, z2):
        tp = TensorProductBasis(
            bspline(degree=3, k=5, penalty=None), bspline(degree=3, k=4, penalty=None)
        ).setup(z2)
        assert tp.penalty is None
        with pytest.raises(ValueError, match="without a penalty"):
            tp.penalty_matrix()

    def test_mixed_penalisation_is_rejected(self):
        with pytest.raises(ValueError, match="uniformly"):
            TensorProductBasis(
                bspline(degree=3, k=5, penalty=2), bspline(degree=3, k=4, penalty=None)
            )

    def test_fewer_than_two_sub_bases_is_rejected(self):
        with pytest.raises(ValueError, match="at least two"):
            TensorProductBasis(bspline(degree=3, k=5, penalty=2))

    def test_wrong_column_count_is_rejected(self, tp, z2):
        with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
            tp.model_matrix(z2[:, :1])


class TestNestedTensorProduct:
    @pytest.fixture
    def nested(self):
        inner = TensorProductBasis(
            bspline(degree=3, k=5, penalty=2), bspline(degree=2, k=4, penalty=1)
        )
        return TensorProductBasis(inner, bspline(degree=3, k=4, penalty=2)).setup()

    def test_shape_flattens_the_nesting(self, nested):
        assert (nested.ndim, nested.k) == (3, 80)
        assert nested.null_space_dim == (2 * 1) * 2

    def test_model_matrix_is_the_three_way_kronecker_product(self, nested, rng):
        z = rng.uniform(0.0, 1.0, (6, 3))
        inner, outer = nested.bases
        mats = [
            inner.bases[0].model_matrix(z[:, 0]),
            inner.bases[1].model_matrix(z[:, 1]),
            outer.model_matrix(z[:, 2]),
        ]
        ref = np.array(
            [np.kron(np.kron(mats[0][i], mats[1][i]), mats[2][i]) for i in range(6)]
        )
        close(nested.model_matrix(z), ref, tol=1e-14)

    def test_gram_is_the_three_way_kronecker_product(self, nested):
        inner, outer = nested.bases
        ref = np.kron(
            np.kron(inner.bases[0].gram_matrix(1), inner.bases[1].gram_matrix(0)),
            outer.gram_matrix(2),
        )
        assert np.array_equal(nested.gram_matrix((1, 0, 2)), ref)

    def test_penalty_matrices_are_per_sub_basis_not_per_axis(self, nested):
        """The documented consequence of nesting: the inner product contributes its
        own already-summed penalty as one entry, so its two axes share a lambda."""
        mats = nested.penalty_matrices()
        assert [M.shape for M in mats] == [(20, 20), (4, 4)]
        assert np.array_equal(mats[0], nested.bases[0].penalty_matrix())

    def test_penalty_is_the_fanned_out_sum_of_those_two(self, nested):
        inner, outer = nested.bases
        ref = np.kron(inner.penalty_matrix(), outer.gram_matrix()) + np.kron(
            inner.gram_matrix(), outer.penalty_matrix()
        )
        close(nested.penalty_matrix(), ref, tol=1e-14)
        assert np.linalg.matrix_rank(ref) == nested.k - nested.null_space_dim


class TestThinPlateMargin:
    """ndim > 1 on a margin, which is also what forces penalty='identity'."""

    @pytest.fixture
    def zm(self):
        rng = np.random.default_rng(2)
        return rng.uniform(0.0, 1.0, (N, 3))

    def test_integral_form_cannot_supply_a_mass_matrix(self, zm):
        """Raised at setup, not at penalty_matrix(): the fan-out is assembled when the
        basis is configured, so an impossible one is reported there."""
        tp = TensorProductBasis(
            ThinPlateBasis(d=2, k=9, m=2, penalty=2), bspline(degree=3, k=5, penalty=2)
        )
        with pytest.raises(NotImplementedError, match="fan_out='identity'"):
            tp.setup(zm)

    def test_identity_form_composes(self, zm):
        tp = tensor_identity(
            ThinPlateBasis(d=2, k=9, m=2, penalty=2), bspline(degree=3, k=5, penalty=2)
        ).setup(zm)
        assert (tp.ndim, tp.k) == (3, 45)
        assert tp.null_space_dim == 3 * 2
        Sr, Sb = tp.bases[0].penalty_matrix(), tp.bases[1].penalty_matrix()
        ref = np.kron(Sr, np.eye(5)) + np.kron(np.eye(9), Sb)
        close(tp.penalty_matrix(), ref, tol=1e-14)
        assert np.linalg.matrix_rank(tp.penalty_matrix()) == 45 - 6

    def test_the_first_two_columns_of_z_go_to_the_radial_margin(self, zm):
        tp = tensor_identity(
            ThinPlateBasis(d=2, k=9, m=2, penalty=2), bspline(degree=3, k=5, penalty=2)
        ).setup(zm)
        radial = tp.bases[0].model_matrix(zm[:, :2])
        spline = tp.bases[1].model_matrix(zm[:, 2])
        ref = np.array([np.kron(radial[i], spline[i]) for i in range(N)])
        close(tp.model_matrix(zm), ref, tol=1e-13)


# ---------------------------------------------------------------------------------
# ThinPlateBasis
# ---------------------------------------------------------------------------------


class TestThinPlateKernel:
    def test_eta_coefficient_matches_the_closed_forms(self):
        close(_eta_coefficient(2, 1), 1 / 12)  # cubic smoothing spline on the line
        close(_eta_coefficient(2, 2), 1 / (8 * np.pi))  # r^2 log r in the plane

    @pytest.mark.parametrize("d", [1, 2])
    def test_the_model_matrix_is_the_hand_written_kernel(self, d, rng):
        """(E_z delta, T_z) with E from the formula above and delta recovered at the
        knots. Fixing delta at the knots leaves the kernel itself as the only thing
        the check at fresh points can be testing."""
        knots = np.sort(rng.uniform(0.0, 1.0, (20, d)), axis=0)
        b = ThinPlateBasis(k=10, m=2, knots=knots, penalty=2).setup()
        delta = radial_coefficients(b, knots)
        zp = rng.uniform(0.05, 0.95, (9, d))
        wiggly = b.k - b.null_space_dim
        close(b.model_matrix(zp)[:, :wiggly], eta(cdist(zp, knots), d) @ delta, tol=1e-11)

    def test_the_polynomial_block_is_the_centred_monomials(self, rng):
        knots = rng.uniform(0.0, 1.0, (20, 2))
        b = ThinPlateBasis(k=10, m=2, knots=knots, penalty=2).setup()
        zp = rng.uniform(0.0, 1.0, (7, 2))
        centred = zp - knots.mean(axis=0)
        # _monomial_powers sorts by (total degree, exponent tuple): 1, then z2, then z1
        T = np.column_stack([np.ones(7), centred[:, 1], centred[:, 0]])
        close(b.model_matrix(zp)[:, b.k - b.null_space_dim :], T, tol=1e-14)

    def test_the_side_conditions_hold_on_the_retained_basis(self, rng):
        """T' delta = 0: the radial part carries no polynomial component, which is
        what makes the last M columns the whole of the null space."""
        knots = rng.uniform(0.0, 1.0, (22, 2))
        b = ThinPlateBasis(k=12, m=2, knots=knots, penalty=2).setup()
        delta = radial_coefficients(b, knots)
        centred = knots - knots.mean(axis=0)
        T = np.column_stack([np.ones(22), centred[:, 1], centred[:, 0]])
        assert np.abs(T.T @ delta).max() < 1e-12 * np.abs(delta).max()

    def test_at_the_knots_the_radial_block_is_u_d_z(self, rng):
        """The class docstring's claim, checked through the invariant that survives
        the arbitrary orthogonal basis `null_space` picks for Z: R R' is determined
        even though R's columns are not."""
        knots = rng.uniform(0.0, 1.0, (18, 2))
        b = ThinPlateBasis(k=10, m=2, knots=knots, penalty=2).setup()
        E = eta(cdist(knots, knots), 2)
        w, U = np.linalg.eigh(E)
        order = np.argsort(np.abs(w))[::-1][: b.k]
        Dk, Uk = w[order], U[:, order]
        centred = knots - knots.mean(axis=0)
        T = np.column_stack([np.ones(18), centred[:, 1], centred[:, 0]])
        ref = (Uk * Dk) @ null_space((Uk.T @ T).T)
        R = b.model_matrix(knots)[:, : b.k - b.null_space_dim]
        close(R @ R.T, ref @ ref.T, tol=1e-12)
        singular = lambda A: np.linalg.svd(A, compute_uv=False)
        close(singular(R), singular(ref), tol=1e-10)


class TestThinPlateStructure:
    @pytest.mark.parametrize("d,m", [(1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3)])
    def test_null_space_dimension_and_penalty_rank(self, d, m):
        knots = np.random.default_rng(9).uniform(0.0, 1.0, (25, d))
        k = comb(m + d - 1, d) + 4
        b = ThinPlateBasis(k=k, m=m, knots=knots, penalty=m).setup()
        assert b.null_space_dim == comb(m + d - 1, d)
        assert np.linalg.matrix_rank(b.penalty_matrix()) == k - b.null_space_dim

    def test_the_penalty_is_the_quadratic_form_in_the_radial_kernel(self, rng):
        """Wood appendix A(d): for delta = U_k Z_k delta_k the penalty is delta' E delta
        exactly, not an approximation to it. E here is the hand-written kernel, whose
        eigenvalues carry both signs -- so this pins D rather than merely |D|, which
        the rank and definiteness of P cannot distinguish."""
        knots = rng.uniform(0.0, 1.0, (20, 2))
        b = ThinPlateBasis(k=11, m=2, knots=knots, penalty=2).setup()
        E = eta(cdist(knots, knots), 2)
        assert np.linalg.eigvalsh(E).min() < 0 < np.linalg.eigvalsh(E).max()
        delta = radial_coefficients(b, knots)
        c = rng.normal(size=b.k)
        wiggly = delta @ c[: b.k - b.null_space_dim]
        close(c @ b.penalty_matrix() @ c, wiggly @ E @ wiggly, tol=1e-10)

    def test_the_last_m_columns_are_unpenalised(self, rng):
        knots = rng.uniform(0.0, 1.0, (20, 2))
        b = ThinPlateBasis(k=11, m=2, knots=knots, penalty=2).setup()
        P = b.penalty_matrix()
        assert (P[:, -3:] == 0.0).all() and (P[-3:, :] == 0.0).all()
        assert np.linalg.eigvalsh(P).min() > -1e-14 * np.abs(P).max()

    def test_bases_of_different_rank_are_nested(self, rng):
        """Wood's claim, and what makes an F-ratio between two ranks legitimate: the
        rank-k column space sits inside the rank-k' one when k' > k."""
        knots = rng.uniform(0.0, 1.0, (24, 2))
        small = ThinPlateBasis(k=8, m=2, knots=knots, penalty=2).setup()
        large = ThinPlateBasis(k=15, m=2, knots=knots, penalty=2).setup()
        zp = rng.uniform(0.0, 1.0, (50, 2))
        Xs, Xl = small.model_matrix(zp), large.model_matrix(zp)
        Q = np.linalg.qr(Xl)[0]
        assert np.abs(Xs - Q @ (Q.T @ Xs)).max() < 1e-12 * np.abs(Xs).max()
        # and not the other way round
        Qs = np.linalg.qr(Xs)[0]
        assert np.abs(Xl - Qs @ (Qs.T @ Xl)).max() > 1e-3 * np.abs(Xl).max()

    def test_m_defaults_to_the_smallest_with_2m_above_d_plus_1(self):
        for d, expected in ((1, 2), (2, 2), (3, 3), (4, 3)):
            knots = np.random.default_rng(3).uniform(0.0, 1.0, (30, d))
            b = ThinPlateBasis(k=20, knots=knots, penalty=None)
            assert b.m == expected

    def test_m_below_the_well_posed_bound_is_rejected(self):
        with pytest.raises(ValueError, match="2m > d"):
            ThinPlateBasis(d=4, k=20, m=2, penalty=2)

    def test_a_penalty_other_than_m_is_rejected(self):
        with pytest.raises(ValueError, match="contradicts"):
            ThinPlateBasis(d=2, k=10, m=2, penalty=3)

    def test_penalty_supplies_m(self):
        assert ThinPlateBasis(d=2, k=10, penalty=3).m == 3

    def test_k_below_the_null_space_is_rejected(self):
        with pytest.raises(ValueError, match="null space dimension"):
            ThinPlateBasis(d=2, k=3, m=2, penalty=2)

    def test_k_above_the_knot_count_is_rejected(self, rng):
        b = ThinPlateBasis(d=2, k=30, m=2, penalty=2)
        with pytest.raises(ValueError, match="exceeds the"):
            b.setup(rng.uniform(0.0, 1.0, (12, 2)))

    def test_gram_matrix_raises(self, rng):
        b = ThinPlateBasis(d=2, k=10, m=2, penalty=2).setup(rng.uniform(0.0, 1.0, (20, 2)))
        with pytest.raises(NotImplementedError, match="no mass matrix"):
            b.gram_matrix()

    def test_setup_without_data_raises(self):
        with pytest.raises(ValueError, match="needs data"):
            ThinPlateBasis(d=2, k=10, m=2, penalty=2).setup()

    def test_unpenalised_is_the_same_basis(self, rng):
        """penalty=None is Wood's pure regression spline: identical columns, no
        penalty matrix."""
        knots = rng.uniform(0.0, 1.0, (20, 2))
        a = ThinPlateBasis(k=10, m=2, knots=knots, penalty=2).setup()
        b = ThinPlateBasis(k=10, m=2, knots=knots, penalty=None).setup()
        zp = rng.uniform(0.0, 1.0, (5, 2))
        assert np.array_equal(a.model_matrix(zp), b.model_matrix(zp))
        with pytest.raises(ValueError, match="without a penalty"):
            b.penalty_matrix()


class TestThinPlateKnots:
    def test_knots_are_the_distinct_rows_of_z_in_order(self):
        """Ties are collapsed -- repeated centres make E singular -- and the survivors
        keep the order they first appear in, not sorted order."""
        rows = [[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [0.5, 4.0], [2.0, 0.5]]
        z = np.array(rows + [rows[1], rows[0], rows[3]])
        b = ThinPlateBasis(d=2, k=4, m=2, penalty=2).setup(z)
        close(b.knot_points, rows, tol=1e-15)

    def test_explicit_knots_ignore_z(self, rng):
        knots = rng.uniform(0.0, 1.0, (20, 2))
        a = ThinPlateBasis(k=10, m=2, knots=knots, penalty=2).setup()
        b = ThinPlateBasis(k=10, m=2, knots=knots, penalty=2).setup(
            rng.uniform(-9.0, 9.0, (60, 2))
        )
        assert np.array_equal(a.knot_points, b.knot_points)
        zp = rng.uniform(0.0, 1.0, (5, 2))
        assert np.array_equal(a.model_matrix(zp), b.model_matrix(zp))

    def test_max_knots_subsamples_reproducibly(self, rng):
        z = rng.uniform(0.0, 1.0, (60, 1))
        a = ThinPlateBasis(d=1, k=6, m=2, penalty=2, max_knots=20, seed=7).setup(z)
        b = ThinPlateBasis(d=1, k=6, m=2, penalty=2, max_knots=20, seed=7).setup(z)
        assert a.knot_points.shape == (20, 1)
        assert np.array_equal(a.knot_points, b.knot_points)
        # a subsample of z, kept in the order the rows appear there
        where = [np.flatnonzero((z == r).all(axis=1))[0] for r in a.knot_points]
        assert where == sorted(where)

    def test_a_different_seed_takes_a_different_subsample(self, rng):
        z = rng.uniform(0.0, 1.0, (60, 1))
        a = ThinPlateBasis(d=1, k=6, m=2, penalty=2, max_knots=20, seed=7).setup(z)
        b = ThinPlateBasis(d=1, k=6, m=2, penalty=2, max_knots=20, seed=8).setup(z)
        assert not np.array_equal(a.knot_points, b.knot_points)

    def test_max_knots_none_keeps_every_distinct_point(self, rng):
        z = rng.uniform(0.0, 1.0, (60, 1))
        b = ThinPlateBasis(d=1, k=6, m=2, penalty=2, max_knots=None).setup(z)
        assert b.knot_points.shape == (60, 1)

    def test_one_dimensional_z_is_accepted_as_a_column(self, rng):
        z = rng.uniform(0.0, 1.0, 30)
        b = ThinPlateBasis(d=1, k=8, m=2, penalty=2).setup(z)
        assert b.knot_points.shape == (30, 1)
        assert np.array_equal(b.model_matrix(z), b.model_matrix(z[:, None]))


# ---------------------------------------------------------------------------------
# the bs factory
# ---------------------------------------------------------------------------------


class TestBsplinesFactory:
    def test_one_axis_returns_a_bspline_basis(self):
        b = bs(degree=2, k=9, knots="uniform", limits=(0.0, 1.0), penalty=1)
        assert type(b) is BSplineBasis
        assert (b.degree, b.k, b.penalty) == (2, 9, 1)

    def test_several_axes_return_a_tensor_product(self):
        b = bs(degree=3, k=(6, 6), knots="uniform", limits=(0.0, 1.0), penalty=2)
        assert type(b) is TensorProductBasis
        assert (b.ndim, b.k, b.penalty) == (2, 36, "integral")
        assert [m.k for m in b.bases] == [6, 6]
        assert [m.penalty for m in b.bases] == [2, 2]

    def test_a_one_entry_k_is_one_axis(self):
        """`k` carries the number of axes, so a 1-tuple is one axis and gets the bare
        basis rather than a one-margin product."""
        b = bs(k=(6,), knots="uniform", limits=(0.0, 1.0), penalty=2)
        assert type(b) is BSplineBasis

    def test_explicit_knot_vectors_imply_the_number_of_axes(self):
        """With `k=None` there is nothing else to read it from: one vector is one axis,
        a list of them is one axis each. A bare vector is one value rather than one
        entry per axis, which is the rule `limits` already follows."""
        long, short = [0.0] * 4 + [0.3, 0.6] + [1.0] * 4, [0.0] * 4 + [0.5] + [1.0] * 4
        assert type(bs(knots=long, penalty=2)) is BSplineBasis
        b = bs(knots=[long, short], penalty=2)
        assert type(b) is TensorProductBasis
        assert (b.ndim, [m.k for m in b.bases]) == (2, [6, 5])

    def test_fan_out_reaches_the_tensor_product(self):
        b = bs(
            degree=3, k=(6, 6), knots="uniform", limits=(0.0, 1.0),
            penalty=2, fan_out="identity",
        )
        assert b.penalty == "identity"

    def test_per_axis_arguments_reach_the_right_margin(self):
        b = bs(
            k=(6, 4), degree=(3, 1), knots="uniform", limits=(0.0, 1.0),
            penalty=(2, 1),
        )
        assert (b.ndim, b.k) == (2, 24)
        assert [(m.degree, m.k, m.penalty) for m in b.bases] == [(3, 6, 2), (1, 4, 1)]

    def test_a_per_axis_argument_of_the_wrong_length_raises(self):
        with pytest.raises(ValueError, match="penalty has 3 entries"):
            bs(k=(6, 4), knots="uniform", limits=(0.0, 1.0), penalty=(2, 1, 1))

    def test_the_tensor_product_class_matches_a_hand_built_one(self, z2):
        a = bs(degree=3, k=(5, 5), knots="uniform", limits=(0.0, 1.0), penalty=2)
        b = TensorProductBasis(
            bspline(degree=3, k=5, penalty=2),
            bspline(degree=3, k=5, penalty=2),
            centre=True,
        )
        a.setup(z2)
        b.setup(z2)
        assert np.array_equal(a.model_matrix(z2), b.model_matrix(z2))
        assert np.array_equal(a.penalty_matrix(), b.penalty_matrix())


# ---------------------------------------------------------------------------------
# the tps factory
# ---------------------------------------------------------------------------------


class TestTpsFactory:
    def test_it_is_centred_and_penalised_at_the_usual_order(self):
        """m == 2 at both d == 1 and d == 2, the cubic smoothing spline and the thin
        plate spline proper, where the class with no penalty would be neither."""
        for d in (1, 2):
            b = tps(k=10, d=d)
            assert type(b) is ThinPlateBasis
            assert (b.centre, b.d, b.m, b.penalty) == (True, d, 2, 2)

    def test_it_matches_a_hand_built_basis(self, z2):
        a = tps(k=12, d=2).setup(z2)
        b = ThinPlateBasis(k=12, d=2, penalty=2, centre=True).setup(z2)
        assert np.array_equal(a.model_matrix(z2), b.model_matrix(z2))
        assert np.array_equal(a.penalty_matrix(), b.penalty_matrix())

    def test_d_is_inferred_from_explicit_knots(self, rng):
        """`d` defaults to None, as on the class, so two-column knots are not
        contradicted by a default of 1."""
        b = tps(k=10, knots=rng.uniform(0.0, 1.0, (30, 2)))
        assert (b.d, b.m) == (2, 2)

    def test_penalty_sets_the_order(self):
        assert tps(k=20, d=3, penalty=3).m == 3

    def test_the_default_order_fails_from_d_equal_to_4(self):
        with pytest.raises(ValueError, match="2m > d"):
            tps(k=20, d=4)


# ---------------------------------------------------------------------------------
# centring
# ---------------------------------------------------------------------------------

CENTRED = {
    "b-spline": (
        lambda centre: BSplineBasis(
            degree=3, k=8, knots="quantiles", penalty=2, centre=centre
        ),
        "curve",
    ),
    "tensor product": (
        # Hand-built rather than from `bs`, which always centres.
        lambda centre: TensorProductBasis(
            *[
                BSplineBasis(degree=3, k=5, knots="quantiles", penalty=2)
                for _ in range(2)
            ],
            centre=centre,
        ),
        "surface",
    ),
    "thin plate": (
        lambda centre: ThinPlateBasis(d=2, k=10, m=2, penalty=2, centre=centre),
        "surface",
    ),
}


def centred_pair(case):
    """The same basis centred and uncentred, set up on the same data. Both see the same
    `z`, so knot placement is identical and the two designs differ only by the shift."""
    build, kind = CENTRED[case]
    z = (curve_data() if kind == "curve" else surface_data())[0]
    return build(True).setup(z), build(False).setup(z), z


@pytest.mark.parametrize("case", list(CENTRED))
class TestCentring:
    def test_the_constraint_holds(self, case):
        """`sum_i f(z_i) = 0`, judged against the size of the column sums that were
        removed rather than against zero absolutely."""
        b, u, z = centred_pair(case)
        removed = np.abs(u.model_matrix(z).sum(axis=0)).max()
        assert removed > 1.0
        assert np.abs(b.model_matrix(z).sum(axis=0)).max() < 1e-12 * removed

    def test_it_is_the_uncentred_design_less_its_column_means(self, case):
        b, u, z = centred_pair(case)
        X = u.model_matrix(z)
        close(b.model_matrix(z), X - X.mean(axis=0), tol=1e-14)

    def test_it_shifts_the_function_by_a_constant(self, case, rng):
        """Away from the training data the two fits differ by one number, not by a
        per-point correction: the shift is a constant per basis function."""
        b, u, z = centred_pair(case)
        c = rng.normal(size=b.k)
        gap = b.eval(z, c) - u.eval(z, c)
        close(gap, np.full(len(z), gap[0]), tol=1e-10)

    def test_the_penalty_is_untouched(self, case):
        """A penalty of order 1 or more annihilates constants, so subtracting one
        changes nothing -- the same matrix, not merely a close one."""
        b, u, _ = centred_pair(case)
        assert np.array_equal(b.penalty_matrix(), u.penalty_matrix())

    def test_setup_without_data_raises(self, case):
        build, _ = CENTRED[case]
        with pytest.raises(ValueError, match="centre=True needs data"):
            build(True).setup()

    def test_clone_carries_the_constraint(self, case):
        b, u, z = centred_pair(case)
        assert b.clone().centre and not u.clone().centre
        close(b.clone().setup(z).model_matrix(z), b.model_matrix(z), tol=1e-14)


def test_centring_does_not_reach_the_derivative():
    """Why `derivative_matrix` takes no centring flag: the shift is a constant, so it
    differentiates away. At order 0 it is therefore the *uncentred* design, which is
    what a caller sweeping over orders wants."""
    build, _ = CENTRED["b-spline"]
    z = curve_data()[0]
    a, b = build(True).setup(z), build(False).setup(z)
    zp = np.linspace(0.1, 0.9, 9)
    assert np.array_equal(a.derivative_matrix(zp, 1), b.derivative_matrix(zp, 1))
    assert np.array_equal(a.derivative_matrix(zp, 0), b.model_matrix(zp))


def test_centring_the_margins_does_not_centre_the_product(z2):
    """Why `bs` centres the product rather than each margin: the column sums of a
    row-wise Kronecker product of centred margins are covariances between the margins'
    columns, and those do not vanish."""
    tp = TensorProductBasis(
        *[
            BSplineBasis(degree=3, k=5, knots="quantiles", penalty=2, centre=True)
            for _ in range(2)
        ]
    ).setup(z2)
    assert np.abs(tp.model_matrix(z2).sum(axis=0)).max() > 1e-3

    factory = bs(degree=3, k=(5, 5), knots="quantiles", penalty=2)
    assert factory.centre and not any(m.centre for m in factory.bases)


# ---------------------------------------------------------------------------------
# lifecycle -- specified, configured, frozen
# ---------------------------------------------------------------------------------

LIFECYCLE = {
    "b-spline, quantile knots": (
        lambda: BSplineBasis(degree=3, k=8, knots="quantiles", penalty=2),
        "curve",
    ),
    "b-spline, explicit knots": (
        lambda: BSplineBasis(
            degree=3, knots=np.array([0.0] * 4 + [0.3, 0.6] + [1.0] * 4), penalty=2
        ),
        "curve",
    ),
    "tensor product": (
        lambda: TensorProductBasis(
            BSplineBasis(degree=3, k=5, knots="quantiles", penalty=2),
            BSplineBasis(degree=2, k=4, knots="quantiles", penalty=1),
        ),
        "surface",
    ),
    "tensor product, identity": (
        lambda: tensor_identity(
            BSplineBasis(degree=3, k=5, knots="quantiles", penalty=2),
            BSplineBasis(degree=2, k=4, knots="quantiles", penalty=1),
        ),
        "surface",
    ),
    "tp b-spline": (
        lambda: bs(degree=3, k=(5, 5), knots="quantiles", penalty=2),
        "surface",
    ),
    "thin plate": (lambda: ThinPlateBasis(d=2, k=10, m=2, penalty=2), "surface"),
}


@pytest.mark.parametrize("case", list(LIFECYCLE))
class TestLifecycle:
    @staticmethod
    def _case(case):
        factory, kind = LIFECYCLE[case]
        return factory, (curve_data()[0] if kind == "curve" else surface_data()[0])

    def test_starts_unconfigured(self, case):
        factory, _ = self._case(case)
        b = factory()
        assert not b.is_setup
        assert b.k > 0 and b.ndim >= 1 and b.penalty is not None
        with pytest.raises(RuntimeError, match="setup"):
            b.model_matrix(np.zeros((3, b.ndim)) if b.ndim > 1 else np.zeros(3))

    def test_setup_returns_self_and_marks_configured(self, case):
        factory, z = self._case(case)
        b = factory()
        assert b.setup(z) is b
        assert b.is_setup

    def test_setup_twice_raises(self, case):
        factory, z = self._case(case)
        b = factory().setup(z)
        with pytest.raises(RuntimeError, match="only once"):
            b.setup(z)

    def test_attribute_assignment_after_setup_raises(self, case):
        factory, z = self._case(case)
        b = factory().setup(z)
        with pytest.raises(AttributeError, match="immutable after setup"):
            b.penalty = 1
        with pytest.raises(AttributeError, match="immutable after setup"):
            b.something_new = 1

    def test_derived_arrays_are_read_only(self, case):
        """What lets a `BasisFunction` hold a live reference rather than a copy."""
        factory, z = self._case(case)
        arrays = held_arrays(factory().setup(z))
        assert arrays
        assert not any(a.flags.writeable for a in arrays)

    def test_a_clone_taken_before_setup_reproduces_the_matrices(self, case):
        factory, z = self._case(case)
        b = factory()
        c = b.clone()
        b.setup(z)
        c.setup(z)
        assert b is not c
        assert np.array_equal(b.model_matrix(z), c.model_matrix(z))
        assert np.array_equal(b.penalty_matrix(), c.penalty_matrix())

    def test_a_clone_taken_after_setup_reproduces_the_matrices(self, case):
        factory, z = self._case(case)
        b = factory().setup(z)
        c = b.clone()
        assert not c.is_setup
        c.setup(z)
        assert np.array_equal(b.model_matrix(z), c.model_matrix(z))
        assert np.array_equal(b.penalty_matrix(), c.penalty_matrix())

    def test_a_clone_keeps_the_hyperparameters_readable(self, case):
        factory, _ = self._case(case)
        b = factory()
        c = b.clone()
        assert type(c) is type(b)
        assert (c.k, c.ndim, c.penalty) == (b.k, b.ndim, b.penalty)

    def test_a_clone_configured_elsewhere_leaves_the_parent_alone(self, case):
        """Fold-wise refitting: the clone is configured from the fold's data, and the
        parent it came from does not move."""
        factory, z = self._case(case)
        b = factory().setup(z)
        X, P = b.model_matrix(z).copy(), b.penalty_matrix()
        c = b.clone().setup(z[: len(z) // 2] * 0.5)
        assert c.model_matrix(z).shape == X.shape
        assert np.array_equal(b.model_matrix(z), X)
        assert np.array_equal(b.penalty_matrix(), P)


class TestFrozenArrays:
    def test_the_knot_vector_cannot_be_written(self):
        b = bspline(degree=3, k=8, penalty=2).setup()
        assert not b.knot_vector.flags.writeable
        with pytest.raises(ValueError):
            b.knot_vector[0] = 5.0

    def test_the_knot_points_cannot_be_written(self, z2):
        b = ThinPlateBasis(d=2, k=10, m=2, penalty=2).setup(z2)
        assert not b.knot_points.flags.writeable
        with pytest.raises(ValueError):
            b.knot_points[0, 0] = 5.0

    def test_a_tensor_products_margins_are_frozen(self, z2):
        tp = TensorProductBasis(
            bspline(degree=3, k=5, penalty=2), bspline(degree=3, k=4, penalty=2)
        ).setup(z2)
        assert all(m.is_setup for m in tp.bases)
        assert not tp.bases[0].knot_vector.flags.writeable

    def test_the_returned_penalty_is_frozen(self, z2):
        """The penalty is built at setup and handed out as that stored array, so a
        caller cannot corrupt the basis -- and finds out immediately rather than
        editing a copy the basis then ignores."""
        for b in (
            bspline(degree=3, k=6, penalty=2).setup(),
            ThinPlateBasis(d=2, k=10, m=2, penalty=2).setup(z2),
            TensorProductBasis(
                bspline(degree=3, k=5, penalty=2), bspline(degree=3, k=4, penalty=2)
            ).setup(z2),
        ):
            P = b.penalty_matrix()
            assert not P.flags.writeable
            with pytest.raises(ValueError):
                P += 1.0
            assert b.penalty_matrix() is P


# ---------------------------------------------------------------------------------
# end to end -- pinned LinearSmoother fits
# ---------------------------------------------------------------------------------

CURVE_GRID = np.linspace(0.1, 0.9, 5)
SURFACE_GRID = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.15], [0.35, 0.9], [0.6, 0.7]])

SMOOTHS = {
    "b-spline": dict(
        basis=lambda: bspline(degree=3, k=8, penalty=2),
        data="curve",
        lmbda=0.1501657419717497,
        edf=6.341757361914052,
        rss=0.3989076919206413,
        predict=[
            0.6609870470148846,
            1.0310728257154687,
            0.11121062534032994,
            -0.8489330256560607,
            -0.7372941511479244,
        ],
    ),
    "tensor product, integral": dict(
        basis=lambda: bs(
            degree=3, k=(5, 5), knots="uniform", limits=(0.0, 1.0), penalty=2
        ),
        data="surface",
        lmbda=1.6883082431706369,
        edf=12.009163299183477,
        rss=0.2499450240642584,
        predict=[
            0.4154646497638652,
            0.03932336502587188,
            -0.05951987932941701,
            -0.8006928709980652,
            -0.35016172057718953,
        ],
    ),
    "tensor product, identity": dict(
        basis=lambda: bs(
            degree=3, k=(5, 5), knots="uniform", limits=(0.0, 1.0),
            penalty=2, fan_out="identity",
        ),
        data="surface",
        lmbda=2.9597818528690008,
        edf=11.413443560504954,
        rss=0.25309003402459285,
        predict=[
            0.4307521889612438,
            0.053256498789186646,
            -0.08178746296070327,
            -0.794895798631797,
            -0.34701226974214555,
        ],
    ),
    "thin plate, d=2": dict(
        basis=lambda: ThinPlateBasis(d=2, k=12, m=2, penalty=2),
        data="surface",
        lmbda=0.061890798408959856,
        edf=11.450441376185175,
        rss=0.2667588143211898,
        predict=[
            0.4380452351549024,
            0.045458369583516844,
            0.1242624871519093,
            -0.7433486053877122,
            -0.3625782279270719,
        ],
    ),
    "thin plate, d=1": dict(
        basis=lambda: ThinPlateBasis(d=1, k=8, m=2, penalty=2),
        data="curve",
        lmbda=0.20923525210874566,
        edf=6.671082434319861,
        rss=0.41512185620499414,
        predict=[
            0.6583930345473312,
            1.0275465507086041,
            0.1084264584908228,
            -0.8481197092183312,
            -0.7299376279297037,
        ],
    ),
}


@pytest.mark.parametrize("case", list(SMOOTHS))
class TestSmootherPins:
    """The values that actually have to keep working: a GCV-selected fit through
    each basis type, on fixed-seed data. Regression pins -- there is no independent
    reference for a smoothing-parameter search -- so a change here is either a bug or
    a decision.

    The two tensor-product `lmbda`s were a decision. Both those designs are rank
    deficient with an `S0` at roundoff (1.8e-16 and 7.1e-15, against floors of 9.4e-15
    and 8.5e-14), so their old literals came through a division by roundoff in
    `_pinv_solve`; flooring it moved them by 1.1e-12 and 2.8e-12. That is inside the
    band those two move over anyway -- perturbing `S` by 1e-15 of its own scale shifts
    them 6.3e-13 and 4.4e-12, with the floor and without it -- so `rtol=1e-12` here is
    tighter than the search's own reproducibility on a rank-deficient design. It is kept
    tight regardless: for fixed code and fixed data these are bit-exact, and that is
    what makes the pin notice anything at all.

    The identity fan-out `lmbda` was a decision too: computing `omega` as
    `1 / sigma**2` from the data's singular values, rather than by eigendecomposing the
    whitened penalty (see `linear_model._diagonalise`), moved it by 1.3e-12 -- inside
    the same band -- with its EDF, RSS and predictions unchanged at their pins."""

    @staticmethod
    def _fit(case):
        spec = SMOOTHS[case]
        z, y = curve_data() if spec["data"] == "curve" else surface_data()
        s = LinearSmoother(z, spec["basis"](), lmbda_criterion=lm.gcv())
        s.set_y(y)
        s.fit_lmbda().fit()
        return s, spec, (CURVE_GRID if spec["data"] == "curve" else SURFACE_GRID)

    def test_selected_lmbda(self, case):
        s, spec, _ = self._fit(case)
        assert_allclose(s.lmbda, spec["lmbda"], rtol=1e-12)

    def test_edf_and_rss(self, case):
        s, spec, _ = self._fit(case)
        assert_allclose(s.edf(), spec["edf"], rtol=1e-9)
        assert_allclose(s.rss(), spec["rss"], rtol=1e-9)

    def test_predictions_on_a_fixed_grid(self, case):
        s, spec, grid = self._fit(case)
        assert_allclose(s.predict(grid), spec["predict"], rtol=1e-9)

    def test_the_fit_is_the_parent_class_on_the_two_matrices(self, case):
        """`LinearSmoother` is sugar, so the pinned numbers must also be what the
        basis's own matrices give through `LinearModel`."""
        s, spec, _ = self._fit(case)
        ref = lm.LinearModel(
            s.basis.model_matrix(s.z), s.basis.penalty_matrix(), lmbda_criterion=lm.gcv()
        )
        ref.set_y(s.y)
        ref.fit_lmbda().fit()
        assert ref.lmbda == s.lmbda
        close(ref.predict(), s.predict(), tol=1e-10)

    def test_edf_is_the_hat_trace(self, case):
        """`LinearModel` normalises `omega`, so `s.lmbda` is not the multiplier on this
        `S` and the two parameterisations of the path differ by a constant. EDF labels
        the path from either side, so it is what lines them up -- and once lined up the
        fitted values must agree too, which an EDF on its own would not catch."""
        s, _, _ = self._fit(case)
        X, S = s.basis.model_matrix(s.z), s.basis.penalty_matrix()
        # lstsq, not solve: `X'X + l S` has condition ~1e17 at every `l` for the
        # tensor bases, so `solve` raises on whichever pivot LAPACK happens to call
        # exactly zero -- which `l` that is depends on where brentq steps.
        hat = lambda l: X @ np.linalg.lstsq(X.T @ X + l * S, X.T, rcond=None)[0]
        rho = brentq(
            lambda rho: np.trace(hat(np.exp(rho))) - s.edf(), np.log(1e-14), np.log(1e14)
        )
        close(hat(np.exp(rho)) @ s.y, s.predict(), tol=1e-8)
