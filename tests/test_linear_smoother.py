"""Tests for `linear_smoother`.

`LinearSmoother` is a thin adapter, so the reference for every numerical test is the
parent class built directly from the basis's own two matrices -- the code path the
class is sugar for. What that leaves to check is the seam: the setup lifecycle, and
that prediction rebuilds the design through the same frozen basis.
"""

import numpy as np
import pytest

from maniprobe import linear_model as lm
from maniprobe.basis import BSplineBasis, ThinPlateBasis, bs
from maniprobe.bilinear import EigenFit
from maniprobe.linear_smoother import LinearSmoother

N, K = 60, 10


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def z(rng):
    return np.sort(rng.uniform(0, 1, N))


@pytest.fixture
def y(rng, z):
    return np.sin(6 * z) + 0.1 * rng.normal(size=N)


def spline(penalty=2):
    """A fresh unconfigured basis: setup is once-only, so tests cannot share one."""
    return BSplineBasis(degree=3, k=K, penalty=penalty)


def fitted(model, y, lmbda=1e-3):
    model.set_y(y)
    model.set_lmbda(lmbda)
    return model.fit()


def reference(basis, z):
    """The parent class, built from the basis by hand."""
    return lm.LinearModel(basis.model_matrix(z), basis.penalty_matrix())


# ---------------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------------


class TestConstruction:
    def test_sets_up_an_unconfigured_basis(self, z):
        b = spline()
        assert not b.is_setup
        s = LinearSmoother(z, b)
        assert b.is_setup
        assert s.X.shape == (N, K)

    def test_takes_a_configured_basis_as_is(self, rng, z):
        """Knots placed on other data survive: the basis is used, not reconfigured."""
        other = np.sort(rng.uniform(-0.5, 1.5, N))
        b = spline()
        b.setup(other)
        knots = b.knot_vector.copy()

        s = LinearSmoother(z, b)

        assert np.array_equal(b.knot_vector, knots)
        assert np.array_equal(s.X, b.model_matrix(z))
        # and those knots really are not the ones z would have placed
        assert not np.allclose(knots, spline().setup(z).knot_vector)

    def test_stores_z_uncoerced(self, z):
        zlist = list(z)
        assert LinearSmoother(zlist, spline()).z is zlist

    def test_design_comes_from_the_basis(self, z):
        b = spline()
        s = LinearSmoother(z, b)
        assert np.array_equal(s.X, b.model_matrix(s.z))

    def test_unpenalised_basis_raises(self, z):
        with pytest.raises(ValueError, match="without a penalty"):
            LinearSmoother(z, spline(penalty=None))

    def test_criterion_is_attached(self, z):
        s = LinearSmoother(z, spline(), lmbda_criterion=lm.gcv())
        assert s.lmbda_criterion.model is s

    def test_singular_penalty_survives_the_cache(self, z):
        """A real spline penalty is rank-deficient -- the parent's Schur path."""
        b = spline()
        s = LinearSmoother(z, b)
        assert np.linalg.matrix_rank(b.penalty_matrix()) == K - b.null_space_dim
        assert np.count_nonzero(s.omega == 0) == b.null_space_dim


# ---------------------------------------------------------------------------------
# equivalence with the parent
# ---------------------------------------------------------------------------------


class TestEquivalence:
    def test_fit_matches_linear_model(self, z, y):
        b = spline()
        s = fitted(LinearSmoother(z, b), y)
        ref = fitted(reference(b, z), y)
        assert np.allclose(s.beta, ref.beta)
        assert np.isclose(s.edf(), ref.edf())
        assert np.isclose(s.rss(), ref.rss())

    @pytest.mark.parametrize("criterion", [lm.gcv, lm.reml, lm.aic, lm.bic])
    def test_criterion_selects_the_same_lmbda(self, z, y, criterion):
        b = spline()
        s = LinearSmoother(z, b, lmbda_criterion=criterion())
        ref = reference(b, z)
        ref.set_lmbda_criterion(criterion())
        s.set_y(y)
        ref.set_y(y)
        assert s.fit_lmbda().lmbda == ref.fit_lmbda().lmbda

    def test_set_edf_round_trip(self, z):
        s = LinearSmoother(z, spline())
        assert np.isclose(s.set_edf(5.0).edf(), 5.0)


# ---------------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------------


class TestPredict:
    def test_default_is_the_training_design(self, z, y):
        s = fitted(LinearSmoother(z, spline()), y)
        assert np.array_equal(s.predict(), s.predict(z))

    def test_new_covariates_go_through_the_basis(self, rng, z, y):
        b = spline()
        s = fitted(LinearSmoother(z, b), y)
        zp = np.linspace(0.05, 0.95, 17)
        assert np.allclose(s.predict(zp), b.model_matrix(zp) @ s.beta)

    def test_raises_before_fit(self, z):
        s = LinearSmoother(z, spline())
        with pytest.raises(ValueError, match="not fitted"):
            s.predict(z)

    def test_derivatives_via_the_basis(self, z, y):
        """Not `predict`'s job: the fitted curve is the basis's callable, and its
        derivative is the basis's own `derivative_matrix` against the same beta."""
        s = fitted(LinearSmoother(z, spline()), y)
        f = s.basis.function(s.beta)
        assert np.allclose(f(z), s.predict())
        zp = np.linspace(0.1, 0.9, 9)
        step = 1e-6
        d = s.basis.derivative_matrix(zp, 1) @ s.beta
        assert np.allclose(d, (f(zp + step) - f(zp - step)) / (2 * step), atol=1e-6)


# ---------------------------------------------------------------------------------
# inherited machinery, exercised through the subclass
# ---------------------------------------------------------------------------------


class TestInherited:
    @pytest.mark.parametrize("route", ["constructor", "add_constraints"])
    def test_constraints_keep_the_invariant(self, z, y, route):
        """Sum-to-zero identifiability: B-spline columns are a partition of unity. Both
        routes, the constructor's being a passthrough this class has to remember to
        make -- the equivalence of the two is `test_linear_model`'s to pin."""
        if route == "constructor":
            s = LinearSmoother(z, spline(), constraints=np.ones(K))
        else:
            s = LinearSmoother(z, spline())
            s.add_constraints(np.ones(K))
        fitted(s, y)
        assert np.array_equal(s.constraints, np.ones((1, K)))
        assert np.allclose(s.X @ s.A, s.U)
        assert np.isclose(s.predict().sum(), 0.0)

    def test_normalize_beta(self, z, y):
        s = fitted(LinearSmoother(z, spline()), y).normalize_beta()
        assert np.isclose(np.sum(s.predict() ** 2) / N, 1.0)

    def test_joint_fit_against_a_linear_model(self, rng, z):
        """A centred basis is what the joint fit assumes: its columns have zero mean, so
        `sum(f) == 0` holds for every coefficient vector, and the design is rank `k - 1`
        because the constant combination is annihilated. Referenced against the parent
        built from the same two matrices."""
        b = BSplineBasis(degree=3, k=K, penalty=2, centre=True)
        s = LinearSmoother(z, b)
        assert np.linalg.matrix_rank(s.X) == K - 1
        assert np.abs(s.X.mean(0)).max() < 1e-14

        other = lm.LinearModel(rng.normal(size=(N, 4)), np.eye(4), fit_intercept=True)
        other.set_lmbda(1e-2)
        s.set_lmbda(1e-3)
        EigenFit(s, other).fit()

        ref = reference(b, z)
        ref.set_lmbda(1e-3)
        other_ref = lm.LinearModel(other.X, np.eye(4), fit_intercept=True)
        other_ref.set_lmbda(1e-2)
        EigenFit(ref, other_ref).fit()

        assert np.allclose(s.beta, ref.beta)
        assert np.allclose(other.beta, other_ref.beta)
        assert s.predict().mean() == pytest.approx(0.0, abs=1e-13)
        assert np.isclose(np.sum(s.predict() ** 2) / N, 1.0)
        # and prediction still goes back through the frozen basis
        zp = np.linspace(0.1, 0.9, 7)
        assert np.allclose(s.predict(zp), b.model_matrix(zp) @ s.beta)


# ---------------------------------------------------------------------------------
# the shape of z is the basis's business
# ---------------------------------------------------------------------------------


class TestMultidimensional:
    @pytest.fixture
    def z2(self, rng):
        return rng.uniform(0, 1, (N, 2))

    @pytest.fixture
    def y2(self, rng, z2):
        return np.sin(4 * z2[:, 0]) * np.cos(3 * z2[:, 1]) + 0.05 * rng.normal(size=N)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: bs(k=(6, 5), penalty=2),
            lambda: bs(k=(6, 5), penalty=2, fan_out="identity"),
            lambda: ThinPlateBasis(d=2, k=15, penalty=2),
        ],
    )
    def test_two_dimensional_bases(self, z2, y2, make):
        b = make()
        s = fitted(LinearSmoother(z2, b), y2)
        ref = fitted(reference(b, z2), y2)
        assert s.X.shape == (N, b.k)
        assert np.allclose(s.beta, ref.beta)
        zp = np.random.default_rng(1).uniform(0.1, 0.9, (7, 2))
        assert np.allclose(s.predict(zp), b.model_matrix(zp) @ s.beta)


# ---------------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------------


class TestSmoke:
    @pytest.mark.parametrize("criterion", [lm.gcv, lm.reml])
    def test_recovers_a_smooth_curve(self, rng, criterion):
        """`bs` centres, so the fit sums to zero over the training points and the
        level of `truth` is not in the model space -- `LinearSmoother` has no
        intercept of its own to put it back. The shape is what is recovered, so the
        truth is centred to match; the zero mean is asserted rather than tolerated."""
        z = np.sort(rng.uniform(0, 1, 200))
        truth = np.sin(6 * z)
        s = LinearSmoother(z, bs(k=20, penalty=2), lmbda_criterion=criterion())
        s.set_y(truth + 0.1 * rng.normal(size=200))
        s.fit_lmbda().fit()

        assert s.predict().mean() == pytest.approx(0.0, abs=1e-13)
        assert np.sqrt(np.mean((s.predict() - (truth - truth.mean())) ** 2)) < 0.05
        # and between the training points, where nothing was fitted
        grid = np.linspace(0.05, 0.95, 500)
        shape = np.sin(6 * grid) - truth.mean()
        assert np.sqrt(np.mean((s.predict(grid) - shape) ** 2)) < 0.05

    def test_recovers_a_latent_index(self, rng):
        """End to end: a smooth `f(z)` that a second model sees only through a noisy
        linear combination. `f` is identified only up to sign and scale, which is what
        the two constraints pin down -- so the truth is standardised to match."""
        z = np.sort(rng.uniform(0, 1, 200))
        truth = np.sin(2 * np.pi * z)
        truth = (truth - truth.mean()) / truth.std()
        X2 = np.column_stack(
            [truth + 0.3 * rng.normal(size=200), rng.normal(size=(200, 4))]
        )

        s = LinearSmoother(z, BSplineBasis(degree=3, k=15, penalty=2, centre=True))
        s.set_lmbda(1e-4)
        other = lm.LinearModel(X2, np.eye(5), fit_intercept=True)
        other.set_lmbda(1e-2)
        EigenFit(s, other).fit()

        f, corr = s.predict(), np.corrcoef(s.predict(), truth)[0, 1]
        assert abs(corr) > 0.99
        assert np.sqrt(np.mean((f * np.sign(corr) - truth) ** 2)) < 0.15

    def test_one_basis_many_responses(self, rng, z):
        """The workflow the cache exists for, with the basis built once."""
        s = LinearSmoother(z, spline())
        s.set_lmbda(1e-3)
        for _ in range(3):
            y = np.sin(6 * z) + 0.1 * rng.normal(size=N)
            s.set_y(y)
            assert np.allclose(s.fit().predict(), fitted(reference(s.basis, z), y).predict())
