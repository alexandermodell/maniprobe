"""Tests for `bilinear`.

The joint fit, checked against a dense generalised eigenproblem and a dense KKT system
assembled independently of the diagonal-basis machinery under test. Moved here whole
from `test_linear_model.py` when `LinearModel.fit_jointly` became `EigenFit.fit`;
the references are unchanged, so a failure here means the same thing it did before.

The dense references are imported from `test_linear_model` rather than copied. They are
that module's checked machinery -- `model_lmbda` in particular translates a `lmbda` in
the units of `S` into the model's own, and is verified there against `trace(dense_hat)`
-- and a second copy would be free to drift away from the one under test.
"""

import numpy as np
import pytest
from scipy.linalg import eigh

from maniprobe.bilinear import BilinearModel, EigenFit
from maniprobe.linear_model import LinearModel
from test_linear_model import N, P, model_lmbda, psd, up_to_sign


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def pair(rng):
    """Two designs and penalties. `X1` is centered, so `sum(X1 @ beta1) == 0` holds for
    every `beta1` -- the precondition `EigenFit` documents and does not check."""
    X1, X2 = rng.normal(size=(N, P)), rng.normal(size=(N, P - 2))
    return X1 - X1.mean(0), psd(rng, P), X2, psd(rng, P - 2)


def joint_objective(X1, S1, l1, X2, S2, l2, m1, m2):
    """The objective as stated, evaluated on two fitted models."""
    r = X1 @ m1.beta - X2 @ m2.beta - m2.b
    return r @ r + l1 * m1.beta @ S1 @ m1.beta + l2 * m2.beta @ S2 @ m2.beta


def dense_joint(X1, S1, l1, X2, S2, l2, fit_intercept2=True):
    """Reference joint fit: explicit hat matrix, then a generalized eigenproblem.

    Profiling the second model out with its own hat matrix `H2` leaves

        min beta' A beta   s.t.   beta' B beta = n,
        A = X1'(I - H2)X1 + l1 S1,      B = X1'X1

    so `beta` is the smallest eigenvector of the *pencil* `(A, B)`, rescaled.

    `B` is singular whenever `X1` is rank-deficient -- which a centred basis always is --
    so `null(X1)` is profiled out of `A` first and the pencil is solved on the row space.
    What is independent here is the formulation: a dense hat matrix and a generalized
    problem in beta-space, against the code's standard problem in the pushforward basis.
    Not independent, and deliberately so: the *idea* of profiling a null space out by a
    Schur complement. It is applied to `A` rather than to `S`, through `pinv` rather than
    `_pinv_solve`, and with its own rank cut -- but a bug in that idea would be invisible
    to this reference.

    Returns `(objective, X1 @ beta)`.
    """
    n = len(X1)
    if fit_intercept2:
        X2 = X2 - X2.mean(0)
        H2 = np.ones((n, n)) / n + X2 @ np.linalg.solve(X2.T @ X2 + l2 * S2, X2.T)
    else:
        H2 = X2 @ np.linalg.solve(X2.T @ X2 + l2 * S2, X2.T)
    A = X1.T @ (np.eye(n) - H2) @ X1 + l1 * S1
    A, B = (A + A.T) / 2, X1.T @ X1

    _, D, Vt = np.linalg.svd(X1, full_matrices=X1.shape[0] < X1.shape[1])
    Q1, Q0 = Vt[: (D > D.max() * 1e-10).sum()].T, Vt[(D > D.max() * 1e-10).sum() :].T
    Z = np.linalg.pinv(Q0.T @ A @ Q0) @ (Q0.T @ A @ Q1)  # empty when X1 is full rank
    Ared = Q1.T @ A @ Q1 - (Q0.T @ A @ Q1).T @ Z
    mu, V = eigh((Ared + Ared.T) / 2, Q1.T @ B @ Q1)
    beta = (Q1 - Q0 @ Z) @ V[:, 0]
    return n * mu[0], X1 @ beta * np.sqrt(n / (beta @ B @ beta))


def fit_pair(X1, S1, l1, X2, S2, l2, fit_intercept=True):
    """Defaults to an intercept on the second model, since the objective has a `b2`.
    Both lmbdas are in the units of the penalties passed, as `dense_joint` and
    `joint_objective` read them."""
    m1, m2 = LinearModel(X1, S1), LinearModel(X2, S2, fit_intercept=fit_intercept)
    m1.set_lmbda(model_lmbda(X1, S1, l1))
    m2.set_lmbda(model_lmbda(X2, S2, l2, fit_intercept))
    return EigenFit(m1, m2).fit().lm1, m2


class TestFit:
    @pytest.mark.parametrize("l1, l2", [(0.0, 0.7), (0.3, 0.7), (1e4, 1e3)])
    @pytest.mark.parametrize("fit_intercept2", [True, False])
    def test_matches_a_dense_generalized_eigenproblem(self, pair, l1, l2,
                                                      fit_intercept2):
        X1, S1, X2, S2 = pair
        m1, m2 = fit_pair(X1, S1, l1, X2, S2, l2, fit_intercept=fit_intercept2)
        want, want_f = dense_joint(X1, S1, l1, X2, S2, l2, fit_intercept2)
        got = joint_objective(X1, S1, l1, X2, S2, l2, m1, m2)
        assert np.isclose(got, want, rtol=1e-10)
        assert up_to_sign(m1.predict(), want_f) < 1e-9

    def test_lmbdas_may_be_passed_or_preset(self, pair):
        """The two spellings have to agree, since `fit(l1, l2)` is defined as setting
        them on the models and then fitting: same coefficients, and the models left
        carrying the `lmbda`s they were fitted at."""
        X1, S1, X2, S2 = pair
        l1, l2 = model_lmbda(X1, S1, 0.3), model_lmbda(X2, S2, 0.7, True)
        preset, preset2 = fit_pair(X1, S1, 0.3, X2, S2, 0.7)

        m1, m2 = LinearModel(X1, S1), LinearModel(X2, S2, fit_intercept=True)
        EigenFit(m1, m2).fit(l1, l2)
        assert m1.lmbda == l1 and m2.lmbda == l2
        assert np.allclose(m1.beta, preset.beta)
        assert np.allclose(m2.beta, preset2.beta)

    def test_satisfies_both_constraints(self, pair):
        X1, S1, X2, S2 = pair
        m1, _ = fit_pair(X1, S1, 0.3, X2, S2, 0.7)
        f = m1.predict()
        assert abs(f.mean()) < 1e-12
        assert np.isclose(f @ f / N, 1.0, rtol=1e-12)

    def test_normalize_beta_afterwards_is_a_no_op(self, pair):
        X1, S1, X2, S2 = pair
        m1, _ = fit_pair(X1, S1, 0.3, X2, S2, 0.7)
        before = m1.beta.copy()
        assert np.allclose(m1.normalize_beta().beta, before)

    def test_second_model_is_fitted_to_the_first(self, pair):
        X1, S1, X2, S2 = pair
        m1, m2 = fit_pair(X1, S1, 0.3, X2, S2, 0.7)
        alone = LinearModel(X2, S2, fit_intercept=True)
        alone.set_lmbda(model_lmbda(X2, S2, 0.7, True))
        alone.set_y(m1.predict())
        alone.fit()
        assert np.allclose(m2.beta, alone.beta)
        assert np.isclose(m2.b, alone.b)

    def test_beats_fit_then_normalize(self, pair):
        """`fit` uses the unconstrained weight of 1 where the joint fit uses the scale
        constraint's multiplier, so it returns a different *direction*, not merely a
        different scale, and must be strictly worse."""
        X1, S1, X2, S2 = pair
        m1, m2 = fit_pair(X1, S1, 0.3, X2, S2, 0.7)
        got = joint_objective(X1, S1, 0.3, X2, S2, 0.7, m1, m2)

        naive = LinearModel(X1, S1)
        naive.set_lmbda(model_lmbda(X1, S1, 0.3))
        naive.set_y(m2.predict())
        naive.fit().normalize_beta()
        refit = LinearModel(X2, S2, fit_intercept=True)
        refit.set_lmbda(model_lmbda(X2, S2, 0.7, True))
        refit.set_y(naive.predict())
        refit.fit()
        worse = joint_objective(X1, S1, 0.3, X2, S2, 0.7, naive, refit)
        assert worse > got * (1 + 1e-6)

    def test_agrees_with_fit_then_normalize_when_unpenalised(self, pair):
        """The one case where they coincide: at lmbda1 = 0 the shrinkage filter is flat
        in `i`, so only the scale differs and normalising recovers the joint answer."""
        X1, S1, X2, S2 = pair
        m1, m2 = fit_pair(X1, S1, 0.0, X2, S2, 0.7)
        naive = LinearModel(X1, S1)
        naive.set_lmbda(0.0)
        naive.set_y(m2.predict())
        naive.fit().normalize_beta()
        assert up_to_sign(naive.predict(), m1.predict()) < 1e-9

    def test_exact_when_the_second_model_can_reproduce_the_first(self, pair):
        """With `col(X1)` inside `col(X2)`, an unpenalised `lm2` reproduces any `f`, so
        `H2` is the identity on `col(X1)`; at lmbda1 = 0 that makes `M = -I`, every
        direction optimal, and the objective exactly 0."""
        X1, S1, _, _ = pair
        X2 = np.column_stack([X1, np.arange(N, dtype=float)])
        m1, m2 = fit_pair(X1, S1, 0.0, X2, None, 0.0)
        assert np.allclose(m1.predict(), m2.predict(), atol=1e-9)
        assert joint_objective(X1, S1, 0.0, X2, np.eye(P + 1), 0.0, m1, m2) < 1e-16

    @pytest.mark.parametrize("shape", [(12, 20), (30, 8)])
    def test_rank_deficient_design(self, rng, shape):
        """`n < p1`, and a design one column short of full rank -- the shape a centred
        basis produces, where `beta` is only determined once the penalty picks a lift
        out of `null(X1)`."""
        n, p = shape
        X1, X2 = rng.normal(size=(n, p)), rng.normal(size=(n, 5))
        if n > p:
            X1[:, -1] = X1[:, :-1].sum(1)  # exactly rank p - 1
        X1 -= X1.mean(0)
        S1, S2 = psd(rng, p), psd(rng, 5)
        m1, m2 = fit_pair(X1, S1, 0.3, X2, S2, 0.7)
        want, want_f = dense_joint(X1, S1, 0.3, X2, S2, 0.7)
        f = m1.predict()
        assert abs(f.mean()) < 1e-12
        assert np.isclose(f @ f / n, 1.0, rtol=1e-12)
        assert np.isclose(joint_objective(X1, S1, 0.3, X2, S2, 0.7, m1, m2), want,
                          rtol=1e-9)
        assert up_to_sign(f, want_f) < 1e-8

    @pytest.mark.parametrize("seed", range(10))
    def test_sign_is_fixed_by_the_largest_coordinate(self, seed):
        """A convention, not something the data decides. Swept over seeds because
        `eigh`'s raw output already satisfies it about half the time, so one fixture
        does not pin it -- dropping the convention left a single-seed test passing."""
        rng = np.random.default_rng(seed)
        X1, X2 = rng.normal(size=(N, P)), rng.normal(size=(N, P - 2))
        m1, _ = fit_pair(X1 - X1.mean(0), psd(rng, P), 0.3, X2, psd(rng, P - 2), 0.7)
        nu = m1.pushforward(m1.beta)
        assert nu[np.abs(nu).argmax()] > 0

    def test_leaves_y_unset(self, pair):
        """Deliberate: with `y` set, a later `fit` would silently return a different
        answer -- see test_beats_fit_then_normalize."""
        X1, S1, X2, S2 = pair
        m1, _ = fit_pair(X1, S1, 0.3, X2, S2, 0.7)
        assert m1.y is None
        with pytest.raises(ValueError, match="y is not set"):
            m1.rss()

    def test_clears_a_stale_y_on_the_second_model(self, pair):
        """`lm2` is fitted to a response that is never materialised -- `U2.T @ y2` is
        read off `K` instead -- so a `y` set before the fit plays no part in it, and
        left in place would outlive the fit it never belonged to and make `rss`
        describe the wrong response."""
        X1, S1, X2, S2 = pair
        m1, m2 = LinearModel(X1, S1), LinearModel(X2, S2, fit_intercept=True)
        m2.set_y(np.arange(N, dtype=float))
        m1.set_lmbda(model_lmbda(X1, S1, 0.3))
        m2.set_lmbda(model_lmbda(X2, S2, 0.7, True))
        EigenFit(m1, m2).fit()
        assert m2.y is None and m2.y_mean is None
        with pytest.raises(ValueError, match="y is not set"):
            m2.rss()

    @pytest.mark.parametrize("which", ["lm1", "lm2"])
    def test_rejects_a_model_constrained_after_construction(self, pair, which):
        """`K` is formed from the two `U`s at construction, and a constraint rebuilds one
        of them: the fit would otherwise pair a cache with designs that no longer match
        it, silently. Caught on the rank, which is the thing a constraint that constrains
        anything always changes."""
        X1, S1, X2, S2 = pair
        m1, m2 = LinearModel(X1, S1), LinearModel(X2, S2)
        m1.set_lmbda(1.0)
        m2.set_lmbda(1.0)
        bl = EigenFit(m1, m2)
        target = m1 if which == "lm1" else m2
        rng = np.random.default_rng(7)
        target.add_constraints(rng.normal(size=target.X.shape[1]))
        with pytest.raises(ValueError, match="no longer carry"):
            bl.fit()

    def test_requires_no_intercept(self, pair):
        X1, S1, X2, S2 = pair
        m1, m2 = LinearModel(X1, S1, fit_intercept=True), LinearModel(X2, S2)
        m1.set_lmbda(0.3)
        m2.set_lmbda(0.7)
        with pytest.raises(ValueError, match="fit_intercept=False"):
            EigenFit(m1, m2).fit()

    @pytest.mark.parametrize("which", ["self", "other"])
    def test_requires_both_lmbdas(self, pair, which):
        X1, S1, X2, S2 = pair
        m1, m2 = LinearModel(X1, S1), LinearModel(X2, S2)
        (m2 if which == "self" else m1).set_lmbda(0.3)
        with pytest.raises(ValueError, match="lmbda is not set"):
            EigenFit(m1, m2).fit()

    @pytest.mark.parametrize("cond, tol", [(1e1, 1e-11), (1e5, 1e-5)])
    def test_accuracy_is_set_by_the_eigenvalue_gap(self, rng, cond, tol):
        """NUMERICAL. The error here scales as `EPS * ||M|| / gap`, not as the usual
        `EPS * ||A_inv||`: `eigh`'s backward error is relative to `||M||`, and what is
        kept is the eigen*vector*, whose sensitivity carries the gap in the denominator.
        `||M||` inherits the dynamic range of `omega`, so the loss tracks `cond(X1)^2`:
        measured 2e-9 relative in `f` at `cond = 1e5`, against 2e-14 at `cond = 10`.
        lmbda barely matters: it inflates `||M||` and the gap together. Accepted, not
        fixed; the tolerances below are pinned so a change in it is visible."""
        U, _ = np.linalg.qr(rng.normal(size=(N, P)))
        V, _ = np.linalg.qr(rng.normal(size=(P, P)))
        X1 = (U * np.geomspace(1.0, 1.0 / cond, P)) @ V.T
        X1 -= X1.mean(0)
        X2, S1, S2 = rng.normal(size=(N, P - 2)), psd(rng, P), psd(rng, P - 2)

        m1, m2 = fit_pair(X1, S1, 1e-2, X2, S2, 0.5)
        _, want_f = dense_joint(X1, S1, 1e-2, X2, S2, 0.5)
        assert up_to_sign(m1.predict(), want_f) / np.sqrt(N) < tol


class TestAddConstraints:
    """Deflation of a sequence of fits, by one call on the pair."""

    def test_constrains_lm1_alone(self, pair):
        """`u` tolerates a constrained `lm1` and refuses a constrained `lm2`, which is
        the way round this constrains -- so the rows land on one model and the other is
        left as it was."""
        X1, S1, X2, S2 = pair
        m1, m2 = LinearModel(X1, S1), LinearModel(X2, S2, fit_intercept=True)
        bl = EigenFit(m1, m2)
        c = np.random.default_rng(7).normal(size=P)

        assert bl.add_constraints(c) is bl
        assert np.array_equal(m1.constraints, np.atleast_2d(c))
        assert m2.constraints is None

    def test_successive_fits_are_orthogonal_on_the_training_rows(self, pair):
        """INDEPENDENT. The deflation itself, in plain numpy on the fitted indices:
        `add_constraints(beta1)` imposes `(X1 beta1_prev).T @ (X1 beta1) == 0`, which is
        `f_prev . f_new == 0`. Three fits, so the second constraint has to hold
        alongside the first rather than replace it, and with the scale constraint that
        makes the three an orthonormal frame under the empirical measure."""
        X1, S1, X2, S2 = pair
        m1, m2 = LinearModel(X1, S1), LinearModel(X2, S2, fit_intercept=True)
        bl = EigenFit(m1, m2)

        fs = []
        for _ in range(3):
            bl.fit(0.3, 0.7)
            fs.append(X1 @ m1.beta)
            bl.add_constraints(m1.beta)

        F = np.column_stack(fs)
        assert np.allclose(F.T @ F / N, np.eye(3), atol=1e-10)

    def test_matches_a_pair_constrained_at_construction(self, pair):
        """INDEPENDENT of the mutator. The same rows through `LinearModel`'s
        constructor, which imposes them inside build step 3 rather than on the finished
        cache: two different code paths that must reach one model, to roundoff rather
        than exactly, as `test_linear_model.test_the_two_routes_agree` holds them
        generally.

        This is also what pins the `K` refresh. Without it `fit` raises on the rank, and
        a refresh formed from anything but the rebuilt `U1` would put this second fit
        somewhere else entirely."""
        X1, S1, X2, S2 = pair
        m1, m2 = LinearModel(X1, S1), LinearModel(X2, S2, fit_intercept=True)
        bl = EigenFit(m1, m2).fit(0.3, 0.7)
        c = m1.beta.copy()
        bl.add_constraints(c).fit(0.3, 0.7)

        fresh1 = LinearModel(X1, S1, constraints=c)
        fresh = EigenFit(fresh1, LinearModel(X2, S2, fit_intercept=True))
        fresh.fit(0.3, 0.7)

        assert np.allclose(m1.omega, fresh1.omega, rtol=1e-11, atol=0.0)
        assert up_to_sign(m1.predict(), fresh1.predict()) / np.sqrt(N) < 1e-10

    def test_converged_starts_true(self, pair):
        """A class attribute rather than something `fit` sets: an eigendecomposition
        has no convergence test to stop short of, so the answer it gives before a fit is
        the one it gives after."""
        X1, S1, X2, S2 = pair
        bl = EigenFit(LinearModel(X1, S1), LinearModel(X2, S2))
        assert bl.converged is True
        assert bl.fit(0.3, 0.7).converged is True


class TestR2:
    """`BilinearModel.r2`, which both subclasses inherit and neither overrides."""

    def test_matches_the_ratio_it_is_defined_as(self, pair):
        """INDEPENDENT. Against the expression written out at arbitrary coefficients,
        rather than at fitted ones: `r2` is a function of two predictions and must not
        depend on their being optimal. `beta` is set directly, so nothing here is a
        second run of the code that produced it."""
        X1, S1, X2, S2 = pair
        m1, m2 = LinearModel(X1, S1), LinearModel(X2, S2)
        rng = np.random.default_rng(7)
        m1.beta, m1.b = rng.normal(size=P), 0.0
        m2.beta, m2.b = rng.normal(size=P - 2), 0.0

        f, g = X1 @ m1.beta, X2 @ m2.beta
        want = 1.0 - np.mean((f - g) ** 2) / f.var()
        assert np.isclose(EigenFit(m1, m2).r2(), want, rtol=1e-12)

    def test_divides_by_the_spread_of_f_not_by_n(self, pair):
        """The denominator is `var(f)`, which off the training rows is not 1. Scaling
        both fitted values by the same factor leaves `r2` unchanged; scaling only `g`
        does not -- so the ratio is doing what the docstring claims and not silently
        dividing by a constant."""
        X1, S1, X2, S2 = pair
        m1, m2 = LinearModel(X1, S1), LinearModel(X2, S2)
        rng = np.random.default_rng(7)
        m1.beta, m1.b = rng.normal(size=P), 0.0
        m2.beta, m2.b = rng.normal(size=P - 2), 0.0
        bl = EigenFit(m1, m2)

        base = bl.r2()
        m1.beta, m2.beta = m1.beta * 3.0, m2.beta * 3.0
        assert np.isclose(bl.r2(), base, rtol=1e-12)
        m2.beta = m2.beta * 2.0
        assert not np.isclose(bl.r2(), base)


class TestU:
    """`BilinearModel.u`, the loadings of `lm2`'s predictors on `lm1`'s index."""

    @pytest.mark.parametrize("fit_intercept2", [True, False])
    def test_matches_the_covariance_it_is_defined_as(self, pair, fit_intercept2):
        """INDEPENDENT, against the sum written out over the rows, and at an arbitrary
        `beta1` rather than a fitted one: `u` is a function of `f` and `X2` and must not
        depend on `f` being optimal. Nothing here re-runs the cache route -- the
        reference centres `X2` explicitly and forms `f` from the design.

        Both intercept settings, because the cache holds `X2` centred with one and
        uncentred without, and `u` is claimed to be the same vector either way. It is
        because `X1` is centred, so `mean(f) == 0` and the term that differs drops.
        """
        X1, S1, X2, S2 = pair
        m1 = LinearModel(X1, S1)
        m2 = LinearModel(X2, S2, fit_intercept=fit_intercept2)
        m1.beta, m1.b = np.random.default_rng(7).normal(size=P), 0.0

        f = X1 @ m1.beta
        want = (X2 - X2.mean(0)).T @ f / N
        assert np.allclose(EigenFit(m1, m2).u(), want, atol=1e-12 * np.abs(want).max())

    def test_is_each_predictors_correlation_scaled_by_its_spread(self, pair):
        """The reading the docstring leads with: `sd(f) == 1` on the training data, so
        `u[j] / sd(x_j)` is `corr(x_j, f)`. Needs a fitted pair, the scale constraint
        being what makes the denominator 1, and `np.corrcoef` is the reference."""
        X1, S1, X2, S2 = pair
        m1, m2 = fit_pair(X1, S1, 0.3, X2, S2, 0.7)
        bl = EigenFit(m1, m2)
        f, u = m1.predict(), bl.u()
        for j in range(X2.shape[1]):
            want = np.corrcoef(X2[:, j], f)[0, 1]
            assert np.isclose(u[j] / X2[:, j].std(), want, rtol=1e-10)

    def test_the_cache_route_agrees_with_forming_f(self, pair):
        """`EigenFit` overrides `_ybar2` to reach `U2.T @ f` as `K.T @ nu`, forming no
        n-sized array; the base class forms `f` and projects it, which is what
        `als.ALSFit` uses. The two must be the same vector, so they are compared on one
        fitted state rather than each against its own."""
        X1, S1, X2, S2 = pair
        m1, m2 = fit_pair(X1, S1, 0.3, X2, S2, 0.7)
        bl = EigenFit(m1, m2)
        cached, formed = bl._ybar2(), BilinearModel._ybar2(bl)
        assert np.allclose(cached, formed, atol=1e-12 * np.abs(formed).max())

    def test_is_unavailable_for_a_constrained_lm2(self, pair):
        """The cache cannot see past a constraint on `lm2`: `U2` no longer spans
        `col(Xc2)`, so `A_inv2.T @ (U2.T @ f)` is the covariance with the projection of
        `f` and not with `f`. A constraint on `lm1` is a different matter -- `f` is
        `X1 @ beta1` regardless -- so that case is checked to still agree with the
        reference rather than to raise."""
        X1, S1, X2, S2 = pair
        rng = np.random.default_rng(7)
        bound2 = LinearModel(X2, S2, constraints=rng.normal(size=P - 2))
        m1 = LinearModel(X1, S1)
        m1.beta, m1.b = rng.normal(size=P), 0.0
        with pytest.raises(ValueError, match="constrained lm2"):
            EigenFit(m1, bound2).u()

        bound1 = LinearModel(X1, S1, constraints=rng.normal(size=P))
        bound1.set_lmbda(1.0)
        m2 = LinearModel(X2, S2)
        m2.set_lmbda(1.0)
        bl = EigenFit(bound1, m2).fit()
        f = bound1.predict()
        want = (X2 - X2.mean(0)).T @ f / N
        assert np.allclose(bl.u(), want, atol=1e-12 * np.abs(want).max())
