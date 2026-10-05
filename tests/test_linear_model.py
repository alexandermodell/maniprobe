"""Tests for `linear_model`.

Everything here checks the diagonal-basis machinery against a dense reference built
independently -- an explicit solve, an explicit hat matrix, a KKT system, or a marginal
likelihood assembled from scratch. The point is that no test reuses the code path it is
testing.

The classes marked NUMERICAL cover traps that produced visibly wrong answers during
development and would not be caught by well-conditioned random data. Do not delete them
without reading the comments.
"""

import warnings

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.optimize import brentq

from maniprobe import linear_model as lm
from maniprobe.bilinear import EigenFit
from maniprobe.linalg import eig, svd
from maniprobe.linear_model import LinearModel


# ---------------------------------------------------------------------------------
# fixtures and dense references
# ---------------------------------------------------------------------------------

N, P = 40, 6


def psd(rng, p, rank=None):
    B = rng.normal(size=(p, rank or p))
    return B @ B.T


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def data(rng):
    """Well-conditioned design, full-rank penalty, offset response."""
    return rng.normal(size=(N, P)), psd(rng, P), rng.normal(size=N) + 2.0


def dense_beta(X, S, y, lmbda):
    """(X'X + lmbda S)^-1 X'y, the textbook solve."""
    S = np.eye(X.shape[1]) if S is None else S
    return np.linalg.solve(X.T @ X + lmbda * S, X.T @ y)


def augment(X, S):
    """Design and penalty with an explicit unpenalised intercept column."""
    p = X.shape[1]
    Sa = np.zeros((p + 1, p + 1))
    Sa[1:, 1:] = S
    return np.column_stack([np.ones(len(X)), X]), Sa


def dense_hat(X, S, lmbda, fit_intercept=False, C=None):
    """`y -> fitted`, from the same KKT system as `dense_kkt` when `C` is given."""
    if fit_intercept:
        X, S = augment(X, S)
    if C is None:
        return X @ np.linalg.solve(X.T @ X + lmbda * S, X.T)
    C = np.atleast_2d(C)
    G = C @ X.T @ X
    k = len(G)
    K = np.block([[X.T @ X + lmbda * S, G.T], [G, np.zeros((k, k))]])
    rhs = np.vstack([X.T, np.zeros((k, len(X)))])
    return X @ np.linalg.lstsq(K, rhs, rcond=None)[0][: X.shape[1]]


def dense_kkt(X, S, y, lmbda, C):
    """Constrained minimiser via Lagrange multipliers: (XC')'(Xb) = 0."""
    C = np.atleast_2d(C)
    G = C @ X.T @ X
    k = len(G)
    K = np.block([[X.T @ X + lmbda * S, G.T], [G, np.zeros((k, k))]])
    rhs = np.concatenate([X.T @ y, np.zeros(k)])
    return np.linalg.lstsq(K, rhs, rcond=None)[0][: X.shape[1]]


def constrained(X, S, C, route):
    """A constrained model, built by whichever route `route` names.

    The two are meant to produce the same cache -- `test_the_two_routes_agree` is where
    that is pinned -- so a test about constraints in general is free to take either.
    """
    if route == "constructor":
        return LinearModel(X, S, constraints=C)
    return LinearModel(X, S).add_constraints(C)


def model_lmbda(X, S, lmbda, fit_intercept=False, C=None):
    """Translate a `lmbda` in the units of `S` into the one the model uses.

    The constructor normalises `omega`, so a model's `lmbda` multiplies that normalised
    penalty rather than the `S` handed to a dense reference here; the two
    parameterisations of the same path differ by one constant. EDF labels that path from
    either side, so it is what lines them up: read `edf(1)` off the model, solve
    `trace(dense_hat(l)) == edf(1)` for `l` on the dense side, and the constant is
    `1 / l`. No test has to know how the constructor chose the normalisation, and `edf`
    against `trace(dense_hat)` is itself checked in `TestEdf`.

    A constraint renormalises `omega` -- build step 4 runs again -- so on a constrained
    model the constant belongs to the constrained spectrum, and the dense side has to be
    constrained the same way. Pass the rows the model carries.
    """
    S = np.eye(X.shape[1]) if S is None else S
    edf1 = LinearModel(X, S, fit_intercept=fit_intercept, constraints=C).edf(1.0)
    rho = brentq(
        lambda rho: np.trace(dense_hat(X, S, np.exp(rho), fit_intercept, C)) - edf1,
        np.log(1e-14),
        np.log(1e14),
    )
    return lmbda / np.exp(rho)


def fitted(X, S, y, lmbda, **kw):
    """Fit at a `lmbda` stated in the units of `S`, so it names the same fit as the
    dense references above."""
    m = LinearModel(X, S, **kw)
    m.set_y(y)
    m.set_lmbda(model_lmbda(X, S, lmbda, kw.get("fit_intercept", False)))
    return m.fit()


def up_to_sign(a, b):
    """`beta` and `-beta` satisfy the scale constraint alike, and nothing in the cache
    fixes the sign of a `U` column, so every comparison under that constraint is made
    against both."""
    return min(np.linalg.norm(a - b), np.linalg.norm(a + b))


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------


class TestHelpers:
    def test_svd_reconstructs_and_ranks(self, rng):
        A = rng.normal(size=(9, 4))
        U, D, Vt = svd(A)
        assert (U.shape, D.shape, Vt.shape) == ((9, 4), (4,), (4, 4))
        assert np.allclose((U * D) @ Vt, A)

    def test_svd_drops_everything_below_the_cut(self, rng):
        A = rng.normal(size=(9, 4))
        A[:, 3] = A[:, 0]
        U, D, Vt = svd(A)
        assert (U.shape, D.shape, Vt.shape) == ((9, 3), (3,), (3, 4))
        assert np.allclose((U * D) @ Vt, A)  # what was dropped was noise
        assert len(svd(A, atol=1e18)[1]) == 0

    def test_svd_null_space_spans_the_kernel(self, rng):
        A = rng.normal(size=(9, 4))
        A[:, 3] = A[:, 0]
        _, _, Vt, V0t = svd(A, return_null=True)
        assert V0t.shape == (1, 4)
        assert np.allclose(A @ V0t.T, 0.0)
        V = np.vstack([Vt, V0t])
        assert np.allclose(V @ V.T, np.eye(4))

    def test_svd_null_space_the_economy_factor_would_truncate(self, rng):
        # n < p: the economy right factor is (4, 9) and is missing 5 null directions.
        A = rng.normal(size=(4, 9))
        U, _, _, V0t = svd(A, return_null=True)
        assert (U.shape, V0t.shape) == ((4, 4), (5, 9))
        assert np.allclose(A @ V0t.T, 0.0)

    def test_svd_asks_for_the_economy_left_factor(self, monkeypatch, rng):
        # The complete left factor is (a, a): for a tall design, the one array the
        # cache exists never to form. `U[:, :r]` slices it away again, so the call
        # itself is the only place the rule can be pinned.
        asked = []
        real = np.linalg.svd

        def spy(M, full_matrices):
            asked.append(full_matrices)
            return real(M, full_matrices=full_matrices)

        monkeypatch.setattr(np.linalg, "svd", spy)
        svd(rng.normal(size=(9, 4)), return_null=True)  # economy V is already complete
        svd(rng.normal(size=(4, 9)), return_null=True)  # only here is full V needed
        assert asked == [False, True]

    def test_atol_is_a_floor_not_a_replacement(self):
        # `atol` may only raise the cut, never lower it: below the matrix's own
        # magnitude the plain relative rule must be untouched.
        M = np.diag([2.0, 1.0, 1e-20])
        assert len(svd(M, atol=1e-30)[1]) == len(svd(M)[1]) == 2
        assert (eig(M, atol=1e-30)[0] == eig(M)[0]).all()

    def test_eig_zeros_noise_and_reconstructs(self):
        w, W = eig(np.diag([2.0, 1e-20, 0.0]), keep_zeros=True)
        assert (w == np.array([0.0, 0.0, 2.0])).all()
        assert np.allclose((W * w) @ W.T, np.diag([2.0, 0.0, 0.0]))

    def test_eig_drops_the_zeros_when_not_keeping_them(self):
        w, W = eig(np.diag([2.0, 1e-20, 0.0]))
        assert (w.shape, W.shape) == ((1,), (3, 1))
        assert w[0] == 2.0

    def test_eig_keeps_genuine_negatives(self):
        # Definiteness is not assumed: a negative above the cut survives with its
        # sign, which is what lets `_build_cache` test for one, while a roundoff
        # negative is still zeroed.
        assert (eig(np.diag([1.0, -1.0]), keep_zeros=True)[0] == [-1.0, 1.0]).all()
        assert eig(np.diag([1.0, -1e-20]), keep_zeros=True)[0][0] == 0.0

    def test_eig_atol_zeroes_an_all_roundoff_matrix(self, rng):
        # A matrix that cancelled to noise has a spectrum of both signs at ~1e-19.
        # Measured against itself none of it is zeroed; measured against the magnitude
        # it would have had, all of it is.
        M = rng.normal(size=(4, 4)) * 1e-19
        M = M + M.T
        assert (eig(M, keep_zeros=True)[0] != 0.0).all()
        assert (eig(M, keep_zeros=True, atol=lm.EPS**2 * 1.0)[0] == 0.0).all()

    def test_eig_n_zero_overrides_the_tolerance(self):
        # The whitened case: this spectrum spans more than 1/EPS, so the relative rule
        # zeroes two genuine eigenvalues. `n_zero` says how many are structurally
        # zero and nothing else is touched.
        M = np.diag([1e-12, 1e-10, 1.0])
        assert (eig(M, keep_zeros=True)[0] == [0.0, 0.0, 1.0]).all()
        assert np.allclose(eig(M, keep_zeros=True, n_zero=0)[0], [1e-12, 1e-10, 1.0])
        w, W = eig(M, keep_zeros=True, n_zero=1)
        assert w[0] == 0.0 and np.allclose(w[1:], [1e-10, 1.0])
        assert W.shape == (3, 3)

    def test_pinv_solve_matches_solve_and_pinv(self, rng):
        S01 = rng.normal(size=(6, 3))
        full = psd(rng, 6)
        assert np.allclose(
            lm._pinv_solve(full, S01, atol=0.0), np.linalg.solve(full, S01)
        )
        deficient = psd(rng, 6, rank=4)
        assert np.allclose(
            lm._pinv_solve(deficient, S01, atol=0.0), np.linalg.pinv(deficient) @ S01
        )

    def test_pinv_solve_floors_a_roundoff_s0(self):
        """Why `atol` is required and not defaulted. A penalty that misses the design's
        null space leaves an `S0` that is roundoff throughout, and the plain relative
        rule zeroes none of it -- a 1x1 `S0` cannot be smaller than its own largest
        entry. Unfloored this inverts roundoff; floored it returns zero, which is what
        a flat direction deserves."""
        S0, S01 = np.array([[1e-13]]), np.ones((1, 2))
        assert np.allclose(lm._pinv_solve(S0, S01, atol=0.0), 1e13)
        assert np.allclose(lm._pinv_solve(S0, S01, atol=1e-11), 0.0)

    def test_pinv_solve_handles_empty(self):
        # The common case: X of full column rank leaves no null space, so S0 is (0, 0).
        got = lm._pinv_solve(np.zeros((0, 0)), np.zeros((0, 3)), atol=0.0)
        assert got.shape == (0, 3)



# ---------------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------------


def cache_cases(rng):
    X = rng.normal(size=(20, P))
    deficient = X.copy()
    deficient[:, 5] = deficient[:, 0] + deficient[:, 1]
    null_dir = np.zeros(P)
    null_dir[[0, 1, 5]] = [1, 1, -1]
    return {
        "full rank, S random": (X, psd(rng, P), P),
        "full rank, S=None": (X, None, P),
        "full rank, S=I": (X, np.eye(P), P),
        "full rank, S rank 3": (X, psd(rng, P, 3), P),
        "rank-deficient, S=I": (deficient, np.eye(P), P - 1),
        "rank-deficient, S on null(X)": (deficient, np.outer(null_dir, null_dir), P - 1),
        "wide n < p": (rng.normal(size=(5, 8)), psd(rng, 8), 5),
    }


@pytest.mark.parametrize("case", list(cache_cases(np.random.default_rng(0))))
class TestCache:
    @staticmethod
    def _setup(case):
        rng = np.random.default_rng(0)
        X, S, rank = cache_cases(rng)[case]
        return LinearModel(X, S), X, S, rank, rng

    def test_invariants(self, case):
        m, X, _, rank, _ = self._setup(case)
        r = len(m.omega)
        assert r == rank
        assert np.allclose(X @ m.A, m.U), "Xc @ A == U"
        assert np.allclose(m.U.T @ m.U, np.eye(r)), "U orthonormal"
        assert np.allclose(m.A_inv, m.U.T @ X), "A_inv == U.T @ Xc"
        assert np.allclose(m.A_inv @ m.A, np.eye(r)), "A_inv is a left inverse of A"
        assert (m.omega >= 0).all()

    def test_solve_matches_dense(self, case):
        m, X, S, _, rng = self._setup(case)
        y = rng.normal(size=len(X))
        m.set_y(y)
        m.set_lmbda(model_lmbda(X, S, 0.37))
        assert np.allclose(m.fit().beta, dense_beta(X, S, y, 0.37))

    def test_pushforward_inverts_pullback(self, case):
        m, _, _, rank, _ = self._setup(case)
        nu = np.arange(1.0, rank + 1)
        assert np.allclose(m.pushforward(m.pullback(nu)), nu)


class TestCacheDetails:
    def test_s_none_matches_explicit_identity(self, rng):
        X, _, y = rng.normal(size=(N, P)), None, rng.normal(size=N)
        a = fitted(X, None, y, 0.37).beta
        b = fitted(X, np.eye(P), y, 0.37).beta
        assert np.allclose(a, b)

    def test_s_none_omega_is_the_ridge_filter(self, rng):
        X = rng.normal(size=(N, P))
        # Ratios, not values: omega is normalised, so only its shape is the filter's.
        expected = 1 / np.linalg.svd(X, compute_uv=False) ** 2
        omega = np.sort(LinearModel(X).omega)
        assert np.allclose(omega / omega[0], np.sort(expected) / np.sort(expected)[0])

    def test_omega_is_normalised_to_unit_median(self, rng):
        X, S = rng.normal(size=(N, P)), psd(rng, P)
        for m in (LinearModel(X), LinearModel(X, S), LinearModel(X, psd(rng, P, 3))):
            assert np.median(m.omega[m.omega > 0]) == pytest.approx(1.0)

    def test_asymmetric_s_is_symmetrised(self, rng):
        X, y = rng.normal(size=(N, P)), rng.normal(size=N)
        S = psd(rng, P)
        K = rng.normal(size=(P, P))
        K -= K.T  # skew: leaves the symmetric part untouched
        # Both caches see the same symmetric part, so they normalise identically and a
        # single lmbda in either parameterisation names the same fit in both.
        a, b = LinearModel(X, S), LinearModel(X, S + 0.5 * K)
        for m in (a, b):
            m.set_y(y)
            m.set_lmbda(0.37)
        assert np.allclose(a.fit().beta, b.fit().beta)

    def test_centred_design_is_not_retained(self, rng):
        assert not hasattr(LinearModel(rng.normal(size=(N, P)), fit_intercept=True), "Xc")

    def test_constraint_becomes_plain_orthogonality(self, rng):
        X, S = rng.normal(size=(N, P)), psd(rng, P)
        m = LinearModel(X, S)
        c, nu = rng.normal(size=P), rng.normal(size=P)
        assert np.allclose((X @ c) @ (X @ m.pullback(nu)), m.pushforward(c) @ nu)

    @pytest.mark.parametrize("rank", range(1, P + 1))
    def test_rank_deficient_penalties_are_accepted(self, rng, rank):
        X = rng.normal(size=(N, P))
        assert len(LinearModel(X, psd(rng, P, rank)).omega) == P

    def test_zero_penalty_is_accepted(self, rng):
        X = rng.normal(size=(N, P))
        assert (LinearModel(X, np.zeros((P, P))).omega == 0).all()

    @pytest.mark.parametrize(
        "bad", ["indefinite", "negative definite", "mildly indefinite"]
    )
    def test_non_psd_penalty_raises(self, rng, bad):
        X, S = rng.normal(size=(N, P)), psd(rng, P)
        top = np.linalg.eigvalsh(S).max()
        S = {
            "indefinite": S - 2 * top * np.eye(P),
            "negative definite": -S,
            "mildly indefinite": psd(rng, P, 5) - 1e-3 * top * np.eye(P),
        }[bad]
        with pytest.raises(ValueError, match="positive semi-definite"):
            LinearModel(X, S)

    def test_roundoff_level_negative_eigenvalue_is_accepted(self, rng):
        X, S = rng.normal(size=(N, P)), psd(rng, P)
        w, Q = np.linalg.eigh(S)
        w[0] = -1e-14 * w[-1]
        LinearModel(X, Q @ np.diag(w) @ Q.T)  # must not raise


# ---------------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------------


class TestFit:
    @pytest.mark.parametrize("lmbda", [0.0, 1e-3, 1.0, 1e3])
    def test_matches_dense(self, data, lmbda):
        X, S, y = data
        assert np.allclose(fitted(X, S, y, lmbda).beta, dense_beta(X, S, y, lmbda))

    def test_lmbda_zero_is_least_squares(self, data):
        X, S, y = data
        expected = np.linalg.lstsq(X, y, rcond=None)[0]
        assert np.allclose(fitted(X, S, y, 0.0).beta, expected)

    def test_intercept_matches_augmented_design(self, data):
        X, S, y = data
        m = fitted(X, S, y, 0.5, fit_intercept=True)
        Xa, Sa = augment(X, S)
        ref = np.linalg.solve(Xa.T @ Xa + 0.5 * Sa, Xa.T @ y)
        assert np.isclose(m.b, ref[0])
        assert np.allclose(m.beta, ref[1:])

    def test_intercept_slopes_match_explicitly_centred_data(self, data):
        X, S, y = data
        a = fitted(X, S, y, 0.5, fit_intercept=True).beta
        b = fitted(X - X.mean(0), S, y - y.mean(), 0.5).beta
        assert np.allclose(a, b)

    def test_recovers_a_known_offset(self, data, rng):
        X, S, _ = data
        beta = rng.normal(size=P)
        m = fitted(X, S, X @ beta + 7.5, 0.0, fit_intercept=True)
        assert np.isclose(m.b, 7.5)
        assert np.allclose(m.beta, beta)

    def test_b_is_exactly_zero_without_an_intercept(self, data):
        X, S, y = data
        assert fitted(X, S, y, 0.5).b == 0.0

    def test_one_cache_serves_many_y_and_lmbda(self, data, rng):
        X, S, _ = data
        m = LinearModel(X, S)
        for _ in range(3):
            y = rng.normal(size=N)
            m.set_y(y)
            for lmbda in (1e-3, 1.0, 1e3):
                m.set_lmbda(model_lmbda(X, S, lmbda))
                assert np.allclose(m.fit().beta, dense_beta(X, S, y, lmbda))

    def test_integer_input(self, rng):
        X, S = rng.integers(0, 5, size=(N, P)), psd(rng, P)
        y = rng.normal(size=N)
        assert np.allclose(fitted(X, S, y, 0.5).beta, dense_beta(X, S, y, 0.5))


class TestPredict:
    def test_defaults_to_training_data(self, data):
        X, S, y = data
        m = fitted(X, S, y, 0.5)
        assert np.allclose(m.predict(), X @ m.beta + m.b)

    def test_on_test_data_with_intercept(self, data, rng):
        X, S, y = data
        m = fitted(X, S, y, 0.5, fit_intercept=True)
        Xt = rng.normal(size=(9, P))
        assert np.allclose(m.predict(Xt), Xt @ m.beta + m.b)


class TestState:
    def test_y_and_lmbda_start_unset(self, data):
        X, S, _ = data
        m = LinearModel(X, S)
        assert m.y is None and m.lmbda is None

    def test_setters_clear_the_fit(self, data):
        X, S, y = data
        m = fitted(X, S, y, 0.5)
        m.set_lmbda(1.0)
        assert m.beta is None
        m.set_lmbda(1.0)
        m.fit()
        m.set_y(y)
        assert m.beta is None

    def test_fit_raises_without_y(self, data):
        X, S, _ = data
        m = LinearModel(X, S)
        m.set_lmbda(1.0)
        with pytest.raises(ValueError, match="y is not set"):
            m.fit()

    def test_fit_raises_without_lmbda(self, data):
        X, S, y = data
        m = LinearModel(X, S)
        m.set_y(y)
        with pytest.raises(ValueError, match="lmbda is not set"):
            m.fit()

    def test_fit_raises_even_with_a_criterion_attached(self, data):
        """Selection is never implicit."""
        X, S, y = data
        m = LinearModel(X, S, lmbda_criterion=lm.gcv())
        m.set_y(y)
        with pytest.raises(ValueError, match="lmbda is not set"):
            m.fit()

    def test_set_y_leaves_lmbda_alone(self, data):
        """set_y sets the response and nothing else about lmbda -- including when a
        criterion is attached. Sweeping many y at a fixed lmbda is the whole point of
        the cache, so a setter must not silently undo the other one."""
        X, S, y = data
        for criterion in (None, lm.gcv()):
            m = LinearModel(X, S, lmbda_criterion=criterion)
            m.set_lmbda(0.5)
            m.set_y(y)
            assert m.lmbda == 0.5
            m.set_y(2 * y)
            assert m.lmbda == 0.5

    def test_criterion_chosen_lmbda_also_survives_set_y(self, data):
        X, S, y = data
        m = LinearModel(X, S, lmbda_criterion=lm.gcv())
        m.set_y(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit_lmbda()
        chosen = m.lmbda
        m.set_y(2 * y)
        assert m.lmbda == chosen

    def test_predict_before_fit_raises(self, data):
        X, S, _ = data
        with pytest.raises(ValueError, match="not fitted"):
            LinearModel(X, S).predict()

    def test_wrong_y_shape_raises(self, data):
        X, S, _ = data
        with pytest.raises(ValueError, match="must have shape"):
            LinearModel(X, S).set_y(np.zeros((N, 1)))

    def test_negative_lmbda_raises(self, data):
        X, S, _ = data
        with pytest.raises(ValueError, match="non-negative"):
            LinearModel(X, S).set_lmbda(-1.0)


# ---------------------------------------------------------------------------------
# degrees of freedom
# ---------------------------------------------------------------------------------


class TestEdf:
    @pytest.mark.parametrize("fit_intercept", [False, True])
    @pytest.mark.parametrize("lmbda", [1e-4, 0.1, 1.0, 10.0, 1e4])
    def test_matches_hat_trace(self, data, fit_intercept, lmbda):
        """The two lmbdas differ by the constant `omega`'s normalisation introduced, so
        the two edf curves are translates in log lmbda. Fixing that constant at one
        point and checking it holds at every other is the strongest form of this
        available without asking the constructor what it chose -- and a stronger claim
        than matching a single point ever was."""
        X, S, _ = data
        m = LinearModel(X, S, fit_intercept=fit_intercept)
        c = model_lmbda(X, S, 1.0, fit_intercept)  # model lmbda per dense lmbda
        assert np.isclose(
            m.edf(c * lmbda), np.trace(dense_hat(X, S, lmbda, fit_intercept))
        )

    def test_needs_no_y(self, data):
        X, S, _ = data
        assert np.isfinite(LinearModel(X, S).edf(1.0))

    def test_vectorises(self, data):
        X, S, _ = data
        m = LinearModel(X, S)
        grid = np.geomspace(1e-4, 1e4, 25)
        assert np.allclose(m.edf(grid), [m.edf(t) for t in grid])

    def test_strictly_decreasing(self, data):
        X, S, _ = data
        assert np.all(np.diff(LinearModel(X, S).edf(np.geomspace(1e-4, 1e4, 25))) < 0)

    def test_limits(self, data):
        X, S, _ = data
        m = LinearModel(X, S)
        assert np.isclose(m.edf(1e-12), P)
        assert m.edf(1e12) < 1e-6

    def test_includes_the_intercept(self, data):
        X, S, _ = data
        assert np.isclose(LinearModel(X, S, fit_intercept=True).edf(1e12), 1.0)


class TestSetEdf:
    @pytest.mark.parametrize("target", [1.5, 3.0, 5.5])
    def test_round_trip(self, data, target):
        X, S, _ = data
        assert np.isclose(LinearModel(X, S).set_edf(target).edf(), target)

    @pytest.mark.parametrize("target", [2.0, 4.0, 6.5])
    def test_round_trip_with_intercept(self, data, target):
        X, S, _ = data
        m = LinearModel(X, S, fit_intercept=True)
        assert np.isclose(m.set_edf(target).edf(), target)

    def test_proportion(self, data):
        X, S, _ = data
        assert np.isclose(LinearModel(X, S).set_edf(0.5, proportion=True).edf(), 0.5 * P)

    def test_proportion_counts_the_intercept_in_the_total(self, data):
        X, S, _ = data
        m = LinearModel(X, S, fit_intercept=True)
        assert np.isclose(m.set_edf(0.5, proportion=True).edf(), 0.5 * (P + 1))

    def test_fit_after_set_edf(self, data):
        X, S, y = data
        m = LinearModel(X, S)
        m.set_y(y)
        m.set_edf(4.0).fit()
        assert np.allclose(m.beta, dense_beta(X, S, y, m.lmbda / model_lmbda(X, S, 1.0)))

    @pytest.mark.parametrize("target", [0.0, 6.5, -1.0])
    def test_unattainable_target_raises(self, data, target):
        X, S, _ = data
        with pytest.raises(ValueError, match="unattainable"):
            LinearModel(X, S).set_edf(target)

    def test_the_last_edf_tol_at_each_end_is_refused(self, data):
        """The range is open at both ends by construction: `lmbda_grid` stops
        `edf_tol` short of `r` and of the floor, so a target inside that margin has no
        finite `lmbda` worth chasing."""
        X, S, _ = data
        m = LinearModel(X, S)
        assert np.isclose(m.set_edf(P - 0.11).edf(), P - 0.11)
        for target in (P - 0.09, 0.09):
            with pytest.raises(ValueError, match="unattainable"):
                LinearModel(X, S).set_edf(target)
        assert np.isclose(LinearModel(X, S).set_edf(0.02, edf_tol=0.01).edf(), 0.02)

    def test_floor_is_set_by_unpenalised_directions(self, rng):
        X = rng.normal(size=(N, P))
        m = LinearModel(X, psd(rng, P, 4))  # 2 directions carry no penalty
        assert np.isclose(m.edf(1e14), 2.0, atol=1e-9)
        with pytest.raises(ValueError, match="unattainable"):
            m.set_edf(1.5)

    def test_an_unpenalised_model_takes_any_lmbda(self, rng):
        """`omega == 0` everywhere: `lmbda` does not act, so `edf` is constant at `r`
        and the only attainable target is that. `set_edf` takes it at `lmbda = 0`
        rather than root-finding on a degenerate bracket."""
        X = rng.normal(size=(N, P))
        m = LinearModel(X, np.zeros((P, P)))
        assert (m.omega == 0).all()
        assert m.set_edf(float(P)).lmbda == 0.0
        with pytest.raises(ValueError, match="unattainable"):
            m.set_edf(P - 1.0)


class TestLmbdaGrid:
    @pytest.mark.parametrize("fit_intercept", [False, True])
    @pytest.mark.parametrize("tol", [0.5, 0.1, 0.01, 1e-4])
    def test_endpoints_give_up_edf_tol(self, data, tol, fit_intercept):
        """Independent reference: `edf` itself, which `lmbda_grid` only ever sees
        through a first-order expansion -- so `rtol=tol` is the claim, not a loose
        tolerance. Measured relative error runs -0.16 at `tol=0.5` down to -4e-5 at
        1e-4, i.e. the expansion is first order and behaves like it.

        Parametrised over the intercept because the derivation drops it: it sits in
        both limits of `edf` and cancels in the difference."""
        X, S, _ = data
        m = LinearModel(X, S, fit_intercept=fit_intercept)
        g = m.lmbda_grid(edf_tol=tol)
        ceiling = len(m.omega) + m.fit_intercept
        floor = int((m.omega == 0).sum()) + m.fit_intercept
        assert np.isclose(ceiling - m.edf(g[0]), tol, rtol=tol)
        assert np.isclose(m.edf(g[-1]) - floor, tol, rtol=tol)

    def test_is_invariant_to_the_scale_of_the_penalty(self, data):
        """`omega` is normalised to unit median, but the derivation does not rest on
        it: scaling `S` scales `omega` and the grid moves with it, so the *fits* the
        grid visits are the same either way."""
        X, S, _ = data
        m, big = LinearModel(X, S), LinearModel(X, 1e6 * S)
        assert np.allclose(m.edf(m.lmbda_grid()), big.edf(big.lmbda_grid()))

    def test_shape_and_spacing(self, data):
        X, S, _ = data
        g = LinearModel(X, S).lmbda_grid(grid_size=37)
        assert g.shape == (37,)
        assert np.allclose(np.diff(np.log(g)), np.diff(np.log(g))[0])

    def test_unpenalised_model_gives_a_single_zero(self, rng):
        X = rng.normal(size=(N, P))
        assert LinearModel(X, np.zeros((P, P))).lmbda_grid() == np.zeros(1)

    def test_widens_with_the_spread_of_omega(self, rng):
        """The point of deriving it: a fixed bracket cannot be right for both."""
        narrow = LinearModel(rng.normal(size=(N, P)), np.eye(P))
        X = rng.normal(size=(N, P)) * np.geomspace(1, 1e-4, P)
        wide = LinearModel(X, np.eye(P))
        span = lambda m: np.log10(m.lmbda_grid()[-1] / m.lmbda_grid()[0])
        assert span(wide) > span(narrow) + 3.0


# ---------------------------------------------------------------------------------
# rss
# ---------------------------------------------------------------------------------


class TestRss:
    @pytest.mark.parametrize("fit_intercept", [False, True])
    @pytest.mark.parametrize("lmbda", [1e-3, 1.0, 1e3])
    def test_equals_actual_residuals(self, data, fit_intercept, lmbda):
        X, S, y = data
        m = fitted(X, S, y, lmbda, fit_intercept=fit_intercept)
        assert np.isclose(m.rss(), ((y - m.predict()) ** 2).sum())

    def test_vectorises(self, data):
        X, S, y = data
        m = LinearModel(X, S)
        m.set_y(y)
        grid = np.geomspace(1e-4, 1e4, 25)
        assert np.allclose(m.rss(grid), [m.rss(t) for t in grid])

    def test_increasing_in_lmbda(self, data):
        X, S, y = data
        m = LinearModel(X, S)
        m.set_y(y)
        assert np.all(np.diff(m.rss(np.geomspace(1e-4, 1e4, 25))) > 0)

    def test_requires_y(self, data):
        X, S, _ = data
        with pytest.raises(ValueError, match="y is not set"):
            LinearModel(X, S).rss(1.0)

    def test_saturated_design_stays_non_negative(self, rng):
        """NUMERICAL: r_perp cancels to zero here and must not go negative, or
        log(rss) in AIC/BIC becomes nan."""
        m = LinearModel(rng.normal(size=(8, 8)))
        m.set_y(rng.normal(size=8))
        assert 0 <= m.rss(1e-12) < 1e-12


# ---------------------------------------------------------------------------------
# lmbda selection
# ---------------------------------------------------------------------------------

GRID = np.geomspace(1e-4, 1e4, 21)


def dense_criterion_inputs(X, S, y, fit_intercept=False):
    """RSS and EDF along `GRID`, which the criteria read in the model's lmbda -- so it
    is converted here, once, and the comparisons below stay point by point."""
    grid = GRID / model_lmbda(X, S, 1.0, fit_intercept)
    rows = [
        (
            ((y - dense_hat(X, S, t, fit_intercept) @ y) ** 2).sum(),
            np.trace(dense_hat(X, S, t, fit_intercept)),
        )
        for t in grid
    ]
    return np.array(rows)[:, 0], np.array(rows)[:, 1]


def dense_reml(X, S, y, lmbda, fit_intercept=False):
    """-2 log L_R assembled from the marginal covariance, sharing no code with REML."""
    n = len(X)
    Xc = X - X.mean(0) if fit_intercept else X
    M = np.eye(n) + Xc @ np.linalg.solve(lmbda * S, Xc.T)
    Minv = np.linalg.inv(M)
    F = np.ones((n, 1)) if fit_intercept else np.zeros((n, 0))
    n_eff = n - F.shape[1]
    if F.shape[1]:
        FMF = F.T @ Minv @ F
        resid = y - F @ np.linalg.solve(FMF, F.T @ Minv @ y)
        extra = np.log(np.linalg.det(FMF))
    else:
        resid, extra = y, 0.0
    quad = resid @ Minv @ resid
    return np.linalg.slogdet(M)[1] + extra + n_eff * np.log(quad / n_eff)


class TestCriteria:
    @pytest.mark.parametrize("fit_intercept", [False, True])
    @pytest.mark.parametrize("gamma", [1.0, 1.4])
    def test_gcv_matches_dense(self, data, fit_intercept, gamma):
        X, S, y = data
        m = LinearModel(X, S, fit_intercept=fit_intercept)
        m.set_y(y)
        c = lm.gcv(gamma=gamma)
        c.attach(m)
        rss, edf = dense_criterion_inputs(X, S, y, fit_intercept)
        assert np.allclose(c.criterion(GRID), N * rss / (N - gamma * edf) ** 2)

    @pytest.mark.parametrize("fit_intercept", [False, True])
    def test_aic_matches_dense(self, data, fit_intercept):
        X, S, y = data
        m = LinearModel(X, S, fit_intercept=fit_intercept)
        m.set_y(y)
        c = lm.aic()
        c.attach(m)
        rss, edf = dense_criterion_inputs(X, S, y, fit_intercept)
        assert np.allclose(c.criterion(GRID), N * np.log(rss / N) + 2 * edf)

    @pytest.mark.parametrize("fit_intercept", [False, True])
    def test_bic_matches_dense(self, data, fit_intercept):
        X, S, y = data
        m = LinearModel(X, S, fit_intercept=fit_intercept)
        m.set_y(y)
        c = lm.bic()
        c.attach(m)
        rss, edf = dense_criterion_inputs(X, S, y, fit_intercept)
        assert np.allclose(c.criterion(GRID), N * np.log(rss / N) + np.log(N) * edf)

    @pytest.mark.parametrize("fit_intercept", [False, True])
    def test_reml_matches_dense_up_to_a_constant(self, data, fit_intercept):
        X, S, y = data
        m = LinearModel(X, S, fit_intercept=fit_intercept)
        m.set_y(y)
        c = lm.reml()
        c.attach(m)
        ours = c.criterion(GRID)
        scale = model_lmbda(X, S, 1.0, fit_intercept)
        ref = np.array([dense_reml(X, S, y, t / scale, fit_intercept) for t in GRID])
        assert np.std(ours - ref) < 1e-8 * max(1.0, np.abs(ours).max())
        assert np.argmin(ours) == np.argmin(ref)

    @pytest.mark.parametrize("name", ["gcv", "reml", "aic", "bic"])
    def test_vectorises(self, data, name):
        X, S, y = data
        m = LinearModel(X, S)
        m.set_y(y)
        c = getattr(lm, name)()
        c.attach(m)
        assert np.allclose(c.criterion(GRID), [c.criterion(t) for t in GRID])

    @pytest.mark.parametrize("name", ["gcv", "reml", "aic", "bic"])
    def test_fit_lmbda_returns_the_grid_argmin(self, data, name):
        X, S, y = data
        c = getattr(lm, name)(grid_size=51)
        m = LinearModel(X, S, lmbda_criterion=c)
        m.set_y(y)
        m.fit_lmbda()
        grid = m.lmbda_grid(51, c.edf_tol)
        assert np.isclose(m.lmbda, grid[np.argmin(c.criterion(grid))])
        dense = m.lmbda / model_lmbda(X, S, 1.0)
        assert np.allclose(m.fit().beta, dense_beta(X, S, y, dense))

    def test_noisier_data_selects_more_smoothing(self, data, rng):
        X, S, _ = data
        beta = rng.normal(size=P)
        picks = []
        for sd in (0.05, 2.0):
            m = LinearModel(X, S, lmbda_criterion=lm.gcv())
            m.set_y(X @ beta + sd * rng.normal(size=N))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                picks.append(m.fit_lmbda().lmbda)
        assert picks[0] < picks[1]

    def test_gamma_above_one_smooths_harder(self, data):
        X, S, y = data
        m = LinearModel(X, S)
        m.set_y(y)
        low, high = lm.gcv(gamma=1.0), lm.gcv(gamma=2.0)
        low.attach(m)
        high.attach(m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert high.fit_lmbda() >= low.fit_lmbda()

    def test_factories_carry_their_options(self):
        assert isinstance(lm.gcv(gamma=3), lm.GCV) and lm.gcv(gamma=3).gamma == 3
        assert isinstance(lm.reml(), lm.REML)
        assert isinstance(lm.aic(edf_tol=0.5), lm.AIC) and lm.aic(edf_tol=0.5).edf_tol == 0.5
        assert isinstance(lm.bic(grid_size=7), lm.BIC) and lm.bic(grid_size=7).grid_size == 7

    def test_constructor_attaches(self, data):
        X, S, _ = data
        c = lm.gcv()
        m = LinearModel(X, S, lmbda_criterion=c)
        assert m.lmbda_criterion is c and c.model is m

    def test_unattached_criterion_raises(self):
        with pytest.raises(ValueError, match="not attached"):
            lm.gcv().fit_lmbda()

    def test_none_detaches(self, data):
        X, S, y = data
        m = LinearModel(X, S, lmbda_criterion=lm.gcv())
        m.set_lmbda_criterion(None)
        assert m.lmbda_criterion is None
        with pytest.raises(ValueError, match="no lmbda_criterion"):
            m.fit_lmbda()

    def test_detaching_then_fitting_by_hand(self, data):
        X, S, y = data
        m = LinearModel(X, S, lmbda_criterion=lm.gcv())
        m.set_lmbda_criterion(None)
        m.set_y(y)
        m.set_lmbda(model_lmbda(X, S, 0.5))
        assert np.allclose(m.fit().beta, dense_beta(X, S, y, 0.5))

    def test_replacing_a_criterion(self, data):
        X, S, _ = data
        m = LinearModel(X, S, lmbda_criterion=lm.gcv())
        second = lm.bic()
        m.set_lmbda_criterion(second)
        assert m.lmbda_criterion is second and second.model is m

    def test_constructing_with_none_is_the_same_as_omitting(self, data):
        X, S, _ = data
        assert LinearModel(X, S, lmbda_criterion=None).lmbda_criterion is None
        assert LinearModel(X, S).lmbda_criterion is None

    def test_fit_lmbda_without_a_criterion_raises(self, data):
        X, S, _ = data
        with pytest.raises(ValueError, match="no lmbda_criterion"):
            LinearModel(X, S).fit_lmbda()

    @pytest.mark.parametrize("name", ["gcv", "reml", "aic", "bic"])
    @pytest.mark.parametrize("signal", [True, False])
    def test_the_derived_range_never_truncates(self, data, rng, name, signal):
        """What replaces the old edge warning, and why it could be deleted rather than
        fixed: no `lmbda` outside the range beats the best one inside by more than
        `edf_tol`. Both minima are taken over the *same* 28-decade grid, one of them
        restricted to the range, so the difference is truncation alone and carries no
        contribution from the spacing."""
        X, S, y = data
        m = LinearModel(X, S)
        m.set_y(X @ rng.normal(size=P) + 0.3 * rng.normal(size=N) if signal else y)
        c = getattr(lm, name)()
        c.attach(m)
        ends = m.lmbda_grid(2, c.edf_tol)
        wide = np.geomspace(1e-14, 1e14, 4000)
        inside = wide[(wide >= ends[0]) & (wide <= ends[-1])]
        best_all = wide[np.argmin(c.criterion(wide))]
        best_inside = inside[np.argmin(c.criterion(inside))]
        assert abs(m.edf(best_inside) - m.edf(best_all)) <= c.edf_tol


# ---------------------------------------------------------------------------------
# constraints and normalisation
# ---------------------------------------------------------------------------------


class TestConstraints:
    def test_single_constraint_matches_kkt(self, data, rng):
        X, S, y = data
        c = rng.normal(size=P)
        m = LinearModel(X, S).add_constraints(c)
        m.set_y(y)
        m.set_lmbda(model_lmbda(X, S, 0.7, C=c))
        m.fit()
        assert len(m.omega) == P - 1
        assert np.allclose(m.beta, dense_kkt(X, S, y, 0.7, c))
        assert abs((X @ c) @ (X @ m.beta)) < 1e-10

    def test_several_constraints_match_kkt(self, data, rng):
        X, S, y = data
        C = rng.normal(size=(3, P))
        m = LinearModel(X, S).add_constraints(C)
        m.set_y(y)
        m.set_lmbda(model_lmbda(X, S, 0.7, C=C))
        m.fit()
        assert len(m.omega) == P - 3
        assert np.allclose(m.beta, dense_kkt(X, S, y, 0.7, C))
        assert np.allclose((X @ C.T).T @ (X @ m.beta), 0, atol=1e-10)

    def test_batch_equals_sequential(self, data, rng):
        """Also pins that the renormalisation composes: normalising after each of two
        constraints is normalising once after both, the first median cancelling out of
        the second eigenproblem. If that failed, `omega` would depend on the order the
        rows arrived in and nothing else would notice."""
        X, S, y = data
        C = rng.normal(size=(3, P))
        batch = LinearModel(X, S).add_constraints(C)
        seq = LinearModel(X, S)
        for row in C:
            seq.add_constraints(row)
        for m in (batch, seq):
            m.set_y(y)
            m.set_lmbda(0.7)
            m.fit()
        assert np.allclose(batch.beta, seq.beta)

    @pytest.mark.parametrize("penalty", ["dense", "ridge"])
    def test_the_two_routes_agree(self, data, rng, penalty):
        """CROSS-PATH PIN, and the whole basis for having two routes: the constructor
        imposes a constraint inside build step 3 and `add_constraints` imposes it on the
        finished cache, so they run different code and must land on the same model.

        Not an independent reference -- `test_single_constraint_matches_kkt` and its
        neighbours are what check either route against dense algebra. This checks they
        are the same route. `ridge` covers `S=None`, whose short-circuit a constraint
        invalidates and which therefore has its own branch.

        Agreement is to roundoff, not approximate: `W` is orthogonal, so the two differ
        only by the eigenvector conventions `linalg.eig` fixes. Measured 5e-16 relative
        on `omega` here and 2e-14 on a spline basis with a singular penalty.
        """
        X, S, y = data
        S = None if penalty == "ridge" else S
        C = rng.normal(size=(2, P))
        a = constrained(X, S, C, "constructor")
        b = constrained(X, S, C, "add_constraints")

        assert len(a.omega) == len(b.omega) == P - 2
        assert (a.omega == 0).sum() == (b.omega == 0).sum()
        assert np.allclose(a.omega, b.omega, rtol=1e-11, atol=0.0)
        # A U column's sign is a convention, and a repeated omega fixes only the span.
        assert np.allclose(np.abs(a.U), np.abs(b.U), atol=1e-11)
        assert np.array_equal(a.constraints, b.constraints)
        for m in (a, b):
            m.set_y(y)
            m.set_edf(3.0)
            m.fit()
        assert np.isclose(a.lmbda, b.lmbda, rtol=1e-9)
        assert np.allclose(a.beta, b.beta, atol=1e-12)

    def test_omega_is_renormalised_by_a_constraint(self, data, rng):
        """The constrained spectrum is a new spectrum, so build step 4 runs again and
        unit median holds for every model the class can build. This is what lets a
        constrained parent's folds be rebuilt through the constructor and searched on
        the parent's own scale -- see `cv.KFoldCV`. Dropping it moves `omega` by 23%,
        which no other test in this file would notice.
        """
        X, S, _ = data
        for m in (
            LinearModel(X, S),
            LinearModel(X, S).add_constraints(rng.normal(size=(2, P))),
            LinearModel(X, S, constraints=rng.normal(size=(2, P))),
        ):
            assert np.isclose(np.median(m.omega[m.omega > 0]), 1.0)

    def test_records_every_row_imposed(self, data, rng):
        """`cv.KFoldCV` rebuilds a fold model from `X`, `S` and this, so a row imposed
        and not recorded is a fold quietly fitting a different model."""
        X, S, _ = data
        C = rng.normal(size=(2, P))
        c = rng.normal(size=P)

        assert LinearModel(X, S).constraints is None
        assert np.array_equal(LinearModel(X, S, constraints=C).constraints, C)
        m = LinearModel(X, S, constraints=C).add_constraints(c)
        assert np.array_equal(m.constraints, np.vstack([C, c]))

    def test_accepts_a_list_of_vectors(self, data, rng):
        X, S, _ = data
        C = rng.normal(size=(2, P))
        assert len(LinearModel(X, S).add_constraints(list(C)).omega) == P - 2

    def test_invariants_survive(self, data, rng):
        X, S, _ = data
        m = LinearModel(X, S).add_constraints(rng.normal(size=(2, P)))
        r = len(m.omega)
        assert np.allclose(X @ m.A, m.U)
        assert np.allclose(m.A_inv @ m.A, np.eye(r))
        assert np.allclose(m.U.T @ m.U, np.eye(r))

    def test_clears_the_fit(self, data, rng):
        X, S, _ = data
        assert LinearModel(X, S).add_constraints(rng.normal(size=P)).beta is None

    def test_redundant_constraints_drop_rank_once(self, data, rng):
        X, S, y = data
        c = rng.normal(size=P)
        m = LinearModel(X, S).add_constraints(np.vstack([c, 2 * c]))
        m.set_y(y)
        m.set_lmbda(model_lmbda(X, S, 0.7, C=np.vstack([c, 2 * c])))
        m.fit()
        assert len(m.omega) == P - 1
        assert np.allclose(m.beta, dense_kkt(X, S, y, 0.7, c))

    def test_tiny_but_real_constraint_is_kept(self, data, rng):
        """Constraints are scale-free, so magnitude must not decide."""
        X, S, _ = data
        assert len(LinearModel(X, S).add_constraints(1e-30 * rng.normal(size=P)).omega) == P - 1

    def test_downstream_quantities_stay_correct(self, data, rng):
        X, S, y = data
        C = rng.normal(size=(3, P))
        m = LinearModel(X, S).add_constraints(C)
        m.set_y(y)

        scale = model_lmbda(X, S, 1.0, C=C)
        for t in (1e-3, 1.0, 1e3):
            assert np.isclose(m.edf(scale * t), np.trace(dense_hat(X, S, t, C=C)))
        m.set_lmbda(scale * 0.7)
        m.fit()
        assert np.isclose(m.rss(), ((y - m.predict()) ** 2).sum())
        m.set_edf(2.0)
        assert np.isclose(m.edf(), 2.0)

    @pytest.mark.parametrize("name", ["gcv", "reml"])
    def test_selection_under_constraints(self, data, rng, name):
        X, S, y = data
        C = rng.normal(size=(2, P))
        m = LinearModel(X, S, lmbda_criterion=getattr(lm, name)()).add_constraints(C)
        m.set_y(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit_lmbda()
        m.fit()
        dense = m.lmbda / model_lmbda(X, S, 1.0, C=C)
        assert np.allclose(m.beta, dense_kkt(X, S, y, dense, C))

    def test_requires_no_intercept(self, data, rng):
        X, S, _ = data
        with pytest.raises(ValueError, match="fit_intercept=False"):
            LinearModel(X, S, fit_intercept=True).add_constraints(rng.normal(size=P))

    def test_wrong_length_raises(self, data, rng):
        X, S, _ = data
        with pytest.raises(ValueError, match="must have length"):
            LinearModel(X, S).add_constraints(rng.normal(size=P + 1))


class TestNormalizeBeta:
    def test_scales_to_unit_mean_square_prediction(self, data):
        X, S, y = data
        m = fitted(X, S, y, 0.7).normalize_beta()
        assert np.isclose((X @ m.beta) @ (X @ m.beta) / N, 1.0)

    def test_preserves_direction_and_is_idempotent(self, data):
        X, S, y = data
        m = fitted(X, S, y, 0.7)
        direction = m.beta / np.linalg.norm(m.beta)
        m.normalize_beta()
        assert np.allclose(m.beta / np.linalg.norm(m.beta), direction)
        m.normalize_beta()
        assert np.isclose((X @ m.beta) @ (X @ m.beta) / N, 1.0)

    def test_composes_with_constraints(self, data, rng):
        X, S, y = data
        C = rng.normal(size=(2, P))
        m = LinearModel(X, S).add_constraints(C)
        m.set_y(y)
        m.set_lmbda(0.7)
        m.fit().normalize_beta()
        assert np.isclose((X @ m.beta) @ (X @ m.beta) / N, 1.0)
        assert np.allclose((X @ C.T).T @ (X @ m.beta), 0, atol=1e-10)

    def test_requires_no_intercept(self, data):
        X, S, y = data
        m = fitted(X, S, y, 0.7, fit_intercept=True)
        with pytest.raises(ValueError, match="fit_intercept=False"):
            m.normalize_beta()

    def test_requires_a_fit(self, data):
        X, S, _ = data
        with pytest.raises(ValueError, match="not fitted"):
            LinearModel(X, S).normalize_beta()


def feasible(rng, X, C=None, size=200):
    """Random `beta`s with `||X beta||^2 / n == 1`, and `(X C').T @ (X beta) == 0` if
    constrained -- the set `minimize_penalty` claims to minimise the penalty over."""
    B = rng.normal(size=(size, X.shape[1]))
    if C is not None:
        G = np.atleast_2d(C) @ X.T @ X
        B = B @ (np.eye(X.shape[1]) - np.linalg.pinv(G) @ G)
    return B * np.sqrt(len(X)) / np.linalg.norm(X @ B.T, axis=0)[:, None]


class TestMinimizePenalty:
    def test_matches_a_dense_generalized_eigenproblem(self, data):
        """Independent reference: the same problem stated in beta-space as the pencil
        `(S, X'X)`, whose smallest eigenvector minimises `beta' S beta` at fixed
        `beta' X'X beta`. Nothing of the diagonal basis is reused."""
        X, S, _ = data
        m = LinearModel(X, S).minimize_penalty()
        beta = eigh(S, X.T @ X)[1][:, 0]
        want = X @ beta * np.sqrt(N / (beta @ X.T @ X @ beta))
        assert up_to_sign(X @ m.beta, want) / np.sqrt(N) < 1e-10

    def test_ridge_gives_the_leading_principal_component(self, data):
        """Independent reference: `numpy.linalg.svd`. With `S=None` the penalty is
        `||beta||^2` and `omega` is `1 / D**2` up to a scale, so the least-penalised
        direction at fixed `||X beta||` is where the design has the most gain."""
        X, _, _ = data
        m = LinearModel(X).minimize_penalty()
        want = np.linalg.svd(X)[0][:, 0] * np.sqrt(N)
        assert up_to_sign(X @ m.beta, want) / np.sqrt(N) < 1e-10

    @pytest.mark.parametrize("constrain", [False, True])
    def test_no_feasible_beta_is_less_penalised(self, data, rng, constrain):
        X, S, _ = data
        C = rng.normal(size=(2, P)) if constrain else None
        m = LinearModel(X, S)
        if constrain:
            m.add_constraints(C)
        m.minimize_penalty()
        got = m.beta @ S @ m.beta
        B = feasible(rng, X, C)
        assert np.isclose((X @ m.beta) @ (X @ m.beta) / N, 1.0)
        assert got <= np.einsum("ij,jk,ik->i", B, S, B).min()
        # nu' diag(omega) nu with one coordinate at sqrt(n): exact, not approached.
        # omega is normalised, so its report of the penalty is in those units.
        assert np.isclose(got, N * m.omega.min() * model_lmbda(X, S, 1.0, C=C))

    def test_needs_neither_y_nor_lmbda(self, data):
        """The point of it: `omega` is a property of the design and the penalty, so
        this is the one fit available before a response exists -- which is what lets
        `ALSFit` start from it."""
        X, S, _ = data
        m = LinearModel(X, S).minimize_penalty()
        assert m.y is None and m.lmbda is None
        assert np.allclose(m.predict(), X @ m.beta)

    def test_is_the_joint_fit_at_large_lmbda(self, data, rng):
        """The docstring's limit claim, and its restriction to the joint fit: that
        objective is a difference at fixed scale, so a large `lmbda` leaves the penalty
        deciding alone. `fit` renormalised does *not* converge here -- uniform shrinkage
        cancels, and its direction keeps a `ybar / omega` weighting."""
        X, S, y = data
        m, other = LinearModel(X, S), LinearModel(rng.normal(size=(N, P - 2)))
        m.set_lmbda(1e10)
        other.set_lmbda(0.7)
        EigenFit(m, other).fit()
        want = X @ LinearModel(X, S).minimize_penalty().beta
        assert up_to_sign(X @ m.beta, want) / np.sqrt(N) < 1e-6
        assert up_to_sign(X @ fitted(X, S, y, 1e10).normalize_beta().beta, want) > 1.0

    def test_requires_no_intercept(self, data):
        X, S, _ = data
        with pytest.raises(ValueError, match="fit_intercept=False"):
            LinearModel(X, S, fit_intercept=True).minimize_penalty()



# ---------------------------------------------------------------------------------
# NUMERICAL -- regressions for bugs that well-conditioned random data cannot see
# ---------------------------------------------------------------------------------


class TestNumerical:
    @pytest.mark.parametrize("seed", range(15))
    def test_unpenalised_directions_are_exactly_unpenalised(self, seed):
        """`eigh` returns a structurally zero eigenvalue as ~1e-17. Clamping negatives
        instead of thresholding leaves that in place, and at lmbda = 1e14 the product
        is large enough to shrink a direction the penalty does not reach -- measured
        edf drifted to 1.956 where the exact floor is 2, on ~85% of seeds."""
        rng = np.random.default_rng(seed)
        m = LinearModel(rng.normal(size=(30, P)), psd(rng, P, 4))
        assert np.isclose(m.edf(1e14), 2.0, atol=1e-9)

    @pytest.mark.parametrize("lmbda", [1e-6, 1e-3, 1.0, 1e3, 1e6])
    def test_ill_conditioned_design_keeps_its_spectrum(self, lmbda):
        """Whitening by D1^-1 multiplies the dynamic range of the spectrum by
        cond(Xc)^2, so an EPS-relative cut on the *whitened* eigenvalues sits above
        genuinely penalised directions. That zeroed two real directions here and gave
        a fit wrong by a factor of 1356 at lmbda=1000, with edf 2.0 against a true
        0.045. The rank must be decided on Stil and transferred by inertia."""
        rng = np.random.default_rng(21)
        X = rng.normal(size=(60, P)) * np.array([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
        y, S = rng.normal(size=60), np.eye(P)
        assert np.linalg.cond(X) > 1e4
        m = fitted(X, S, y, lmbda)
        ref = dense_beta(X, S, y, lmbda)
        assert np.linalg.norm(m.beta - ref) / np.linalg.norm(ref) < 1e-6
        assert np.isclose(m.edf(), np.trace(dense_hat(X, S, lmbda)))

    def test_penalty_supported_on_the_null_space_is_accepted(self, rng):
        """S = uu' with u spanning null(X) is valid and PSD, but makes the Schur
        complement cancel to exact zero -- its computed spectrum is roundoff of both
        signs. A definiteness test relative to that spectrum rejects valid input; the
        `scale` argument is what prevents it."""
        X = rng.normal(size=(30, P))
        X[:, 5] = X[:, 0] + X[:, 1]
        u = np.zeros(P)
        u[[0, 1, 5]] = [1, 1, -1]
        for magnitude in (1e-6, 1.0, 1e6):
            m = LinearModel(X, magnitude * np.outer(u, u))
            assert (m.omega == 0).all()
            assert np.isclose(m.edf(1e14), 5.0)

    def test_a_tiny_schur_complement_survives_a_large_penalty(self, rng):
        """The converse of the test above, and why `_build_cache` damps its `scale` by
        EPS before passing it. Most of S sits on null(X) at 1e6, but a real component
        sits on the row space at 1e-3, so Stil is legitimately nine decades below
        abs(S).max(). Passing that bound undamped puts the cut at EPS * 1e6 = 1.5e-2,
        above the surviving direction: omega is zeroed entirely and edf(1e14) reads 5
        rather than 4."""
        X = rng.normal(size=(30, P))
        X[:, 5] = X[:, 0] + X[:, 1]
        u = np.zeros(P)
        u[[0, 1, 5]] = [1, 1, -1]
        u /= np.linalg.norm(u)
        v = np.linalg.svd(X, full_matrices=True)[2].T[:, 0]  # unit, in row(X)
        m = LinearModel(X, 1e6 * np.outer(u, u) + 1e-3 * np.outer(v, v))
        assert len(m.omega) == P - 1
        assert (m.omega > 0).sum() == 1
        assert np.isclose(m.edf(1e14), 4.0)

    def test_penalty_blind_to_the_null_space_of_the_design(self):
        """The third corner of the two above, and a defect until `_pinv_solve` was
        floored: `S` is PSD with `S @ u == 0` for the `u` spanning `null(X)`, so `S0` is
        roundoff throughout -- 2.0e-26 here, against an `abs(S).max()` of 1.3e+05.
        Inverting that amplified by 5e25 and left the Schur complement with a large
        negative eigenvalue, so this raised `S must be positive semi-definite` and blamed
        the penalty for a tolerance. The seed is one of 14 in 400 that did.

        Independent reference: the dense normal equations by `lstsq`. `dense_hat` and
        `model_lmbda` cannot serve here -- `u` lies in the null space of both `X` and
        `S`, so `X'X + lmbda S` is singular and their `solve` has nothing to return --
        hence `lstsq` and the EDF alignment done inline. Fitted values rather than
        `beta`, which the flat direction leaves non-unique while the fit is unique.
        """
        rng = np.random.default_rng(1)
        p = 8
        X = rng.normal(size=(40, p))
        X[:, p - 1] = X[:, 0] + X[:, 1]
        u = np.zeros(p)
        u[[0, 1, p - 1]] = [1.0, 1.0, -1.0]
        B = rng.normal(size=(p - 1, p)) @ (np.eye(p) - np.outer(u, u) / (u @ u))
        S = 1e4 * (B.T @ B)  # PSD, and S @ u == 0
        y = rng.normal(size=40)

        m = LinearModel(X, S)
        m.set_y(y)
        assert len(m.omega) == p - 1  # the rank of the design
        assert (m.omega > 0).all()  # what was dropped was the direction S never reached
        for dense_lmbda in (1e-3, 1.0, 1e3):
            H = X @ np.linalg.lstsq(X.T @ X + dense_lmbda * S, X.T, rcond=None)[0]
            rho = brentq(
                lambda r: m.edf(np.exp(r)) - np.trace(H), np.log(1e-12), np.log(1e12)
            )
            m.set_lmbda(np.exp(rho))
            want = H @ y
            got = m.fit().predict()
            assert np.allclose(got, want, rtol=0, atol=1e-11 * np.abs(want).max())

    @pytest.mark.parametrize("route", ["constructor", "add_constraints"])
    def test_constraint_in_the_null_space_is_a_no_op(self, rng, route):
        """Xc == 0 constrains nothing, so it must not consume a degree of freedom.
        Ctilde is then roundoff, and its rank cannot be judged against its own
        largest singular value.

        Both routes, because each states the floor in its own coordinates -- the
        constructor as `EPS * D1.max() * max(norm(c))` before `A_inv` exists, and the
        mutator as `EPS * norm(A_inv, 2) * max(norm(c))`. Those are the same number,
        which is the claim under test: unfloor either and that route alone eats a
        dimension."""
        X = rng.normal(size=(30, P))
        X[:, 5] = X[:, 0] + X[:, 1]
        u = np.zeros(P)
        u[[0, 1, 5]] = [1, 1, -1]
        assert len(LinearModel(X, np.eye(P)).omega) == P - 1
        assert len(constrained(X, np.eye(P), u, route).omega) == P - 1

    @pytest.mark.parametrize("route", ["constructor", "add_constraints"])
    def test_vacuous_constraint_at_a_size_where_roundoff_accumulates(self, route):
        """The same trap, sized so that the margin actually matters. Ctilde's roundoff
        grows with the problem: at (30, 6) it sits at 1.8e-16 * scale, at (200, 40) at
        4.8e-16. Both routes therefore pass `scale` to `svd` undamped -- an EPS-damped
        floor puts the cut at eps * scale, level with the roundoff itself, and this
        constraint silently eats a degree of freedom."""
        rng = np.random.default_rng(0)
        n, p = 200, 40
        X = rng.normal(size=(n, p))
        X[:, p - 1] = X[:, 0] + X[:, 1]
        u = np.zeros(p)
        u[[0, 1, p - 1]] = [1, 1, -1]
        assert len(LinearModel(X, np.eye(p)).omega) == p - 1
        assert len(constrained(X, np.eye(p), u, route).omega) == p - 1

    def test_null_space_constraint_mixed_with_a_real_one(self, rng):
        X = rng.normal(size=(30, P))
        X[:, 5] = X[:, 0] + X[:, 1]
        u = np.zeros(P)
        u[[0, 1, 5]] = [1, 1, -1]
        C = np.vstack([rng.normal(size=P), u])
        assert len(LinearModel(X, np.eye(P)).add_constraints(C).omega) == P - 2

    @pytest.mark.parametrize("lmbda", [1e-6, 1.0, 1e6])
    def test_constraints_on_an_ill_conditioned_design(self, lmbda):
        rng = np.random.default_rng(31)
        X = rng.normal(size=(60, P)) * np.array([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
        y, S = rng.normal(size=60), psd(rng, P)
        C = rng.normal(size=(2, P))
        m = LinearModel(X, S).add_constraints(C)
        m.set_y(y)
        m.set_lmbda(model_lmbda(X, S, lmbda, C=C))
        m.fit()
        ref = dense_kkt(X, S, y, lmbda, C)
        assert np.linalg.norm(m.beta - ref) / np.linalg.norm(ref) < 1e-5
        assert np.abs((X @ C.T).T @ (X @ m.beta)).max() / np.abs(X @ m.beta).max() < 1e-12

    def test_reml_log_term_survives_a_tiny_penalty_product(self, rng):
        """The log-determinant term collapses algebraically to -sum(log(1 - a_i)),
        which is prettier -- but when lmbda*omega drops below machine precision `a`
        rounds to exactly 1, 1 - a is 0, and the term is +inf across the bottom of the
        range. Written in log1p(lmbda*omega) it stays finite.

        Reaching it takes a range, not a design: `omega` is normalised, so no scaling of
        X or S moves the product any more, and the bottom of the search range is the
        only thing that decides. This one is set well below the default to get there."""
        X = rng.normal(size=(N, P))
        m = LinearModel(X, np.eye(P))
        m.set_y(rng.normal(size=N))
        grid = np.geomspace(1e-20, 1e6, 60)
        assert (grid[0] * m.omega).min() < 1e-16, "range does not exercise the trap"
        c = lm.reml()
        c.attach(m)
        assert np.all(np.isfinite(c.criterion(grid)))


# ---------------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------------


def test_penalised_spline_recovers_a_smooth_signal():
    """The workflow the class exists for: build one cache, then sweep responses and
    lmbdas off it. A truncated cubic basis with a wiggle-only penalty, fitted to noisy
    sin(2 pi t) with each of the four selectors."""
    rng = np.random.default_rng(11)
    t = np.linspace(0, 1, 120)
    knots = np.linspace(0, 1, 14)
    X = np.column_stack(
        [np.ones_like(t), t] + [np.clip(t - k, 0, None) ** 3 for k in knots]
    )
    S = np.zeros((X.shape[1],) * 2)
    S[2:, 2:] = np.eye(len(knots))
    truth = np.sin(2 * np.pi * t)

    models = {
        name: LinearModel(
            X, S, lmbda_criterion=getattr(lm, name)()
        )
        for name in ("gcv", "reml", "aic", "bic")
    }
    for _ in range(5):
        y = truth + 0.1 * rng.normal(size=len(t))
        for name, m in models.items():
            m.set_y(y)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit_lmbda()
            m.fit()
            rmse = np.sqrt(((m.predict() - truth) ** 2).mean())
            assert rmse < 0.06, f"{name} rmse {rmse}"
