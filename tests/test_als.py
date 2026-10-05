"""Tests for `als`.

The module's claims are about an *iteration*, so the references here are about where it
goes and how fast, not only about the fit it ends on.

What is independent
-------------------

One full sweep is `lm2` fitted to `f`, then `lm1` fitted to that, so in terms of the two
dense hat matrices it is

    f <- H1 @ H2 @ f,   rescaled to ||f||^2 / n == 1

-- power iteration on `H1 @ H2`, whose fixed point is that matrix's dominant eigenvector
and whose rate is its eigenvalue gap. `H1` and `H2` are assembled here from `X`, `S` and
a `lmbda` in the units of `S`, by `np.linalg.solve` on the normal equations; the module
under test never forms them, working in the pushforward basis throughout. So both the
limit and the rate are checked against a formulation that shares no arithmetic with the
code, and `A1`, `B`, `K` and `omega` appear nowhere below.

The second reference is `EigenFit`. The module docstring claims the two trace the same
solution path at different indexings of `lmbda1` -- specifically that `EigenFit`'s
solution at `lmbda1` is the ALS fixed point at `lmbda1 / (-theta)`, where `1 + theta` is
the optimal objective over `n`. `theta` is recovered here from `joint_objective`, which
is the objective written out in `test_bilinear`, so the correspondence is checked
without reading an eigenvalue out of either implementation.

Fixing `lmbda` so the iteration can be studied
----------------------------------------------

`ALSFit.fit` re-selects both `lmbda`s every sweep, which is its whole reason to exist
and is also what would make the sweep a moving target: the convergence results above are
stated at fixed `lmbda`, and `A1 B` is not one operator if `A1` changes underneath it.
`Pin` is a `LmbdaCriterion` that selects a constant, so the alternation can be watched
against a fixed dense operator. It is scaffolding for these tests and asserts nothing by
itself; `TestSelection` covers the selecting behaviour it suppresses.
"""

import warnings

import numpy as np
import pytest

from maniprobe.als import ALSFit, Anderson
from maniprobe.bilinear import BilinearModel, EigenFit
from maniprobe.linear_model import LinearModel, LmbdaCriterion
from test_bilinear import joint_objective
from test_linear_model import N, P, model_lmbda, psd, up_to_sign


class Pin(LmbdaCriterion):
    """A criterion that selects a constant; see the module docstring."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def fit_lmbda(self):
        return self.value


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def pair(rng):
    """Two designs and penalties, `X1` centred as the joint fit requires. Well
    separated: the gap of `H1 @ H2` is 0.67 at the `lmbda`s used below, so the plain
    sweep converges in tens of iterations and the tolerance means what it says."""
    X1 = rng.normal(size=(N, P))
    X2 = rng.normal(size=(N, P - 2))
    return X1 - X1.mean(0), psd(rng, P), X2, psd(rng, P - 2)


@pytest.fixture
def tight_pair():
    """A design whose gap is 0.991, built by giving `X2` two directions of `X1` of
    near-equal predictability. This is the regime the acceleration exists for, and the
    one that separates a tolerance on the iterate from one on the objective; nothing
    else here needs it."""
    rng = np.random.default_rng(1)
    X1 = rng.normal(size=(N, P))
    X1 = X1 - X1.mean(0)
    v1, v2 = rng.normal(size=P), rng.normal(size=P)
    X2 = np.column_stack(
        [
            X1 @ v1,
            X1 @ v2,
            X1 @ v1 + 1e-3 * rng.normal(size=N),
            X1 @ v2 + 1e-3 * rng.normal(size=N),
        ]
    )
    return X1, psd(rng, P), X2, psd(rng, 4)


def als(pair, l1, l2, **kwargs):
    """An `ALSFit` over fresh models with both `lmbda`s pinned. `l1` and `l2` are in the
    units of the penalties, as the dense references below read them."""
    X1, S1, X2, S2 = pair
    return ALSFit(
        LinearModel(X1, S1, lmbda_criterion=Pin(model_lmbda(X1, S1, l1))),
        LinearModel(X2, S2, lmbda_criterion=Pin(model_lmbda(X2, S2, l2))),
        **kwargs,
    )


def sweep_eigen(pair, l1, l2):
    """Reference: the eigenpairs of the dense sweep operator `H1 @ H2`.

    Returns `(mu, vectors)` in decreasing order of eigenvalue, each vector scaled to
    `||v||^2 / n == 1` so it can be compared with a fitted `f` directly. The hat
    matrices are formed from the normal equations, which is the formulation the module
    avoids; see the module docstring.
    """
    X1, S1, X2, S2 = pair
    H1 = X1 @ np.linalg.solve(X1.T @ X1 + l1 * S1, X1.T)
    H2 = X2 @ np.linalg.solve(X2.T @ X2 + l2 * S2, X2.T)
    mu, V = np.linalg.eig(H1 @ H2)
    order = np.argsort(mu.real)[::-1]
    V = V.real[:, order]
    return mu.real[order], V * np.sqrt(N) / np.linalg.norm(V, axis=0)


def iterate(pair, l1, l2, k, **kwargs):
    """`f` after exactly `k` sweeps. `tol=0` disables the stopping rule so the sweep
    count is the one asked for, which costs the `max_iter` warning on every call."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return als(pair, l1, l2, max_iter=k, tol=0.0, **kwargs).fit().lm1.predict()


class TestSweep:
    def test_fixed_point_is_the_dominant_eigenvector(self, pair):
        """INDEPENDENT. Against the dominant eigenvector of the dense `H1 @ H2`, which
        is what the alternation is power iteration on. This is the module's central
        claim and the reference shares no arithmetic with it."""
        mu, V = sweep_eigen(pair, 0.3, 0.7)
        f = als(pair, 0.3, 0.7, tol=1e-14, max_iter=5000).fit().lm1.predict()
        assert up_to_sign(f, V[:, 0]) / np.sqrt(N) < 1e-10

    def test_rate_is_the_eigenvalue_gap(self, pair):
        """INDEPENDENT, and the sharpest test here. The error against the fixed point
        must fall by `mu2 / mu1` per sweep -- not merely `O(rate^k)` but that ratio
        itself, to four decimals.

        On the tail only, and that is the claim rather than a weakening of it: the rate
        is what the error settles to *once the subdominant direction dominates it*. The
        first ratios run 0.653, 0.685, 0.682 while the third and later eigenvectors are
        still dying off; from the eleventh sweep on, every ratio is within 3.5e-5 of a
        gap of 0.672362."""
        mu, V = sweep_eigen(pair, 0.3, 0.7)
        gap = mu[1] / mu[0]
        errs = np.array(
            [up_to_sign(iterate(pair, 0.3, 0.7, k), V[:, 0]) for k in range(1, 26)]
        )
        assert np.all(np.diff(errs) < 0)  # monotone, with no acceleration
        ratios = errs[1:] / errs[:-1]
        assert np.allclose(ratios[10:], gap, atol=1e-3)
        assert not np.allclose(ratios[:3], gap, atol=1e-3)

    def test_constraint_holds_after_every_sweep(self, pair):
        """`normalize_beta` is the rescaling step of the power iteration, so the scale
        constraint is exact at every `k`, not only in the limit. `sum(f) == 0` comes
        free from the centred design."""
        for k in (1, 2, 5, 20):
            f = iterate(pair, 0.3, 0.7, k)
            assert abs(f.var() - 1.0) < 1e-12
            assert abs(f.mean()) < 1e-12

    def test_starts_from_the_least_penalised_direction(self, pair):
        """The documented initialisation, and the one sweep at which it is observable:
        `lm2` opens the loop fitted to `lm1.minimize_penalty()`'s fitted values, which
        is why the sweep runs `lm2` first and why no `lmbda` is needed to state a start.

        Worth pinning only here. At fixed `lmbda` the sweep is power iteration with a
        single attractor, so from the second sweep on any start converges to the same
        place and the choice becomes unobservable in the answer -- replacing it with a
        different deterministic start breaks nothing further down this file."""
        X1, S1, X2, S2 = pair
        f0 = LinearModel(X1, S1).minimize_penalty().predict()

        a = als(pair, 0.3, 0.7, max_iter=1, tol=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            a.fit()

        want = LinearModel(X2, S2)
        want.set_lmbda(model_lmbda(X2, S2, 0.7))
        want.set_y(f0)
        want.fit()
        assert np.allclose(a.lm2.beta, want.beta, rtol=1e-14)

    def test_is_deterministic(self, pair):
        """Started from `lm1.minimize_penalty()`, which needs no `lmbda` and draws no
        random numbers, so two runs agree to the bit."""
        a = als(pair, 0.3, 0.7).fit()
        b = als(pair, 0.3, 0.7).fit()
        assert np.array_equal(a.lm1.beta, b.lm1.beta)
        assert np.array_equal(a.lm2.beta, b.lm2.beta)
        assert a.n_iter == b.n_iter

    def test_r2_is_the_inherited_one(self, pair):
        """`ALSFit` gets `r2` from `BilinearModel` and does not override it; that it is
        reached at all through the new base is what this checks."""
        a = als(pair, 0.3, 0.7).fit()
        f, g = a.lm1.predict(), a.lm2.predict()
        assert isinstance(a, BilinearModel)
        assert np.isclose(a.r2(), 1.0 - np.mean((f - g) ** 2) / f.var(), rtol=1e-12)

    def test_u_is_the_inherited_one(self, pair):
        """`u` too, and here it is the base class's own `_ybar2` that runs: `ALSFit`
        holds no `K` to reach `U2.T @ f` through, which is the whole reason that method
        exists. `beta1` has been through Anderson mixing and a rescaling by then, so
        this also covers the claim that `pushforward` recovers `nu` from any `beta1` the
        loop can leave behind."""
        X2 = pair[2]
        a = als(pair, 0.3, 0.7).fit()
        f = a.lm1.predict()
        want = (X2 - X2.mean(0)).T @ f / N
        assert np.allclose(a.u(), want, atol=1e-12 * np.abs(want).max())


class TestCorrespondenceWithEigenFit:
    @pytest.mark.parametrize("l1, l2", [(0.3, 0.7), (0.05, 0.05), (2.0, 0.4)])
    def test_same_path_at_a_rescaled_lmbda1(self, pair, l1, l2):
        """INDEPENDENT, and the module docstring's main structural claim: `EigenFit`'s
        solution at `lmbda1` is the ALS fixed point at `lmbda1 / (-theta)`, so the two
        trace one path differently indexed. `theta` is recovered from the optimal
        objective (`1 + theta == J* / n`) using `test_bilinear.joint_objective`, so no
        eigenvalue is read out of either implementation."""
        X1, S1, X2, S2 = pair
        e1 = LinearModel(X1, S1)
        e2 = LinearModel(X2, S2)
        EigenFit(e1, e2).fit(model_lmbda(X1, S1, l1), model_lmbda(X2, S2, l2))

        theta = joint_objective(X1, S1, l1, X2, S2, l2, e1, e2) / N - 1.0
        assert -1.0 < theta < 0.0  # the sign that makes the reindexing increasing

        f = als(pair, l1 / -theta, l2, tol=1e-14, max_iter=5000).fit().lm1.predict()
        assert up_to_sign(f, e1.predict()) / np.sqrt(N) < 1e-9

    def test_at_a_common_lmbda_they_differ(self, pair):
        """The other half of that claim: the paths agree but the labels do not, so at
        the *same* `lmbda1` the two give different fits -- ALS's the less smoothed.

        Measured as the penalty actually incurred, `beta1' S1 beta1` (12.86 against
        10.31), and not as `edf`: `edf` is a function of `lmbda` alone, so at a common
        label the two are equal to the bit and could not show this either way."""
        X1, S1, X2, S2 = pair
        e1, e2 = LinearModel(X1, S1), LinearModel(X2, S2)
        EigenFit(e1, e2).fit(model_lmbda(X1, S1, 0.3), model_lmbda(X2, S2, 0.7))
        a = als(pair, 0.3, 0.7, tol=1e-14, max_iter=5000).fit()

        assert up_to_sign(a.lm1.predict(), e1.predict()) / np.sqrt(N) > 1e-3
        assert a.lm1.beta @ S1 @ a.lm1.beta > e1.beta @ S1 @ e1.beta


class TestAcceleration:
    def test_depth_one_is_a_no_op(self, pair):
        """`accel=1` leaves no difference to extrapolate from, so the step must not
        touch `beta` however often it is called -- which is what makes 1 the plain
        alternation rather than a special case of the code path."""
        X1, S1, _, _ = pair
        m = LinearModel(X1, S1)
        m.set_lmbda(0.1)
        m.set_y(X1 @ np.arange(1.0, P + 1))
        m.fit()
        m.normalize_beta()

        step, before = Anderson(1), m.beta.copy()
        for _ in range(3):
            step(m)
        assert np.array_equal(m.beta, before)

    def test_reaches_the_same_point_far_sooner(self, tight_pair):
        """What the acceleration is for. At a gap of 0.991 the plain sweep needs some
        hundreds of sweeps to reach the fixed point and `accel=2` needs single figures,
        both landing on the dominant eigenvector to 1e-8. It is the rate it attacks, so
        this is the regime it pays in; `pair`, at a gap of 0.67, would show nothing.

        `tol=1e-12` rather than something smaller: the tolerance is now a residual, and
        below machine precision on a unit-variance `f` it is unreachable, so a tighter
        request would only buy `max_iter` sweeps and a warning.

        That both land on the *dominant* one is a property of this design and not a
        guarantee: the step is a root-finder, every eigenvector of the sweep is a root,
        and the module docstring accepts that it sometimes settles on the second."""
        mu, V = sweep_eigen(tight_pair, 0.05, 0.05)
        assert mu[1] / mu[0] > 0.95

        plain = als(tight_pair, 0.05, 0.05, tol=1e-12, max_iter=20000).fit()
        quick = als(tight_pair, 0.05, 0.05, tol=1e-12, max_iter=20000, accel=2).fit()
        for a in (plain, quick):
            assert up_to_sign(a.lm1.predict(), V[:, 0]) / np.sqrt(N) < 1e-8
        assert quick.n_iter < plain.n_iter / 20

    def test_keeps_the_constraint_while_extrapolating(self, tight_pair):
        """The extrapolated iterate does not satisfy it -- a combination of unit-norm
        vectors is not unit-norm -- so `Anderson` rescales after replacing `beta`.

        Checked over the *early* sweeps, and that is the whole content of the test. Run
        to convergence it asserts nothing: the extrapolation coefficients go to zero
        there, so the combination is the identity and the norm is already 1 whether or
        not anything rescaled it. With the rescale removed the deviation appears only at
        sweeps 3 to 5, peaking at 5.6e-4 and gone again by sweep 8."""
        for k in (2, 3, 4, 5, 6, 8):
            f = iterate(tight_pair, 0.05, 0.05, k, accel=4)
            assert abs(f.var() - 1.0) < 1e-12


class TestStopping:
    def test_warns_when_max_iter_is_exhausted(self, tight_pair):
        # `tol=0` so the residual test cannot fire: at this gap the default tolerance
        # is satisfied within three sweeps, which is the point of the test below.
        with pytest.warns(UserWarning, match="no convergence"):
            a = als(tight_pair, 0.05, 0.05, max_iter=3, tol=0.0).fit()
        assert a.n_iter == 3

    def test_n_iter_is_none_before_fit(self, pair):
        assert als(pair, 0.3, 0.7).n_iter is None

    def test_tolerance_predicts_the_distance_to_the_fixed_point(self, pair, tight_pair):
        """NUMERICAL, and the property the residual test exists for. `||f - f_old||` is
        first order in the distance to the fixed point, with factor `(1 - rho) / rho`,
        so the error on stopping should be about `tol * rho / (1 - rho)` -- at *both*
        gaps, which is what makes `tol` a number a caller can reason about.

        Measured at the default `tol=1e-4`: 1.5e-4 at a gap of 0.672 against 2.0e-4
        predicted, and 1.09e-2 at 0.991 against 1.1e-2. A factor of three either way is
        pinned, since the relation is asymptotic in the sweep count."""
        for p, l1, l2 in ((pair, 0.3, 0.7), (tight_pair, 0.05, 0.05)):
            mu, V = sweep_eigen(p, l1, l2)
            rho = mu[1] / mu[0]
            a = als(p, l1, l2).fit()
            e = up_to_sign(a.lm1.predict(), V[:, 0]) / np.sqrt(N)
            assert 1 / 3 < e / (a.tol * rho / (1.0 - rho)) < 3

    def test_does_not_stop_while_the_iterate_is_still_moving(self, tight_pair):
        """REGRESSION. The defect the residual test replaced: at a gap of 0.991 a
        tolerance on the objective's decrement stopped after 3 sweeps, 0.028 from the
        fixed point, having left 0.035 of training `R^2` unclaimed -- more than
        converging to the wrong eigenvector costs, and silently, since `max_iter` was
        never reached. The residual test runs 106 sweeps and leaves 9e-8."""
        mu, V = sweep_eigen(tight_pair, 0.05, 0.05)
        a = als(tight_pair, 0.05, 0.05).fit()
        assert 50 < a.n_iter < a.max_iter
        assert up_to_sign(a.lm1.predict(), V[:, 0]) / np.sqrt(N) < 2e-2

    def test_converged_records_which_way_the_loop_ended(self, pair, tight_pair):
        """The fact the warning already carries, left where a caller can read it: a
        display marking a fit that stopped short cannot use the warning, which Python's
        default filter fires once per session rather than once per fit."""
        assert als(pair, 0.3, 0.7).fit().converged is True
        with pytest.warns(UserWarning, match="no convergence"):
            a = als(tight_pair, 0.05, 0.05, max_iter=3, tol=0.0).fit()
        assert a.converged is False


class TestSelection:
    def test_requires_a_criterion_on_both_models(self, pair):
        """`ALSFit` needs no `lmbda` set, since the first sweep selects both -- but it
        does need something to select with, and the failure is `fit_lmbda`'s."""
        X1, S1, X2, S2 = pair
        for which in ("lm1", "lm2"):
            m1 = LinearModel(X1, S1, lmbda_criterion=None if which == "lm1" else Pin(1))
            m2 = LinearModel(X2, S2, lmbda_criterion=None if which == "lm2" else Pin(1))
            with pytest.raises(ValueError, match="no lmbda_criterion"):
                ALSFit(m1, m2).fit()

    def test_ignores_any_lmbda_already_set(self, pair):
        """The constructor's promise: a `lmbda` on the models is replaced on the first
        sweep, so two runs differing only in it agree."""
        a = als(pair, 0.3, 0.7)
        a.lm1.set_lmbda(123.0)
        a.lm2.set_lmbda(456.0)
        assert np.allclose(a.fit().lm1.beta, als(pair, 0.3, 0.7).fit().lm1.beta)

    def test_selects_both_lmbdas_from_their_criteria(self, pair):
        """With a real criterion attached rather than `Pin`, both `lmbda`s must end at
        values their own criterion chose -- the behaviour the rest of the file pins
        down. `gcv` on each side, checked by re-selecting at the converged fit."""
        from maniprobe.linear_model import gcv

        X1, S1, X2, S2 = pair
        m1 = LinearModel(X1, S1, lmbda_criterion=gcv())
        m2 = LinearModel(X2, S2, lmbda_criterion=gcv())
        a = ALSFit(m1, m2, tol=1e-12, max_iter=2000).fit()

        assert m1.lmbda > 0 and m2.lmbda > 0
        assert np.isclose(m1.lmbda_criterion.fit_lmbda(), m1.lmbda, rtol=1e-12)
        assert np.isclose(m2.lmbda_criterion.fit_lmbda(), m2.lmbda, rtol=1e-12)


class TestReport:
    """The callback that replaced `verbose`. Nothing here prints: the one display is
    `probe.Progress`, which drives a sequence of these fits and cannot have each of them
    writing its own line."""

    def test_is_called_once_per_sweep(self, pair):
        calls = []
        a = als(pair, 0.3, 0.7, report=lambda i, total, m: calls.append((i, total)))
        a.fit()
        assert [i for i, _ in calls] == list(range(1, a.n_iter + 1))
        assert {total for _, total in calls} == {a.max_iter}

    def test_reports_the_training_r2_it_claims_to(self, pair):
        """`1 - mean((f - g)**2)`, which the scale constraint on `f` makes the training
        `R^2` without a second pass over the data. `g` is the previous half-sweep's, so
        the two agree exactly only in the limit -- hence a converged fit and a tolerance
        rather than an identity."""
        calls = []
        a = als(
            pair, 0.3, 0.7, tol=1e-12, max_iter=5000,
            report=lambda i, total, m: calls.append(m),
        ).fit()
        assert set(calls[-1]) == {"train R²"}
        assert np.isclose(calls[-1]["train R²"], a.r2(), rtol=1e-9)

    def test_reports_the_test_r2_only_with_a_test_set(self, pair):
        """`test` is the argument pair for the two `predict`s -- held-out rows of the
        two designs here. The last sweep reports at the state the fit ends on, so that
        one is exact against `r2` afterwards."""
        X1, _, X2, _ = pair
        calls = []
        a = als(
            pair, 0.3, 0.7, test=(X1[:10], X2[:10]),
            report=lambda i, total, m: calls.append(m),
        ).fit()
        assert set(calls[-1]) == {"train R²", "test R²"}
        assert calls[-1]["test R²"] == a.r2(X1[:10], X2[:10])

    def test_prints_nothing(self, pair, capsys):
        als(pair, 0.3, 0.7, report=lambda i, total, m: None).fit()
        assert capsys.readouterr().out == ""


class TestCriterion:
    """`als()` and `ALSCriterion` -- the alternation as a specification, which is what a
    caller hands to `probe.Probe` rather than a fitter over models it has not built.

    `als` is imported inside each test because this module's own helper of that name
    would otherwise shadow it, as `gcv` already is one test below.
    """

    def test_defaults_are_the_probe_facing_ones(self):
        """`gcv()` when nothing is named, and `accel=3` rather than `ALSFit`'s own 1:
        this is where a caller who never meets the alternation gets a depth."""
        from maniprobe.als import als
        from maniprobe.linear_model import GCV

        c = als()
        assert isinstance(c.criterion, GCV)
        assert (c.tol, c.max_iter, c.accel) == (1e-4, 200, 3)
        assert (als(accel=5).accel, als(accel=5).max_iter) == (5, 200)

    def test_fitter_attaches_a_copy_to_each_model(self, pair):
        """Two instances and not one: `attach` binds a criterion to a single model, so
        one object handed to both would leave each half-sweep selecting against the
        other's cache. The copies carry the original's settings, `copy.copy` being exact
        for an object holding scalars and a reference `attach` overwrites."""
        from maniprobe.als import als
        from maniprobe.linear_model import gcv

        X1, S1, X2, S2 = pair
        criterion = gcv(gamma=1.4)
        sm, lm = LinearModel(X1, S1), LinearModel(X2, S2)
        a = als(criterion, tol=1e-6, max_iter=7, accel=2).fitter(sm, lm, None)

        assert isinstance(a, ALSFit) and a.lm1 is sm and a.lm2 is lm
        assert (a.tol, a.max_iter, a.accel, a.test) == (1e-6, 7, 2, None)
        assert sm.lmbda_criterion is not criterion
        assert lm.lmbda_criterion is not sm.lmbda_criterion
        assert sm.lmbda_criterion.model is sm and lm.lmbda_criterion.model is lm
        assert sm.lmbda_criterion.gamma == lm.lmbda_criterion.gamma == 1.4

    def test_the_fitter_it_builds_selects_both_lmbdas(self, pair):
        """End to end, since the copies are only useful if each is left selecting for
        the model it was attached to."""
        from maniprobe.als import als

        X1, S1, X2, S2 = pair
        sm, lm = LinearModel(X1, S1), LinearModel(X2, S2)
        a = als(tol=1e-12, max_iter=2000).fitter(sm, lm, None).fit()

        assert sm.lmbda > 0 and lm.lmbda > 0
        assert np.isclose(sm.lmbda_criterion.fit_lmbda(), sm.lmbda, rtol=1e-12)
        assert np.isclose(lm.lmbda_criterion.fit_lmbda(), lm.lmbda, rtol=1e-12)
