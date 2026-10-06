"""Tests for `probe`.

`Probe` composes machinery that is already checked elsewhere -- the joint fit against a
dense reference in `test_bilinear.py`, the folds in `test_cv.py`, the alternation in
`test_als.py` -- so what is under test here is the *composition*: that the right
coefficients are recorded, from the right place, at the right moment, and that the right
row is fed back as the next component's constraint.

What is independent
-------------------

`F.T @ F / n == I` after `d` components. `F` is rebuilt here from the basis and the
recorded `c` alone, so nothing about `Probe` enters the reference: if the deflation used
the wrong row, or used it at the wrong moment, or was not applied at all, the
off-diagonal entries say so. It is the strongest structural check available, and it is
independent of `probe.py` rather than of `basis` and `linear_model`, which have their
own tests -- an orthonormal frame is a property of what this module composed, not of the
arithmetic that built each column.

`u` against `Xc.T @ f / n` written out in numpy. That one is independent arithmetic
as well as an independent formulation: `bilinear.u` never forms `Xc` and never forms
`f`, taking the whole of it through the cache as `A_inv.T @ K.T @ nu / n`.

Both `R^2`s recomputed from the recorded `c`, `w` and `b`. The formula is
`BilinearModel.r2`'s own, so what this checks is not the formula but that the three
coefficient vectors were taken off the models the fit left them on -- and, on the test
set, that `test_set` reached `predict` in the order those two models read it.

`fold_scores` after two components against a fresh `EigenCVFit` built on parents already
carrying the same rows. `test_cv.py` holds the two routes to a constrained fold to each
other directly; what is added here is the composition, that `Probe.fit` sends the
coefficients it recorded to `add_constraints` and sends them once each.

The pin
-------

`test_r2_over_three_components_is_pinned` is a regression pin, not a reference: it
records what the alternation reaches on this construction, and a change to it is a
behavioural change to be explained rather than a bug.
"""

import numpy as np
import pytest

from maniprobe.als import ALSFit, als
from maniprobe.basis import bs
from maniprobe.basis.splines import BSplineBasis
from maniprobe.cv import EigenCVFit, kf
from maniprobe.linear_model import GCV, LinearModel, gcv, reml
from maniprobe.linear_smoother import LinearSmoother
from maniprobe.probe import Probe, Progress
from maniprobe.search import grid

N = 200
N_TEST = 50
P = 6
K = 10


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def data(rng):
    """Two smooth functions of `z` loaded onto random directions in `R^P`, plus noise,
    split into a training and a test set.

    The construction is a reference for nothing below -- no test asserts that the fit
    recovers it. It is here so that the first two components are well determined and
    anything after them is mostly noise, which is the sequence the tests watch.
    """
    z = rng.uniform(size=N + N_TEST)
    F = np.column_stack([np.sin(2 * np.pi * z), np.cos(4 * np.pi * z)])
    X = F @ rng.normal(size=(2, P)) + 0.3 * rng.normal(size=(N + N_TEST, P))
    return X[:N], z[:N], (X[N:], z[N:])


def frame(basis, z, probe):
    """`F`, the fitted index of every component, from the basis and the recorded `c`."""
    return basis.model_matrix(z) @ np.column_stack([c.c for c in probe.components])


class TestConstruction:
    def test_rejects_an_uncentred_basis(self, data):
        """Without centring the constant is reachable, and the objective is zero at a
        fit that says nothing. `bs` always centres, so this is a hand-built basis."""
        X, z, _ = data
        with pytest.raises(ValueError, match="centred"):
            Probe(X, z, BSplineBasis(k=K, penalty=2))

    def test_a_bare_criterion_is_shorthand_for_alternating_on_it(self, data):
        """INDEPENDENT of any fit: `criterion=None`, `gcv()` and `als(gcv())` must
        build one `ALSFit` carrying one set of settings, which is the whole content of
        the shorthand."""
        X, z, _ = data
        fitters = [
            Probe(X, z, bs(K), criterion=c).fitter for c in (None, gcv(), als(gcv()))
        ]
        for fitter in fitters:
            assert isinstance(fitter, ALSFit)
            assert (fitter.tol, fitter.max_iter, fitter.accel) == (1e-4, 200, 3)
            for model in (fitter.lm1, fitter.lm2):
                criterion = model.lmbda_criterion
                assert isinstance(criterion, GCV)
                assert (criterion.gamma, criterion.grid_size, criterion.edf_tol) == (
                    1,
                    300,
                    0.1,
                )
                assert criterion.model is model
            # Two attachments and so two instances; one object bound to both models
            # would leave each half-sweep selecting against the other's cache.
            assert fitter.lm1.lmbda_criterion is not fitter.lm2.lmbda_criterion

    def test_als_changes_only_what_it_names(self, data):
        """`als(reml(), accel=5)` is the same specification with two fields moved."""
        X, z, _ = data
        fitter = Probe(X, z, bs(K), criterion=als(reml(), accel=5)).fitter
        assert (fitter.tol, fitter.max_iter, fitter.accel) == (1e-4, 200, 5)
        assert type(fitter.lm1.lmbda_criterion).__name__ == "REML"

    def test_a_cross_validated_criterion_builds_its_folds_at_construction(self, data):
        """`CVCriterion.fitter` is an `EigenCVFit`, whose folds are the expensive part
        and are built before anything is fitted."""
        X, z, _ = data
        probe = Probe(X, z, bs(K), criterion=kf(k=3, seed=0))
        assert isinstance(probe.fitter, EigenCVFit)
        assert len(probe.fitter.cv.folds) == 3
        assert probe.components == []


class TestTheFrame:
    def test_the_frame_is_orthonormal(self, data):
        """INDEPENDENT. `F.T @ F / n == I` after three components, `F` rebuilt from the
        basis and the recorded coefficients; see the module docstring. Measured 8e-16.

        Mutation-checked: dropping the `add_constraints` call, or moving it above the
        record so that the row constrained is not the row kept, breaks the off-diagonal
        entries or the record outright."""
        X, z, _ = data
        basis = bs(K)
        probe = Probe(X, z, basis).fit(3)
        F = frame(basis, z, probe)
        assert F.T @ F / N == pytest.approx(np.eye(3), abs=1e-12)

    def test_u_is_the_covariance_of_the_columns_with_the_index(self, data):
        """INDEPENDENT. `Xc.T @ f / n` in plain numpy, against a `u` that forms neither
        `Xc` nor `f`. Measured 2e-15."""
        X, z, _ = data
        basis = bs(K)
        probe = Probe(X, z, basis).fit(3)
        F = frame(basis, z, probe)
        for k, component in enumerate(probe.components):
            reference = (X - X.mean(0)).T @ F[:, k] / N
            assert component.u == pytest.approx(reference, abs=1e-11)

    def test_r2_is_recomputed_from_the_recorded_coefficients(self, data):
        """INDEPENDENT of where the numbers came from, though not of the formula: `f`
        from the basis and `c`, `g` from `X`, `w` and `b`, and
        `1 - mean((f - g)**2) / var(f)` written out. What it checks is that all three
        vectors were taken off the models the fit left them on."""
        X, z, _ = data
        basis = bs(K)
        probe = Probe(X, z, basis).fit(3)
        F = frame(basis, z, probe)
        for k, component in enumerate(probe.components):
            g = X @ component.w + component.b
            reference = 1.0 - np.mean((F[:, k] - g) ** 2) / F[:, k].var()
            assert component.r2 == pytest.approx(reference, rel=1e-10)

    def test_the_test_set_is_scored_in_the_order_it_was_given(self, data):
        """INDEPENDENT, and the check on the swap: `test_set` is `(X_test, z_test)` in
        the order of `(X, z)`, while `r2` takes the arguments to `lm1.predict` and
        `lm2.predict` -- the smoother first. Scored here against explicit numpy on the
        held-out rows."""
        X, z, (X_test, z_test) = data
        basis = bs(K)
        probe = Probe(X, z, basis, test_set=(X_test, z_test)).fit(2)
        F_test = basis.model_matrix(z_test) @ np.column_stack(
            [c.c for c in probe.components]
        )
        for k, component in enumerate(probe.components):
            g = X_test @ component.w + component.b
            reference = 1.0 - np.mean((F_test[:, k] - g) ** 2) / F_test[:, k].var()
            assert component.test_r2 == pytest.approx(reference, rel=1e-10)

    def test_there_is_no_test_r2_without_a_test_set(self, data):
        X, z, _ = data
        probe = Probe(X, z, bs(K)).fit(2)
        assert [c.test_r2 for c in probe.components] == [None, None]


class TestTheSequence:
    def test_n_components_is_a_total(self, data):
        """`fit(4, continue_fit=True)` after `fit(2)` fits two more and leaves the first
        two as the objects they were."""
        X, z, _ = data
        probe = Probe(X, z, bs(K)).fit(2)
        first = list(probe.components)
        probe.fit(4, continue_fit=True)
        assert len(probe.components) == 4
        assert probe.components[:2] == first

    def test_a_total_already_reached_does_nothing(self, data):
        """Keeping fewer components is a question about the map, not a reason to
        refit."""
        X, z, _ = data
        probe = Probe(X, z, bs(K)).fit(3)
        components, fitter = probe.components, probe.fitter
        probe.fit(2, continue_fit=True)
        assert probe.components is components and len(components) == 3
        assert probe.fitter is fitter

    def test_a_restart_rebuilds_the_smoother_and_keeps_the_linear_model(self, data):
        """Constraints cannot be removed, so starting over means a fresh smoother; the
        linear model is never constrained and every field the next fit touches is
        overwritten, so rebuilding it would cost a full SVD of `X` for nothing.

        Reproducing the fit exactly is what says the smoother is genuinely fresh: a
        kept one would fit its first component orthogonal to the three before it.

        Mutation-checked: rebuilding `lm`, or failing to rebuild `sm`.
        """
        X, z, _ = data
        probe = Probe(X, z, bs(K)).fit(3)
        sm, lm = probe.sm, probe.lm
        r2 = [c.r2 for c in probe.components]

        probe.fit(3)
        assert probe.sm is not sm
        assert probe.lm is lm
        assert [c.r2 for c in probe.components] == r2

    def test_warns_and_stops_when_the_smoother_runs_out_of_rank(self, data):
        """A centred basis of four functions has rank three, and each constraint costs
        one. Asking for more is a statement about the basis, so the three already
        fitted stand."""
        X, z, _ = data
        probe = Probe(X, z, bs(4))
        with pytest.warns(UserWarning, match="stopping at 3 components"):
            probe.fit(5)
        assert len(probe.components) == 3
        assert len(probe.sm.omega) == 0

    def test_r2_over_three_components_is_pinned(self, data):
        """REGRESSION PIN, not a reference: what the alternation reaches under `gcv()`
        on this construction. A failure here is a number that moved, and someone has to
        decide whether the move is acceptable."""
        X, z, _ = data
        probe = Probe(X, z, bs(K)).fit(3)
        assert [c.r2 for c in probe.components] == pytest.approx(
            [0.9830934823401034, 0.8853601011846952, 2.254021143111018e-05], rel=1e-9
        )


class TestTheFolds:
    def test_the_folds_reach_what_a_rebuild_would_have_reached(self, data):
        """INDEPENDENT, and the test that licenses building the fitter once. After two
        components, `fold_scores` at a *fixed* pair must agree with the same call on a
        fresh `EigenCVFit` whose parents were constructed carrying the rows the probe
        recorded.

        `test_cv.test_fold_scores_match_folds_built_constrained` holds the mutator and
        constructor routes to each other directly; what this adds is the composition --
        that `Probe.fit` sends the coefficients it recorded to `add_constraints`, once
        each and in order. To roundoff rather than exactly, for the reason that test
        gives, and at a fixed pair rather than at the searched outcome, so that a flat
        surface's argmax is not what is being measured.

        Mutation-checked: skipping the `K` refresh in `EigenFit.add_constraints`, which
        leaves every later fit raising on a stale cache.
        """
        X, z, _ = data
        basis = bs(K)
        probe = Probe(X, z, basis, criterion=kf(k=3, method=grid(grid_size=3), seed=0))
        probe.fit(2)

        fresh = EigenCVFit(
            LinearSmoother(z, basis, constraints=[c.c for c in probe.components]),
            LinearModel(X, fit_intercept=True),
            k=3,
            seed=0,
        )
        reference = fresh.cv.fold_scores(0.5, 2.0)
        assert probe.fitter.cv.fold_scores(0.5, 2.0) == pytest.approx(
            reference, rel=1e-9
        )
        # And the constraints are doing something: unconstrained folds score elsewhere.
        free = EigenCVFit(
            LinearSmoother(z, basis),
            LinearModel(X, fit_intercept=True),
            k=3,
            seed=0,
        )
        assert not np.allclose(reference, free.cv.fold_scores(0.5, 2.0))

    def test_the_parents_are_the_probes_own_models(self, data):
        """One shared reference, so the constraint the folds carry is the one the
        probe's smoother carries, and `sm.beta` after a fit is the component."""
        X, z, _ = data
        probe = Probe(X, z, bs(K), criterion=kf(k=3, method=grid(grid_size=2), seed=0))
        probe.fit(1)
        assert probe.fitter.cv.lm1 is probe.sm
        assert probe.fitter.cv.lm2 is probe.lm
        for fold, *_ in probe.fitter.cv.folds:
            assert np.array_equal(
                fold.lm1.constraints, np.atleast_2d(probe.components[0].c)
            )


class TestProgress:
    """`Progress` on its own, against lines written out by hand: it formats whatever
    dict it is handed and dispatches on nothing, so a fit is not needed to check it."""

    def test_the_running_line_is_rewritten_in_place(self, capsys):
        p = Progress(3)
        p(17, 200, {"train R²": 0.498120, "test R²": 0.451002})
        assert capsys.readouterr().out == (
            "\r  •  Component 3     17/200   train R²  0.498120   test R²  0.451002"
        )

    def test_the_counter_is_right_aligned_to_its_total(self, capsys):
        """So the metrics after it do not jog as the counter fills. Two counts under
        one total, since a full counter needs no padding and would pass either way."""
        p = Progress(2)
        p(7, 200, {"train R²": 0.524011})
        early = capsys.readouterr().out
        p(200, 200, {"train R²": 0.524011})
        assert "Component 2      7/200   " in early
        assert "Component 2    200/200   " in capsys.readouterr().out

    def test_finish_adds_to_what_was_last_reported(self, capsys):
        """A search's running metric is its cross-validated score and the two `R^2`s
        exist only once the winner has been refitted, so the closing line is the union
        of the two -- and under alternation the same labels are overwritten in place."""
        p = Progress(1)
        p(40, 40, {"cv R²": 0.560100})
        capsys.readouterr()
        p.finish({"train R²": 0.713301, "test R²": 0.690244})
        assert capsys.readouterr().out == (
            "\r  ✓  Component 1    40/40   cv R²  0.560100   train R²  0.713301"
            "   test R²  0.690244\n"
        )

    def test_an_unconverged_fit_is_marked_rather_than_raised_on(self, capsys):
        p = Progress(2)
        p(200, 200, {"train R²": 0.524011})
        capsys.readouterr()
        p.finish({"train R²": 0.524011}, converged=False)
        assert capsys.readouterr().out.startswith("\r  ⚠  Component 2    200/200")

    def test_a_shorter_line_is_padded_over_the_one_before_it(self, capsys):
        """A `\\r` leaves whatever the previous line wrote past its end."""
        p = Progress(0)
        p(1, 2, {"train R²": 0.5, "test R²": 0.5})
        long = capsys.readouterr().out
        p(2, 2, {"train R²": 0.5})
        short = capsys.readouterr().out
        assert len(short) == len(long)
        assert short.endswith(" " * len("   test R²  0.500000"))

    def test_a_verbose_fit_closes_one_line_per_component(self, data, capsys):
        """Split on the newlines only -- one per component -- and take what follows the
        last `\\r` of each, which is what a terminal is left showing."""
        X, z, (X_test, z_test) = data
        Probe(X, z, bs(K), test_set=(X_test, z_test), verbose=True).fit(2)
        lines = capsys.readouterr().out.split("\n")[:-1]
        assert len(lines) == 2
        for i, line in enumerate(lines):
            final = line.split("\r")[-1]
            assert final.startswith(f"  ✓  Component {i}")
            assert "train R²" in final and "test R²" in final

    def test_a_quiet_fit_prints_nothing(self, data, capsys):
        X, z, _ = data
        probe = Probe(X, z, bs(K)).fit(2)
        assert capsys.readouterr().out == ""
        assert probe.fitter.report is None


class TestInspect:
    def test_inspect_prints_a_row_per_component(self, data, capsys):
        X, z, (X_test, z_test) = data
        probe = Probe(X, z, bs(K), test_set=(X_test, z_test)).fit(2)
        probe.inspect()
        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == "Components fitted: 2"
        assert lines[1] == (
            f"  Component 0   train R² {probe.components[0].r2:9.6f}"
            f"   test R² {probe.components[0].test_r2:9.6f}"
        )
        assert len(lines) == 3

    def test_inspect_leaves_the_test_column_out_without_a_test_set(self, data, capsys):
        X, z, _ = data
        Probe(X, z, bs(K)).fit(1).inspect()
        lines = capsys.readouterr().out.splitlines()
        assert lines[1].startswith("  Component 0   train R² ")
        assert "test R²" not in lines[1]


class TestManifold:
    def test_the_frame_handed_over_is_the_one_the_probe_fitted(self, data):
        """INDEPENDENT, and the acceptance check on the two together: the map is built
        from the components as recorded, so its coordinates are still orthonormal under
        the empirical measure -- `f().T @ f() / n == I`. Measured 5e-16."""
        X, z, _ = data
        manifold = Probe(X, z, bs(K)).fit(3).manifold()
        F = manifold.f()
        assert F.T @ F / N == pytest.approx(np.eye(3), abs=1e-12)
        assert np.array_equal(manifold.T, np.eye(3))

    def test_no_argument_is_every_component(self, data):
        X, z, _ = data
        probe = Probe(X, z, bs(K)).fit(3)
        assert probe.manifold().r2 == pytest.approx([c.r2 for c in probe.components])

    def test_n_components_takes_the_first_of_them(self, data):
        X, z, _ = data
        probe = Probe(X, z, bs(K)).fit(3)
        manifold = probe.manifold(n_components=2)
        assert manifold.C.shape[1] == 2
        assert manifold.r2 == pytest.approx([c.r2 for c in probe.components[:2]])

    def test_component_idx_takes_them_in_the_order_given(self, data):
        """An index list is a frame and not only a selection, so the order is kept."""
        X, z, _ = data
        probe = Probe(X, z, bs(K)).fit(3)
        manifold = probe.manifold(component_idx=[2, 0])
        assert manifold.r2 == pytest.approx(
            [probe.components[2].r2, probe.components[0].r2]
        )

    def test_the_test_set_reaches_the_map(self, data):
        """In the order `Manifold` reads it, which is the order `Probe` was given."""
        X, z, (X_test, z_test) = data
        probe = Probe(X, z, bs(K), test_set=(X_test, z_test)).fit(2)
        assert probe.manifold().test_r2 == pytest.approx(
            [c.test_r2 for c in probe.components]
        )

    def test_rejects_both_selections_at_once(self, data):
        """Two different answers to one question."""
        X, z, _ = data
        probe = Probe(X, z, bs(K)).fit(2)
        with pytest.raises(ValueError, match="not both"):
            probe.manifold(n_components=1, component_idx=[0])

    def test_rejects_an_unfitted_probe(self, data):
        """An empty map is not a map, and the words are `LinearModel.predict`'s."""
        X, z, _ = data
        with pytest.raises(ValueError, match="no components fitted; call fit first"):
            Probe(X, z, bs(K)).manifold()


# ---------------------------------------------------------------------------------
# a basis the data covers only part of
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("criterion", [None, reml()])
def test_sparse_support_recovers_every_feature(criterion):
    """END-TO-END PIN for the precision of `omega`. A tensor-product spline over a
    disc inside its square leaves most corner functions without data, the regime of
    a rectangular basis over the mainland U.S. There the small penalty weights used to
    be roundoff: the first fit selected `lmbda` at a NaN and stalled, and each
    deflation constraint zeroed most of the weights, so later components overfit --
    train R^2 near 1, test R^2 near 0. Three features are planted; all three must
    come back, and out of sample."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, size=(12000, 2))
    z = pts[(pts**2).sum(1) < 0.5][:3000]
    F = np.column_stack([z[:, 0], z[:, 1], np.exp(-8 * ((z - [0.2, -0.1]) ** 2).sum(1))])
    # Unequal strengths, so the joint directions are well separated and the
    # alternation converges well inside its sweep limit.
    F = (F - F.mean(0)) / F.std(0) * [1.5, 1.0, 0.7]
    X = F @ rng.normal(size=(3, 20)) + 0.5 * rng.normal(size=(len(z), 20))

    basis = bs(k=(30, 30), limits=[(-1, 1), (-1, 1)])
    probe = Probe(X[:1500], z[:1500], basis, criterion=criterion,
                  test_set=(X[1500:], z[1500:]))
    probe.fit(3)
    for c in probe.components:
        assert c.test_r2 > 0.9
        assert c.r2 - c.test_r2 < 0.03
