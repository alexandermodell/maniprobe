"""Tests for `cv`.

`KFoldCV`'s own content is bookkeeping: which rows go where, what centring each fold's
design carries, which of the parents' settings are inherited, and how the fold scores
are combined. The joint fit it drives is `EigenFit`'s, checked against a dense
reference in `test_bilinear.py` and not re-checked here -- except once, in
`test_matches_a_dense_reference_on_a_known_fold`, which runs the whole pipeline against
an explicit hat matrix and a generalized eigenproblem so that at least one test depends
on neither module's cache.

Recovering the split without reimplementing it
---------------------------------------------

`X2` is passed through untouched, so its held-out rows are literally rows of the parent
design. The fixture puts the row number in column 0, which makes `X2_test[:, 0]` the
fold's held-out indices -- read off the object under test rather than derived by running
`permutation` and `array_split` a second time, which would only assert that the code
agrees with itself.
"""

import numpy as np
import pytest
from scipy.linalg import eigh

from maniprobe.bilinear import EigenFit
from maniprobe.cv import EigenCVFit, KFoldCV, ho, kf
from maniprobe.linear_model import LinearModel
from maniprobe.search import bayes, grid
from test_linear_model import N, P, model_lmbda, psd


N_FOLDS = 5


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def pair(rng):
    """Two designs and penalties, `X1` centred as the joint fit requires. Column 0 of
    `X2` is the row number; see the module docstring."""
    X1 = rng.normal(size=(N, P))
    X2 = rng.normal(size=(N, P - 2))
    X2[:, 0] = np.arange(N)
    return X1 - X1.mean(0), psd(rng, P), X2, psd(rng, P - 2)


def held_out(fold):
    """The fold's held-out row numbers, read off column 0 of its `X2_test`."""
    return fold[2][:, 0].astype(int)


def dense_fold_score(X1, S1, l1, X2, S2, l2, train, test, fit_intercept2):
    """Reference: one fold's held-out `R^2`, from a dense hat matrix and a generalized
    eigenproblem in beta-space.

    The formulation is `dense_joint`'s in `test_bilinear.py`, carried through to
    coefficients so it can predict at held-out rows. Profiling `lm2` out with its own
    hat matrix `H2` leaves

        min beta' A beta   s.t.   beta' B beta = n,
        A = X1'(I - H2)X1 + l1 S1,      B = X1'X1

    over the fold's training rows, so `beta1` is the smallest eigenvector of the pencil
    `(A, B)` rescaled to `||X1 beta1||^2 == n`; `beta2` is then an ordinary penalised
    solve against the fitted `f`. `B` is nonsingular here because the fixture's `X1` is
    random and stays full rank under centring -- a partition-of-unity basis would not,
    which is the case `dense_joint` profiles a null space out for.

    `l1` and `l2` arrive in the units the *model* uses, since that is what `score`
    takes; `model_lmbda` converts them to the units of `S1` and `S2` that this reference
    reads, and it must be given the fold's own design, not the parent's.
    """
    A1 = X1[train] - X1[train].mean(0)
    A2 = X2[train]
    n = len(train)
    l1 = l1 / model_lmbda(A1, S1, 1.0)
    l2 = l2 / model_lmbda(A2, S2, 1.0, fit_intercept2)

    x_mean2 = A2.mean(0) if fit_intercept2 else np.zeros(A2.shape[1])
    A2c = A2 - x_mean2
    H2 = A2c @ np.linalg.solve(A2c.T @ A2c + l2 * S2, A2c.T)
    if fit_intercept2:
        H2 = H2 + np.ones((n, n)) / n

    A = A1.T @ (np.eye(n) - H2) @ A1 + l1 * S1
    _, V = eigh((A + A.T) / 2, A1.T @ A1)
    beta1 = V[:, 0]
    f = A1 @ beta1
    beta1 = beta1 * np.sqrt(n) / np.linalg.norm(f)

    # `f` is mean-zero on the training rows because `A1` is centred, so the intercept
    # of the second model reduces to its `x_mean` term.
    f = A1 @ beta1
    beta2 = np.linalg.solve(A2c.T @ A2c + l2 * S2, A2c.T @ f)
    b2 = -x_mean2 @ beta2

    f_test = (X1[test] - X1[train].mean(0)) @ beta1
    g_test = X2[test] @ beta2 + b2
    return 1.0 - np.mean((f_test - g_test) ** 2) / f_test.var()


def build(pair, fit_intercept2=False, k=N_FOLDS, seed=0, constraints2=None):
    X1, S1, X2, S2 = pair
    return KFoldCV(
        LinearModel(X1, S1),
        LinearModel(X2, S2, fit_intercept=fit_intercept2, constraints=constraints2),
        k=k,
        seed=seed,
    )


class TestFolds:
    def test_every_row_is_held_out_exactly_once(self, pair):
        cv = build(pair)
        assert len(cv.folds) == N_FOLDS
        assert sorted(np.concatenate([held_out(f) for f in cv.folds])) == list(range(N))

    def test_training_rows_are_the_complement_of_the_held_out_ones(self, pair):
        """`lm2`'s fold design is untouched, so its rows carry the row numbers too --
        which is what lets train and test be compared as sets."""
        cv = build(pair)
        for fold in cv.folds:
            train = fold[0].lm2.X[:, 0].astype(int)
            assert set(train).isdisjoint(held_out(fold))
            assert sorted(np.concatenate([train, held_out(fold)])) == list(range(N))

    def test_lm1_is_centred_on_its_own_training_rows(self, pair):
        """Exact, not approximate: it is what makes `sum(f) == 0` and `var(f) == 1`
        hold on the training rows of every fold."""
        X1 = pair[0]
        for fold in build(pair).folds:
            assert np.abs(fold[0].lm1.X.mean(0)).max() < 1e-14 * np.abs(X1).max()

    def test_held_out_rows_carry_the_training_centring(self, pair):
        """The held-out design is shifted by the *training* rows' means. Centring it on
        itself would be fitting to held-out data, and would leave the fold's two halves
        on different origins."""
        X1 = pair[0]
        for fold in build(pair).folds:
            test = held_out(fold)
            train = np.setdiff1d(np.arange(N), test)
            want = X1[test] - X1[train].mean(0)
            assert np.allclose(fold[1], want, rtol=0, atol=1e-14 * np.abs(X1).max())
            assert np.abs(fold[1].mean(0)).max() > 1e-8  # i.e. not centred on itself

    def test_lm2_is_passed_through_untouched(self, pair):
        """No centring on this side: an intercept is what a mean shift is for. Rows are
        matched by row number rather than by position, since the training rows arrive in
        the permutation's order -- which the fit is indifferent to."""
        X2 = pair[2]
        for fold in build(pair).folds:
            train = fold[0].lm2.X[:, 0].astype(int)
            assert np.array_equal(fold[0].lm2.X, X2[train])
            assert np.array_equal(fold[2], X2[held_out(fold)])

    @pytest.mark.parametrize("fit_intercept2", [True, False])
    def test_fold_models_inherit_fit_intercept(self, pair, fit_intercept2):
        for fold in build(pair, fit_intercept2=fit_intercept2).folds:
            assert fold[0].lm1.fit_intercept is False
            assert fold[0].lm2.fit_intercept is fit_intercept2

    def test_fold_models_inherit_the_penalty(self, pair):
        """`S` is read off the parent, which is the whole reason `LinearModel` retains
        it: the basis and penalty are built once on all the data."""
        S1, S2 = pair[1], pair[3]
        for fold in build(pair).folds:
            assert fold[0].lm1.S is S1
            assert fold[0].lm2.S is S2

    def test_fold_models_inherit_the_constraint_rows(self, pair, rng):
        """Read off the parent like `S`, and imposed on each fold's own rows -- which
        is what `(X c).T @ (X beta) == 0` says, the reading the per-fold centring gets
        too. The rank drops on the folds exactly as it does on the parent, and the
        constraint holds on the rows each fold was fitted to.
        """
        c = rng.normal(size=P - 2)
        cv = build(pair, constraints2=c)
        assert len(cv.lm2.omega) == P - 3
        for fold in cv.folds:
            assert np.array_equal(fold[0].lm2.constraints, np.atleast_2d(c))
            assert len(fold[0].lm2.omega) == P - 3
        cv.score(0.5, 2.0)
        for fold in cv.folds:
            lm2 = fold[0].lm2
            g = lm2.X @ lm2.beta
            assert abs((lm2.X @ c) @ g) / abs(g).max() < 1e-10

    def test_the_score_is_the_constrained_model_s(self, pair):
        """REGRESSION PIN, for a defect rather than a number. The folds were rebuilt
        from `X`, `S` and `fit_intercept` alone, so a constrained parent was scored by
        unconstrained folds and these two numbers were *equal*: the search ranked
        `lmbda` pairs for a model with a degree of freedom the closing refit does not
        have. Measured 0.124 apart here, on a constraint chosen to bite.
        """
        free = build(pair).score(0.5, 2.0)
        bound = build(pair, constraints2=np.ones(P - 2)).score(0.5, 2.0)
        assert abs(free - bound) > 1e-3

    def test_seed_fixes_the_split_and_changes_it(self, pair):
        same = [held_out(f) for f in build(pair, seed=0).folds]
        again = [held_out(f) for f in build(pair, seed=0).folds]
        other = [held_out(f) for f in build(pair, seed=1).folds]
        assert all(np.array_equal(a, b) for a, b in zip(same, again))
        assert not all(np.array_equal(a, b) for a, b in zip(same, other))

    @pytest.mark.parametrize("k", [N_FOLDS, 0.75])
    def test_folds_are_shuffled_rather_than_contiguous(self, pair, k):
        """Contiguous folds would make every fold an extrapolation on data sorted by
        its covariate, which a smoother's routinely is. Against a contiguous block
        starting wherever this one does, rather than against `arange`: a single split
        holds out the *last* rows, so a block at 0 is not what an unshuffled one would
        look like."""
        first = np.sort(held_out(build(pair, k=k).folds[0]))
        assert not np.array_equal(first, np.arange(first[0], first[0] + len(first)))

    def test_a_fractional_k_is_one_split_at_that_training_proportion(self, pair):
        """The proportion is of the rows and not of the folds: 0.75 of 40 rows are
        trained on and the remaining 10 scored, in one split rather than four."""
        cv = build(pair, k=0.75)
        assert len(cv.folds) == 1
        train = cv.folds[0][0].lm2.X[:, 0].astype(int)
        assert (len(train), len(held_out(cv.folds[0]))) == (30, 10)
        assert sorted(np.concatenate([train, held_out(cv.folds[0])])) == list(range(N))

    def test_requires_matching_row_counts(self, pair):
        X1, S1, X2, S2 = pair
        with pytest.raises(ValueError, match="same number of rows"):
            KFoldCV(LinearModel(X1, S1), LinearModel(X2[:10], S2))

    @pytest.mark.parametrize("k", [1, 1.0, 0, -0.5, 2.5])
    def test_rejects_a_k_that_is_neither(self, pair, k):
        """Both silent otherwise: `array_split` reads 2.5 as 2, and `k = 1.0` would
        train on everything and score on nothing."""
        with pytest.raises(ValueError, match="k must be"):
            build(pair, k=k)


class TestScore:
    @pytest.mark.parametrize("fit_intercept2", [True, False])
    @pytest.mark.parametrize("l1, l2", [(1e-3, 1e-3), (0.5, 2.0)])
    def test_matches_a_dense_reference_on_a_known_fold(
        self, pair, l1, l2, fit_intercept2
    ):
        """INDEPENDENT REFERENCE. The one test here that depends on neither `cv`'s
        bookkeeping nor `bilinear`'s cache: the split is read off the object, and each
        fold is then refitted from a dense hat matrix and a generalized eigenproblem."""
        X1, S1, X2, S2 = pair
        cv = build(pair, fit_intercept2=fit_intercept2)
        cv.score(l1, l2)

        for fold in cv.folds:
            test = held_out(fold)
            train = np.setdiff1d(np.arange(N), test)
            want = dense_fold_score(
                X1, S1, l1, X2, S2, l2, train, test, fit_intercept2
            )
            assert np.isclose(fold[0].r2(fold[1], fold[2]), want, rtol=1e-8)

    def test_is_the_mean_of_the_fold_scores(self, pair):
        cv = build(pair)
        total = cv.score(0.5, 2.0)
        each = [f[0].r2(f[1], f[2]) for f in cv.folds]
        assert np.isclose(total, sum(each) / N_FOLDS)
        assert not np.allclose(each, each[0])  # the folds genuinely differ

    def test_is_repeatable(self, pair):
        """A search strategy is entitled to assume it: `EigenFit.fit` is a
        function of the two `lmbda`s and a cache fixed at construction."""
        cv = build(pair)
        assert cv.score(0.5, 2.0) == cv.score(0.5, 2.0)
        cv.score(1e-4, 1e-4)
        assert cv.score(0.5, 2.0) == cv.score(0.5, 2.0)

    def test_constraint_is_exact_on_every_fold(self, pair):
        """What the per-fold centring buys, and the reason it is not merely tidiness:
        `sum(f) == 0` and `||f||^2 / n == 1` hold to machine precision on the rows each
        fold was fitted to."""
        cv = build(pair)
        cv.score(0.5, 2.0)
        for fold in cv.folds:
            f = fold[0].lm1.predict()
            assert abs(f.mean()) < 1e-12
            assert abs(f.var() - 1.0) < 1e-12

    def test_held_out_variance_is_not_one(self, pair):
        """Which is why `r2` divides by it rather than substituting 1: the scale
        constraint is imposed on the training rows and says nothing about the others."""
        cv = build(pair)
        cv.score(1e-4, 1e-4)
        spreads = [fold[0].lm1.predict(fold[1]).var() for fold in cv.folds]
        assert not np.allclose(spreads, 1.0, atol=1e-3)

    def test_fold_scores_average_to_score(self, pair):
        cv = build(pair)
        s = cv.fold_scores(0.5, 2.0)
        assert len(s) == N_FOLDS
        assert np.isclose(s.mean(), cv.score(0.5, 2.0))

    def test_a_fractional_k_scores_on_its_one_split(self, pair):
        """Exactly, not closely: with one fold there is no averaging left for `score` to
        do, and the held-out `R^2` of that split is the whole objective."""
        cv = build(pair, k=0.75)
        s = cv.fold_scores(0.5, 2.0)
        assert len(s) == 1
        assert s[0] == cv.score(0.5, 2.0)

    def test_edf_is_the_parents_summed(self, pair):
        cv = build(pair)
        assert np.isclose(cv.edf(0.5, 2.0), cv.lm1.edf(0.5) + cv.lm2.edf(2.0))

    def test_lmbda_means_the_same_in_a_fold_as_in_the_parent(self, pair):
        """REGRESSION PIN. The docstring's claim that a winning `lmbda` transfers to a
        full-data refit: dropping rows scales the whole spectrum by roughly
        `n_train / n`, and the unit-median normalisation removes a uniform factor. What
        survives is the sampling fluctuation, so this pins a tolerance rather than an
        identity: measured 0.085 EDF at `N = 40` with folds of 32 rows, pinned at 0.24,
        which is 4% of the model's 6 coordinates. A failure means the normalisation
        changed, not that the fit is wrong -- and the headroom is under 3x, so at a much
        smaller `N` this would need re-measuring rather than loosening.
        """
        X1, S1 = pair[0], pair[1]
        parent = LinearModel(X1, S1)
        for fold in build(pair).folds:
            for lmbda in [1e-2, 1.0, 1e2]:
                assert abs(fold[0].lm1.edf(lmbda) - parent.edf(lmbda)) < 0.04 * P


# Wider than `P`, and deliberately so: `signal_pair` needs a regime where smoothing
# helps. At `P = 6` a fold has 27 training rows against 6 columns, the unpenalised fit
# is fine, and the best `lmbda` is whichever end of the box is smallest. At 25 columns
# the fold is nearly saturated and the optimum moves inside.
P1_SIGNAL, P2_SIGNAL = 25, 20


@pytest.fixture
def signal_pair(rng):
    """As `pair`, but with `X2` carrying the direction `X1` spans, so a cross-validated
    score has something to find. `pair`'s two designs are independent and score -0.18 at
    their best, which is all the exact comparisons above need and useless for anything
    asserting *where* the maximum is."""
    X1 = rng.normal(size=(N, P1_SIGNAL))
    X1 = X1 - X1.mean(0)
    S1, S2 = psd(rng, P1_SIGNAL), psd(rng, P2_SIGNAL)
    f = X1 @ rng.normal(size=P1_SIGNAL)
    return X1, S1, rng.normal(size=(N, P2_SIGNAL)) + f[:, None] * 0.5, S2


def build_fit(pair, k=N_FOLDS, seed=0, edf_tol=0.1, constraints2=None, **kwargs):
    """`**kwargs` is `method` -- a `grid()` or `bayes()` carrying its own budget --
    which lives on the constructor with everything else that decides what `fit()`
    costs."""
    X1, S1, X2, S2 = pair
    return EigenCVFit(
        LinearModel(X1, S1),
        LinearModel(X2, S2, constraints=constraints2),
        k=k,
        seed=seed,
        edf_tol=edf_tol,
        **kwargs,
    )


class TestEigenCVFit:
    def test_box_is_the_two_grids_in_logs(self, pair):
        """INDEPENDENT. Against `lmbda_grid`'s ends read off the parents directly. The
        box is the one place a `lmbda` changes coordinates, so a swapped `exp`/`log` or
        a transposed pair of axes has to be visible somewhere.

        Built by `fit` and None before it, which is the same statement from the other
        side: the box is derived from `omega`, not chosen alongside `k` and the budget.
        """
        m = build_fit(pair, k=2, edf_tol=0.05, method=grid(grid_size=3))
        assert m.bounds is None

        m.fit()
        for row, lm in zip(m.bounds, (m.lm1, m.lm2)):
            assert np.allclose(np.exp(row), lm.lmbda_grid(2, 0.05))
        assert (m.bounds[:, 0] < m.bounds[:, 1]).all()

    def test_the_box_is_rebuilt_after_a_constraint(self, pair):
        """INDEPENDENT, and why the box belongs to `fit`. A constraint renormalises
        `omega`, so the ends `lmbda_grid` derives from it move -- on the constrained
        axis only, `lm2` being untouched -- and a box formed once at construction would
        search the wrong range for every fit after the first.

        Against the parents' own grids afterwards, so what is pinned is that the box
        follows `omega` rather than merely that it changed.
        """
        m = build_fit(pair, k=2, method=grid(grid_size=3)).fit()
        before = m.bounds.copy()
        m.add_constraints(m.lm1.beta).fit()

        assert np.abs(m.bounds[0] - before[0]).max() > 0.1
        assert np.array_equal(m.bounds[1], before[1])
        for row, lm in zip(m.bounds, (m.lm1, m.lm2)):
            assert np.allclose(np.exp(row), lm.lmbda_grid(2, 0.1))

    def test_report_reaches_the_searcher(self, pair):
        """One callback down two layers: the searcher calls it after every evaluation,
        and reassigning it between fits is how a caller labels a sequence of them."""
        calls = []
        m = build_fit(
            pair, k=2, method=grid(grid_size=3),
            report=lambda i, total, metrics: calls.append((i, total)),
        ).fit()
        assert [i for i, _ in calls] == list(range(1, 10))
        assert {total for _, total in calls} == {9}

        m.report = None
        m.fit()
        assert len(calls) == 9

    def test_selects_the_argmax_of_what_it_evaluated(self, pair):
        """INDEPENDENT of `search.py`. The scores are recomputed here by calling the
        objective again at every evaluated point, and the winner is found by a plain
        loop; `fit` must land on the same pair."""
        m = build_fit(pair, k=3, method=grid(grid_size=4)).fit()

        best, best_at = -np.inf, None
        for v in m.search.points:
            s = m.cv.score(*np.exp(v))
            if s > best:
                best, best_at = s, v
        assert np.allclose([m.lm1.lmbda, m.lm2.lmbda], np.exp(best_at))
        assert np.isclose(best, m.search.scores.max())

    def test_refit_is_the_full_data_fit_at_the_chosen_pair(self, pair):
        """INDEPENDENT of how the pair was chosen. A fresh `EigenFit` over fresh models,
        fitted at the winning `lmbda`s, must reproduce the fit `EigenCVFit` is left
        holding -- exactly, since both are one `eigh` on the same matrix."""
        X1, S1, X2, S2 = pair
        m = build_fit(pair, k=3, method=grid(grid_size=3)).fit()

        want = EigenFit(LinearModel(X1, S1), LinearModel(X2, S2))
        want.fit(m.lm1.lmbda, m.lm2.lmbda)
        assert np.allclose(m.lm1.beta, want.lm1.beta, rtol=1e-12)
        assert np.allclose(m.lm2.beta, want.lm2.beta, rtol=1e-12)
        assert np.isclose(m.r2(), want.r2(), rtol=1e-12)

    def test_constraint_holds_on_all_the_rows(self, pair):
        """The closing refit is over the whole design, not a fold's: `sum(f) == 0` and
        `||f||^2 / n == 1` hold on all `N` rows, which they would not if the object were
        left holding a training-subset fit."""
        m = build_fit(pair, k=3, method=grid(grid_size=3)).fit()
        f = m.lm1.predict()
        assert len(f) == N
        assert abs(f.mean()) < 1e-12
        assert abs(f.var() - 1.0) < 1e-12

    def test_runs_on_constrained_parents(self, pair):
        """End to end on a constrained `lm2`: the box comes from the parents' grids, the
        folds are constrained the same way, and the closing refit is over all the rows.
        All three describe one model now, where the folds used to describe another."""
        c = np.ones(P - 2)
        m = build_fit(pair, k=3, constraints2=c, method=grid(grid_size=3)).fit()
        X2, lm2 = pair[2], m.lm2
        g = X2 @ lm2.beta
        assert len(lm2.omega) == P - 3
        assert abs((X2 @ c) @ g) / abs(g).max() < 1e-10
        lo, hi = np.exp(m.bounds[1])
        assert lo <= lm2.lmbda <= hi

    def test_shares_the_parents_with_its_folds(self, pair):
        """`KFoldCV` keeps the parents for `edf` and this fits them; one shared
        reference, not a second copy that could drift."""
        m = build_fit(pair)
        assert m.lm1 is m.cv.lm1 and m.lm2 is m.cv.lm2

    @pytest.mark.parametrize(
        "method, n", [(grid(grid_size=4), 16), (bayes(n_eval=9), 9)]
    )
    def test_budget_reaches_the_searcher(self, pair, method, n):
        """The budget lives on the specification rather than on this constructor, so
        what is checked is that it reaches the searcher through both of them."""
        m = build_fit(pair, k=2, method=method).fit()
        assert m.search is not None and len(m.search.points) == n
        assert len(m.search.scores) == n

    def test_a_named_search_is_resolved_at_construction(self, pair):
        """The shorthand, and where it is resolved: `method` is the object from then on,
        so a name cannot survive as far as `fit`, which is after the folds are built.
        Against the helpers rather than against their default budgets, which are
        `search.as_spec`'s business and are pinned there."""
        assert build_fit(pair, k=2, method="grid").method.grid_size == grid().grid_size
        assert build_fit(pair, k=2, method="bayes").method.n_eval == bayes().n_eval

    def test_rejects_an_unrecognised_method(self, pair):
        """At construction, so the fold pairs are never built. `fit` takes no arguments
        and has nothing left to reject."""
        with pytest.raises(ValueError, match="method must be"):
            build_fit(pair, method="grd")

    def test_search_is_none_before_fit(self, pair):
        assert build_fit(pair).search is None

    def test_seed_fixes_the_run_and_changes_it(self, pair):
        """One seed covers both the fold split and the Sobol' sequence, so it has to
        reproduce a whole `bayes` run and a different one has to move it."""
        a = build_fit(pair, k=3, seed=0, method=bayes(n_eval=9)).fit()
        b = build_fit(pair, k=3, seed=0, method=bayes(n_eval=9)).fit()
        c = build_fit(pair, k=3, seed=1, method=bayes(n_eval=9)).fit()
        assert np.allclose(a.search.points, b.search.points)
        assert a.lm1.lmbda == b.lm1.lmbda and a.lm2.lmbda == b.lm2.lmbda
        assert not np.allclose(a.search.points, c.search.points)

    def test_grid_needs_no_seed(self, pair):
        """It is deterministic, which is half of what it is kept for."""
        a = build_fit(pair, k=2, seed=None, method=grid(grid_size=3)).fit()
        b = build_fit(pair, k=2, seed=None, method=grid(grid_size=3)).fit()
        assert np.allclose(a.search.points, b.search.points)

    def test_chosen_pair(self, signal_pair):
        """REGRESSION PIN, not an independent check: nothing here says these two numbers
        are right, only that they have not moved. A change means the box, the objective
        or the search moved, and someone has to say which.

        On `signal_pair` rather than `pair`, and that is the whole point of that
        fixture: the standard pair is two independent designs whose best cross-validated
        score is -0.18, so its maximiser is a wander over noise and pinning it would pin
        the noise. Here the best score is 0.49 and the winners sit 0.25 and 0.59 of the
        way into their axes -- interior on both, so a change that moves the optimum
        moves this, rather than being absorbed by a box end that is an answer in its own
        right."""
        m = build_fit(signal_pair, k=3, seed=0, method=bayes(n_eval=12)).fit()
        assert np.allclose([m.lm1.lmbda, m.lm2.lmbda], [0.05067417, 40.1963027])
        interior = (m.bounds[:, 0] + 0.5 < np.log([m.lm1.lmbda, m.lm2.lmbda])) & (
            np.log([m.lm1.lmbda, m.lm2.lmbda]) < m.bounds[:, 1] - 0.5
        )
        assert interior.all()


class TestAddConstraints:
    """A constraint imposed in place, which is how a sequence of deflated fits reuses
    one object instead of rebuilding the folds for every one of them."""

    def test_folds_and_parents_move_together(self, pair):
        """A fold left unconstrained cross-validates a model carrying a degree of
        freedom the closing refit does not have -- the defect `constraints` was made a
        specification field for. The rank drop is the visible half of it, on the parent
        and on every fold alike."""
        m = build_fit(pair, k=3, method=grid(grid_size=3)).fit()
        ranks = [len(m.lm1.omega)] + [len(bl.lm1.omega) for bl, *_ in m.cv.folds]
        c = m.lm1.beta.copy()
        assert m.add_constraints(c) is m

        assert [len(m.lm1.omega)] + [len(bl.lm1.omega) for bl, *_ in m.cv.folds] == [
            r - 1 for r in ranks
        ]
        for lm in [m.lm1] + [bl.lm1 for bl, *_ in m.cv.folds]:
            assert np.array_equal(lm.constraints, np.atleast_2d(c))
        assert m.lm2.constraints is None

    def test_fold_scores_match_folds_built_constrained(self, pair):
        """INDEPENDENT, and the test that licenses not rebuilding the folds. After two
        components' worth of constraints, `fold_scores` at a *fixed* pair must agree
        with the same call on a fresh `EigenCVFit` whose parents were constructed
        carrying the same rows -- the fold models a rebuild would have produced, since
        `KFoldCV` copies `constraints` off its parents verbatim.

        To roundoff rather than exactly: the reused folds reach their constraint through
        `add_constraints` and the fresh ones through `LinearModel`'s constructor, and
        `test_linear_model.test_the_two_routes_agree` is what holds those two routes to
        each other. Measured 3e-15 relative here.

        At a fixed pair, and deliberately not at the searched outcome: a
        cross-validation surface is routinely flat, so an argmax can turn a last-digit
        difference into a visibly different `lmbda`, and the test would then be
        measuring the search rather than the folds.
        """
        X1, S1, X2, S2 = pair
        m = build_fit(pair, k=3, seed=0, method=grid(grid_size=3))
        rows = []
        for _ in range(2):
            m.fit()
            rows.append(m.lm1.beta.copy())
            m.add_constraints(m.lm1.beta)

        fresh = EigenCVFit(
            LinearModel(X1, S1, constraints=np.array(rows)),
            LinearModel(X2, S2),
            k=3,
            seed=0,
            method=grid(),
        )
        got = fresh.cv.fold_scores(0.5, 2.0)
        assert np.allclose(m.cv.fold_scores(0.5, 2.0), got, rtol=1e-11, atol=0.0)
        # And the constraint is doing something: unconstrained folds score elsewhere.
        free = build_fit(pair, k=3, seed=0).cv.fold_scores(0.5, 2.0)
        assert not np.allclose(got, free)


class TestCriterion:
    """`kf`, `ho` and `CVCriterion` -- the cross-validation stated before there are two
    models to run it over."""

    def test_kf_builds_a_fold_average(self, pair):
        X1, S1, X2, S2 = pair
        m = kf(k=3, method=grid(grid_size=3), seed=4).fitter(
            LinearModel(X1, S1), LinearModel(X2, S2), test=None
        )
        assert isinstance(m, EigenCVFit) and len(m.cv.folds) == 3
        assert (m.method.grid_size, m.seed) == (3, 4)

    def test_ho_builds_one_split_at_that_proportion(self, pair):
        """One class implements both, and this is the distinction the two helpers exist
        to make: a mean over folds against a single held-out estimate."""
        X1, S1, X2, S2 = pair
        m = ho(train=0.75, seed=0).fitter(
            LinearModel(X1, S1), LinearModel(X2, S2), test=None
        )
        assert len(m.cv.folds) == 1
        assert len(m.cv.folds[0][0].lm1.X) == 30

    def test_the_test_set_is_accepted_and_ignored(self, pair):
        """A cross-validated fit has a held-out score by construction and wants no
        second one. The signature matches `als.ALSCriterion.fitter` all the same, since
        that is what lets a caller dispatch on nothing."""
        X1, S1, X2, S2 = pair
        m = kf(k=2).fitter(
            LinearModel(X1, S1), LinearModel(X2, S2), test=(X1[:5], X2[:5])
        )
        assert not hasattr(m, "test")

    @pytest.mark.parametrize("k", [0.8, 1, 1.0, 2.5])
    def test_kf_refuses_anything_but_a_fold_count(self, k):
        """`KFoldCV` accepts the union of the two forms, so `kf(k=0.8)` would otherwise
        be a hold-out split without saying so."""
        with pytest.raises(ValueError, match="k must be an integer"):
            kf(k=k)

    @pytest.mark.parametrize("train", [5, 1.0, 0, -0.5])
    def test_ho_refuses_anything_but_a_proportion(self, train):
        with pytest.raises(ValueError, match="train must be"):
            ho(train=train)
