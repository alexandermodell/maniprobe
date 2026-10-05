"""Tests for `search`.

The surrogate is checked against references from outside this file: the Matern 5/2
correlation against the general Matern formula evaluated through a Bessel function, the
log marginal likelihood against `scipy.stats.multivariate_normal`, and the posterior
against a densely formed and densely solved one. The length-scale fit is checked
statistically rather than algebraically -- data drawn from the prior at a known scale,
and the scale recovered -- since there is no closed form for the maximiser to compare
against.

What each reference is independent *of* is worth naming, because none is independent of
everything. `dense_posterior` calls `_matern52`, so it checks the Cholesky path, the
contraction that forms the variance and the length-scale selection, and not the kernel;
the kernel has its own reference. Nothing here re-derives expected improvement: it is
checked through the property that makes it usable, invariance to an affine rescaling of
the objective, which a recomputed formula would satisfy by construction.

The searches themselves run on a surface whose maximiser is known in closed form -- an
anisotropic bump on a box wide enough that most of it is flat, which is the shape a
cross-validation score has -- and on one real `KFoldCV`. `BayesSearch` is required to
beat a 400-point grid using 40 evaluations, which is the whole claim the module makes
for it.

Mutation checks
---------------

Each of these was introduced deliberately and the suite re-run; the test named first is
the sharpest one that failed. Recorded because a test that survives the bug it was
written for is not coverage.

- `argmin` for `argmax` in the acquisition: all six seeds of
  `test_bayes_finds_the_analytic_maximiser`, and
  `test_bayes_beats_a_grid_of_ten_times_the_budget`.
- The scores left unstandardised before the surrogate is fitted:
  `test_bayes_is_invariant_to_rescaling_the_objective`.
- `mu * norm.pdf(z)` for `sd * norm.pdf(z)`, an exploration term with the wrong units:
  `test_expected_improvement_is_the_expectation_it_claims_to_be`.
- `d**2 / 3` dropped from the correlation, leaving Matern 3/2:
  `test_matern52_matches_the_bessel_form`.
- The posterior variance contracted along the wrong axis (`.sum(0)` for `.sum(1)`,
  which broadcasts rather than raising): `test_posterior_matches_a_dense_solve`.
- The length-scale fixed at `ELL[0]` instead of chosen:
  `test_posterior_uses_the_length_scale_it_fitted`.
- The grid built with `indexing="xy"`: `test_grid_is_the_box_in_reading_order`.
- The standard deviation floored at `abs(var)` instead of at `JITTER`:
  `test_posterior_floors_the_standard_deviation_at_duplicate_points`.

Two of those are the reason the acquisition and the posterior are tested as units at
all. A mis-scaled exploration term and a length-scale frozen at the shortest offered
both survive *every* search-level test here: the searches recover from either by
exploring more, reaching the same answer more slowly, and 40 evaluations is enough
slack to hide it. Only the regression pin at the end and the two unit tests notice.
"""

import warnings

import numpy as np
import pytest
from scipy.special import gamma, kv
from scipy.stats import multivariate_normal

from maniprobe.cv import KFoldCV
from maniprobe.linear_model import LinearModel
from maniprobe.search import (
    ELL,
    JITTER,
    N_INIT,
    BayesSearch,
    GridSearch,
    _expected_improvement,
    _log_marginal,
    _matern52,
    _posterior,
    as_spec,
    bayes,
    grid,
)
from test_linear_model import N, P, psd

# An anisotropic bump: the maximiser is exact, the two axes have different widths, and
# the box is wide enough that most of it is flat to floating point -- which is what a
# cross-validation surface looks like and what makes the search's job non-trivial.
PEAK = np.array([1.3, -2.1])
BOX = np.array([[-5.0, 5.0], [-5.0, 5.0]])


def bump(v):
    return np.exp(-(((v[0] - PEAK[0]) / 2.0) ** 2) - ((v[1] - PEAK[1]) / 0.7) ** 2)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def unit_points(rng):
    """Sixty points on the unit square, where the surrogate does its work."""
    return rng.random((60, 2))


@pytest.fixture
def cv(rng):
    """A pair with a real bilinear signal: `X2` is built from four of `X1`'s columns,
    so some direction in `span(X1)` is genuinely predictable and the score surface has
    an optimum rather than a ridge of noise."""
    X1 = rng.normal(size=(N, P))
    X2 = X1[:, :4] @ rng.normal(size=(4, P - 2)) + 0.5 * rng.normal(size=(N, P - 2))
    lm1 = LinearModel(X1 - X1.mean(0), psd(rng, P), fit_intercept=False)
    lm2 = LinearModel(X2, psd(rng, P - 2), fit_intercept=True)
    return KFoldCV(lm1, lm2, k=5, seed=1)


def dense_posterior(U, y, V, ell):
    """Reference: the textbook GP posterior, formed whole and solved with `solve`.

    Independent of the Cholesky factorisation, of `cho_solve`, and of the contraction
    `_posterior` uses to get one variance per row rather than a full covariance. Not
    independent of `_matern52`, which has its own reference.
    """
    K = _matern52(U, U, ell) + JITTER * np.eye(len(U))
    Ks = _matern52(V, U, ell)
    mu = Ks @ np.linalg.solve(K, y)
    var = np.array([1.0 - k @ np.linalg.solve(K, k) for k in Ks])
    return mu, np.sqrt(np.maximum(var, JITTER))


# -- the kernel ---------------------------------------------------------------------


def test_matern52_matches_the_bessel_form(rng):
    """Independent reference: the general Matern kernel,

        2**(1 - nu) / Gamma(nu) * d**nu * K_nu(d),    d = sqrt(2 nu) r / ell

    at `nu = 5/2`, through `scipy.special.kv`. The closed form in `_matern52` follows
    from that only by a half-integer identity, so agreement is a check on the identity
    and not a restatement of it. A failure means the kernel is not Matern 5/2 -- most
    likely that it is Matern 3/2, which differs by one term.
    """
    A, B = rng.normal(size=(5, 2)), rng.normal(size=(4, 2))
    ell, nu = 0.37, 2.5
    d = np.sqrt(2 * nu) * np.linalg.norm(A[:, None] - B[None], axis=-1) / ell
    reference = 2 ** (1 - nu) / gamma(nu) * d**nu * kv(nu, d)
    assert _matern52(A, B, ell) == pytest.approx(reference, abs=1e-14)


def test_matern52_is_a_correlation(rng):
    """Unit diagonal, exactly. `_posterior` writes the prior variance as the literal
    `1.0`, so this is the assumption that expression rests on rather than a property of
    the kernel worth knowing on its own."""
    A = rng.normal(size=(6, 2))
    assert np.diag(_matern52(A, A, 0.4)) == pytest.approx(1.0, abs=0.0)


# -- the marginal likelihood --------------------------------------------------------


def test_log_marginal_is_a_log_density(unit_points, rng):
    """Independent reference: `scipy.stats.multivariate_normal.logpdf` of the same
    correlation matrix. This is why the `2 pi` term is kept even though it is constant
    in `ell` and changes no choice made from it -- without it there is nothing outside
    this file that knows what the number should be.

    The tolerance is `eps * cond(K)` rather than a fixed number, which is the most two
    solvers of the same system can be asked to agree to -- scipy goes through an
    eigendecomposition, this through a Cholesky. It has to scale, because `cond(K)`
    runs from `3.6e2` at `ell = 0.1` to `8.2e8` at `ell = 1.5`: a long length-scale
    makes the observations nearly collinear. Measured agreement over that range is
    `8e-15` to `7.5e-9`, inside the bound by one to two orders of magnitude throughout,
    so what is being checked is the expression and not the arithmetic.
    """
    y = rng.normal(size=len(unit_points))
    for ell in [0.1, 0.4, 1.5]:
        K = _matern52(unit_points, unit_points, ell) + JITTER * np.eye(len(y))
        reference = multivariate_normal.logpdf(y, np.zeros(len(y)), K)
        tol = np.finfo(float).eps * np.linalg.cond(K)
        assert _log_marginal(unit_points, y, ell) == pytest.approx(reference, rel=tol)


def test_log_marginal_recovers_the_scale_it_was_generated_at(unit_points, rng):
    """Statistical reference: draw `y` from the prior at a known length-scale and ask
    the likelihood for it back. The data comes from the model rather than from the
    estimator, so this checks the whole quantity -- the quadratic form, the log
    determinant and their relative weight -- in a way `logpdf` at a fixed `ell` cannot.

    Six percent at sixty points; the likelihood is broad in `log(ell)`, which is the
    same fact that lets `ELL` be a coarse grid.
    """
    grid = np.geomspace(0.05, 2.0, 40)
    for true_ell in [0.15, 0.3, 0.6, 1.0]:
        K = _matern52(unit_points, unit_points, true_ell) + 1e-10 * np.eye(60)
        y = np.linalg.cholesky(K) @ rng.normal(size=60)
        fitted = grid[np.argmax([_log_marginal(unit_points, y, e) for e in grid])]
        assert fitted == pytest.approx(true_ell, rel=0.06)


# -- the posterior ------------------------------------------------------------------


def test_posterior_matches_a_dense_solve(unit_points, rng, monkeypatch):
    """`ELL` is pinned to one value so the reference has no choice to reproduce; the
    choosing is checked by `test_log_marginal_recovers_the_scale_it_was_generated_at`
    and by the searches themselves."""
    monkeypatch.setattr("maniprobe.search.ELL", np.array([0.4]))
    y = rng.normal(size=len(unit_points))
    V = rng.random((11, 2))
    mu, sd = _posterior(unit_points, y, V)
    mu_ref, sd_ref = dense_posterior(unit_points, y, V, 0.4)
    assert mu == pytest.approx(mu_ref, abs=1e-8)
    assert sd == pytest.approx(sd_ref, abs=1e-8)


def test_posterior_interpolates(unit_points, rng):
    """With a jitter and no noise term the GP passes through its data, which is the
    claim `search.py` makes for holding the diagonal at a conditioning level rather
    than fitting it. Asserted at `1e-6` rather than at a loose tolerance because the
    only thing standing between the posterior mean and `y` is the jitter."""
    y = rng.normal(size=len(unit_points))
    mu, sd = _posterior(unit_points, y, unit_points)
    assert mu == pytest.approx(y, abs=1e-6)
    assert sd == pytest.approx(np.sqrt(JITTER), rel=1e-6)


def test_posterior_floors_the_standard_deviation_at_duplicate_points(rng):
    """Repeated points are what the acquisition drives towards, and they are where the
    posterior variance collapses: with five rows duplicated the raw variance falls to
    `3.3e-9`, a third of the jitter, and roundoff can take it below zero from there.
    The floor is what stops `gain / sd` dividing by zero in the acquisition, so it is
    checked where it binds rather than only where it does not."""
    U = rng.random((30, 2))
    U = np.vstack([U, U[:5]])
    _, sd = _posterior(U, rng.normal(size=len(U)), U)
    assert sd == pytest.approx(np.sqrt(JITTER), rel=1e-12)


def test_posterior_uses_the_length_scale_it_fitted(rng):
    """Data drawn from the prior at `ell = 0.8`, and the posterior compared against a
    dense one at two of `ELL`'s values: the grid point nearest the truth, and the
    shortest. It agrees with the first to `1e-12` and differs from the second by 1.2,
    which is the whole spread of the data -- so this pins that the length-scale reaches
    `_posterior` and not merely that the likelihood could have found it.

    `test_log_marginal_recovers_the_scale_it_was_generated_at` checks that the
    likelihood knows the answer; this checks that `_posterior` asks it.
    """
    U, V = rng.random((40, 2)), rng.random((15, 2))
    K = _matern52(U, U, 0.8) + 1e-10 * np.eye(len(U))
    y = np.linalg.cholesky(K) @ rng.normal(size=len(U))
    mu, _ = _posterior(U, y, V)
    nearest, _ = dense_posterior(U, y, V, ELL[np.argmin(np.abs(ELL - 0.8))])
    shortest, _ = dense_posterior(U, y, V, ELL[0])
    assert mu == pytest.approx(nearest, abs=1e-10)
    assert np.abs(mu - shortest).max() > 1.0


def test_posterior_returns_to_the_prior_far_from_the_data(rng):
    """Far outside the box the correlation with every observation underflows, so the
    mean is the prior mean and the standard deviation the prior one. The acquisition
    depends on this: it is what makes unexplored regions attractive."""
    U, y = rng.random((12, 2)), rng.normal(size=12)
    mu, sd = _posterior(U, y, np.array([[40.0, 40.0]]))
    assert mu[0] == pytest.approx(0.0, abs=1e-12)
    assert sd[0] == pytest.approx(1.0, abs=1e-12)


# -- the acquisition ----------------------------------------------------------------


def test_expected_improvement_is_the_expectation_it_claims_to_be(rng):
    """Independent reference: `E[max(Y - best, 0)]` by Monte Carlo, a million draws per
    case. This is the definition rather than a second copy of the closed form, so it
    catches a term with the wrong factor in it -- which nothing else here does, since
    the searches recover from a mis-scaled acquisition by exploring instead.

    Four decimal places is what a million draws supports; the standard error of the
    mean is around `1e-3 * sd`.
    """
    best = 0.4
    mu = np.array([-0.5, 0.0, 0.4, 0.9, 3.0])
    sd = np.array([1e-3, 0.2, 1.0, 0.5, 0.1])
    draws = mu + sd * rng.normal(size=(1_000_000, len(mu)))
    reference = np.maximum(draws - best, 0.0).mean(0)
    assert _expected_improvement(mu, sd, best) == pytest.approx(reference, abs=2e-3)


def test_expected_improvement_is_invariant_to_an_affine_map(rng):
    """`EI` under `c * y + d` is `c` times `EI` under `y`, so its argmax is unchanged.
    This is the property `search` relies on when it standardises the scores, stated on
    the acquisition itself rather than inferred from two whole runs agreeing."""
    mu, sd, best = rng.normal(size=7), rng.random(7) + 0.1, 0.3
    c, d = 3.5, -2.0
    assert _expected_improvement(c * mu + d, c * sd, c * best + d) == pytest.approx(
        c * _expected_improvement(mu, sd, best)
    )


# -- GridSearch ---------------------------------------------------------------------


def test_grid_is_the_box_in_reading_order():
    """Independent of `meshgrid`, `stack` and the reshape: the same points built by two
    nested loops. The order is what makes `scores.reshape(g, g)` the surface with the
    first dimension of `bounds` on the first axis, which the docstring promises."""
    g = 7
    s = GridSearch(bump, BOX).search(grid_size=g)
    axes = [np.linspace(lo, hi, g) for lo, hi in BOX]
    expected = np.array([[x, y] for x in axes[0] for y in axes[1]])
    assert s.points == pytest.approx(expected)
    assert s.scores.reshape(g, g)[3, 5] == bump(np.array([axes[0][3], axes[1][5]]))


def test_grid_reaches_the_corners():
    """The ends of `bounds` are evaluated rather than approached, which is what makes
    an optimum at an endpoint an answer: `lmbda_grid`'s ends are chosen to be the
    extremes worth looking at."""
    s = GridSearch(bump, BOX).search(grid_size=5)
    assert s.points.min(0) == pytest.approx(BOX[:, 0])
    assert s.points.max(0) == pytest.approx(BOX[:, 1])


def test_grid_finds_the_maximiser_to_its_own_spacing():
    """A grid cannot do better than a grid: the reported point is the grid point
    nearest the maximiser, so the error is bounded by half a spacing on each axis and
    nothing here should ever assert more."""
    g = 21
    s = GridSearch(bump, BOX).search(grid_size=g)
    spacing = (BOX[:, 1] - BOX[:, 0]) / (g - 1)
    assert (np.abs(s.points[s.scores.argmax()] - PEAK) <= spacing / 2).all()


# -- BayesSearch --------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_bayes_finds_the_analytic_maximiser(seed):
    """Within a fiftieth of the box on both axes, from six different Sobol' scramblings
    of the same 40 evaluations. The bump is narrow relative to the box -- its half-width
    is 0.7 of 10 on the tighter axis -- so this is a search that has to localise, not
    one that can succeed by covering."""
    s = BayesSearch(bump, BOX, seed=seed).search(n_eval=40)
    assert np.abs(s.points[s.scores.argmax()] - PEAK).max() < 0.2
    assert s.scores.max() > 0.99


def test_bayes_beats_a_grid_of_ten_times_the_budget():
    """The claim the module exists to make. 40 evaluations against 400, on a surface
    that is flat almost everywhere, and the acquisition has to be pointed the right way
    for it to hold: maximising the wrong end of expected improvement fails here."""
    b = BayesSearch(bump, BOX, seed=0).search(n_eval=40)
    g = GridSearch(bump, BOX).search(grid_size=20)
    assert b.scores.max() > g.scores.max()


def test_bayes_is_invariant_to_rescaling_the_objective():
    """`3 * f + 10` is searched at exactly the same points as `f`, bit for bit.

    Standardising the scores before fitting the surrogate is what buys this, and
    expected improvement is invariant to a positive affine map for the same reason: it
    scales with the spread and the shift cancels. Without it the length-scale fit and
    the acquisition would both depend on the units the objective happens to be in, and
    a score in `R^2` would search differently from the same score in percent.
    """
    a = BayesSearch(bump, BOX, seed=0).search(n_eval=30)
    b = BayesSearch(lambda v: 3.0 * bump(v) + 10.0, BOX, seed=0).search(n_eval=30)
    assert b.points == pytest.approx(a.points, abs=0.0)
    assert b.scores == pytest.approx(3.0 * a.scores + 10.0)


def test_bayes_stays_inside_the_box():
    """Every evaluation is in `bounds`, including the ones the acquisition chose. The
    box is not a preference: outside it `lmbda_grid`'s guarantee about `edf_tol` does
    not hold and the caller's `exp` can overflow."""
    box = np.array([[-2.0, 3.0], [10.0, 11.5]])
    s = BayesSearch(lambda v: -np.sum(v**2), box, seed=0).search(n_eval=20)
    assert (s.points >= box[:, 0]).all() and (s.points <= box[:, 1]).all()


def test_bayes_repeats_exactly_given_a_seed():
    """A run is one Sobol' sequence and a deterministic objective, so it repeats to the
    bit -- which is what makes the regression pin below meaningful."""
    a = BayesSearch(bump, BOX, seed=4).search(n_eval=20)
    b = BayesSearch(bump, BOX, seed=4).search(n_eval=20)
    c = BayesSearch(bump, BOX, seed=5).search(n_eval=20)
    assert b.points == pytest.approx(a.points, abs=0.0)
    assert not np.allclose(c.points, a.points)


def test_bayes_refuses_a_budget_below_the_initial_design():
    """Truncating the initial design would give a smaller search than the one asked
    for and running it in full a larger one; neither should happen silently."""
    with pytest.raises(ValueError, match="n_eval"):
        BayesSearch(bump, BOX).search(n_eval=N_INIT - 1)


def test_bayes_survives_a_constant_objective():
    """Standardising divides by the spread, which is zero here. The run degenerates to
    its Sobol' sequence rather than producing nan, and every point is distinct."""
    s = BayesSearch(lambda v: 2.5, BOX, seed=0).search(n_eval=N_INIT + 5)
    assert np.isfinite(s.points).all()
    assert len(np.unique(s.points, axis=0)) == N_INIT + 5


def test_search_emits_no_warnings():
    """`N_INIT` and `N_CAND` are powers of two, which is what Sobol' points need for
    their balance property; scipy warns otherwise, and a warning arriving at a caller
    who cannot act on it is not something this project does."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BayesSearch(bump, BOX, seed=0).search(n_eval=N_INIT + 2)


def test_ell_spans_the_scales_that_get_chosen():
    """`ELL`'s ends are a margin, not a range the fit uses. Over 3840 fits on eight
    real cross-validation surfaces the chosen length-scale ran from 0.155 to 0.739, so
    both ends sit a factor of nearly three outside anything observed. A fit landing on
    an end is therefore worth noticing, and this pins that the margin exists."""
    assert ELL[0] < 0.155 / 2 and ELL[-1] > 0.739 * 2


# -- the specifications -------------------------------------------------------------


def test_a_spec_runs_the_search_it_names():
    """Bit for bit against the search run directly, both of them: the specification is
    the budget and nothing else, so what it produces has to be what the caller would
    have written out by hand."""
    assert (grid().grid_size, bayes().n_eval) == (15, 40)

    g, want = grid(grid_size=4).search(bump, BOX), GridSearch(bump, BOX).search(4)
    assert isinstance(g, GridSearch)
    assert g.points == pytest.approx(want.points, abs=0.0)
    assert g.scores == pytest.approx(want.scores, abs=0.0)

    b = bayes(n_eval=10).search(bump, BOX, 3)
    want = BayesSearch(bump, BOX, seed=3).search(n_eval=10)
    assert isinstance(b, BayesSearch)
    assert b.points == pytest.approx(want.points, abs=0.0)


def test_the_grid_spec_ignores_the_seed():
    """Accepted so that a caller holding one of the two specifications need not know
    which; ignored because the grid is deterministic, which is half of what it is kept
    for."""
    a = grid(grid_size=3).search(bump, BOX, 0)
    b = grid(grid_size=3).search(bump, BOX, 1)
    assert a.points == pytest.approx(b.points, abs=0.0)


def test_a_budget_belonging_to_the_other_search_is_refused_at_once():
    """The whole point of stating the search early: `EigenCVFit` builds its folds before
    it ever calls `search`, so a keyword the chosen searcher does not take used to cost
    that construction before failing. Python's own `TypeError` is the check."""
    with pytest.raises(TypeError):
        bayes(grid_size=20)
    with pytest.raises(TypeError):
        grid(n_eval=40)


def test_a_spec_holds_no_state_and_can_be_reused():
    """Which is what lets one be shared between fits: the results belong to the searcher
    each call returns, not to the specification."""
    spec = grid(grid_size=3)
    first, second = spec.search(bump, BOX), spec.search(bump, BOX)
    assert first is not second
    assert not hasattr(spec, "points") and not hasattr(spec, "scores")


def test_a_name_is_shorthand_for_the_default_budget():
    """The spelling a caller with no view on the budget writes, and the object is how a
    budget is stated. A specification passes through whatever it is, so the resolution
    is idempotent and a caller may supply one of its own."""
    assert (as_spec("grid").grid_size, as_spec("bayes").n_eval) == (15, 40)
    assert as_spec(None).n_eval == 40

    spec = grid(grid_size=3)
    assert as_spec(spec) is spec


@pytest.mark.parametrize("method", ["grd", "Bayes", "", "grid "])
def test_a_name_that_is_neither_is_refused(method):
    """A misspelling would otherwise reach the caller as an `AttributeError` from
    wherever the search was first run -- for `cv.EigenCVFit`, after every fold has been
    built."""
    with pytest.raises(ValueError, match="method must be"):
        as_spec(method)


def test_a_spec_forwards_the_report():
    calls = []
    bayes(n_eval=N_INIT + 1).search(
        bump, BOX, 0, report=lambda i, total, m: calls.append((i, total))
    )
    assert [i for i, _ in calls] == list(range(1, N_INIT + 2))
    assert {total for _, total in calls} == {N_INIT + 1}


# -- against a real cross-validation ------------------------------------------------


# Pinned, not derived: the best score each search reaches on the `cv` fixture. See
# `test_matches_a_grid_on_a_real_cross_validation`.
GRID_BEST = 0.9745720779813849
BAYES_BEST = 0.9745977254933511


def test_matches_a_grid_on_a_real_cross_validation(cv):
    """Regression pin. No independent reference exists for the maximiser of a
    cross-validation surface, so what is pinned is that the two searches agree: 40
    Bayesian evaluations reach the best score of a 625-point grid, and in fact pass it
    by 0.003 of a fold standard error, since the grid can only report a grid point.

    A failure means the numbers moved and someone has to decide whether that is
    acceptable -- not that the code is wrong.
    """
    bounds = np.log([cv.lm1.lmbda_grid(2, 0.1), cv.lm2.lmbda_grid(2, 0.1)])
    f = lambda v: cv.score(*np.exp(v))  # noqa: E731

    g = GridSearch(f, bounds).search(grid_size=25)
    b = BayesSearch(f, bounds, seed=0).search(n_eval=40)
    best = g.points[g.scores.argmax()]
    fold = cv.fold_scores(*np.exp(best))
    se = fold.std(ddof=1) / np.sqrt(len(fold))

    assert b.scores.max() > g.scores.max() - 0.1 * se
    # The literals, so that a change in either search is a change to be explained.
    assert g.scores.max() == pytest.approx(GRID_BEST, abs=1e-9)
    assert b.scores.max() == pytest.approx(BAYES_BEST, abs=1e-9)


# -- the progress callback ----------------------------------------------------------


def test_grid_reports_every_evaluation_and_the_best_so_far():
    """The count is 1-based over evaluations completed, against `grid_size**2`, and the
    value is the running maximum -- checked against `np.maximum.accumulate` of the
    scores the search itself reports, on a surface whose later points are worse, so that
    best-so-far and last-value differ."""
    calls = []
    s = GridSearch(bump, BOX).search(
        grid_size=4, report=lambda i, t, m: calls.append((i, t, m["cv R²"]))
    )

    assert [i for i, _, _ in calls] == list(range(1, 17))
    assert {t for _, t, _ in calls} == {16}
    assert [v for _, _, v in calls] == pytest.approx(np.maximum.accumulate(s.scores))
    assert calls[-1][2] != s.scores[-1]


def test_bayes_reports_its_initial_design_too():
    """`total` is the whole budget and the initial design is counted against it: those
    evaluations cost the objective what any other does, which is what a progress count
    is about."""
    calls = []
    s = BayesSearch(bump, BOX, seed=0).search(
        n_eval=N_INIT + 4, report=lambda i, t, m: calls.append((i, t, m["cv R²"]))
    )

    assert [i for i, _, _ in calls] == list(range(1, N_INIT + 5))
    assert {t for _, t, _ in calls} == {N_INIT + 4}
    assert [v for _, _, v in calls] == pytest.approx(np.maximum.accumulate(s.scores))


def test_a_search_without_a_report_is_unchanged():
    """The callback is the whole of the addition: the points evaluated and the scores
    are the same, bit for bit, whether or not anyone is watching."""
    a = BayesSearch(bump, BOX, seed=0).search(n_eval=N_INIT + 3)
    b = BayesSearch(bump, BOX, seed=0).search(n_eval=N_INIT + 3, report=lambda *_: None)
    assert b.points == pytest.approx(a.points, abs=0.0)
    assert b.scores == pytest.approx(a.scores, abs=0.0)
