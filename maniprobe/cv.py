"""K-fold cross-validation for the joint fit of two penalised models.

`KFoldCV` holds a fresh `EigenFit` per training fold, each over a pair of models built
on its own training rows, and scores a `(lmbda1, lmbda2)` pair by

    fold.fit(lmbda1, lmbda2)                  on the training rows
    R^2 of `lm2`'s prediction against `lm1`'s on the held-out rows

averaged over the folds. It selects nothing and searches nothing: it is the objective a
search strategy is pointed at, and the strategies live in `search.py`.

`EigenCVFit` is the composition -- a `KFoldCV`, a box, and a search -- and the only
class here a user need construct. It is an `EigenFit` that chooses its own two `lmbda`s
before fitting, so at that level of abstraction regularisation is not something the
caller has to know about: build the pair, call `fit()`, read the fitted models.

Why a joint fit and not an `ALSFit` sweep
-----------------------------------------

`ALSFit` exists so that both `lmbda`s can be re-selected from their own criteria as the
alternation proceeds. With the selection moved outside, that reason is gone and what
would be left of it is iteration error, a tolerance, a sweep limit, and -- under
acceleration -- the second-eigenvector trap. `EigenFit.fit` is the map
`(lmbda1, lmbda2) -> fit` exactly and deterministically, in one eigendecomposition, and
its `lmbda`s are the ones a search box would be stated in.

Holding an `EigenFit` per fold rather than re-forming the joint fit each time is
what makes a search affordable: `K = U1.T @ U2` is constant in both `lmbda`s, and at
`r = 2000`, `n = 50000` it is 1.21 s of the 1.73 s a fold would otherwise spend per
evaluation. Cached, a five-fold score costs 2.6 s instead of 8.7. Even so an evaluation
is seconds, which is why the search pointed at this should be one that spends them
carefully rather than a grid.

The score
---------

`f = X1 beta1` is the fitted direction and `g = X2 beta2 + b2` what the second model
makes of it, so the objective's data term on held-out rows is `mean((f - g)**2)` and the
score is `1 - mean((f - g)**2) / var(f)`, which is `BilinearModel.r2` at held-out data.

The denominator is not decoration. The objective is invariant to rescaling both
coefficient vectors together -- `||X1 beta1||^2 / n == 1` exists only to remove that
freedom -- and it is imposed on the *training* rows, so on held-out rows `var(f)` is
neither 1 nor constant in `lmbda1`: `beta1` is chosen to be unit-norm where it was
fitted while being maximally predictable by `lm2`, so as `lmbda1 -> 0` the direction
gets wigglier and carries more of its energy off the rows that normalised it.
Substituting 1 would fold that drift into the score rather than divide it out: on a
30-function basis at `n = 200` it runs monotonically from 0.99 under heavy smoothing to
1.08 as `lmbda1 -> 0`, so it moves the criterion's shape and not merely its level.

A `lmbda` means the same thing in every fold, and the winning pair transfers to a
full-data refit, because build step 4 normalises `omega` to unit median: dropping rows
scales the whole spectrum by roughly `n_train / n`, and a uniform factor is what that
normalisation removes.

That argument needs a fold model to be the same model as its parent up to the rows,
which is why `constraints` is reproduced along with `X` and `S`. A fold built without
them is a larger model, and searched on a different scale besides: a constraint
renormalises `omega` over the spectrum it leaves behind, so the unconstrained fold's
`lmbda` is not the constrained parent's by a factor no argument about rows removes.

Selecting from it
-----------------

`EigenCVFit` builds the box the search runs over from `LinearModel.lmbda_grid`, whose
ends each give up exactly `edf_tol` degrees of freedom, and searches it in the
logarithm: the useful range of a `lmbda` spans decades and the score moves smoothly in
`log(lmbda)`. Both transformations belong to the call site rather than to `search.py`,
and this is the call site. Because the ends are derived rather than fixed, a maximiser
landing on one is an answer and not a truncation, which is why nothing warns about it.

The winner is the argmax of what was evaluated, and it is refitted on the full data --
which the parents are for. The surface is routinely flat, so that argmax is often not a
meaningful choice among the fits near it; what makes it defensible is not precision but
that every point in that region is a fit the score cannot separate. Reading the surface,
rather than trusting its maximiser, is what `grid()` is kept for.

Centring
--------

`lm1` carries no intercept -- `EigenFit` refuses one, since the scale constraint
would invalidate it -- so the centring that makes `sum(f) == 0` has to live in the
design. That is a precondition of the joint fit rather than a detail of this module:
with the constant reachable, `f = 1` satisfies the scale constraint, is matched exactly
by `lm2`'s intercept, and lies in the null space of any derivative penalty, so the
objective is exactly zero at a fit that says nothing. Each fold therefore recentres
`lm1`'s design on its own training rows, and shifts the held-out rows by the same means.

Where the basis is centred already this is exactness rather than rescue. `X1 @ c0 == 0`
for the `c0` with `B @ c0 == 1` is an identity on the whole vector, so it holds on every
subset of rows and no fold can reach the constant either way; recentring only moves
`sum(f)` over the training rows from `O(1/sqrt(n))` to zero. `lm2` needs none of this:
an intercept is what a mean shift is for, and a fold model inherits its parent's.

Cost, and why `k` is a memory dial
----------------------------------

Construction is one design cache per fold -- an SVD of that fold's training design, plus
a `K` -- and that is the dominant one-off cost. What decides usability afterwards is
memory: every fold holds two full models, and at `r = p = 2000`, `n = 50000` each model
carries an `(n_train, p)` design and an `(n_train, r)` orthonormal design at 0.64 GB
apiece, so a fold of 40000 rows is 2.6 GB and five of them are 12.8 -- before the parent
models and the held-out slices.

`k` is therefore a memory dial as much as a statistical choice, and at large `n` it is
barely the latter: two folds of 25000 rows each estimate the score with ample precision,
and cost 3.2 GB rather than 12.8. Where `n` is small enough for the fold-to-fold
variance to matter, so is the memory.

A fractional `k` is the far end of that dial, and it turns the same dial twice. At
`k = 0.8` the fit sees exactly the rows a five-fold fold sees, at 2.6 GB rather than
12.8; and because a score is one fit per fold, every evaluation costs a fifth of what
it did, which at seconds apiece is what decides how long a search can run. What it gives
up is the averaging: the score is one split's estimate rather than the mean of five, and
`fold_scores` returns a single number, so there is no fold-to-fold spread to read an
uncertainty off. That trade is the same regime argument as above -- worth making where
`n` is large enough for one split to be precise on its own, which is where the memory
bites in the first place.
"""

import numpy as np

from .bilinear import EigenFit
from .linear_model import LinearModel
from .search import as_spec


class KFoldCV:
    """A pair of models rebuilt on training folds, scoring a `(lmbda1, lmbda2)` pair.

    Attributes
    ----------
    folds : list of tuple
        `(bl, X1_test, X2_test)` per fold -- an `EigenFit` over two fresh models
        built on that fold's training rows, and the held-out designs it is scored on.
        `X1_test` carries the training rows' centring; `X2_test` is as supplied.
    lm1, lm2 : LinearModel
        The parents, kept for `edf` alone -- the complexity a caller would order by
        belongs to the models that will be refitted, not to any fold's. Never fitted
        here.

    Every `score` call refits the fold models in place, so they hold the last-scored
    fit and nothing here is reentrant. Nothing is accumulated across calls: what was
    evaluated is the searcher's business, not the objective's.
    """

    def __init__(self, lm1, lm2, k=5, seed=None):
        """Build a copy of the pair per training fold.

        Parameters
        ----------
        lm1, lm2 : LinearModel
            Read as specifications and never fitted: `X`, `S`, `fit_intercept` and
            `constraints` are taken off them, the fold models are built from those, and
            they are kept afterwards only so `edf` can be read from them. `lm1` must
            carry no intercept, and its design must be centred -- see the module
            docstring for what goes wrong otherwise. Constraint rows are reproduced as
            given and read on each fold's own rows, which is what they say:
            `(X c).T @ (X beta) == 0` is a statement about the rows the fit sees, as the
            centring below is.
        k : int or float
            A fold count of at least 2, each pair built on `(k - 1) / k` of the rows
            and scored on the rest, with every row held out exactly once; or a training
            proportion in `(0, 1)`, which is one pair built on `k` of the rows and
            scored on the remainder. Also a memory dial; see the module docstring.
        seed : int or Generator, optional
            Seeds the permutation assigning rows to folds. The assignment is shuffled
            rather than contiguous because a design whose covariate is sorted -- which
            a smoother's routinely is -- would otherwise make every fold an
            extrapolation.

        Raises
        ------
        ValueError
            If the two models were built on different numbers of rows, which would
            otherwise index `lm2`'s design with `lm1`'s row numbers and say nothing.
            If `k` is neither of the two things above: `array_split` reads 2.5 as 2 and
            `k = 1.0` would leave nothing held out, so both would score something other
            than what was asked for without saying so.
        """
        n = len(lm1.X)
        if len(lm2.X) != n:
            raise ValueError(
                f"lm1 and lm2 must have the same number of rows, got {n} and "
                f"{len(lm2.X)}"
            )
        if not (0 < k < 1 or (k >= 2 and k == int(k))):
            raise ValueError(
                f"k must be a proportion in (0, 1) or an integer of at least 2, got {k}"
            )

        self.lm1, self.lm2 = lm1, lm2
        perm = np.random.default_rng(seed).permutation(n)
        if k < 1:
            n_train = round(k * n)
            splits = [(perm[:n_train], perm[n_train:])]
        else:
            idx = np.array_split(perm, k)
            splits = [
                (np.concatenate(idx[:i] + idx[i + 1 :]), test)
                for i, test in enumerate(idx)
            ]

        self.folds = []
        for train, test in splits:
            # The centring constraint belongs to the rows the fit sees, so the mean is
            # taken over the training rows and the held-out ones are shifted by that
            # same mean: centring them on themselves would be fitting to held-out data.
            x_mean = lm1.X[train].mean(0)
            self.folds.append(
                (
                    EigenFit(
                        LinearModel(
                            lm1.X[train] - x_mean,
                            lm1.S,
                            fit_intercept=lm1.fit_intercept,
                            constraints=lm1.constraints,
                        ),
                        LinearModel(
                            lm2.X[train],
                            lm2.S,
                            fit_intercept=lm2.fit_intercept,
                            constraints=lm2.constraints,
                        ),
                    ),
                    lm1.X[test] - x_mean,
                    lm2.X[test],
                )
            )

    def add_constraints(self, c):
        """Impose a constraint row on every fold, on that fold's own training rows.

        The row is stated in the parents' coefficient space and read on the rows each
        fold sees, which is what a constraint says -- `(X c).T @ (X beta) == 0` is a
        statement about rows, as the per-fold centring is -- so one `c` goes to all of
        them unchanged.

        This reaches the fold models a rebuild would have reached: the constructor
        copies `constraints` off the parents verbatim, and the two routes to a
        constrained cache agree to roundoff. What it costs instead is re-forming each
        fold's `K`, `O(n r1 r2)`, where a rebuild would re-decompose both of that
        fold's designs -- including `lm2`'s, which a constraint on `lm1` never changes.

        The parents are untouched; `EigenCVFit.add_constraints` is what moves both.

        Parameters
        ----------
        c : ndarray, shape (p1,) or (k, p1), or list of ndarray
            As `bilinear.BilinearModel.add_constraints`.

        Returns
        -------
        self
        """
        for bl, *_ in self.folds:
            bl.add_constraints(c)
        return self

    def fold_scores(self, lmbda1, lmbda2):
        """Held-out `R^2` in each fold, at these two `lmbda`s.

        Each fold is fitted from scratch: `EigenFit.fit` is a function of the two
        `lmbda`s and of a cache fixed at construction, so nothing carries over from the
        previous call and repeated evaluation at one point repeats exactly -- which a
        search strategy is entitled to assume.

        Split out from `score` so that the averaging is visible: the folds are combined
        by a mean of ratios, not pooled, and `score` being one line over this is what
        says so.

        Parameters
        ----------
        lmbda1, lmbda2 : float
            Non-negative, and relative to each model's normalised `omega`; see
            `linear_model`'s module docstring.

        Returns
        -------
        ndarray, shape (len(folds),) -- length 1 under a single train-test split.
        """
        return np.array(
            [
                bl.fit(lmbda1, lmbda2).r2(X1_test, X2_test)
                for bl, X1_test, X2_test in self.folds
            ]
        )

    def edf(self, lmbda1, lmbda2):
        """Total effective degrees of freedom of the pair: `edf1 + edf2`.

        Read off the *parent* models, since they are the ones a chosen `lmbda` pair is
        refitted on. Reported in the same coordinates as `score`, so that a caller
        comparing evaluated points has both numbers without reaching past this object
        into the models.

        A complexity *ordering* over `(lmbda1, lmbda2)`, not a claim about the joint
        fit's degrees of freedom: being no linear smoother of a fixed response, it has
        no such number, and the sum is what makes "simpler" a total order over two
        `lmbda`s at all.

        Parameters
        ----------
        lmbda1, lmbda2 : float

        Returns
        -------
        float
        """
        return self.lm1.edf(lmbda1) + self.lm2.edf(lmbda2)

    def score(self, lmbda1, lmbda2):
        """Mean held-out `R^2` over the folds -- the searchable map.

        A pure function of the two `lmbda`s: nothing is recorded, and repeated
        evaluation at one point repeats exactly.

        Parameters
        ----------
        lmbda1, lmbda2 : float

        Returns
        -------
        float
            Averaged over folds rather than pooled: each fold's ratio is scale-free on
            its own, and pooling would mix folds whose `f` differ in held-out spread.
        """
        return self.fold_scores(lmbda1, lmbda2).mean()


# ---------------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------------


class EigenCVFit(EigenFit):
    """A pair of `LinearModel`s fitted to each other, cross-validating both `lmbda`s.

    Attributes
    ----------
    lm1, lm2 : LinearModel
        The pair, as `BilinearModel` describes them. The same objects `cv` holds as its
        parents -- one shared reference, not a second copy -- and the ones `fit` leaves
        holding the full-data fit.
    cv : KFoldCV
        The objective. Never selects anything itself.
    edf_tol : float
        As given, and read by `fit`, which builds the box from it.
    bounds : ndarray, shape (2, 2)
        `log(lmbda)` box, rows `(lmbda1, lmbda2)` and columns `(lo, hi)`. Built by
        `fit`, not held from construction; None before it, and see `fit` for why.
    seed : int or Generator, optional
        As given, and used twice: the fold split and the Sobol' sequence.
    method : GridSpec or BayesSpec
        Which search `fit` will run, and its budget: `grid()` or `bayes()`, holding no
        state of its own, so one may be shared between fits. A name given as a string
        is resolved to one at construction, so this is always the object.
    report : callable, optional
        Handed to that search, which calls it after every evaluation; see
        `search.GridSearch.search`. Reassignable, so that a caller driving a sequence
        of fits can label each one without rebuilding the folds.
    search : GridSearch or BayesSearch
        The last search run, so its `points` and `scores` are the surface `fit` chose
        from. None before `fit`.

    Construction is the expensive part: one fold pair per split, plus this object's own
    `K` over the parents. That `K` is used once, at the closing refit, and costs about
    a third of one fold's construction -- paid up front so that `EigenFit`'s intercept
    check fires before any fold work does.
    """

    def __init__(
        self, lm1, lm2, k=5, seed=None, edf_tol=0.1, method=None, report=None
    ):
        """
        Parameters
        ----------
        lm1, lm2 : LinearModel
            As `EigenFit`, and fitted in place by `fit`. Any `lmbda` on them is ignored;
            `fit` selects both.
        k, seed
            Passed to `KFoldCV`; see it for both. `seed` also seeds `BayesSearch`, so
            one number reproduces a whole run and None leaves neither reproducible.
        edf_tol : float
            Degrees of freedom given up at each end of the box, as in `lmbda_grid`.
            Both ends of both axes are derived from it, which is what makes a step in
            either normalised coordinate the same amount of fit given up -- the
            isotropy `BayesSearch`'s single length-scale assumes. Kept rather than
            spent here: `fit` is where the box is formed.
        method : GridSpec, BayesSpec, {"grid", "bayes"} or None
            `bayes(n_eval=40)` costs that many evaluations and reports the path it took
            to one point; `grid(grid_size=15)` costs `grid_size**2` and reports the
            whole surface. Their answers are equally good -- worst-case regret over
            `search.py`'s benchmark surfaces is 0.041 of a fold standard error against
            0.030 -- so None is the cheap one, `bayes()`, at 40 evaluations against 225.
            Ask for the grid when the surface is what you want, or when the objective is
            fast enough that determinism costs nothing.

            The bare name is shorthand for that search at its default budget, and the
            object is how a budget is stated: `grid(grid_size=30)`. A budget belonging
            to the other search is then a `TypeError` from `grid` or `bayes` rather than
            a keyword nobody checks, which is what this replaced.
        report : callable, optional
            Passed to the search; see the class attributes.

        Raises
        ------
        ValueError
            As `search.as_spec` raises, on a name that is neither -- before the folds,
            which are the expensive part.
        """
        super().__init__(lm1, lm2)
        self.cv = KFoldCV(lm1, lm2, k, seed)
        self.seed = seed
        self.edf_tol = edf_tol
        # Resolved here rather than at first use: a misspelled name would otherwise
        # surface as an AttributeError from `fit`, after every fold had been built.
        self.method = as_spec(method)
        self.report = report
        self.bounds = self.search = None

    def fit(self):
        """Cross-validate both `lmbda`s, then fit the pair at the winner.

        Takes nothing: what the search costs and over what box -- `k`, `edf_tol`,
        `method` and its budget -- is one decision, settled at construction, which is
        also where everything this will evaluate was built.

        The box is the exception, and it is derived state rather than an option:
        `lmbda_grid`'s ends are a function of `omega`, and `add_constraints`
        renormalises `omega` -- by 23% where it was measured -- so a box formed once
        would search the wrong range for every fit after a constraint.

        The winner is the argmax of the scores actually measured, not of a surrogate's
        posterior mean: the latter is a prediction, and it is largest where the search
        has been least. Refitting is `EigenFit.fit` on the parents, which is the only
        fit here over all the rows -- everything before it happened on training
        subsets.

        Returns
        -------
        self
        """
        self.bounds = np.log(
            [
                self.lm1.lmbda_grid(2, self.edf_tol),
                self.lm2.lmbda_grid(2, self.edf_tol),
            ]
        )

        def objective(v):
            return self.cv.score(*np.exp(v))

        self.search = self.method.search(
            objective, self.bounds, self.seed, report=self.report
        )

        best = self.search.points[self.search.scores.argmax()]
        return super().fit(*np.exp(best))

    def add_constraints(self, c):
        """Constrain the folds and the parents alike, so that both describe one model.

        A fold left unconstrained cross-validates a model carrying a degree of freedom
        the closing refit does not have -- the defect the fold rebuild's `constraints`
        field exists for, measured at 0.124 of held-out `R^2` -- and the parents are
        what the box and that refit are read off. `super()` re-forms this object's own
        `K` in passing.

        Returns
        -------
        self
        """
        self.cv.add_constraints(c)
        return super().add_constraints(c)


# ---------------------------------------------------------------------------------
# the specification
# ---------------------------------------------------------------------------------


class CVCriterion:
    """Cross-validation of the pair's two `lmbda`s, as a specification.

    What an `EigenCVFit` will be built from, held before there are two models to build
    it over. `als.ALSCriterion` is the other object of this shape, and the two share
    only `fitter` -- no base, since two classes with one method and no common state
    would leave one owning nothing.

    It shares no base with `LmbdaCriterion` either, and that is a statement rather than
    an omission: that one picks a single model's `lmbda` from the training data by a
    criterion computed on it, this one picks the pair's two by held-out score. They
    have the word `criterion` and the argument they are passed to in common, and
    nothing else.

    Attributes
    ----------
    k, method, seed
        As `kf` and `ho` take them and `EigenCVFit` reads them; `k` is a fold count
        from one and a training proportion from the other.
    """

    def __init__(self, k, method, seed):
        """
        Parameters
        ----------
        k, method, seed
            See the class attributes. Built by `kf` or `ho`, which is where `k`'s two
            readings are told apart.
        """
        self.k = k
        self.method = method
        self.seed = seed

    def fitter(self, sm, lm, test):
        """Build the `EigenCVFit` this specifies.

        Parameters
        ----------
        sm, lm : LinearModel
            The pair, `sm` the constrained side. Construction builds the folds, so it
            is the expensive part, as `EigenCVFit`'s always is.
        test : tuple, optional
            Accepted and ignored: a cross-validated fit has a held-out score by
            construction and wants no second test set. The signature matches
            `als.ALSCriterion.fitter` so that a caller can dispatch on nothing.

        Returns
        -------
        EigenCVFit
        """
        return EigenCVFit(sm, lm, k=self.k, seed=self.seed, method=self.method)


def kf(k=5, method=None, seed=None):
    """Construct a `CVCriterion` scoring by `k`-fold cross-validation.

    One class implements this and `ho`, and they are kept as two helpers because a mean
    over folds and a single held-out split are different statements about what the
    score is -- and because each then checks its own first argument, which is the
    distinction they exist to make.

    Parameters
    ----------
    k : int
        Folds, each pair built on `(k - 1) / k` of the rows and scored on the rest.
        Also a memory dial; see the module docstring.
    method : GridSpec or BayesSpec, optional
        `grid()` or `bayes()`, carrying its own budget; as `EigenCVFit`.
    seed : int or Generator, optional
        The fold split and the Sobol' sequence both, so one number reproduces a run.

    Returns
    -------
    CVCriterion

    Raises
    ------
    ValueError
        If `k` is not an integer of at least 2. `KFoldCV` accepts the union of the two
        forms, so `kf(0.8)` would otherwise be a hold-out split without saying so.
    """
    if not (k >= 2 and k == int(k)):
        raise ValueError(f"k must be an integer of at least 2, got {k}")
    return CVCriterion(k, method, seed)


def ho(train=0.8, method=None, seed=None):
    """Construct a `CVCriterion` scoring on one held-out split.

    Parameters
    ----------
    train : float
        Training proportion in `(0, 1)`; the rest is scored on. One fit per evaluation
        rather than `k`, and no fold-to-fold spread to read an uncertainty off; see the
        module docstring for when that trade is the right one.
    method, seed
        As `kf`.

    Returns
    -------
    CVCriterion

    Raises
    ------
    ValueError
        If `train` is not in `(0, 1)`. `KFoldCV` accepts the union of the two forms, so
        `ho(5)` would otherwise be a five-fold average without saying so.
    """
    if not 0 < train < 1:
        raise ValueError(f"train must be a proportion in (0, 1), got {train}")
    return CVCriterion(train, method, seed)
