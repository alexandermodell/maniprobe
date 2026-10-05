"""Joint fitting of two penalised models by alternating least squares.

`ALSFit` holds a pair of `LinearModel`s and alternates between them,

    lm2.set_y(lm1.predict()); lm2.fit_lmbda(); lm2.fit()
    lm1.set_y(lm2.predict()); lm1.fit_lmbda(); lm1.fit(); lm1.normalize_beta()

which minimises, over both models' coefficients,

    ||X1 beta1 - X2 beta2 - b2||^2 + lmbda1 * beta1.T @ S1 @ beta1
                                   + lmbda2 * beta2.T @ S2 @ beta2

subject to `||X1 beta1||^2 / n == 1`. `bilinear.EigenFit` solves the same problem
in one eigendecomposition; the reason to alternate instead is that each half-sweep is an
ordinary penalised regression, so both `lmbda`s can be re-selected from their own
criteria as the fit proceeds. The eigenproblem takes both as given.

What the sweep is
-----------------

In `lm1`'s orthonormal-design coordinates (`nu = lm1.pushforward(beta1)`, so
`f = U1 @ nu`), write `A1`, `A2` for the two shrinkage vectors, `K = U1.T @ U2`, and

    B = K diag(A2) K.T = U1.T @ H2 @ U1

for `lm2`'s smoother seen from `lm1`. Then `lm2`'s half-sweep is `g = H2 @ f` and
`lm1`'s is `nu <- A1 * (U1.T @ g)`, so one full sweep is

    nu <- A1 * (B @ nu),   rescaled to ||nu|| == sqrt(n)

-- power iteration on `A1 B`, with `normalize_beta` as its rescaling step. `A1 B` is
similar to the symmetric positive semi-definite `A1^.5 B A1^.5`, so its spectrum lies in
[0, 1], it has a single attractor reached from any start, and the Rayleigh quotient
`nu.T B nu / nu.T (I + lmbda1 Omega1) nu` increases on every sweep. Convergence is
linear at `mu2 / mu1` -- the same spectral gap that sets `EigenFit`'s eigenvector
accuracy, appearing here as a rate rather than as a conditioning term. It is started at
the least-penalised direction, `lm1.minimize_penalty()`: at fixed `lmbda` any start
would do, and this one is deterministic and needs no `lmbda` to state.

Relation to `EigenFit`
----------------------

`EigenFit` takes the smallest eigenpair of `M = lmbda1 Omega1 - B`. Writing
`M = (I + lmbda1 Omega1) - (I + B)`, its eigenvector equation `M nu == theta nu`
rearranges to

    A1 (B + c I) nu == nu,      c = 1 + theta == J* / n

for the optimal objective value `J*`. The two methods therefore differ by a *scalar
shift of B*, ALS being the `c == 0` case. A uniform shift is exactly what a rescaling of
`lmbda1` absorbs, and indeed the `EigenFit` solution at `lmbda1` is the ALS fixed
point at `lmbda1 / (-theta)`; that map is a strictly increasing bijection of [0, inf),
so the two trace the *same* solution path, differently indexed. ALS reaches nothing off
that path and misses nothing on it.

The paths agree but the labels do not: at a common `lmbda1` the two give different
answers, ALS's being the less smoothed. Since both `lmbda`s are selected here rather
than supplied, what matters is that the criteria are evaluated in the parametrisation
they were derived for -- `lm1`'s half-sweep is a linear smoother
`y -> U1 diag(A1) U1.T y`, which is what GCV, REML, AIC and BIC assume, and which the
constrained map (whose multiplier depends on the data) is not.

Each sweep costs `O(n (p1 + p2 + r1 + r2))` against `O(n r1 r2 + r1^3)` for the single
eigendecomposition, so alternating wins per sweep from about `r = 500` up and needs
`log(tol) / log(mu2 / mu1)` of them.

Acceleration
------------

`accel` attacks that sweep count. Writing `G` for the sweep and `r_k = G(nu_k) - nu_k`
for its residual, the type-II Anderson step fits that residual by differences of the
last `accel` iterates,

    gamma = argmin ||r_k - dR gamma||,   nu <- G(nu_k) - (dNu + dR) gamma

rescaled to the constraint, which the combination does not preserve. `accel = 1` is the
plain sweep, a one-iterate history leaving no difference to extrapolate from. Since it
is the rate it attacks, it pays where `mu2 / mu1` is near 1 and not otherwise: five to
ten times fewer sweeps at 0.95 and above, nothing at all at 0.5.

It is unguarded, and what that costs is a different limit, not a failure to converge.
Every eigenvector of `A1 B` is a fixed point of the sweep, hence a root of `r(nu) == 0`;
power iteration is attracted to the dominant one, a root-finder is not, and on designs
with `mu2 / mu1` around 0.98 about a third of draws settle on the second instead --
silently and quickly, since the residual there is genuinely zero. This is accepted
rather than safeguarded because at such a gap the two roots are near-indistinguishable
where it matters: the second costs some 0.02 of training `R^2`, being a subdominant
stationary point of the training objective, and out of sample nothing measurable.

One approximation is made knowingly: each `fit_lmbda` selects against a response that is
itself a fitted value, so the criterion's `n` and `edf` are not quite the ones its
derivation assumes -- standard for backfitting-style alternation.

Stopping
--------

`tol` is a threshold on `||f - f_old|| / sqrt(n)`, the sweep-to-sweep change in the
fitted direction. Since `f_new = G(f)`, that is the residual of the fixed-point equation
being solved, and it is first order in the distance to the solution: measured against a
converged reference it tracked that distance to four decimals over a whole run, at the
`(1 - mu2/mu1) / (mu2/mu1)` the theory gives, so `tol` implies an error of about
`tol * mu1 / (mu1 - mu2)`.

It is *not* a change in the objective, and that is the point rather than an oversight.
The solution is a stationary point of the objective, so successive objective values
differ at second order and stop moving long before the iterate does -- no rescaling of
that quantity helps, since a square root of something converging to a non-zero limit
changes its units and not its order. Testing `mean((f - g)**2)` instead, at a gap of
0.991 and a tolerance of 1e-6, stopped after 3 sweeps having left 0.035 of training
`R^2` on the table: more than the 0.02 that converging to the wrong eigenvector costs
above, and silently, since `max_iter` was never reached. Nor is such a test worth
keeping alongside this one: being second order where this is first, it is *implied* by
it, and would pass on every sweep where this one does.

The default of 1e-4 is where the two effects balance. At a gap of 0.991 it takes 106
sweeps and leaves 9e-8 of training `R^2`, against 361 and 615 sweeps at 1e-5 and 1e-6 to
reach 7e-9 and 7e-10 -- accuracy in a direction the objective is flat in, which is the
one nobody needs. At a gap of 0.672 it takes 22 sweeps and leaves 2e-6. A fixed residual
tolerance grants a wider error where the gap is tighter, which is the right way round:
the objective loss from an error `e` scales as `(mu1 - mu2) e^2`, so the same `e` costs
less exactly where it is dearer to remove.

Because `lmbda` moves every sweep -- and, with `accel > 1`, because the extrapolation is
not a descent step -- no objective is guaranteed to decrease monotonically, which is why
`max_iter` is a real backstop rather than a formality.
"""

import copy
import warnings
from collections import deque

import numpy as np

from . import linalg
from .bilinear import BilinearModel
from .linear_model import gcv


class ALSFit(BilinearModel):
    """A pair of `LinearModel`s fitted to each other, selecting both `lmbda`s.

    Attributes
    ----------
    lm1, lm2 : LinearModel
        The pair, as `BilinearModel` describes them; the scale constraint on `lm1` is
        reimposed after every sweep rather than by a Lagrange multiplier.
    test, tol, max_iter, accel, report
        As given to the constructor.
    n_iter : int
        Sweeps taken by the last `fit`; None before it.
    converged : bool
        False if that fit exhausted `max_iter` instead of meeting `tol`. The only
        subclass that writes it; see `BilinearModel`.

    Both models need a criterion attached before `fit`; neither needs a `lmbda`, since
    the initialisation sets `beta1` directly and the first sweep selects both. Not
    checked here: `fit_lmbda` raises on a missing criterion at the first call that needs
    it.
    """

    def __init__(
        self, lm1, lm2, test=None, tol=1e-4, max_iter=200, accel=1, report=None
    ):
        """
        Parameters
        ----------
        lm1, lm2 : LinearModel
            As `BilinearModel`, and fitted in place. Any `lmbda` already set on them is
            ignored: `fit` starts from `lm1.minimize_penalty()`, which needs none, and
            selects both on the first sweep.
        test : tuple, optional
            `(t1, t2)`, the arguments to `lm1.predict` and `lm2.predict` at held-out
            data -- designs, covariates, or one of each, depending on what the two
            models are. Used only for reporting.
        tol : float
            Threshold on `||f - f_old|| / sqrt(n)`, the sweep-to-sweep change in the
            fitted direction; see the module docstring for why it is that and not a
            change in the objective, and for what the default buys.
        max_iter : int
            Sweep limit. The rate is `mu2 / mu1`, so a design whose leading joint
            direction is poorly separated legitimately needs many sweeps -- 106 at a
            gap of 0.991 against 22 at 0.672, both at the default `tol`.
        accel : int
            History depth for the Anderson acceleration; 1 is the plain alternation.
        report : callable, optional
            Called once per sweep as `report(i, max_iter, metrics)`, `metrics` holding
            the training `R^2` and, where `test` is given, the held-out one -- which
            costs a `predict` over those rows every sweep. What is done with them is
            the caller's business: this module prints nothing, so that a caller driving
            several fits in turn has one display rather than one per fit.
        """
        super().__init__(lm1, lm2)
        self.test = test
        self.tol = tol
        self.max_iter = max_iter
        self.accel = accel
        self.report = report
        self.n_iter = None

    def fit(self):
        """Alternate until the residual mean square settles.

        Initialised from `lm1.minimize_penalty()` -- the least-penalised `f` the scale
        constraint allows, which needs neither a `lmbda` nor a `y` to state. That `f` is
        the response the loop opens on, which is why the sweep runs `lm2` first.

        `EigenFit(lm1, lm2).fit()` would also serve, and would start nearer the
        limit, but it needs both `lmbda`s set -- two numbers the caller has no basis
        for choosing, since the first sweep replaces them -- and it costs the
        `O(n r1 r2 + r1^3)` eigendecomposition this module exists to avoid.

        Ending on `lm1` leaves the constraint applied, so `f` -- the one quantity
        carried across the loop boundary, and the one the tolerance is measured on -- is
        normalised at every sweep. `mean((f - g)**2)` is the objective's data term at
        the parameters actually held, though not at the partially minimised ones: `lm2`
        was fitted to the previous sweep's `f`. It is computed only where there is a
        `report` to hand it to.

        Returns
        -------
        self

        Warns
        -----
        UserWarning
            If `max_iter` is exhausted before the tolerance is met.
        """
        self.lm1.minimize_penalty()
        f = self.lm1.predict()
        step = Anderson(self.accel)

        for i in range(1, self.max_iter + 1):
            f_old = f

            self.lm2.set_y(f)
            self.lm2.fit_lmbda()
            self.lm2.fit()
            g = self.lm2.predict()

            self.lm1.set_y(g)
            self.lm1.fit_lmbda()
            self.lm1.fit()
            self.lm1.normalize_beta()
            step(self.lm1)
            f = self.lm1.predict()

            self.n_iter = i
            if self.report is not None:
                # `f` has unit variance by the scale constraint, so `1 - mse` is
                # already the training R^2 and needs no second pass over the data.
                mse = np.mean((f - g) ** 2)
                metrics = {"train R²": 1.0 - mse}
                if self.test is not None:
                    metrics["test R²"] = self.r2(*self.test)
                self.report(i, self.max_iter, metrics)
            # The residual of the fixed-point equation, `f` being unit-variance already
            # so that no relative scaling is needed. First order in the distance to the
            # solution, which the objective's own decrement is not.
            if np.sqrt(np.mean((f - f_old) ** 2)) <= self.tol:
                self.converged = True
                break
        else:
            self.converged = False
            warnings.warn(
                f"no convergence in {self.max_iter} sweeps; raise max_iter, loosen "
                "tol, or set accel > 1 -- the rate is the eigenvalue gap mu2 / mu1, "
                "which a poorly separated design makes slow"
            )
        return self


# ---------------------------------------------------------------------------------
# acceleration
# ---------------------------------------------------------------------------------


class Anderson:
    """The Anderson step of the module docstring, applied to `lm1` after each sweep.

    Holds the last `depth` iterates and their residuals, and the one it last left on
    the model -- so an instance belongs to a single `fit`, and `ALSFit.fit` builds its
    own.
    """

    def __init__(self, depth):
        self.depth = depth
        self.nu = None
        self.nus = deque(maxlen=depth)
        self.rs = deque(maxlen=depth)

    def __call__(self, lm):
        """Replace `lm.beta` with the extrapolated iterate, in place.

        The iterate is `nu`, of length `r1` rather than `n`, so the least squares is
        free next to a sweep -- and it is the iterate the convergence result is about.
        A no-op on the first call, which has no incoming `nu` to difference against, and
        whenever fewer than two are stored, `depth == 1` included. `dR`'s columns are
        differences of residuals themselves converging to zero, so the solve is
        truncated at `linalg.EPS`.

        Parameters
        ----------
        lm : LinearModel
            `ALSFit`'s `lm1`, just fitted and normalised. Mutated in place.
        """
        g = lm.pushforward(lm.beta)  # `G(nu)`, the sweep's output in `nu` coordinates
        nu, self.nu = self.nu, g
        if nu is None:
            return
        self.nus.append(nu)
        self.rs.append(g - nu)
        if len(self.nus) < 2:
            return

        dNu = np.diff(np.column_stack(self.nus), axis=1)
        dR = np.diff(np.column_stack(self.rs), axis=1)
        gamma = np.linalg.lstsq(dR, self.rs[-1], rcond=linalg.EPS)[0]

        lm.beta = lm.pullback(g - (dNu + dR) @ gamma)
        lm.normalize_beta()
        self.nu = lm.pushforward(lm.beta)


# ---------------------------------------------------------------------------------
# the specification
# ---------------------------------------------------------------------------------


class ALSCriterion:
    """Alternation, re-selecting each model's `lmbda` from `criterion` every sweep.

    A specification rather than a fit: the criterion and the three settings of the
    sweep, and one method that turns a pair of models into an `ALSFit` over them. It is
    what a caller who is not driving `ALSFit` by hand hands to `probe.Probe`, and
    `cv.CVCriterion` is the other thing that can be handed there -- two classes with
    one method and no common state, so they duck-type rather than share a base that
    would own nothing.

    Attributes
    ----------
    criterion : LmbdaCriterion
        Copied per model by `fitter`; see it for why.
    tol, max_iter, accel
        As `ALSFit` reads them.
    """

    def __init__(self, criterion, tol, max_iter, accel):
        """
        Parameters
        ----------
        criterion, tol, max_iter, accel
            See the class attributes, and `als` for the defaults.
        """
        self.criterion = criterion
        self.tol = tol
        self.max_iter = max_iter
        self.accel = accel

    def fitter(self, sm, lm, test):
        """Build the `ALSFit` this specifies, a criterion attached to each model.

        Two attachments and therefore two instances: `attach` binds a criterion to one
        model, so a single object handed to both would leave each half-sweep selecting
        against the other's cache. This is the one place that knows two are needed,
        which is why the copying is here rather than at the call site. `copy.copy`
        rather than a `clone` on `LmbdaCriterion`: the four criteria hold scalars and a
        model reference `attach` overwrites, so a shallow copy is exact.

        Parameters
        ----------
        sm, lm : LinearModel
            The pair, `sm` the constrained side. Mutated by the attachment and, later,
            by the fit.
        test : tuple, optional
            `(t1, t2)`, as `ALSFit` reads it.

        Returns
        -------
        ALSFit
        """
        sm.set_lmbda_criterion(copy.copy(self.criterion))
        lm.set_lmbda_criterion(copy.copy(self.criterion))
        return ALSFit(
            sm, lm, test=test, tol=self.tol, max_iter=self.max_iter, accel=self.accel
        )


def als(criterion=None, tol=1e-4, max_iter=200, accel=3):
    """Construct an `ALSCriterion`.

    Parameters
    ----------
    criterion : LmbdaCriterion, optional
        Selected from every sweep, on each model separately. None is `gcv()`, so
        `als(accel=5)` says only what it is changing.
    tol, max_iter : float, int
        As `ALSFit`; see it and the module docstring for what the defaults buy.
    accel : int
        History depth for the Anderson acceleration, and the one default here that is
        not `ALSFit`'s: 3 rather than the plain alternation, the standard range being 3
        to 5 and `Anderson`'s `dR` columns -- differences of residuals converging to
        zero -- making the shallowest of them the best conditioned. What it accepts is
        the second-eigenvector trap the module docstring describes: about a third of
        draws at a gap near 0.98, costing some 0.02 of training `R^2` on their own, and
        rather more in a sequence of deflated fits, where the wrong direction is frozen
        into every later fit's constraints. There is no cheap test for it -- the honest
        one is the eigendecomposition the alternation exists to avoid -- so it is
        recorded rather than guarded, and a surprising sequence has a suspect.

    Returns
    -------
    ALSCriterion
    """
    return ALSCriterion(gcv() if criterion is None else criterion, tol, max_iter, accel)
