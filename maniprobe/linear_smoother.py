"""Penalised regression against a functional basis.

`LinearSmoother` is `LinearModel` with a `Basis` in front of it. The covariate `z`
becomes the design and the penalty,

    X = basis.model_matrix(z)
    S = basis.penalty_matrix()

and prediction at new covariates goes back through the same map. That is the whole
class: fitting, EDF, `lmbda` selection, constraints and the rest are inherited
unchanged, because the parent is written entirely in terms of the cache built from
`X` and `S`.

A basis is frozen once set up, so `model_matrix` is from then on a pure function of
its argument -- which is what makes prediction safe here. The design a prediction is
built from is necessarily the one that was fitted.
"""

from .linear_model import LinearModel


class LinearSmoother(LinearModel):
    """Generalised ridge regression against a `Basis`.

    Whether the basis spans a constant, and whether that constant is penalised, are
    properties of the basis -- so `fit_intercept` is not exposed. A B-spline basis
    penalised at derivative order 2 spans the constant and leaves it in the penalty's
    null space, hence unshrunk, which is the intercept one would otherwise fit.

    Attributes
    ----------
    z
        The covariates as supplied, uncoerced: interpreting them is the basis's job.
    basis : Basis
        Set up, hence frozen. `self.X == basis.model_matrix(self.z)`.

    Plus the fitted state and design cache of `LinearModel`, which mean the same
    things here.
    """

    def __init__(self, z, basis, constraints=None, lmbda_criterion=None):
        """Set the basis up if it isn't already, then build the design cache.

        Parameters
        ----------
        z : array-like
            Covariates. Their shape is the basis's business -- an `(n,)` vector for a
            one-dimensional basis, `(n, d)` for a tensor product or a thin plate
            spline -- and they are passed through untouched.
        basis : Basis
            Configured or not. An unconfigured basis is set up from `z`; one that is
            already set up is taken exactly as it is, knots and all, even if they were
            placed on other data. Pass `basis.clone()` to place them afresh.
        constraints : ndarray, shape (p,) or (k, p), or list of ndarray, optional
            As for `LinearModel`, and stated in the basis's coefficient space:
            `np.ones(k)` is the sum-to-zero identifiability constraint on a basis whose
            columns are a partition of unity.
        lmbda_criterion : LmbdaCriterion, optional
            As for `LinearModel`.

        Raises
        ------
        ValueError
            If `basis` was constructed without a penalty. `penalty=None` says the
            basis does not participate in penalisation, which is not the same as a
            penalty of zero; for an unpenalised fit use `LinearModel` directly.
        """
        if not basis.is_setup:
            basis.setup(z)
        self.z = z
        self.basis = basis
        super().__init__(
            basis.model_matrix(z),
            basis.penalty_matrix(),
            constraints=constraints,
            lmbda_criterion=lmbda_criterion,
        )

    def predict(self, z_pred=None):
        """Predict at `z_pred`, defaulting to the training covariates.

        For derivatives of the fitted curve, where the basis differentiates,
        `basis.derivative_matrix(z_pred, order) @ model.beta`.

        Parameters
        ----------
        z_pred : array-like, optional
            Covariates, as for the constructor. None reuses the training design
            rather than rebuilding the same matrix.

        Returns
        -------
        ndarray, shape (m,)
        """
        X_test = None if z_pred is None else self.basis.model_matrix(z_pred)
        return super().predict(X_test)
