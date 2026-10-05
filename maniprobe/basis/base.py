"""Base class for functional bases: a model matrix and a penalty matrix.

A basis is those two matrices. `model_matrix(z)` evaluates the basis functions at `z`,
giving the design a fitting routine regresses against; `penalty_matrix()` is the
roughness penalty it shrinks against. Everything else here exists to build them once and
then hold them still.

`penalty` is *opaque* state. This class may inspect only whether it is None; its
meaning, its validation and the timing of that validation belong entirely to the
concrete basis. By convention -- documented, not enforced -- an integer names the order
of the penalised derivative, so `penalty=2` is broadly comparable across bases without
any class asserting that it is. None says the basis does not participate in
penalisation, which is not the same as a penalty of zero. The matrix itself is built in
`_setup` and stored, so `penalty_matrix` returns fixed state; `gram_matrix`, where a
basis has one, stays lazy, since it takes a derivative order and cannot be precomputed
over all of them.

`model_matrix` takes covariates and nothing else, anywhere: it is *the* design, not a
family of them. A basis with more to offer names it separately --
`BSplineBasis.derivative_matrix` -- so that the option need not exist on the classes
with no use for it. Not every basis differentiates: a thin plate regression spline does
not.

Centring
--------
`centre=True` imposes `sum_i f(z_i) = 0` over the `z` given to `setup`, by subtracting
each basis function's mean over those points. So `setup` then requires data, and the
constraint holds for every coefficient vector rather than being enforced during fitting.
It costs one extra evaluation of the design at setup, and a retained vector of length
`k`.

The shift is a constant per basis function, which has three consequences worth stating.
A penalty of order 1 or more is unchanged, since it annihilates constants. A derivative
is unchanged, which is why `derivative_matrix` is not centred and needs no flag saying
so. But `gram_matrix`, and the unusual `penalty=0` mass matrix, are integrals of the
basis functions themselves and so refer to the *uncentred* ones.

Where the constant function lies in the span -- which is every basis here -- the centred
columns sum to zero, so the design has `k` columns and rank `k - 1`. That is left as it
is: what to do with a rank-deficient design belongs to the fitting routine.

Lifecycle
---------
A basis is *specified* by `__init__`, which takes hyperparameters only and never data;
*configured* by `setup(z=None)`, which derives knot placement and the like, then
discards `z`; and thereafter *frozen*, with attribute rebinding raising and derived
arrays marked read-only. `setup` owns all three transitions -- it guards, calls the
subclass's `_setup` and freezes -- so no concrete basis bookends its own configuration.

Freezing is what lets a `BasisFunction` hold a live reference rather than a copy, and it
is what makes `LinearSmoother.predict` safe: the design a prediction is built from is
necessarily the one that was fitted.

So a user can specify a basis and hand it to an algorithm, which calls `setup` with the
training data under the hood. Repeated fits -- folds, bootstrap resamples -- use
`clone()` for a fresh unconfigured instance, which makes "knots placed once on all the
data, or per replicate?" an explicit choice rather than an accident of mutation order.
`clone` works on a configured basis too: it rebuilds from `_params`, which `setup` never
touches.
"""

import numpy as np


class Basis:
    """A functional basis: hyperparameters, then a design and a penalty.

    Concrete bases add the two attributes describing their shape -- `ndim`, the number
    of covariate axes, and `k`, the number of basis functions -- which are theirs to
    define rather than defaults to inherit.

    Attributes
    ----------
    penalty
        Opaque penalty option; None means this basis does not participate in
        penalisation. Interpreted by the concrete basis, never here.
    centre : bool
        Whether the basis functions are shifted to sum to zero over the setup data.
    is_setup : bool
        False until `setup` has run, after which the object is immutable.
    """

    penalty = None
    centre = False
    is_setup = False
    _params = None

    # -- lifecycle -----------------------------------------------------------------

    def setup(self, z=None):
        """Configure internal state from optional data, then freeze.

        Concrete bases implement `_setup`; this owns the guard, the centring offset,
        the freeze and the return.

        Parameters
        ----------
        z : array-like, optional
            Training covariates. Their shape is the concrete basis's business, and a
            basis whose hyperparameters already fix the model space needs none at all
            -- unless it is centred, since the constraint is defined by the points it
            holds at.

        Returns
        -------
        self

        Raises
        ------
        RuntimeError
            If this basis has already been set up.
        ValueError
            If this basis is centred and no `z` is given.
        """
        if self.is_setup:
            raise RuntimeError(
                f"{type(self).__name__}.setup() may be called only once; "
                "use clone() for a fresh instance."
            )
        if self.centre and z is None:
            raise ValueError(
                "centre=True needs data: the constraint is that the basis functions "
                "sum to zero over the setup points, and without z there are none."
            )
        self._setup(z)
        # Through `_model_matrix` rather than `model_matrix`, whose `_require_setup`
        # guard is for callers and does not hold yet: this basis is not marked set up
        # until `setup` returns.
        self._offset = self._model_matrix(z).mean(axis=0) if self.centre else None
        # Top-level arrays only: one nested in a list or dict is missed, which is fine
        # here because a tensor product's sub-bases freeze themselves.
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        object.__setattr__(self, "is_setup", True)
        return self

    def _setup(self, z):
        """Derive internal state from `z`. Implemented by concrete bases."""
        raise NotImplementedError

    def clone(self):
        """A fresh, unconfigured instance with identical hyperparameters.

        Works on a configured basis as well as a specified one, since `_params` records
        the constructor arguments and `setup` never touches them. `TensorProductBasis`
        relies on that to clone its margins.

        Returns
        -------
        Basis
        """
        return type(self)(**self._params)

    def __setattr__(self, name, value):
        if self.is_setup:
            raise AttributeError(
                f"{type(self).__name__} is immutable after setup(); "
                f"cannot set {name!r}. Use clone() for a fresh instance."
            )
        object.__setattr__(self, name, value)

    # -- guards --------------------------------------------------------------------

    def _require_setup(self):
        if not self.is_setup:
            raise RuntimeError(f"call {type(self).__name__}.setup() before use.")

    def _require_penalty(self):
        if self.penalty is None:
            raise ValueError(
                f"{type(self).__name__} was constructed without a penalty. "
                "penalty=None means this basis does not participate in penalisation, "
                "which is not the same as a zero penalty matrix."
            )

    # -- matrices ------------------------------------------------------------------

    def model_matrix(self, z):
        """Basis functions evaluated at `z`, centred if this basis is.

        Concrete bases implement `_model_matrix`; this owns the setup guard and the
        centring shift.

        Parameters
        ----------
        z : array-like

        Returns
        -------
        ndarray, shape (n, k)
        """
        self._require_setup()
        B = self._model_matrix(z)
        return B if self._offset is None else B - self._offset

    def _model_matrix(self, z):
        """Basis functions evaluated at `z`. Implemented by concrete bases."""
        raise NotImplementedError

    def penalty_matrix(self):
        """The roughness penalty. Implemented by concrete bases.

        Built in `_setup` and stored, so this hands back fixed state: a read-only array,
        not a copy.

        Returns
        -------
        ndarray, shape (k, k)
        """
        raise NotImplementedError

    # -- evaluation ----------------------------------------------------------------

    def eval(self, z, coefficients):
        """Evaluate the function these coefficients name: `model_matrix(z) @ c`.

        Parameters
        ----------
        z : array-like
        coefficients : ndarray, shape (k,)

        Returns
        -------
        ndarray, shape (n,)
        """
        return self.model_matrix(z) @ coefficients

    def function(self, coefficients):
        """Bind coefficients to this basis, giving a callable.

        Parameters
        ----------
        coefficients : ndarray, shape (k,)

        Returns
        -------
        BasisFunction
        """
        return BasisFunction(self, coefficients)


class BasisFunction:
    """A callable binding fitted coefficients to a `Basis`.

    Holds a live reference rather than a snapshot, which is safe because a basis is
    immutable once set up.

    Attributes
    ----------
    basis : Basis
    coefficients : ndarray, shape (k,)
    """

    def __init__(self, basis, coefficients):
        """
        Parameters
        ----------
        basis : Basis
        coefficients : ndarray, shape (k,)
        """
        self.basis = basis
        self.coefficients = np.asarray(coefficients, dtype=float)

    def __call__(self, z):
        """Evaluate the fitted function at `z`.

        Parameters
        ----------
        z : array-like

        Returns
        -------
        ndarray, shape (n,)
        """
        return self.basis.eval(z, self.coefficients)
