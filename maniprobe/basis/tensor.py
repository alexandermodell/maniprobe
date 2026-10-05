"""Tensor-product basis, generically composed from any sub-bases.

The model matrix is the row-wise Kronecker product of the sub-bases' model matrices, and
the Gram matrix is the ordinary Kronecker product of theirs, at their respective
derivative orders. There is no special case anywhere: `gram_matrix()` at the zero
multi-index *is* the mass matrix, because that is what the product evaluates to.

The roughness penalty is a separate object, and that separation is the whole reason
`gram_matrix` and `penalty_matrix` are two methods rather than one. For derivative
orders m_j on each axis the penalty is the sum over axes of S_j fanned out against a
chosen factor on every other axis,

    P = sum_j (F_1 kron ... kron S_j kron ... kron F_r)

with `fan_out='integral'` taking F_i to be the undifferentiated Gram (mass) matrix, and
`fan_out='identity'` taking F_i = I. The integral form is the honest integral of the
squared partial derivative over the product domain, and is invariant under
reparameterisation of the other margins -- the penalty is a property of the function.
The identity form matches mgcv's `te()`: it penalises the roughness of each *column of
the coefficient array* instead, which is not invariant, but weights every marginal
coefficient equally regardless of knot spacing. Under the identity form the result is
not the Gram matrix of anything, which is precisely why the choice lives here and not in
`gram_matrix`.

The margins are stored at their original size and fanned out only on demand.
`penalty_matrices()` exposes them, which is what a fitting layer would consume to attach
one smoothing parameter per axis. Summed with equal weights, as the assembled penalty
does, they give the isotropic single-lambda penalty -- where the relative weight of each
axis is fixed by the units of the covariates.

A derivative here -- `gram_matrix`'s, the only one this class takes -- is a multi-index
with one entry per axis. A bare 0 is accepted as the zero multi-index, being the one
scalar that says the same thing on every axis; any other scalar is rejected rather than
reinterpreted, since it is ambiguous between "differentiate every axis" and "which
axis?". The model matrix is not differentiated at all: that would need every margin to
differentiate, and a thin plate margin does not.

The cost to keep in view is that `k` is *multiplicative*: three margins of 20 functions
each give an 8000-column design, and the penalty is 8000 x 8000. That number, not the
quadrature or the Kronecker algebra, is what decides whether a tensor product is usable.
"""

import math

import numpy as np

from .base import Basis


class TensorProductBasis(Basis):
    """Tensor product of two or more sub-bases.

    Sub-bases must be uniformly penalised or uniformly unpenalised. A mix is rejected:
    an unpenalised margin of any appreciable size contributes free wiggly directions
    with no roughness control, which shows up only as a fit that appears to overfit for
    no reason.

    Attributes
    ----------
    bases : tuple of Basis
        The sub-bases, in the order their covariates appear in the columns of `z`.
        Replaced by configured clones at setup.
    fan_out : {'integral', 'identity'}
        How a margin's penalty is spread over the axes it does not penalise.
    penalty
        `fan_out` when the margins are penalised, None when they are not -- this class's
        reading of the base's opaque penalty option. Two questions, deliberately apart:
        whether there is a penalty at all, and how it is fanned out.
    ndim, k : int
        Summed and multiplied over the sub-bases respectively.
    """

    def __init__(self, *bases, fan_out="integral", centre=False):
        """
        Parameters
        ----------
        *bases : Basis
            Two or more sub-bases. Duck-typed on capability rather than by class, so any
            conforming object composes -- including another `TensorProductBasis`, which
            nests.
        fan_out : {'integral', 'identity'}
            Fan-out scheme; see the module docstring. Ignored, harmlessly, when the
            margins carry no penalty to fan out.
        centre : bool, optional
            Constrain the product's basis functions to sum to zero over the setup data;
            see `Basis`. This is the level at which to ask for it: centring each margin
            instead leaves the product uncentred, since the column means of a row-wise
            Kronecker product are covariances rather than zeros.
        """
        if len(bases) < 2:
            raise ValueError("a tensor product needs at least two sub-bases.")
        if fan_out not in ("integral", "identity"):
            raise ValueError(
                f"fan_out must be 'integral' or 'identity'; got {fan_out!r}."
            )
        penalised = [b.penalty is not None for b in bases]
        if any(penalised) and not all(penalised):
            raise ValueError(
                "sub-bases must be uniformly penalised or uniformly unpenalised; "
                f"penalties are set on sub-bases "
                f"{[i for i, p in enumerate(penalised) if p]} and missing on the rest."
            )

        self.bases = tuple(bases)
        self.fan_out = fan_out
        self.penalty = fan_out if any(penalised) else None
        self.centre = centre

    # -- shape ---------------------------------------------------------------------

    @property
    def ndim(self):
        return sum(b.ndim for b in self.bases)

    @property
    def k(self):
        return math.prod(b.k for b in self.bases)

    def _as_points(self, z):
        """Covariates as an (n, ndim) array, one column per axis."""
        z = np.asarray(z, dtype=float)
        if z.ndim != 2 or z.shape[1] != self.ndim:
            raise ValueError(f"z must have shape (n, {self.ndim}); got {np.shape(z)}.")
        return z

    def _column_slices(self):
        """Which columns of `z` each sub-basis reads.

        An `int` for a one-dimensional margin and a `slice` for a wider one, so that
        `z[:, columns]` hands a B-spline margin a vector and a thin plate margin a
        matrix -- the shape each one wants.
        """
        out, start = [], 0
        for b in self.bases:
            stop = start + b.ndim
            out.append(slice(start, stop) if b.ndim > 1 else start)
            start = stop
        return out

    def _split_multi_index(self, alpha):
        """`alpha` split one piece per sub-basis, an `int` for a 1-D margin.

        The same running offset as `_column_slices`, kept separate because a derivative
        is not a set of columns: `gram_matrix` needs this and never those, and `_setup`
        the reverse.
        """
        out, start = [], 0
        for b in self.bases:
            stop = start + b.ndim
            sub = alpha[start:stop]
            out.append(sub[0] if b.ndim == 1 else sub)
            start = stop
        return out

    def _as_multi_index(self, derivative):
        """Normalise `derivative` to a tuple with one entry per axis."""
        if np.isscalar(derivative):
            if derivative != 0:
                raise TypeError(
                    f"{type(self).__name__} spans {self.ndim} axes, so derivative must "
                    f"be a multi-index of length {self.ndim}, not the scalar "
                    f"{derivative!r}. A scalar order is ambiguous here."
                )
            return (0,) * self.ndim
        alpha = tuple(derivative)
        if len(alpha) != self.ndim:
            raise ValueError(
                f"derivative multi-index has length {len(alpha)}, expected {self.ndim}."
            )
        return alpha

    # -- lifecycle -----------------------------------------------------------------

    def clone(self):
        """A fresh, unconfigured instance with identical hyperparameters.

        Overridden because the constructor takes its sub-bases positionally, which
        `_params` cannot express. Cloning the margins is the whole job: `clone` returns
        an unconfigured basis whether or not its receiver was configured, so
        `self.bases` serves before and after setup alike.

        Returns
        -------
        TensorProductBasis
        """
        return type(self)(
            *[b.clone() for b in self.bases], fan_out=self.fan_out, centre=self.centre
        )

    def _setup(self, z):
        """Configure clones of the sub-bases, then assemble the penalty.

        Cloning first means an externally configured sub-basis is simply irrelevant, and
        that `TensorProductBasis(b, b)` is valid rather than aliased. The cost is that
        hand-placed state on a sub-basis survives only if it is expressible as a
        hyperparameter -- which is what `BSplineBasis`'s `knots` argument is for.

        The penalty is assembled here rather than on demand, so it is fixed state of a
        configured basis. Everything it needs -- each margin's penalty and, under the
        integral form, each margin's mass matrix -- exists once the clones are set up.
        """
        self.bases = tuple(b.clone() for b in self.bases)
        if z is None:
            for b in self.bases:
                b.setup()
        else:
            z = self._as_points(z)
            for b, columns in zip(self.bases, self._column_slices()):
                b.setup(z[:, columns])

        if self.penalty is not None:
            # Read from the margins rather than through `penalty_matrices`, whose
            # `_require_setup` guard is for callers and does not hold yet: this basis
            # is not marked set up until `_setup` returns.
            P = [b.penalty_matrix() for b in self.bases]
            F = self._other_factors()
            self._penalty_matrix = sum(
                _kron_all([P[j] if i == j else F[i] for i in range(len(P))])
                for j in range(len(P))
            )

    # -- matrices ------------------------------------------------------------------

    def _model_matrix(self, z):
        """Row-wise Kronecker product of the sub-bases' model matrices.

        Parameters
        ----------
        z : array-like, shape (n, ndim)

        Returns
        -------
        ndarray, shape (n, k)
        """
        z = self._as_points(z)
        mats = [
            b.model_matrix(z[:, columns])
            for b, columns in zip(self.bases, self._column_slices())
        ]
        result = mats[0]
        for M in mats[1:]:
            result = _row_kron(result, M)
        return result

    def gram_matrix(self, derivative=0):
        """Kronecker product of the sub-bases' Gram matrices.

        Exact, not an approximation: the integral over a product domain of a product of
        separable functions factorises.

        Parameters
        ----------
        derivative : int or tuple of int, optional

        Returns
        -------
        ndarray, shape (k, k)
        """
        self._require_setup()
        subs = self._split_multi_index(self._as_multi_index(derivative))
        return _kron_all([b.gram_matrix(a) for b, a in zip(self.bases, subs)])

    def penalty_matrices(self):
        """Marginal penalties, one per sub-basis, at their original size.

        One entry per *sub-basis*, not per axis: a nested tensor product contributes its
        own already-summed penalty as a single entry, so its axes share a smoothing
        parameter. The two coincide only when every margin is one-dimensional.

        Returns
        -------
        list of ndarray
        """
        self._require_penalty()
        self._require_setup()
        return [b.penalty_matrix() for b in self.bases]

    def _other_factors(self):
        """What each margin's penalty is fanned out against on the other axes."""
        if self.fan_out == "identity":
            return [np.eye(b.k) for b in self.bases]
        return [b.gram_matrix() for b in self.bases]

    def penalty_matrix(self):
        """The fanned-out penalty, assembled at setup.

        Returns
        -------
        ndarray, shape (k, k)
        """
        self._require_penalty()
        self._require_setup()
        return self._penalty_matrix

    @property
    def null_space_dim(self):
        """Both forms leave the product of the margins' null spaces unpenalised."""
        self._require_penalty()
        return math.prod(b.null_space_dim for b in self.bases)


def _row_kron(A, B):
    """Row-wise Kronecker product: row i is the outer product of row i of each."""
    return np.einsum("ni,nj->nij", A, B).reshape(A.shape[0], -1)


def _kron_all(blocks):
    K = blocks[0]
    for M in blocks[1:]:
        K = np.kron(K, M)
    return K
