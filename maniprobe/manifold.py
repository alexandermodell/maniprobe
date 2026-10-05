"""The map a `probe.Probe` fitted, and the freedom in how it is written.

`Manifold` holds the two maps into `R^p` the components define,

    phi(z) = u_1 f_1(z) + ... + u_d f_d(z)
    psi(x) = u_1 g_1(x) + ... + u_d g_d(x),    g_k(x) = w_k.T x + b_k

and everything about *exploring* them. The split from `Probe` is the reason two classes
exist: after the last component is fitted there is nothing left to decide about the fit,
and everything left to decide about the frame it is written in.

The frame is the state
----------------------

Rescaling one component and unscaling its loadings, reordering the components, and
rotating them are all one thing -- a change of frame -- so one `d x d` matrix `T`
carries every one of them, and the class is that matrix over a set of originals that
are never modified:

    C == C0 @ T      W == W0 @ T      b == b0 @ T      U == U0 @ inv(T).T
    f() == F0 @ T    g() == G0 @ T

`reorder` composes a permutation onto `T`, `rescale` a diagonal, `rotate` an orthogonal
matrix, and `reset` sets it back to the identity. The consequence worth stating first:

    embed == f() @ U.T == F0 @ T @ inv(T) @ U0.T == F0 @ U0.T

so **`embed` and `project` are invariant to everything this class offers**, and only the
coordinates and the loadings move. That is the whole reason to hold `T` explicitly
rather than to mutate arrays in place -- the thing being changed is a description, not
the map -- and it is what makes `reset` exact rather than an accumulation of inverses.

Only four `n x d` arrays exist, the originals, and no snapshot duplicates them.
"""

import numpy as np

# Kaiser's iteration converges in a handful of steps at the `d` a probe produces, so the
# cap is a backstop rather than a budget; neither is a parameter, there being no
# question here a caller is better placed to answer than this file is.
_VARIMAX_TOL = 1e-9
_VARIMAX_MAX_ITER = 100


class Manifold:
    """The two maps a sequence of components defines, in a frame that can be changed.

    Attributes
    ----------
    basis : Basis
        The smoother's, so `f` can be evaluated at new covariates.
    T : ndarray, shape (d, d)
        The frame, the identity as fitted. Every operation right-multiplies onto it.
    C : ndarray, shape (k, d)
        Smoother coefficients, one column per component: `f(z) == B(z) @ C`.
    W : ndarray, shape (p, d)
    b : ndarray, shape (d,)
        Linear-model coefficients and intercepts: `g(x) == x @ W + b`.
    U : ndarray, shape (p, d)
        Loadings, the columns `embed` and `project` sum over. Transformed by
        `inv(T).T` and not by `T`, which is what makes the two maps invariant.
    r2, test_r2 : ndarray, shape (d,), or None
        `1 - mean((f_k - g_k)**2) / var(f_k)` per component, on the training rows and on
        the test rows where there are any. Recomputed after every change of frame, never
        permuted: permuting would be right for `reorder` and silently wrong for
        `rotate`, which mixes the residuals.

    Plus the originals `_C0`, `_W0`, `_b0`, `_U0`, `_F0`, `_G0`, `_F0_test`, `_G0_test`,
    which are never modified. `z` and `X` are not retained -- `_F0` and `_G0` are what
    was wanted from them, and `f`, `g`, `embed` and `project` take fresh covariates.

    `k` is the basis size, `p` the number of data columns, `d` the number of components.
    """

    def __init__(self, basis, components, z, X, test_set=None):
        """Stack the components, evaluate them on the data, and start at `T == I`.

        Parameters
        ----------
        basis : Basis
            Set up, as the smoother left it.
        components : list of probe.Component
            In the order they are to be held; `Probe.manifold` is what selects them.
        z, X
            The covariates and the data the components were fitted on. Evaluated here
            and then dropped.
        test_set : tuple, optional
            `(X_test, z_test)`, in the order of `(X, z)`, as `Probe` takes it.
        """
        self.basis = basis
        self._C0 = np.column_stack([c.c for c in components])
        self._W0 = np.column_stack([c.w for c in components])
        self._b0 = np.array([c.b for c in components])
        self._U0 = np.column_stack([c.u for c in components])
        self._F0 = basis.model_matrix(z) @ self._C0
        self._G0 = X @ self._W0 + self._b0

        if test_set is None:
            self._F0_test = self._G0_test = None
        else:
            X_test, z_test = test_set
            self._F0_test = basis.model_matrix(z_test) @ self._C0
            self._G0_test = X_test @ self._W0 + self._b0

        self.T = np.eye(len(components))
        self._refresh()

    def _refresh(self):
        """Recompute everything `T` determines: `C`, `W`, `b`, `U` and the two `R^2`s.

        `inv` rather than a `solve` against `_U0.T`, which would be the careful spelling
        for a general matrix: `T` is a product of permutations, orthogonal factors and
        diagonals, and the one thing that can make it ill-conditioned is a `rescale`
        driving a component's scale to zero -- which is the caller asking for it, and
        not a conditioning problem this could hide.
        """
        self.C = self._C0 @ self.T
        self.W = self._W0 @ self.T
        self.b = self._b0 @ self.T
        self.U = self._U0 @ np.linalg.inv(self.T).T
        self.r2 = _column_r2(self.f(), self.g())
        self.test_r2 = (
            None
            if self._F0_test is None
            else _column_r2(self._F0_test @ self.T, self._G0_test @ self.T)
        )

    def _compose(self, delta):
        """Right-multiply `delta` onto the frame and recompute. Every operation is one.

        Returns
        -------
        self
        """
        self.T = self.T @ delta
        self._refresh()
        return self

    # -- the maps ------------------------------------------------------------------

    def f(self, z=None):
        """The smooth coordinates at `z`, defaulting to the training covariates.

        Parameters
        ----------
        z : array-like, optional
            As the basis reads them. None takes `F0 @ T` rather than rebuilding a design
            already evaluated.

        Returns
        -------
        ndarray, shape (m, d)
        """
        if z is None:
            return self._F0 @ self.T
        return self.basis.model_matrix(z) @ self.C

    def g(self, x=None):
        """The linear readout of those coordinates at `x`, defaulting to the data.

        Parameters
        ----------
        x : ndarray, shape (m, p), optional

        Returns
        -------
        ndarray, shape (m, d)
        """
        if x is None:
            return self._G0 @ self.T
        return x @ self.W + self.b

    def embed(self, z=None):
        """`phi(z)`: the point of the manifold at covariates `z`.

        Invariant to every operation this class offers; see the module docstring. There
        is no `x_mean` and none is added: `phi` and `psi` both land in the centred
        space, and an offset on one of them would break the agreement between the two,
        which is the pair's whole content.

        Parameters
        ----------
        z : array-like, optional

        Returns
        -------
        ndarray, shape (m, p)
        """
        return self.f(z) @ self.U.T

    def project(self, x=None):
        """`psi(x)`: the point of the manifold estimated from data `x`.

        Invariant, as `embed` is.

        Parameters
        ----------
        x : ndarray, shape (m, p), optional

        Returns
        -------
        ndarray, shape (m, p)
        """
        return self.g(x) @ self.U.T

    # -- changes of frame ----------------------------------------------------------

    def rescale(self, unit="f"):
        """Make each component's coordinate, or its loadings, unit-norm.

        `U` transforms by `inv(T).T` where `f` transforms by `T`, so scaling a column of
        `T` up scales the matching column of `U` down, and the two options are the same
        diagonal read in opposite directions: to divide `u_k` by its norm, *multiply*
        that column of `T` by it.

        Parameters
        ----------
        unit : {'f', 'u'}
            Which of the two is made unit-norm -- and in two different norms, the
            asymmetry being in the spaces rather than in the code. `'f'` is unit mean
            square under the empirical measure on the training rows, `||f||^2 / n == 1`,
            which is the convention the fit itself imposes. `'u'` is the ordinary
            Euclidean norm in `R^p`, there being no measure there to use.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            On any other value.
        """
        if unit == "u":
            delta = np.linalg.norm(self.U, axis=0)
        elif unit == "f":
            delta = 1.0 / np.sqrt(np.mean(self.f() ** 2, axis=0))
        else:
            raise ValueError(f"unit must be 'f' or 'u', got {unit!r}")
        return self._compose(np.diag(delta))

    def reorder(self, by="r2"):
        """Order the components by one of the two `R^2`s, descending.

        Parameters
        ----------
        by : {'r2', 'test_r2'}
            Which score to order on. The fit produces them in no particular order:
            each component is the best remaining direction orthogonal to the ones
            before it, which is not the same as the best-fitting one.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            On any other value, or on `'test_r2'` with no test set.
        """
        if by not in ("r2", "test_r2"):
            raise ValueError(f"by must be 'r2' or 'test_r2', got {by!r}")
        scores = self.r2 if by == "r2" else self.test_r2
        if scores is None:
            raise ValueError("cannot reorder by test_r2 without a test set")
        # A permutation matrix is its own transpose's inverse, so `U` is permuted the
        # same way `f` is -- as it must be, the two being paired term by term.
        return self._compose(np.eye(len(scores))[:, np.argsort(scores)[::-1]])

    def rotate(self, method="varimax"):
        """Rotate the components to simple structure.

        Acts on the *current* coordinates, so rotations compose, and is a no-op at
        `d == 1`, where the only orthogonal transforms are the two signs.

        Parameters
        ----------
        method : {'varimax'}
            The one criterion implemented; see `_varimax`.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            On any other value.
        """
        if method != "varimax":
            raise ValueError(f"method must be 'varimax', got {method!r}")
        if len(self.T) == 1:
            return self
        return self._compose(_varimax(self.f()))

    def reset(self):
        """Return to the frame the probe fitted.

        Exact, not an accumulation of inverses: the originals were never modified, so
        this is `T = I` and a recomputation.

        Returns
        -------
        self
        """
        self.T = np.eye(len(self.T))
        self._refresh()
        return self

    def inspect(self):
        """Print the number of components, whether the frame has moved, and the `R^2`s.

        Original or transformed, and no more. Which operations were applied, and in what
        order, is a history nobody has asked for; `T` is the answer to the question that
        matters -- whether what is being read is still the fit as it came off the probe.
        """
        d = len(self.T)
        frame = "original" if np.array_equal(self.T, np.eye(d)) else "transformed"
        print(f"Components: {d}   frame: {frame}")
        for i in range(d):
            metrics = f"train R² {self.r2[i]:9.6f}"
            if self.test_r2 is not None:
                metrics += f"   test R² {self.test_r2[i]:9.6f}"
            print(f"  Component {i}   {metrics}")


# ---------------------------------------------------------------------------------
# the two computations
# ---------------------------------------------------------------------------------


def _column_r2(F, G):
    """`bilinear.BilinearModel.r2` column by column, for a whole frame at once.

    Written once and called for both the training and the test values, since it is the
    expression carrying the decision that `r2` is recomputed rather than permuted. The
    variance is measured rather than assumed: `var(f_k) == 1` holds as the probe fitted
    it and a `rescale` breaks it, and dividing by the measured spread is what makes that
    need no branch.

    Parameters
    ----------
    F, G : ndarray, shape (m, d)

    Returns
    -------
    ndarray, shape (d,)
    """
    return 1.0 - np.mean((F - G) ** 2, axis=0) / F.var(axis=0)


def _varimax(F):
    """An orthogonal `R` maximising the varimax criterion of `F @ R`.

    Kaiser's iteration, and without his row normalisation: that step exists to stop
    high-communality rows from dominating factor loadings of differing scales, and the
    columns here are already commensurate, being unit-norm under one measure. Each step
    is the polar factor of `F.T @ B` for

        B = L**3 - L @ diag(mean(L**2, axis=0)),   L = F @ R

    the gradient of the criterion, which is the classical result that the fixed point of
    that map is a stationary point of the varimax objective. The criterion itself is
    never formed: `sum(s)` is the nuclear norm of `F.T @ B`, which increases with it and
    stops when it does.

    `np.linalg.svd` rather than `linalg.svd`, and that is the exception this file has to
    state: it is wanted here as a *polar factorisation*, so every singular value is kept
    whatever its size, and dropping the small ones -- the shared helper's whole job --
    would return something that is not orthogonal.

    Parameters
    ----------
    F : ndarray, shape (n, d)
        The current coordinates, `d >= 2`.

    Returns
    -------
    ndarray, shape (d, d)
    """
    R = np.eye(F.shape[1])
    previous = 0.0
    for _ in range(_VARIMAX_MAX_ITER):
        L = F @ R
        B = L**3 - L @ np.diag(np.mean(L**2, axis=0))
        left, s, right = np.linalg.svd(F.T @ B)
        R = left @ right
        total = s.sum()
        if total - previous <= _VARIMAX_TOL * total:
            break
        previous = total
    return R
