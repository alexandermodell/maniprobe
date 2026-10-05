"""Shared numerical conventions: one tolerance, and the decompositions that use it.

Every rank decision here is the same comparison,

    keep the value if   |value| > max(EPS * spectrum.max(), atol)

a relative cut against the matrix's own largest value, floored by an absolute one. `atol`
defaults to 0, leaving the plain relative rule, and matters when the matrix was formed by
a cancellation: its computed spectrum is then roundoff, and measured against itself
nothing is zeroed. Each caller states the absolute level below which its own matrix
carries no information:

- `LinearModel._build_cache` passes `EPS**2 * abs(S).max()` twice. `EPS**2` is machine
  epsilon, so that is the rounding error in forming a congruence of `S`: both the Schur
  complement whose definiteness it checks, and the `S0` that `_pinv_solve` inverts,
  which is roundoff throughout wherever the penalty misses the design's null space.
- A constraint passes `EPS * norm(A_inv, 2) * max(norm(c))`, one `EPS` below the
  magnitude a constraint that constrains anything actually reaches -- written
  `EPS * D1.max() * max(norm(c))` where `_build_cache` imposes one during the build,
  `norm(A_inv, 2)` being `D1.max()` at that point.

One case is beyond any tolerance: a spectrum stretched by a congruence, where the rank
must be decided before the transform and the count carried across. That is `eig`'s
`n_zero`.
"""

import numpy as np

# Relative threshold for treating singular/eigen values as zero. Square root of machine
# precision: the usual choice, since squaring a design matrix squares its conditioning.
EPS = np.sqrt(np.finfo(float).eps)


def svd(M, return_null=False, atol=0.0):
    """Compact SVD of `M`, dropping singular values below the tolerance.

    Parameters
    ----------
    M : ndarray, shape (a, b)
    return_null : bool
        If True, also return a basis of `null(M)`. The *left* null space is never
        returned: it is `(a, a - r)`, and for a tall `M` that is the one array this
        module exists to avoid forming. Transpose `M` to put the null space you want on
        the right -- `null(M.T)` is what `svd(M.T, return_null=True)` gives you.
    atol : float
        Absolute floor on the cut; see the module docstring.

    Returns
    -------
    U : ndarray, shape (a, r)
    D : ndarray, shape (r,)
        Descending, every entry above the cut.
    Vt : ndarray, shape (r, b)
    V0t : ndarray, shape (b - r, b)
        Only if `return_null`. Its rows are an orthonormal basis of `null(M)`.
    """
    a, b = M.shape
    # np.linalg.svd truncates the right factor only when a < b, so this one expression
    # gives an economy U either way and a complete (b, b) Vt exactly when asked.
    U, D, Vt = np.linalg.svd(M, full_matrices=return_null and a < b)
    r = int((D > max(EPS * D.max(initial=0.0), atol)).sum())
    if return_null:
        return U[:, :r], D[:r], Vt[:r], Vt[r:]
    return U[:, :r], D[:r], Vt[:r]


def eig(M, keep_zeros=False, atol=0.0, n_zero=None):
    """Eigendecomposition of a symmetric `M`, zeroing noise-level eigenvalues.

    Thresholding rather than clamping at zero matters: `eigh` returns a structurally zero
    eigenvalue as ~1e-17, and at large `lmbda` that product is enough to shrink a
    direction the penalty does not reach.

    Parameters
    ----------
    M : ndarray, shape (r, r)
        Symmetric. Definiteness is not assumed: a negative eigenvalue above the cut
        survives with its sign, so a caller that needs positive semi-definiteness can
        test `(w < 0).any()` on the result.
    keep_zeros : bool
        If True, zeroed eigenvalues stay in place and `W` keeps all `r` columns. If
        False both are dropped, which is what you want before dividing by `w`.
    atol : float
        Absolute floor on the cut; see the module docstring.
    n_zero : int, optional
        Zero the `n_zero` leading eigenvalues instead of applying any tolerance. For a
        congruence `M = C.T @ N @ C` whose rank was decided on `N`: by Sylvester's law
        of inertia `dim null(M) == dim null(N)` exactly, whereas the eigenvalues
        themselves are rescaled by factors spanning `cond(C)**2` and so cannot be
        thresholded here at all. `eigh` returns them ascending, so for a PSD `M` the
        zeros are the leading entries.

    Returns
    -------
    w : ndarray, shape (r,) or (k,)
        Ascending, with noise-level entries exactly 0, or dropped.
    W : ndarray, shape (r, r) or (r, k)
    """
    w, W = np.linalg.eigh(M)
    if n_zero is None:
        keep = np.abs(w) > max(EPS * np.abs(w).max(initial=0.0), atol)
    else:
        keep = np.arange(len(w)) >= n_zero
    if keep_zeros:
        return np.where(keep, w, 0.0), W
    return w[keep], W[:, keep]
