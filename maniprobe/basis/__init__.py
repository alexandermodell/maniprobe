"""Functional bases: a model matrix and a penalty matrix.

A basis turns covariates into a design a fitting routine can regress against, and
carries the roughness penalty that routine should shrink against. `LinearSmoother` is
the caller these exist for; `Basis` states the contract and the lifecycle they all
share.

Three concrete bases:

`BSplineBasis`
    One axis, local support, knots placed by rule or given outright. The default choice
    for a single covariate.
`TensorProductBasis`
    A product of any two or more bases, one per group of axes. Anisotropic: each margin
    keeps its own units and its own knot spacing.
`ThinPlateBasis`
    Isotropic and knot-free in d dimensions, truncated to rank k by Wood's construction.
    Right for spatial coordinates, wrong for covariates in unrelated units.

`bs(k, ...)` builds the first or the second from one set of arguments, and `tps(k, ...)`
builds the third. Both return a basis centred and penalised, ready to smooth, where the
classes themselves take no view on either.
"""

from .base import Basis, BasisFunction
from .splines import BSplineBasis, bs
from .tensor import TensorProductBasis
from .thinplate import ThinPlateBasis, tps

__all__ = [
    "Basis",
    "BasisFunction",
    "BSplineBasis",
    "TensorProductBasis",
    "ThinPlateBasis",
    "bs",
    "tps",
]
