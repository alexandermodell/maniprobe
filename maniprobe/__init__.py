"""The Manifold Probe: a sequence of joint fits, and the map they define.

`Probe` fits a smooth function of covariates `z` against a linear readout of the data
`X`, one component at a time, deflating between them. `Manifold` is what the components
add up to -- two maps into `R^p` -- and everything about exploring them.

What is re-exported here is what a caller writes to fit a probe: the two classes, the
two basis helpers, the two models underneath, and the criteria that say how each fit
chooses its smoothing. The machinery those compose -- the joint fit itself (`bilinear`,
`als`, `cv`), the searches (`search`), the shared decompositions (`linalg`) -- is one
import deeper, in the module named for it, which is where anyone reaching for it already
knows to look.

The basis classes are among it, and deliberately: `bs` and `tps` return a basis centred
and penalised, which is what a probe needs, whereas the classes default to neither. A
probe rejects an uncentred basis, so reaching past the helpers is safe, but it is a
choice to make knowingly rather than the first thing an import offers.
"""

from .als import als
from .basis import bs, tps
from .cv import ho, kf
from .linear_model import LinearModel, aic, bic, gcv, reml
from .linear_smoother import LinearSmoother
from .manifold import Manifold
from .probe import Probe
from .search import bayes, grid

__all__ = [
    "Probe",
    "Manifold",
    "bs",
    "tps",
    "LinearModel",
    "LinearSmoother",
    "gcv",
    "reml",
    "aic",
    "bic",
    "als",
    "kf",
    "ho",
    "grid",
    "bayes",
]
