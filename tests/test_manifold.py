"""Tests for `manifold`.

The components here are *chosen*, not fitted: `c`, `w`, `b` and `u` are written down by
this file, so every quantity below has a value it set, and nothing about `probe.py` or
the joint fit enters. `X` is built as `F @ pinv(W)` plus noise, which makes `X @ W` the
coordinates back again to within that noise, so the `R^2`s are ordinary numbers rather
than arbitrary ones -- and their spread across components, which `reorder` needs, comes
from giving `F`'s columns different scales rather than from anything the fit does.

`Probe.manifold` is covered in `test_probe.py`, which is where the composition of the
two belongs.

What is independent
-------------------

`embed` and `project` unchanged to roundoff across `rescale`, `reorder`, `rotate`,
`reset` and any composition of them. This is the strongest structural check available
and it is what pins the `inv(T).T`: `U` transformed by `T` instead leaves `f` and `U`
scaling the same way, and the two maps then move under every operation.

`f()` against `f(z)` on the training covariates after the frame has moved. The two are
`F0 @ T` and `B(z) @ (C0 @ T)`, the same product associated differently, so agreement to
roundoff says the cached route and `C` describe one thing.

The varimax criterion, written out here as `mean(L**4) - mean(L**2)**2` summed over
columns, must increase under the returned `R`, and `R` must be orthogonal; and a matrix
already at simple structure must come back a signed permutation. `_varimax` forms that
criterion nowhere -- it iterates on the gradient and watches a nuclear norm -- so the
objective is checked against an expression the code does not contain.

The `R^2`s recomputed in plain numpy after a rotation, together with the fact that the
rotated values are not a permutation of the unrotated ones. That pair is what separates
recomputing from permuting.
"""

import numpy as np
import pytest

from maniprobe.basis import bs
from maniprobe.manifold import Manifold, _varimax
from maniprobe.probe import Component

N = 150
N_TEST = 40
P = 6
K = 10
D = 3
SCALES = np.array([1.0, 0.4, 2.0])


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def varimax_criterion(L):
    """The varimax criterion of `L`: the variance of the squared loadings, summed over
    columns. Written out here; `_varimax` never forms it."""
    return np.sum(np.mean(L**4, axis=0) - np.mean(L**2, axis=0) ** 2)


def build(rng, test_set=True, d=D):
    """A `Manifold` over `d` chosen components, and the training `z` and `X` it was
    built on -- which it drops, and two tests need in order to rebuild what it cached.

    The column scales are what give the components different `R^2`s, the noise being the
    same for all of them: the numerator of `1 - mean((f - g)**2) / var(f)` is fixed and
    the denominator is not.
    """
    z = rng.uniform(size=N + N_TEST)
    basis = bs(K)
    basis.setup(z[:N])

    C = rng.normal(size=(K, D)) * SCALES
    W = rng.normal(size=(P, D))
    b = rng.normal(size=D)
    U = rng.normal(size=(P, D))
    F = basis.model_matrix(z) @ C
    X = F @ np.linalg.pinv(W) + 0.05 * rng.normal(size=(N + N_TEST, P))

    components = [
        Component(c=C[:, k], w=W[:, k], b=b[k], u=U[:, k], r2=None, test_r2=None)
        for k in range(d)
    ]
    manifold = Manifold(
        basis,
        components,
        z[:N],
        X[:N],
        (X[N:], z[N:]) if test_set else None,
    )
    return manifold, z[:N], X[:N]


@pytest.fixture
def manifold(rng):
    return build(rng)[0]


def move(m):
    """One of each operation, composed, so that `T` is no kind of matrix in
    particular."""
    return m.rescale("u").reorder("r2").rotate().rescale("f")


class TestTheMaps:
    def test_embed_and_project_are_invariant(self, manifold):
        """INDEPENDENT, and the strongest check here: the frame is a description and
        the map is not, so nothing this class offers may move either map.

        Mutation-checked: transforming `U` by `T` instead of by `inv(T).T` breaks every
        one of these; so does dropping the transpose.
        """
        embed, project = manifold.embed(), manifold.project()
        for operation in (
            lambda: manifold.rescale("u"),
            lambda: manifold.rescale("f"),
            lambda: manifold.reorder("r2"),
            lambda: manifold.reorder("test_r2"),
            lambda: manifold.rotate(),
            lambda: manifold.reset(),
            lambda: move(manifold),
        ):
            operation()
            assert manifold.embed() == pytest.approx(embed, abs=1e-12)
            assert manifold.project() == pytest.approx(project, abs=1e-12)

    def test_embed_and_project_are_invariant_at_new_covariates(self, manifold, rng):
        """The same, off the rows the originals were evaluated on -- which is the route
        through `basis.model_matrix` and `C` rather than through `_F0`."""
        z = rng.uniform(size=20)
        x = rng.normal(size=(20, P))
        embed, project = manifold.embed(z), manifold.project(x)
        move(manifold)
        assert manifold.embed(z) == pytest.approx(embed, abs=1e-12)
        assert manifold.project(x) == pytest.approx(project, abs=1e-12)

    def test_the_cached_coordinates_agree_with_the_rebuilt_ones(self, rng):
        """INDEPENDENT of the shortcut. `f()` is `F0 @ T` and `f(z)` at the training
        covariates is `B(z) @ (C0 @ T)` -- the same product associated the other way --
        and likewise `G0 @ T` against `X @ W + b`. Checked after the frame has moved,
        since at `T == I` the two are the same arithmetic.

        Mutation-checked: transforming `C`, `W` or `b` by anything other than `T`.
        """
        manifold, z, X = build(rng)
        move(manifold)
        assert manifold.f() == pytest.approx(manifold.f(z), abs=1e-12)
        assert manifold.g() == pytest.approx(manifold.g(X), abs=1e-12)


class TestRescale:
    def test_f_is_made_unit_mean_square(self, manifold):
        """`'f'` is the norm the fit itself imposes: `||f||^2 / n == 1` under the
        empirical measure on the training rows."""
        manifold.rotate().rescale("f")
        assert np.mean(manifold.f() ** 2, axis=0) == pytest.approx(np.ones(D))

    def test_u_is_made_unit_euclidean_norm(self, manifold):
        """`'u'` is the ordinary norm in `R^p`, there being no measure there to use --
        so the two options are unit-norm in two different norms, and doing one undoes
        the other."""
        manifold.rescale("u")
        assert np.linalg.norm(manifold.U, axis=0) == pytest.approx(np.ones(D))
        assert not np.allclose(np.mean(manifold.f() ** 2, axis=0), 1.0)

        manifold.rescale("f")
        assert np.mean(manifold.f() ** 2, axis=0) == pytest.approx(np.ones(D))
        assert not np.allclose(np.linalg.norm(manifold.U, axis=0), 1.0)

    def test_rejects_any_other_unit(self, manifold):
        with pytest.raises(ValueError, match="unit must be"):
            manifold.rescale(unit="norm")


class TestReorder:
    def test_the_components_are_sorted_descending(self, manifold):
        """Descending, and the columns that move with the score are the same columns.

        Mutation-checked: sorting ascending fails the first assertion; permuting the
        coordinates without permuting `r2`, or the reverse, fails the second.
        """
        before, coordinates = manifold.r2.copy(), manifold.f().copy()
        manifold.reorder(by="r2")
        assert np.all(np.diff(manifold.r2) <= 0)
        assert manifold.r2 == pytest.approx(np.sort(before)[::-1])
        for k, score in enumerate(manifold.r2):
            source = int(np.argmin(np.abs(before - score)))
            assert manifold.f()[:, k] == pytest.approx(coordinates[:, source])

    def test_test_r2_orders_on_the_held_out_score(self, manifold):
        manifold.reorder(by="test_r2")
        assert np.all(np.diff(manifold.test_r2) <= 0)

    def test_rejects_any_other_score(self, manifold):
        with pytest.raises(ValueError, match="by must be"):
            manifold.reorder(by="edf")

    def test_rejects_test_r2_without_a_test_set(self, rng):
        m, *_ = build(rng, test_set=False)
        assert m.test_r2 is None
        with pytest.raises(ValueError, match="without a test set"):
            m.reorder(by="test_r2")


class TestRotate:
    def test_varimax_is_orthogonal_and_increases_the_criterion(self, manifold):
        """INDEPENDENT: the criterion is written out in this file and `_varimax` forms
        it nowhere, iterating on its gradient and stopping on a nuclear norm."""
        F = manifold.f()
        R = _varimax(F)
        assert R.T @ R == pytest.approx(np.eye(D), abs=1e-12)
        assert varimax_criterion(F @ R) > varimax_criterion(F)

    def test_varimax_leaves_simple_structure_alone(self, rng):
        """INDEPENDENT. Columns already concentrated on disjoint blocks of rows are at
        simple structure, so `R` must be a signed permutation: orthogonality forces the
        rest to zero once the three largest absolute entries are one."""
        F = np.zeros((3 * N, D))
        for k in range(D):
            F[k * N : (k + 1) * N, k] = rng.normal(size=N)
        R = _varimax(F)
        assert np.sort(np.abs(R).ravel())[-D:] == pytest.approx(np.ones(D), abs=1e-6)

    def test_rotating_a_rotated_frame_acts_on_the_current_coordinates(self, manifold):
        """Which is what makes rotations compose: a second rotation of a frame already
        at a stationary point is the identity, not the first rotation applied again."""
        manifold.rotate()
        rotated = manifold.T.copy()
        manifold.rotate()
        assert manifold.T == pytest.approx(rotated, abs=1e-7)

    def test_rotation_is_a_no_op_at_one_component(self, rng):
        """There is no simple structure to rotate a single column into, and `_varimax`
        is written for `d >= 2`; `rotate`'s guard is what holds it to that.

        Not mutation-checked, and recorded as such rather than left to look like it is:
        at `d == 1` the polar factor is the sign of `n * (mean(f**4) - mean(f**2)**2)`,
        a variance and so never negative, so `_varimax` returns the identity too and
        removing the guard changes no number. What the guard buys is the precondition
        stated where it is met, not a repair.
        """
        manifold, *_ = build(rng, d=1)
        assert np.array_equal(manifold.rotate().T, np.eye(1))

    def test_rejects_any_other_method(self, manifold):
        with pytest.raises(ValueError, match="method must be"):
            manifold.rotate(method="promax")


class TestScores:
    def test_r2_is_recomputed_after_a_rotation(self, manifold):
        """INDEPENDENT, in plain numpy from the rotated `f` and `g`.

        Mutation-checked, and this is the pair that does it: permuting the old values
        would pass the first assertion trivially and fail the second, a rotation mixing
        the residuals rather than relabelling them.
        """
        before = manifold.r2.copy()
        manifold.rotate()
        F, G = manifold.f(), manifold.g()
        reference = 1.0 - np.mean((F - G) ** 2, axis=0) / F.var(axis=0)
        assert manifold.r2 == pytest.approx(reference, rel=1e-12)
        assert not np.allclose(np.sort(manifold.r2), np.sort(before))

    def test_r2_after_a_rescale_does_not_assume_unit_variance(self, manifold):
        """`var(f_k) == 1` holds as a probe fits it and a rescale breaks it, so the
        spread is measured. Rescaling is a change of frame and must not move a score."""
        before = manifold.r2.copy()
        manifold.rescale("u")
        assert manifold.r2 == pytest.approx(before, rel=1e-10)

    def test_test_r2_is_recomputed_on_the_test_rows(self, manifold):
        """INDEPENDENT: the same expression on `_F0_test` and `_G0_test`, which the
        constructor built from the test set and nothing since has touched."""
        move(manifold)
        F, G = manifold._F0_test @ manifold.T, manifold._G0_test @ manifold.T
        reference = 1.0 - np.mean((F - G) ** 2, axis=0) / F.var(axis=0)
        assert manifold.test_r2 == pytest.approx(reference, rel=1e-12)


class TestReset:
    def test_reset_is_exact(self, manifold):
        """Exact rather than an accumulation of inverses: the originals were never
        modified, so this is `T = I` and a recomputation."""
        original = [manifold.C, manifold.W, manifold.b, manifold.U, manifold.r2]
        move(manifold)
        manifold.reset()
        for was, now in zip(
            original, [manifold.C, manifold.W, manifold.b, manifold.U, manifold.r2]
        ):
            assert np.array_equal(was, now)

    def test_reset_returns_the_identity_frame(self, manifold):
        move(manifold)
        assert np.array_equal(manifold.reset().T, np.eye(D))


class TestInspect:
    def test_inspect_reports_the_frame_and_a_row_per_component(self, manifold, capsys):
        manifold.inspect()
        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == "Components: 3   frame: original"
        assert lines[1] == (
            f"  Component 0   train R² {manifold.r2[0]:9.6f}"
            f"   test R² {manifold.test_r2[0]:9.6f}"
        )
        assert len(lines) == 4

    def test_inspect_says_when_the_frame_has_moved(self, manifold, capsys):
        manifold.rotate()
        manifold.inspect()
        assert "frame: transformed" in capsys.readouterr().out

    def test_inspect_leaves_the_test_column_out_without_a_test_set(self, rng, capsys):
        build(rng, test_set=False)[0].inspect()
        assert "test R²" not in capsys.readouterr().out
