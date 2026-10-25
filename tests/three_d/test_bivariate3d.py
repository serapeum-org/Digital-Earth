"""The 3-D bivariate verb: two rasters draped as one colour from an n×n matrix (#492).

`Scene3D.bivariate(a, b, ...)` classifies two rasters engine-neutrally and drapes the resulting RGBA on a
ground plane. These cover that it records a drawn `bivariate` layer, honours an explicit extent and matrix, and
refuses bad input — the colour maths itself is covered by `tests/base/test_bivariate.py`.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.three_d import Scene3D  # noqa: E402


def _pair() -> tuple[np.ndarray, np.ndarray]:
    """Return two 6x6 rasters with spread, for a bivariate classification.

    Returns:
        The ``(a, b)`` arrays.
    """
    a = np.add.outer(np.arange(6.0), np.arange(6.0))
    b = np.add.outer(np.arange(6.0)[::-1], np.arange(6.0))
    return a, b


@pytest.fixture
def scene():
    """Yield an off-screen scene, closed on the way out.

    Yields:
        A scene under test.
    """
    built = Scene3D(off_screen=True)
    yield built
    built.close()


class TestBivariate:
    """`Scene3D.bivariate` drapes a bivariate raster as one drawn layer."""

    def test_it_records_a_bivariate_layer(self, scene):
        """The call records one layer of the `bivariate` kind.

        Args:
            scene: The scene under test.
        """
        a, b = _pair()
        scene.bivariate(a, b)
        assert scene.get_layer(scene.layer_ids[-1]).kind == "bivariate", scene.layer_ids

    def test_the_draped_plane_spans_the_index_extent(self, scene):
        """With no bbox the plane spans the array's ``(0, 0, cols, rows)`` extent.

        Args:
            scene: The scene under test.
        """
        a, b = _pair()
        scene.bivariate(a, b)
        bounds = scene.mesh_of(scene.layer_ids[-1]).bounds
        assert (bounds[0], bounds[1], bounds[2], bounds[3]) == (0.0, 6.0, 0.0, 6.0), (
            bounds
        )

    def test_an_explicit_bbox_sets_the_plane_extent(self, scene):
        """A bbox places the plane there instead of at the index extent.

        Args:
            scene: The scene under test.
        """
        a, b = _pair()
        scene.bivariate(a, b, bbox=(-10.0, 35.0, 30.0, 60.0))
        bounds = scene.mesh_of(scene.layer_ids[-1]).bounds
        assert (bounds[0], bounds[1], bounds[2], bounds[3]) == (
            -10.0,
            30.0,
            35.0,
            60.0,
        ), bounds

    def test_a_bbox_that_is_not_four_numbers_is_refused(self, scene):
        """A three-number box cannot be a (west, south, east, north).

        Args:
            scene: The scene under test.
        """
        a, b = _pair()
        with pytest.raises(ValueError, match="west, south, east, north"):
            scene.bivariate(a, b, bbox=(1.0, 2.0, 3.0))

    def test_mismatched_shapes_are_refused(self, scene):
        """`a` and `b` must be 2-D arrays of the same shape.

        Args:
            scene: The scene under test.
        """
        a = np.zeros((6, 6))
        b = np.zeros((5, 5))
        with pytest.raises(ValueError, match="same shape"):
            scene.bivariate(a, b)

    def test_a_custom_matrix_is_honoured(self, scene):
        """A caller's own 2×2 matrix is accepted when n matches.

        Args:
            scene: The scene under test.
        """
        a, b = _pair()
        matrix = (("#000000", "#111111"), ("#222222", "#333333"))
        scene.bivariate(a, b, scheme="equal_interval", n=2, matrix=matrix)
        assert scene.get_layer(scene.layer_ids[-1]).kind == "bivariate", scene.layer_ids
