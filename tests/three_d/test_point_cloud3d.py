"""Tests for the D3.2 point-cloud renderer (digitalearth.three_d.PointCloudMixin.point_cloud).

Gated on the optional ``3d`` extra (pyvista). Exercises the numpy xyz path (LiDAR-style), the 2-D lift-to-z=0
path, per-point colouring, the GeoDataFrame (``get_cell_points``-style) path read by duck-typing, and that the
guard module imports no GIS competitor.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.three_d import Scene3D
from digitalearth.three_d.point_cloud import SCALAR, _coords_from_array


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def test_point_cloud_from_xyz_array():
    """A coloured (N, 3) xyz table renders and registers one layer with the points and scalar."""
    pts = np.column_stack([np.arange(40.0), np.arange(40.0), np.linspace(0.0, 9.0, 40)])
    scene = Scene3D(off_screen=True)
    actor = scene.point_cloud(pts, values=pts[:, 2])
    assert actor is not None
    assert len(scene.layers) == 1
    cloud = scene.layers[0][0]
    assert cloud.n_points == 40
    assert SCALAR in cloud.point_data
    scene.close()


def test_two_column_table_is_lifted_to_z0():
    """A (N, 2) table is lifted to z=0 (so 2-D points still build a valid cloud)."""
    out = _coords_from_array(np.random.default_rng(0).random((25, 2)))
    assert out.shape == (25, 3)
    assert np.all(out[:, 2] == 0.0)


def test_bad_shape_raises():
    """A non-(N, 2)/(N, 3) array is rejected with a clear error."""
    zeros = np.zeros((5, 4))
    with pytest.raises(ValueError):
        _coords_from_array(zeros)


def test_uncoloured_cloud_has_no_scalar():
    """Without values/value_column the cloud carries no colour scalar."""
    scene = Scene3D(off_screen=True)
    scene.point_cloud(np.random.default_rng(1).random((30, 3)), eye_dome_lighting=False)
    assert SCALAR not in scene.layers[0][0].point_data
    scene.close()


def test_point_cloud_from_geodataframe(points):
    """A GeoDataFrame of points (e.g. get_cell_points) renders via duck-typed coordinate reads."""
    scene = Scene3D(off_screen=True)
    actor = scene.point_cloud(points, size=8.0)
    assert actor is not None
    assert scene.layers[0][0].n_points == len(points)
    scene.close()


def test_a_null_geometry_reads_as_nan_rather_than_raising():
    """A frame with one `null` geometry still builds a cloud, as it did before the PointArrays migration.

    `main` read `.x` straight off the series, and geopandas answers NaN there for a missing geometry.
    Routing the read through `PointArrays.from_features(..., centroids=False)` classified such a frame as
    "not all points" — a null's `geom_type` is NaN — and raised instead, turning a cloud that rendered into
    an exception on an ordinary GeoJSON feature.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    from digitalearth.three_d.point_cloud import _coords_from_geodataframe

    gdf = gpd.GeoDataFrame(geometry=[Point(0, 0), None, Point(2, 2)], crs="EPSG:3857")
    coords, values = _coords_from_geodataframe(gdf, None)
    assert coords.shape == (3, 3), f"every row survives the read, got {coords.shape}"
    assert np.isnan(coords[1, 0]), (
        "the missing row reads as NaN, the way geopandas answers for it"
    )
    assert values is None, "and no value column was asked for"


def test_renders_a_nonempty_frame():
    """The point cloud produces a real off-screen frame with content."""
    pts = np.random.default_rng(2).random((300, 3)) * 10
    scene = Scene3D(off_screen=True)
    scene.point_cloud(pts, values=pts[:, 2])
    img = scene.screenshot()
    assert img.ndim == 3 and bool(img.any())
    scene.close()


def test_values_length_mismatch_raises():
    """point_cloud() rejects a values array whose length does not match the points."""
    scene = Scene3D(off_screen=True)
    zeros = np.zeros((10, 3))
    zeros2 = np.zeros(7)
    with pytest.raises(ValueError, match="does not match"):
        scene.point_cloud(zeros, values=zeros2)
    scene.close()


class TestKeywordsTheDrawerWouldOverwrite:
    """H4 — `**kwargs` a caller passes must not be silently overwritten by the drawer.

    PyVista refuses a keyword it does not know (`_common_arg_parser` raises `TypeError`), so an arbitrary
    typo is already caught. What is not caught is a keyword PyVista *does* take and this drawer writes
    itself: `point_size`, the deleted spelling of `size` that the size fold overwrites, and the colour
    keywords the classification derives. Those drew a picture the caller did not ask for, with no
    diagnostic.
    """

    @staticmethod
    def _table() -> np.ndarray:
        """A small xyz point table.

        Returns:
            An ``(N, 3)`` float array of ascending points.
        """
        return np.column_stack([np.arange(12.0), np.arange(12.0), np.zeros(12)])

    def test_the_deleted_point_size_spelling_is_refused(self):
        """`point_size=` is the deleted spelling of `size=` and must raise, not draw at the default.

        Test scenario:
            The builder's `**kwargs` swallowed `point_size` into the layer's props, and the drawer then
            overwrote it from `size` — so `point_size=40` rendered at 5.0. Every other tier refuses the
            spelling because the parameter is gone from the signature; this one has to refuse it by name.
        """
        pts = self._table()
        scene = Scene3D(off_screen=True)
        try:
            with pytest.raises(TypeError) as excinfo:
                scene.point_cloud(pts, point_size=40)
        finally:
            scene.close()
        message = str(excinfo.value)
        assert "point_size=" in message, (
            f"the refusal must name the keyword it refuses, got {message}"
        )
        assert "size=" in message, (
            f"the refusal must name the live spelling to use instead, got {message}"
        )

    def test_the_live_spelling_still_reaches_the_actor(self):
        """`size=` is the live spelling and the actor must draw at exactly what it was given."""
        asked = 40.0
        scene = Scene3D(off_screen=True)
        try:
            actor = scene.point_cloud(self._table(), size=asked)
            drawn = float(actor.prop.point_size)
        finally:
            scene.close()
        assert drawn == asked, f"the cloud must draw at size={asked}, got {drawn}"

    @pytest.mark.parametrize("keyword", ["clim", "n_colors", "scalars"])
    def test_a_colour_keyword_the_classification_owns_is_named(self, keyword):
        """A pinned colour keyword on a coloured cloud is refused by name, not silently replaced.

        Args:
            keyword: A colour setting `classified_scalars` derives, or the scalar binding itself.

        Test scenario:
            `props.update(scalars=SCALAR, **style)` let the derived style win over whatever the caller
            pinned, so `clim=(0, 100)` on a classified cloud drew the class-index range instead and said
            nothing. The extruded-polygon drawer already refuses the same clash by name; this one did not.
        """
        values = {
            "clim": (0.0, 100.0),
            "n_colors": 3,
            "scalars": "elevation",
        }[keyword]
        pts = self._table()
        scene = Scene3D(off_screen=True)
        try:
            with pytest.raises(TypeError) as excinfo:
                scene.point_cloud(
                    pts, values=pts[:, 0], scheme="quantiles", k=4, **{keyword: values}
                )
        finally:
            scene.close()
        message = str(excinfo.value)
        assert f"{keyword}=" in message, (
            f"the refusal must name the keyword {keyword}=, got {message}"
        )

    def test_the_missing_data_colour_is_the_callers_to_choose(self):
        """`nan_color` is a choice, not a consequence, so a coloured cloud takes the caller's own.

        Test scenario:
            It was refused with `clim` and `n_colors`, which is the wrong company: those are the class-index
            range and the class count, and overriding either re-colours the wrong classes, while `nan_color`
            is filled from the shared missing-data colour as a *default*. `point_cloud` has no `nan_color`
            parameter, so refusing it meant "give up the colouring" for anyone who wanted a different one
            (review R2-L2). Read off the actor's lookup table, which is what VTK actually paints with.
        """
        pts = self._table()
        scene = Scene3D(off_screen=True)
        try:
            actor = scene.point_cloud(
                pts, values=pts[:, 0], scheme="quantiles", k=3, nan_color="red"
            )
            painted = tuple(actor.mapper.lookup_table.nan_color)
        finally:
            scene.close()
        assert painted == (1.0, 0.0, 0.0, 1.0), (
            f"the cloud paints missing data {painted} rather than the red that was asked for"
        )

    def test_the_default_missing_data_colour_is_the_shared_one(self):
        """Built the other way round: without a `nan_color` the shared neutral grey is what is painted.

        Test scenario:
            Without this the test above would pass for a cloud that happened to be red anyway, and the
            default is the thing the classifier is responsible for.
        """
        pts = self._table()
        scene = Scene3D(off_screen=True)
        try:
            actor = scene.point_cloud(pts, values=pts[:, 0], scheme="quantiles", k=3)
            painted = tuple(actor.mapper.lookup_table.nan_color)
        finally:
            scene.close()
        assert painted != (1.0, 0.0, 0.0, 1.0), (
            f"the default missing-data colour is {painted}, which is the red the other test asks for"
        )

    def test_the_refusal_does_not_call_a_continuous_ramp_a_scheme(self):
        """On the continuous path there is no classification, so "scheme=None" named nothing.

        Test scenario:
            The message read "together with the values it colours by (scheme=None)", which reads as a scheme
            actually called None; the values reach the engine unchanged there and the ramp owns the range.
        """
        pts = self._table()
        scene = Scene3D(off_screen=True)
        try:
            with pytest.raises(TypeError) as excinfo:
                scene.point_cloud(pts, values=pts[:, 0], scalars="elevation")
        finally:
            scene.close()
        assert "scheme=None" not in str(excinfo.value), excinfo.value

    def test_the_same_keyword_is_fine_on_an_uncoloured_cloud(self):
        """An uncoloured cloud derives no colour style, so a pinned `clim` is the caller's to set."""
        scene = Scene3D(off_screen=True)
        try:
            actor = scene.point_cloud(self._table(), clim=(0.0, 30.0))
            assert actor is not None, (
                "an uncoloured cloud with a pinned clim still draws"
            )
        finally:
            scene.close()
