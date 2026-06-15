"""DX.1 (3-D half) — ``quickplot(backend="3d")`` dispatches to a :class:`Scene3D`.

Mirrors ``tests/test_interactive_autostyle.py::TestQuickplotBackend`` for the PyVista backend: a raster becomes
3-D ``terrain``, a point ``FeatureCollection`` a ``point_cloud``, a polygon one ``extruded_polygons``; lines and
unknown backends raise. Gated on the ``3d`` extra (pyvista).
"""
import pytest

pv = pytest.importorskip("pyvista")

from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

from digitalearth.scene import Map


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless so ``Scene3D()`` (built inside quickplot) never opens a window."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


@pytest.fixture
def raster():
    """A small projected raster (EPSG:32618)."""
    return Dataset.read_file("examples/data/acc4000.tif")


class TestQuickplot3DBackend:
    """``quickplot(backend="3d")`` dispatch (DX.1, the 3-D half)."""

    def test_raster_returns_scene3d_terrain(self, raster):
        from digitalearth.api import quickplot
        from digitalearth.three_d import Scene3D

        out = quickplot(raster, backend="3d")
        assert isinstance(out, Scene3D), f"expected Scene3D, got {type(out)}"
        assert len(out.layers) == 1  # one terrain layer
        out.close()

    def test_points_return_scene3d_point_cloud(self):
        from digitalearth.api import quickplot
        from digitalearth.three_d import Scene3D

        fc = FeatureCollection.read_file("examples/data/rhine_gauges.geojson")
        out = quickplot(fc, backend="3d")
        assert isinstance(out, Scene3D) and len(out.layers) == 1
        out.close()

    def test_polygons_extrude_and_colour_by_column(self):
        from digitalearth.api import quickplot
        from digitalearth.three_d import Scene3D

        fc = FeatureCollection.read_file("examples/data/rhine_basin.geojson")
        out = quickplot(fc, backend="3d", column="scalerank", height=50_000.0)
        assert isinstance(out, Scene3D) and len(out.layers) == 1
        out.close()

    def test_polygons_default_flat_height_no_column(self):
        """Polygons with neither ``column`` nor ``height`` extrude flat (height=1.0, column=None)."""
        from digitalearth.api import quickplot
        from digitalearth.three_d import Scene3D

        fc = FeatureCollection.read_file("examples/data/rhine_basin.geojson")
        out = quickplot(fc, backend="3d")
        assert isinstance(out, Scene3D) and len(out.layers) == 1
        out.close()

    def test_lines_raise_typeerror(self):
        from digitalearth.api import quickplot

        fc = FeatureCollection.read_file("examples/data/rhine_river_centerline.geojson")
        with pytest.raises(TypeError, match="uniformly point, polygon, or raster"):
            quickplot(fc, backend="3d")

    def test_unknown_backend_message_lists_3d(self, raster):
        from digitalearth.api import quickplot

        with pytest.raises(ValueError, match="3d"):
            quickplot(raster, backend="webgl")

    def test_matplotlib_default_still_returns_static_map(self, raster):
        from digitalearth.api import quickplot

        out = quickplot(raster, crs=raster.epsg)
        assert isinstance(out, Map), "default backend must remain the static Map"

    def test_colorbar_false_builds_without_error(self, raster):
        from digitalearth.api import quickplot
        from digitalearth.three_d import Scene3D

        out = quickplot(raster, backend="3d", colorbar=False)
        assert isinstance(out, Scene3D) and len(out.layers) == 1
        out.close()

    def test_colorbar_false_on_points_builds(self):
        """colorbar=False forwards show_scalar_bar=False into the point-cloud (add_points) path too."""
        from digitalearth.api import quickplot
        from digitalearth.three_d import Scene3D

        fc = FeatureCollection.read_file("examples/data/rhine_gauges.geojson")
        out = quickplot(fc, backend="3d", colorbar=False)
        assert isinstance(out, Scene3D) and len(out.layers) == 1
        out.close()

    def test_empty_featurecollection_raises_valueerror(self):
        from digitalearth.api import quickplot

        empty = FeatureCollection.read_file("examples/data/rhine_gauges.geojson").iloc[:0]
        with pytest.raises(ValueError, match="empty FeatureCollection"):
            quickplot(empty, backend="3d")

    def test_non_vector_input_raises_typeerror(self):
        from digitalearth.api import quickplot

        with pytest.raises(TypeError, match="cannot draw"):
            quickplot(123, backend="3d")

    def test_mixed_geometry_raises_clear_message(self):
        """Mixed point+polygon input raises the general message, not the misleading 'line geometries' one (N1)."""
        import pandas as pd
        from digitalearth.api import quickplot

        pts = FeatureCollection.read_file("examples/data/rhine_gauges.geojson").iloc[:2]
        poly = FeatureCollection.read_file("examples/data/rhine_basin.geojson").iloc[:1]
        mixed = pd.concat([pts, poly])
        with pytest.raises(TypeError, match="uniformly point, polygon, or raster"):
            quickplot(mixed, backend="3d")
