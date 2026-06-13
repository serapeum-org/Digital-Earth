"""DI.6 — unstructured meshes, hexbin & 2-D KDE (trimesh / hexbin / kde, recipe I7).

Element-type, connectivity-source and aggregation assertions. Runs in the ``interactive`` pixi env;
every test ``importorskip``s geoviews.
"""

import numpy as np
import pytest

from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")

rng = np.random.default_rng(7)


@pytest.fixture()
def m() -> InteractiveMap:
    """A fresh Web-Mercator map for each test."""
    return InteractiveMap()


@pytest.fixture()
def point_fc():
    """A scattered point FeatureCollection (EPSG:32618, numeric 'fid')."""
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


class _FakeMesh:
    """A minimal UGRID-mesh duck type exposing node coords + fan triangles (like pyramids Mesh2d)."""

    def __init__(self):
        self.node_x = np.array([0.0, 1.0, 0.0, 1.0])
        self.node_y = np.array([0.0, 0.0, 1.0, 1.0])
        self.fan_triangles = np.array([[0, 1, 2], [1, 3, 2]])


class TestTrimesh:
    """``trimesh`` — UGRID connectivity or local Delaunay of scattered points."""

    def test_scattered_points_delaunay(self, m, point_fc):
        m.trimesh(point_fc, value_column="fid")
        assert isinstance(m.layers[0], gv.TriMesh), f"got {type(m.layers[0])}"
        assert len(m.layers[0].nodes) > 0, "the triangulated nodes must be populated"

    def test_ugrid_mesh_uses_supplied_connectivity(self, m):
        """A UGRID mesh (node_x/node_y/fan_triangles) is used directly, no triangulation."""
        m.trimesh(_FakeMesh())
        trimesh = m.layers[0]
        assert isinstance(trimesh, gv.TriMesh)
        assert len(trimesh.nodes) == 4, "the four mesh nodes must be carried through"

    def test_auto_rasterize_above_threshold(self, m, point_fc):
        m.trimesh(point_fc, value_column="fid", rasterize_threshold=1)
        assert isinstance(
            m.layers[0], hv.DynamicMap
        ), "above the face threshold the mesh must rasterize to an image"

    def test_forced_no_rasterize_keeps_trimesh(self, m, point_fc):
        m.trimesh(point_fc, value_column="fid", rasterize=False)
        assert isinstance(m.layers[0], gv.TriMesh)

    def test_scattered_without_value_column(self, m, point_fc):
        """Delaunay of bare points (no value column) still builds a TriMesh."""
        m.trimesh(point_fc, rasterize=False)
        assert isinstance(m.layers[0], gv.TriMesh)


class TestHexbin:
    """``hexbin`` — equal-area hex density binning."""

    @pytest.fixture()
    def cloud(self):
        import geopandas as gpd

        x = rng.uniform(-9e6, -8e6, 2_000)
        y = rng.uniform(4e6, 5e6, 2_000)
        return gpd.GeoDataFrame(
            {"v": rng.normal(0, 1, 2_000)},
            geometry=gpd.points_from_xy(x, y),
            crs="EPSG:3857",
        )

    def test_registers_hextiles(self, m, cloud):
        m.hexbin(cloud)
        assert isinstance(m.layers[0], gv.HexTiles), f"got {type(m.layers[0])}"

    def test_gridsize_and_aggregator_recorded(self, m, cloud):
        import numpy as np

        m.hexbin(cloud, gridsize=40, aggregator="mean", column="v")
        plot = hv.Store.lookup_options("bokeh", m.layers[0], "plot").kwargs
        assert plot["gridsize"] == 40, f"gridsize not recorded: {plot.get('gridsize')}"
        assert plot["aggregator"] is np.mean, "aggregator must be the numpy reducer HexTiles expects"

    def test_count_uses_np_size_reducer(self, m, cloud):
        """The default 'count' aggregator maps to ``np.size`` (what HexTiles needs for counting)."""
        import numpy as np

        m.hexbin(cloud)
        plot = hv.Store.lookup_options("bokeh", m.layers[0], "plot").kwargs
        assert plot["aggregator"] is np.size, "count must map to np.size, not the string 'count'"

    def test_mpl_render_smoke(self, m, cloud, tmp_path):
        out = tmp_path / "hex.png"
        m.hexbin(cloud).save(str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_bokeh_render_count(self, m, cloud):
        """Count hexbin must render under the *bokeh* backend (the .save smoke test uses matplotlib)."""
        hv.renderer("bokeh").get_plot(m.hexbin(cloud).render())

    def test_bokeh_render_value(self, m, cloud):
        """Value-aggregated hexbin must also render under bokeh (geometry projected from x/y arrays)."""
        hv.renderer("bokeh").get_plot(m.hexbin(cloud, column="v", aggregator="mean").render())


class TestKde:
    """``kde`` — 2-D kernel density of point positions."""

    def test_registers_bivariate(self, m, point_fc):
        m.kde(point_fc)
        assert isinstance(m.layers[0], hv.Bivariate), f"got {type(m.layers[0])}"

    def test_filled_toggle_recorded(self, m, point_fc):
        m.kde(point_fc, filled=False)
        plot = hv.Store.lookup_options("bokeh", m.layers[0], "plot").kwargs
        assert plot["filled"] is False
