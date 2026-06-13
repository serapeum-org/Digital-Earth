"""DI.5 — vector fields, flow & labels (vectorfield / streamlines / barbs / text / labels).

Element-type, subsampling and CRS assertions plus mpl-backend render smokes for the matplotlib-only
barbs/streamlines. Runs in the ``interactive`` pixi env; every test ``importorskip``s geoviews.
"""

import pytest

from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")


@pytest.fixture()
def m() -> InteractiveMap:
    """A fresh Web-Mercator map for each test."""
    return InteractiveMap()


@pytest.fixture(scope="module")
def uv():
    """Two single-band pyramids Datasets (u, v) sharing the acc4000 grid."""
    from pyramids.dataset import Dataset

    base = Dataset.read_file("examples/data/acc4000.tif")
    return base, base  # same grid; values identical is fine for structural tests


class TestVectorField:
    """``vectorfield`` — u/v arrows via gv.VectorField.from_uv (recipe I6)."""

    def test_registers_gv_vectorfield(self, m, uv):
        u, v = uv
        out = m.vectorfield(u, v)
        assert out is m, "vectorfield() must chain"
        assert isinstance(m.layers[0], gv.VectorField), f"got {type(m.layers[0])}"

    def test_density_subsamples_the_grid(self, m, uv):
        u, v = uv
        m.vectorfield(u, v, density=1.0)
        m.vectorfield(u, v, density=0.25)
        dense = len(m.layers[0])
        sparse = len(m.layers[1])
        assert (
            sparse < dense
        ), f"lower density must mean fewer arrows: {sparse} !< {dense}"

    def test_invalid_density_raises(self, m, uv):
        u, v = uv
        with pytest.raises(ValueError, match="density must be in"):
            m.vectorfield(u, v, density=0.0)

    def test_magnitude_colour_recorded(self, m, uv):
        u, v = uv
        m.vectorfield(u, v, color_by="magnitude", cmap="plasma")
        style = hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs
        assert style["color"] == "Magnitude" and style["cmap"] == "plasma"

    def test_no_colour_when_color_by_none(self, m, uv):
        u, v = uv
        m.vectorfield(u, v, color_by=None)
        style = hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs
        assert "color" not in style or style.get("color") != "Magnitude"


class TestStreamlinesAndBarbs:
    """``streamlines`` / ``barbs`` — matplotlib-backend flow renders."""

    def test_streamlines_registers_and_logs(self, m, uv):
        u, v = uv
        m.streamlines(u, v)
        assert isinstance(m.layers[0], gv.VectorField)

    def test_streamlines_density_subsamples(self, m, uv):
        """density must actually subsample (M1: it was a dead parameter)."""
        u, v = uv
        m.streamlines(u, v, density=1.0)
        m.streamlines(u, v, density=0.25)
        assert len(m.layers[1]) < len(
            m.layers[0]
        ), "lower density must mean fewer streamline seeds"

    def test_barbs_register_or_clear_error(self, m, uv):
        u, v = uv
        if hasattr(gv, "WindBarbs"):
            m.barbs(u, v)
            assert type(m.layers[0]).__name__ == "WindBarbs"
        else:  # pragma: no cover - GeoViews build without WindBarbs
            with pytest.raises(ImportError, match="WindBarbs"):
                m.barbs(u, v)

    def test_barbs_density_subsamples(self, m, uv):
        """barbs exposes density for parity with vectorfield/streamlines (N1)."""
        u, v = uv
        if not hasattr(gv, "WindBarbs"):  # pragma: no cover - GeoViews without WindBarbs
            pytest.skip("GeoViews build without WindBarbs")
        m.barbs(u, v, density=1.0)
        m.barbs(u, v, density=0.25)
        assert len(m.layers[1]) < len(
            m.layers[0]
        ), "lower density must mean fewer barbs"

    def test_barbs_density_out_of_range_raises(self, m, uv):
        """density outside (0, 1] is rejected before building the element."""
        u, v = uv
        if not hasattr(gv, "WindBarbs"):  # pragma: no cover - GeoViews without WindBarbs
            pytest.skip("GeoViews build without WindBarbs")
        with pytest.raises(ValueError, match="density must be in"):
            m.barbs(u, v, density=0.0)

    def test_streamlines_mpl_render_smoke(self, m, uv, tmp_path):
        u, v = uv
        out = tmp_path / "stream.png"
        m.streamlines(u, v).save(str(out))
        assert out.exists() and out.stat().st_size > 0


class TestTextAndLabels:
    """``text`` / ``labels`` — annotations reprojected through pyramids."""

    def test_text_reprojects_lonlat_to_display_crs(self, m):
        m.text(12.5, 41.9, "Rome")
        element = m.layers[0]
        assert isinstance(element, gv.Text), f"got {type(element)}"
        from pyramids.feature.geometry import reproject_coordinates

        (x,), (y,) = reproject_coordinates([12.5], [41.9], from_crs=4326, to_crs=3857)
        assert element.x == pytest.approx(x, rel=1e-6)
        assert element.y == pytest.approx(y, rel=1e-6)

    def test_text_already_display_crs_passes_through(self, m):
        m.text(1_391_493.6, 5_146_011.7, "X", crs=3857)
        assert m.layers[0].x == pytest.approx(1_391_493.6, rel=1e-6)

    def test_labels_from_feature_column(self, m):
        from pyramids.feature import FeatureCollection

        fc = FeatureCollection.read_file("tests/data/points.geojson")
        m.labels(fc, "fid")
        element = m.layers[0]
        assert isinstance(element, gv.Labels), f"got {type(element)}"
        assert "fid" in [d.name for d in element.vdims]

    def test_labels_missing_column_raises(self, m):
        from pyramids.feature import FeatureCollection

        fc = FeatureCollection.read_file("tests/data/points.geojson")
        with pytest.raises(KeyError, match="nope"):
            m.labels(fc, "nope")

    def test_labels_render_under_bokeh_after_reproject(self):
        """Labels built from reprojected points must render under bokeh.

        Regression: handing GeoViews the geometry GeoDataFrame mis-projected the points at render
        (a boolean-mask length mismatch). Building from explicit display-CRS x/y arrays fixes it. The
        construct-only test above could not catch a render-time failure.
        """
        from pyramids.feature import FeatureCollection

        fc = FeatureCollection.read_file("tests/data/points.geojson")  # EPSG:32618
        m = InteractiveMap(crs=3857)  # forces a reproject through pyramids
        m.points(fc).labels(fc, "fid")
        hv.renderer("bokeh").get_plot(m.render())  # raises if the element mis-projects


class TestGraph:
    """``graph`` / ``flow`` — network / origin-destination flows (DI.15, recipe I9)."""

    @pytest.fixture()
    def nodes(self):
        import geopandas as gpd

        return gpd.GeoDataFrame(
            {"id": [0, 1, 2]},
            geometry=gpd.points_from_xy([-9e6, -8.5e6, -8e6], [4e6, 4.5e6, 4e6]),
            crs="EPSG:3857",
        )

    def test_graph_registers_gv_graph(self, m, nodes):
        edges = [(0, 1, 5.0), (1, 2, 3.0)]
        m.graph(nodes, edges, weight="weight")
        assert isinstance(m.layers[0], gv.Graph), f"got {type(m.layers[0])}"

    def test_flow_is_graph_alias(self, m, nodes):
        m.flow(nodes, [(0, 1), (1, 2)])
        assert isinstance(m.layers[0], gv.Graph)

    def test_bundled_graph_datashades(self, m, nodes):
        m.graph(nodes, [(0, 1), (1, 2)], bundle=True)
        assert isinstance(
            m.layers[0], (hv.RGB, hv.DynamicMap)
        ), f"got {type(m.layers[0])}"

    def test_weight_on_weightless_edges_draws_unweighted(self, m, nodes):
        """weight= on 2-tuple edges draws unweighted (logged, not a crash) — L2."""
        m.graph(nodes, [(0, 1), (1, 2)], weight="weight")
        element = m.layers[0]
        assert isinstance(element, gv.Graph)
        assert "weight" not in [d.name for d in element.vdims]
