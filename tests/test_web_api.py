"""DX.1 (web half) — ``quickplot(backend="web")`` dispatches to a :class:`WebMap`.

A raster becomes ``add_raster``, a point ``FeatureCollection`` ``points``, a polygon one a ``choropleth``
(with a ``column``) or outline ``polygons``. Dispatch/validation tests need no engine; the drawing tests
``importorskip`` maplibre.
"""

import pytest

from digitalearth.api import quickplot
from pyramids.feature import FeatureCollection


@pytest.fixture()
def polys_fc(tmp_path):
    """A 4-polygon pyramids ``FeatureCollection`` (lon/lat) with a spread ``pop`` column for classifying."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    geoms = [Polygon([(i, 0), (i + 1, 0), (i + 0.5, 1)]) for i in range(4)]
    gdf = gpd.GeoDataFrame({"pop": [1.0, 5.0, 9.0, 3.0]}, geometry=geoms, crs=4326)
    path = tmp_path / "polys.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return FeatureCollection.read_file(str(path))


class TestWebBackendDispatch:
    """Validation/dispatch errors that do not need the engine."""

    def test_unknown_backend_message_lists_web(self, dataset):
        with pytest.raises(ValueError, match="web"):
            quickplot(dataset, backend="nope")

    def test_empty_featurecollection_raises(self):
        empty = FeatureCollection.read_file("examples/data/rhine_gauges.geojson").iloc[:0]
        with pytest.raises(ValueError, match="empty FeatureCollection"):
            quickplot(empty, backend="web")

    def test_non_vector_input_raises_typeerror(self):
        with pytest.raises(TypeError, match="cannot draw"):
            quickplot(123, backend="web")


class TestWebBackendDraw:
    """Input-type dispatch into the WebMap builders (engine required)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_raster_returns_webmap(self, dataset):
        from digitalearth.web import WebMap

        out = quickplot(dataset, backend="web")
        assert isinstance(out, WebMap)
        assert len(out.layers) >= 1 and out._last_layer_id is not None

    def test_points_return_webmap(self):
        from digitalearth.web import WebMap

        fc = FeatureCollection.read_file("examples/data/rhine_gauges.geojson")
        out = quickplot(fc, backend="web")
        assert isinstance(out, WebMap) and out._last_layer_id is not None

    def test_polygons_choropleth_by_column(self, polys_fc):
        out = quickplot(polys_fc, backend="web", column="pop", k=4)
        assert out.last_breaks is not None, "a column choropleth must classify the values"

    def test_polygons_default_no_column(self, polys_fc):
        from digitalearth.web import WebMap

        out = quickplot(polys_fc, backend="web")
        assert isinstance(out, WebMap) and len(out.layers) >= 1

    def test_basemap_adds_underlay(self, polys_fc):
        out = quickplot(polys_fc, backend="web", column="pop", k=4, basemap=True)
        assert len(out.layers) >= 2, "basemap=True should add a tile underlay beneath the data"
