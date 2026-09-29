"""DX.1 (web half) — ``quickplot(backend="web")`` dispatches to a :class:`WebMap`.

A raster becomes ``field``, a point ``FeatureCollection`` ``points``, a polygon one a ``choropleth``
(with a ``column``) or outline ``polygons``. Dispatch/validation tests need no engine; the drawing tests
``importorskip`` maplibre.
"""

import pytest
from pyramids.feature import FeatureCollection

from digitalearth.api import quickplot


@pytest.fixture
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
        empty = FeatureCollection.read_file("examples/data/rhine_gauges.geojson").iloc[
            :0
        ]
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
        assert len(out.layers) >= 1
        assert out._last_layer_id is not None

    def test_points_return_webmap(self):
        from digitalearth.web import WebMap

        fc = FeatureCollection.read_file("examples/data/rhine_gauges.geojson")
        out = quickplot(fc, backend="web")
        assert isinstance(out, WebMap)
        assert out._last_layer_id is not None

    def test_polygons_choropleth_by_column(self, polys_fc):
        out = quickplot(polys_fc, backend="web", column="pop", k=4)
        assert out.last_breaks is not None, (
            "a column choropleth must classify the values"
        )

    def test_polygons_default_no_column(self, polys_fc):
        from digitalearth.web import WebMap

        out = quickplot(polys_fc, backend="web")
        assert isinstance(out, WebMap)
        assert len(out.layers) >= 1

    def test_basemap_adds_underlay(self, polys_fc):
        out = quickplot(polys_fc, backend="web", column="pop", k=4, basemap=True)
        assert len(out.layers) >= 2, (
            "basemap=True should add a tile underlay beneath the data"
        )

    @pytest.mark.parametrize("flag", [True, False])
    def test_the_key_decision_is_recorded_either_way(self, polys_fc, flag):
        """``colorbar=`` records a guide on the keyed layer whichever way it is set.

        Args:
            polys_fc: A 4-polygon collection with a spread ``pop`` column.
            flag: What the caller passed as ``colorbar=``.

        Test scenario:
            This tier gated the whole key call on the flag, so ``colorbar=False`` left the layer carrying no
            guide at all while the other three backends recorded `Guide(show=False)` — a web figure written
            out after ``colorbar=False`` was indistinguishable from one where nobody asked (review M1). The
            picture agreed either way, which is why nothing caught it: the *record* is what travels through
            `FigureSpec`, and it is what lets a switcher offer the key back.

            Both values are parametrised rather than asserting the ``False`` case alone, so the fix cannot
            be "always record `show=False`". The panel is read too: a recorded-but-switched-off key must
            draw nothing, which is what keeps the record and the picture from disagreeing in the other
            direction.
        """
        out = quickplot(
            polys_fc,
            backend="web",
            column="pop",
            scheme="quantiles",
            k=3,
            colorbar=flag,
        )
        keyed = [
            layer_id
            for layer_id in out.layer_ids
            if (out.get_layer(layer_id).symbology.encoding("color") or None) is not None
            and out.get_layer(layer_id).symbology.guide() is not None
        ]
        assert keyed, (
            f"colorbar={flag} recorded no guide on any layer, so the figure carries no note that the key "
            "was decided"
        )
        guides = [out.get_layer(layer_id).symbology.guide() for layer_id in keyed]
        assert [guide.show for guide in guides] == [flag], (
            f"colorbar={flag} must be recorded as show={flag}, got {guides}"
        )
