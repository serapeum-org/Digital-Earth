"""WB-16 — Natural-Earth land/ocean/lakes polygon fills as reference geography on the web tier.

``coastlines()`` and ``borders()`` drew reference *lines*; the fill half was blocked while cleopatra's
``natural_earth`` returned exterior rings only (holes dropped, so an ocean fill painted over every continent).
cleopatra 0.42.0 added ``natural_earth_polygons`` — hole-aware ``[exterior, *holes]`` rings — so ``land()``,
``ocean()`` and ``lakes()`` now draw correct filled polygons as MapLibre ``fill`` layers in the reference band
(over the basemap, under the data), built as GeoJSON so they embed in a saved page.

The whole file ``importorskip``s maplibre, so it SKIPs under ``-e dev`` and RUNs under ``-e web``.
"""

import pytest

pytest.importorskip("maplibre", reason="the web tier needs the web environment")

from digitalearth.web import WebMap  # noqa: E402


def _recorded(web_map):
    """Replay a map's queued entries against a recorder and return its layers and sources.

    Args:
        web_map: The map whose queue is applied.

    Returns:
        A ``(layers, sources)`` pair: the ``maplibre`` ``Layer`` objects and the source specs the queued
        entries registered, in order.
    """
    layers = []
    sources = []

    class Recorder:
        """Stands in for the MapLibre widget, recording what each entry registers."""

        def add_source(self, src_id, source):
            """Record the source spec."""
            sources.append(source)

        def add_layer(self, layer):
            """Record the built layer."""
            layers.append(layer)

    for entry in web_map._queued:
        web_map._apply_layer(Recorder(), entry)
    return layers, sources


class TestFillsDrawReferencePolygons:
    """``land()``/``ocean()``/``lakes()`` add a Natural-Earth polygon fill in the reference band."""

    @pytest.mark.parametrize("builder", ["land", "ocean", "lakes"])
    def test_the_builder_is_chainable(self, builder):
        """Every fill builder returns the map, so calls chain.

        Args:
            builder: The fill builder under test.
        """
        web_map = WebMap()
        assert getattr(web_map, builder)() is web_map

    @pytest.mark.parametrize("builder", ["land", "ocean", "lakes"])
    def test_it_joins_the_reference_band(self, builder):
        """A fill is reference geography, so it joins the reference band, not the data band.

        Args:
            builder: The fill builder under test.
        """
        web_map = getattr(WebMap(), builder)()
        assert web_map._reference_count == 1

    @pytest.mark.parametrize("builder", ["land", "ocean", "lakes"])
    def test_it_registers_a_fill_layer(self, builder):
        """The overlay is a MapLibre ``fill``, not a line — these are areas.

        Args:
            builder: The fill builder under test.
        """
        layers, _ = _recorded(getattr(WebMap(), builder)())
        assert any(layer.type == "fill" for layer in layers), [
            layer.type for layer in layers
        ]

    def test_the_fill_colour_reaches_the_paint(self):
        """A caller's colour is what the drawn fill is painted with."""
        layers, _ = _recorded(WebMap().land(color="#123456"))
        painted = [
            layer.paint["fill-color"]
            for layer in layers
            if "fill-color" in (layer.paint or {})
        ]
        assert "#123456" in painted, painted

    def test_the_geometry_comes_from_natural_earth_polygons(self):
        """The fill is real Natural-Earth geometry, so the source carries many Polygon features."""
        _, sources = _recorded(WebMap().land())
        features = sources[0]["data"]["features"]
        assert len(features) > 50, len(features)

    def test_every_feature_is_a_polygon(self):
        """Fills are polygons only — no lines reach the source."""
        _, sources = _recorded(WebMap().ocean())
        kinds = {
            feature["geometry"]["type"] for feature in sources[0]["data"]["features"]
        }
        assert kinds == {"Polygon"}, kinds

    def test_ocean_holes_are_preserved(self):
        """The ocean's continent-shaped holes survive: some part carries more than one ring.

        Test scenario:
            cleopatra 0.42.0's ``natural_earth_polygons`` is hole-aware, so the main ocean polygon is
            ``[exterior, *continent_holes]``. A fill built from exterior rings alone would paint the sea
            over the continents (the cleopatra#384 bug); the emitted GeoJSON must keep the holes.
        """
        _, sources = _recorded(WebMap().ocean())
        features = sources[0]["data"]["features"]
        max_rings = max(len(f["geometry"]["coordinates"]) for f in features)
        assert max_rings > 1, (
            f"the ocean fill must keep its holes, got max rings {max_rings}"
        )

    def test_every_ring_is_closed(self):
        """GeoJSON requires each polygon ring closed (first vertex == last); open source rings are closed."""
        _, sources = _recorded(WebMap().land())
        features = sources[0]["data"]["features"]
        assert all(
            ring[0] == ring[-1]
            for f in features
            for ring in f["geometry"]["coordinates"]
        )


class TestFillsSitUnderTheData:
    """A fill added after the data is still drawn beneath it (the band, not call order)."""

    def test_a_fill_added_last_is_drawn_first(self):
        """``ocean()`` after the data is queued into the reference band, ahead of the data layer."""
        import geopandas as gpd
        from shapely.geometry import Point

        frame = gpd.GeoDataFrame({"v": [1.0]}, geometry=[Point(4.9, 52.4)], crs=4326)
        m = WebMap().points(frame, name="obs").ocean()
        assert m._reference_count == 1, m._queued
        assert len(m._queued) == 2, m._queued


class TestMalformedCallsAreRefused:
    """A resolution Natural-Earth does not publish is refused at the call."""

    def test_an_unknown_resolution_is_refused(self):
        """Natural Earth ships 110m/50m/10m; anything else is a mistake named here."""
        web_map = WebMap()
        with pytest.raises(ValueError, match="resolution"):
            web_map.land(resolution="1m")

    def test_a_non_finite_opacity_is_refused(self):
        """A NaN opacity could not be written into a saved figure, so it is refused at the call."""
        web_map = WebMap()
        nan = float("nan")
        with pytest.raises(ValueError, match="opacity"):
            web_map.ocean(opacity=nan)


class TestItRenders:
    """The queued closure adds the source and fill layer when the widget is built."""

    def test_a_fill_map_renders(self):
        """Rendering runs the closure, so a crash building the fill would surface here."""
        from maplibre.ipywidget import MapWidget

        assert isinstance(WebMap().basemap().ocean().land().render(), MapWidget)
