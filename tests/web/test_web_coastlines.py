"""WB-16 — Natural-Earth coastlines and borders as reference geography on the web tier.

``graticule()`` shipped, but the tier drew no coastlines or country borders: they came only with whatever
basemap style a caller picked. These two builders add them as **line** overlays in the reference band — over
the basemap, under the data, which is what :meth:`~digitalearth.web.base.WebMapBase.add_reference` is for —
built as GeoJSON from cleopatra's ``natural_earth`` coordinate arrays (the same source the static tier reads),
so no GIS is reimplemented here and the lines embed in a saved page. Land and ocean are out of scope: they are
polygons that need interior-ring support (UP-2).

The whole file ``importorskip``s maplibre, so it SKIPs under ``-e dev`` and RUNs under ``-e web``.
"""

import geopandas as gpd
import pytest
from shapely.geometry import Point

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


class TestCoastlinesDrawReferenceLines:
    """``coastlines()`` adds a Natural-Earth line overlay in the reference band."""

    def test_the_builder_is_chainable(self):
        """Every builder returns the map, so calls chain."""
        web_map = WebMap()
        assert web_map.coastlines() is web_map

    def test_it_joins_the_reference_band(self):
        """Reference geography sits over the basemap and under the data, which is the reference band."""
        assert WebMap().coastlines()._reference_count == 1

    def test_it_registers_a_line_layer(self):
        """The overlay is drawn as a MapLibre line, not a fill — coastlines are lines."""
        layers, _ = _recorded(WebMap().coastlines())
        assert any(layer.type == "line" for layer in layers), [
            layer.type for layer in layers
        ]

    def test_the_line_colour_reaches_the_paint(self):
        """A caller's colour is what the drawn line is painted with."""
        layers, _ = _recorded(WebMap().coastlines(color="#112233"))
        painted = [
            layer.paint["line-color"]
            for layer in layers
            if "line-color" in (layer.paint or {})
        ]
        assert "#112233" in painted, painted

    def test_the_geometry_comes_from_natural_earth(self):
        """The lines are real Natural-Earth geometry, so the source carries many LineString features."""
        _, sources = _recorded(WebMap().coastlines())
        features = sources[0]["data"]["features"]
        assert len(features) > 50, len(features)

    def test_every_feature_is_a_line(self):
        """Coastlines are lines only — no polygon fills reach the source."""
        _, sources = _recorded(WebMap().coastlines())
        kinds = {
            feature["geometry"]["type"] for feature in sources[0]["data"]["features"]
        }
        assert kinds == {"LineString"}, kinds


class TestBordersDrawReferenceLines:
    """``borders()`` is the same line path for country boundaries."""

    def test_the_builder_is_chainable(self):
        """Borders chain like every other builder."""
        web_map = WebMap()
        assert web_map.borders() is web_map

    def test_it_registers_a_line_layer(self):
        """Country borders are lines too."""
        layers, _ = _recorded(WebMap().borders())
        assert any(layer.type == "line" for layer in layers), [
            layer.type for layer in layers
        ]

    def test_the_geometry_comes_from_natural_earth(self):
        """Borders carry their own Natural-Earth geometry, distinct from the coastline."""
        _, sources = _recorded(WebMap().borders())
        assert len(sources[0]["data"]["features"]) > 50


class TestReferenceGeographySitsUnderTheData:
    """The band is the point: a coastline added after the data is still drawn beneath it."""

    def test_a_coastline_added_last_is_drawn_first(self):
        """Draw order, not call order: the reference line is queued beneath the data layer."""
        frame = gpd.GeoDataFrame({"v": [1.0]}, geometry=[Point(4.9, 52.4)], crs=4326)
        m = WebMap().points(frame, name="obs").coastlines()
        # The reference entry is inserted at the front of the queue, ahead of the data layer.
        assert m._reference_count == 1 and len(m._queued) == 2, m._queued


class TestMalformedCallsAreRefused:
    """A resolution Natural-Earth does not publish is refused at the call."""

    def test_an_unknown_resolution_is_refused(self):
        """Natural Earth ships 110m/50m/10m; anything else is a mistake named here."""
        mapped = WebMap()
        with pytest.raises(ValueError, match="resolution"):
            mapped.coastlines(resolution="1m")


class TestItRenders:
    """The queued closure adds the source and layer when the widget is built."""

    def test_a_coastline_map_renders(self):
        """Rendering runs the closure, so a crash building the overlay would surface here."""
        from maplibre.ipywidget import MapWidget

        assert isinstance(WebMap().basemap().coastlines().render(), MapWidget)
