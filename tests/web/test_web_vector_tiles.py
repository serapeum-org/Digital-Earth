"""MVT vector tiles on the web tier (WB-6).

MapLibre serves a vector tile set (an ``.mvt``/``.pbf`` pyramid, or a TileJSON describing one) as a
``vector`` source, and draws one named layer out of it with an ordinary circle / line / fill layer. The tier
already drew a *raster* tile pyramid under the data through ``tiles``/``basemap`` — this is the vector
counterpart, and it draws among the data rather than beneath it.

The source is the tile URL, not a feature collection, so the layer draws from its description alone — like a
basemap, and unlike the GeoJSON vector builders. The tile set is a :class:`~digitalearth.web.VectorTileSource`
value object — the five fields that name the MapLibre source travel as one thing, and refuse an ambiguous
pairing at construction. Lives under ``tests/web/``, which the ``test-web`` pixi task runs in the ``web``
environment; without MapLibre the whole module skips.
"""

import pytest

from digitalearth.web import VectorTileSource, WebMap

#: A roads tile template reused across the styling and round-trip cases, where the source is incidental.
ROADS = "https://tiles.example.org/{z}/{x}/{y}.pbf"


@pytest.fixture(autouse=True)
def _need_engine():
    """Skip the module when the web extra is absent."""
    pytest.importorskip("maplibre")


def _payload(html: str) -> str:
    """Return the page's call payload — what this map does, not what the library contains.

    Args:
        html: A page from ``to_html``.

    Returns:
        The substring from ``var data =`` to the end of the page.
    """
    marker = html.rfind("var data = ")
    assert marker != -1, "the exported page carries no call payload"
    return html[marker:]


class TestTheSourceValueObjectIsBuiltAndRefused:
    """``VectorTileSource`` holds the MapLibre source and refuses an ambiguous tile set at construction."""

    def test_a_template_builds_a_tiles_source(self):
        """A ``{z}/{x}/{y}`` template becomes a ``vector`` source listing that one template."""
        source = VectorTileSource(tiles=ROADS).to_source()
        assert source == {"type": "vector", "tiles": [ROADS]}, source

    def test_a_tilejson_url_builds_a_url_source(self):
        """A TileJSON ``url`` becomes a ``vector`` source carrying ``url`` and no ``tiles``."""
        source = VectorTileSource(
            url="https://tiles.example.org/roads.json"
        ).to_source()
        assert source == {
            "type": "vector",
            "url": "https://tiles.example.org/roads.json",
        }, source

    def test_the_zoom_range_and_attribution_reach_the_source(self):
        """The optional coverage fields fold into the source dict the layer draws from."""
        source = VectorTileSource(
            tiles=ROADS, min_zoom=4, max_zoom=14, attribution="© Example"
        ).to_source()
        assert (source["minzoom"], source["maxzoom"], source["attribution"]) == (
            4,
            14,
            "© Example",
        ), source

    def test_naming_neither_is_refused_at_construction(self):
        """A vector source needs a tile set; with neither given there is nothing to draw."""
        with pytest.raises(ValueError, match=r"^vector_tiles\(\) takes exactly one of"):
            VectorTileSource()

    def test_naming_both_is_refused_at_construction(self):
        """A template and a TileJSON are two ways to say the same thing; giving both is ambiguous."""
        with pytest.raises(ValueError, match=r"^vector_tiles\(\) takes exactly one of"):
            VectorTileSource(tiles=ROADS, url="https://tiles.example.org/roads.json")


class TestAnMvtSourceAndLayerAreDescribed:
    """A vector-tile call records one layer whose source is the tile set, drawn from its description."""

    def test_a_line_layer_draws_the_named_source_layer(self):
        """A roads tile set is a ``vector`` source and a ``line`` layer reading one source-layer out of it."""
        m = WebMap().vector_tiles(
            VectorTileSource(tiles=ROADS),
            source_layer="roads",
            name="roads",
        )
        drawn = m._renderer.drawn["roads"]
        assert drawn.source_spec["type"] == "vector", drawn.source_spec
        assert drawn.layer.type == "line", drawn.layer.type

    def test_the_source_layer_is_carried_onto_the_maplibre_layer(self):
        """A vector source holds many named layers, so the one to draw is named on the layer itself."""
        m = WebMap().vector_tiles(VectorTileSource(tiles=ROADS), source_layer="roads")
        drawn = m._renderer.drawn[m.layer_ids[-1]]
        assert drawn.layer.source_layer == "roads", drawn.layer.source_layer

    def test_the_tile_template_reaches_the_source(self):
        """The ``{z}/{x}/{y}`` template the caller passed is what the source requests tiles from."""
        m = WebMap().vector_tiles(VectorTileSource(tiles=ROADS), source_layer="roads")
        source = m._renderer.drawn[m.layer_ids[-1]].source_spec
        assert source["tiles"] == [ROADS], source

    def test_the_recorded_kind_is_vector_tiles(self):
        """The figure describes the layer as a vector-tile layer, not as a plain line.

        Test scenario:
            The kind is what a dispatcher, a layer switcher and ``replace_layer`` read; a tile layer
            labelled ``lines`` would be offered a GeoJSON replacement it cannot take.
        """
        m = WebMap().vector_tiles(
            VectorTileSource(tiles=ROADS), source_layer="roads", name="r"
        )
        assert m.get_layer("r").kind == "vector_tiles", m.get_layer("r")

    def test_the_call_chains(self):
        """Every builder returns the map, so a tile layer composes with the rest of the chain."""
        m = WebMap()
        returned = m.vector_tiles(VectorTileSource(tiles=ROADS), source_layer="roads")
        assert returned is m

    def test_a_polygon_tile_set_is_a_fill_layer(self):
        """``geometry="fill"`` draws a building/landuse tile set as a fill layer."""
        m = WebMap().vector_tiles(
            VectorTileSource(tiles=ROADS),
            source_layer="buildings",
            geometry="fill",
        )
        assert m._renderer.drawn[m.layer_ids[-1]].layer.type == "fill"

    def test_a_point_tile_set_is_a_circle_layer(self):
        """``geometry="circle"`` draws a POI tile set as a circle layer."""
        m = WebMap().vector_tiles(
            VectorTileSource(tiles=ROADS),
            source_layer="pois",
            geometry="circle",
        )
        assert m._renderer.drawn[m.layer_ids[-1]].layer.type == "circle"


class TestTheSourceIsNamedOneOfTwoWays:
    """A vector source is reached by a tile template or a TileJSON ``url``, and by exactly one of them."""

    def test_a_tilejson_url_is_recorded_as_a_url_not_a_template(self):
        """Passing ``url=`` names a TileJSON the service publishes, so the source carries ``url``."""
        m = WebMap().vector_tiles(
            VectorTileSource(url="https://tiles.example.org/roads.json"),
            source_layer="roads",
        )
        source = m._renderer.drawn[m.layer_ids[-1]].source_spec
        assert source["url"] == "https://tiles.example.org/roads.json", source
        assert "tiles" not in source, source

    def test_an_unknown_geometry_is_refused(self):
        """A vector tile is drawn as a point, line or area; nothing else has a MapLibre layer type."""
        mapped = WebMap()
        source = VectorTileSource(tiles=ROADS)
        with pytest.raises(ValueError, match="geometry="):
            mapped.vector_tiles(source, source_layer="roads", geometry="raster")


class TestTheSourceCarriesItsCoverage:
    """The optional zoom range and attribution reach the MapLibre source through the value object."""

    def test_the_zoom_range_reaches_the_source(self):
        """A service that serves only some zooms says so, so MapLibre over-zooms rather than 404s."""
        m = WebMap().vector_tiles(
            VectorTileSource(tiles=ROADS, min_zoom=4, max_zoom=14),
            source_layer="roads",
        )
        source = m._renderer.drawn[m.layer_ids[-1]].source_spec
        assert (source["minzoom"], source["maxzoom"]) == (4, 14), source

    def test_the_attribution_reaches_the_source(self):
        """A tile service's attribution is shown in the map's attribution control."""
        m = WebMap().vector_tiles(
            VectorTileSource(tiles=ROADS, attribution="© Example"),
            source_layer="roads",
        )
        source = m._renderer.drawn[m.layer_ids[-1]].source_spec
        assert source["attribution"] == "© Example", source


class TestTheLayerIsStyled:
    """The colour and the geometry-specific size land on the MapLibre paint the layer draws with."""

    def test_the_colour_reaches_the_line_paint(self):
        """A caller's colour is what the roads are drawn in."""
        m = WebMap().vector_tiles(
            VectorTileSource(tiles=ROADS),
            source_layer="roads",
            color="#ff0000",
        )
        assert m._renderer.drawn[m.layer_ids[-1]].layer.paint["line-color"] == "#ff0000"

    def test_the_width_reaches_the_line_paint(self):
        """``width=`` is the line width in pixels."""
        m = WebMap().vector_tiles(
            VectorTileSource(tiles=ROADS),
            source_layer="roads",
            width=3.0,
        )
        assert m._renderer.drawn[m.layer_ids[-1]].layer.paint["line-width"] == 3.0


class TestTheTileLayerRoundTrips:
    """A figure holding a vector-tile layer is drawable from its description alone."""

    def test_the_layer_redraws_from_the_figure(self):
        """The renderer has a drawer for the kind, and it rebuilds the recorded source and type.

        Test scenario:
            The source is the tile URL in the description, not a feature collection in the figure's
            sources — so a redraw that only reopened sources would find nothing. Drawing from the
            description is what lets a saved page carry the tile layer.
        """
        m = WebMap().vector_tiles(
            VectorTileSource(tiles=ROADS), source_layer="roads", name="r"
        )
        redrawn = m._renderer.draw_layer(m.figure_spec, "r")
        assert redrawn.layer.type == "line", redrawn.layer.type
        assert redrawn.source_spec["type"] == "vector", redrawn.source_spec

    def test_the_source_type_is_emitted_to_the_page(self):
        """The exported page carries the vector source, so a browser can request the tiles."""
        m = WebMap().vector_tiles(VectorTileSource(tiles=ROADS), source_layer="roads")
        assert '"vector"' in _payload(m.to_html())

    def test_a_hidden_layer_is_described_hidden(self):
        """``visible=False`` builds the layer hidden and describes it hidden, so a switcher can show it."""
        m = WebMap().vector_tiles(
            VectorTileSource(tiles=ROADS),
            source_layer="roads",
            visible=False,
        )
        layout = m._renderer.drawn[m.layer_ids[-1]].layer.layout
        assert layout["visibility"] == "none", layout
