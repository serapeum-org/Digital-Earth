"""#396 — the big-data deck.gl layers own a `LayerSpec`, so they are described, rebuilt and portable.

The spec-level builder/registry behaviour (``layer_ids``, remove/hide/reorder) is exercised in
``test_web_bigdata.py``. This file fills the gaps that line coverage alone leaves on the changed code: the
``draw_deck_geojson`` drawer's refusal when a hand-built description carries no ``deck`` template, the caller's
fill/line colours reaching the recorded template through the drawer, and the headline round-trip claim — a
deck layer that names a path survives ``figure_spec``'s ``to_dict``/``from_dict`` and is rebuilt from the
description, the ``draw_deck_geojson`` path a live build never takes.

Lives under ``tests/web/``, run by the ``test-web`` pixi task in the ``web`` env; ``importorskip("maplibre")``.
"""

import pytest

pytest.importorskip("maplibre")


@pytest.fixture
def points_gdf():
    """Three points in lon/lat with a ``value`` column.

    Returns:
        A GeoDataFrame in EPSG:4326 holding point geometries.
    """
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    return gpd.GeoDataFrame(
        {"value": [0.0, 1.0, 2.0]},
        geometry=[Point(x, x) for x in range(3)],
        crs=4326,
    )


@pytest.fixture
def polygons_gdf():
    """Two triangles in lon/lat with a ``pop`` column.

    Returns:
        A GeoDataFrame in EPSG:4326 holding polygon geometries.
    """
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    return gpd.GeoDataFrame(
        {"pop": [1.0, 5.0]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3)]),
        ],
        crs=4326,
    )


class TestDrawDeckGeojsonDescriptionContract:
    """``draw_deck_geojson`` rebuilds from the recorded ``deck`` template, and refuses a description missing it."""

    def test_a_description_without_a_deck_template_is_refused(self, points_gdf):
        """A hand-built `deck_scatter` description carrying no ``deck`` template is refused by layer, not key.

        Test scenario:
            A ``LayerSpec`` of kind ``deck_scatter`` with empty symbology props reaches the drawer the way a
            figure written before #396 recorded the template would. ``draw_deck_geojson`` must raise a
            ``ValueError`` naming the layer and the missing ``deck`` prop (via ``required_props``), rather
            than a bare ``KeyError`` deep in the template merge.
        """
        from digitalearth.base.spec import LayerSpec, Symbology
        from digitalearth.web import WebMap
        from digitalearth.web.bigdata import draw_deck_geojson

        web_map = WebMap()
        bare = LayerSpec("ghost", "deck_scatter", symbology=Symbology(props={}))
        with pytest.raises(
            ValueError, match=r"ghost.*deck_scatter.*none of \['deck'\]"
        ):
            draw_deck_geojson(web_map, points_gdf, bare)


class TestDeckStyleReachesTheRecordedTemplate:
    """The caller's colours/size are recorded on the template and reach the deck.gl layer the drawer rebuilds."""

    def test_deck_scatter_records_fill_colour_and_radius(self, points_gdf):
        """``fill_color``/``size`` reach the GeoJsonLayer's ``getFillColor`` and both radius keys.

        Test scenario:
            ``deck_scatter`` records a template without id/data; the drawer fills those in. The caller's
            RGBA and radius must survive onto the built deck.gl layer dict, not be replaced by defaults.
        """
        from digitalearth.web import WebMap

        m = WebMap().deck_scatter(points_gdf, fill_color=(10, 20, 30, 40), size=7.0)
        layer = m.layers[0]
        assert layer["@@type"] == "GeoJsonLayer", layer
        assert layer["getFillColor"] == [10, 20, 30, 40], layer["getFillColor"]
        assert layer["getPointRadius"] == 7.0, layer["getPointRadius"]
        assert layer["pointRadiusMinPixels"] == 7.0, layer["pointRadiusMinPixels"]

    def test_deck_polygons_records_fill_and_line_colour(self, polygons_gdf):
        """``fill_color``/``line_color`` reach the GeoJsonLayer's ``getFillColor``/``getLineColor``.

        Test scenario:
            ``deck_polygons`` records a stroked/filled template; the caller's two RGBA channels must reach
            the built layer dict, with the outline width the builder sets.
        """
        from digitalearth.web import WebMap

        m = WebMap().deck_polygons(
            polygons_gdf, fill_color=(1, 2, 3, 4), line_color=(5, 6, 7, 8)
        )
        layer = m.layers[0]
        assert layer["@@type"] == "GeoJsonLayer", layer
        assert layer["getFillColor"] == [1, 2, 3, 4], layer["getFillColor"]
        assert layer["getLineColor"] == [5, 6, 7, 8], layer["getLineColor"]
        assert layer["lineWidthMinPixels"] == 1, layer["lineWidthMinPixels"]


class TestDeckLayerSurvivesAFigureRoundTrip:
    """A deck layer that names a path is portable: ``to_dict``/``from_dict`` reload it and redraw from it."""

    @staticmethod
    def _geojson_path(gdf, tmp_path, name):
        """Write ``gdf`` to a GeoJSON file and return its path as a string.

        Args:
            gdf: The GeoDataFrame to serialise.
            tmp_path: pytest's per-test directory.
            name: The file stem.

        Returns:
            The path to the written ``.geojson`` file, as a string.
        """
        out = tmp_path / f"{name}.geojson"
        gdf.to_file(str(out), driver="GeoJSON")
        return str(out)

    def test_deck_scatter_from_a_path_reloads_and_redraws(self, points_gdf, tmp_path):
        """A `deck_scatter` naming a path survives a figure round-trip and is rebuilt by the drawer.

        Test scenario:
            The source is a GeoJSON file (a pyramids object in memory records an ``object:`` reference that
            ``to_dict`` refuses, exactly as ``points(gdf)`` does). ``FigureSpec.from_dict(to_dict())``
            reloads the description, and ``draw_layer`` rebuilds a deck.gl ``GeoJsonLayer`` from it — the
            ``draw_deck_geojson`` path a live build never takes, opening the recorded reference rather than
            the placed frame.
        """
        from digitalearth.base.spec import FigureSpec
        from digitalearth.web import WebMap
        from digitalearth.web.renderer import DECK_ROUTE

        path = self._geojson_path(points_gdf, tmp_path, "pts")
        built = WebMap().deck_scatter(path, size=6.0, name="obs")
        reloaded = FigureSpec.from_dict(built.figure_spec.to_dict())
        assert reloaded.layers.get("obs").kind == "deck_scatter", reloaded.layers.get(
            "obs"
        )

        drawn = WebMap()._renderer.draw_layer(reloaded, "obs")
        assert drawn is not None, "a reloaded deck description must be drawable"
        assert drawn.route == DECK_ROUTE, drawn.route
        assert drawn.layer["@@type"] == "GeoJsonLayer", drawn.layer
        assert drawn.layer["id"] == "obs", drawn.layer
        assert drawn.layer["getPointRadius"] == 6.0, drawn.layer

    def test_deck_polygons_from_a_path_reloads_and_redraws(
        self, polygons_gdf, tmp_path
    ):
        """The polygon counterpart: a path-backed `deck_polygons` reloads and is rebuilt by the drawer.

        Test scenario:
            As for scatter, but a stroked/filled polygon template; the rebuilt layer must carry the id the
            description names and deck.gl's ``GeoJsonLayer`` type.
        """
        from digitalearth.base.spec import FigureSpec
        from digitalearth.web import WebMap
        from digitalearth.web.renderer import DECK_ROUTE

        path = self._geojson_path(polygons_gdf, tmp_path, "areas")
        built = WebMap().deck_polygons(path, name="zones")
        reloaded = FigureSpec.from_dict(built.figure_spec.to_dict())
        assert reloaded.layers.get("zones").kind == "deck_polygons", (
            reloaded.layers.get("zones")
        )

        drawn = WebMap()._renderer.draw_layer(reloaded, "zones")
        assert drawn is not None, "a reloaded deck description must be drawable"
        assert drawn.route == DECK_ROUTE, drawn.route
        assert drawn.layer["@@type"] == "GeoJsonLayer", drawn.layer
        assert drawn.layer["id"] == "zones", drawn.layer
