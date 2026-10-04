"""The geometry a user draws on a web map, handed back to Python (#363).

MapboxDraw runs in the browser, and py-maplibregl syncs what it holds back to the widget's
``draw_feature_collection_all`` trait on every create/update/delete. These tests stand in for the browser by
setting that trait on the real ``MapWidget`` the map rendered, which is exactly what the front-end's sync does, so
nothing here needs a browser or the network.
"""

import pytest

pytest.importorskip("maplibre")

from pyramids.feature import FeatureCollection  # noqa: E402

from digitalearth.web import WebMap  # noqa: E402

#: A triangle as MapboxDraw reports it: an ``id`` beside the geometry, and empty properties.
TRIANGLE = {
    "id": "a1",
    "type": "Feature",
    "properties": {},
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[10.0, 50.0], [12.0, 50.0], [11.0, 52.0], [10.0, 50.0]]],
    },
}

#: A two-vertex path, the shape ``measure(distance=True)`` lets a user draw.
PATH = {
    "id": "b2",
    "type": "Feature",
    "properties": {"note": "river"},
    "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [3.0, 4.0]]},
}


def _draw(widget, *features):
    """Set the widget's synced draw state the way the front-end does after a ``draw.create``.

    Args:
        widget: The ``MapWidget`` the map rendered.
        *features: The GeoJSON features MapboxDraw holds.
    """
    widget.draw_feature_collection_all = {
        "type": "FeatureCollection",
        "features": list(features),
    }


class TestDrawnFeatures:
    """``drawn_features()`` reads the last rendered widget's draw state as a ``FeatureCollection``."""

    def test_a_drawn_polygon_comes_back_as_a_feature_collection(self):
        """One drawn polygon is one row, in lon/lat, with the triangle's own extent."""
        m = WebMap().measure()
        _draw(m.render(), TRIANGLE)
        drawn = m.drawn_features()
        assert isinstance(drawn, FeatureCollection)
        assert len(drawn) == 1
        assert drawn.epsg == 4326
        # The triangle's corners, read off TRIANGLE by hand: x from 10 to 12, y from 50 to 52.
        assert tuple(drawn.geometry.iloc[0].bounds) == (10.0, 50.0, 12.0, 52.0)

    def test_the_draw_id_is_kept_so_events_can_be_matched(self):
        """MapboxDraw's feature id is what the created/updated/deleted events name, so it survives."""
        m = WebMap().measure()
        _draw(m.render(), TRIANGLE, PATH)
        drawn = m.drawn_features()
        assert list(drawn["id"]) == ["a1", "b2"]

    def test_feature_properties_become_columns(self):
        """Properties a feature carries reach the frame as columns."""
        m = WebMap().measure()
        _draw(m.render(), PATH)
        assert m.drawn_features()["note"].iloc[0] == "river"

    def test_a_line_keeps_its_geometry_type(self):
        """A path drawn for distance comes back as a line, not coerced to anything else."""
        m = WebMap().measure()
        _draw(m.render(), PATH)
        assert m.drawn_features().geometry.iloc[0].geom_type == "LineString"

    def test_nothing_drawn_is_an_empty_collection(self):
        """A rendered map nobody drew on returns no rows, still in lon/lat — not an error."""
        m = WebMap().measure()
        m.render()
        drawn = m.drawn_features()
        assert isinstance(drawn, FeatureCollection)
        assert len(drawn) == 0
        assert drawn.epsg == 4326

    def test_everything_deleted_is_an_empty_collection(self):
        """After a ``draw.delete`` of the last shape the front-end syncs an empty collection."""
        m = WebMap().measure()
        widget = m.render()
        _draw(widget, TRIANGLE)
        _draw(widget)
        assert len(m.drawn_features()) == 0

    def test_the_latest_render_is_the_one_read(self):
        """Each render builds a new widget; the one on screen is the last one built."""
        m = WebMap().measure()
        first = m.render()
        _draw(first, TRIANGLE)
        m.render()
        assert len(m.drawn_features()) == 0

    def test_an_explicit_widget_is_read_instead(self):
        """A caller holding a widget from an earlier render can ask about that one."""
        m = WebMap().measure()
        first = m.render()
        _draw(first, PATH)
        m.render()
        assert list(m.drawn_features(first)["id"]) == ["b2"]

    def test_a_map_never_rendered_says_so(self):
        """There is no widget to read before the map is rendered, and the error says what to do."""
        m = WebMap().measure()
        with pytest.raises(RuntimeError, match="render"):
            m.drawn_features()

    def test_something_that_is_not_a_map_widget_is_refused(self):
        """An object without the draw trait is named in a TypeError rather than read as nothing drawn."""
        m = WebMap()
        not_a_widget = object()
        with pytest.raises(TypeError, match="draw_feature_collection_all"):
            m.drawn_features(not_a_widget)
