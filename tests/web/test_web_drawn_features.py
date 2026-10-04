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

#: The same triangle, carrying an ``id`` of its own in its properties — what ``draw.setFeatureProperty`` or an
#: imported feature leaves behind, and what must not displace MapboxDraw's id.
TRIANGLE_WITH_OWN_ID = {
    **TRIANGLE,
    "properties": {"id": "USER-SET"},
}

#: A point carrying its own ``id`` property and **no** draw id — what a caller-supplied widget or a trait set
#: from Python holds, since MapboxDraw itself always writes one.
POINT_WITHOUT_DRAW_ID = {
    "type": "Feature",
    "properties": {"id": "mine", "label": "plot A"},
    "geometry": {"type": "Point", "coordinates": [8.0, 51.0]},
}

#: The same point with the draw id present but ``None``, which is the other way the key can be missing.
POINT_WITH_NULL_DRAW_ID = {**POINT_WITHOUT_DRAW_ID, "id": None}

#: A point with no ``id`` anywhere — neither a draw id nor a property.
POINT_WITH_NO_ID_AT_ALL = {
    "type": "Feature",
    "properties": {"label": "plot B"},
    "geometry": {"type": "Point", "coordinates": [9.0, 52.0]},
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

    def test_a_feature_s_own_id_property_does_not_displace_the_draw_id(self):
        """The ``id`` column is MapboxDraw's id even when the feature carries an ``id`` property of its own.

        Test scenario:
            - Draw one triangle whose properties already hold ``id="USER-SET"`` while MapboxDraw's own id
              for it is ``"a1"``.
            - The ``id`` column is the draw id, which is what the create/update/delete events name; the
              feature's own value is not what a caller matching an event would be handed.
        """
        m = WebMap().measure()
        _draw(m.render(), TRIANGLE_WITH_OWN_ID)
        assert list(m.drawn_features()["id"]) == [TRIANGLE["id"]]

    def test_an_own_id_property_survives_a_feature_with_no_draw_id(self):
        """With no draw id to prefer, the feature's own ``id`` property is the only one there is.

        Test scenario:
            - A point carrying ``id="mine"`` in its properties and no top-level draw id, which is what a
              caller-supplied widget or a trait set from Python holds; MapboxDraw itself always writes one.
            - The merge used to be unconditional, so ``None`` won and the property was destroyed —
              measured: ``id column: [None]``. Before round 1's draw-id fix the property survived, so this
              was a regression, not a gap.
        """
        m = WebMap().measure()
        _draw(m.render(), POINT_WITHOUT_DRAW_ID)
        assert list(m.drawn_features()["id"]) == ["mine"], (
            f"the feature's own id property must survive, got "
            f"{list(m.drawn_features()['id'])}"
        )

    def test_the_other_properties_survive_a_feature_with_no_draw_id(self):
        """Nothing else on the feature is disturbed by the missing draw id.

        Test scenario:
            - The same point. Its ``label`` always came through; this pins that the narrower merge did not
              cost it.
        """
        m = WebMap().measure()
        _draw(m.render(), POINT_WITHOUT_DRAW_ID)
        assert list(m.drawn_features()["label"]) == ["plot A"], (
            f"the other properties must survive, got {list(m.drawn_features()['label'])}"
        )

    def test_a_draw_id_that_is_explicitly_null_does_not_displace_the_property(self):
        """A present-but-``None`` draw id is a missing one, not a value to write.

        Test scenario:
            - The same point with ``"id": None`` beside the geometry rather than the key absent. Reading
              ``feature.get("id")`` cannot tell the two apart, and neither should the merge: both leave the
              property standing.
        """
        m = WebMap().measure()
        _draw(m.render(), POINT_WITH_NULL_DRAW_ID)
        assert list(m.drawn_features()["id"]) == ["mine"], (
            f"an explicit null draw id must not displace the property, got "
            f"{list(m.drawn_features()['id'])}"
        )

    def test_a_feature_with_no_id_anywhere_still_has_an_id_column(self):
        """The ``id`` column is there whatever the feature holds, so no caller has to branch on it.

        Test scenario:
            - A point with neither a draw id nor an ``id`` property. The unconditional merge gave the
              column for free; only writing the key when there is a draw id would have taken it away and
              made ``drawn["id"]`` a ``KeyError`` — the very failure round 1's M5 fix removed from the
              empty path.
            - The column is therefore still declared, holding ``None`` for the row that has no id.
        """
        m = WebMap().measure()
        _draw(m.render(), POINT_WITH_NO_ID_AT_ALL)
        assert list(m.drawn_features()["id"]) == [None], (
            f"the id column must exist even with no id anywhere, got "
            f"{list(m.drawn_features()['id'])}"
        )

    def test_a_drawn_feature_still_prefers_the_draw_id_over_the_property(self):
        """The draw id keeps winning when there is one — the behaviour round 1 fixed stays fixed.

        Test scenario:
            - The triangle carrying ``id="USER-SET"`` in its properties while MapboxDraw's id is ``"a1"``,
              read through the narrower merge rather than the unconditional one.
        """
        m = WebMap().measure()
        _draw(m.render(), TRIANGLE_WITH_OWN_ID)
        assert list(m.drawn_features()["id"]) == ["a1"], (
            f"the draw id must still win, got {list(m.drawn_features()['id'])}"
        )

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

    def test_the_empty_collection_carries_the_columns_the_drawn_one_does(self):
        """``drawn["id"]`` is readable on the empty path too, so a caller need not branch on ``len()``.

        Test scenario:
            - Read one map nobody drew on and a second one holding a single triangle with no properties of
              its own, so the only columns either can have are the geometry and the draw id.
            - The two column sets are equal, and the empty collection's ``id`` column is an empty list
              rather than a ``KeyError``.
        """
        empty = WebMap().measure()
        empty.render()
        drawn_on = WebMap().measure()
        _draw(drawn_on.render(), TRIANGLE)
        assert set(empty.drawn_features().columns) == set(
            drawn_on.drawn_features().columns
        )
        assert list(empty.drawn_features()["id"]) == []

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
