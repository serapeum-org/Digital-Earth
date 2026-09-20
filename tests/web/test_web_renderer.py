"""The web renderer's own API: reconciling two figures, and refusing what it cannot draw (#296).

`tests/web/test_web_seam.py` checks the seam through the map — that a description is what gets drawn.
These reach the renderer directly, for the paths a builder does not take: reconciling an arbitrary pair of
figures, removing and re-showing a layer, and the guards that turn a malformed description into a message
rather than a `KeyError` from somewhere far away.
"""

from dataclasses import replace as with_fields

import pytest

from digitalearth.base.spec import LayerSpec, Symbology
from digitalearth.web.renderer import DRAWN_KINDS, Renderer, drawer_for, required_props

pytest.importorskip("maplibre", reason="the web tier needs the web environment")


@pytest.fixture
def points_gdf():
    """Return a small point collection in EPSG:4326.

    Returns:
        A two-point GeoDataFrame.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    return gpd.GeoDataFrame(
        {"value": [1.0, 2.0]},
        geometry=[Point(4.9, 52.4), Point(5.1, 52.1)],
        crs=4326,
    )


@pytest.fixture
def drawn_map(points_gdf):
    """Return a map with one drawn point layer.

    Args:
        points_gdf: The features to draw.

    Returns:
        The map.
    """
    from digitalearth.web import WebMap

    return WebMap().points(points_gdf, name="obs")


class TestADescriptionItCannotDrawIsRefusedByName:
    """A bare `KeyError` on a MapLibre key names neither the layer nor what is wrong with it."""

    def test_a_layer_with_no_symbology_names_itself_and_what_is_missing(self):
        """A `LayerSpec` can reach a drawer without what its builder would have recorded."""
        with pytest.raises(ValueError, match="cannot be drawn by the web tier"):
            required_props(LayerSpec("x", "points"), "paint")

    def test_the_message_names_the_layer_its_kind_and_the_missing_props(self):
        """Each of the three is something the reader needs to find the malformed description."""
        with pytest.raises(ValueError) as caught:
            required_props(LayerSpec("wells", "polygons"), "paint", "maplibre_type")
        message = str(caught.value)
        assert "'wells'" in message, message
        assert "polygons" in message, message
        assert "maplibre_type" in message, message

    def test_a_description_that_carries_them_is_returned_as_a_plain_dict(self):
        """The guard must not refuse what it exists to let through."""
        layer = LayerSpec("x", "points", symbology=Symbology(props={"paint": {"a": 1}}))
        assert required_props(layer, "paint") == {"paint": {"a": 1}}


class TestAKindThisTierDoesNotDraw:
    """A figure written for another backend should say so."""

    def test_it_is_refused_by_name_and_told_what_is_drawn(self):
        """The message lists the kinds, so the caller can see what they meant."""
        with pytest.raises(KeyError, match="does not draw 'point_cloud'"):
            drawer_for("point_cloud")

    def test_the_drawer_table_is_held_against_the_declared_kinds(self, monkeypatch):
        """Drift either way is a defect in this module, not in the caller.

        Args:
            monkeypatch: Used to make the two lists disagree.

        Test scenario:
            A kind in the table and not in `DRAWN_KINDS` would be refused with a message that is false; one
            in `DRAWN_KINDS` with no drawer would raise a bare `KeyError` where the message belongs.
        """
        from digitalearth.web import renderer

        monkeypatch.setattr(renderer, "DRAWN_KINDS", ("points", "nonesuch"))
        with pytest.raises(KeyError, match="drawer table and DRAWN_KINDS disagree"):
            renderer.drawer_for("points")


class TestReconcilingTwoFigures:
    """`apply` is how a map moves from one description to another."""

    def test_a_removed_layer_is_no_longer_drawn(self, drawn_map):
        """What the figure drops, the renderer drops.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        # The panel lists the layers it shows, so dropping a layer drops it from both — a figure that
        # named a layer it no longer holds is refused by the spec, which is the point of that check.
        panel = with_fields(figure.panels[0], layers=())
        emptied = with_fields(
            figure, layers=figure.layers.remove("obs"), panels=(panel,)
        )
        drawn_map._renderer.apply(figure, emptied)
        assert drawn_map._renderer.drawn == {}, drawn_map._renderer.drawn

    def test_a_layer_whose_data_changed_is_drawn_again(self, drawn_map, points_gdf):
        """A new source means a new MapLibre source, so the layer is rebuilt rather than restyled.

        Args:
            drawn_map: A map with one drawn layer.
            points_gdf: Used to register a second source to point at.
        """
        figure = drawn_map.figure_spec
        was = drawn_map._renderer.drawn["obs"].layer
        moved = with_fields(figure.layers.get("obs"), kind="lines")
        rebuilt = with_fields(figure, layers=figure.layers.replace(moved))
        drawn_map._renderer.apply(figure, rebuilt)
        assert drawn_map._renderer.drawn["obs"].layer is not was, (
            "the layer was not rebuilt"
        )

    def test_a_label_change_alone_does_not_reach_maplibre(self, drawn_map):
        """`diff` groups a label with a colour; only one of them is drawn.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        was = drawn_map._renderer.drawn["obs"].layer
        renamed = with_fields(figure.layers.get("obs"), label="something else")
        relabelled = with_fields(figure, layers=figure.layers.replace(renamed))
        drawn_map._renderer.apply(figure, relabelled)
        assert drawn_map._renderer.drawn["obs"].layer is was, (
            "a rename redrew the layer"
        )

    def test_a_style_change_does_reach_maplibre(self, drawn_map):
        """The other half of the same guard.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        was = drawn_map._renderer.drawn["obs"].layer
        layer = figure.layers.get("obs")
        paint = {**dict(layer.symbology.props["paint"]), "circle-radius": 11.0}
        restyled = with_fields(
            figure,
            layers=figure.layers.replace(
                with_fields(
                    layer,
                    symbology=Symbology(
                        props={**layer.symbology.props, "paint": paint}
                    ),
                )
            ),
        )
        drawn_map._renderer.apply(figure, restyled)
        drawn = drawn_map._renderer.drawn["obs"].layer
        assert drawn is not was, "a restyle must redraw"
        assert drawn.paint["circle-radius"] == 11.0, drawn.paint

    def test_hiding_and_showing_a_layer_moves_its_visibility(self, drawn_map):
        """A layer switcher toggles what is already drawn rather than rebuilding it.

        Args:
            drawn_map: A map with one drawn layer.
        """
        renderer = drawn_map._renderer
        renderer.set_visible("obs", False)
        assert renderer.drawn["obs"].layer.layout["visibility"] == "none"
        renderer.set_visible("obs", True)
        assert renderer.drawn["obs"].layer.layout["visibility"] == "visible"

    def test_a_figure_that_hides_a_layer_hides_what_was_drawn(self, drawn_map):
        """A layer switcher's toggle is a change to the description, reconciled like any other.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        hidden = with_fields(
            figure,
            layers=figure.layers.replace(
                with_fields(figure.layers.get("obs"), visible=False)
            ),
        )
        drawn_map._renderer.apply(figure, hidden)
        assert drawn_map._renderer.drawn["obs"].layer.layout["visibility"] == "none"

        drawn_map._renderer.apply(hidden, figure)
        assert drawn_map._renderer.drawn["obs"].layer.layout["visibility"] == "visible"

    def test_toggling_a_drawing_that_carries_no_layer_is_quiet(self, drawn_map):
        """A drawer may produce a source and no layer; toggling it must not raise.

        Args:
            drawn_map: A map with one drawn layer.
        """
        from digitalearth.web.renderer import DrawnLayer

        drawn_map._renderer._drawn["headless"] = DrawnLayer(
            source_id=None, source_spec=None, layer=None
        )
        assert drawn_map._renderer.set_visible("headless", False) is None

    def test_showing_a_layer_nothing_drew_is_quiet(self, drawn_map):
        """A declined layer has no drawing to toggle, and toggling it is not an error.

        Args:
            drawn_map: A map with one drawn layer.
        """
        assert drawn_map._renderer.set_visible("nobody", True) is None

    def test_removing_a_layer_nothing_drew_is_quiet(self, drawn_map):
        """So a caller can remove a layer the tier declined to draw.

        Args:
            drawn_map: A map with one drawn layer.
        """
        held = dict(drawn_map._renderer.drawn)
        drawn_map._renderer.remove("nobody")
        assert drawn_map._renderer.drawn == held, (
            "removing an unknown id touched something"
        )


class TestWhatTheRendererReports:
    """The record is read by `WebMap.layers` and by the widget builder."""

    def test_a_layer_not_in_both_figures_is_treated_as_a_redraw(self, drawn_map):
        """`restyled` only names ids both hold, so this is about a direct call.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        added = with_fields(
            figure, layers=figure.layers.add(LayerSpec("later", "points"))
        )
        assert Renderer._reaches_maplibre(figure, added, "later") is True

    def test_the_declared_kinds_are_the_ones_drawn_from_a_description(self):
        """The tuple is the tier's answer to "what do you draw?"."""
        assert "points" in DRAWN_KINDS
        assert "terrain" not in DRAWN_KINDS, (
            "terrain is a deck.gl path, still queued, and must not claim to be described"
        )

    def test_band_for_prefers_the_layer_s_own_band(self, drawn_map):
        """A custom layer's band is the only thing that says where it belongs.

        Args:
            drawn_map: A map whose renderer is under test.
        """
        own = LayerSpec("x", "custom:maplibre", band="underlay")
        assert drawn_map._renderer.band_for(own) == "underlay"

    def test_band_for_falls_back_to_the_kind_s_band(self, drawn_map):
        """Every other kind declares its own band once, at registration.

        Args:
            drawn_map: A map whose renderer is under test.
        """
        from digitalearth.base.registry import band_of

        assert drawn_map._renderer.band_for(LayerSpec("x", "graticule")) == band_of(
            "graticule"
        )
