"""The one ``layer_control`` signature, on the interactive half of it (#264).

``layer_control`` existed on two tiers and shared **not one parameter** between them: the web tier took
``(position, layer_ids, theme)`` and this one ``(opacity, reorder, basemap_switch)``, returning a Panel
``Viewable`` where the web tier returned the map. So no caller could add a layer control without knowing
which backend it was talking to. The settled signature is the layers to include, the position, and the
controls to expose — with **both** tiers returning the map.

Returning the map is the part that took work here, because the thing this tier builds is the object a caller
*displays*. It is kept on the map instead, the way the static tier keeps its ``fig``/``ax`` and this tier
keeps its ``layers``, and read back through :attr:`InteractiveMap.layer_control_panel`. That is also what
makes ``layer_control()`` composable with the rest of the builders for the first time.

``opacity=`` and ``basemap_switch=`` were the same request spelled as two booleans, and are entries in
``controls=`` now. ``reorder=`` is **not** one of them: it names a manipulation this tier cannot do at all,
which is refused rather than exposed, so it stays its own flag.
"""

import pytest

pytest.importorskip("geoviews")
pytest.importorskip("panel")

import panel as pn  # noqa: E402  (after the engine guard)

from digitalearth.interactive import InteractiveMap  # noqa: E402


@pytest.fixture
def point_fc():
    """Return the point fixture as a pyramids ``FeatureCollection``.

    Returns:
        A ``FeatureCollection`` of points in EPSG:32618.
    """
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


@pytest.fixture
def two_layers(dataset, point_fc):
    """Return a Web-Mercator map carrying a raster field and a point overlay.

    Args:
        dataset: The raster fixture.
        point_fc: The point ``FeatureCollection`` fixture.

    Returns:
        An ``InteractiveMap`` with two layers and no layer control yet.
    """
    return InteractiveMap().field(dataset).points(point_fc)


def _toggles(built) -> list:
    """Return the options of the built control's visibility toggle group.

    Args:
        built: The map ``layer_control()`` returned.

    Returns:
        The toggle labels, read off the ``CheckBoxGroup`` inside the panel the map is holding.
    """
    groups = built.layer_control_panel.select(pn.widgets.CheckBoxGroup)
    assert groups, "the control has no visibility toggles"
    return list(groups[0].options)


class TestTheMapComesBack:
    """Returning the map is what let one call be written against either tier (#264)."""

    def test_the_call_returns_the_map_itself(self, two_layers):
        """Not the panel — the panel is held, so the builder can chain like every other one.

        Args:
            two_layers: The map under test.
        """
        assert two_layers.layer_control() is two_layers

    def test_the_control_chains_into_another_builder(self, two_layers, point_fc):
        """The reason the return type mattered: a control mid-chain used to end the chain.

        Args:
            two_layers: The map under test.
            point_fc: A second point layer to add after the control.
        """
        built = two_layers.layer_control().points(point_fc)
        assert len(built.layers) == 3

    def test_the_panel_is_reachable_and_is_a_viewable(self, two_layers):
        """A held panel nobody can read is a panel that was not built.

        Args:
            two_layers: The map under test.
        """
        assert isinstance(
            two_layers.layer_control().layer_control_panel, pn.viewable.Viewable
        )

    def test_a_map_with_no_control_holds_no_panel(self, two_layers):
        """The accessor answers "none built yet" rather than raising or inventing one.

        Args:
            two_layers: The map under test.
        """
        assert two_layers.layer_control_panel is None

    def test_a_second_call_replaces_the_first_panel(self, two_layers):
        """Two calls are one control reconsidered, not two stacked in the same corner.

        Args:
            two_layers: The map under test.
        """
        first = two_layers.layer_control().layer_control_panel
        second = two_layers.layer_control(controls=("visibility",)).layer_control_panel
        assert first is not second

    def test_one_toggle_per_layer_in_draw_order(self, two_layers):
        """What the panel is *for*, unchanged by the new return type.

        Args:
            two_layers: The map under test.
        """
        assert _toggles(two_layers.layer_control()) == ["0: Image", "1: Points"]


class TestTheLayersToInclude:
    """``layers=`` is the Tier-2 name, and it means layer ids on both tiers."""

    def test_only_the_named_layers_get_a_toggle(self, two_layers):
        """A subset hides a layer from the switch, which is what the web tier's ``layers=`` does.

        Args:
            two_layers: The map under test.
        """
        keep = two_layers.layer_ids[0]
        assert _toggles(two_layers.layer_control(layers=[keep])) == ["0: Image"]

    def test_a_layer_left_out_of_the_switch_is_still_drawn(self, two_layers):
        """Hidden from the switch is not hidden from the map — the web tier's own wording.

        Args:
            two_layers: The map under test.

        Test scenario:
            Nothing is ticked, so the only element the composed view can contain is the layer that was
            never offered. One element is therefore the whole claim: it survived a control that does not
            list it.
        """
        keep = two_layers.layer_ids[0]
        two_layers.layer_control(layers=[keep])
        composed = two_layers._compose_visible_layers([], always=(1,))
        assert len(composed.dimension_values(0)) > 0

    def test_an_unknown_layer_is_refused(self, two_layers):
        """A typo would produce a control over nothing rather than an error.

        Args:
            two_layers: The map under test.
        """
        with pytest.raises(ValueError, match="not on this map"):
            two_layers.layer_control(layers=["image-999"])

    def test_a_map_with_no_layers_is_still_refused(self):
        """An empty control in the corner explains nothing — unchanged by the rename."""
        empty = InteractiveMap()
        with pytest.raises(ValueError, match="at least one layer"):
            empty.layer_control()


class TestWherTheControlSits:
    """``position`` is the second shared name; here it places the widget column beside the map."""

    @pytest.mark.parametrize(
        ("position", "controls_first"),
        [("top-left", True), ("bottom-left", True), ("top-right", False)],
    )
    def test_a_left_position_puts_the_widgets_before_the_map(
        self, two_layers, position, controls_first
    ):
        """The corner names the side, which is the only part of a corner a two-cell row can honour.

        Args:
            two_layers: The map under test.
            position: The corner asked for.
            controls_first: Whether the widget column should come first in the row.
        """
        row = two_layers.layer_control(position=position).layer_control_panel
        first_holds_widgets = bool(row[0].select(pn.widgets.CheckBoxGroup))
        assert first_holds_widgets is controls_first

    @pytest.mark.parametrize(
        ("position", "expected"), [("top-right", "start"), ("bottom-right", "end")]
    )
    def test_the_half_of_the_corner_a_row_can_honour_is_the_alignment(
        self, two_layers, position, expected
    ):
        """Top and bottom become the column's own vertical alignment inside the row.

        Args:
            two_layers: The map under test.
            position: The corner asked for.
            expected: The Panel alignment the column must carry.
        """
        row = two_layers.layer_control(position=position).layer_control_panel
        assert row[1].align == expected

    def test_a_position_that_is_not_a_corner_is_refused(self, two_layers):
        """The four corners are the web tier's, and this tier is held to the same four.

        Args:
            two_layers: The map under test.
        """
        with pytest.raises(ValueError, match="control position"):
            two_layers.layer_control(position="middle")


class TestTheControlsToExpose:
    """``controls=`` replaces two booleans that said the same thing in this tier's own words."""

    def test_the_default_offers_every_control_this_tier_can_draw(self, two_layers):
        """Unchanged from the old defaults: toggles, an opacity slider and a basemap switch.

        Args:
            two_layers: The map under test.
        """
        panel_obj = two_layers.layer_control().layer_control_panel
        assert len(panel_obj[1]) == 3

    def test_visibility_alone_builds_no_other_widget(self, two_layers):
        """Asking for one control has to drop the others, or the argument does nothing.

        Args:
            two_layers: The map under test.
        """
        panel_obj = two_layers.layer_control(
            controls=("visibility",)
        ).layer_control_panel
        assert panel_obj.select(pn.widgets.FloatSlider) == []

    def test_opacity_named_builds_the_slider(self, two_layers):
        """The control named is the control built.

        Args:
            two_layers: The map under test.
        """
        panel_obj = two_layers.layer_control(
            controls=("visibility", "opacity")
        ).layer_control_panel
        assert panel_obj.select(pn.widgets.FloatSlider), "opacity slider missing"

    def test_a_control_no_tier_has_is_refused_by_the_vocabulary(self, two_layers):
        """A name outside the shared vocabulary is a typo, and reads as one.

        Args:
            two_layers: The map under test.
        """
        with pytest.raises(ValueError) as refused:
            two_layers.layer_control(controls=("opactiy",))
        assert "opactiy" in str(refused.value), refused.value

    def test_visibility_cannot_be_dropped(self, two_layers):
        """A layer control with no per-layer toggle is not a layer control.

        Args:
            two_layers: The map under test.
        """
        with pytest.raises(ValueError) as refused:
            two_layers.layer_control(controls=("opacity",))
        assert "visibility" in str(refused.value), refused.value


class TestAskingForABasemapSwitchOutright:
    """Naming ``"basemap"`` is a request, and a request this tier cannot honour is refused."""

    def test_an_unprompted_switch_is_dropped_on_a_non_mercator_map(self, dataset):
        """Left to the default the switch is furniture, so a map that cannot take it drops it.

        Args:
            dataset: The raster fixture.
        """
        other = InteractiveMap(crs=4326).field(dataset).layer_control()
        assert len(other.layer_control_panel[1]) == 2

    def test_naming_it_on_a_non_mercator_map_is_refused(self, dataset):
        """The same condition, asked for outright, with the display CRS named.

        Args:
            dataset: The raster fixture.
        """
        other = InteractiveMap(crs=4326).field(dataset)
        with pytest.raises(ValueError) as refused:
            other.layer_control(controls=("visibility", "basemap"))
        assert "crs=3857" in str(refused.value), refused.value

    def test_naming_it_on_a_mercator_map_builds_the_select(self, two_layers):
        """Where the tier can honour the request, it builds the widget.

        Args:
            two_layers: The map under test.
        """
        panel_obj = two_layers.layer_control(
            controls=("visibility", "basemap")
        ).layer_control_panel
        assert panel_obj.select(pn.widgets.Select), "basemap switch missing"


class TestReorderIsNotAControl:
    """``reorder`` names a manipulation this tier cannot do, so it is refused rather than exposed."""

    def test_reorder_true_is_still_refused(self, two_layers):
        """#242 — an inert flag is worse than a missing one, and this one stays refused.

        Args:
            two_layers: The map under test.
        """
        with pytest.raises(NotImplementedError, match="reorder"):
            two_layers.layer_control(reorder=True)

    def test_reorder_is_not_in_the_shared_control_vocabulary(self, two_layers):
        """Naming it through ``controls=`` would read as a control the tier exposes.

        Args:
            two_layers: The map under test.
        """
        with pytest.raises(ValueError) as refused:
            two_layers.layer_control(controls=("visibility", "reorder"))
        assert "reorder" in str(refused.value), refused.value
