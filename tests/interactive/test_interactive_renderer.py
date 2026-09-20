"""The interactive renderer's own API: reconciling two figures, and refusing what it cannot draw (#300).

`tests/interactive/test_interactive_seam.py` checks the seam through the map — that a description is what
gets drawn. These reach the renderer directly, for the paths a builder does not take: reconciling an
arbitrary pair of figures, rolling back a refusal, and the guards that turn a malformed description into a
message rather than a `KeyError` from somewhere far away.

Unlike the other tiers, this one's drawer table is keyed by kind **and then by recipe**: several builders
draw one kind by different means, and each records how it built its layer under `via`. So there are two
refusals to hold, not one — an unknown kind, and a known kind with an unknown recipe.
"""

from dataclasses import replace as with_fields

import pytest

from digitalearth.base.registry import band_of
from digitalearth.base.spec import LayerSpec, Symbology
from digitalearth.interactive import InteractiveMap
from digitalearth.interactive.renderer import DRAWN_KINDS, Renderer, drawer_for

pytest.importorskip(
    "geoviews", reason="the interactive tier needs the interactive environment"
)


@pytest.fixture
def point_fc():
    """Return a small point collection read through pyramids.

    Returns:
        A `FeatureCollection` of scattered points.
    """
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


@pytest.fixture
def drawn_map(point_fc):
    """Yield a map with one drawn point layer, closed on the way out.

    Args:
        point_fc: The features to draw.

    Yields:
        The map, whose single layer is `points-1`.
    """
    interactive_map = InteractiveMap().points(point_fc)
    yield interactive_map
    interactive_map.close()


def _refused_figure(interactive_map):
    """Return a figure whose second new layer names a kind this tier does not draw.

    The drawable layer comes first so `apply` has to draw something before it refuses — otherwise there is
    nothing for a rollback to undo, and a renderer with no rollback at all passes the check.

    Args:
        interactive_map: The map whose figure is extended.

    Returns:
        A `FigureSpec` holding the map's layers, a copy of the drawn one, and an undrawable one.
    """
    figure = interactive_map.figure_spec
    drawable = figure.layers.get(interactive_map.layer_ids[-1])
    tree = figure.layers.add(with_fields(drawable, id="second")).add(
        LayerSpec("refused", "terrain", source_id=drawable.source_id)
    )
    return with_fields(figure, layers=tree)


class TestAKindThisTierDoesNotDraw:
    """A figure written for another backend should say so, not fail obscurely."""

    def test_it_is_refused_by_name(self):
        """`terrain` is the 3-D tier's; the message names it rather than raising on a dict key."""
        with pytest.raises(KeyError, match="does not draw 'terrain'"):
            drawer_for("terrain")

    def test_the_refusal_lists_the_kinds_it_does_draw(self):
        """The caller needs to see the vocabulary to spot which name they meant."""
        with pytest.raises(KeyError) as caught:
            drawer_for("point_cloud")
        message = str(caught.value)
        assert "points" in message, message
        assert "raster" in message, message

    def test_a_kind_with_no_drawer_is_caught_in_this_module(self, monkeypatch):
        """Drift one way: a name on the list nothing draws.

        Args:
            monkeypatch: Used to add a kind the table has no entry for.

        Test scenario:
            Such a kind would raise a bare `KeyError` on a dict lookup, where the message that names the
            tier and its kinds belongs.
        """
        from digitalearth.interactive import renderer

        monkeypatch.setattr(renderer, "DRAWN_KINDS", DRAWN_KINDS + ("nonesuch",))
        with pytest.raises(KeyError, match="drawer table and DRAWN_KINDS disagree"):
            renderer.drawer_for("points")

    def test_a_drawer_for_an_undeclared_kind_is_caught_too(self, monkeypatch):
        """Drift the other way: a drawer the tier does not admit to having.

        Args:
            monkeypatch: Used to drop a kind the table still draws.

        Test scenario:
            A kind in the table but not in `DRAWN_KINDS` is refused with a message that is simply false —
            the tier says it cannot draw something it has a drawer for.
        """
        from digitalearth.interactive import renderer

        monkeypatch.setattr(renderer, "DRAWN_KINDS", ("points",))
        with pytest.raises(KeyError, match="drawer table and DRAWN_KINDS disagree"):
            renderer.drawer_for("points")


class TestARecipeThisKindDoesNotHave:
    """The second key: a kind says *what* a layer is, `via` says how this tier built it."""

    def test_an_unknown_recipe_is_refused_naming_the_layers_kind(self, drawn_map):
        """A builder routed to a kind without recording how it draws it fails here, by name.

        Args:
            drawn_map: A map whose renderer is under test.
        """
        draw = drawer_for("points")
        layer = LayerSpec("x", "points", symbology=Symbology(props={"via": "nonesuch"}))
        with pytest.raises(KeyError, match="records 'nonesuch' as how it was drawn"):
            draw(drawn_map, None, layer)

    def test_the_refusal_lists_the_recipes_that_kind_has(self, drawn_map):
        """Two builders draw points — a frame of geometry and a datashaded aggregate.

        Args:
            drawn_map: A map whose renderer is under test.
        """
        draw = drawer_for("points")
        layer = LayerSpec("x", "points", symbology=Symbology(props={"via": "nonesuch"}))
        with pytest.raises(KeyError) as caught:
            draw(drawn_map, None, layer)
        message = str(caught.value)
        assert "datashade" in message, message
        assert "geometry" in message, message

    def test_a_recipe_belonging_to_another_kind_is_refused(self, drawn_map):
        """`geometry` draws a vector layer; a raster has four recipes and none of them is that.

        Args:
            drawn_map: A map whose renderer is under test.

        Test scenario:
            One flat table keyed by kind alone would have drawn this — the kind resolves, and the recipe
            would never have been consulted. The two-level table is what makes it a refusal.
        """
        draw = drawer_for("raster")
        layer = LayerSpec("x", "raster", symbology=Symbology(props={"via": "geometry"}))
        with pytest.raises(KeyError, match="'image'"):
            draw(drawn_map, None, layer)

    def test_a_layer_with_no_symbology_at_all_is_refused_the_same_way(self, drawn_map):
        """A `LayerSpec` can reach a drawer without what its builder would have recorded.

        Args:
            drawn_map: A map whose renderer is under test.
        """
        draw = drawer_for("points")
        layer = LayerSpec("x", "points")
        with pytest.raises(KeyError, match="records None as how it was drawn"):
            draw(drawn_map, None, layer)


class TestReconcilingTwoFigures:
    """`apply` is how a map moves from one description to another."""

    def test_a_removed_layer_is_no_longer_drawn(self, drawn_map):
        """What the figure drops, the renderer drops.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        # The panel lists the layers it shows, so dropping a layer drops it from both — a figure that
        # named a layer it no longer holds is refused by the spec.
        panel = with_fields(figure.panels[0], layers=())
        emptied = with_fields(
            figure, layers=figure.layers.remove("points-1"), panels=(panel,)
        )
        drawn_map._renderer.apply(figure, emptied)
        assert drawn_map._renderer.drawn == {}, drawn_map._renderer.drawn

    def test_a_layer_whose_data_changed_is_drawn_again(self, drawn_map):
        """A HoloViews element is a value, so new data means a new element rather than a restyle.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        was = drawn_map._renderer.drawn["points-1"].element
        moved = with_fields(figure.layers.get("points-1"), kind="lines")
        rebuilt = with_fields(figure, layers=figure.layers.replace(moved))
        drawn_map._renderer.apply(figure, rebuilt)
        assert drawn_map._renderer.drawn["points-1"].element is not was, (
            "the layer was not rebuilt"
        )

    def test_a_label_change_alone_does_not_reach_holoviews(self, drawn_map):
        """`diff` groups a label with a colormap; only one of them is drawn.

        Args:
            drawn_map: A map with one drawn layer.

        Test scenario:
            The redraw guard shipped over the wrong collection on another tier, so a layer pointing at new
            data kept its old drawing while the figure advertised the new one. This is the cheap half of
            the same guard: a rename must cost nothing.
        """
        figure = drawn_map.figure_spec
        was = drawn_map._renderer.drawn["points-1"].element
        renamed = with_fields(figure.layers.get("points-1"), label="something else")
        relabelled = with_fields(figure, layers=figure.layers.replace(renamed))
        drawn_map._renderer.apply(figure, relabelled)
        assert drawn_map._renderer.drawn["points-1"].element is was, (
            "a rename redrew the layer"
        )

    def test_a_style_change_does_reach_holoviews(self, drawn_map):
        """The other half of the same guard: what the drawer reads must trigger a redraw.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        was = drawn_map._renderer.drawn["points-1"].element
        layer = figure.layers.get("points-1")
        common = {**dict(layer.symbology.props["common"]), "size": 11.0}
        restyled = with_fields(
            figure,
            layers=figure.layers.replace(
                with_fields(
                    layer,
                    symbology=Symbology(
                        props={**layer.symbology.props, "common": common}
                    ),
                )
            ),
        )
        drawn_map._renderer.apply(figure, restyled)
        drawn = drawn_map._renderer.drawn["points-1"]
        assert drawn.element is not was, "a restyle must redraw"
        assert drawn.style["size"] == 11.0, drawn.style

    def test_an_added_layer_is_drawn(self, drawn_map):
        """The fourth arm of the reconciliation, so the other three are not the whole story.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        added = with_fields(
            figure,
            layers=figure.layers.add(
                LayerSpec(
                    "grid",
                    "graticule",
                    symbology=Symbology(props={"via": "graticule", "step": 15}),
                )
            ),
        )
        drawn_map._renderer.apply(figure, added)
        assert "grid" in drawn_map._renderer.drawn, sorted(drawn_map._renderer.drawn)

    def test_a_layer_only_one_figure_holds_is_treated_as_a_redraw(self, drawn_map):
        """`restyled` names only ids both figures hold, so this is about a direct call.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        added = with_fields(
            figure, layers=figure.layers.add(LayerSpec("later", "points"))
        )
        assert Renderer._reaches_holoviews(figure, added, "later") is True


class TestARefusalLeavesTheRecordAsItWas:
    """`apply` draws layer by layer, so a refusal can come after something was already drawn."""

    def test_a_refused_apply_rolls_the_record_back(self, drawn_map):
        """Rolling back only the caller's description would leave a record no figure owns.

        Args:
            drawn_map: A map with one drawn layer.

        Test scenario:
            The figure below holds a drawable layer *and then* an undrawable one, so the renderer really
            draws before it refuses. The companion test below shows the partial draw is genuinely there
            without the rollback, so this is not passing because nothing happened.
        """
        held = {
            layer_id: id(drawn.element)
            for layer_id, drawn in drawn_map._renderer.drawn.items()
        }
        figure = drawn_map.figure_spec
        refused = _refused_figure(drawn_map)
        with pytest.raises(KeyError):
            drawn_map._renderer.apply(figure, refused)
        after = {
            layer_id: id(drawn.element)
            for layer_id, drawn in drawn_map._renderer.drawn.items()
        }
        assert after == held, (
            f"a refused change left {sorted(after)} behind, was {sorted(held)}"
        )

    def test_the_rollback_is_what_removes_the_partial_draw(self, drawn_map):
        """Without the `try`/`except`, the drawn record keeps a layer the caller's figure does not.

        Args:
            drawn_map: A map with one drawn layer.

        Test scenario:
            `apply` is `_reconcile` wrapped in a rollback. Calling the inner method is the same run with
            the rollback removed, which is how the check above is shown to have teeth: this asserts the
            half-drawn state exists, so restoring the wrapper is what makes it go away.
        """
        figure = drawn_map.figure_spec
        refused = _refused_figure(drawn_map)
        with pytest.raises(KeyError):
            drawn_map._renderer._reconcile(figure, refused)
        assert "second" in drawn_map._renderer.drawn, sorted(drawn_map._renderer.drawn)

    def test_the_map_still_reports_the_figure_it_can_draw(self, drawn_map):
        """A refused change must not leave the map advertising a figure it never drew.

        Args:
            drawn_map: A map with one drawn layer.
        """
        held = drawn_map.figure_spec
        refused = _refused_figure(drawn_map)
        with pytest.raises(KeyError):
            drawn_map._renderer.apply(held, refused)
        assert drawn_map.figure_spec == held, "the map kept a figure it could not draw"


class TestRemovingALayer:
    """`remove` is called for a layer the tier declined to draw as well as for one it drew."""

    def test_removing_an_id_nothing_drew_is_quiet(self, drawn_map):
        """So a caller can remove a declined layer without knowing it was declined.

        Args:
            drawn_map: A map with one drawn layer.
        """
        held = dict(drawn_map._renderer.drawn)
        drawn_map._renderer.remove("nobody")
        assert drawn_map._renderer.drawn == held, (
            "removing an unknown id touched something"
        )

    def test_removing_a_drawn_layer_forgets_it(self, drawn_map):
        """The other half, so the check above is not passing because `remove` does nothing at all.

        Args:
            drawn_map: A map with one drawn layer.
        """
        drawn_map._renderer.remove("points-1")
        assert "points-1" not in drawn_map._renderer.drawn, sorted(
            drawn_map._renderer.drawn
        )

    def test_removing_the_same_layer_twice_is_quiet(self, drawn_map):
        """A reconciliation removes before it redraws, so double removal is an ordinary path.

        Args:
            drawn_map: A map with one drawn layer.
        """
        drawn_map._renderer.remove("points-1")
        drawn_map._renderer.remove("points-1")
        assert drawn_map._renderer.drawn == {}, drawn_map._renderer.drawn


class TestWhereTheRendererPutsALayer:
    """Draw order is asked of the renderer, which asks the kind — unless the layer answers first."""

    def test_band_for_prefers_the_layer_s_own_band(self, drawn_map):
        """`custom:holoviews` names an engine, which says nothing about what the layer draws.

        Args:
            drawn_map: A map whose renderer is under test.
        """
        own = LayerSpec("x", "custom:holoviews", band="underlay")
        assert drawn_map._renderer.band_for(own) == "underlay"

    def test_band_for_falls_back_to_the_kind_s_band(self, drawn_map):
        """Every other kind declares its band once, at registration.

        Args:
            drawn_map: A map whose renderer is under test.
        """
        placed = drawn_map._renderer.band_for(LayerSpec("x", "graticule"))
        assert placed == band_of("graticule"), placed

    def test_the_two_answers_are_not_the_same_answer(self, drawn_map):
        """A `band_for` that ignored the layer would pass both checks above if they agreed.

        Args:
            drawn_map: A map whose renderer is under test.
        """
        asked = LayerSpec("x", "graticule", band="overlay")
        assert drawn_map._renderer.band_for(asked) == "overlay"
        assert band_of("graticule") != "overlay", band_of("graticule")
