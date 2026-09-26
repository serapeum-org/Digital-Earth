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
from digitalearth.interactive.renderer import (
    DRAWN_KINDS,
    Renderer,
    _is_shown,
    _visibility_keywords,
    drawer_for,
)

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


def _all_hidden(figure):
    """Return the same figure with every layer switched off.

    Args:
        figure: The figure to hide.

    Returns:
        A `FigureSpec` whose layers are all `visible=False`, which is what a layer switcher's toggle
        hands `apply`.
    """
    tree = figure.layers
    for layer in list(tree):
        tree = tree.replace(with_fields(layer, visible=False))
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

    def test_a_hidden_layer_is_hidden_on_the_element(self, drawn_map):
        """The fifth arm: a figure that switches a layer off has to reach HoloViews' options.

        Args:
            drawn_map: A map with one drawn layer.

        Test scenario:
            A layer switcher's toggle arrives as a changed description and is reconciled through `apply`.
            `diff` groups it as neither rebuilt nor restyled, so nothing but the `hidden` arm carries it —
            and a reconciliation that skipped that arm left the element drawn with the description saying
            it was off.
        """
        figure = drawn_map.figure_spec
        drawn_map._renderer.apply(figure, _all_hidden(figure))
        assert drawn_map._renderer.is_visible("points-1") is False, (
            "the reconciliation's hidden arm must reach the element"
        )

    def test_a_shown_layer_comes_back(self, drawn_map):
        """The other direction, so a reconciliation that only ever hid could not pass.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        hidden = _all_hidden(figure)
        drawn_map._renderer.apply(figure, hidden)
        drawn_map._renderer.apply(hidden, figure)
        assert drawn_map._renderer.is_visible("points-1") is True, (
            "the reconciliation's shown arm must reach the element too"
        )


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


class TestApplyReachesTheOverlay:
    """`apply` moves the elements `render()` composes, and leaves the figure the map *reports* alone."""

    @staticmethod
    def _emptied(drawn_map):
        """Return the map's figure with its one layer taken out.

        Args:
            drawn_map: A map with one drawn layer.

        Returns:
            The figure, and the panel emptied with it so `FigureSpec` accepts it.
        """
        figure = drawn_map.figure_spec
        panel = with_fields(figure.panels[0], layers=())
        return with_fields(
            figure, layers=figure.layers.remove("points-1"), panels=(panel,)
        )

    def test_a_successful_apply_takes_the_element_out_of_the_overlay(self, drawn_map):
        """A removed layer has to leave the picture, not only the record beside it.

        Args:
            drawn_map: A map with one drawn layer.

        Test scenario:
            `apply` was record-only here until order 23: it reconciled `drawn` and nothing a viewer would
            see, so a figure that removed a layer left `render()` overlaying it (review M1). This is the
            same call, asked of the list the overlay is composed from.
        """
        drawn_map._renderer.apply(drawn_map.figure_spec, self._emptied(drawn_map))
        assert drawn_map.layers == [], drawn_map.layers

    def test_a_successful_apply_moves_the_record_too(self, drawn_map):
        """So the check above is not passing because `apply` emptied the overlay and nothing else.

        Args:
            drawn_map: A map with one drawn layer.
        """
        drawn_map._renderer.apply(drawn_map.figure_spec, self._emptied(drawn_map))
        assert drawn_map._renderer.drawn == {}, drawn_map._renderer.drawn

    def test_a_successful_apply_leaves_the_figure_the_map_reports_alone(
        self, drawn_map
    ):
        """The description is the map's own, installed by `_change` once `apply` has returned.

        Args:
            drawn_map: A map with one drawn layer.

        Test scenario:
            The split matters: `apply` called directly — by a caller composing a figure of their own, or by
            the conformance adapter — must not leave the map describing something nobody asked it to. That
            is what makes "a figure this refuses is never one the map reports" true of the public methods,
            which install the description themselves.
        """
        held = drawn_map.figure_spec
        drawn_map._renderer.apply(held, self._emptied(drawn_map))
        assert drawn_map.figure_spec == held, drawn_map.layer_ids

    def test_a_refused_apply_puts_the_overlay_back(self, drawn_map):
        """The overlay rolls back with the record, or the two disagree about what is drawn.

        Args:
            drawn_map: A map with one drawn layer.

        Test scenario:
            `apply` is not atomic — it draws layer by layer — so a refusal on the second layer has already
            drawn the first. Rolling back only the record would leave that element overlaid under an id no
            figure owns, which is the first defect the shared renderer contract states.
        """
        held = list(drawn_map.layers)
        with pytest.raises(KeyError):
            drawn_map._renderer.apply(drawn_map.figure_spec, _refused_figure(drawn_map))
        assert drawn_map.layers == held, drawn_map.layers


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


class TestAskingWhetherALayerIsDrawn:
    """Review M4 — `is_visible` is the read-back `set_visible` writes, and every tier answers it."""

    def test_a_drawn_layer_reads_back_as_drawn(self, drawn_map):
        """A reader that always said `False` would pass every hiding check on its own.

        Args:
            drawn_map: A map with one drawn layer.
        """
        assert drawn_map._renderer.is_visible("points-1") is True, (
            "a layer nothing hid must read back drawn"
        )

    def test_a_hidden_layer_reads_back_hidden(self, drawn_map):
        """The answer comes from the option the element was drawn with, so hiding must move it.

        Args:
            drawn_map: A map with one drawn layer.
        """
        drawn_map._renderer.set_visible("points-1", False)
        assert drawn_map._renderer.is_visible("points-1") is False, (
            "set_visible must be readable back through is_visible"
        )

    def test_hiding_an_id_nothing_drew_is_quiet(self, drawn_map):
        """A caller may toggle a layer this tier declined to draw, as it may remove one.

        Args:
            drawn_map: A map with one drawn layer.

        Test scenario:
            The three other tiers ignore the id too, and a reconcile reaches `set_visible` for every id
            the description switches — including one whose drawer returned nothing.
        """
        drawn_map._renderer.set_visible("nope", False)
        assert sorted(drawn_map._renderer.drawn) == ["points-1"], (
            f"toggling an unknown id must touch nothing; got {sorted(drawn_map._renderer.drawn)}"
        )

    def test_asking_about_an_id_nothing_drew_is_refused_by_name(self, drawn_map):
        """A layer the overlay does not hold has no visibility, and guessing one would hide a defect.

        Args:
            drawn_map: A map with one drawn layer.

        Test scenario:
            The conformance suite asks this of every tier through `drawn_is_hidden`. Answering `True`
            for an id nothing drew would let a check pass against a layer that was never drawn at all.
        """
        renderer = drawn_map._renderer
        with pytest.raises(
            KeyError, match="nothing is drawn for layer 'nope'"
        ) as refused:
            renderer.is_visible("nope")
        assert "points-1" in str(refused.value), (
            f"the refusal must list the drawn ids; got {refused.value}"
        )


class TestALayerDrawnByAProducerReadsBackHidden:
    """A `DynamicMap` layer must answer `is_visible` about the frames it draws (round 2 `/test`)."""

    def test_a_hidden_dynamic_layer_is_not_reported_as_drawn(self, dataset):
        """Hiding a windowed raster must be visible to the read-back, not only to Bokeh.

        Test scenario:
            `_visibility_keywords` resolves a `DynamicMap` to the type of the element it produces, but the
            read-back then looked the applied options up against the `DynamicMap` itself, whose own style
            kwargs are empty — so the defaults won and a hidden layer answered "drawn". The hide reached
            Bokeh correctly; only the answer was wrong, and wrong exactly for the layers the shared
            contract's visibility check covers.
        """
        interactive_map = InteractiveMap()
        try:
            interactive_map.large_image(dataset, dynamic=True)
            layer_id = interactive_map.layer_ids[-1]
            interactive_map.layers[-1][
                ()
            ]  # realise a frame, which is what carries the style
            interactive_map._renderer.set_visible(layer_id, False)
            assert interactive_map._renderer.is_visible(layer_id) is False, (
                "a hidden producer-drawn layer must read back hidden"
            )
        finally:
            interactive_map.close()

    def test_a_dynamic_layer_left_alone_still_reads_back_drawn(self, dataset):
        """The fix must not make every producer-drawn layer look hidden."""
        interactive_map = InteractiveMap()
        try:
            interactive_map.large_image(dataset, dynamic=True)
            layer_id = interactive_map.layer_ids[-1]
            interactive_map.layers[-1][()]
            assert interactive_map._renderer.is_visible(layer_id) is True, (
                "a layer nobody hid must read back drawn"
            )
        finally:
            interactive_map.close()


class TestAnElementBokehGivesNoWayToHide:
    """The tier says so rather than saying nothing — the silence review M4 reports, one kind narrower."""

    def test_a_tile_element_declares_no_visibility_keyword(self):
        """A `WMTS`' image *is* the basemap, and Bokeh's tile renderer takes no `visible` style.

        Test scenario:
            The keyword list is read off the backend's own option store rather than hard-coded, so a
            HoloViews release that gave tiles a `visible` option would be picked up rather than missed.
        """
        interactive_map = InteractiveMap().tiles("OSM")
        element = interactive_map._renderer.drawn["basemap-1"].element
        keywords = _visibility_keywords(element)
        interactive_map.close()
        assert keywords == (), (
            f"a tile element must offer no way to hide it; got {keywords}"
        )

    def test_hiding_one_warns_instead_of_failing_silently(self, recwarn):
        """A figure that describes a hidden basemap would otherwise be drawn with it showing.

        Args:
            recwarn: pytest's warning recorder.

        Test scenario:
            `set_visible` returned without doing anything and without saying so, so a stored figure that
            switched the basemap off drew it back on with nothing to tell the caller why.
        """
        interactive_map = InteractiveMap().tiles("OSM")
        interactive_map._renderer.set_visible("basemap-1", False)
        interactive_map.close()
        warned = [
            str(w.message) for w in recwarn.list if "no way to hide" in str(w.message)
        ]
        assert warned, (
            f"hiding a tile layer must warn; got {[str(w.message) for w in recwarn.list]}"
        )
        assert "basemap-1" in warned[0], (
            f"the warning must name the layer; got {warned[0]}"
        )

    def test_one_that_cannot_be_hidden_reads_back_drawn(self):
        """An element with no such option is one that is never hidden, which is the honest answer.

        Test scenario:
            Reading `False` for it would make the conformance suite's "is this layer drawn hidden?"
            answer yes for a basemap that is plainly on the screen.
        """
        interactive_map = InteractiveMap().tiles("OSM")
        element = interactive_map._renderer.drawn["basemap-1"].element
        answer = _is_shown(element)
        interactive_map.close()
        assert answer is True, "a layer that cannot be hidden must not read back hidden"

    def test_a_dynamic_layer_with_no_frame_yet_has_no_element_type_to_ask(
        self, dataset
    ):
        """A `DynamicMap` is a producer of elements, so the options belong to what it produces.

        Args:
            dataset: The raster fixture the dynamic layer reads.

        Test scenario:
            `large_image` draws a `DynamicMap`, and `DynamicMap` itself is not in the backend's option
            store — looking its own name up there raises. Until it has produced a frame its `type` is
            `None`, so there is nothing to ask and the honest answer is "no keyword".
        """
        interactive_map = InteractiveMap(crs=4326).large_image(dataset)
        element = interactive_map._renderer.drawn["raster-1"].element
        before_a_frame = _visibility_keywords(element)
        produced = type(element).__name__
        interactive_map.close()
        assert produced == "DynamicMap", (
            f"large_image must draw a DynamicMap; got {produced}"
        )
        assert before_a_frame == (), (
            f"a producer with no frame yet has no element type to ask; got {before_a_frame}"
        )

    def test_a_dynamic_layer_that_has_produced_a_frame_is_asked_about_that_frame(
        self, dataset
    ):
        """Once it has produced one, the keywords are the produced element's, not the producer's.

        Args:
            dataset: The raster fixture the dynamic layer reads.

        Test scenario:
            This is the arm the `DynamicMap` branch exists for: a `DynamicMap` over `Image` takes
            `visible`, and reading the producer's own name off the store would answer "no keyword" for
            a layer that can perfectly well be hidden.
        """
        interactive_map = InteractiveMap(crs=4326).large_image(dataset)
        element = interactive_map._renderer.drawn["raster-1"].element
        element[()]
        after_a_frame = _visibility_keywords(element)
        interactive_map.close()
        assert after_a_frame == ("visible",), (
            f"a realised dynamic layer must be asked about its element; got {after_a_frame}"
        )
