"""The static renderer's own API: reconciling two figures, and refusing what it cannot draw (#303).

``tests/static/test_static_seam.py`` checks the seam through the map — that every builder describes what it
draws. These reach the renderer directly, for the paths a builder does not take: reconciling an arbitrary
pair of figures, rolling a refusal back off the axes, and the guards that turn a malformed description into
a message rather than a `KeyError` from somewhere far away.

Two things make this tier's renderer different from the web and interactive ones, and both are covered here.
Its drawer table is keyed by kind **and then by recipe**, because five builders draw a ``choropleth`` and
four draw a raster field — so there are two refusals to hold, not one. And it **mutates a live axes** rather
than rebuilding one, so a rollback has to take artists off the axes and put the old ones back, not just
restore a dict.
"""

from dataclasses import replace as with_fields

import numpy as np
import pytest
from pyramids.feature import FeatureCollection

from digitalearth.base.registry import _OBJECTS, band_of
from digitalearth.base.spec import LayerSpec, Symbology
from digitalearth.static import Map
from digitalearth.static.capabilities import CAPABILITIES
from digitalearth.static.renderer import (
    DRAWN_KINDS,
    DrawnLayer,
    Renderer,
    drawer_for,
    drawing_opts,
)


@pytest.fixture
def drawn_map(dataset):
    """Yield a map with one drawn raster layer, closed on the way out.

    Args:
        dataset: The raster fixture the layer draws.

    Yields:
        The map, whose single layer is ``raster-1``.
    """
    canvas = Map(crs=dataset.epsg)
    canvas.imshow(dataset)
    yield canvas
    canvas.close()


def _text_layer(layer_id, crs):
    """Return a drawable text layer, the cheapest thing this tier really puts on an axes.

    A *label* rather than a second raster on purpose: it is the cheapest drawable thing this tier has — one
    ``Text``, no reprojection and no glyph — so a rollback test spends its time on the rollback. It used to
    be forced as well: until #313 a second raster took the first one's image off and the rollback had
    nothing recognisable to restore. Layers compose now, and the label is kept for the first reason.

    Args:
        layer_id: The id to give it.
        crs: The map's display CRS, recorded as the label's own so the point needs no reprojection — the
            fixture raster is in a UTM zone, and a lon/lat origin warps to infinity there and would be
            skipped as off-limb.

    Returns:
        The `LayerSpec`.
    """
    return LayerSpec(
        layer_id,
        "text",
        symbology=Symbology(
            props={"via": "text", "lon": 0.0, "lat": 0.0, "s": "here", "crs": crs}
        ),
    )


def _refused_figure(canvas):
    """Return a figure whose second new layer names a kind this tier does not draw.

    The drawable layer comes first so ``apply`` has to draw something before it refuses — otherwise there is
    nothing for a rollback to undo, and a renderer with no rollback at all passes the check.

    Args:
        canvas: The map whose figure is extended.

    Returns:
        A `FigureSpec` holding the map's layers, a drawable text layer, and an undrawable one.

    Note:
        The undrawable layer is put in the label's own band on purpose. `LayerTree.add` files a layer by
        band, and `FigureDiff.added` follows the tree's order — so a ``terrain`` layer, which belongs among
        the data, would otherwise come *before* the label and be refused with nothing yet drawn. The
        rollback checks below would then pass with nothing to roll back.
    """
    figure = canvas.figure_spec
    tree = figure.layers.add(_text_layer("also-drawn", canvas.crs)).add(
        LayerSpec("refused", "terrain", band="overlay")
    )
    return with_fields(figure, layers=tree)


def _artist_ids(canvas):
    """Return the identity of every artist the renderer holds, by layer id.

    A redraw removes a layer and adds it again under the same id, so a set of ids reads identically before
    and after; the artists it built do not.

    Args:
        canvas: The map to read.

    Returns:
        Layer id -> the ids of the artists drawn for it.
    """
    return {
        layer_id: tuple(id(artist) for artist in drawn.artists)
        for layer_id, drawn in canvas._renderer.drawn.items()
    }


class TestAKindThisTierDoesNotDraw:
    """A figure written for another backend should say so, not fail obscurely."""

    def test_it_is_refused_by_name(self):
        """``terrain`` is the 3-D tier's; the message names it rather than raising on a dict key."""
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
        from digitalearth.static import renderer

        monkeypatch.setattr(renderer, "DRAWN_KINDS", DRAWN_KINDS + ("nonesuch",))
        with pytest.raises(KeyError, match="drawer table and DRAWN_KINDS disagree"):
            renderer.drawer_for("raster")

    def test_a_drawer_for_an_undeclared_kind_is_caught_too(self, monkeypatch):
        """Drift the other way: a drawer the tier does not admit to having.

        Args:
            monkeypatch: Used to drop every kind but one while the table still draws them all.

        Test scenario:
            A kind in the table but not in `DRAWN_KINDS` is refused with a message that is simply false —
            the tier says it cannot draw something it has a drawer for.
        """
        from digitalearth.static import renderer

        monkeypatch.setattr(renderer, "DRAWN_KINDS", ("raster",))
        with pytest.raises(KeyError, match="drawer table and DRAWN_KINDS disagree"):
            renderer.drawer_for("raster")


class TestARecipeThisKindDoesNotHave:
    """The second key: a kind says *what* a layer is, ``via`` says how this tier built it."""

    def test_an_unknown_recipe_is_refused_naming_the_layers_kind(self, drawn_map):
        """A builder routed to a kind without recording how it draws it fails here, by name.

        Args:
            drawn_map: A map whose renderer is under test.
        """
        draw = drawer_for("choropleth")
        layer = LayerSpec(
            "x", "choropleth", symbology=Symbology(props={"via": "nonesuch"})
        )
        with pytest.raises(KeyError, match="records 'nonesuch' as how it was drawn"):
            draw(drawn_map, None, layer)

    def test_the_refusal_lists_the_recipes_that_kind_has(self, drawn_map):
        """Five builders draw a choropleth; the message has to show which names are available.

        Args:
            drawn_map: A map whose renderer is under test.
        """
        draw = drawer_for("choropleth")
        layer = LayerSpec(
            "x", "choropleth", symbology=Symbology(props={"via": "nonesuch"})
        )
        with pytest.raises(KeyError) as caught:
            draw(drawn_map, None, layer)
        message = str(caught.value)
        assert "voronoi" in message, message
        assert "quadtree" in message, message

    def test_a_recipe_belonging_to_another_kind_is_refused(self, drawn_map):
        """``imshow`` draws a raster; a choropleth has five recipes and none of them is that.

        Args:
            drawn_map: A map whose renderer is under test.

        Test scenario:
            One flat table keyed by kind alone would have drawn this — the kind resolves, and the recipe
            would never have been consulted. The two-level table is what makes it a refusal.
        """
        draw = drawer_for("choropleth")
        layer = LayerSpec(
            "x", "choropleth", symbology=Symbology(props={"via": "imshow"})
        )
        with pytest.raises(KeyError, match="'choropleth'"):
            draw(drawn_map, None, layer)

    def test_a_layer_with_no_symbology_at_all_is_refused_the_same_way(self, drawn_map):
        """A `LayerSpec` can reach a drawer without what its builder would have recorded.

        Args:
            drawn_map: A map whose renderer is under test.
        """
        draw = drawer_for("raster")
        with pytest.raises(KeyError, match="records None as how it was drawn"):
            draw(drawn_map, None, LayerSpec("x", "raster"))


class TestTheOptionsADrawerReads:
    """A symbology freezes what it is given; a drawer has to hand cleopatra back what the caller wrote."""

    def test_a_recorded_list_is_a_list_again(self):
        """HoloViews is not the only engine that reads a tuple as something else than a list."""
        layer = LayerSpec(
            "x", "raster", symbology=Symbology(props={"opts": {"levels": [1, 2, 3]}})
        )
        assert drawing_opts(layer) == {"levels": [1, 2, 3]}

    def test_a_layer_that_recorded_none_reads_as_empty(self):
        """A decoration layer records no styling at all, and a drawer must not have to check."""
        assert drawing_opts(LayerSpec("x", "graticule")) == {}

    def test_what_comes_back_is_a_copy(self):
        """A drawer mutates its options the way a builder used to mutate the caller's keywords."""
        layer = LayerSpec(
            "x", "raster", symbology=Symbology(props={"opts": {"cmap": "viridis"}})
        )
        first = drawing_opts(layer)
        first["cmap"] = "plasma"
        assert drawing_opts(layer)["cmap"] == "viridis", "the description was mutated"


class TestReconcilingTwoFigures:
    """``apply`` is how a map moves from one description to another."""

    def test_a_removed_layer_is_no_longer_drawn(self, drawn_map):
        """What the figure drops, the renderer drops — off the axes as well as out of its record.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        # The panel lists the layers it shows, so dropping a layer drops it from both — a figure that
        # named a layer it no longer holds is refused by the spec.
        panel = with_fields(figure.panels[0], layers=())
        emptied = with_fields(
            figure, layers=figure.layers.remove("raster-1"), panels=(panel,)
        )
        drawn_map._renderer.apply(figure, emptied)
        assert drawn_map._renderer.drawn == {}, drawn_map._renderer.drawn
        assert len(drawn_map.ax.images) == 0, "the artist is still on the axes"

    def test_a_removed_layer_leaves_the_colorbar_registry(self, drawn_map):
        """`Scene.layers` is what a colorbar is keyed to, so a stale entry would key it to nothing.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        panel = with_fields(figure.panels[0], layers=())
        emptied = with_fields(
            figure, layers=figure.layers.remove("raster-1"), panels=(panel,)
        )
        drawn_map._renderer.apply(figure, emptied)
        assert drawn_map.layers == [], drawn_map.layers

    def test_a_layer_whose_data_changed_is_drawn_again(self, drawn_map):
        """New data means new artists: a cleopatra glyph bakes its values into what it built.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        was = drawn_map._renderer.drawn["raster-1"].artist
        moved = with_fields(
            figure.layers.get("raster-1"),
            kind="mesh",
            symbology=Symbology(
                props={
                    **dict(figure.layers.get("raster-1").symbology.props),
                    "via": "pcolormesh",
                }
            ),
        )
        rebuilt = with_fields(figure, layers=figure.layers.replace(moved))
        drawn_map._renderer.apply(figure, rebuilt)
        assert drawn_map._renderer.drawn["raster-1"].artist is not was, (
            "the layer was not rebuilt"
        )

    def test_a_label_change_alone_does_not_reach_matplotlib(self, drawn_map):
        """``diff`` groups a label with a colormap; only one of them is drawn.

        Args:
            drawn_map: A map with one drawn layer.

        Test scenario:
            The redraw guard shipped over the wrong collection on another tier, so a layer pointing at new
            data kept its old drawing while the figure advertised the new one. This is the cheap half of
            the same guard: a rename must cost nothing.
        """
        figure = drawn_map.figure_spec
        was = drawn_map._renderer.drawn["raster-1"].artist
        renamed = with_fields(figure.layers.get("raster-1"), label="something else")
        relabelled = with_fields(figure, layers=figure.layers.replace(renamed))
        drawn_map._renderer.apply(figure, relabelled)
        assert drawn_map._renderer.drawn["raster-1"].artist is was, (
            "a rename redrew the layer"
        )

    def test_a_style_change_does_reach_matplotlib(self, drawn_map):
        """The other half of the same guard: what the drawer reads must trigger a redraw.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        was = drawn_map._renderer.drawn["raster-1"].artist
        layer = figure.layers.get("raster-1")
        restyled = with_fields(
            figure,
            layers=figure.layers.replace(
                with_fields(
                    layer,
                    symbology=Symbology(
                        props={**dict(layer.symbology.props), "cmap": "plasma"}
                    ),
                )
            ),
        )
        drawn_map._renderer.apply(figure, restyled)
        drawn = drawn_map._renderer.drawn["raster-1"]
        assert drawn.artist is not was, "a restyle must redraw"
        assert drawn.artist.get_cmap().name == "plasma", drawn.artist.get_cmap().name

    def test_an_added_layer_is_drawn(self, drawn_map):
        """The fourth arm of the reconciliation, so the other three are not the whole story.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        added = with_fields(
            figure, layers=figure.layers.add(_text_layer("label", drawn_map.crs))
        )
        drawn_map._renderer.apply(figure, added)
        assert "label" in drawn_map._renderer.drawn, sorted(drawn_map._renderer.drawn)

    def test_a_hidden_layer_is_hidden_on_the_axes(self, drawn_map):
        """Visibility is the one change that does not rebuild anything.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        hidden = with_fields(
            figure, layers=figure.layers.set_visible("raster-1", False)
        )
        drawn_map._renderer.apply(figure, hidden)
        assert drawn_map.ax.images[-1].get_visible() is False

    def test_a_shown_layer_comes_back(self, drawn_map):
        """And the way back, so the toggle is not one-directional.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        hidden = with_fields(
            figure, layers=figure.layers.set_visible("raster-1", False)
        )
        drawn_map._renderer.apply(figure, hidden)
        drawn_map._renderer.apply(hidden, figure)
        assert drawn_map.ax.images[-1].get_visible() is True

    def test_a_layer_only_one_figure_holds_is_treated_as_a_redraw(self, drawn_map):
        """``restyled`` names only ids both figures hold, so this is about a direct call.

        Args:
            drawn_map: A map with one drawn layer.
        """
        figure = drawn_map.figure_spec
        added = with_fields(
            figure, layers=figure.layers.add(LayerSpec("later", "points"))
        )
        assert Renderer._reaches_matplotlib(figure, added, "later") is True


class TestARefusalLeavesTheAxesAsItFoundThem:
    """``apply`` draws layer by layer, so a refusal can come after something was already drawn."""

    def test_a_refused_apply_rolls_the_record_back(self, drawn_map):
        """Rolling back only the caller's description would leave a record no figure owns.

        Args:
            drawn_map: A map with one drawn layer.

        Test scenario:
            The figure below holds a drawable layer *and then* an undrawable one, so the renderer really
            draws before it refuses. The companion test below shows the partial draw is genuinely there
            without the rollback, so this is not passing because nothing happened.
        """
        held = _artist_ids(drawn_map)
        figure = drawn_map.figure_spec
        with pytest.raises(KeyError):
            drawn_map._renderer.apply(figure, _refused_figure(drawn_map))
        after = _artist_ids(drawn_map)
        assert after == held, (
            f"a refused change left {sorted(after)} behind, was {sorted(held)}"
        )

    def test_a_refused_apply_takes_its_artists_off_the_axes(self, drawn_map):
        """This tier mutates a live axes, so a rollback the web tier can do with a dict is not enough.

        Args:
            drawn_map: A map with one drawn layer.

        Test scenario:
            The refused figure's drawable half is a text label, so the count below is the count of what
            the refused change put on the axes and nothing else.
        """
        held = len(drawn_map.ax.texts)
        figure = drawn_map.figure_spec
        with pytest.raises(KeyError):
            drawn_map._renderer.apply(figure, _refused_figure(drawn_map))
        assert len(drawn_map.ax.texts) == held, (
            f"{len(drawn_map.ax.texts)} labels on the axes, {held} before the refusal"
        )

    def test_the_rollback_is_what_removes_the_partial_draw(self, drawn_map):
        """Without the ``try``/``except``, the record and the axes keep a layer the figure does not.

        Args:
            drawn_map: A map with one drawn layer.

        Test scenario:
            ``apply`` is ``_reconcile`` wrapped in a rollback. Calling the inner method is the same run
            with the rollback removed, which is how the check above is shown to have teeth: this asserts
            the half-drawn state exists, so restoring the wrapper is what makes it go away.
        """
        figure = drawn_map.figure_spec
        with pytest.raises(KeyError):
            drawn_map._renderer._reconcile(figure, _refused_figure(drawn_map), [])
        assert "also-drawn" in drawn_map._renderer.drawn, sorted(
            drawn_map._renderer.drawn
        )

    def test_the_map_still_reports_the_figure_it_can_draw(self, drawn_map):
        """A refused change must not leave the map advertising a figure it never drew.

        Args:
            drawn_map: A map with one drawn layer.
        """
        held = drawn_map.figure_spec
        with pytest.raises(KeyError):
            drawn_map._renderer.apply(held, _refused_figure(drawn_map))
        assert drawn_map.figure_spec == held, "the map kept a figure it could not draw"

    def test_a_restored_layer_is_drawn_again_rather_than_re_attached(self, drawn_map):
        """Matplotlib unlinks a removed artist, so a rollback of a *removal* has to redraw it.

        Args:
            drawn_map: A map with one drawn layer.

        Test scenario:
            The refused figure below removes the drawn layer and then names one this tier cannot draw, so
            the rollback has to put back something it has already taken off the axes.
        """
        figure = drawn_map.figure_spec
        panel = with_fields(figure.panels[0], layers=())
        refused = with_fields(
            figure,
            panels=(panel,),
            layers=figure.layers.remove("raster-1").add(
                LayerSpec("refused", "terrain")
            ),
        )
        with pytest.raises(KeyError):
            drawn_map._renderer.apply(figure, refused)
        assert "raster-1" in drawn_map._renderer.drawn, sorted(
            drawn_map._renderer.drawn
        )
        assert len(drawn_map.ax.images) == 1, len(drawn_map.ax.images)

    def test_a_refused_builder_call_describes_nothing(self, dataset, monkeypatch):
        """The same guarantee one level up: a drawer that raises must not leave a named layer behind.

        Args:
            dataset: The raster the builder draws.
            monkeypatch: Used to make the drawer fail the way a broken description would.
        """
        from digitalearth.static.maps import raster

        def refuse(_scene, _data, _layer):
            """Refuse to draw, whatever it is handed.

            Args:
                _scene: The map being drawn on, unread.
                _data: The raster, unread.
                _layer: The layer's description, unread.

            Raises:
                ValueError: always.
            """
            raise ValueError("this description cannot be drawn")

        monkeypatch.setattr(raster, "draw_field", refuse)
        canvas = Map(crs=dataset.epsg)
        with pytest.raises(ValueError, match="cannot be drawn"):
            canvas.imshow(dataset)
        assert canvas.layer_ids == [], canvas.layer_ids
        canvas.close()


class TestRemovingALayer:
    """Removal is the one operation that has to reach three places at once."""

    def test_removing_an_id_nothing_drew_is_quiet(self, drawn_map):
        """A caller may remove a layer the tier declined to draw.

        Args:
            drawn_map: A map with one drawn layer.
        """
        drawn_map._renderer.remove("never-drawn")
        assert sorted(drawn_map._renderer.drawn) == ["raster-1"]

    def test_removing_the_same_layer_twice_is_quiet(self, drawn_map):
        """Removal is idempotent, so a reconcile that overlaps a manual removal is safe.

        Args:
            drawn_map: A map with one drawn layer.
        """
        drawn_map._renderer.remove("raster-1")
        drawn_map._renderer.remove("raster-1")
        assert drawn_map._renderer.drawn == {}

    def test_removing_a_layer_forgets_the_data_it_registered(self, drawn_map):
        """The object table is process-global and holds strong references.

        Args:
            drawn_map: A map with one drawn layer.
        """
        uri = drawn_map.figure_spec.sources["raster-1"].uri
        key = uri.split(":", 1)[1]
        assert key in _OBJECTS, "the raster was never registered"
        drawn_map._renderer.remove("raster-1")
        assert key not in _OBJECTS, f"{key} is still registered"

    def test_a_reconcile_keeps_the_data_until_the_change_is_through(self, drawn_map):
        """A rollback redraws from the old figure, so its sources cannot be dropped as it goes.

        Args:
            drawn_map: A map with one drawn layer.

        Test scenario:
            The refused figure removes the drawn layer, so the removal really runs before the refusal —
            and the rollback that follows has to be able to open the source again.
        """
        uri = drawn_map.figure_spec.sources["raster-1"].uri
        figure = drawn_map.figure_spec
        panel = with_fields(figure.panels[0], layers=())
        refused = with_fields(
            figure,
            panels=(panel,),
            layers=figure.layers.remove("raster-1").add(
                LayerSpec("refused", "terrain")
            ),
        )
        with pytest.raises(KeyError):
            drawn_map._renderer.apply(figure, refused)
        assert uri.split(":", 1)[1] in _OBJECTS, "the rollback could not have redrawn"


class TestTheSceneLetsGoOfWhatItRegistered:
    """A tier that registers a caller's data must forget it, or it leaks for the life of the process."""

    def test_closing_forgets_every_source_the_scene_held(self, dataset):
        """`close` is the caller saying they are finished with the figure's data.

        Args:
            dataset: The raster drawn and then forgotten.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.imshow(dataset)
        key = canvas.figure_spec.sources["raster-1"].uri.split(":", 1)[1]
        canvas.close()
        assert key not in _OBJECTS, f"{key} outlived the scene that registered it"

    def test_closing_twice_is_quiet(self, dataset):
        """Closing is a lifecycle call, and one that refused a second call would be a trap.

        Args:
            dataset: The raster drawn before the two closes.
        """
        canvas = Map(crs=dataset.epsg)
        canvas.imshow(dataset)
        canvas.close()
        canvas.close()
        assert canvas.figure_spec.layers.ids == ("raster-1",)

    def test_a_with_block_closes_the_scene(self, dataset):
        """``__exit__`` closes rather than only shutting the figure, so the two cannot drift.

        Args:
            dataset: The raster drawn inside the block.
        """
        with Map(crs=dataset.epsg) as canvas:
            canvas.imshow(dataset)
            key = canvas.figure_spec.sources["raster-1"].uri.split(":", 1)[1]
        assert key not in _OBJECTS, f"{key} outlived the with block"

    def test_a_with_block_still_closes_the_figure(self, dataset):
        """The behaviour ``__exit__`` had before it grew a second job.

        Args:
            dataset: The raster drawn inside the block.
        """
        import matplotlib.pyplot as plt

        with Map(crs=dataset.epsg) as canvas:
            canvas.imshow(dataset)
            number = canvas.fig.number
        assert plt.fignum_exists(number) is False

    def test_a_scene_that_drew_nothing_closes_quietly(self):
        """A figure with no layers has nothing to forget, and must not make that an error."""
        canvas = Map()
        canvas.close()
        assert canvas.layer_ids == []


class TestWhatADrawerNeedsThatAFigureCannotCarry:
    """A clip boundary is a geometry and a credential is a secret; neither belongs in a symbology."""

    def test_a_clip_boundary_travels_with_the_scene(self):
        """It is held under the layer's id rather than written into the description."""
        features = FeatureCollection.read_file("tests/data/points.geojson")
        canvas = Map(crs=features.epsg)
        canvas.voronoi(features, column="fid", clip=features.geometry.buffer(5000.0))
        held = canvas._layer_keys[canvas.layer_ids[-1]]
        assert held is not None, canvas._layer_keys
        assert (
            "clip"
            not in canvas.figure_spec.layers.get(canvas.layer_ids[-1]).symbology.props
        )
        canvas.close()

    def test_forgetting_the_layer_forgets_its_key(self):
        """The key is the layer's, so it goes when the layer does."""
        features = FeatureCollection.read_file("tests/data/points.geojson")
        canvas = Map(crs=features.epsg)
        canvas.quadtree(features, nmax=1)
        layer_id = canvas.layer_ids[-1]
        canvas._renderer.remove(layer_id)
        assert layer_id not in canvas._layer_keys, canvas._layer_keys
        canvas.close()


class TestWhereTheRendererPutsALayer:
    """``band_for`` is how a layer that names no band still finds its place in the stack."""

    def test_band_for_prefers_the_layer_s_own_band(self, drawn_map):
        """A backdrop is the ``raster`` kind drawn somewhere other than where that kind goes.

        Args:
            drawn_map: A map whose renderer answers the question.
        """
        layer = LayerSpec("x", "raster", band="underlay")
        assert drawn_map._renderer.band_for(layer) == "underlay"

    def test_band_for_falls_back_to_the_kind_s_band(self, drawn_map):
        """Most layers declare none, and the registry is what decides for them.

        Args:
            drawn_map: A map whose renderer answers the question.
        """
        layer = LayerSpec("x", "raster")
        assert drawn_map._renderer.band_for(layer) == band_of("raster")

    def test_the_two_answers_are_not_the_same_answer(self, drawn_map):
        """Otherwise the preference above would be untestable.

        Args:
            drawn_map: A map whose renderer answers the question.
        """
        own = drawn_map._renderer.band_for(LayerSpec("x", "raster", band="underlay"))
        assert own != band_of("raster")


class TestEveryDeclaredKindIsDrawnFromItsDescription:
    """The static tier finished its seam in one step, and that is worth pinning rather than assuming."""

    def test_nothing_is_declared_that_is_not_drawn(self):
        """A kind the tier advertises and cannot draw is a refusal a caller cannot predict."""
        undrawn = sorted(set(CAPABILITIES.kinds) - set(DRAWN_KINDS))
        assert undrawn == [], f"{undrawn} are declared but drawn the old way"

    def test_nothing_is_drawn_that_is_not_declared(self):
        """And the other direction: the refusal message would name a false vocabulary."""
        undeclared = sorted(set(DRAWN_KINDS) - set(CAPABILITIES.kinds))
        assert undeclared == [], f"{undeclared} are drawn without being declared"

    def test_every_kind_resolves_to_a_drawer(self):
        """A kind on the list with no drawer raises a bare `KeyError` far from the cause."""
        unresolved = [kind for kind in DRAWN_KINDS if drawer_for(kind) is None]
        assert unresolved == [], unresolved


class TestWhatADrawerHandsBack:
    """`DrawnLayer` is the one shape every drawer answers in, whatever it put on the axes."""

    def test_a_layer_with_no_artists_is_still_a_drawn_layer(self):
        """A graticule's lines are drawn by the projection frame, long after the layer is."""
        assert DrawnLayer().artists == ()

    def test_hiding_a_layer_with_no_artists_is_quiet(self, drawn_map):
        """So a figure that hides a graticule does not have to special-case it.

        Args:
            drawn_map: A map whose renderer is under test.
        """
        drawn_map._renderer._drawn["empty"] = DrawnLayer()
        drawn_map._renderer.set_visible("empty", False)
        assert "empty" in drawn_map._renderer.drawn

    def test_an_artist_that_is_not_on_the_axes_is_removed_quietly(self, drawn_map):
        """Matplotlib answers a double removal with `ValueError`; a reconcile must survive it.

        Args:
            drawn_map: A map whose renderer is under test.
        """
        stray = np.array([1.0, 2.0])  # not an artist at all
        drawn_map._renderer._drawn["stray"] = DrawnLayer(artist=stray, artists=(stray,))
        drawn_map._renderer.remove("stray")
        assert "stray" not in drawn_map._renderer.drawn
