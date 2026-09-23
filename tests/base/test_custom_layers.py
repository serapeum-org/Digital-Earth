"""A caller's own engine object, described well enough to address it (DE-49, #293).

Every tier lets a caller hand it a PyVista mesh, a HoloViews element or a MapLibre layer. None of them could be
addressed, described or saved around: on the web tier the object never reached the layer tree, so `layer_ids` did
not list it and `remove_layer` raised. These cover the description that replaces that — `custom:<engine>`, a band
the caller chooses, and the one rule for a figure whose object is not here.
"""

import importlib
import json
import logging

import pytest

from digitalearth.base.custom import (
    MissingObject,
    custom_engine,
    custom_kind,
    held_object,
    is_custom,
)
from digitalearth.base.registry import kind_info
from digitalearth.base.spec import (
    DataRef,
    FigureSpec,
    LayerSpec,
    LayerTree,
    PanelSpec,
    Symbology,
)


class TestTheCustomKind:
    """`custom:<engine>` — the kind a caller's own object is recorded under."""

    @pytest.mark.parametrize("engine", ["pyvista", "holoviews", "maplibre"])
    def test_each_engine_has_a_registered_kind(self, engine):
        """A tier can look the kind up, as it looks up any other.

        Args:
            engine: The engine under test.
        """
        assert kind_info(custom_kind(engine)).takes == "none", kind_info(
            custom_kind(engine)
        )

    def test_a_name_that_could_not_be_spelled_as_a_kind_is_refused(self):
        """The refusal names the engine, rather than arriving later as a `LayerSpec` error."""
        with pytest.raises(ValueError, match="is not an engine name"):
            custom_kind("PyVista")

    def test_a_custom_layer_is_told_apart_from_a_plugin_kind(self):
        """Both are namespaced; only one means "the object is the caller's"."""
        assert (is_custom("custom:pyvista"), is_custom("mypkg:hexbin")) == (
            True,
            False,
        ), "only the custom: prefix marks a caller's own object"

    def test_the_engine_is_read_back_out_of_the_kind(self):
        """A renderer asks which engine built the object before looking for it."""
        assert custom_engine("custom:holoviews") == "holoviews", custom_engine(
            "custom:holoviews"
        )

    def test_a_drawn_layer_has_no_engine_of_its_own(self):
        """Every tier can draw a `points` layer, so nothing holds an object for it."""
        assert custom_engine("points") is None, custom_engine("points")


class TestALayerThatDescribesACustomObject:
    """`LayerSpec` holds the description, and only the description."""

    def test_a_custom_layer_round_trips_through_json(self):
        """Id, kind, label, band and visibility survive; there is nothing else to write."""
        layer = LayerSpec(
            "wells",
            custom_kind("pyvista"),
            label="Wells",
            band="overlay",
            visible=False,
            group="survey",
        )
        rebuilt = LayerSpec.from_dict(json.loads(json.dumps(layer.to_dict())))
        assert rebuilt == layer, rebuilt.to_dict()

    def test_the_stored_form_holds_no_object(self):
        """What a figure stores is a placeholder: five keys, none of them a mesh."""
        stored = LayerSpec("wells", custom_kind("pyvista"), label="Wells").to_dict()
        assert stored == {
            "id": "wells",
            "kind": "custom:pyvista",
            "label": "Wells",
        }, stored

    def test_a_custom_layer_takes_the_band_the_caller_chose(self):
        """A caller's object can draw anything, so the band is the caller's to say."""
        tree = (
            LayerTree()
            .add(LayerSpec("dem", "raster"))
            .add(LayerSpec("tiles", custom_kind("maplibre"), band="underlay"))
        )
        assert tree.ids == ("tiles", "dem"), tree.ids

    def test_without_a_band_it_is_drawn_among_the_data(self):
        """The default is what most callers' objects are: another data layer."""
        tree = (
            LayerTree()
            .add(LayerSpec("dem", "raster"))
            .add(LayerSpec("wells", custom_kind("maplibre")))
        )
        assert tree.ids == ("dem", "wells"), tree.ids

    def test_a_band_nothing_could_draw_is_refused(self):
        """A misspelt band is caught where the layer is built."""
        kind = custom_kind("maplibre")
        with pytest.raises(ValueError, match="band must be one of"):
            LayerSpec("wells", kind, band="middle")

    def test_a_band_override_works_for_any_layer(self):
        """The override is not custom-only: a caller can put their coastlines under the data."""
        tree = (
            LayerTree()
            .add(LayerSpec("dem", "raster"))
            .add(LayerSpec("coast", "coastlines", band="reference"))
        )
        assert tree.ids == ("coast", "dem"), tree.ids


class TestAMissingObject:
    """One rule for a figure that names a custom layer whose object is not here — contract C7.

    The clause is stated once, in :data:`digitalearth.base.contract_clauses.CLAUSES`. A missing object is
    its *nothing to draw* half; `TestWhatC7DoesNotCover` pins the other half, a kind the tier does not draw
    at all, which is refused by name rather than skipped (#320).
    """

    @staticmethod
    def _loaded():
        """Return a figure loaded from a dict, holding one custom layer and no object.

        Returns:
            The figure.
        """
        return FigureSpec.from_dict(
            {
                "schema_version": 1,
                "layers": {
                    "layers": [{"id": "wells", "kind": "custom:maplibre"}],
                },
                "panels": [{"id": "map", "viewport": {"crs": 4326}}],
            }
        )

    def test_the_object_is_handed_back_when_the_renderer_holds_it(self):
        """The common case: the map that was given the object is the one drawing it."""
        drawn = held_object(
            "wells",
            "custom:maplibre",
            {"wells": "a layer"},
            engine="maplibre",
            backend="web",
        )
        assert drawn == "a layer", drawn

    def test_a_figure_loaded_from_a_dict_says_the_object_is_not_carried(self):
        """A description cannot rebuild a MapLibre layer somebody built in a notebook."""
        layer = self._loaded().layers.get("wells")
        with pytest.raises(MissingObject, match="this figure does not carry"):
            held_object(layer.id, layer.kind, {}, engine="maplibre", backend="web")

    def test_another_backend_says_it_cannot_draw_that_engine(self):
        """A MapLibre layer in a 3-D scene is a different problem, and reads as one."""
        layer = self._loaded().layers.get("wells")
        with pytest.raises(
            MissingObject, match=r"custom:maplibre layer and cannot be drawn by the 3d"
        ):
            held_object(layer.id, layer.kind, {}, engine="pyvista", backend="3d")

    def test_the_message_names_the_layer(self):
        """A figure with a dozen layers needs to say which one was skipped."""
        with pytest.raises(MissingObject, match="'wells'"):
            held_object("wells", "custom:pyvista", {}, engine="pyvista", backend="3d")


class TestWhatC7DoesNotCover:
    """The two situations C7 used to be read for, pinned apart so neither drifts into the other (#320).

    Both halves of the clause are stated once, in :data:`digitalearth.base.contract_clauses.CLAUSES`. The
    *nothing to draw* half is pinned by the rest of this file and by each tier's contract file; this class
    exists to keep the other half, because the text once read as if ``strict=`` governed it too, and
    softening it would swallow a typo'd or cross-backend kind in silence.

    Both halves are asked of one lenient scene, so a future change cannot satisfy one by breaking the other.
    """

    #: The layer id every figure here uses, named once so the assertions and the builder cannot drift.
    LAYER_ID = "wells"

    #: A kind each tier does **not** draw, with the module its `drawer_for` lives in. Every module imports
    #: without its engine: `drawer_for` refuses a kind before it imports any builder behind one, which is
    #: what lets all four tiers be asked from the engine-free base env.
    UNDRAWN_PER_TIER = [
        ("static", "digitalearth.static.renderer", "terrain"),
        ("interactive", "digitalearth.interactive.renderer", "terrain"),
        ("3d", "digitalearth.three_d.renderer", "choropleth"),
        ("web", "digitalearth.web.renderer", "mesh"),
    ]

    @classmethod
    def _figure(cls, kind):
        """Return a one-layer figure naming `kind` and carrying no object for it.

        Args:
            kind: The layer kind to name.

        Returns:
            The figure.
        """
        layer = LayerSpec(
            cls.LAYER_ID, kind, symbology=Symbology(props={"via": "custom"})
        )
        return FigureSpec(
            sources={},
            layers=LayerTree((layer,)),
            panels=(PanelSpec("map", layers=(cls.LAYER_ID,)),),
        )

    @pytest.mark.parametrize(
        ("tier", "module_name", "kind"),
        UNDRAWN_PER_TIER,
        ids=[row[0] for row in UNDRAWN_PER_TIER],
    )
    def test_a_kind_the_tier_does_not_draw_is_refused_by_name(
        self, tier, module_name, kind
    ):
        """All four tiers refuse it, and none of them has a scene to be lenient with.

        Args:
            tier: The tier's name, for the parametrisation id.
            module_name: The module its `drawer_for` lives in.
            kind: A kind that tier does not draw.

        Test scenario:
            `drawer_for` is a module function: there is no scene, so no ``strict`` flag can reach it and
            lenience is not expressible. That is the point — the refusal is unconditional by construction,
            not by a branch someone could later put a ``strict`` test in.
        """
        drawer_for = importlib.import_module(module_name).drawer_for
        with pytest.raises(KeyError, match=f"does not draw '{kind}'"):
            drawer_for(kind)

    def test_the_refusal_reaches_a_lenient_scene_unchanged(self):
        """A figure naming an undrawable kind raises on a scene that skips everything else.

        Test scenario:
            Asked through the render path rather than of `drawer_for` directly, on a `Map` left at its
            default ``strict=False``. A caller who sent a 3-D figure to the static tier is told which kind
            it was; the alternative is a blank figure that claims a layer.
        """
        from digitalearth.static import Map

        with Map(crs=4326) as scene:
            assert scene.strict is False, "the probe must be a lenient scene"
            with pytest.raises(KeyError, match="does not draw 'terrain'"):
                scene._renderer.draw_layer(self._figure("terrain"), self.LAYER_ID)

    def test_a_layer_with_nothing_to_draw_still_skips_on_that_same_scene(self, caplog):
        """The half the decision must not have broken: nothing to draw is still a warning, not a raise.

        Args:
            caplog: Captures the warning the skip logs.

        Test scenario:
            The same lenient `Map`, the same shape of figure — but a kind the tier *does* draw, whose
            object this process never held. C7 applies, so the layer is skipped and the render carries on.
        """
        from digitalearth.static import Map

        with Map(crs=4326) as scene:
            figure = self._figure(custom_kind("matplotlib"))
            with caplog.at_level(logging.WARNING):
                drawn = scene._renderer.draw_layer(figure, self.LAYER_ID)
        assert drawn is None, f"a layer with nothing to draw must draw nothing: {drawn}"
        assert "this figure does not carry" in caplog.text, (
            f"the skip was not reported: {caplog.text!r}"
        )

    def test_and_the_same_layer_raises_under_strict(self):
        """``strict=True`` is the dial for C7's case, and it still turns the skip back into an error.

        Test scenario:
            Only the flag differs from the test above, so this pins that the dial is read — a skip that
            ignored ``strict`` would pass the previous test just as well.
        """
        from digitalearth.static import Map

        with Map(crs=4326, strict=True) as scene:
            figure = self._figure(custom_kind("matplotlib"))
            with pytest.raises(MissingObject, match="this figure does not carry"):
                scene._renderer.draw_layer(figure, self.LAYER_ID)


class TestTheDiffOfACustomLayer:
    """`FigureSpec.diff` reports what it can see, and nothing about the object."""

    @staticmethod
    def _figure(*, visible=True, order=("dem", "wells"), symbology=None):
        """Return a figure with a raster and a custom layer.

        Args:
            visible: Whether the custom layer is visible.
            order: The ids in draw order.
            symbology: The custom layer's symbology, if any.

        Returns:
            The figure.
        """
        layers = {
            "dem": LayerSpec("dem", "raster", source_id="dem"),
            "wells": LayerSpec(
                "wells",
                custom_kind("maplibre"),
                visible=visible,
                symbology=symbology or Symbology(),
            ),
        }
        return FigureSpec(
            sources={"dem": DataRef("file:///dem.tif")},
            layers=LayerTree(tuple(layers[layer_id] for layer_id in order)),
            panels=(PanelSpec("map", layers=order),),
        )

    def test_hiding_a_custom_layer_is_reported(self):
        """Visibility is in the description, so the renderer is told."""
        difference = self._figure().diff(self._figure(visible=False))
        assert difference.hidden == ("wells",), difference

    def test_reordering_a_custom_layer_is_reported(self):
        """Draw order is in the description too."""
        difference = self._figure().diff(self._figure(order=("wells", "dem")))
        assert difference.order == ("wells", "dem"), difference

    def test_nothing_is_reported_about_the_object(self):
        """Two figures describing the same custom layer differ in nothing, whatever the objects were.

        Test scenario:
            The accepted limit from the issue: the contents of a caller's object are invisible to `diff`, so
            changing them means replacing the layer.
        """
        difference = self._figure().diff(self._figure())
        assert not difference, difference


class TestLettingAnObjectGo:
    """Review H2 — `forget_object` is the counterpart to `register_object`."""

    def test_a_forgotten_uri_no_longer_resolves(self):
        """A figure that no longer draws an object should not keep it alive."""
        from digitalearth.base.registry import (
            forget_object,
            register_object,
            resolve_uri,
        )

        uri = register_object([1, 2, 3], name="test-forget-one")
        forget_object(uri)
        with pytest.raises(KeyError, match="test-forget-one"):
            resolve_uri(uri)

    def test_a_bare_id_is_accepted_too(self):
        """The id on its own, for a caller holding the key rather than the URI."""
        from digitalearth.base.registry import (
            forget_object,
            register_object,
            resolve_uri,
        )

        register_object([1], name="test-forget-two")
        forget_object("test-forget-two")
        with pytest.raises(KeyError, match="test-forget-two"):
            resolve_uri("object:test-forget-two")

    def test_forgetting_what_was_never_registered_leaves_the_rest_alone(self):
        """So a caller can forget the same layer twice, and nothing else goes with it."""
        from digitalearth.base.registry import _OBJECTS, forget_object, register_object

        register_object([1], name="test-forget-bystander")
        held = len(_OBJECTS)
        forget_object("object:never-registered")
        assert len(_OBJECTS) == held, "forgetting an unknown id must touch nothing"
        forget_object("test-forget-bystander")

    def test_an_empty_id_is_nothing_to_forget(self):
        """A layer that drew from no data has no reference to forget (review N5)."""
        from digitalearth.base.registry import _OBJECTS, forget_object, register_object

        register_object([1], name="test-forget-empty-bystander")
        held = len(_OBJECTS)
        forget_object("")
        forget_object(None)
        assert len(_OBJECTS) == held, "an empty id must not reach the table"
        forget_object("test-forget-empty-bystander")

    def test_a_namespace_goes_all_at_once_and_takes_nothing_else(self):
        """What a figure calls when it closes, so a notebook does not hold every dataset ever drawn.

        Test scenario:
            Namespacing stopped the entries colliding, so re-running a cell left the previous figure's
            data in the table for the life of the session (review M2).
        """
        from digitalearth.base.registry import (
            forget_namespace,
            object_namespace,
            register_object,
            resolve_uri,
        )

        mine, theirs = object_namespace(), object_namespace()
        ours = [
            register_object([1], name=f"{mine}:a"),
            register_object([2], name=f"{mine}:b"),
        ]
        kept = register_object([3], name=f"{theirs}:a")
        forget_namespace(mine)
        for uri in ours:
            with pytest.raises(KeyError, match="no object is registered"):
                resolve_uri(uri)
        assert resolve_uri(kept) == [3], "another figure's namespace is untouched"
        forget_namespace(theirs)

    def test_an_object_of_another_engine_is_refused_before_it_is_handed_over(self):
        """The one case the function exists to tell apart, and the one the reordering changed.

        Test scenario:
            The held object was returned before the engine was checked, so a layer id that collides across
            tiers handed a renderer an object built with another engine (review L4). The engine is read
            first, so the answer does not depend on whether the id happens to be in this table.
        """
        from digitalearth.base.custom import MissingObject, held_object

        with pytest.raises(MissingObject, match="cannot be drawn by the web backend"):
            held_object(
                "wells",
                "custom:pyvista",
                {"wells": "a pyvista mesh"},
                engine="maplibre",
                backend="web",
            )
