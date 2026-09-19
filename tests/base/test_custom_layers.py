"""A caller's own engine object, described well enough to address it (DE-49, #293).

Every tier lets a caller hand it a PyVista mesh, a HoloViews element or a MapLibre layer. None of them could be
addressed, described or saved around: on the web tier the object never reached the layer tree, so `layer_ids` did
not list it and `remove_layer` raised. These cover the description that replaces that — `custom:<engine>`, a band
the caller chooses, and the one rule for a figure whose object is not here.
"""

import json

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
    """One rule for a figure that names a custom layer whose object is not here (contract C7)."""

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
