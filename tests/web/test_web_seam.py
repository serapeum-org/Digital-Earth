"""The web seam: what the map describes is what the map draws (#296).

The tier used to keep two parallel structures — a description in the layer tree and a drawing in a queue of
closures — with nothing holding them to each other. These are the checks that they are now one thing.

The renderer conformance suite (#305) is inherited separately, at the bottom of this file: the web tier
signs the same contract the 3-D tier does.
"""

import importlib.util

import pytest

from digitalearth.base.registry import band_of
from digitalearth.base.spec import LayerSpec
from digitalearth.web.capabilities import CAPABILITIES
from digitalearth.web.renderer import DRAWN_KINDS, drawer_for

from ..base.test_renderer_conformance import RendererConformance, RendererContract

pytest.importorskip("maplibre", reason="the web tier needs the web environment")


@pytest.fixture
def points_gdf():
    """Return a small point collection in EPSG:4326.

    Returns:
        A two-point GeoDataFrame, enough for any builder that takes points.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    return gpd.GeoDataFrame(
        {"value": [1.0, 2.0]},
        geometry=[Point(4.9, 52.4), Point(5.1, 52.1)],
        crs=4326,
    )


class TestTheTwoPathsAreDisjoint:
    """A builder either records a description or queues a drawing — never both."""

    def test_every_drawable_kind_is_declared(self):
        """A kind drawn but not declared would be invisible to `Capabilities`."""
        undeclared = sorted(set(DRAWN_KINDS) - CAPABILITIES.kinds)
        assert undeclared == [], f"the tier draws {undeclared} without declaring them"

    def test_every_drawable_kind_resolves_to_a_drawer(self):
        """A kind on the list with no drawer raises a bare `KeyError` far from the cause."""
        unresolved = []
        for kind in DRAWN_KINDS:
            try:
                drawer_for(kind)
            except KeyError:
                unresolved.append(kind)
        assert unresolved == [], f"{unresolved} are declared drawable with no drawer"

    def test_a_kind_from_another_tier_is_refused_by_name(self):
        """A figure written for another backend should say so, not fail obscurely."""
        with pytest.raises(KeyError, match="does not draw 'terrain'"):
            drawer_for("terrain")

    def test_no_builder_both_describes_and_queues_a_layer(self, points_gdf):
        """Drawing a layer twice is what a half-open seam looks like.

        Args:
            points_gdf: A small point collection.

        Test scenario:
            A converted builder records a description and the renderer draws it; an unconverted one queues
            a closure. If a builder did both, the layer would be added to the widget twice — once from the
            description and once from the queue — and the second would win silently.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().points(points_gdf, name="obs").graticule(name="grid")
        drawn = set(m._renderer.drawn)
        queued_ids = {
            getattr(entry, "_digitalearth_layer_id", None) for entry in m._queued
        }
        described_and_queued = {
            layer_id
            for layer_id in drawn
            if layer_id in queued_ids and not _is_marker(m, layer_id)
        }
        assert described_and_queued == set(), (
            f"{sorted(described_and_queued)} are both drawn from a description and queued as a drawing"
        )

    def test_the_widget_holds_each_layer_once(self, points_gdf):
        """The observable form of the check above.

        Args:
            points_gdf: A small point collection.
        """
        from digitalearth.web import WebMap

        m = WebMap().points(points_gdf, name="obs").graticule(name="grid")
        added = []

        class Recorder:
            """Stands in for the widget, recording what is added to it."""

            def add_source(self, source_id, spec):
                """Ignore the source; only the layers are under test.

                Args:
                    source_id: Unused.
                    spec: Unused.
                """

            def add_layer(self, layer):
                """Record one added layer.

                Args:
                    layer: The layer reaching the widget.
                """
                added.append(getattr(layer, "id", layer))

        for entry in m.layers:
            m._apply_layer(Recorder(), entry)
        assert len(added) == len(set(added)), f"a layer was added twice: {added}"
        for layer_id in m.layer_ids:
            assert layer_id in added, f"{layer_id} is described but never drawn"


def _is_marker(web_map, layer_id: str) -> bool:
    """Whether a queue entry for `layer_id` is a description marker rather than a drawing.

    Args:
        web_map: The map holding the queue.
        layer_id: The layer to look for.

    Returns:
        `True` when the entry standing for that layer is the seam's marker, which is expected, rather than
        a closure that would draw it a second time.
    """
    from digitalearth.web.base import _Described

    for entry in web_map._queued:
        if getattr(entry, "_digitalearth_layer_id", None) == layer_id:
            return isinstance(entry, _Described)
    return False


class TestADescriptionIsWhatIsDrawn:
    """The figure is the source of truth, not a label beside one."""

    def test_a_layer_is_drawn_from_its_recorded_symbology(self, points_gdf):
        """Change the description and the drawing follows, because one is built from the other.

        Args:
            points_gdf: A small point collection.
        """
        from dataclasses import replace as with_fields

        from digitalearth.base.spec import Symbology
        from digitalearth.web import WebMap

        m = WebMap().points(points_gdf, name="obs", size=4.0)
        layer = m.figure_spec.layers.get("obs")
        paint = dict(layer.symbology.props["paint"])
        assert paint["circle-radius"] == 4.0, paint

        bigger = with_fields(
            layer,
            symbology=Symbology(
                props={
                    **layer.symbology.props,
                    "paint": {**paint, "circle-radius": 9.0},
                }
            ),
        )
        redrawn = m._renderer.draw_layer(
            with_fields(m.figure_spec, layers=m.figure_spec.layers.replace(bigger)),
            "obs",
        )
        assert redrawn.layer.paint["circle-radius"] == 9.0, redrawn.layer.paint

    def test_generated_ids_follow_the_layer_rather_than_a_counter(self, points_gdf):
        """Drawing the same description twice gives the same ids.

        Args:
            points_gdf: A small point collection.

        Test scenario:
            Source ids came from a counter, so re-drawing a figure produced different ids for the same
            layer and nothing downstream could match them up. They are derived from the layer id now.
        """
        from digitalearth.web import WebMap

        m = WebMap().points(points_gdf, name="obs")
        first = m._renderer.draw_layer(m.figure_spec, "obs")
        second = m._renderer.draw_layer(m.figure_spec, "obs")
        assert first.source_id == second.source_id == "obs-src", first.source_id
        assert first.layer.id == second.layer.id == "obs", first.layer.id


class TestAFigureSurvivesBeingWrittenDown:
    """The seam's point: a description carries what the drawing needs, so it can be saved and reloaded."""

    def test_a_saved_figure_reloads_and_draws_on_another_map(self):
        """Before the seam this was impossible: the drawing lived in a closure, not in the figure.

        Test scenario:
            A closure captured the compiled MapLibre source and layer, so `to_dict` could only record that
            a layer existed — never enough to draw it again. Symbology is recorded as values now, so a
            figure written to a dict draws on a map that never saw the builder call.
        """
        from digitalearth.base.spec import FigureSpec
        from digitalearth.web import WebMap

        saved = (
            WebMap()
            .graticule(name="grid", lon_step=15.0)
            .text(4.9, 52.4, "Amsterdam", name="lbl")
            .figure_spec.to_dict()
        )
        reloaded = FigureSpec.from_dict(saved)
        assert list(reloaded.layers.ids) == ["grid", "lbl"], reloaded.layers.ids
        assert reloaded.layers.get("grid").symbology.props["lon_step"] == 15.0
        assert reloaded.layers.get("lbl").symbology.props["s"] == "Amsterdam"

        elsewhere = WebMap()
        drawn = elsewhere._renderer.draw_layer(reloaded, "grid")
        assert drawn is not None, "a reloaded description must be drawable"
        assert drawn.layer.id == "grid", drawn.layer.id

    def test_an_in_memory_source_is_refused_rather_than_written_as_a_dead_reference(
        self, points_gdf
    ):
        """An `object:` reference resolves only in the process that made it, and the figure says so.

        Args:
            points_gdf: A collection held in memory rather than read from a path.
        """
        from digitalearth.web import WebMap

        m = WebMap().points(points_gdf, name="obs")
        with pytest.raises(ValueError, match="only resolves in the process"):
            m.figure_spec.to_dict()


class TestADrawerThatDeclinesLeavesNothingBehind:
    """A described layer nothing draws is exactly the drift this seam removes."""

    def test_an_unplaceable_raster_is_not_registered(self, monkeypatch, dataset):
        """The builder's caller sees a map that never registered it.

        Args:
            monkeypatch: Used to make the raster's corners unrepresentable.
            dataset: A small raster.

        Test scenario:
            The description is written before the drawer runs, so a drawer that declines has to take the
            description with it — otherwise `figure_spec` names a layer the map cannot draw.
        """
        from digitalearth.web import WebMap

        monkeypatch.setattr(WebMap, "_lonlat_corners", lambda self, source: None)
        m = WebMap().field(dataset)
        assert m.layer_ids == [], m.layer_ids
        assert m._renderer.drawn == {}, m._renderer.drawn
        assert m.layers == [], m.layers

    def test_a_custom_layer_this_process_does_not_hold_is_not_drawn(self):
        """A figure loaded from a dict names the object but cannot rebuild it."""
        from digitalearth.web import WebMap

        m = WebMap()
        m.add_layer(_fake_layer("wells"), name="wells")
        # Forget the object the way loading a saved figure would: the description survives, the object
        # does not.
        m._custom.clear()
        built = m._renderer.draw_layer(m.figure_spec, "wells")
        assert built is None, built

    def test_a_widget_skips_a_marker_whose_drawing_is_gone(self):
        """The marker and the drawing are removed together, but the widget must not assume it."""
        from digitalearth.web import WebMap

        m = WebMap()
        m.add_layer(_fake_layer("wells"), name="wells")
        m._renderer.remove("wells")
        added = []

        class Recorder:
            """Records what reaches the widget."""

            def add_source(self, source_id, spec):
                """Ignore the source; only the layers are under test.

                Args:
                    source_id: Unused.
                    spec: Unused.
                """

            def add_layer(self, layer):
                """Record one added layer.

                Args:
                    layer: The layer reaching the widget.
                """
                added.append(layer)

        for entry in m._queued:
            m._apply_layer(Recorder(), entry)
        assert added == [], "a marker with no drawing must add nothing"


class TestTheBandOfALayerIsItsKinds:
    """Where a layer is drawn is a property of what it is, not of which path drew it."""

    def test_reference_geography_sits_over_the_basemap_and_under_the_data(
        self, points_gdf
    ):
        """A graticule beneath an opaque basemap is invisible; over the data it obscures it.

        Args:
            points_gdf: A small point collection.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().points(points_gdf, name="obs").graticule(name="grid")
        assert band_of("graticule") == "reference", band_of("graticule")
        assert m.layer_ids.index("grid") < m.layer_ids.index("obs"), m.layer_ids

    def test_a_custom_layer_is_placed_in_the_band_it_asked_for(self):
        """`custom:maplibre` names an engine, which says nothing about what the layer draws."""
        from digitalearth.web import WebMap

        m = WebMap()
        m.add_layer(_fake_layer("tiles"), name="tiles", band="underlay")
        m.add_layer(_fake_layer("obs"), name="obs")
        assert m.layer_ids == ["tiles", "obs"], m.layer_ids


def _fake_layer(layer_id: str):
    """Return an object standing for a caller's own MapLibre layer.

    Args:
        layer_id: The id it carries.

    Returns:
        A minimal object with an `id`, which is all `add_layer` reads.
    """

    class _Layer:
        """A caller's own layer."""

        def __init__(self, id_):
            """Hold the id `add_layer` reads.

            Args:
                id_: The layer id.
            """
            self.id = id_

    return _Layer(layer_id)


needs_maplibre = pytest.mark.skipif(
    importlib.util.find_spec("maplibre") is None,
    reason="the web tier needs the web environment",
)


class WebContract(RendererContract):
    """The web tier's adapter for the shared renderer contract (#305)."""

    backend = "web"

    def make(self):
        """Return an empty map.

        Returns:
            The map.
        """
        from digitalearth.web import WebMap

        return WebMap()

    def draw_one(self, tier) -> str:
        """Draw one point layer, which reaches MapLibre.

        Args:
            tier: The map.

        Returns:
            The drawn layer's id.
        """
        import geopandas as gpd
        from shapely.geometry import Point

        features = gpd.GeoDataFrame(
            {"value": [1.0, 2.0]},
            geometry=[Point(4.9, 52.4), Point(5.1, 52.1)],
            crs=4326,
        )
        tier.points(features, name="obs")
        return "obs"

    def refused_figure(self, tier):
        """Return a figure whose second layer names a kind this tier does not draw.

        Args:
            tier: The map.

        Returns:
            The figure.
        """
        from dataclasses import replace as with_fields

        figure = tier.figure_spec
        drawable = figure.layers.get(tier.layer_ids[-1])
        # The first layer must really draw, or the refusal below is never reached and the rollback check
        # passes for the wrong reason — which is what mutation-testing this contract caught. It copies the
        # drawn layer's symbology rather than inventing one.
        tree = figure.layers.add(with_fields(drawable, id="second")).add(
            LayerSpec("refused", "terrain", source_id=drawable.source_id)
        )
        return with_fields(figure, layers=tree)

    def apply_figure(self, tier, figure) -> None:
        """Move the map to `figure` through the renderer.

        Args:
            tier: The map.
            figure: The figure.
        """
        tier._renderer.apply(tier.figure_spec, figure)

    def engine_holds(self, tier):
        """Return which layers are drawn, *and what was drawn for them*.

        A redraw replaces a layer under the same id, so a set of ids reads identically before and after;
        the identity of the MapLibre object does not.

        Args:
            tier: The map.

        Returns:
            Layer id to the identity of what was drawn for it.
        """
        return {
            layer_id: id(built.layer)
            for layer_id, built in tier._renderer.drawn.items()
        }

    def relabel(self, tier, layer_id: str):
        """Return the map's figure with one layer's label changed and nothing else.

        Args:
            tier: The map.
            layer_id: The layer to relabel.

        Returns:
            The figure.
        """
        from dataclasses import replace as with_fields

        figure = tier.figure_spec
        renamed = with_fields(figure.layers.get(layer_id), label="a different name")
        return with_fields(figure, layers=figure.layers.replace(renamed))

    def drawn_is_in_view(self, tier):
        """The web tier frames on its data rather than on a camera it could mis-place.

        Args:
            tier: The map.

        Returns:
            `None` — there is no view a drawn layer can fall outside of, so the check skips.
        """
        return None

    def declared_kinds(self) -> frozenset:
        """Return the kinds the tier declares.

        Returns:
            The declared kinds.
        """
        return CAPABILITIES.kinds

    def drawn_kinds(self) -> tuple:
        """Return the kinds the renderer draws from a description.

        Returns:
            The drawable kinds.
        """
        return DRAWN_KINDS

    def drawer_for(self, kind: str):
        """Return the drawer registered for `kind`.

        Args:
            kind: The layer kind.

        Returns:
            The drawer.
        """
        return drawer_for(kind)


@needs_maplibre
class TestWebRendererConformance(RendererConformance):
    """The web tier, signing the contract the 3-D tier already passes."""

    contract = WebContract()
