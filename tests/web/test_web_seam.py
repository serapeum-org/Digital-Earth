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
from digitalearth.web.renderer import DRAWN_KINDS, derived_ids, drawer_for

from ..base.test_renderer_conformance import RendererConformance, RendererContract
from .test_web_capabilities import DESCRIBED_AND_DRAWN

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

            The queue is read by *count*, not by looking up the first entry under an id: the marker is
            always queued first, so asking what the first entry under an id is answered "a marker" whether
            or not a closure followed it, and a builder that did both passed (review M11). Every id in the
            queue must appear once, and the ids in the queue must be exactly the layers described.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().points(points_gdf, name="obs").graticule(name="grid")
        queued = [
            layer_id
            for entry in m._queued
            if (layer_id := getattr(entry, "_digitalearth_layer_id", None)) is not None
        ]
        repeated = sorted(
            {layer_id for layer_id in queued if queued.count(layer_id) > 1}
        )
        assert repeated == [], (
            f"{repeated} are queued twice — described and queued as a drawing, so the widget adds them twice"
        )
        assert sorted(queued) == sorted(m.layer_ids), (
            f"the queue names {sorted(queued)}; the map describes {sorted(m.layer_ids)}"
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

    def test_a_kind_the_seam_has_not_reached_is_described_but_left_to_the_queue(self):
        """The seam opens one kind at a time, so a kind it has not reached must not be drawn from its record.

        Test scenario:
            `DRAWN_KINDS` is a growing subset: a kind joins it in the same step its builder stops queuing
            a closure. Drawing every described layer regardless of that list would hand an unconverted
            kind to `drawer_for`, which refuses it by name — so a builder that has not been converted yet
            would stop working the moment it recorded a description. Its layer is still registered; it is
            simply drawn by its own queued closure rather than from the tree.
        """
        from digitalearth.web import WebMap

        unconverted = "terrain"
        assert unconverted in CAPABILITIES.kinds, (
            f"{unconverted} must be a kind the tier declares, or this asks nothing"
        )
        assert unconverted not in DRAWN_KINDS, (
            f"{unconverted} is drawn from its description now; ask about a kind that is not"
        )

        m = WebMap()
        registered = m._index_layer("dem", "dem", kind=unconverted)
        assert registered is True, "an unconverted kind is still a registered layer"
        assert m.layer_ids == ["dem"], m.layer_ids
        assert m._renderer.drawn == {}, (
            f"an unconverted kind was drawn from its description: {m._renderer.drawn}"
        )
        assert m._queued == [], (
            f"an unconverted kind was queued as a description: {m._queued}"
        )

    def test_a_custom_layer_hands_the_widget_no_source_of_its_own(self):
        """A caller's own object carries its data inside it, so nothing is added beside it.

        Test scenario:
            Every other description builds a MapLibre source the widget is given before the layer.
            `draw_custom` has none — `custom:maplibre` means the object *is* the layer — so the widget
            builder has to skip that step rather than call `add_source(None, None)`, which MapLibre would
            take as a real source under an unusable id.
        """
        from digitalearth.web import WebMap

        m = WebMap()
        own = _fake_layer("wells")
        m.add_layer(own, name="wells")
        sources = []
        added = []

        class Recorder:
            """Records both halves of what reaches the widget."""

            def add_source(self, source_id, spec):
                """Record one added source.

                Args:
                    source_id: The source id reaching the widget.
                    spec: The source definition reaching the widget.
                """
                sources.append((source_id, spec))

            def add_layer(self, layer):
                """Record one added layer.

                Args:
                    layer: The layer reaching the widget.
                """
                added.append(layer)

        for entry in m._queued:
            m._apply_layer(Recorder(), entry)
        assert sources == [], f"a custom layer added a source of its own: {sources}"
        assert added == [own], f"the caller's own object must be what is added: {added}"


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

    def test_the_drawing_uses_the_recorded_values_rather_than_its_own_defaults(self):
        """A description that survives but is not *read* is a description in name only.

        Test scenario:
            The round-trip check above asserts the props are written and read back. It does not assert the
            drawer uses them — and a drawer that ignored them and drew its defaults passed the whole suite.
            Two grids an octave apart must differ in the drawing, not only in the record.
        """
        from digitalearth.web import WebMap

        coarse = WebMap().graticule(name="grid", lon_step=60.0, lat_step=60.0)
        fine = WebMap().graticule(name="grid", lon_step=15.0, lat_step=15.0)
        coarse_lines = coarse._renderer.drawn["grid"].source_spec["data"]["features"]
        fine_lines = fine._renderer.drawn["grid"].source_spec["data"]["features"]
        assert len(fine_lines) > len(coarse_lines), (
            f"a finer grid must draw more lines; got {len(fine_lines)} vs {len(coarse_lines)}"
        )

    def test_a_text_annotation_draws_the_string_it_recorded(self):
        """The same question for the other source-less kind."""
        from digitalearth.web import WebMap

        m = WebMap().text(4.9, 52.4, "Amsterdam", name="lbl", text_size=18.0)
        built = m._renderer.drawn["lbl"]
        assert built.source_spec["data"]["properties"]["text"] == "Amsterdam"
        assert built.layer.layout["text-size"] == 18.0, built.layer.layout

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

    def test_a_declined_cluster_layer_leaves_no_tooltip_target_behind(
        self, monkeypatch, points_gdf
    ):
        """`cluster` files a tooltip target its own drawer produced, so a refusal must not file one.

        Args:
            monkeypatch: Used to make the cluster drawer decline.
            points_gdf: A small point collection.

        Test scenario:
            The bubbles are aggregates whose only properties are point counts, so `cluster` files the
            *unclustered* points as the layer a later `.tooltip()` binds to — an id that exists only
            because the drawer drew it. Filing it whether or not the drawer drew would leave the tooltip
            bound to a layer no widget holds, which is exactly the drift this seam removes.
        """
        from digitalearth.web import WebMap, bigdata

        def _declines(web_map, data, layer):
            """Behave like a drawer that found nothing to draw.

            Args:
                web_map: Unused — the drawer declines before it would read the map.
                data: Unused — the collection it would have clustered.
                layer: Unused — the description it would have drawn.

            Returns:
                `None`, the shape a drawer answers in when it declines.
            """
            return None

        monkeypatch.setattr(bigdata, "draw_clusters", _declines)
        m = WebMap().points(points_gdf, name="obs")
        m.cluster(points_gdf)
        assert m.layer_ids == ["obs"], m.layer_ids
        assert m._last_layer_id == "obs", (
            f"a declined cluster layer moved the tooltip target to {m._last_layer_id!r}"
        )

    def test_a_raster_redrawn_where_it_cannot_be_placed_is_declined(
        self, monkeypatch, warning_log, dataset
    ):
        """The drawer keeps its own placement guard, because a figure can reach it without its builder.

        Args:
            monkeypatch: Used to make the display warp place none of the data.
            warning_log: The tier's loguru warnings.
            dataset: A small raster.

        Test scenario:
            `field` checks placement before it records anything, so on that path its drawer never meets
            an unplaceable source. A *saved* figure does not go through `field`: it is handed straight to
            `draw_layer`, possibly on a map whose display CRS cannot place the raster at all. Without the
            drawer's own guard that path raises where every other unplaceable layer is skipped with a
            warning, and the record is left holding a drawing the figure no longer describes.
        """
        from digitalearth.base.crs import OffLimbError
        from digitalearth.web import WebMap

        m = WebMap().field(dataset, name="band")
        first = m._renderer.drawn["band"]

        def _off_limb(self, data, band=1):
            """Behave like a warp that placed none of the data.

            Args:
                self: The map being drawn on.
                data: Unused — the source that would have been warped.
                band: Unused — the band that would have been read.

            Raises:
                OffLimbError: always, which is what an off-limb warp raises.
            """
            raise OffLimbError("the data lies outside what 4326 can show")

        monkeypatch.setattr(WebMap, "_to_display_source", _off_limb)
        assert m._renderer.draw_layer(m.figure_spec, "band") is None, (
            "an unplaceable raster must be declined rather than drawn somewhere wrong"
        )
        assert m._renderer.drawn["band"] is first, (
            "a declined redraw replaced what was already drawn"
        )
        assert any("field" in line for line in warning_log), warning_log


def _unplaceable(monkeypatch) -> None:
    """Make every raster's corners unrepresentable in lon/lat, so a raster drawer declines — or, strict, raises.

    Args:
        monkeypatch: pytest's patcher, which restores the method after the test (or on `undo()`).
    """
    from digitalearth.web import WebMap

    monkeypatch.setattr(WebMap, "_lonlat_corners", lambda self, source: None)


def _registered(web_map) -> list:
    """Return the ids this map holds in the process-global object registry.

    Read from the registry itself rather than from `figure_spec.sources`, which filters by the layers in the
    tree — so a source a declined layer left registered is invisible there (review M6).

    Args:
        web_map: The map whose namespace is read.

    Returns:
        The registry keys under the map's namespace, sorted.
    """
    from digitalearth.base import registry

    return sorted(
        key for key in registry._OBJECTS if key.startswith(f"{web_map._objects_ns}:")
    )


#: The two raster builders, each drawn under the name the tests below reuse.
_RASTER_BUILDERS = [
    pytest.param(lambda m, ds: m.field(ds, name="dem"), id="field"),
    pytest.param(
        lambda m, ds: m.rgb_composite(ds, bands=(1, 1, 1), name="dem"),
        id="rgb_composite",
    ),
]


class TestALayerThatIsNotDrawnLeavesNothingBehind:
    """Review M6: a declined or refused layer gives back its record, its data and its name."""

    @pytest.mark.parametrize("build", _RASTER_BUILDERS)
    def test_a_declined_layer_lets_its_data_go(self, monkeypatch, dataset, build):
        """The object registry is process-global, so a source left in it is held for the process's life.

        Args:
            monkeypatch: Used to make the drawer decline.
            dataset: A small raster.
            build: The raster builder under test.
        """
        from digitalearth.web import WebMap

        _unplaceable(monkeypatch)
        m = WebMap()
        build(m, dataset)
        assert _registered(m) == [], "a declined layer left its data registered"

    @pytest.mark.parametrize("build", _RASTER_BUILDERS)
    def test_a_declined_name_is_free_again(self, monkeypatch, dataset, build):
        """A caller who names a layer, watches it skip, and names it again gets the name they asked for.

        Args:
            monkeypatch: Used to make the drawer decline once.
            dataset: A small raster.
            build: The raster builder under test.

        Test scenario:
            The id stayed issued, so the retry became `dem-2` — and a layer switcher captions a row with
            its id, so the viewer read the suffix.
        """
        from digitalearth.web import WebMap

        _unplaceable(monkeypatch)
        m = WebMap()
        build(m, dataset)
        monkeypatch.undo()
        build(m, dataset)
        assert m.layer_ids == ["dem"], m.layer_ids

    @pytest.mark.parametrize("build", _RASTER_BUILDERS)
    def test_a_refused_layer_is_not_described(self, monkeypatch, dataset, build):
        """Under `strict` the drawer raises, and the record must go with the drawing just as on a decline.

        Args:
            monkeypatch: Used to make the drawer refuse.
            dataset: A small raster.
            build: The raster builder under test.

        Test scenario:
            The description is written before the drawer runs. A decline took it back; a raise did not, so
            the figure named a layer no widget would ever hold.
        """
        from digitalearth.base.crs import OffLimbError
        from digitalearth.web import WebMap

        _unplaceable(monkeypatch)
        m = WebMap(strict=True)
        with pytest.raises(OffLimbError):
            build(m, dataset)
        assert m.layer_ids == [], m.layer_ids

    @pytest.mark.parametrize("build", _RASTER_BUILDERS)
    def test_a_refused_layer_lets_its_data_go(self, monkeypatch, dataset, build):
        """The raise path forgets the source too, not only the decline path.

        Args:
            monkeypatch: Used to make the drawer refuse.
            dataset: A small raster.
            build: The raster builder under test.
        """
        from digitalearth.base.crs import OffLimbError
        from digitalearth.web import WebMap

        _unplaceable(monkeypatch)
        m = WebMap(strict=True)
        with pytest.raises(OffLimbError):
            build(m, dataset)
        assert _registered(m) == [], "a refused layer left its data registered"

    @pytest.mark.parametrize("build", _RASTER_BUILDERS)
    def test_a_refused_name_is_free_again(self, monkeypatch, dataset, build):
        """A retry after a refusal gets the name it asked for.

        Args:
            monkeypatch: Used to make the drawer refuse once.
            dataset: A small raster.
            build: The raster builder under test.
        """
        from digitalearth.base.crs import OffLimbError
        from digitalearth.web import WebMap

        _unplaceable(monkeypatch)
        m = WebMap(strict=True)
        with pytest.raises(OffLimbError):
            build(m, dataset)
        monkeypatch.undo()
        build(m, dataset)
        assert m.layer_ids == ["dem"], m.layer_ids

    def test_a_refused_classification_is_not_filed_under_its_id(
        self, monkeypatch, points_gdf
    ):
        """The key a refused layer classified must not stay filed under an id that is free again.

        Args:
            monkeypatch: Used to make the vector drawer refuse.
            points_gdf: A small point collection.

        Test scenario:
            `_index_layer` files the builder's classification under the layer's id before it draws. With the
            id released, a later layer reusing it would inherit a key it was never drawn with; and `legend()`
            with no id would describe a layer that is not on the map.
        """
        from digitalearth.web import WebMap, vector

        m = WebMap().points(points_gdf, name="kept")
        monkeypatch.setattr(vector, "draw_vector", _refusing_drawer)
        with pytest.raises(ValueError, match="cannot be drawn"):
            m.points(points_gdf, column="value", name="classified")
        assert "classified" not in m._legends, sorted(m._legends)
        assert m.last_legend is None, m.last_legend

    def test_a_later_unclassified_layer_inherits_no_key(self, monkeypatch, points_gdf):
        """The surviving key takes `last_legend` back, and must not be filed again under the next layer.

        Args:
            monkeypatch: Used to make the vector drawer refuse once.
            points_gdf: A small point collection.

        Test scenario:
            `_index_layer` files `last_legend` under a new layer when it differs from the one it last filed.
            A refusal hands `last_legend` back to the surviving layer's key; unless the filing marker
            follows, the next *unclassified* layer is filed under that key and a colour key is drawn for a
            layer that was never classified.
        """
        from digitalearth.web import WebMap, vector

        m = WebMap().points(points_gdf, column="value", name="first")
        drawer = vector.draw_vector
        monkeypatch.setattr(vector, "draw_vector", _refusing_drawer)
        with pytest.raises(ValueError, match="cannot be drawn"):
            m.points(points_gdf, column="value", name="refused")
        monkeypatch.setattr(vector, "draw_vector", drawer)
        m.points(points_gdf, name="plain")
        assert "plain" not in m._legends, sorted(m._legends)

    def test_removing_a_classified_layer_files_no_key_under_the_next(self, points_gdf):
        """The same marker, on :meth:`remove_layer`'s path — the one the shared helper was lifted from.

        Args:
            points_gdf: A small point collection.
        """
        from digitalearth.web import WebMap

        m = (
            WebMap()
            .points(points_gdf, column="value", name="first")
            .points(points_gdf, column="value", name="second")
        )
        m.remove_layer("second").points(points_gdf, name="plain")
        assert "plain" not in m._legends, sorted(m._legends)


def _refusing_drawer(web_map, data, layer):
    """Behave like a drawer that cannot draw what it was given.

    Args:
        web_map: Unused — the drawer refuses before it would read the map.
        data: Unused — the collection it would have drawn.
        layer: Unused — the description it would have drawn.

    Raises:
        ValueError: always.
    """
    raise ValueError("this description cannot be drawn")


def _widget_layer_ids(web_map) -> list:
    """Return every MapLibre layer id the widget would be given, in the order it is given them.

    Args:
        web_map: The map whose queue is replayed.

    Returns:
        The id of each layer added — a description's own layer, the extra layers it owns (a graticule's
        labels, a cluster's counts and loose points), and whatever a queued closure adds.
    """
    added = []

    class Recorder:
        """Stands in for the widget, recording the id of every layer added to it."""

        def add_source(self, source_id, spec):
            """Ignore the source; MapLibre keeps sources in a namespace of their own.

            Args:
                source_id: Unused.
                spec: Unused.
            """

        def add_layer(self, layer):
            """Record one added layer's id.

            Args:
                layer: The layer reaching the widget.
            """
            added.append(getattr(layer, "id", layer))

    for entry in web_map._queued:
        web_map._apply_layer(Recorder(), entry)
    return added


def _duplicates(ids: list) -> list:
    """Return the ids that appear more than once.

    Args:
        ids: MapLibre layer ids.

    Returns:
        Each repeated id once, sorted — MapLibre keeps the first layer with an id and drops the rest.
    """
    return sorted({layer_id for layer_id in ids if ids.count(layer_id) > 1})


class TestADerivedIdIsReserved:
    """Review M5: an id a drawer derives from a layer's own is as taken as the layer's own id."""

    def test_a_caller_name_after_a_graticule_does_not_take_its_label_id(self):
        """`g` draws its degree labels as `g-label`, so a later `name="g-label"` must go elsewhere."""
        from digitalearth.web import WebMap

        m = WebMap().graticule(name="g").text(4.9, 52.4, "here", name="g-label")
        assert _duplicates(_widget_layer_ids(m)) == [], _widget_layer_ids(m)

    def test_a_graticule_after_a_caller_name_does_not_take_it_either(self):
        """The other order: the name is taken first, so the graticule's own id has to move."""
        from digitalearth.web import WebMap

        m = WebMap().text(4.9, 52.4, "here", name="g-label").graticule(name="g")
        assert _duplicates(_widget_layer_ids(m)) == [], _widget_layer_ids(m)

    @pytest.mark.parametrize("suffix", ["count", "unclustered"])
    def test_a_caller_name_after_a_cluster_does_not_take_its_derived_ids(
        self, points_gdf, suffix
    ):
        """A cluster draws its counts and its loose points under its own id, suffixed.

        Args:
            points_gdf: A small point collection.
            suffix: Which of the two derived layers the caller's name collides with.
        """
        from digitalearth.web import WebMap

        m = WebMap().cluster(points_gdf)
        clusters = m.layer_ids[0]
        m.text(4.9, 52.4, "here", name=f"{clusters}-{suffix}")
        assert _duplicates(_widget_layer_ids(m)) == [], _widget_layer_ids(m)

    def test_a_cluster_after_a_caller_name_does_not_take_it_either(self, points_gdf):
        """`cluster` numbers itself, so the name it would derive from is taken by the caller first.

        Args:
            points_gdf: A small point collection.
        """
        from digitalearth.web import WebMap

        m = (
            WebMap()
            .text(4.9, 52.4, "here", name="clusters-1-count")
            .cluster(points_gdf)
        )
        assert _duplicates(_widget_layer_ids(m)) == [], _widget_layer_ids(m)

    def test_a_caller_s_own_layer_under_a_derived_id_is_refused(self):
        """`add_layer` draws an object under the id it carries, so it cannot be moved — only refused.

        Test scenario:
            The id check covered the tree's ids, and a derived id is not in the tree, so the object was
            accepted and MapLibre dropped one of the two layers with only a console error.
        """
        from digitalearth.web import WebMap

        m = WebMap().graticule(name="g")
        own = _fake_layer("g-label")
        with pytest.raises(ValueError, match="already on this map"):
            m.add_layer(own)

    def test_a_graticule_after_a_caller_s_own_layer_moves_around_it(self):
        """The object's id is reserved when it is added, so a later derived id steps around it."""
        from digitalearth.web import WebMap

        m = WebMap().add_layer(_fake_layer("Graticule-label")).graticule()
        assert _duplicates(_widget_layer_ids(m)) == [], _widget_layer_ids(m)

    def test_a_declined_layer_gives_its_derived_ids_back(self, monkeypatch, points_gdf):
        """A cluster nothing drew holds no ids, derived ones included.

        Args:
            monkeypatch: Used to make the cluster drawer decline.
            points_gdf: A small point collection.
        """
        from digitalearth.web import WebMap, bigdata

        monkeypatch.setattr(bigdata, "draw_clusters", lambda web_map, data, layer: None)
        m = WebMap().cluster(points_gdf)
        m.text(4.9, 52.4, "here", name="clusters-1-count")
        assert m.layer_ids == ["clusters-1-count"], m.layer_ids

    @pytest.mark.parametrize("kind", sorted(DESCRIBED_AND_DRAWN))
    def test_the_table_names_every_extra_layer_a_drawer_adds(self, kind):
        """The reservation is only as good as the table it reads, so the table is held to the drawers.

        Args:
            kind: The drawn kind under test.

        Test scenario:
            A drawer that grew an extra layer without a row in `DERIVED_SUFFIXES` would add an id nothing
            reserved — the defect this class exists for. Every drawn kind is built, and the ids of the
            extra layers its drawer produced must be exactly the ones the table derives.
        """
        from digitalearth.web import WebMap

        m = WebMap()
        DESCRIBED_AND_DRAWN[kind](m)
        added = {
            layer_id: [extra.id for extra in built.extra_layers]
            for layer_id, built in m._renderer.drawn.items()
        }
        derived = {
            layer.id: list(derived_ids(layer.kind, layer.id))
            for layer in m.figure_spec.layers
        }
        assert added == derived, (added, derived)


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


class _Same:
    """One engine object in a snapshot, compared by identity and kept alive while the snapshot is held.

    A MapLibre `Layer` is a pydantic model and compares by value, so a layer drawn again from the same
    description would read as unchanged; identity is what a redraw changes. Holding the object also stops a
    freed object's `id()` being handed to the one that replaced it.

    Attributes:
        obj: The object.
    """

    __slots__ = ("obj",)

    def __init__(self, obj):
        """Hold one object.

        Args:
            obj: The object.
        """
        self.obj = obj

    def __eq__(self, other):
        """Whether `other` holds this same object.

        Args:
            other: Another snapshot entry.

        Returns:
            `True` only for the same object, not an equal one.
        """
        return isinstance(other, _Same) and other.obj is self.obj

    def __hash__(self):
        """Hash by identity, consistently with `__eq__`.

        Returns:
            The object's `id()`.
        """
        return id(self.obj)

    def __repr__(self):
        """Name the object's type and id, for a failure message.

        Returns:
            The representation.
        """
        return f"<{type(self.obj).__name__} {getattr(self.obj, 'id', '')!s}>".strip()


class _RecordingWidget:
    """Stands in for the MapLibre widget, recording every call the map's queue makes on it, in order."""

    def __init__(self):
        """Start with nothing recorded."""
        self.calls = []

    def __getattr__(self, method):
        """Return a recorder for any widget method — `add_source`, `add_layer`, `add_control`, ….

        Args:
            method: The method a queue entry calls.

        Returns:
            A callable recording the method's name and each argument, by identity.
        """

        def record(*args, **kwargs):
            """Record one call.

            Args:
                *args: The positional arguments, recorded by identity.
                **kwargs: The keyword arguments, recorded by identity.
            """
            self.calls.append(
                (
                    method,
                    *(_Same(arg) for arg in args),
                    *((key, _Same(value)) for key, value in sorted(kwargs.items())),
                )
            )

        return record


class WebContract(RendererContract):
    """The web tier's adapter for the shared renderer contract (#305).

    **`apply` does not reach this tier's engine, nor the figure it reports.** The widget is built from the
    map's queue, `_queued`, which only the builders and `remove_layer` write. `Renderer.apply` reconciles the
    renderer's own record and stops there: a layer it adds is never queued, so the widget never adds it, and
    a layer it removes stays queued (review M1). That is the decision for this wave — the renderer's module
    docstring says so — and wiring `apply` through the map's own state is later work. So
    `apply_reaches_engine` is `False` and the engine checks skip here rather than pass whatever `apply` did;
    `apply_reaches_description` is `False` because `apply_figure` calls the renderer, which never touches
    `figure_spec`. The rollback and the redraw guard are held to the renderer's record instead, which is the
    one thing `apply` changes here.
    """

    backend = "web"
    apply_reaches_engine = False
    apply_reaches_description = False

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
        """Return what the widget is built from: the queue, replayed the way the widget build replays it.

        `_build_map_widget` hands every queue entry to `_apply_layer`, so that is what this does, into a
        widget that records each call. A described layer resolves through the renderer's record only because
        its marker is in the queue — a layer the record holds and the queue does not is never added, which is
        exactly what reading the record directly could not see. No check reads this while
        `apply_reaches_engine` is `False`; it is what they will read once `apply` reaches the queue.

        Args:
            tier: The map.

        Returns:
            Every widget call, in order, with its arguments compared by identity — so a layer drawn again from
            the same description reads as a change.
        """
        widget = _RecordingWidget()
        for entry in tier._queued:
            tier._apply_layer(widget, entry)
        return tuple(widget.calls)

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
