"""The interactive seam: what the map describes is what the map draws (#300).

The tier held its drawing and nothing else — `self.layers`, a list of built HoloViews elements, and
`self._styles`, keyed by `id(element)`. Neither is a description, so nothing could be written down, read
back or diffed, and the basemap and the Natural-Earth underlays reached the element list by an insert that
recorded nothing at all.

These are the checks that the description and the drawing are now one thing. The companion file
`tests/interactive/test_interactive_renderer.py` reaches the renderer directly, for the paths a builder
does not take; the shared renderer conformance suite (#305) is signed in
`tests/base/test_renderer_conformance.py`.
"""

import json
from dataclasses import replace as with_fields

import pytest

from digitalearth.base.registry import band_of, kinds
from digitalearth.base.spec import LayerSpec, Symbology
from digitalearth.interactive import InteractiveMap
from digitalearth.interactive.capabilities import CAPABILITIES
from digitalearth.interactive.renderer import DRAWN_KINDS, drawer_for

pytest.importorskip(
    "geoviews", reason="the interactive tier needs the interactive environment"
)

#: A credential that is not a real key, used to show one never reaches a written figure.
FAKE_KEY = "FAKE-KEY-NOT-REAL"

#: The kinds this tier draws that the shared vocabulary does not hold, and which its `Capabilities`
#: therefore cannot declare — `Capabilities` refuses a kind nobody registered.
#:
#: Each of the four names a *builder* rather than a registered kind, and the registry already has a name
#: for what it draws: `hexbin` and `kde` are `heatmap` ("point density"), `graph` is `flow` ("flows between
#: places"), and `barbs` is a recipe of `vectors`, whose registry entry reads "interactive
#: vectorfield/barbs" in so many words. Pinned here rather than waved through: this list must shrink as the
#: four builders start recording the registered name, and a *fifth* undeclared kind must fail here rather
#: than be discovered by a reader of a figure that cannot be read back.
UNREGISTERED_KINDS = ("barbs", "graph", "hexbin", "kde")


@pytest.fixture
def point_fc():
    """Return a small point collection read through pyramids.

    Returns:
        A `FeatureCollection` of scattered points, enough for any builder that takes points.
    """
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


@pytest.fixture
def new_map():
    """Yield a factory for maps, closing every map it made on the way out.

    The object registry is process-global and holds strong references, so a map that is never closed
    keeps the data it drew for the rest of the session.

    Yields:
        A callable taking the same keywords as `InteractiveMap` and returning a map.
    """
    built = []

    def make(**options):
        """Return a map that will be closed when the test ends.

        Args:
            **options: Passed straight to `InteractiveMap`.

        Returns:
            The map.
        """
        interactive_map = InteractiveMap(**options)
        built.append(interactive_map)
        return interactive_map

    yield make
    for interactive_map in built:
        interactive_map.close()


class TestWhatTheTierSaysItDraws:
    """`DRAWN_KINDS` is the tier's answer to "what do you draw?", and it has to be true."""

    def test_every_drawable_kind_resolves_to_a_drawer(self):
        """A kind on the list with no drawer raises a bare `KeyError` far from the cause."""
        unresolved = []
        for kind in DRAWN_KINDS:
            try:
                drawer_for(kind)
            except KeyError:
                unresolved.append(kind)
        assert unresolved == [], f"{unresolved} are declared drawable with no drawer"

    def test_the_kinds_it_draws_are_registered_names_bar_four_that_name_builders(self):
        """A kind nobody registered is a name only this tier knows, so no reader can resolve it.

        Test scenario:
            A figure records `kind` as a value and is read back by whatever opens it — another tier, a
            saved file, the registry's own `band_of`. Four of this tier's builders record their own method
            name instead of the registered kind that already means what they draw, so `band_of` answers
            them from its default rather than from an entry, and `Capabilities` cannot declare them at all.
            The list is pinned so it can only shrink.
        """
        unregistered = tuple(sorted(set(DRAWN_KINDS) - set(kinds())))
        assert unregistered == UNREGISTERED_KINDS, (
            f"the drawable kinds nobody registered changed: {unregistered}"
        )

    def test_the_undeclared_kinds_are_exactly_the_unregistered_ones(self):
        """One gap, not two: the declaration is short only where the registry is.

        Test scenario:
            `Capabilities` refuses a kind nobody registered, so a tier cannot declare what it draws until
            the kind exists. Holding the two differences against each other says that registering those
            four names is the whole of the fix, rather than the first half of it.
        """
        undeclared = tuple(sorted(set(DRAWN_KINDS) - CAPABILITIES.kinds))
        assert undeclared == UNREGISTERED_KINDS, (
            f"the tier draws {undeclared} without declaring them"
        )

    def test_a_declared_kind_the_tier_does_not_draw_is_still_declared(self):
        """The declaration may be wider than the drawer table; it must not be narrower.

        Test scenario:
            `custom:holoviews` is declared and never drawn from a description on purpose — a caller's own
            element has no description to rebuild it from. This asserts the direction the contract allows,
            so the check above cannot be read as "the two lists are the same list".
        """
        assert "custom:holoviews" in CAPABILITIES.kinds, sorted(CAPABILITIES.kinds)
        assert "custom:holoviews" not in DRAWN_KINDS, DRAWN_KINDS


class TestAFigureNamesEveryLayerItDrew:
    """The figure is the source of truth, not a label beside one."""

    def test_a_built_map_describes_each_layer_it_drew_exactly_once(
        self, new_map, point_fc, dataset
    ):
        """A layer drawn twice, or drawn and never described, is what a half-open seam looks like.

        Args:
            new_map: The map factory.
            point_fc: A small point collection.
            dataset: A small raster.
        """
        interactive_map = (
            new_map().tiles("CartoLight").graticule().points(point_fc).image(dataset)
        )
        described = list(interactive_map.figure_spec.layers.ids)
        assert len(described) == 4, described
        assert len(set(described)) == len(described), (
            f"a layer is described twice: {described}"
        )
        assert set(interactive_map._renderer.drawn) == set(described), (
            f"drawn {sorted(interactive_map._renderer.drawn)} but described {sorted(described)}"
        )

    def test_the_element_list_holds_one_element_per_described_layer(
        self, new_map, point_fc, dataset
    ):
        """The observable form of the check above: nothing is composed that no layer owns.

        Args:
            new_map: The map factory.
            point_fc: A small point collection.
            dataset: A small raster.
        """
        interactive_map = (
            new_map().tiles("CartoLight").graticule().points(point_fc).image(dataset)
        )
        assert len(interactive_map.layers) == len(interactive_map.layer_ids), (
            f"{len(interactive_map.layers)} elements for {len(interactive_map.layer_ids)} layers"
        )

    def test_each_layer_carries_the_source_it_was_given(self, new_map, point_fc):
        """The figure has to name the data, or it describes a picture of nothing.

        Args:
            new_map: The map factory.
            point_fc: The collection handed to the builder.
        """
        interactive_map = new_map().points(point_fc)
        figure = interactive_map.figure_spec
        layer_id = interactive_map.layer_ids[0]
        assert figure.layers.get(layer_id).source_id == layer_id, figure.layers.get(
            layer_id
        )
        assert figure.sources[layer_id].open() is point_fc, (
            "the source is not the data given"
        )

    def test_a_layer_drawn_from_nothing_records_no_source(self, new_map):
        """A graticule is cut from Natural Earth, so naming a source for it would be a fiction.

        Args:
            new_map: The map factory.
        """
        interactive_map = new_map().graticule()
        figure = interactive_map.figure_spec
        layer_id = interactive_map.layer_ids[0]
        assert figure.layers.get(layer_id).source_id is None, figure.layers.get(
            layer_id
        )
        assert figure.sources == {}, figure.sources

    def test_a_figure_with_no_in_memory_source_can_be_written_down(self, new_map):
        """The seam's point: a description carries what the drawing needs, so it can be saved.

        Args:
            new_map: The map factory.

        Test scenario:
            Before the seam the drawing lived in an element the figure never held, so `to_dict` could
            record only that a layer existed. A graticule and a basemap now write out in full.
        """
        interactive_map = (
            new_map().tiles("CartoLight").graticule(lon_step=15, lat_step=15)
        )
        written = interactive_map.figure_spec.to_dict()
        layers = {layer["id"]: layer for layer in written["layers"]["layers"]}
        graticule = next(
            entry for entry in layers.values() if entry["kind"] == "graticule"
        )
        assert graticule["symbology"]["props"]["step"] == 15, graticule
        basemap = next(entry for entry in layers.values() if entry["kind"] == "basemap")
        assert basemap["symbology"]["props"]["provider"] == "CartoLight", basemap


class TestIdsFollowTheLayer:
    """An id is how a caller addresses a layer again, so it has to come from the layer."""

    def test_a_callers_name_becomes_the_id(self, new_map):
        """`add_element(name=...)` is the tier's one naming surface, and it is taken literally.

        Args:
            new_map: The map factory.
        """
        interactive_map = new_map()
        interactive_map.add_element("an element", name="obs")
        assert interactive_map.layer_ids == ["obs"], interactive_map.layer_ids

    def test_a_second_layer_asking_for_the_same_name_is_suffixed(self, new_map):
        """Two layers sharing an id makes the second unaddressable.

        Args:
            new_map: The map factory.
        """
        interactive_map = new_map()
        interactive_map.add_element("first", name="obs")
        interactive_map.add_element("second", name="obs")
        assert interactive_map.layer_ids == ["obs", "obs-1"], interactive_map.layer_ids

    def test_an_unnamed_layer_is_counted_under_its_kind(self, new_map):
        """A generated id says what the layer is, which is what a reader of `layer_ids` needs.

        Args:
            new_map: The map factory.
        """
        interactive_map = new_map().graticule()
        assert interactive_map.layer_ids == ["graticule-1"], interactive_map.layer_ids


class TestADrawerThatDeclinesLeavesNothingBehind:
    """A described layer nothing draws is exactly the drift this seam removes."""

    def test_a_declined_layer_is_not_described_and_registers_no_source(
        self, new_map, point_fc, monkeypatch
    ):
        """The description is written before the drawer runs, so a refusal has to take it with it.

        Args:
            new_map: The map factory.
            point_fc: The collection the builder is given.
            monkeypatch: Used to make the drawer decline.

        Test scenario:
            `add_element` records the layer and its source, then asks the renderer to draw it. A drawer
            that declines leaves `figure_spec` naming a layer the map cannot draw, and — worse — leaves
            the caller's data in the process-global object registry under that layer's id.
        """
        from digitalearth.interactive import vector

        monkeypatch.setattr(vector, "draw_vector", lambda *args, **kwargs: None)
        interactive_map = new_map().points(point_fc)
        assert interactive_map.layer_ids == [], interactive_map.layer_ids
        assert interactive_map.layers == [], interactive_map.layers
        assert interactive_map._renderer.drawn == {}, interactive_map._renderer.drawn
        assert interactive_map.figure_spec.sources == {}, (
            interactive_map.figure_spec.sources
        )

    def test_the_builder_still_returns_the_map_when_its_drawer_declines(
        self, new_map, point_fc, monkeypatch
    ):
        """Builders chain, so a declined layer must not break the chain it sits in.

        Args:
            new_map: The map factory.
            point_fc: The collection the builder is given.
            monkeypatch: Used to make the drawer decline.
        """
        from digitalearth.interactive import vector

        monkeypatch.setattr(vector, "draw_vector", lambda *args, **kwargs: None)
        interactive_map = new_map()
        assert interactive_map.points(point_fc) is interactive_map


class TestTheDrawingUsesTheRecordedValues:
    """A description that survives but is not *read* is a description in name only."""

    def test_a_raster_is_coloured_by_the_recorded_cmap(self, new_map, dataset):
        """Change the record and the drawing follows, because one is built from the other.

        Args:
            new_map: The map factory.
            dataset: A small raster.

        Test scenario:
            A drawer that ignored the record and recomputed the colormap from the builder's own argument
            would pass every round-trip check — the props would be written and read back correctly and
            the picture would still be wrong. So the record is changed *after* the builder ran.
        """
        interactive_map = new_map().image(dataset, cmap="magma")
        layer_id = interactive_map.layer_ids[0]
        figure = interactive_map.figure_spec
        layer = figure.layers.get(layer_id)
        assert interactive_map._renderer.drawn[layer_id].style["cmap"] == "magma"

        swapped = with_fields(
            layer,
            symbology=Symbology(props={**layer.symbology.props, "cmap": "plasma"}),
        )
        redrawn = interactive_map._renderer.draw_layer(
            with_fields(figure, layers=figure.layers.replace(swapped)), layer_id
        )
        assert redrawn.style["cmap"] == "plasma", redrawn.style

    def test_a_vector_is_styled_by_the_recorded_common_options(self, new_map, point_fc):
        """The same question for the style block every vector kind shares.

        Args:
            new_map: The map factory.
            point_fc: A small point collection.
        """
        interactive_map = new_map().points(point_fc, size=6.0)
        layer_id = interactive_map.layer_ids[0]
        figure = interactive_map.figure_spec
        layer = figure.layers.get(layer_id)
        common = dict(layer.symbology.props["common"])

        bigger = with_fields(
            layer,
            symbology=Symbology(
                props={**layer.symbology.props, "common": {**common, "size": 21.0}}
            ),
        )
        redrawn = interactive_map._renderer.draw_layer(
            with_fields(figure, layers=figure.layers.replace(bigger)), layer_id
        )
        assert redrawn.style["size"] == 21.0, redrawn.style

    def test_the_recorded_options_are_what_reach_the_element(self, new_map, point_fc):
        """`.opts()` writes into HoloViews' global `Store`, so the tier records what it applied.

        Args:
            new_map: The map factory.
            point_fc: A small point collection.

        Test scenario:
            The check above reads the renderer's own note of what it resolved. This one reads the map's
            styling record for the element that came back, which is the value a dashboard override is
            merged over — so a drawer that resolved the right options and applied different ones fails.
        """
        interactive_map = new_map().points(point_fc, size=6.0)
        layer_id = interactive_map.layer_ids[0]
        figure = interactive_map.figure_spec
        layer = figure.layers.get(layer_id)
        common = dict(layer.symbology.props["common"])

        bigger = with_fields(
            layer,
            symbology=Symbology(
                props={**layer.symbology.props, "common": {**common, "size": 21.0}}
            ),
        )
        redrawn = interactive_map._renderer.draw_layer(
            with_fields(figure, layers=figure.layers.replace(bigger)), layer_id
        )
        applied = interactive_map.style_of(redrawn.element)["common"]
        assert applied["size"] == 21.0, applied

    def test_two_graticules_an_octave_apart_draw_different_grids(self, new_map):
        """The source-less kind: nothing but the record says which grid to cut.

        Args:
            new_map: The map factory.
        """
        coarse = new_map().graticule(lon_step=30, lat_step=30)
        fine = new_map().graticule(lon_step=10, lat_step=10)
        coarse_grid = coarse._renderer.drawn["graticule-1"].element.data.name
        fine_grid = fine._renderer.drawn["graticule-1"].element.data.name
        assert coarse_grid == "graticules_30", coarse_grid
        assert fine_grid == "graticules_10", fine_grid


class TestTheUnderlaysAreDescribedAndStillUnderneath:
    """The basemap and the Natural-Earth underlays used to reach the element list by a bare insert."""

    def test_the_basemap_is_described_rather_than_only_inserted(self, new_map, dataset):
        """A basemap nothing describes cannot be saved, reloaded, hidden or reordered.

        Args:
            new_map: The map factory.
            dataset: A small raster.
        """
        interactive_map = new_map().image(dataset).tiles("CartoLight")
        figure = interactive_map.figure_spec
        basemap = next(
            figure.layers.get(layer_id)
            for layer_id in interactive_map.layer_ids
            if figure.layers.get(layer_id).kind == "basemap"
        )
        assert basemap.symbology.props["provider"] == "CartoLight", (
            basemap.symbology.props
        )
        assert basemap.symbology.props["via"] == "tiles", basemap.symbology.props

    def test_the_basemap_element_is_composed_under_the_data(self, new_map, dataset):
        """An underlay basemap drawn over the data hides the figure's subject.

        Args:
            new_map: The map factory.
            dataset: A small raster.
        """
        import geoviews as gv
        import holoviews as hv

        interactive_map = new_map().image(dataset).tiles("CartoLight")
        types = [type(element) for element in interactive_map.layers]
        assert types.index(gv.element.WMTS) < types.index(hv.Image), types

    def test_the_basemap_is_described_under_the_data_too(self, new_map, dataset):
        """The description has to agree with the drawing about which one is on top.

        Args:
            new_map: The map factory.
            dataset: A small raster.
        """
        interactive_map = new_map().image(dataset).tiles("CartoLight")
        figure = interactive_map.figure_spec
        placed = {
            figure.layers.get(layer_id).kind: index
            for index, layer_id in enumerate(interactive_map.layer_ids)
        }
        assert placed["basemap"] < placed["raster"], interactive_map.layer_ids

    def test_each_natural_earth_layer_records_its_own_kind(self, new_map):
        """Rivers recorded as borders is a figure that cannot be read back as what was drawn.

        Args:
            new_map: The map factory.

        Test scenario:
            The five features were drawn by one loop that tagged every one of them `borders`, so a figure
            said "borders" five times for five different pieces of geography.
        """
        interactive_map = new_map().features(
            land=True, ocean=True, borders=True, rivers=True, lakes=True
        )
        figure = interactive_map.figure_spec
        drawn = sorted(
            figure.layers.get(layer_id).kind for layer_id in interactive_map.layer_ids
        )
        assert drawn == ["borders", "lakes", "land", "ocean", "rivers"], drawn

    def test_land_and_ocean_are_composed_under_the_data(self, new_map, dataset):
        """They are opaque fills, so over the raster they would erase it.

        Args:
            new_map: The map factory.
            dataset: A small raster.
        """
        import holoviews as hv

        interactive_map = new_map().image(dataset)
        interactive_map.features(land=True, ocean=True)
        types = [type(element) for element in interactive_map.layers]
        assert types.index(hv.Image) == len(types) - 1, types

    def test_land_and_ocean_are_described_under_the_data(self, new_map, dataset):
        """And the description says the same, through the band their kinds declare.

        Args:
            new_map: The map factory.
            dataset: A small raster.
        """
        interactive_map = new_map().image(dataset)
        interactive_map.features(land=True, ocean=True)
        figure = interactive_map.figure_spec
        raster_at = next(
            index
            for index, layer_id in enumerate(interactive_map.layer_ids)
            if figure.layers.get(layer_id).kind == "raster"
        )
        assert raster_at == len(interactive_map.layer_ids) - 1, (
            interactive_map.layer_ids
        )


class TestAKeyedBasemapKeepsItsCredentialOffTheFigure:
    """A figure is written to JSON and read back; a key written into one leaks with it."""

    def test_the_key_is_held_on_the_map(self, new_map):
        """It still has to reach the drawer, which reads it from the map rather than the figure.

        Args:
            new_map: The map factory.
        """
        interactive_map = new_map().tiles(
            "Planet.NICFI", api_key=FAKE_KEY, preset={"date": "2024-01"}
        )
        layer_id = interactive_map.layer_ids[0]
        assert interactive_map._layer_keys[layer_id] == FAKE_KEY, (
            interactive_map._layer_keys
        )

    def test_the_key_reaches_the_tiles_that_were_drawn(self, new_map):
        """Keeping it off the figure must not mean losing it.

        Args:
            new_map: The map factory.
        """
        interactive_map = new_map().tiles(
            "Planet.NICFI", api_key=FAKE_KEY, preset={"date": "2024-01"}
        )
        url = interactive_map.layers[0].data
        assert FAKE_KEY in url, "the credential never reached the element"

    def test_a_written_figure_does_not_carry_the_key(self, new_map):
        """The whole written figure is searched, not just the field the key would obviously go in.

        Args:
            new_map: The map factory.

        Test scenario:
            The credential used to be one of the values `tiles()` recorded, so `figure_spec.to_dict()` —
            the thing a caller saves next to their notebook — carried a live API key in plain text.
        """
        interactive_map = new_map().tiles(
            "Planet.NICFI", api_key=FAKE_KEY, preset={"date": "2024-01"}
        )
        written = json.dumps(interactive_map.figure_spec.to_dict())
        assert FAKE_KEY not in written, written

    def test_the_symbology_records_the_provider_but_not_the_key(self, new_map):
        """What is left in the description is enough to draw it again, given the key.

        Args:
            new_map: The map factory.
        """
        interactive_map = new_map().tiles(
            "Planet.NICFI", api_key=FAKE_KEY, preset={"date": "2024-01"}
        )
        props = interactive_map.figure_spec.layers.get(
            interactive_map.layer_ids[0]
        ).symbology.props
        assert props["provider"] == "Planet.NICFI", props
        assert FAKE_KEY not in repr(props), props


class TestTheBandOfALayerIsItsKinds:
    """Where a layer is drawn is a property of what it is, not of which builder drew it."""

    @pytest.mark.parametrize("kind", DRAWN_KINDS)
    def test_band_for_places_each_drawable_kind_where_its_kind_says(
        self, kind, new_map
    ):
        """The renderer must not hold a second opinion about draw order.

        Args:
            kind: One of the kinds the tier draws.
            new_map: The map factory.
        """
        renderer = new_map()._renderer
        assert renderer.band_for(LayerSpec("x", kind)) == band_of(kind), kind

    def test_reference_geography_is_described_over_the_basemap_and_under_the_data(
        self, new_map, point_fc
    ):
        """A graticule beneath an opaque basemap is invisible; over the data it obscures it.

        Args:
            new_map: The map factory.
            point_fc: A small point collection.
        """
        interactive_map = new_map().tiles("CartoLight").points(point_fc).graticule()
        placed = interactive_map.layer_ids
        assert band_of("graticule") == "reference", band_of("graticule")
        assert placed.index("graticule-3") < placed.index("points-2"), placed
