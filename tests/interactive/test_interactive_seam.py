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

#: The description of a plain graticule, for the tests that add a layer by hand under a name they choose.
_GRATICULE = Symbology(props={"via": "graticule", "step": 30, "opts": {}})


def _registered_under(interactive_map):
    """Return the object-registry entries a map registered, straight from the process-global table.

    Args:
        interactive_map: The map whose entries are wanted.

    Returns:
        The sorted registry keys under the map's own namespace. Read from the table rather than from
        `figure_spec.sources`, which only lists the sources of layers the tree still holds.
    """
    from digitalearth.base import registry

    prefix = f"{interactive_map._objects_ns}:"
    return sorted(key for key in registry._OBJECTS if key.startswith(prefix))


def _composed_kinds(interactive_map):
    """Return the kind of each element the map composes, in the order `render()` overlays them.

    Args:
        interactive_map: The map whose drawing is read.

    Returns:
        One kind per entry of `interactive_map.layers`, found by matching the element to the layer it was
        drawn for. Read from the drawing, not from `layer_ids`: the tree sorts by band by construction, so
        an order check against it passes however the elements were actually placed (review M13).
    """
    figure = interactive_map.figure_spec
    kind_of = {
        id(record.element): figure.layers.get(layer_id).kind
        for layer_id, record in interactive_map._renderer.drawn.items()
        if layer_id in figure.layers
    }
    return [kind_of.get(id(element), "?") for element in interactive_map.layers]


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

    def test_every_kind_it_draws_is_a_registered_name(self):
        """A kind nobody registered is a name only this tier knows, so no reader can resolve it.

        Test scenario:
            A figure records `kind` as a value and is read back by whatever opens it — another tier, a
            saved file, the registry's own `band_of`. Four builders here once recorded their own method
            name (`barbs`, `graph`, `hexbin`, `kde`) instead of the registered kind that already meant
            what they drew, so `band_of` answered them from its default rather than from an entry and
            `Capabilities` could not declare them at all. Two builders may still draw one kind — the
            recipe under `via` keeps them apart — but neither may invent a word for it.
        """
        unregistered = sorted(set(DRAWN_KINDS) - set(kinds()))
        assert unregistered == [], (
            f"the tier draws {unregistered}, which no reader of a figure can resolve"
        )

    def test_everything_it_draws_is_declared(self):
        """What a tier draws and what it says it can draw are one list, or the declaration is fiction.

        Test scenario:
            `Capabilities` refuses a kind nobody registered, so while four builders recorded invented
            names the tier could not declare them — it drew four kinds it did not claim and claimed a
            `heatmap` nothing drew. `api.BACKEND_CAPABILITIES` is derived from these declarations, so an
            undeclared kind is a layer the dispatcher will route here and find unbuildable.
        """
        undeclared = sorted(set(DRAWN_KINDS) - CAPABILITIES.kinds)
        assert undeclared == [], f"the tier draws {undeclared} without declaring them"

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
        # The registry itself, not `figure_spec.sources`: that view is filtered to the ids the tree still
        # holds, so it reads empty whether or not the object was let go (review M6).
        assert _registered_under(interactive_map) == [], _registered_under(
            interactive_map
        )

    def test_a_declined_layer_gives_its_name_back(self, new_map, monkeypatch):
        """A caller who names a layer, watches it decline, and names it again gets the name they asked for.

        Args:
            new_map: The map factory.
            monkeypatch: Used to make the drawer decline, and then to stop it declining.

        Test scenario:
            The declined layer's id stayed in the map's set of issued ids, so the retry was suffixed —
            `grid-1` for a layer the caller called `grid`, naming a layer that never existed.
        """
        from digitalearth.interactive import projection

        interactive_map = new_map()
        monkeypatch.setattr(projection, "draw_graticule", lambda *args, **kwargs: None)
        interactive_map.add_element(
            None, kind="graticule", name="grid", symbology=_GRATICULE
        )
        monkeypatch.undo()
        interactive_map.add_element(
            None, kind="graticule", name="grid", symbology=_GRATICULE
        )
        assert interactive_map.layer_ids == ["grid"], interactive_map.layer_ids

    def test_a_declined_keyed_basemap_does_not_keep_its_credential(
        self, new_map, monkeypatch
    ):
        """A key held for a layer nothing draws is a secret held for nothing.

        Args:
            new_map: The map factory.
            monkeypatch: Used to make the tile drawer decline.
        """
        from digitalearth.interactive import decoration

        monkeypatch.setattr(decoration, "draw_tiles", lambda *args, **kwargs: None)
        interactive_map = new_map().tiles(
            "Planet.NICFI", api_key=FAKE_KEY, preset={"date": "2024-01"}
        )
        assert interactive_map._layer_keys == {}, interactive_map._layer_keys

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


def _off_limb(*_args, **_kwargs):
    """Stand in for a drawer whose reprojection places none of the data.

    Args:
        *_args: The drawer's arguments, unused.
        **_kwargs: Its keyword arguments, unused.

    Raises:
        OffLimbError: always, as a warp that lands nowhere in the display CRS does.
    """
    from digitalearth.base.crs import OffLimbError

    raise OffLimbError("none of the data lands in the display CRS")


class TestADrawerThatRaisesLeavesNothingBehind:
    """A drawer runs after the layer is described, so its failure has to take the description with it."""

    def test_a_refused_column_leaves_no_layer_and_no_object(self, new_map, point_fc):
        """The error reaches the caller, and the map is as it was before the call.

        Args:
            new_map: The map factory.
            point_fc: The collection the builder is given.

        Test scenario:
            `hexbin` reads its value column only in the drawer, so a name that matches nothing raises
            there — after `add_element` had described the layer and registered its source. The layer
            stayed in `layer_ids` with no element drawn for it, and its data stayed registered (review H4).
        """
        interactive_map = new_map()
        with pytest.raises(KeyError, match="nope"):
            interactive_map.hexbin(point_fc, column="nope")
        assert interactive_map.layer_ids == [], interactive_map.layer_ids
        assert _registered_under(interactive_map) == [], _registered_under(
            interactive_map
        )

    def test_a_strict_map_raises_and_describes_nothing(
        self, new_map, dataset, monkeypatch
    ):
        """Under `strict=True` an off-limb layer raises — and must not stay described.

        Args:
            new_map: The map factory.
            dataset: A small raster.
            monkeypatch: Used to make the raster drawer land nowhere.
        """
        from digitalearth.base.crs import OffLimbError
        from digitalearth.interactive import raster

        monkeypatch.setattr(raster, "draw_image", _off_limb)
        interactive_map = new_map(strict=True)
        with pytest.raises(OffLimbError):
            interactive_map.image(dataset)
        assert interactive_map.layer_ids == [], interactive_map.layer_ids

    def test_a_skipped_layer_does_not_move_the_layers_after_it(
        self, new_map, dataset, point_fc, monkeypatch
    ):
        """Later layers are placed by the tree's index, so a ghost in the tree misplaces every one of them.

        Args:
            new_map: The map factory.
            dataset: A small raster, which the patched drawer cannot place.
            point_fc: A small point collection.
            monkeypatch: Used to make the raster drawer land nowhere.

        Test scenario:
            The skipped raster stayed in the tree as `raster-1`. Coastlines were then inserted at the index
            the tree gave them — one past the ghost — and the points at the ghost's side, so the drawing
            came out coastlines-then-points: the coastlines under the data the figure said they sat over.
        """
        from digitalearth.interactive import raster

        monkeypatch.setattr(raster, "draw_image", _off_limb)
        interactive_map = new_map().image(dataset).coastlines().points(point_fc)
        assert _composed_kinds(interactive_map) == ["points", "coastlines"], (
            _composed_kinds(interactive_map)
        )


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

    def test_the_basemap_is_drawn_where_the_description_puts_it(self, new_map, dataset):
        """The description has to agree with the drawing about which one is on top.

        Args:
            new_map: The map factory.
            dataset: A small raster.

        Test scenario:
            The basemap is added after the raster, against band order. This read `layer_ids`, which the
            `LayerTree` sorts by band by construction, so it passed with the elements appended in call order
            and the tiles drawn over the raster (review M13). It reads the drawing now.
        """
        interactive_map = new_map().image(dataset).tiles("CartoLight")
        assert _composed_kinds(interactive_map) == ["basemap", "raster"], (
            _composed_kinds(interactive_map)
        )

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

    def test_land_and_ocean_are_drawn_in_the_order_the_description_gives(
        self, new_map, dataset
    ):
        """Under the data, and in the order they were asked for — the band the kinds declare, then call order.

        Args:
            new_map: The map factory.
            dataset: A small raster.

        Test scenario:
            This read where the raster sat in `layer_ids`, the band-sorted tree, so it could not fail. Read
            off the drawing it catches both ways of getting it wrong: appending in call order puts the fills
            over the raster, and inserting every underlay at the front — what the tier did before the tree —
            draws ocean beneath land, the reverse of the order the figure reports (review M13).
        """
        interactive_map = new_map().image(dataset)
        interactive_map.features(land=True, ocean=True)
        assert _composed_kinds(interactive_map) == ["land", "ocean", "raster"], (
            _composed_kinds(interactive_map)
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

    def test_reference_geography_is_drawn_over_the_basemap_and_under_the_data(
        self, new_map, point_fc
    ):
        """A graticule beneath an opaque basemap is invisible; over the data it obscures it.

        Args:
            new_map: The map factory.
            point_fc: A small point collection.

        Test scenario:
            The three are added in exactly the reverse of their band order, and the drawing is read rather
            than `layer_ids`. The old form added them almost in band order and read the tree, which the
            `LayerTree` sorts by construction, so it passed with the elements appended in call order and with
            only the underlays moved to the front (review M13).
        """
        interactive_map = new_map().points(point_fc).graticule().tiles("CartoLight")
        assert band_of("graticule") == "reference", band_of("graticule")
        assert _composed_kinds(interactive_map) == ["basemap", "graticule", "points"], (
            _composed_kinds(interactive_map)
        )


class TestTheDescribedOrderIsTheDrawnOrder:
    """One list decides draw order, or a figure shows something other than what it reports."""

    def test_a_reference_layer_added_last_is_still_composed_under_the_data(
        self, new_map, point_fc
    ):
        """The band decides where a layer is drawn, whenever the builder was called.

        Test scenario:
            `figure_spec` read the band-sorted tree while `_compose` overlaid insertion order, so
            `points(fc).graticule()` *described* the graticule under the points and *drew* it over them.
            The two orders are one construction now: a layer is inserted where the tree puts it.
        """
        interactive_map = new_map()
        interactive_map.points(point_fc)
        interactive_map.graticule()
        drawn = interactive_map._renderer.drawn
        composed = [id(element) for element in interactive_map.layers]
        described = [
            id(drawn[layer_id].element) for layer_id in interactive_map.layer_ids
        ]
        assert composed == described, (
            f"composed {composed} but described {interactive_map.layer_ids}"
        )

    def test_the_graticule_is_drawn_beneath_the_points_it_annotates(
        self, new_map, point_fc
    ):
        """The order itself, spelled out: a reference layer belongs under the data.

        Args:
            new_map: The map factory.
            point_fc: A small point collection.

        Test scenario:
            Spelled out on the drawing. Spelled out on `layer_ids` it could not fail: the tree sorts by band
            whatever order the elements were composed in (review M13).
        """
        interactive_map = new_map()
        interactive_map.points(point_fc)
        interactive_map.graticule()
        assert _composed_kinds(interactive_map) == ["graticule", "points"], (
            _composed_kinds(interactive_map)
        )

    def test_reference_geography_is_composed_over_the_basemap_not_under_it(
        self, new_map
    ):
        """Land added after a basemap must sit on top of it in the overlay, or it is invisible.

        Test scenario:
            Both are underlays, and both used to be inserted at the *front* of the element list, so the
            one added second went beneath the first: `tiles()` then `features(land=True)` put opaque
            land under an opaque basemap, where no viewer could ever see it. Asserted against the
            composed elements rather than the tree, because the tree ordered these two correctly even
            while the drawing did not — reading it alone is a check that cannot fail.
        """
        interactive_map = new_map()
        interactive_map.tiles()
        interactive_map.features(land=True)
        drawn = interactive_map._renderer.drawn
        composed = [id(element) for element in interactive_map.layers]
        kinds_by_element = {
            id(record.element): interactive_map.figure_spec.layers.get(layer_id).kind
            for layer_id, record in drawn.items()
        }
        assert [kinds_by_element[marker] for marker in composed] == [
            "basemap",
            "land",
        ], "land must be composed after the basemap it sits on"


class TestClosingAMapLetsItsDataGo:
    """`close()` is how a figure says it is finished with the data it registered."""

    def test_a_closed_map_can_no_longer_open_the_source_it_registered(self, point_fc):
        """Closing is only observable through the figure: the source stops resolving.

        Args:
            point_fc: The collection the builder is given.

        Test scenario:
            The object table is process-global and holds a strong reference, so a session that builds
            maps kept every dataset they drew for the rest of the process. `close()` drops the entries
            this map made, which is what the figure's source failing to open proves.
        """
        interactive_map = InteractiveMap().points(point_fc)
        source = interactive_map.figure_spec.sources[interactive_map.layer_ids[0]]
        assert source.open() is point_fc, "the source never resolved"
        interactive_map.close()
        with pytest.raises(KeyError):
            source.open()

    def test_closing_a_map_lets_go_of_the_credentials_it_held(self):
        """A closed map has no layer left to draw, so it has no reason to hold a key.

        Test scenario:
            `close()` forgot the map's data and kept `_layer_keys`, so a keyed basemap's API key lived on in
            a map the caller had finished with (review M6).
        """
        interactive_map = InteractiveMap().tiles(
            "Planet.NICFI", api_key=FAKE_KEY, preset={"date": "2024-01"}
        )
        assert FAKE_KEY in interactive_map._layer_keys.values(), (
            "the key was never held"
        )
        interactive_map.close()
        assert interactive_map._layer_keys == {}, interactive_map._layer_keys

    def test_closing_twice_is_quiet(self, point_fc):
        """A caller that closes a map a finalizer already closed must not see an error.

        Args:
            point_fc: The collection the builder is given.

        Test scenario:
            A notebook drops a map by re-running its cell, so a finalizer and an explicit `close()` both
            reach the same figure. Forgetting a namespace twice has to be quiet — and the map has to be
            readable afterwards, since closing releases the data, not the description.
        """
        interactive_map = InteractiveMap().points(point_fc)
        interactive_map.close()
        interactive_map.close()
        assert interactive_map.layer_ids == ["points-1"], interactive_map.layer_ids

    def test_closing_one_map_leaves_another_maps_data_alone(self, point_fc):
        """Each figure registers under its own namespace, so closing is not a global clear.

        Args:
            point_fc: The collection one map draws.

        Test scenario:
            Both maps call their first point layer `points-1` and the object table is keyed by name, so
            without a per-figure namespace the second map's registration re-pointed the first map's
            already-captured source at its own data, and closing either took the other's entry with it.
            Two separately read collections are used so the two sides are told apart by identity — a
            second map drawing a *different kind* would never collide and so could not catch this.
        """
        from pyramids.feature import FeatureCollection

        other_fc = FeatureCollection.read_file("tests/data/points.geojson")
        assert other_fc is not point_fc, "the two maps must draw two distinct objects"
        closed = InteractiveMap().points(point_fc)
        kept = InteractiveMap().points(other_fc)
        source = kept.figure_spec.sources[kept.layer_ids[0]]
        closed.close()
        try:
            assert source.open() is other_fc, (
                "closing one map forgot another map's source"
            )
        finally:
            kept.close()

    def test_the_context_manager_hands_back_the_map_it_was_given(self, point_fc):
        """`with InteractiveMap() as m` has to bind the map, not whatever `__enter__` returned.

        Args:
            point_fc: The collection the builder is given.

        Test scenario:
            An `__enter__` that returned anything else — `None` is the usual slip, since the body is one
            statement — leaves the name bound to something the builders are not on, and every `with`
            block in a caller's notebook fails on its first builder call rather than here.
        """
        built = InteractiveMap()
        with built as bound:
            bound.points(point_fc)
        assert bound is built, "the context manager bound a different object"

    def test_leaving_the_with_block_closes_the_map(self, point_fc):
        """The reason the tier is a context manager at all: the data goes when the block ends.

        Args:
            point_fc: The collection the builder is given.

        Test scenario:
            The source is opened inside the block first, so the check after it cannot pass because the
            source never resolved at all — which is what an assertion on the closed state alone would
            allow.
        """
        with InteractiveMap() as interactive_map:
            interactive_map.points(point_fc)
            source = interactive_map.figure_spec.sources[interactive_map.layer_ids[0]]
            assert source.open() is point_fc, "the source never resolved"
        with pytest.raises(KeyError):
            source.open()

    def test_an_exception_inside_the_block_still_closes_the_map(self, point_fc):
        """`__exit__` closes unconditionally, as a file does, so a raising block leaks nothing.

        Args:
            point_fc: The collection the builder is given.

        Test scenario:
            An `__exit__` that returned early on an exception — or one written as a `finally` around the
            body rather than as the protocol method — would leave the data registered exactly when the
            caller is least likely to notice.
        """
        interactive_map = InteractiveMap().points(point_fc)
        source = interactive_map.figure_spec.sources[interactive_map.layer_ids[0]]
        with pytest.raises(RuntimeError, match="deliberate"), interactive_map:
            raise RuntimeError("deliberate")
        with pytest.raises(KeyError):
            source.open()
