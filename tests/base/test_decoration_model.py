"""Where a decoration lives: bands, furniture, guides and the tooltip channel (DE-48, #292).

The code and the design disagreed about where a decoration belongs. A graticule was a sourceless layer here and
an anchored item there; every tier sorted its decorations into bands its own way, with `lakes` under the data on
static and over it on interactive; and nothing in the spec could hold a legend's placement or a tooltip at all.
These cover the four homes that replace that: a layer in its band, furniture on the panel, a guide on the
encoding, and the panel's own title.
"""

import json
from dataclasses import fields

import pytest

from digitalearth.base.registry import (
    KIND_BANDS,
    FurnitureInfo,
    KindInfo,
    band_of,
    furniture_info,
    furniture_kinds,
    kind_info,
    kinds,
    register_furniture,
    temporary_furniture,
    temporary_kind,
)
from digitalearth.base.spec import (
    CHANNELS,
    DataRef,
    Encoding,
    FigureSpec,
    Furniture,
    Guide,
    LayerSpec,
    LayerTree,
    LegendSpec,
    PanelSpec,
    Scale,
    Symbology,
    Viewport,
)


class TestKindBands:
    """Every kind says where it is drawn relative to the data."""

    def test_every_registered_kind_declares_a_band(self):
        """A kind with no band could not be placed, so the registry has none.

        Test scenario:
            The band is what `LayerTree.add` reads; a kind outside the vocabulary would place its layers
            wherever the caller happened to add them, which is the behaviour this replaces.
        """
        unplaced = [name for name in kinds() if kind_info(name).band not in KIND_BANDS]
        assert unplaced == [], f"kinds with no band in {KIND_BANDS}: {unplaced}"

    @pytest.mark.parametrize(
        "kind, band",
        [
            ("basemap", "underlay"),
            ("land", "underlay"),
            ("lakes", "underlay"),
            ("graticule", "reference"),
            ("raster", "data"),
            ("choropleth", "data"),
            ("text", "overlay"),
            ("labels", "overlay"),
            ("coastlines", "overlay"),
        ],
    )
    def test_the_built_in_vocabulary_is_sorted_by_what_it_draws(self, kind, band):
        """The bands settle the disagreements the tiers had: ground cover under, line geography over.

        Args:
            kind: The registered kind under test.
            band: The band it belongs to.

        Test scenario:
            `lakes` was drawn under the data on static and over it on interactive; it is ground cover, so it is
            an underlay. Coastlines and borders are drawn over the field by both 2-D tiers, where a thin line
            stays visible over an opaque raster, so they are overlays rather than reference.
        """
        assert band_of(kind) == band, f"{kind!r} is in the {band_of(kind)!r} band"

    def test_an_unregistered_kind_is_drawn_among_the_data(self):
        """A plugin's kind, or one whose plugin is not installed, is data rather than ground cover."""
        assert band_of("mypkg:hexbin") == "data", band_of("mypkg:hexbin")

    def test_a_kind_may_declare_its_own_band(self):
        """A plugin that draws a halo over the data registers it as an overlay."""
        with temporary_kind(KindInfo("demo:halo", "points", "a halo", "overlay")):
            placed = band_of("demo:halo")
        assert placed == "overlay", placed

    def test_a_band_nothing_could_draw_is_refused_where_the_entry_is_built(self):
        """A misspelt band is caught at registration, not when a layer of that kind is added."""
        with pytest.raises(ValueError, match="band must be one of"):
            KindInfo("demo:halo", "points", "a halo", "middle")


class TestBandedTree:
    """`LayerTree.add` and `move` keep every layer in its band."""

    def test_a_basemap_added_last_is_still_drawn_first(self):
        """Ground cover goes to the bottom however late it arrives — the web tier's `add_underlay`, in the tree.

        Test scenario:
            The reproduction: a caller adds the data, then remembers the basemap. Appending would have drawn the
            tiles over the raster and hidden it.
        """
        tree = (
            LayerTree()
            .add(LayerSpec("dem", "raster"))
            .add(LayerSpec("tiles", "basemap"))
        )
        assert tree.ids == ("tiles", "dem"), tree.ids

    def test_a_label_added_first_is_still_drawn_last(self):
        """A place name stays over the field that arrives after it."""
        tree = (
            LayerTree().add(LayerSpec("place", "text")).add(LayerSpec("dem", "raster"))
        )
        assert tree.ids == ("dem", "place"), tree.ids

    def test_the_four_bands_sort_a_map_whatever_order_it_was_built_in(self):
        """A map built back to front comes out in draw order.

        Test scenario:
            One layer per band, added top band first. Underlay, reference, data, overlay is the order every
            tier was hand-rolling.
        """
        tree = LayerTree()
        for layer in (
            LayerSpec("place", "text"),
            LayerSpec("dem", "raster"),
            LayerSpec("grid", "graticule"),
            LayerSpec("tiles", "basemap"),
        ):
            tree = tree.add(layer)
        assert tree.ids == ("tiles", "grid", "dem", "place"), tree.ids

    def test_two_layers_of_one_band_keep_the_order_they_arrived_in(self):
        """Within a band, adding still appends: the second raster is drawn over the first."""
        tree = (
            LayerTree()
            .add(LayerSpec("dem", "raster"))
            .add(LayerSpec("slope", "raster"))
        )
        assert tree.ids == ("dem", "slope"), tree.ids

    def test_a_position_inside_the_band_is_honoured(self):
        """A caller who knows where in its band a layer goes can still say so."""
        tree = (
            LayerTree()
            .add(LayerSpec("tiles", "basemap"))
            .add(LayerSpec("dem", "raster"))
            .add(LayerSpec("slope", "raster"), index=1)
        )
        assert tree.ids == ("tiles", "slope", "dem"), tree.ids

    def test_a_position_outside_the_band_is_refused(self):
        """Asking for a basemap above the data names the band's range rather than clamping."""
        tree = LayerTree().add(LayerSpec("dem", "raster"))
        with pytest.raises(
            IndexError, match="the underlay band runs from position 0 to 0"
        ):
            tree.add(LayerSpec("tiles", "basemap"), index=1)

    def test_a_layer_moves_within_its_band(self):
        """Reordering the data layers is what a layer switcher does, and it still works."""
        tree = (
            LayerTree()
            .add(LayerSpec("tiles", "basemap"))
            .add(LayerSpec("dem", "raster"))
            .add(LayerSpec("slope", "raster"))
        )
        assert tree.move("dem", 2).ids == ("tiles", "slope", "dem"), tree.ids

    def test_a_layer_cannot_be_moved_out_of_its_band(self):
        """Dragging the basemap to the top would hide the map under its own tiles."""
        tree = (
            LayerTree()
            .add(LayerSpec("tiles", "basemap"))
            .add(LayerSpec("dem", "raster"))
        )
        with pytest.raises(
            IndexError, match="the underlay band runs from position 0 to 0"
        ):
            tree.move("tiles", -1)

    def test_a_move_the_tree_could_not_hold_is_still_refused_by_position(self):
        """A position outside the tree is refused before the band is consulted."""
        tree = LayerTree().add(LayerSpec("dem", "raster"))
        with pytest.raises(IndexError, match="the tree holds 1 layers"):
            tree.move("dem", 4)

    def test_a_tree_built_out_of_band_order_places_a_layer_above_its_ground(self):
        """A hand-built tree is not reordered, and a new layer lands above the lower bands it belongs over.

        Test scenario:
            Nothing stops `LayerTree((label, basemap))`, and `from_dict` reads whatever was stored. The bounds
            then meet at the topmost lower-band layer, which keeps the half of the order that still means
            something: the raster is drawn over the tiles.
        """
        tree = LayerTree((LayerSpec("place", "text"), LayerSpec("tiles", "basemap")))
        assert tree.add(LayerSpec("dem", "raster")).ids == (
            "place",
            "tiles",
            "dem",
        ), tree.ids


class TestFurniture:
    """Items anchored to the panel frame, which have no place on the ground."""

    def test_an_item_takes_the_corner_its_kind_is_registered_with(self):
        """A scale bar goes bottom left without the panel repeating it."""
        assert Furniture("scale_bar").anchor == "bottom-left", Furniture(
            "scale_bar"
        ).anchor

    def test_a_caller_can_put_it_somewhere_else(self):
        """The registered corner is a default, not a rule."""
        item = Furniture("navigation", anchor="bottom-right")
        assert item.anchor == "bottom-right", item.anchor

    def test_a_name_no_tier_knows_is_refused(self):
        """A typo is caught where the figure is written, not silently dropped at draw time."""
        with pytest.raises(ValueError, match="no furniture 'scalebar' is registered"):
            Furniture("scalebar")

    def test_a_corner_no_tier_could_place_is_refused(self):
        """The four corners are what matplotlib, MapLibre and PyVista all name."""
        with pytest.raises(ValueError, match="anchor must be one of"):
            Furniture("scale_bar", anchor="middle")

    def test_options_are_held_read_only(self):
        """A caller who keeps the dict cannot re-configure the item afterwards."""
        given = {"units": ["km", "mi"]}
        item = Furniture("scale_bar", options=given)
        given["units"] = ["nm"]
        assert item.options["units"] == ("km", "mi"), item.options

    def test_options_that_are_not_a_mapping_are_refused(self):
        """A list of pairs is a caller who meant a dict, and would read as option names."""
        with pytest.raises(ValueError, match="options must be a mapping"):
            Furniture("scale_bar", options=[("units", "km")])

    def test_an_option_name_that_is_not_a_name_is_refused(self):
        """An option keyed by a number could not be handed to any tier as a keyword."""
        with pytest.raises(ValueError, match="option names must be non-empty strings"):
            Furniture("scale_bar", options={1: "km"})

    def test_a_stored_item_reads_back_as_the_item_it_was_written_from(self):
        """The JSON round trip is what a saved figure depends on."""
        written = json.loads(
            json.dumps(Furniture("time_slider", options={"step": 3}).to_dict())
        )
        assert Furniture.from_dict(written) == Furniture(
            "time_slider", anchor="bottom-left", options={"step": 3}
        ), written

    def test_the_stored_form_pins_the_corner_it_was_placed_in(self):
        """A figure read back years later places its furniture where it was placed when it was written."""
        assert Furniture("fullscreen").to_dict() == {
            "kind": "fullscreen",
            "anchor": "top-right",
        }, Furniture("fullscreen").to_dict()

    def test_a_plugin_can_register_its_own_item(self):
        """The registry is the growth axis here too: an inset map is a row, not a field."""
        with temporary_furniture(FurnitureInfo("demo:inset", "top-left", "an inset")):
            placed = Furniture("demo:inset").anchor
        assert placed == "top-left", placed

    def test_a_name_that_could_not_be_looked_up_is_refused(self):
        """A furniture name follows the kind grammar, so it can be stored and read back."""
        with pytest.raises(ValueError, match="is not a furniture name"):
            FurnitureInfo("Scale Bar", "top-left", "a scale bar")

    def test_an_entry_anchored_nowhere_is_refused(self):
        """An anchor outside the four corners could not be placed by any tier."""
        with pytest.raises(ValueError, match="anchor must be one of"):
            FurnitureInfo("demo:compass", "middle", "a compass")

    def test_an_entry_with_no_description_is_refused(self):
        """The registry is listed to callers, so every row says what it is."""
        with pytest.raises(ValueError, match="needs a description"):
            FurnitureInfo("demo:blank", "top-left", "  ")

    def test_registering_something_that_is_not_an_entry_is_refused(self):
        """A bare name carries no anchor, so it could not be placed."""
        with pytest.raises(TypeError, match="needs a FurnitureInfo"):
            register_furniture("scale_bar")

    def test_a_second_meaning_for_one_name_is_refused(self):
        """Two plugins claiming one name would otherwise overwrite each other silently."""
        with pytest.raises(ValueError, match="is already registered"):
            register_furniture(FurnitureInfo("measure", "top-right", "another meaning"))

    def test_registering_the_same_entry_again_is_accepted(self):
        """A module imported twice registers the same row twice, which is not a clash."""
        register_furniture(furniture_info("measure"))
        assert furniture_info("measure").anchor == "top-left", furniture_info("measure")

    def test_a_temporary_entry_needs_an_entry_too(self):
        """The block form refuses what the permanent form refuses."""
        with pytest.raises(TypeError, match="needs a FurnitureInfo"):
            with temporary_furniture("scale_bar"):
                pass

    def test_swapping_a_built_in_for_a_block_puts_it_back(self):
        """A test that moves the scale bar must not move it for the next test."""
        with temporary_furniture(FurnitureInfo("scale_bar", "top-right", "a stand-in")):
            inside = furniture_info("scale_bar").anchor
        assert (inside, furniture_info("scale_bar").anchor) == (
            "top-right",
            "bottom-left",
        ), "the built-in must come back"

    def test_the_built_in_items_cover_what_the_tiers_already_draw(self):
        """Scale bar, north arrow, attribution, navigation, fullscreen, switcher, slider, measure."""
        missing = {
            "scale_bar",
            "north_arrow",
            "attribution",
            "navigation",
            "fullscreen",
            "layer_switcher",
            "time_slider",
            "measure",
        }.difference(furniture_kinds())
        assert missing == set(), (
            f"furniture the tiers draw but nothing registers: {missing}"
        )

    def test_a_registered_item_says_what_it_is(self):
        """The registry is self-describing, as the kind registry is."""
        assert furniture_info("measure").doc.startswith("a tool that measures"), (
            furniture_info("measure").doc
        )


class TestPanelFurniture:
    """`PanelSpec.furniture` — the panel's own anchored items."""

    def test_a_panel_holds_its_items_in_order(self):
        """Two items, each in its own corner, recorded as given."""
        panel = PanelSpec(
            "main", furniture=(Furniture("scale_bar"), Furniture("north_arrow"))
        )
        assert [item.kind for item in panel.furniture] == [
            "scale_bar",
            "north_arrow",
        ], panel.furniture

    def test_one_kind_twice_is_refused(self):
        """Two scale bars in one corner is a caller who added the same item twice."""
        with pytest.raises(ValueError, match="lists furniture \\['scale_bar'\\] more"):
            PanelSpec(
                "main",
                furniture=(
                    Furniture("scale_bar"),
                    Furniture("scale_bar", anchor="top-left"),
                ),
            )

    def test_something_that_is_not_an_item_is_refused(self):
        """A bare kind name is not an item, and would have no anchor to place."""
        with pytest.raises(ValueError, match="furniture must be Furniture items"):
            PanelSpec("main", furniture=("scale_bar",))

    def test_a_panel_round_trips_through_json_with_its_furniture(self):
        """What a figure stores is what it reads back."""
        panel = PanelSpec(
            "main",
            Viewport(4326),
            layers=("dem",),
            furniture=(Furniture("attribution", options={"text": "CC-BY"}),),
        )
        rebuilt = PanelSpec.from_dict(json.loads(json.dumps(panel.to_dict())))
        assert rebuilt == panel, rebuilt.to_dict()

    def test_a_panel_stored_before_this_change_still_loads(self):
        """A figure written when panels had no furniture reads back as a panel with none."""
        stored = {"id": "main", "viewport": {"crs": 3857}, "layers": ["dem"]}
        assert PanelSpec.from_dict(stored).furniture == (), PanelSpec.from_dict(
            stored
        ).furniture


class TestGuides:
    """What explains an encoding lives on the encoding."""

    def test_an_encoding_carries_its_guide(self):
        """A colorbar's title and corner sit with the channel they explain."""
        encoding = Encoding.by_field(
            "color",
            "dem",
            scale=Scale.from_limits(0.0, 100.0),
            guide=Guide(title="Elevation (m)", anchor="bottom-right"),
        )
        assert (encoding.guide.title, encoding.guide.anchor) == (
            "Elevation (m)",
            "bottom-right",
        ), encoding.guide

    def test_a_guide_can_switch_itself_off(self):
        """A channel chosen for contrast rather than meaning needs no legend."""
        assert (
            Encoding.constant("color", "#444", guide=Guide(show=False)).guide.show
            is False
        ), "show=False must survive onto the encoding"

    def test_a_guide_round_trips_through_json(self):
        """An encoding's stored form carries the guide, and reads back equal."""
        encoding = Encoding.by_field("size", "pop", guide=Guide(title="People"))
        rebuilt = Encoding.from_dict(json.loads(json.dumps(encoding.to_dict())))
        assert rebuilt == encoding, rebuilt.to_dict()

    def test_an_encoding_without_a_guide_stores_nothing_extra(self):
        """The field is additive: an encoding that carries no guide is the dict it was before."""
        assert "guide" not in Encoding.constant("opacity", 0.5).to_dict(), (
            Encoding.constant("opacity", 0.5).to_dict()
        )

    def test_a_default_guide_stores_an_empty_dict(self):
        """Only what was asked for is written, so a guide costs nothing in JSON."""
        assert Guide().to_dict() == {}, Guide().to_dict()

    def test_a_guide_writes_what_it_was_asked_for(self):
        """Each field is stored under its own key, so a stored guide reads as it was written."""
        stored = Guide(show=False, title="Rainfall", anchor="top-left").to_dict()
        assert stored == {
            "show": False,
            "title": "Rainfall",
            "anchor": "top-left",
        }, stored

    @pytest.mark.parametrize(
        "guide, message",
        [
            ({"show": 1}, "show must be True or False"),
            ({"title": " "}, "title must be a non-empty string"),
            ({"anchor": "middle"}, "anchor must be one of"),
        ],
    )
    def test_a_guide_that_could_not_be_drawn_is_refused(self, guide, message):
        """Each field is checked where the guide is built.

        Args:
            guide: The keyword that is wrong.
            message: What the refusal must say.
        """
        with pytest.raises(ValueError, match=message):
            Guide(**guide)

    def test_something_that_is_not_a_guide_is_refused(self):
        """A bare title would be read as a guide with no placement and no switch."""
        with pytest.raises(ValueError, match="guide must be a Guide or None"):
            Encoding.constant("color", "#f00", guide="Elevation")

    def test_the_legend_content_type_is_untouched(self):
        """`LegendSpec` stays content-only: placement is the guide's, not the legend's."""
        placement = {"anchor", "show"}.intersection(
            field.name for field in fields(LegendSpec)
        )
        assert placement == set(), f"LegendSpec grew placement fields: {placement}"


class TestTooltipChannel:
    """Per-layer interaction is a channel, not another keyword per builder."""

    def test_the_channel_is_declared(self):
        """`tooltip` is a row in the table, alongside colour and text."""
        assert CHANNELS["tooltip"].kind == "text", CHANNELS["tooltip"]

    def test_a_layer_carries_the_fields_to_show(self):
        """The fields a hover shows are bound like any other channel."""
        symbology = Symbology.of(tooltip=("name", "population"))
        assert symbology.encoding("tooltip").resolve() == ("name", "population"), (
            symbology.encodings
        )

    def test_a_tooltip_can_follow_a_column(self):
        """A single-column tooltip is a field binding, which is what a renderer reads."""
        assert Encoding.by_field("tooltip", "name").field == "name", "field binding"


class TestAFigureWrittenBeforeThisChange:
    """Everything here is additive: an older figure still loads."""

    def test_an_old_figure_dict_equals_one_built_the_new_way(self):
        """The figure a previous release wrote reads back as the figure this release builds.

        Test scenario:
            The DoD's compatibility check: the stored form gained `furniture` on a panel and `guide` on an
            encoding, both written only when set, so a dict without them is complete.
        """
        stored = {
            "schema_version": 1,
            "sources": {"dem": {"uri": "file:///dem.tif"}},
            "layers": {
                "layers": [
                    {"id": "dem", "kind": "raster", "source_id": "dem"},
                    {"id": "roads", "kind": "lines"},
                ]
            },
            "panels": [
                {
                    "id": "map",
                    "viewport": {"crs": 4326},
                    "layers": ["dem", "roads"],
                    "title": "Elevation",
                }
            ],
        }
        built = FigureSpec(
            sources={"dem": DataRef("file:///dem.tif")},
            layers=LayerTree()
            .add(LayerSpec("dem", "raster", source_id="dem"))
            .add(LayerSpec("roads", "lines")),
            panels=(
                PanelSpec(
                    "map", Viewport(4326), layers=("dem", "roads"), title="Elevation"
                ),
            ),
        )
        assert FigureSpec.from_dict(stored) == built, FigureSpec.from_dict(
            stored
        ).to_dict()

    def test_a_stored_order_is_reproduced_rather_than_re_banded(self):
        """Loading a figure draws it as it was drawn, even where the bands would place a layer elsewhere.

        Test scenario:
            A figure written before the bands existed can hold a graticule over the data. Re-banding it on load
            would quietly redraw someone's saved figure, so `from_dict` keeps the order it was given; the bands
            decide where a *new* layer goes.
        """
        stored = {
            "schema_version": 1,
            "layers": {
                "layers": [
                    {"id": "dem", "kind": "raster"},
                    {"id": "grid", "kind": "graticule"},
                ]
            },
            "panels": [{"id": "map", "viewport": {"crs": 4326}}],
        }
        assert FigureSpec.from_dict(stored).layers.ids == ("dem", "grid"), (
            FigureSpec.from_dict(stored).layers.ids
        )


class TestTheDiffSeesDecorations:
    """`FigureSpec.diff` already reports the new homes, because they sit inside what it compares."""

    @staticmethod
    def _figure(symbology=None, furniture=()):
        """Return a one-layer figure, optionally restyled or furnished.

        Args:
            symbology: The layer's symbology, or `None` for the default.
            furniture: The panel's anchored items.

        Returns:
            The figure.
        """
        layer = LayerSpec(
            "dem", "raster", source_id="dem", symbology=symbology or Symbology()
        )
        return FigureSpec(
            sources={"dem": DataRef("file:///dem.tif")},
            layers=LayerTree().add(layer),
            panels=(PanelSpec("map", layers=("dem",), furniture=furniture),),
        )

    def test_a_guide_change_is_a_restyle(self):
        """Adding a colorbar title restyles the layer; it does not rebuild it."""
        before = self._figure(Symbology.of(color=Encoding.by_field("color", "dem")))
        after = self._figure(
            Symbology.of(
                color=Encoding.by_field("color", "dem", guide=Guide(title="m"))
            )
        )
        assert before.diff(after).restyled == ("dem",), before.diff(after)

    def test_added_furniture_is_a_panel_change(self):
        """A scale bar the panel did not have is reported as a changed panel, not a changed layer."""
        difference = self._figure().diff(
            self._figure(furniture=(Furniture("scale_bar"),))
        )
        assert (difference.panels, difference.restyled) == (("map",), ()), difference
