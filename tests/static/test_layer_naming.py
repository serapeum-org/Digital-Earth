"""Naming and hiding a static layer as it is built (#321, #327).

The static tier drew every layer under a generated id and offered no way to say otherwise: `name=` fell into
the builder's `**kwargs` and went on to cleopatra, which either refused it or styled something with it. The
same was true of `visible=` — a caller could not ask for a layer that starts hidden, so a figure built for a
layer switcher had to be drawn and then hidden, one call per layer, after the fact.

These cover the builder side of that on this tier: the id a caller asks for, what a second caller asking for
the same one gets, and that a layer built hidden is hidden **on the axes** as well as in the description. The
cross-tier half — that the web and interactive tiers answer identically — is in
`tests/base/test_map_conformance.py`; the rule itself is in `tests/base/test_layer.py`.
"""

from typing import Tuple

import matplotlib
import pytest

matplotlib.use("Agg")

import geopandas as gpd
from matplotlib.font_manager import font_family_aliases
from matplotlib.text import Text
from pyramids.feature import FeatureCollection
from shapely.geometry import Point, Polygon

from digitalearth.static import Map
from digitalearth.static.maps.decoration import (
    _font_family_a_name_still_stands_for,
)

#: The name the probes ask for, and the id a second layer asking for it again must be given.
ASKED = "wells"
SUFFIXED = "wells-2"

#: The column the polygon fixture carries, so a `choropleth` has something to colour by.
COLUMN = "pop"

#: A family matplotlib ships with itself, so it resolves on every machine that can import matplotlib.
FONT = "DejaVu Serif"

#: A second shipped family, for the probe that an explicit font keyword outranks the layer's name.
OTHER_FONT = "DejaVu Sans Mono"

#: What `Text.get_fontfamily()` answers when nobody chose a family — matplotlib's `font.family` default.
DEFAULT_FAMILY = "sans-serif"

#: One of matplotlib's generic family aliases — a family name that resolves without being registered under
#: that spelling anywhere. Deliberately not `DEFAULT_FAMILY`: an alias that is also the default would draw
#: the same picture whether or not the name was read as a font, so the case could not fail.
GENERIC_ALIAS = "monospace"

#: The canonical `Text` properties that set a font family. Every keyword matplotlib takes for one is an alias
#: of one of these three.
FAMILY_PROPERTIES = frozenset({"fontfamily", "fontname", "fontproperties"})

#: A name that *parses* as a fontconfig pattern naming a family this machine has — `-` is the size
#: separator there — while naming no family at all. It is also exactly what the id allocator mints for a
#: second layer called `FONT`, so the package generates this shape itself (review R2-H6).
PATTERN_ONLY = f"{FONT}-2"

#: The other half of the same class: `,` separates families in a fontconfig pattern, so this parses too and
#: is no family either.
PATTERN_LIST = f"{FONT}, {ASKED}"


def _family_spellings() -> Tuple[str, ...]:
    """Return every keyword matplotlib accepts for a text artist's font family, without `name`.

    Asked of matplotlib rather than typed out here: a list typed out is what the package itself had, and it
    was short by one — `font_properties=` went unrecognised and the caller's font was dropped (review
    R2-H5). `name` is left out because these two builders spend it on the layer's id, which is the whole
    subject of this file.

    Returns:
        The aliases matplotlib declares for `fontfamily`/`fontname`/`fontproperties`, together with the
        three canonical spellings, sorted so the parametrisation reads the same on every run.
    """
    declared = getattr(Text, "_alias_to_prop", None)
    if declared is None:  # matplotlib < 3.11 declared the mapping the other way round
        declared = {
            alias: prop
            for prop, aliases in getattr(Text, "_alias_map", {}).items()
            for alias in aliases
        }
    aliases = {alias for alias, prop in declared.items() if prop in FAMILY_PROPERTIES}
    return tuple(sorted((aliases | FAMILY_PROPERTIES) - {"name"}))


#: Every spelling the probes below pass a font under, derived once.
FAMILY_SPELLINGS = _family_spellings()


@pytest.fixture
def drawn():
    """Yield an empty map in EPSG:4326, closed on the way out.

    Yields:
        A `Map` whose display CRS matches every fixture below, so no builder reprojects.
    """
    built = Map(crs=4326)
    yield built
    built.close()


def _points() -> FeatureCollection:
    """Return two point features with a numeric column.

    Returns:
        A pyramids `FeatureCollection` in EPSG:4326.
    """
    return FeatureCollection(
        gpd.GeoDataFrame(
            {COLUMN: [1.0, 9.0]},
            geometry=[Point(0.0, 0.0), Point(1.0, 1.0)],
            crs=4326,
        )
    )


def _polygons() -> FeatureCollection:
    """Return two polygon features with a numeric column.

    Returns:
        A pyramids `FeatureCollection` in EPSG:4326.
    """
    return FeatureCollection(
        gpd.GeoDataFrame(
            {COLUMN: [1.0, 9.0]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            ],
            crs=4326,
        )
    )


class TestTheNameACallerGives:
    """`name=` reaches the layer id, across the builder families this tier has."""

    def test_a_vector_builder_files_the_layer_under_the_name(self, drawn):
        """`scatter(name=)` is the id `layer_ids` reports.

        Args:
            drawn: The map under test.
        """
        drawn.scatter(_points(), name=ASKED)
        assert drawn.layer_ids == [ASKED], drawn.layer_ids

    def test_a_raster_builder_does_too(self, dataset):
        """`imshow(name=)` reaches the same funnel through `_field`.

        Args:
            dataset: The committed raster fixture.

        Test scenario:
            `imshow` forwards its keywords to `_field`, so the keyword has to be declared in both places to
            be discoverable in either. A caller reading `help(Map.imshow)` should find it there.
        """
        with Map(crs=dataset.epsg) as built:
            built.imshow(dataset, name=ASKED)
            assert built.layer_ids == [ASKED], built.layer_ids

    def test_a_decoration_builder_does_too(self, drawn):
        """`text(name=)` names a label, which is a layer like any other.

        Args:
            drawn: The map under test.
        """
        drawn.text(0.5, 0.5, "Amsterdam", name=ASKED)
        assert drawn.layer_ids == [ASKED], drawn.layer_ids

    def test_the_graticule_takes_the_name_it_was_created_with(self, drawn):
        """A graticule is one layer per map, so its name is set by the call that creates it.

        Args:
            drawn: The map under test.
        """
        drawn.graticule(spacing=30.0, name=ASKED)
        assert drawn.layer_ids == [ASKED], drawn.layer_ids

    def test_an_unnamed_layer_is_still_numbered_by_its_kind(self, drawn):
        """The default is unchanged: no name asked for, an id generated from the kind.

        Args:
            drawn: The map under test.
        """
        drawn.scatter(_points())
        assert drawn.layer_ids == ["points-1"], drawn.layer_ids


class TestANameUsedTwice:
    """The suffix rule, on the tier whose answer changed (#321)."""

    def test_the_second_layer_is_suffixed(self, drawn):
        """Two layers under one name get one id each, and the second counts from two.

        Args:
            drawn: The map under test.
        """
        drawn.scatter(_points(), name=ASKED)
        drawn.scatter(_points(), name=ASKED)
        assert drawn.layer_ids == [ASKED, SUFFIXED], drawn.layer_ids

    def test_the_suffix_counts_the_name_and_not_the_figure(self, drawn):
        """The change: the second `wells` is `wells-2` however many other layers were drawn first.

        Args:
            drawn: The map under test.

        Test scenario:
            This tier minted a colliding name from the scene-wide id counter, so three unnamed layers ahead
            of the reuse made it `wells-4` — while the web and 3-D tiers answered `wells-2` to the same
            script. Three unnamed layers are drawn here for exactly that reason; the old rule cannot
            produce the expected id.
        """
        drawn.scatter(_points())
        drawn.scatter(_points())
        drawn.scatter(_points())
        drawn.scatter(_points(), name=ASKED)
        drawn.scatter(_points(), name=ASKED)
        assert drawn.layer_ids[-2:] == [ASKED, SUFFIXED], drawn.layer_ids

    def test_a_third_layer_keeps_counting(self, drawn):
        """`wells`, `wells-2`, `wells-3` — never two layers under one id.

        Args:
            drawn: The map under test.
        """
        for _ in range(3):
            drawn.scatter(_points(), name=ASKED)
        assert drawn.layer_ids == [ASKED, SUFFIXED, "wells-3"], drawn.layer_ids


class TestANameThatIsWhatTheGeneratorWouldMint:
    """The other collision: the caller's name lands *on* the generated sequence rather than beside it.

    `TestANameUsedTwice` covers two callers asking for one name. This is the case the suffix rule cannot
    answer, because the caller asks first: `scatter(name="points-1")` takes the very id the next unnamed
    `scatter` would generate. The counter is per-scene and the issued set is per-id, so the generator has
    to step over what is already out — and the cost of it not doing so is not a cosmetic id clash. The
    renderer keys everything it drew by layer id, so a second layer minting an id already in use makes the
    first one unaddressable: `set_visible`, `is_visible` and `remove_layer` would all reach the wrong
    artist, or the only surviving one.
    """

    def test_an_unnamed_layer_steps_over_the_id_a_caller_already_took(self, drawn):
        """The generator skips `points-1` because the caller holds it, and mints `points-2`.

        Args:
            drawn: The map under test.

        Test scenario:
            The kind a `scatter` counts under is `points`, so the first generated id is `points-1` — which
            this caller asks for by name before any unnamed layer exists. Without the skip the second call
            re-mints it and the scene describes two layers under one id.
        """
        drawn.scatter(_points(), name="points-1")
        drawn.scatter(_points())
        assert drawn.layer_ids == ["points-1", "points-2"], drawn.layer_ids

    def test_each_layer_is_still_the_only_artist_its_id_toggles(self, drawn):
        """The half the id list cannot show: two ids, two separately switchable sets of artists.

        Args:
            drawn: The map under test.

        Test scenario:
            `Renderer._drawn` is a mapping from layer id to the artists drawn for it, so a re-minted id
            does not merely read oddly — it overwrites the entry, and the first layer's artists are left on
            the axes with nothing able to reach them. Hiding the caller's own layer and reading the other
            one back is what tells the two apart, because it asks the renderer rather than the description.
        """
        drawn.scatter(_points(), name="points-1")
        drawn.scatter(_points())
        drawn._renderer.set_visible("points-1", False)
        assert drawn._renderer.is_visible("points-2") is True, drawn.layer_ids


class TestALayerBuiltHidden:
    """`visible=False` describes the layer hidden **and** leaves it hidden on the axes (#327)."""

    def test_the_figure_describes_it_hidden(self, drawn):
        """A switcher reads the figure, so the figure has to know.

        Args:
            drawn: The map under test.
        """
        drawn.scatter(_points(), name=ASKED, visible=False)
        assert drawn.figure_spec.layers.get(ASKED).visible is False

    def test_the_artist_is_hidden_too(self, drawn):
        """The other half: the description and the drawing must agree about the same layer.

        Args:
            drawn: The map under test.

        Test scenario:
            A description that says `visible=False` over an artist matplotlib is still drawing is the same
            disagreement the interactive tier had in the opposite direction (#327), so both halves are
            asserted rather than one.
        """
        drawn.scatter(_points(), name=ASKED, visible=False)
        assert drawn._renderer.is_visible(ASKED) is False

    def test_a_layer_nobody_hid_stays_visible(self, drawn):
        """The complement, so the two probes above cannot pass by hiding everything.

        Args:
            drawn: The map under test.
        """
        drawn.scatter(_points(), name=ASKED)
        assert drawn._renderer.is_visible(ASKED) is True

    def test_a_choropleth_built_hidden_is_hidden(self, drawn):
        """The same for the classified builder, which reaches the funnel by its own path.

        Args:
            drawn: The map under test.
        """
        drawn.choropleth(_polygons(), COLUMN, name=ASKED, visible=False)
        assert drawn.figure_spec.layers.get(ASKED).visible is False


class TestTheSquareGraticuleShorthand:
    """`graticule(spacing=)` — the web tier's shorthand, now spelled the same here (#324)."""

    def test_one_spacing_sets_both_steps(self, drawn):
        """`spacing=10` is `lon_step=10, lat_step=10`, which is what the contract declares it means.

        Args:
            drawn: The map under test.
        """
        drawn.graticule(spacing=10.0, name=ASKED)
        props = drawn.figure_spec.layers.get(ASKED).symbology.props
        assert (props["lon_step"], props["lat_step"]) == (10.0, 10.0), props

    def test_the_two_steps_still_win_when_no_spacing_is_given(self, drawn):
        """The shorthand is an override, not a replacement: an asymmetric grid still draws.

        Args:
            drawn: The map under test.
        """
        drawn.graticule(lon_step=20.0, lat_step=5.0, name=ASKED)
        props = drawn.figure_spec.layers.get(ASKED).symbology.props
        assert (props["lon_step"], props["lat_step"]) == (20.0, 5.0), props

    def test_a_step_the_interactive_tier_refuses_still_draws_here(self, drawn):
        """The keyword is shared across the tiers; the values it takes are each engine's (R-L10).

        Args:
            drawn: The map under test.

        Test scenario:
            This tier generates its own meridians, so any positive step draws — and so does the web
            tier. The interactive tier draws Natural Earth's pre-cut `graticules_<n>` layers and honours
            only 1, 5, 10, 15, 20 and 30, which
            `tests/interactive/test_interactive_projection.py::test_unsupported_spacing_raises` pins from
            that side. This pins the static half, so a change that quietly narrows this tier to the
            other's domain is caught rather than read as parity.
        """
        drawn.graticule(spacing=7.5, name=ASKED)
        props = drawn.figure_spec.layers.get(ASKED).symbology.props
        assert (props["lon_step"], props["lat_step"]) == (7.5, 7.5), props


class TestTheFontMatplotlibReadsFromAName:
    """`name=` names the layer without taking matplotlib's font alias away from the caller (R-M2).

    `matplotlib.axes.Axes.text` and `.annotate` document `name=` as an alias of the font family, so
    `text(..., name="DejaVu Serif")` chose a font for as long as this package forwarded the keyword. #321
    gave `name=` to the layer on all thirty-six of this tier's builders — counted as the public methods of
    `Map` whose signature takes one — and on these two, the only two whose artist is a `Text` and so the
    only two of matplotlib's artists that answer to `name` at all, that silently reinterpreted a font as an
    id. These probes read the font **off the artist**, not off the description:
    the defect was a value that looked recorded and never reached the engine.
    """

    def test_a_name_that_is_a_font_family_still_reaches_the_artist(self, drawn):
        """The old spelling keeps choosing the font it always chose.

        Args:
            drawn: The map under test.
        """
        placed = drawn.text(0.5, 0.5, "Amsterdam", name=FONT)
        assert placed.get_fontfamily() == [FONT], placed.get_fontfamily()

    def test_the_layer_is_named_for_it_all_the_same(self, drawn):
        """And the layer is still filed under what the caller asked for, like every other builder.

        Args:
            drawn: The map under test.
        """
        drawn.text(0.5, 0.5, "Amsterdam", name=FONT)
        assert drawn.layer_ids == [FONT], drawn.layer_ids

    def test_annotate_keeps_the_font_too(self, drawn):
        """`annotate` draws an `Annotation`, which is a `Text` and reads `name` the same way.

        Args:
            drawn: The map under test.
        """
        placed = drawn.annotate(0.5, 0.5, "Amsterdam", name=FONT)
        assert placed.get_fontfamily() == [FONT], placed.get_fontfamily()

    def test_a_name_that_is_no_font_leaves_the_font_alone(self, drawn):
        """The complement: an ordinary layer name must not become a font matplotlib then fails to find.

        Args:
            drawn: The map under test.

        Test scenario:
            Forwarding every `name=` on to matplotlib would answer the finding and cost a `findfont: Font
            family 'wells' not found.` on every named label. The name is only read as a font when it
            resolves to one.
        """
        placed = drawn.text(0.5, 0.5, "Amsterdam", name=ASKED)
        assert placed.get_fontfamily() == [DEFAULT_FAMILY], placed.get_fontfamily()

    @pytest.mark.parametrize("spelling", FAMILY_SPELLINGS)
    def test_a_font_the_caller_spelled_any_way_outranks_the_name(self, drawn, spelling):
        """Whichever of matplotlib's six spellings the caller chose a font with, that font is the one drawn.

        Args:
            drawn: The map under test.
            spelling: One keyword matplotlib accepts for a text artist's font family.

        Test scenario:
            The name is a fallback, so a call that says which font it wants decides. The check behind that
            was a hand-written list of spellings and `font_properties=` was not on it: the layer's name
            went on as `fontname=`, which `Text` applies *after* `fontproperties`, so the caller's font
            lost with nothing said (review R2-H5). Parametrising over the whole class is what stops the
            list being short again — the spellings come from matplotlib, not from this file.

            The pair is never an error, either: `Axes.text(fontname=..., fontfamily=...)` draws, with the
            later keyword winning, so what this pins is priority rather than a refusal.
        """
        placed = drawn.text(0.5, 0.5, "Amsterdam", name=FONT, **{spelling: OTHER_FONT})
        assert placed.get_fontfamily() == [OTHER_FONT], (
            spelling,
            placed.get_fontfamily(),
        )

    def test_the_spellings_probed_include_the_one_that_was_missed(self):
        """The derivation above really found the class, rather than quietly finding nothing.

        Test scenario:
            `_family_spellings` reads a private matplotlib mapping, so it can go empty on an upgrade and
            take the parametrisation with it — every case would then pass by not existing.
            `font_properties` is the spelling R2-H5 reports, so its presence is the cheapest proof the
            list is real.
        """
        assert "font_properties" in FAMILY_SPELLINGS, FAMILY_SPELLINGS

    @pytest.mark.parametrize("named", [PATTERN_ONLY, PATTERN_LIST])
    def test_a_name_only_a_pattern_parser_reads_as_a_font_is_not_one(
        self, drawn, named
    ):
        """A name that resolves only because it parses as a *pattern* must not be drawn as a family.

        Args:
            drawn: The map under test.
            named: A layer name `FontProperties(family=...)` parses as a fontconfig pattern.

        Test scenario:
            The check asked `findfont(FontProperties(family=name))`, where a lone string is read as a
            fontconfig pattern — `-` separates the size, `,` the families — and then forwarded the string
            verbatim, where matplotlib reads it as one literal family. So `"DejaVu Serif-2"` resolved, was
            forwarded, and left the artist asking for a family nothing has: the `findfont: ... not found`
            the check exists to avoid, on the very shape the id allocator mints for a duplicate name
            (review R2-H6). What decides and what is forwarded have to be the same thing.
        """
        placed = drawn.text(0.5, 0.5, "Amsterdam", name=named)
        assert placed.get_fontfamily() == [DEFAULT_FAMILY], placed.get_fontfamily()

    def test_the_font_a_name_implies_is_not_written_into_the_figure(self, drawn):
        """A keyword the caller never wrote must not appear in the description of what they asked for.

        Args:
            drawn: The map under test.

        Test scenario:
            The font was synthesised into the layer's `opts` as it was built, so the same source produced
            two figures: on a machine that has the family the layer carried `{'fontname': ...}`, and on
            one that does not it carried no `opts` at all (review R2-M3). The name is what the caller
            wrote and the name is what travels; the font is read from it when the label is drawn.
        """
        drawn.text(0.5, 0.5, "Amsterdam", name=FONT)
        props = drawn.figure_spec.layers.get(FONT).symbology.props
        assert "opts" not in props, props

    def test_a_font_the_caller_did_write_still_travels(self, drawn):
        """The complement: a font the caller *did* ask for is described, like any other plain keyword.

        Args:
            drawn: The map under test.
        """
        drawn.text(0.5, 0.5, "Amsterdam", name=ASKED, fontname=OTHER_FONT)
        props = drawn.figure_spec.layers.get(ASKED).symbology.props
        assert props["opts"] == {"fontname": OTHER_FONT}, props

    def test_a_second_label_asking_for_the_same_font_is_drawn_in_it_too(self, drawn):
        """The suffixed id is the layer's; the font follows the name the caller asked for.

        Args:
            drawn: The map under test.

        Test scenario:
            Reading the font from the layer's id at draw time would give the second label
            `"DejaVu Serif-2"`, which names no family — the R2-H6 shape, reached from the other side.
            It is read from the label, which keeps the caller's spelling.
        """
        drawn.text(0.5, 0.5, "Amsterdam", name=FONT)
        second = drawn.text(0.6, 0.6, "Rotterdam", name=FONT)
        assert second.get_fontfamily() == [FONT], second.get_fontfamily()

    def test_the_second_label_is_still_filed_under_the_suffixed_id(self, drawn):
        """And the id is still suffixed, so the font is not bought by giving two layers one name.

        Args:
            drawn: The map under test.
        """
        drawn.text(0.5, 0.5, "Amsterdam", name=FONT)
        drawn.text(0.6, 0.6, "Rotterdam", name=FONT)
        assert drawn.layer_ids == [FONT, PATTERN_ONLY], drawn.layer_ids

    def test_a_generic_family_alias_is_read_off_a_name_as_well(self, drawn):
        """A family matplotlib resolves by alias is a family, and the registry does not list it.

        Args:
            drawn: The map under test.

        Test scenario:
            `serif`, `monospace` and the rest name families without being registered under those spellings
            — the registry holds `DejaVu Sans Mono`, never `monospace` — so a membership test that only
            read the registry would answer `False` and drop the caller's font. Asked through the artist,
            as every case here is: this reads back as the family the label is actually drawn in.
        """
        placed = drawn.text(0.5, 0.5, "Amsterdam", name=GENERIC_ALIAS)
        assert placed.get_fontfamily() == [GENERIC_ALIAS], placed.get_fontfamily()

    def test_the_alias_probed_is_one_matplotlib_declares_and_is_not_the_default(self):
        """The case above can only fail if the alias differs from the font a bare label already gets.

        Test scenario:
            `get_fontfamily()` answers `DEFAULT_FAMILY` for a label nobody gave a font, and `sans-serif`
            is both the default and a generic alias — so probing with that one would pass whether the name
            was read as a font or ignored entirely. This is the cheapest proof the probe is a real one,
            and it reads the alias set from matplotlib rather than from this file.
        """
        assert GENERIC_ALIAS in font_family_aliases - {DEFAULT_FAMILY}, (
            f"{GENERIC_ALIAS!r} is not an alias matplotlib declares, or it is the default family"
        )

    def test_a_pair_matplotlib_will_refuse_has_no_third_spelling_added_to_it(self):
        """Two aliases of one property: the call is lost, and this package must not make the pile worse.

        Test scenario:
            Asked of the resolution rather than through a draw, because a draw cannot tell the two answers
            apart: `Axes.text(family=..., fontfamily=...)` raises on the pair whatever this decides, with
            the same message, so the only observable difference is whether a `fontname=` the caller never
            wrote was added on the way. Both ways of getting this wrong are caught here — letting the
            refusal escape from inside the check, and reading the unreadable set as "no font chosen" and
            so adding the layer's name to it.
        """
        refused = {"family": OTHER_FONT, "fontfamily": OTHER_FONT}
        assert _font_family_a_name_still_stands_for(FONT, refused) == {}, (
            "a keyword set matplotlib cannot read is not one to add a third spelling to"
        )

    def test_that_pair_is_still_refused_by_matplotlib_in_its_own_words(self, drawn):
        """And the refusal reaches the caller unchanged, rather than being swallowed here.

        Args:
            drawn: The map under test.

        Test scenario:
            The complement of the case above. Deciding to add nothing must not become deciding to drop the
            caller's keywords: the pair is theirs, it is wrong, and matplotlib is the one that says so.
        """
        with pytest.raises(TypeError) as refusal:
            drawn.text(
                0.5,
                0.5,
                "Amsterdam",
                name=FONT,
                family=OTHER_FONT,
                fontfamily=OTHER_FONT,
            )
        assert "aliases of one another" in str(refusal.value), str(refusal.value)


class TestAGraticuleAskedForMoreThanItCanHonour:
    """`graticule()` says so when a call's own arguments cannot all be obeyed (review R2-L3).

    Both were silent. `spacing=` overwrote the two steps the same call had just given, and a `name=` on a
    *replacing* call was accepted and dropped. Neither behaviour is wrong — one grid has one spacing, and
    a layer cannot be renamed under the callers holding its id — but a caller whose argument is discarded
    has to hear it.
    """

    def test_spacing_beside_a_step_says_which_argument_is_dropped(self, drawn):
        """Naming both a step and a spacing warns, and the warning names the argument that lost.

        Args:
            drawn: The map under test.
        """
        with pytest.warns(UserWarning, match="lon_step"):
            drawn.graticule(30.0, 60.0, spacing=10.0, name=ASKED)

    def test_the_spacing_is_still_the_one_that_wins(self, drawn):
        """The warning is a warning, not a change of mind: `spacing` still sets both steps.

        Args:
            drawn: The map under test.
        """
        with pytest.warns(UserWarning):
            drawn.graticule(30.0, 60.0, spacing=10.0, name=ASKED)
        props = drawn.figure_spec.layers.get(ASKED).symbology.props
        assert (props["lon_step"], props["lat_step"]) == (10.0, 10.0), props

    def test_spacing_on_its_own_is_silent(self, drawn, recwarn):
        """A caller who names only `spacing` has nothing discarded, so nothing is said.

        Args:
            drawn: The map under test.
            recwarn: pytest's recorder for whatever the call warns.
        """
        drawn.graticule(spacing=10.0, name=ASKED)
        assert [str(caught.message) for caught in recwarn] == [], recwarn.list

    def test_two_steps_on_their_own_are_silent_too(self, drawn, recwarn):
        """And so is a caller who names only the two steps.

        Args:
            drawn: The map under test.
            recwarn: pytest's recorder for whatever the call warns.
        """
        drawn.graticule(30.0, 60.0, name=ASKED)
        assert [str(caught.message) for caught in recwarn] == [], recwarn.list

    def test_the_two_steps_are_what_was_recorded(self, drawn):
        """The complement of the override: with no `spacing`, the two steps are drawn as given.

        Args:
            drawn: The map under test.
        """
        drawn.graticule(30.0, 60.0, name=ASKED)
        props = drawn.figure_spec.layers.get(ASKED).symbology.props
        assert (props["lon_step"], props["lat_step"]) == (30.0, 60.0), props

    def test_naming_no_step_at_all_still_draws_the_default_grid(self, drawn):
        """A call that names no step draws the 30-degree grid it always did.

        Args:
            drawn: The map under test.

        Test scenario:
            The two steps default to `None` now, so a discarded step can be told from an unwritten one.
            This pins that the documented default came through that change intact.
        """
        drawn.graticule(name=ASKED)
        props = drawn.figure_spec.layers.get(ASKED).symbology.props
        assert (props["lon_step"], props["lat_step"]) == (30.0, 30.0), props

    def test_one_step_alone_still_leaves_the_other_at_the_default(self, drawn):
        """Naming one of the two steps sets that one and leaves the other where it was.

        Args:
            drawn: The map under test.
        """
        drawn.graticule(lat_step=15.0, name=ASKED)
        props = drawn.figure_spec.layers.get(ASKED).symbology.props
        assert (props["lon_step"], props["lat_step"]) == (30.0, 15.0), props

    def test_a_name_on_a_replacing_call_says_it_is_ignored(self, drawn):
        """Renaming the one graticule is not on offer, so a call that tries is told.

        Args:
            drawn: The map under test.
        """
        drawn.graticule(name=ASKED)
        with pytest.warns(UserWarning, match="other-name"):
            drawn.graticule(spacing=20.0, name="other-name")

    def test_the_graticule_keeps_the_id_it_was_created_under(self, drawn):
        """And the id stays the creating call's, which is why the second name has nowhere to go.

        Args:
            drawn: The map under test.
        """
        drawn.graticule(name=ASKED)
        with pytest.warns(UserWarning):
            drawn.graticule(spacing=20.0, name="other-name")
        assert drawn.layer_ids == [ASKED], drawn.layer_ids

    def test_repeating_the_name_it_already_has_is_silent(self, drawn, recwarn):
        """Asking again for the name the layer already carries discards nothing, so nothing is said.

        Args:
            drawn: The map under test.
            recwarn: pytest's recorder for whatever the call warns.
        """
        drawn.graticule(name=ASKED)
        drawn.graticule(spacing=20.0, name=ASKED)
        assert [str(caught.message) for caught in recwarn] == [], recwarn.list


class TestAGraticuleCalledASecondTime:
    """A replacing `graticule()` changes what the call named, and nothing it did not (R-L5)."""

    def test_a_replacing_call_leaves_a_hidden_graticule_hidden(self, drawn):
        """Changing the spacing is not a request to put the grid back on screen.

        Args:
            drawn: The map under test.

        Test scenario:
            The replacing branch passed its own `visible` through, so a caller who changed only the
            spacing had `visible=True` — the *default*, not anything they wrote — reset the flag they
            had set on the creating call.
        """
        drawn.graticule(name=ASKED, visible=False)
        drawn.graticule(spacing=20.0)
        assert drawn.figure_spec.layers.get(ASKED).visible is False

    def test_a_replacing_call_can_still_hide_a_graticule_that_was_showing(self, drawn):
        """The flag is left alone, not ignored: a replacing call that asks for `False` gets it.

        Args:
            drawn: The map under test.

        Test scenario:
            The mirror of the probe above, so "leave the flag alone when it is not named" cannot be
            satisfied by never reading it.
        """
        drawn.graticule(name=ASKED)
        drawn.graticule(spacing=20.0, visible=False)
        assert drawn.figure_spec.layers.get(ASKED).visible is False

    def test_a_replacing_call_that_asks_for_it_does_show_it_again(self, drawn):
        """The complement: naming the flag still sets it, so the fix is not "ignore `visible` on replace".

        Args:
            drawn: The map under test.
        """
        drawn.graticule(name=ASKED, visible=False)
        drawn.graticule(spacing=20.0, visible=True)
        assert drawn.figure_spec.layers.get(ASKED).visible is True

    def test_the_replacement_still_moves_the_spacing(self, drawn):
        """And the replacement itself is unchanged — one graticule, restyled in place.

        Args:
            drawn: The map under test.
        """
        drawn.graticule(name=ASKED, visible=False)
        drawn.graticule(spacing=20.0)
        props = drawn.figure_spec.layers.get(ASKED).symbology.props
        assert (props["lon_step"], props["lat_step"]) == (20.0, 20.0), props
