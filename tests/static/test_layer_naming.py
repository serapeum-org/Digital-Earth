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

import matplotlib
import pytest

matplotlib.use("Agg")

import geopandas as gpd
from pyramids.feature import FeatureCollection
from shapely.geometry import Point, Polygon

from digitalearth.static import Map

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


class TestTheFontMatplotlibReadsFromAName:
    """`name=` names the layer without taking matplotlib's font alias away from the caller (R-M2).

    `matplotlib.axes.Axes.text` and `.annotate` document `name=` as an alias of the font family, so
    `text(..., name="DejaVu Serif")` chose a font for as long as this package forwarded the keyword. #321
    gave `name=` to the layer on all thirty-five builders, and on these two — the only two whose artist is a
    `Text`, and so the only two of matplotlib's artists that answer to `name` at all — that silently
    reinterpreted a font as an id. These probes read the font **off the artist**, not off the description:
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

    def test_a_font_the_caller_spelled_out_outranks_the_name(self, drawn):
        """`fontname=` is the unambiguous spelling, so it decides — and matplotlib is never handed both.

        Args:
            drawn: The map under test.

        Test scenario:
            Passing a family under two of matplotlib's aliases at once raises `TypeError: Got both`, so the
            name is only read as a font when the call has not already said which font it wants.
        """
        placed = drawn.text(0.5, 0.5, "Amsterdam", name=FONT, fontname=OTHER_FONT)
        assert placed.get_fontfamily() == [OTHER_FONT], placed.get_fontfamily()
