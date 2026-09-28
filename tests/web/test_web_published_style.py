"""What this tier publishes as the caller's own style, builder by builder (review R2-H1/H2/M4/M5/M12, #334).

`Symbology.encodings` is the one style reading another tier — and `to_backend()` at order 33 — can act on, so
what lands there has to be what a caller *asked for* rather than what this tier's builder resolved. The paint
dict is the *resolved* style: a builder writes `circle-radius` whether or not `size=` was passed. So each
builder records the keys it was **given** — `tests/web/test_web_asked_style.py` checks that record — and the
lift publishes those and no others. This module checks the end of that chain, which is the only part another
tier sees.

It is the module that used to test the two tables of measured defaults this replaced, and it keeps every
question that is about the answer rather than about the table. Those tables were exact in one direction only:
a caller who asked for exactly their own builder's default was indistinguishable from one who asked for
nothing, so the channel published nothing — and a *kind with no row* published its whole resolved style,
which is how a bare `basemap()` published `{'opacity': 1.0}`. Both are checks here now. The questions only a
table needed — which kind a row is found by, what keys it lists, what values it lists — went with it.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

import pytest

pytest.importorskip("maplibre", reason="the web tier needs the web environment")

from digitalearth.base.spec import Symbology  # noqa: E402
from digitalearth.web import WebMap  # noqa: E402
from digitalearth.web.renderer import portable_encodings  # noqa: E402

#: The contour levels every contour probe traces, so no probe depends on an auto-resolved level set.
LEVELS = [10.0, 100.0, 1000.0]

#: The DEM the raster and contour builders draw. Relative, as every other test in this suite is.
DEM = "examples/data/acc4000.tif"


def _points():
    """Return a small point frame with one numeric column.

    Returns:
        A five-point GeoDataFrame in EPSG:4326.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    return gpd.GeoDataFrame(
        {"pop": [1.0, 2.0, 3.0, 4.0, 5.0]},
        geometry=[Point(4.0 + step * 0.1, 52.0 + step * 0.1) for step in range(5)],
        crs="EPSG:4326",
    )


def _lines():
    """Return a small line frame.

    Returns:
        A two-line GeoDataFrame in EPSG:4326.
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    return gpd.GeoDataFrame(
        {"pop": [1.0, 2.0]},
        geometry=[
            LineString([(4.0, 52.0), (5.0, 53.0)]),
            LineString([(5.0, 52.0), (6.0, 53.0)]),
        ],
        crs="EPSG:4326",
    )


def _polygons():
    """Return a small polygon frame with one numeric column.

    Returns:
        A two-polygon GeoDataFrame in EPSG:4326.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon

    return gpd.GeoDataFrame(
        {"pop": [1.0, 9.0]},
        geometry=[
            Polygon([(4, 52), (5, 52), (5, 53), (4, 53)]),
            Polygon([(5, 52), (6, 52), (6, 53), (5, 53)]),
        ],
        crs="EPSG:4326",
    )


def _dem():
    """Return the DEM the raster builders draw.

    Returns:
        A pyramids `Dataset`.
    """
    from pyramids.dataset import Dataset

    return Dataset.read_file(DEM)


@dataclass(frozen=True)
class Probe:
    """One builder, and how to ask it the questions below.

    Attributes:
        builder: The `WebMap` method, for the failure messages — the kind is the identity, but "polygons"
            is what a reader has to go and look at.
        unstyled: Draws the builder with no style keyword at all.
        styled: Draws it with an explicit style, spelled the way that builder takes it.
        recorded: Where that builder records its resolved style — `"paint"` for the MapLibre dict,
            `"props"` for the builders that record flat and compile at draw time.
        publishes: What an unstyled call is nonetheless expected to publish. Empty for all but
            `extrusion`, whose `height=` is a required argument and therefore always an ask.
    """

    builder: str
    unstyled: Callable[[Any], Any]
    styled: Callable[[Any, dict], Any]
    recorded: str
    publishes: Mapping[str, Any]


#: Every builder on this tier that records a style the lift can read, keyed by the kind it records under.
#:
#: Still keyed by kind, though nothing looks a row up by it any more: the kind is what tells `polygons` from
#: `choropleth` from filled `contours`, which write the same three paint keys, so it is the only name under
#: which these probes read as three different builders rather than one repeated.
PROBES: Dict[str, Probe] = {
    "points": Probe(
        "points",
        lambda m: m.points(_points()),
        lambda m, style: m.points(_points(), **style),
        "paint",
        {},
    ),
    "lines": Probe(
        "lines",
        lambda m: m.lines(_lines()),
        lambda m, style: m.lines(_lines(), **style),
        "paint",
        {},
    ),
    "contours": Probe(
        "contours",
        lambda m: m.contours(_dem(), levels=LEVELS),
        lambda m, style: m.contours(_dem(), levels=LEVELS, **style),
        "paint",
        {},
    ),
    "polygons": Probe(
        "polygons",
        lambda m: m.polygons(_polygons()),
        lambda m, style: m.polygons(_polygons(), **style),
        "paint",
        {},
    ),
    "choropleth": Probe(
        "choropleth",
        lambda m: m.choropleth(_polygons(), "pop"),
        lambda m, style: m.choropleth(_polygons(), "pop", **style),
        "paint",
        {},
    ),
    "filled_contours": Probe(
        "contours(filled=True)",
        lambda m: m.contours(_dem(), filled=True, levels=LEVELS),
        lambda m, style: m.contours(_dem(), filled=True, levels=LEVELS, **style),
        "paint",
        {},
    ),
    "labels": Probe(
        "labels",
        lambda m: m.labels(_points(), "pop"),
        lambda m, style: m.labels(_points(), "pop", **style),
        "paint",
        {},
    ),
    "heatmap": Probe(
        "heatmap",
        lambda m: m.heatmap(_points()),
        lambda m, style: m.heatmap(_points(), **style),
        "paint",
        {},
    ),
    "extrusion": Probe(
        "extrusion",
        lambda m: m.extrusion(_polygons(), height=10.0),
        lambda m, style: m.extrusion(_polygons(), height=10.0, **style),
        "paint",
        {"height": 10.0},
    ),
    "raster": Probe(
        "field",
        lambda m: m.field(_dem()),
        lambda m, style: m.field(_dem(), **style),
        "props",
        {},
    ),
    "rgb": Probe(
        "rgb_composite",
        lambda m: m.rgb_composite(_dem(), bands=(1, 1, 1)),
        lambda m, style: m.rgb_composite(_dem(), bands=(1, 1, 1), **style),
        "props",
        {},
    ),
    "graticule": Probe(
        "graticule",
        lambda m: m.graticule(),
        lambda m, style: m.graticule(**style),
        "props",
        {},
    ),
    # Added with the tables' deletion, and it is the reason the deletion is a fix rather than a refactor:
    # `basemap` records its opacity flat under a kind neither table had a row for, so it was charged nothing
    # and published `{'opacity': 1.0}` from a bare call — one tier's default crossing as caller intent, the
    # R-H2 finding itself, live. Nothing is charged now, so it publishes what was asked for and nothing else.
    "basemap": Probe(
        "basemap",
        lambda m: m.basemap(),
        lambda m, style: m.basemap(**style),
        "props",
        {},
    ),
}

#: An explicit ask whose value is a default **somewhere else**, and must still be published.
#:
#: Kept exactly as the deleted tables had to answer them, because "somebody else's default is still my ask"
#: is a property of the answer rather than of how it is reached. Each was a real failure once: the first
#: table was keyed by paint property and pooled every builder's defaults, and the second by the key *set*,
#: which cannot tell apart the three builders that write `{fill-opacity, fill-outline-color, fill-color}` or
#: the two that write the line trio — so `polygons(opacity=0.85)` was read as choropleth's default,
#: `choropleth(opacity=0.6)` as polygons', and `lines(width=1.5)` as contours' (review R2-H2). Nothing is
#: compared any more, so all eight are now the same question as any other ask; they stay because a
#: regression here is the one that cost two review rounds.
BORROWED_DEFAULT: Tuple[Tuple[str, str, Any, str, Any], ...] = (
    ("points", "size", 6.0, "size", 6.0),
    ("heatmap", "opacity", 0.9, "opacity", 0.9),
    ("raster", "opacity", 0.6, "opacity", 0.6),
    ("graticule", "opacity", 1.0, "opacity", 1.0),
    ("polygons", "opacity", 0.85, "opacity", 0.85),
    ("choropleth", "opacity", 0.6, "opacity", 0.6),
    ("filled_contours", "opacity", 0.6, "opacity", 0.6),
    ("lines", "width", 1.5, "width", 1.5),
)

#: The loss the subtractive tables took, and what each of those calls publishes now (#334).
#:
#: A caller who asked for exactly **their own** builder's default was indistinguishable from one who asked
#: for nothing, because the figure recorded the resolved value and never the fact that a keyword was passed.
#: Every one of these published `{}` before the builders recorded the ask; each publishes the channel it
#: names now. `extrusion` is not here because its `height=` was always an ask and always published.
OWN_DEFAULT_ASKED_FOR: Tuple[Tuple[str, dict, str, Any], ...] = (
    ("points", {"size": 5.0}, "size", 5.0),
    ("points", {"color": "#3388ff"}, "color", "#3388ff"),
    ("points", {"opacity": 0.9}, "opacity", 0.9),
    ("polygons", {"opacity": 0.6}, "opacity", 0.6),
    ("lines", {"width": 2.0}, "width", 2.0),
    ("contours", {"width": 1.5}, "width", 1.5),
    ("filled_contours", {"opacity": 1.0}, "opacity", 1.0),
    ("choropleth", {"opacity": 0.85}, "opacity", 0.85),
    ("labels", {"color": "#ffffff"}, "color", "#ffffff"),
    ("heatmap", {"opacity": 0.8}, "opacity", 0.8),
    ("raster", {"opacity": 1.0}, "opacity", 1.0),
    ("rgb", {"opacity": 1.0}, "opacity", 1.0),
    ("graticule", {"opacity": 0.6}, "opacity", 0.6),
    ("basemap", {"opacity": 1.0}, "opacity", 1.0),
)


def _last(drawn) -> Any:
    """Return the symbology of the last layer a map described.

    Args:
        drawn: The map.

    Returns:
        Its last layer's `Symbology`.
    """
    figure = drawn.figure_spec
    return figure.layers.get(figure.layers.ids[-1]).symbology


def _published(drawn) -> Dict[str, Any]:
    """Return the **constant** channels the last layer publishes as the caller's own style.

    Constants only, which is the question this module asks: whether a value the builder resolved for itself
    crossed as the caller's intent (#334). A field-driven encoding is not a value at all — it names the
    column the channel varies with, which a classified builder publishes because the caller passed
    `column=`, not because a default leaked. Those are asserted separately, in
    :class:`TestAClassifiedLayerPublishesTheFieldItsColourVariesWith`.

    Args:
        drawn: The map.

    Returns:
        `{channel: value}`, resolved, for the channels bound to a constant.
    """
    symbology = _last(drawn)
    return {
        channel: symbology.encoding(channel).resolve()
        for channel in sorted(symbology.encodings)
        if symbology.encoding(channel).is_constant
    }


def _color_field(drawn) -> Any:
    """Return the column the last layer's colour varies with, or `None` for a flat colour.

    Args:
        drawn: The map.

    Returns:
        The field name of a field-driven `color` encoding, else `None` — which covers both a constant colour
        and no colour encoding at all, since neither names a column.
    """
    encoding = _last(drawn).encoding("color")
    return None if encoding is None or encoding.is_constant else encoding.field


def _resolved(drawn, where: str) -> Mapping[str, Any]:
    """Return the resolved style the builder recorded, from whichever half it records in.

    Args:
        drawn: The map.
        where: `"paint"` or `"props"`.

    Returns:
        The recorded mapping.
    """
    props = dict(_last(drawn).props)
    return dict(props["paint"]) if where == "paint" else props


#: Which builders bind their colour to a column, and which column, from the probe's own bare call.
#:
#: The classifying builders are the ones order 24 gives a portable colour to: each already built a `Scale` to
#: compile its MapLibre expression from, and each now publishes `Encoding.by_field("color", column, scale=…)`
#: from it. `contours` is in here with no `column=` in its probe because it colours the traced level along a
#: ramp unless a flat `color=` is passed — the attribute it traces is the field. The rest bind a constant
#: colour or none, and must stay that way: a flat colour has nothing to vary with, which is what makes a
#: colour key on those layers refusable.
COLOUR_BY_FIELD: Tuple[Tuple[str, Any], ...] = (
    ("choropleth", "pop"),
    ("contours", "level"),
    # The bands between levels carry `level_min`, not `level` — the lower bound of the band a fill covers.
    ("filled_contours", "level_min"),
    ("points", None),
    ("lines", None),
    ("polygons", None),
    ("labels", None),
    ("heatmap", None),
    ("extrusion", None),
    ("raster", None),
    ("rgb", None),
    ("graticule", None),
    ("basemap", None),
)


class TestAClassifiedLayerPublishesTheFieldItsColourVariesWith:
    """The gap order 24 closed: a data-driven fill described the layer not at all (DE-48)."""

    @pytest.mark.parametrize(
        ("kind", "field"),
        COLOUR_BY_FIELD,
        ids=[name for name, _ in COLOUR_BY_FIELD],
    )
    def test_the_colour_channel_names_its_column_or_nothing(self, kind, field):
        """A classified layer says what its colour varies with; a flat one says nothing.

        Args:
            kind: The kind under test.
            field: The column the colour must be bound to, or `None` when it must not be bound to one.

        Test scenario:
            `portable_encodings` reads the resolved paint dict, and a classified `fill-color` there is a
            MapLibre expression — a list — so it published no colour channel at all and the layer crossed to
            another tier with its whole thematic meaning missing. The fix is not to invent a constant from
            the expression (that would describe the layer wrongly) but to publish the binding the builder
            already holds. Asserted over **every** builder, both directions, because "the classified ones
            publish it" is only worth anything if the flat ones still do not — that is what keeps a colour
            key refusable on an unclassified layer.
        """
        drawn = WebMap()
        try:
            PROBES[kind].unstyled(drawn)
            published = _color_field(drawn)
        finally:
            drawn.close()
        assert published == field, (
            f"{PROBES[kind].builder} bound its colour to {published!r}, expected {field!r}"
        )

    @pytest.mark.parametrize("scheme", [None, "quantiles", "categorical"])
    def test_every_classification_shape_carries_the_scale_it_drew_with(self, scheme):
        """The scale on the encoding is the one the swatches and the expression came from.

        Args:
            scheme: The classification shape under test — continuous, graduated, categorical.

        Test scenario:
            The three arms of `_color_expr` each build a `Scale` for their own reasons, and each hands *that*
            object back rather than a second one built from `last_breaks`. A recomputation is how the top
            swatch once read `12.900000000000002` for a ramp drawn to `12.9`. Checked against
            `last_breaks`, which is the independently recorded record of what was drawn, so the two sides of
            the comparison are not one expression twice.
        """
        drawn = WebMap()
        try:
            drawn.choropleth(_polygons(), "pop", scheme=scheme, k=2)
            scale = _last(drawn).encoding("color").scale
            breaks = list(drawn.last_breaks)
        finally:
            drawn.close()
        assert scale is not None, (
            "a classified colour must carry the scale it was drawn with"
        )
        if scheme == "categorical":
            assert list(scale.categories) == breaks, (
                f"the scale's categories {list(scale.categories)} must be the drawn ones {breaks}"
            )
        elif scheme is None:
            assert [scale.vmin, scale.vmax] == [breaks[0], breaks[-1]], (
                f"the ramp's limits {(scale.vmin, scale.vmax)} must span the drawn stops {breaks}"
            )
        else:
            assert list(scale.breaks) == breaks, (
                f"the scale's edges {list(scale.breaks)} must be the drawn ones {breaks}"
            )


class TestAnUnstyledLayerPublishesNothing:
    """The question that catches a row the table is missing (review R2-H1)."""

    @pytest.mark.parametrize("kind", sorted(PROBES))
    def test_a_builder_drawn_bare_publishes_no_channel_it_was_not_asked_for(self, kind):
        """Every builder on this tier, drawn with no style keyword at all.

        Args:
            kind: The kind under test, which is what its row is keyed by.

        Test scenario:
            Two ways of getting this wrong have been shipped. The first guard drew `points()` and nothing
            else, so `fill-opacity: 1.0` — what filled `contours` writes — was in no row and an unstyled
            filled contour published `{'opacity': 1.0}`. The second was structural: a kind the table had no
            row for was charged nothing and published its whole resolved style, which is why a bare
            `basemap()` published `{'opacity': 1.0}` until the record replaced the tables. Nothing is
            charged now — a builder that records no ask publishes no channel — and every builder is still
            drawn, because "publishes nothing" is only worth anything if it is asked of all of them.
            `extrusion` is the one builder that publishes from a bare call, because its `height=` is a
            required argument and so is never a default.
        """
        probe = PROBES[kind]
        drawn = WebMap()
        try:
            probe.unstyled(drawn)
            published = _published(drawn)
        finally:
            drawn.close()
        assert published == dict(probe.publishes), (
            f"an unstyled {probe.builder} published {published}; a value its builder defaulted to is the "
            "tier's business, not a portable ask"
        )


class TestAnExplicitAskIsStillPublished:
    """The other direction: an ask has to reach `encodings`, whatever value it carries."""

    @pytest.mark.parametrize(
        ("kind", "keyword", "value", "channel", "expected"),
        BORROWED_DEFAULT,
        ids=[f"{name}-{keyword}" for name, keyword, _, _, _ in BORROWED_DEFAULT],
    )
    def test_a_value_another_builder_defaults_to_is_still_this_callers_ask(
        self, kind, keyword, value, channel, expected
    ):
        """A number is only a default for the builder that defaults it.

        Args:
            kind: The kind under test.
            keyword: The keyword the ask is written with.
            value: A value some *other* builder on this tier resolves to unasked.
            channel: The declared channel it must be published under.
            expected: The value it must be published with.

        Test scenario:
            The table was keyed by paint property first and by the recorded key set second, and neither
            could tell `polygons` from `choropleth` from filled `contours` — all three write the same three
            keys — so each was charged with the others' defaults and three explicit, non-default asks
            published nothing. The kind is what tells them apart (review R2-H2).
        """
        probe = PROBES[kind]
        drawn = WebMap()
        try:
            probe.styled(drawn, {keyword: value})
            published = _published(drawn)
        finally:
            drawn.close()
        assert published.get(channel) == expected, (
            f"{probe.builder}({keyword}={value!r}) published {published}; {value!r} is another builder's "
            "default, not this one's"
        )

    @pytest.mark.parametrize(
        ("kind", "asked", "channel", "expected"),
        OWN_DEFAULT_ASKED_FOR,
        ids=[f"{name}-{channel}" for name, _, channel, _ in OWN_DEFAULT_ASKED_FOR],
    )
    def test_a_caller_asking_for_their_own_builders_default_publishes_it(
        self, kind, asked, channel, expected
    ):
        """The case no subtraction could answer, and the one this module used to pin as a loss (#334).

        Args:
            kind: The kind under test.
            asked: An explicit ask whose value is that builder's own default.
            channel: The declared channel it must be published under.
            expected: The value it must be published with.

        Test scenario:
            Executed before the change, every one of these published `{}`: the figure held the resolved
            value and no record of the keyword, so an ask for the tier's own default read as no ask and a
            figure carried elsewhere was drawn at *that* tier's default. `basemap` is in the list from the
            other side — it published `{'opacity': 1.0}` whether or not anybody asked, because its kind had
            no row at all.
        """
        probe = PROBES[kind]
        drawn = WebMap()
        try:
            probe.styled(drawn, asked)
            published = _published(drawn)
        finally:
            drawn.close()
        assert published.get(channel) == expected, (
            f"{probe.builder}({asked}) published {published}; the value a caller names is theirs even when "
            "the builder would have used it anyway"
        )


class TestTheRecordIsWhatDecides:
    """Not the value, and not a table: what the builder recorded about the call."""

    @pytest.mark.parametrize("spelling", [5, 5.0])
    def test_the_value_published_is_the_value_written(self, spelling):
        """An `int` and a `float` of one number are one ask, and both publish (review R2-M4).

        Args:
            spelling: `5` or `5.0` — this tier's own default marker size, written both ways.

        Test scenario:
            The comparison that used to decide this was type-strict — `type(value) is type(default)` — so
            the two spellings of one number answered differently. The web tier survived it by coercing
            through `float()` and the interactive tier did not; nothing is compared on either now, so what
            the caller wrote is what is published.
        """
        drawn = WebMap()
        try:
            drawn.points(_points(), size=spelling)
            published = _published(drawn)
        finally:
            drawn.close()
        assert published == {"size": float(spelling)}, (
            f"points(size={spelling!r}) published {published}"
        )

    def test_a_description_that_records_no_ask_publishes_nothing_it_holds(self):
        """Asked of the lift directly, so the fail-closed half does not depend on a builder.

        Test scenario:
            A `Symbology` written by hand — or read out of a figure saved before the record existed — holds
            resolved paint and no record, and must publish nothing rather than publish what it holds. The
            deleted tables failed the other way: a kind with no row published everything.
        """
        held = Symbology(
            props={"paint": {"circle-radius": 7.0, "circle-color": "#f00"}}
        )
        assert portable_encodings(held) == {}, (
            "a description that records no ask published something it merely holds"
        )

    def test_a_layer_publishes_the_channels_its_caller_named_and_no_others(self):
        """One call, two asks out of three channels the builder writes.

        Test scenario:
            The sharpest statement of the change, in one draw: `points` resolves and writes all three of
            `circle-radius`, `circle-opacity` and `circle-color` whatever the caller passed, so the resolved
            paint cannot distinguish them — and every value here is this builder's own default, so no
            subtraction could either.
        """
        drawn = WebMap()
        try:
            drawn.points(_points(), size=5.0, color="#3388ff")
            published = _published(drawn)
        finally:
            drawn.close()
        assert published == {"size": 5.0, "color": "#3388ff"}, (
            f"points(size=5.0, color='#3388ff') published {published}; the opacity was not asked for"
        )
