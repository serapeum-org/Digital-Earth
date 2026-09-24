"""What this tier publishes as the caller's own style, builder by builder (review R2-H1/H2/M4/M5/M12).

`Symbology.encodings` is the one style reading another tier — and `to_backend()` at order 33 — can act on, so
what lands there has to be what a caller *asked for* rather than what this tier's builder resolved. The
subtraction that makes that true reads :data:`~digitalearth.web.renderer.UNASKED_PAINT` and
:data:`~digitalearth.web.renderer.UNASKED_PROPS`, and until this module existed **nothing tested either of
them**: `grep -rn UNASKED tests/` returned no match, and the one behavioural guard drew `points()` and
nothing else, so it pinned three of the fourteen values and an unstyled filled contour went on publishing
`{'opacity': 1.0}` (review R2-H1/R2-M5).

Every case here draws a real layer through a real builder and reads `figure_spec`, so a row that stops
matching the builder it describes fails here rather than rotting quietly. Four questions are asked of every
row: what an unstyled call publishes, which kind the row is found by, what the builder really writes, and
what an explicit ask publishes.

The tables are keyed by the layer **kind** the builder records under, which is the only thing that can tell
`polygons` from `choropleth` from filled `contours` — all three write the same three paint keys. The last
question is where that shows: three explicit, non-default asks published nothing while a sibling's default
could be charged to them (review R2-H2). What no keying fixes is asked too, by
:meth:`TestAnExplicitAskIsStillPublished.test_a_caller_asking_for_their_own_builders_default_still_publishes_nothing`
— the table is still subtractive, so a caller asking for exactly their own builder's default is
indistinguishable from one who asked for nothing, and that is what #334 stays open for.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

import pytest

pytest.importorskip("maplibre", reason="the web tier needs the web environment")

from digitalearth.web import WebMap  # noqa: E402
from digitalearth.web.renderer import (  # noqa: E402
    DRAWN_KINDS,
    UNASKED_PAINT,
    UNASKED_PROPS,
    UnaskedStyle,
)

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
#: Keyed by **kind** because that is what the table is keyed by: the parametrised checks below walk
#: :data:`~digitalearth.web.renderer.UNASKED_PAINT` and :data:`~digitalearth.web.renderer.UNASKED_PROPS` and
#: look each row's kind up here, so a row naming a kind with no probe fails rather than going unmeasured.
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
}

#: Every row of both tables, as `(kind, row)`, for parametrisation.
ROWS: Tuple[Tuple[str, UnaskedStyle], ...] = tuple(
    (row.kind, row) for row in (*UNASKED_PAINT, *UNASKED_PROPS)
)

#: An explicit ask whose value is a default **somewhere else**, and must still be published.
#:
#: The half of the ledger a wrong row cannot be caught by otherwise: the unstyled probes catch a row that is
#: too small, and only this catches one that is too large.
#:
#: The last four are the cases the *shape*-keyed table could not answer. Three builders wrote
#: `{fill-opacity, fill-outline-color, fill-color}` and two wrote the line trio, so a row matched by the key
#: set could not tell them apart and pooled their defaults: `polygons(opacity=0.85)` was read as
#: choropleth's default, `choropleth(opacity=0.6)` as polygons', and `lines(width=1.5)` as contours' — every
#: one of them an explicit, non-default ask that published nothing. The kind is what tells them apart
#: (review R2-H2).
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

#: The loss a subtractive table takes even when it knows exactly which builder wrote the style.
#:
#: Keying by kind removes the *pooling* — a builder is no longer charged its siblings' defaults — and it does
#: not remove the subtraction. A caller who asks for exactly **their own** builder's default is
#: indistinguishable from one who asked for nothing, because the figure records the resolved value and not
#: the fact that a keyword was passed. That is the whole of what #334 has left to close, and it closes only
#: by recording the ask at the builder rather than by any table. Pinned here so the real cost is measured
#: rather than described (review R2-H2).
OWN_DEFAULT_ASKED_FOR: Tuple[Tuple[str, dict], ...] = (
    ("points", {"size": 5.0}),
    ("polygons", {"opacity": 0.6}),
    ("lines", {"width": 2.0}),
    ("raster", {"opacity": 1.0}),
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
    """Return what the last layer publishes as the caller's own style.

    Args:
        drawn: The map.

    Returns:
        `{channel: value}`, resolved.
    """
    symbology = _last(drawn)
    return {
        channel: symbology.encoding(channel).resolve()
        for channel in sorted(symbology.encodings)
    }


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


class TestAnUnstyledLayerPublishesNothing:
    """The question that catches a row the table is missing (review R2-H1)."""

    @pytest.mark.parametrize("kind", sorted(PROBES))
    def test_a_builder_drawn_bare_publishes_no_channel_it_was_not_asked_for(self, kind):
        """Every builder on this tier, drawn with no style keyword at all.

        Args:
            kind: The kind under test, which is what its row is keyed by.

        Test scenario:
            The old guard drew `points()` and nothing else, so `fill-opacity: 1.0` — what filled
            `contours` writes — was in no row and an unstyled filled contour published `{'opacity': 1.0}`
            as the caller's own style. Drawing every builder is what turns "add the missing row" into a
            check that no row can go missing again. `extrusion` is the one builder that publishes from a
            bare call, because its `height=` is a required argument and so is never a default.
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


class TestEveryRowStillDescribesItsBuilder:
    """The question that catches a row that has rotted — the kind it claims, the keys, and the values."""

    @pytest.mark.parametrize(("kind", "row"), ROWS, ids=[name for name, _ in ROWS])
    def test_a_rows_kind_is_the_kind_its_builder_records_under(self, kind, row):
        """A row is looked up by the kind the builder indexes its layer as, so that has to match.

        Args:
            kind: The row's kind.
            row: The row under test.

        Test scenario:
            The kind is the row's identity, so a row keyed to a kind no builder records subtracts nothing
            at all — and the builder goes straight back to publishing its own defaults as the caller's ask,
            which is the R2-H1 defect one indirection further along.
        """
        probe = PROBES[kind]
        drawn = WebMap()
        try:
            probe.unstyled(drawn)
            figure = drawn.figure_spec
            recorded = figure.layers.get(figure.layers.ids[-1]).kind
        finally:
            drawn.close()
        assert recorded == row.kind, (
            f"{probe.builder} records kind {recorded!r} and its row is keyed {row.kind!r}, so the row "
            "matches nothing it describes"
        )

    @pytest.mark.parametrize(("kind", "row"), ROWS, ids=[name for name, _ in ROWS])
    def test_a_row_knows_every_key_its_builder_writes(self, kind, row):
        """The keys are no longer the identity, and they are still the drift guard they were.

        Args:
            kind: The row's kind.
            row: The row under test.

        Test scenario:
            A builder that gains a key gains a default nobody listed, which is exactly how
            `fill-opacity: 1.0` came to be missing (review R2-H1). Matching moved to the kind, so a new key
            no longer breaks the lookup — it just goes unsubtracted, silently. This is what says so.
        """
        probe = PROBES[kind]
        drawn = WebMap()
        try:
            probe.unstyled(drawn)
            written = frozenset(_resolved(drawn, probe.recorded))
        finally:
            drawn.close()
        assert written == row.keys, (
            f"{probe.builder} records {sorted(written)} and its row knows {sorted(row.keys)}; a key the row "
            "has not seen carries a default nothing subtracts"
        )

    @pytest.mark.parametrize(("kind", "row"), ROWS, ids=[name for name, _ in ROWS])
    def test_every_default_a_row_lists_is_what_its_builder_writes(self, kind, row):
        """The values, measured rather than remembered.

        Args:
            kind: The row's kind.
            row: The row under test.

        Test scenario:
            A row that lists a value the builder no longer writes stops subtracting anything, and the
            builder goes back to publishing its default as an ask. Reading the resolved style back off a
            real call is the only way that cannot drift, and it is what the round-1 table was missing.
        """
        probe = PROBES[kind]
        drawn = WebMap()
        try:
            probe.unstyled(drawn)
            written = _resolved(drawn, probe.recorded)
        finally:
            drawn.close()
        listed = {key: written.get(key) for key in row.defaults}
        assert listed == dict(row.defaults), (
            f"{probe.builder} writes {listed} where its row says {dict(row.defaults)}"
        )


class TestAnExplicitAskIsStillPublished:
    """The question that catches a row that is too large (review R2-H2)."""

    def test_no_two_rows_claim_the_same_kind(self):
        """The kind is the row's identity, so two rows claiming one kind make one of them dead.

        Test scenario:
            Re-points the guard that used to assert which *shapes* builders shared. That question retired
            with the shape: a row is keyed by kind now, so nothing is pooled and there is no shared-shape
            residue left to bound. What the identity needs instead is uniqueness — `unasked_here` answers
            with the first row that matches, so a duplicate kind silently charges one builder the other's
            defaults, which is the very failure the pooled table had.
        """
        claimed = [row.kind for row in (*UNASKED_PAINT, *UNASKED_PROPS)]
        duplicated = sorted({kind for kind in claimed if claimed.count(kind) > 1})
        assert duplicated == [], (
            f"{duplicated} are claimed by more than one row; the lookup answers with the first, so the "
            "others are dead and their builders are charged someone else's defaults"
        )

    @pytest.mark.parametrize(("kind", "row"), ROWS, ids=[name for name, _ in ROWS])
    def test_every_rows_kind_is_one_this_tier_actually_draws(self, kind, row):
        """The other half of the identity: a row keyed to a kind nobody draws subtracts nothing.

        Args:
            kind: The row's kind.
            row: The row under test.

        Test scenario:
            Read against :data:`~digitalearth.web.renderer.DRAWN_KINDS`, which is the tier's own list, so
            a kind renamed under `src/` fails here rather than turning the row off in silence.
        """
        assert row.kind in DRAWN_KINDS, (
            f"{row.builder}'s row is keyed {row.kind!r}, which this tier does not draw: {list(DRAWN_KINDS)}"
        )

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
        ("kind", "asked"),
        OWN_DEFAULT_ASKED_FOR,
        ids=[name for name, _ in OWN_DEFAULT_ASKED_FOR],
    )
    def test_a_caller_asking_for_their_own_builders_default_still_publishes_nothing(
        self, kind, asked
    ):
        """The loss a subtractive table takes however well it knows the builder.

        Args:
            kind: The kind under test.
            asked: An explicit ask whose value is that builder's own default.

        Test scenario:
            Not a promise that this is right — it is what #334 is open for. Keying by kind removed the
            pooling; it did not remove the subtraction, because a figure records the resolved value and
            never the fact that a keyword was passed. So this is the real, remaining cost, measured rather
            than described. The day a builder records its own ask, this fails and the rows come out.
        """
        probe = PROBES[kind]
        drawn = WebMap()
        try:
            probe.styled(drawn, asked)
            published = _published(drawn)
        finally:
            drawn.close()
        assert published == dict(probe.publishes), (
            f"{probe.builder}({asked}) published {published}; a subtractive table cannot tell this from an "
            "unstyled call, and #334 is what closes it"
        )


class TestTheSpellingOfANumberDoesNotDecideWhatIsPublished:
    """`type(value) is type(default)` made `size=5` publish and `size=5.0` not (review R2-M4)."""

    @pytest.mark.parametrize("spelling", [5, 5.0])
    def test_the_tier_s_own_default_is_read_as_a_default_however_it_is_spelled(
        self, spelling
    ):
        """An `int` and a `float` of the same value are one ask, not two.

        Args:
            spelling: `5` or `5.0` — the web tier's own default marker size.

        Test scenario:
            The comparison was type-strict, so the two spellings of one number answered differently. The web
            tier survived it by coercing through `float()`; the interactive tier did not, and the rule is
            shared, so it is checked on both.
        """
        drawn = WebMap()
        try:
            drawn.points(_points(), size=spelling)
            published = _published(drawn)
        finally:
            drawn.close()
        assert published == {}, (
            f"points(size={spelling!r}) published {published}; 5 and 5.0 are one ask"
        )

    @pytest.mark.parametrize("spelling", [12, 12.0])
    def test_a_value_the_tier_does_not_default_is_published_however_it_is_spelled(
        self, spelling
    ):
        """The other direction, so the check above cannot pass by publishing nothing ever.

        Args:
            spelling: `12` or `12.0` — a size no builder on this tier defaults.
        """
        drawn = WebMap()
        try:
            drawn.points(_points(), size=spelling)
            published = _published(drawn)
        finally:
            drawn.close()
        assert published.get("size") == 12.0, (
            f"points(size={spelling!r}) published {published}"
        )
