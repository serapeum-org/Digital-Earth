"""What this tier publishes as the caller's own style, builder by builder (review R2-H1/H2/M4/M5/M12).

`Symbology.encodings` is the one style reading another tier — and `to_backend()` at order 33 — can act on, so
what lands there has to be what a caller *asked for* rather than what this tier's builder resolved. The
subtraction that makes that true reads :data:`~digitalearth.web.renderer.UNASKED_PAINT` and
:data:`~digitalearth.web.renderer.UNASKED_PROPS`, and until this module existed **nothing tested either of
them**: `grep -rn UNASKED tests/` returned no match, and the one behavioural guard drew `points()` and
nothing else, so it pinned three of the fourteen values and an unstyled filled contour went on publishing
`{'opacity': 1.0}` (review R2-H1/R2-M5).

Every case here draws a real layer through a real builder and reads `figure_spec`, so a row that stops
matching the builder it describes fails here rather than rotting quietly. Three questions are asked of every
row: what an unstyled call publishes, what the builder really writes, and what an explicit ask publishes.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

import pytest

pytest.importorskip("maplibre", reason="the web tier needs the web environment")

from digitalearth.web import WebMap  # noqa: E402
from digitalearth.web.renderer import (  # noqa: E402
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
    """One builder, and how to ask it the three questions.

    Attributes:
        unstyled: Draws the builder with no style keyword at all.
        recorded: Where that builder records its resolved style — `"paint"` for the MapLibre dict,
            `"props"` for the builders that record flat and compile at draw time.
        publishes: What an unstyled call is nonetheless expected to publish. Empty for all but
            `extrusion`, whose `height=` is a required argument and therefore always an ask.
    """

    unstyled: Callable[[Any], Any]
    recorded: str
    publishes: Mapping[str, Any]


#: Every builder on this tier that records a style the lift can read, and how to draw it bare.
#:
#: Keyed by the name the tables' own rows carry, so a row and its probe cannot drift apart — the parametrised
#: checks below walk :data:`~digitalearth.web.renderer.UNASKED_PAINT` and
#: :data:`~digitalearth.web.renderer.UNASKED_PROPS` and look each row's builder up here, and a row naming a
#: builder with no probe fails rather than going unmeasured.
PROBES: Dict[str, Probe] = {
    "points": Probe(lambda m: m.points(_points()), "paint", {}),
    "lines": Probe(lambda m: m.lines(_lines()), "paint", {}),
    "contours": Probe(lambda m: m.contours(_dem(), levels=LEVELS), "paint", {}),
    "polygons": Probe(lambda m: m.polygons(_polygons()), "paint", {}),
    "choropleth": Probe(lambda m: m.choropleth(_polygons(), "pop"), "paint", {}),
    "contours(filled=True)": Probe(
        lambda m: m.contours(_dem(), filled=True, levels=LEVELS), "paint", {}
    ),
    "labels": Probe(lambda m: m.labels(_points(), "pop"), "paint", {}),
    "heatmap": Probe(lambda m: m.heatmap(_points()), "paint", {}),
    "extrusion": Probe(
        lambda m: m.extrusion(_polygons(), height=10.0), "paint", {"height": 10.0}
    ),
    "field": Probe(lambda m: m.field(_dem()), "props", {}),
    "rgb_composite": Probe(
        lambda m: m.rgb_composite(_dem(), bands=(1, 1, 1)), "props", {}
    ),
    "graticule": Probe(lambda m: m.graticule(), "props", {}),
}

#: Every row of both tables, flattened to `(builder name, row)` for parametrisation.
ROWS: Tuple[Tuple[str, UnaskedStyle], ...] = tuple(
    (name, row) for row in (*UNASKED_PAINT, *UNASKED_PROPS) for name in row.builders
)

#: An explicit ask that is **another** builder's default, for the shapes a builder has to itself.
#:
#: The half of the ledger a wrong row cannot be caught by otherwise: the unstyled probes catch a row that is
#: too small, and only this catches one that is too large. Each value below is a number this tier defaults
#: *somewhere* — 6.0 is the interactive tier's marker size, 0.9 is `points`' circle opacity, 0.6 is
#: `graticule`'s opacity and 1.0 is `field`'s — so a table that pooled its rows would swallow every one of
#: them (review R2-H2).
BORROWED_DEFAULT: Tuple[Tuple[str, str, Any, str, Any], ...] = (
    ("points", "size", 6.0, "size", 6.0),
    ("heatmap", "opacity", 0.9, "opacity", 0.9),
    ("field", "opacity", 0.6, "opacity", 0.6),
    ("graticule", "opacity", 1.0, "opacity", 1.0),
)

#: The loss the shape-keyed table still takes, named rather than assumed.
#:
#: Three builders record `{fill-opacity, fill-outline-color, fill-color}` and two record the line trio, and
#: a figure carries the resolved style rather than the builder that wrote it — so within one shape the
#: defaults are still pooled and an explicit ask that happens to be a *shape-mate's* default is read as a
#: default. This is what #334 closes by recording the ask at the builder; until then the residue is bounded
#: by the shape, and pinned here so it cannot quietly widen (review R2-H2).
POOLED_WITHIN_ONE_SHAPE: Tuple[Tuple[str, dict, str], ...] = (
    (
        "polygons",
        {"opacity": 0.85},
        "0.85 is choropleth's fill-opacity, and the two record the same keys",
    ),
    (
        "contours(filled=True)",
        {"opacity": 0.6},
        "0.6 is polygons' fill-opacity, and the two record the same keys",
    ),
    (
        "lines",
        {"width": 1.5},
        "1.5 is contours' line-width, and the two record the same keys",
    ),
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

    @pytest.mark.parametrize("builder", sorted(PROBES))
    def test_a_builder_drawn_bare_publishes_no_channel_it_was_not_asked_for(
        self, builder
    ):
        """Every builder on this tier, drawn with no style keyword at all.

        Args:
            builder: The builder under test.

        Test scenario:
            The old guard drew `points()` and nothing else, so `fill-opacity: 1.0` — what filled
            `contours` writes — was in no row and an unstyled filled contour published `{'opacity': 1.0}`
            as the caller's own style. Drawing every builder is what turns "add the missing row" into a
            check that no row can go missing again. `extrusion` is the one builder that publishes from a
            bare call, because its `height=` is a required argument and so is never a default.
        """
        drawn = WebMap()
        try:
            PROBES[builder].unstyled(drawn)
            published = _published(drawn)
        finally:
            drawn.close()
        assert published == dict(PROBES[builder].publishes), (
            f"an unstyled {builder} published {published}; a value its builder defaulted to is the tier's "
            "business, not a portable ask"
        )


class TestEveryRowStillDescribesItsBuilder:
    """The question that catches a row that has rotted — the value, and the shape it is matched by."""

    @pytest.mark.parametrize(
        ("builder", "row"), ROWS, ids=[f"{name}" for name, _ in ROWS]
    )
    def test_the_shape_a_row_is_matched_by_is_the_shape_its_builder_records(
        self, builder, row
    ):
        """A row is looked up by the set of keys its builder writes, so that set has to be right.

        Args:
            builder: The builder the row names.
            row: The row under test.

        Test scenario:
            The figure records the resolved style and not the builder that wrote it, so the key set is the
            row's identity. A builder that gains or loses a key matches no row from that moment on, which
            would silently publish its whole style as the caller's — the failure this catches first.
        """
        probe = PROBES[builder]
        drawn = WebMap()
        try:
            probe.unstyled(drawn)
            written = frozenset(_resolved(drawn, probe.recorded))
        finally:
            drawn.close()
        assert written == row.keys, (
            f"{builder} records {sorted(written)} and its row is matched by {sorted(row.keys)}, so the row "
            "matches nothing it describes"
        )

    @pytest.mark.parametrize(
        ("builder", "row"), ROWS, ids=[f"{name}" for name, _ in ROWS]
    )
    def test_every_default_a_row_lists_is_what_its_builder_writes(self, builder, row):
        """The values, measured rather than remembered.

        Args:
            builder: The builder the row names.
            row: The row under test.

        Test scenario:
            A row that lists a value the builder no longer writes stops subtracting anything, and the
            builder goes back to publishing its default as an ask. Reading the resolved style back off a
            real call is the only way that cannot drift, and it is what the round-1 table was missing.
        """
        probe = PROBES[builder]
        drawn = WebMap()
        try:
            probe.unstyled(drawn)
            written = _resolved(drawn, probe.recorded)
        finally:
            drawn.close()
        listed = {key: written.get(key) for key in row.defaults}
        assert listed == dict(row.defaults), (
            f"{builder} writes {listed} where its row says {dict(row.defaults)}"
        )


class TestAnExplicitAskIsStillPublished:
    """The question that catches a row that is too large (review R2-H2)."""

    @pytest.mark.parametrize(
        ("builder", "keyword", "value", "channel", "expected"),
        BORROWED_DEFAULT,
        ids=[f"{name}-{keyword}" for name, keyword, _, _, _ in BORROWED_DEFAULT],
    )
    def test_a_value_another_builder_defaults_to_is_still_this_caller_s_ask(
        self, builder, keyword, value, channel, expected
    ):
        """A number is only a default for the builder that defaults it.

        Args:
            builder: The builder under test.
            keyword: The keyword the ask is written with.
            value: A value some *other* builder on this tier resolves to unasked.
            channel: The declared channel it must be published under.
            expected: The value it must be published with.

        Test scenario:
            The table was keyed by paint property, so every builder was charged with its siblings' defaults
            as well as its own and a real ask was dropped. Keyed by builder, the same number asked for
            somewhere else is the caller's and is published.
        """
        drawn = WebMap()
        try:
            _ask(drawn, builder, {keyword: value})
            published = _published(drawn)
        finally:
            drawn.close()
        assert published.get(channel) == expected, (
            f"{builder}({keyword}={value!r}) published {published}; {value!r} is another builder's default, "
            "not this one's"
        )

    @pytest.mark.parametrize(
        ("builder", "asked", "why"),
        POOLED_WITHIN_ONE_SHAPE,
        ids=[name for name, _, _ in POOLED_WITHIN_ONE_SHAPE],
    )
    def test_the_residue_a_shared_shape_leaves_is_the_one_that_is_written_down(
        self, builder, asked, why
    ):
        """The loss the table still takes, measured so it cannot widen unnoticed.

        Args:
            builder: The builder under test.
            asked: The explicit, non-default ask it is drawn with.
            why: The shape-mate whose default that value is.

        Test scenario:
            Not a promise that this is right — it is not, and #334 is what fixes it by recording the ask at
            the builder. It is a promise that the loss is **exactly** this: three fill builders and two line
            builders share a recorded shape, and nothing else on this tier does. The day a builder records
            its own ask, this fails and the row comes out.
        """
        drawn = WebMap()
        try:
            _ask(drawn, builder, asked)
            published = _published(drawn)
        finally:
            drawn.close()
        assert published == {}, (
            f"{builder}({asked}) published {published}; the residue is meant to be exactly the shape-mates "
            f"({why}), so this is either a fix (#334) or a widening"
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


def _ask(drawn, builder: str, style: dict) -> None:
    """Draw one builder with an explicit style, spelled the way that builder takes it.

    Args:
        drawn: The map to draw on.
        builder: The builder's name, as :data:`PROBES` keys it.
        style: The style keywords to pass.

    Raises:
        KeyError: when the builder has no styled form here, which means a new row was added to a table
            without a way to ask it for an explicit value.
    """
    styled = {
        "points": lambda: drawn.points(_points(), **style),
        "lines": lambda: drawn.lines(_lines(), **style),
        "polygons": lambda: drawn.polygons(_polygons(), **style),
        "contours(filled=True)": lambda: drawn.contours(
            _dem(), filled=True, levels=LEVELS, **style
        ),
        "heatmap": lambda: drawn.heatmap(_points(), **style),
        "field": lambda: drawn.field(_dem(), **style),
        "graticule": lambda: drawn.graticule(**style),
    }
    if builder not in styled:
        raise KeyError(
            f"no styled probe for {builder!r}; add one beside its unstyled probe"
        )
    styled[builder]()
