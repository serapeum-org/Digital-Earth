"""What this tier's builders record about the style they were **asked** for (#334).

`Symbology.encodings` is the one style reading another tier — and `to_backend()` at order 33 — can act on, so
what lands there has to be what a caller asked for rather than what this tier's builder resolved. A keyword
defaulted in a signature destroys that: `points(size=5.0)` and `points(features)` are the same call by the
time the paint dict exists. So each style keyword's default is now
:data:`~digitalearth.base.ask.UNSET` and each builder records the keys it was actually given, under
:data:`~digitalearth.base.ask.ASKED_PROP`.

These check the record, builder by builder, against real draws: that an unstyled call records nothing, that an
ask is recorded **including** one whose value is that builder's own default — the case no subtraction of
measured defaults could ever answer — that the key it is recorded under is one the lift can translate *and*
one the builder really writes, and that it survives a figure written to JSON and read back. What the lift then
does with the record is asked separately.
"""

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Tuple

import pytest

pytest.importorskip("maplibre", reason="the web tier needs the web environment")

from digitalearth.base.ask import ASKED_PROP, asked_style  # noqa: E402
from digitalearth.base.spec import FigureSpec  # noqa: E402
from digitalearth.web import WebMap  # noqa: E402
from digitalearth.web.renderer import FLAT_CHANNELS, PAINT_CHANNELS  # noqa: E402

#: The contour levels every contour probe traces, so no probe depends on an auto-resolved level set.
LEVELS = [10.0, 100.0, 1000.0]

#: The DEM the raster and contour builders draw. Relative, as every other test in this suite is.
DEM = "examples/data/acc4000.tif"

#: A tile template the `tiles` probe names. Never fetched — the description is what is under test.
TILE_URL = "https://tiles.example.org/{z}/{x}/{y}.png"

#: Every spelling this tier records a style under that the lift can translate to a declared channel. A key
#: recorded under anything else publishes nothing and says nothing about why, which is the one way the record
#: can rot in silence.
TRANSLATABLE: frozenset = frozenset(PAINT_CHANNELS) | frozenset(FLAT_CHANNELS)


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
    """Return the raster the raster builders draw.

    Returns:
        A pyramids `Dataset`.
    """
    from pyramids.dataset import Dataset

    return Dataset.read_file(DEM)


@dataclass(frozen=True)
class Probe:
    """One builder, and the ask to put to it.

    Attributes:
        draw: Draws the builder with the style keywords it is handed — `{}` for an unstyled call.
        own_default: An ask whose value is **this** builder's own default, spelled the way the builder takes
            it. The hard case, and the only one worth probing: an ask for anything else was published even
            by the subtractive tables this replaces.
        records: The keys that ask has to be recorded under, in this tier's own spelling. Empty for a
            builder whose style drives no declared channel — `cluster`'s bubble colour, `text`'s text colour
            and `point_cloud`'s size are each a *second* colour or size on a layer, which is why
            `FLAT_CHANNELS` is deliberately one row.
        writes: Where the builder records its resolved style — `"paint"` for the MapLibre dict, `"props"`
            for the builders that record flat and compile at draw time.
        bare: What the builder records even when nobody styled it. Empty for all but `extrusion`, whose
            `height=` is a required argument and therefore never a default.
    """

    draw: Callable[[Any, dict], Any]
    own_default: Mapping[str, Any]
    records: Tuple[str, ...]
    writes: str = "paint"
    bare: Tuple[str, ...] = field(default=())


#: Every builder on this tier that records a style, and the ask each one is put to.
#:
#: Every builder, not a subset: "which builders record a channel" is exactly the question, and a probe table
#: that draws some of them can only ever confirm the ones it draws. The three at the end record style that
#: drives no channel and are here to pin that, so a later builder cannot quietly start publishing a second
#: colour as the layer's own.
PROBES: Dict[str, Probe] = {
    "points": Probe(
        lambda m, style: m.points(_points(), **style),
        MappingProxyType({"size": 5.0}),
        ("circle-radius",),
    ),
    "points-colour": Probe(
        lambda m, style: m.points(_points(), **style),
        MappingProxyType({"color": "#3388ff"}),
        ("circle-color",),
    ),
    "points-opacity": Probe(
        lambda m, style: m.points(_points(), **style),
        MappingProxyType({"opacity": 0.9}),
        ("circle-opacity",),
    ),
    "lines": Probe(
        lambda m, style: m.lines(_lines(), **style),
        MappingProxyType({"width": 2.0}),
        ("line-width",),
    ),
    "contours": Probe(
        lambda m, style: m.contours(_dem(), levels=LEVELS, **style),
        MappingProxyType({"width": 1.5}),
        ("line-width",),
    ),
    "filled_contours": Probe(
        lambda m, style: m.contours(_dem(), filled=True, levels=LEVELS, **style),
        MappingProxyType({"opacity": 1.0}),
        ("fill-opacity",),
    ),
    "polygons": Probe(
        lambda m, style: m.polygons(_polygons(), **style),
        MappingProxyType({"opacity": 0.6}),
        ("fill-opacity",),
    ),
    "choropleth": Probe(
        lambda m, style: m.choropleth(_polygons(), "pop", **style),
        MappingProxyType({"opacity": 0.85}),
        ("fill-opacity",),
    ),
    "labels": Probe(
        lambda m, style: m.labels(_points(), "pop", **style),
        MappingProxyType({"color": "#ffffff"}),
        ("text-color",),
    ),
    "heatmap": Probe(
        lambda m, style: m.heatmap(_points(), **style),
        MappingProxyType({"opacity": 0.8}),
        ("heatmap-opacity",),
    ),
    "extrusion": Probe(
        lambda m, style: m.extrusion(_polygons(), height=10.0, **style),
        MappingProxyType({"opacity": 0.9}),
        ("fill-extrusion-opacity",),
        bare=("fill-extrusion-height",),
    ),
    "field": Probe(
        lambda m, style: m.field(_dem(), **style),
        MappingProxyType({"opacity": 1.0}),
        ("opacity",),
        writes="props",
    ),
    "rgb_composite": Probe(
        lambda m, style: m.rgb_composite(_dem(), bands=(1, 1, 1), **style),
        MappingProxyType({"opacity": 1.0}),
        ("opacity",),
        writes="props",
    ),
    "hsv_composite": Probe(
        lambda m, style: m.hsv_composite(_dem(), bands=(1, 1, 1), **style),
        MappingProxyType({"opacity": 1.0}),
        ("opacity",),
        writes="props",
    ),
    "graticule": Probe(
        lambda m, style: m.graticule(**style),
        MappingProxyType({"opacity": 0.6}),
        ("opacity",),
        writes="props",
    ),
    "basemap": Probe(
        lambda m, style: m.basemap(**style),
        MappingProxyType({"opacity": 1.0}),
        ("opacity",),
        writes="props",
    ),
    "tiles": Probe(
        lambda m, style: m.tiles(TILE_URL, **style),
        MappingProxyType({"opacity": 1.0}),
        ("opacity",),
        writes="props",
    ),
    "cluster": Probe(
        lambda m, style: m.cluster(_points(), **style),
        MappingProxyType({"color": "#51bbd6"}),
        (),
        writes="props",
    ),
    "text": Probe(
        lambda m, style: m.text(4.5, 52.5, "Amsterdam", **style),
        MappingProxyType({"color": "#ffffff"}),
        (),
        writes="props",
    ),
    "point_cloud": Probe(
        lambda m, style: m.point_cloud(_points(), **style),
        MappingProxyType({"size": 2.0}),
        (),
        writes="props",
    ),
}


def _props(probe: Probe, style: Mapping[str, Any]) -> Dict[str, Any]:
    """Draw one probe and return its last layer's recorded props.

    Args:
        probe: The probe.
        style: The style keywords to draw it with.

    Returns:
        The `Symbology.props` mapping, as a plain dict.
    """
    drawn = WebMap()
    try:
        probe.draw(drawn, dict(style))
        figure = drawn.figure_spec
        return dict(figure.layers.get(figure.layers.ids[-1]).symbology.props)
    finally:
        drawn.close()


def _drawn(probe: Probe, style: Mapping[str, Any]):
    """Draw one probe and return what its last layer recorded.

    Args:
        probe: The probe.
        style: The style keywords to draw it with.

    Returns:
        ``(asked, resolved)`` — the keys the builder recorded as asks, sorted, and the resolved style it
        wrote, read out of whichever half that builder records in.
    """
    props = _props(probe, style)
    resolved = dict(props["paint"]) if probe.writes == "paint" else props
    return sorted(asked_style(props)), resolved


class TestAnUnstyledCallRecordsNoAsk:
    """The half that keeps an unstyled layer's description exactly what it was."""

    @pytest.mark.parametrize("label", sorted(PROBES))
    def test_a_builder_drawn_bare_records_no_ask(self, label):
        """Every builder on this tier, drawn with no style keyword at all.

        Args:
            label: The probe under test.

        Test scenario:
            A builder resolves its defaults whether or not anybody asked, so it would be easy to record the
            resolution rather than the ask — which is the R-H2 defect with an extra field. `extrusion` is
            the one builder that records from a bare call, because its `height=` is a required argument and
            so is never a default of this tier's.
        """
        recorded, _ = _drawn(PROBES[label], {})
        assert recorded == sorted(PROBES[label].bare), (
            f"an unstyled {label} recorded {recorded} as the caller's asks; a value its builder defaulted "
            "to is the tier's business, not a portable ask"
        )

    @pytest.mark.parametrize("label", sorted(PROBES))
    def test_a_builder_with_nothing_to_record_writes_no_record_key_at_all(self, label):
        """An empty record is not written, so an unstyled description is byte-for-byte what it was.

        Args:
            label: The probe under test.

        Test scenario:
            Recording `[]` would read the same to the lift and still change every unstyled layer's
            description — a new key in every saved figure, and in every test that compares a props mapping.
            The record is additive precisely because absence is spelled by the key's absence. `extrusion`
            is the exception, and for the reason above: its `height=` is always an ask.
        """
        props = _props(PROBES[label], {})
        assert (ASKED_PROP in props) is bool(PROBES[label].bare), (
            f"an unstyled {label} described {sorted(props)}; a builder with nothing to record must add no "
            f"{ASKED_PROP!r} key"
        )


class TestAnAskIsRecordedEvenWhenItIsTheTiersOwnDefault:
    """The case no table of measured defaults could answer (#334)."""

    @pytest.mark.parametrize("label", sorted(PROBES))
    def test_a_caller_asking_for_their_own_builders_default_is_recorded(self, label):
        """`points(size=5.0)` is a different call from `points(features)`, and now says so.

        Args:
            label: The probe under test.

        Test scenario:
            Measured before this: the subtractive tables compared the resolved value against the builder's
            own default, so an explicit ask for exactly that value was indistinguishable from no ask, the
            channel published nothing, and a figure carried to another tier was drawn at *that* tier's
            default. The three probes at the end of the table record nothing, because their style drives no
            declared channel on this tier.
        """
        probe = PROBES[label]
        recorded, _ = _drawn(probe, probe.own_default)
        expected = sorted({*probe.bare, *probe.records})
        assert recorded == expected, (
            f"{label} asked for {dict(probe.own_default)} recorded {recorded}; the ask is the record, not "
            "a comparison against the value the builder would have used anyway"
        )

    @pytest.mark.parametrize("label", sorted(PROBES))
    def test_the_record_does_not_depend_on_the_value_asked_for(self, label):
        """The other direction, so the check above cannot pass by recording every key always.

        Args:
            label: The probe under test.

        Test scenario:
            Written from the opposite side: the same keyword with a value that is *no* tier's default has to
            be recorded under the same key. A record that tracked the value rather than the ask would
            answer these two differently, and one of them would be wrong whichever way it leaned.
        """
        probe = PROBES[label]
        unusual = {
            keyword: "#0f0f0f" if isinstance(value, str) else 0.37
            for keyword, value in probe.own_default.items()
        }
        recorded, _ = _drawn(probe, unusual)
        expected = sorted({*probe.bare, *probe.records})
        assert recorded == expected, (
            f"{label} asked for {unusual} recorded {recorded}, where the same keyword at its own default "
            f"records {expected}"
        )


class TestTheKeyTheAskIsRecordedUnderIsUsable:
    """A record the lift cannot translate, or that names nothing the builder wrote, publishes in silence."""

    @pytest.mark.parametrize("label", sorted(PROBES))
    def test_every_recorded_key_is_one_the_lift_can_translate(self, label):
        """The record is read against this tier's own channel maps, so it has to be spelled their way.

        Args:
            label: The probe under test.

        Test scenario:
            The record is a list of strings, so `circle_radius` for `circle-radius` is not an error
            anywhere — it simply translates to no channel, and the ask is dropped with nothing said. Read
            against `PAINT_CHANNELS`/`FLAT_CHANNELS` rather than against a list written here, so a key
            renamed under `src/` fails here instead of turning the record off.
        """
        probe = PROBES[label]
        recorded, _ = _drawn(probe, probe.own_default)
        unusable = sorted(set(recorded) - TRANSLATABLE)
        assert unusable == [], (
            f"{label} records {unusable}, which neither PAINT_CHANNELS nor FLAT_CHANNELS translates, so "
            "the ask publishes nothing and says nothing about why"
        )

    @pytest.mark.parametrize("label", sorted(PROBES))
    def test_every_recorded_key_is_one_the_builder_really_wrote(self, label):
        """The other half: a record pointing at a key the style has no value under lifts nothing.

        Args:
            label: The probe under test.

        Test scenario:
            The lift reads the recorded key out of the resolved style, so a record and the style it names
            have to agree. This is the guard the deleted tables kept as their `keys` field, pointed the
            other way round — at what the record claims rather than at what the builder writes.
        """
        probe = PROBES[label]
        recorded, resolved = _drawn(probe, probe.own_default)
        missing = sorted(set(recorded) - set(resolved))
        assert missing == [], (
            f"{label} records {missing} as asked for, and writes no style under them: {sorted(resolved)}"
        )


class TestTheRecordSurvivesBeingWrittenDown:
    """A figure is saved as JSON, so a record that cannot be written down helps no other tier."""

    def test_an_ask_is_still_recorded_after_a_figure_round_trip(self):
        """The record has to reach the reader, not only the process that drew the layer.

        Test scenario:
            `Symbology` freezes the recorded list to a tuple and JSON reads a tuple back as a list, so the
            shape crosses the trip twice. A record written as a tuple or a set is refused by the travel rule
            and dropped from every saved figure, which is exactly how the documented `clim` pair was lost
            (review R2-M13) — one field along.
        """
        drawn = WebMap()
        try:
            drawn.tiles(TILE_URL, opacity=1.0).graticule(opacity=0.6)
            written = drawn.figure_spec.to_dict()
        finally:
            drawn.close()
        reloaded = FigureSpec.from_dict(json.loads(json.dumps(written)))
        carried = [
            sorted(asked_style(reloaded.layers.get(layer_id).symbology.props))
            for layer_id in reloaded.layers.ids
        ]
        assert carried == [["opacity"], ["opacity"]], (
            f"a saved figure carried {carried}; both layers were asked for the opacity they were already "
            "going to be drawn at, which is the whole case the record exists for"
        )
