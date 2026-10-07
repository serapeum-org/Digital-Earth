"""What this tier's builders record about the style they were **asked** for (#334).

`Symbology.encodings` is the one style reading another tier — and `to_backend()` at order 33 — can act on, so
what lands there has to be what a caller asked for rather than what this tier's builder resolved. A keyword
defaulted in a signature destroys that: `points(size=6.0)` and `points(features)` are the same call by the
time the options exist.

This tier answers that question in **two** places, and both are checked here.

* A keyword the caller writes in ``**opts`` is already theirs by construction — it lands in
  :data:`~digitalearth.interactive.style_fold.ASKED_BUCKET`, which the lift reads unfiltered — and `opacity=`
  reaches the same bucket through :func:`~digitalearth.interactive.vector.apply_opacity`, which writes it only
  when it was passed. Those need no record, and the probes below pin that they do not get one.
* A builder *parameter* resolves its default before anything is recorded, and there are exactly two on this
  tier that drive a declared channel: ``points(size=)`` and ``field(alpha=)``. Those now default to
  :data:`~digitalearth.base.ask.UNSET` and record the ask under :data:`~digitalearth.base.ask.ASKED_PROP`.

Measured rather than assumed: every builder on this tier was drawn bare and its recorded style read back, and
only those two derive anything a declared channel can hold.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pytest

pytest.importorskip(
    "holoviews", reason="the interactive tier needs the interactive environment"
)
pytest.importorskip(
    "geoviews", reason="the interactive tier needs the interactive environment"
)

from digitalearth.base.ask import ASKED_PROP, asked_style  # noqa: E402
from digitalearth.interactive import InteractiveMap  # noqa: E402

#: The raster the raster probes draw. Relative, as every other test in this suite is.
DEM = "examples/data/acc4000.tif"


def _points(count: int = 5):
    """Return a small point frame with one numeric column.

    Args:
        count: How many points.

    Returns:
        A GeoDataFrame in EPSG:4326.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    return gpd.GeoDataFrame(
        {"pop": [float(step + 1) for step in range(count)]},
        geometry=[Point(4.0 + step * 0.1, 52.0 + step * 0.1) for step in range(count)],
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
        own_default: An ask whose value is what this builder would have used anyway. The hard case, and the
            only one worth probing: an ask for anything else was published even by the table this replaces.
        channel: The declared channel that ask drives, for the `**opts` probes to read back.
        records: The keys the ask has to be recorded under, in this tier's own spelling. Empty for the
            keywords that reach the caller's own bucket, which needs no record.
    """

    draw: Callable[[Any, dict], Any]
    own_default: Mapping[str, Any]
    channel: str
    records: tuple[str, ...]


#: The two builder parameters on this tier that drive a declared channel and default to a value.
#:
#: Measured, not chosen: drawing all twenty-seven builders bare and reading `symbology.props` back shows only
#: these two deriving anything `CHANNEL_KEYWORDS` translates — `points` writes `common['size']` and `field` a
#: flat `alpha`. `polygons` derives `fill_alpha`, which drives no channel, and a classified `color` is a value
#: dimension rather than a colour.
PARAMETERS: dict[str, Probe] = {
    "points": Probe(
        lambda m, style: m.points(_points(), **style),
        MappingProxyType({"size": 6.0}),
        "size",
        ("size",),
    ),
    "field": Probe(
        lambda m, style: m.field(_dem(), **style),
        MappingProxyType({"alpha": 1.0}),
        "opacity",
        ("alpha",),
    ),
}

#: The keywords that reach the caller's own bucket instead, and so were never mis-attributed.
#:
#: `opacity=` goes through `apply_opacity`, which writes `opts['alpha']` only when the caller passed
#: something — the same "record the ask" shape, arrived at before #334 and spelled with `None`. `line_width`
#: and `alpha` on the builders that take no such parameter go straight through `**opts`. Kept as probes
#: because "this needs no record" is a claim about the builders, and a builder that started resolving one of
#: these into a parameter would quietly lose the attribution again.
THROUGH_OPTS: dict[str, Probe] = {
    "polygons-opacity": Probe(
        lambda m, style: m.polygons(_polygons(), **style),
        MappingProxyType({"opacity": 1.0}),
        "opacity",
        (),
    ),
    "choropleth-opacity": Probe(
        lambda m, style: m.choropleth(_polygons(), "pop", **style),
        MappingProxyType({"opacity": 1.0}),
        "opacity",
        (),
    ),
    "quadmesh-alpha": Probe(
        lambda m, style: m.quadmesh(_dem(), **style),
        MappingProxyType({"alpha": 1.0}),
        "opacity",
        (),
    ),
    "points-through-opts": Probe(
        lambda m, style: m.points(_points(), **style),
        MappingProxyType({"alpha": 1.0}),
        "opacity",
        (),
    ),
}

#: Every probe, for the questions that hold of all of them.
EVERY: dict[str, Probe] = {**PARAMETERS, **THROUGH_OPTS}


def _drawn(probe: Probe, style: Mapping[str, Any]):
    """Draw one probe and return its last layer's description.

    Args:
        probe: The probe.
        style: The style keywords to draw it with.

    Returns:
        ``(props, published)`` — the recorded props as a plain dict, and `{channel: value}` for every channel
        the layer publishes.
    """
    drawn = InteractiveMap()
    try:
        probe.draw(drawn, dict(style))
        figure = drawn.figure_spec
        symbology = figure.layers.get(figure.layers.ids[-1]).symbology
    finally:
        drawn.close()
    # Constants only. A style keyword lifts to a constant, which is all this module is about; a **field**
    # binding is what the layer is coloured *by* — `field(dem)` publishes one whether or not anybody styled it
    # (order 24) — and `Encoding.resolve()` needs that field's values, which no probe here has.
    published = {
        channel: symbology.encoding(channel).resolve()
        for channel in sorted(symbology.encodings)
        if symbology.encoding(channel).is_constant
    }
    return dict(symbology.props), published


class TestAnUnstyledCallRecordsNoAsk:
    """The half that keeps an unstyled layer's description exactly what it was."""

    @pytest.mark.parametrize("label", sorted(EVERY))
    def test_a_builder_drawn_bare_records_no_ask(self, label):
        """Every probe, drawn with no style keyword at all.

        Args:
            label: The probe under test.

        Test scenario:
            A builder resolves its defaults whether or not anybody asked, so it would be easy to record the
            resolution rather than the ask — which is the R-H2 defect with an extra field in it.
        """
        props, _ = _drawn(EVERY[label], {})
        recorded = sorted(asked_style(props))
        assert recorded == [], (
            f"an unstyled {label} recorded {recorded} as the caller's asks; a value its builder defaulted "
            "to is the tier's business, not a portable ask"
        )

    @pytest.mark.parametrize("label", sorted(EVERY))
    def test_a_builder_with_nothing_to_record_writes_no_record_key_at_all(self, label):
        """An empty record is not written, so an unstyled description is what it always was.

        Args:
            label: The probe under test.

        Test scenario:
            Recording `[]` would read the same to the lift and still add a key to every unstyled layer in
            every saved figure. The record is additive precisely because absence is spelled by the key's
            absence.
        """
        props, _ = _drawn(EVERY[label], {})
        assert ASKED_PROP not in props, (
            f"an unstyled {label} described {sorted(props)}; a builder with nothing to record must add no "
            f"{ASKED_PROP!r} key"
        )


class TestABuilderParameterRecordsTheAsk:
    """The two keywords that resolve a default of their own before anything is recorded."""

    @pytest.mark.parametrize("label", sorted(PARAMETERS))
    def test_a_caller_asking_for_their_own_builders_default_is_recorded(self, label):
        """`points(size=6.0)` is a different call from `points(features)`, and now says so.

        Args:
            label: The probe under test.

        Test scenario:
            Measured before this: the subtractive table compared the resolved value against the builder's
            own default, so an explicit ask for exactly that value published nothing and a figure carried
            elsewhere was drawn at the other tier's default instead.
        """
        probe = PARAMETERS[label]
        props, _ = _drawn(probe, probe.own_default)
        recorded = sorted(asked_style(props))
        assert recorded == sorted(probe.records), (
            f"{label} asked for {dict(probe.own_default)} recorded {recorded}; the ask is the record, not a "
            "comparison against the value the builder would have used anyway"
        )

    @pytest.mark.parametrize("label", sorted(PARAMETERS))
    def test_the_record_does_not_depend_on_the_value_asked_for(self, label):
        """The other direction, so the check above cannot pass by recording every key always.

        Args:
            label: The probe under test.

        Test scenario:
            The same keyword with a value that is no tier's default has to be recorded under the same key. A
            record that tracked the value rather than the ask would answer these two differently.
        """
        probe = PARAMETERS[label]
        unusual = {keyword: 0.37 for keyword in probe.own_default}
        props, _ = _drawn(probe, unusual)
        recorded = sorted(asked_style(props))
        assert recorded == sorted(probe.records), (
            f"{label} asked for {unusual} recorded {recorded}, where the same keyword at its own default "
            f"records {sorted(probe.records)}"
        )

    @pytest.mark.parametrize("label", sorted(PARAMETERS))
    def test_the_recorded_key_is_one_the_builder_really_writes(self, label):
        """A record naming a key the style holds no value under lifts nothing at all.

        Args:
            label: The probe under test.

        Test scenario:
            The lift reads the recorded key out of the resolved style — the flat props for `field`, the
            derived bucket for `points` — so the record and the style it names have to agree. A misspelling
            is not an error anywhere; it simply publishes nothing, with nothing said about why.
        """
        probe = PARAMETERS[label]
        props, _ = _drawn(probe, probe.own_default)
        style = {**props, **dict(props.get("common") or {})}
        missing = sorted(set(asked_style(props)) - set(style))
        assert missing == [], (
            f"{label} records {missing} as asked for, and writes no style under them: {sorted(style)}"
        )


class TestAKeywordTheCallerWroteNeedsNoRecord:
    """The other half of the tier: `**opts` and `apply_opacity` were already attributing correctly."""

    @pytest.mark.parametrize("label", sorted(THROUGH_OPTS))
    def test_a_keyword_that_reaches_the_callers_bucket_is_not_recorded_as_a_parameter(
        self, label
    ):
        """Nothing was added where nothing was needed.

        Args:
            label: The probe under test.

        Test scenario:
            `ASKED_BUCKET` is lifted unfiltered on the promise that every key in it was written by the
            caller. A parameter record beside it would be a second answer to one question, and the two
            could disagree.
        """
        probe = THROUGH_OPTS[label]
        props, _ = _drawn(probe, probe.own_default)
        recorded = sorted(asked_style(props))
        assert recorded == [], (
            f"{label} asked for {dict(probe.own_default)} recorded {recorded}; that keyword is already the "
            "caller's own, in the bucket the lift trusts without filtering"
        )

    @pytest.mark.parametrize("label", sorted(THROUGH_OPTS))
    def test_that_keyword_publishes_its_channel_even_at_the_tiers_own_value(
        self, label
    ):
        """Which is why it needs no record: it was never subtracted in the first place.

        Args:
            label: The probe under test.

        Test scenario:
            The claim above is only safe if the caller's bucket really does publish an ask whose value is
            the tier's own default. Asked directly, so "no record needed here" is measured rather than
            asserted from how the code reads.
        """
        probe = THROUGH_OPTS[label]
        _, published = _drawn(probe, probe.own_default)
        expected = next(iter(probe.own_default.values()))
        assert published.get(probe.channel) == expected, (
            f"{label} asked for {dict(probe.own_default)} published {published}; a keyword the caller wrote "
            "is theirs whatever its value"
        )
