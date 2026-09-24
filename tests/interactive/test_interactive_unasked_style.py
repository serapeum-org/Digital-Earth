"""What this tier publishes as the caller's own style, builder by builder (review R2-H2/H3/M4/M5).

`Symbology.encodings` is the one style reading another tier — and `to_backend()` at order 33 — can act on,
so what lands there has to be what a caller *asked for* rather than what this tier's builder resolved.
`tests/interactive/test_interactive_style_fold.py` checks the lift against hand-written props; these check
it against the **builders**, which is where the defaults actually come from and where the round-1 fix's two
defects were found: a table keyed by style keyword rather than by builder, and a builder laundering a
derived colour through the bucket the lift trusts unfiltered (review R2-M5 — nothing tested either table).

Every case here draws a real layer and reads `figure_spec`, so a row of
:data:`~digitalearth.interactive.style_fold.UNASKED_STYLE` that stops matching its builder fails here rather
than rotting quietly.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

import pytest

pytest.importorskip(
    "holoviews", reason="the interactive tier needs the interactive environment"
)
pytest.importorskip(
    "geoviews", reason="the interactive tier needs the interactive environment"
)

from digitalearth.base.spec import Symbology  # noqa: E402
from digitalearth.interactive import InteractiveMap  # noqa: E402
from digitalearth.interactive.style_fold import (  # noqa: E402
    ASKED_BUCKET,
    TIER_BUCKET,
    UNASKED_STYLE,
    builder_of,
    portable_encodings,
)

#: The DEM the raster builders draw. Relative, as every other test in this suite is.
DEM = "examples/data/acc4000.tif"

#: The contour levels every contour probe traces, so no probe depends on an auto-resolved level set.
LEVELS = [10.0, 100.0]

#: A `spaghetti` cycle spelled the one way a figure cannot carry: bokeh's RGB triple. Every colour the
#: tier ships is a hex string, which travels, so this is what a derived value the description has to hold
#: rather than record looks like.
UNTRAVELLED_CYCLE = ((255, 0, 0), (0, 255, 0), (0, 0, 255))


def _points(count: int = 5):
    """Return a small point frame with the columns the aggregating builders need.

    Args:
        count: How many points.

    Returns:
        A GeoDataFrame in EPSG:4326.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    return gpd.GeoDataFrame(
        {
            "pop": [float(step + 1) for step in range(count)],
            "u": [1.0] * count,
            "v": [0.5] * count,
        },
        geometry=[Point(4.0 + step * 0.1, 52.0 + step * 0.1) for step in range(count)],
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


def _collection(members: int = 3):
    """Return a `DatasetCollection` of repeated DEMs.

    Args:
        members: How many members it holds.

    Returns:
        The collection `spaghetti` draws one layer per member of.
    """
    from pyramids.dataset.collection import DatasetCollection

    return DatasetCollection.from_files([DEM] * members)


#: Every builder on this tier that records a layer, and how to draw it with no style keyword at all.
#:
#: All twenty-three data builders plus the four decorations, because "which builders derive a channel" is
#: exactly the question the table answers and a probe that draws a subset can only ever confirm the subset.
#: `grep -rn UNASKED tests/` returned nothing before this (review R2-M5).
UNSTYLED: Dict[str, Callable[[Any], Any]] = {
    "points": lambda m: m.points(_points()),
    "path": lambda m: m.path(_lines()),
    "polygons": lambda m: m.polygons(_polygons()),
    "choropleth": lambda m: m.choropleth(_polygons(), "pop"),
    "labels": lambda m: m.labels(_points(), "pop"),
    "hexbin": lambda m: m.hexbin(_points()),
    "kde": lambda m: m.kde(_points()),
    "graph": lambda m: m.graph(_points(), [(0, 1)]),
    "flow": lambda m: m.flow(_points(), [(0, 1)]),
    "trimesh": lambda m: m.trimesh(_points()),
    "vectorfield": lambda m: m.vectorfield(_dem(), _dem()),
    "streamlines": lambda m: m.streamlines(_dem(), _dem()),
    "barbs": lambda m: m.barbs(_dem(), _dem()),
    "image": lambda m: m.image(_dem()),
    "rgb": lambda m: m.rgb(_dem(), bands=(1, 1, 1)),
    "quadmesh": lambda m: m.quadmesh(_dem()),
    "contours": lambda m: m.contours(_dem(), levels=LEVELS),
    "filled_contours": lambda m: m.filled_contours(_dem(), levels=LEVELS),
    "large_image": lambda m: m.large_image(_dem()),
    "spaghetti": lambda m: m.spaghetti(_collection(2), levels=4),
    "rasterize": lambda m: m.rasterize(_points()),
    "datashade": lambda m: m.datashade(_points()),
    "trajectory": lambda m: m.trajectory(_points()),
    "tiles": lambda m: m.tiles(),
    "coastlines": lambda m: m.coastlines(),
    "graticule": lambda m: m.graticule(),
    "text": lambda m: m.text(5.0, 52.0, "Amsterdam"),
}

#: Which builder each row of :data:`~digitalearth.interactive.style_fold.UNASKED_STYLE` describes.
#:
#: Written the other way round from the table so the two have to agree: the checks below walk the **table**
#: and look its key up here, so a row added without a probe fails rather than going unmeasured.
ROW_PROBES: Dict[Tuple[str, str], str] = {
    ("geometry", "Points"): "points",
    ("image", ""): "image",
}


@dataclass(frozen=True)
class Borrowed:
    """One explicit ask whose value is a default *somewhere else*.

    Attributes:
        draw: Draws the builder with that ask.
        channel: The channel it must be published under.
        value: The value it must be published with.
        whose: Whose default the value is, for the failure message.
    """

    draw: Callable[[Any], Any]
    channel: str
    value: Any
    whose: str


#: Explicit asks that a table keyed by style keyword would have swallowed (review R2-H2).
#:
#: 1.0 is `image`'s own `alpha` default on this tier, and 5.0 and 0.9 are the web tier's marker size and
#: circle opacity — the very pair the module docstring of the conformance suite quotes. None of them is
#: `points`' default, so every one of them is the caller's.
BORROWED: Dict[str, Borrowed] = {
    "size-is-another-builders-alpha": Borrowed(
        lambda m: m.points(_points(), size=1.0), "size", 1.0, "image's alpha"
    ),
    "size-is-the-web-tiers-size": Borrowed(
        lambda m: m.points(_points(), size=5.0), "size", 5.0, "the web tier's size"
    ),
    "alpha-is-the-web-tiers-opacity": Borrowed(
        lambda m: m.image(_dem(), alpha=0.9),
        "opacity",
        0.9,
        "the web tier's circle opacity",
    ),
}


def _symbologies(figure) -> Tuple[Symbology, ...]:
    """Return every layer's recorded symbology, in draw order.

    Args:
        figure: A tier's `figure_spec`.

    Returns:
        One `Symbology` per layer.
    """
    return tuple(layer.symbology for layer in figure.layers)


def _published(figure) -> list:
    """Return every layer's published channels, in draw order.

    Args:
        figure: A tier's `figure_spec`.

    Returns:
        One `{channel: value}` dict per layer.
    """
    return [
        {
            channel: symbology.encoding(channel).resolve()
            for channel in sorted(symbology.encodings)
        }
        for symbology in _symbologies(figure)
    ]


def _drawn(call: Callable[[Any], Any]):
    """Draw one builder on a fresh map and return its figure.

    Args:
        call: The builder call.

    Returns:
        The `FigureSpec`.
    """
    drawn = InteractiveMap()
    try:
        call(drawn)
        return drawn.figure_spec
    finally:
        drawn.close()


class TestAnUnstyledLayerPublishesNothing:
    """The question that catches a row the table is missing, asked of every builder."""

    @pytest.mark.parametrize("builder", sorted(UNSTYLED))
    def test_a_builder_drawn_bare_publishes_no_channel(self, builder):
        """A layer nobody styled must publish no caller intent, because there was none to publish.

        Args:
            builder: The builder under test.

        Test scenario:
            `Symbology.encodings` is the field `to_backend()` (order 33) will read, so a tier that lifts
            its own builder defaults into it publishes them **as** the caller's intent — and the two 2-D
            tiers' defaults differ, so an unstyled layer carried across would be drawn in the other tier's
            colours (review R-H2). Every builder is drawn, because a table row can only be missing for a
            builder nobody drew.
        """
        published = _published(_drawn(UNSTYLED[builder]))
        assert published == [{}] * len(published), (
            f"an unstyled {builder} published {published}; a value its builder resolved is the tier's "
            "business, not a portable ask"
        )

    @pytest.mark.parametrize("builder", sorted(UNSTYLED))
    def test_a_builder_drawn_bare_writes_nothing_into_the_callers_own_bucket(
        self, builder
    ):
        """The invariant the lift depends on, stated at last as a check.

        Args:
            builder: The builder under test.

        Test scenario:
            `portable_encodings` lifts :data:`ASKED_BUCKET` **unfiltered**, on the promise that every key
            there was written by the caller. `spaghetti` broke that promise — it passed its per-member
            colour cycle through `**opts`, so three unstyled members published three different colours as
            three caller intents (review R2-H3). A derived value now goes to
            :data:`~digitalearth.interactive.style_fold.TIER_BUCKET`, and this is what stops the next
            builder reopening it.
        """
        written = [
            dict(symbology.props.get(ASKED_BUCKET) or {})
            for symbology in _symbologies(_drawn(UNSTYLED[builder]))
        ]
        assert written == [{}] * len(written), (
            f"an unstyled {builder} filed {written} under {ASKED_BUCKET!r}, the bucket the lift trusts "
            "without filtering; style the builder worked out itself belongs in the tier bucket"
        )


class TestEveryRowStillDescribesItsBuilder:
    """The question that catches a row that has rotted — the builder it names, and the value."""

    @pytest.mark.parametrize("key", sorted(UNASKED_STYLE))
    def test_a_rows_builder_is_the_builder_that_records_that_key(self, key):
        """A row is looked up by what the builder records under `via`/`hv_type`, so that has to match.

        Args:
            key: The row's `(via, hv_type)` key.

        Test scenario:
            The lookup is the whole improvement over a keyword-keyed table, so a row whose key no builder
            answers to subtracts nothing at all — and the builder goes straight back to publishing its
            default as an ask.
        """
        recorded = [
            builder_of(symbology.props)
            for symbology in _symbologies(_drawn(UNSTYLED[ROW_PROBES[key]]))
        ]
        assert recorded == [key], (
            f"{ROW_PROBES[key]} records {recorded} and its row is keyed {key}, so the row matches nothing"
        )

    @pytest.mark.parametrize("key", sorted(UNASKED_STYLE))
    def test_every_default_a_row_lists_is_what_its_builder_writes(self, key):
        """The values, measured rather than remembered.

        Args:
            key: The row's `(via, hv_type)` key.

        Test scenario:
            A row listing a value the builder no longer writes stops subtracting, silently. Reading the
            resolved style back off a real call is the only way that cannot drift.
        """
        symbology = _symbologies(_drawn(UNSTYLED[ROW_PROBES[key]]))[0]
        flat: Mapping[str, Any] = {
            **dict(symbology.props),
            **dict(symbology.props.get("common") or {}),
        }
        listed = {name: flat.get(name) for name in UNASKED_STYLE[key]}
        assert listed == dict(UNASKED_STYLE[key]), (
            f"{ROW_PROBES[key]} writes {listed} where its row says {dict(UNASKED_STYLE[key])}"
        )


class TestAnExplicitAskIsStillPublished:
    """The question that catches a row that is too large (review R2-H2)."""

    @pytest.mark.parametrize("case", sorted(BORROWED))
    def test_a_value_another_builder_defaults_to_is_still_this_callers_ask(self, case):
        """A number is only a default for the builder that defaults it.

        Args:
            case: The borrowed-default case under test.

        Test scenario:
            A table keyed by style keyword pools every builder's defaults under one name, so an explicit
            ask that happens to equal a *sibling's* default is dropped. Keyed by builder — which this tier
            can do, because every builder records `via` — the same number asked for elsewhere is the
            caller's and is published.
        """
        borrowed = BORROWED[case]
        published = _published(_drawn(borrowed.draw))[0]
        assert published.get(borrowed.channel) == borrowed.value, (
            f"{case} published {published}; {borrowed.value!r} is {borrowed.whose}, not this builder's"
        )


class TestTheDerivedColourIsNotTheCallersOwn:
    """`spaghetti` derives a colour per member; nobody asked for it (review R2-H3)."""

    def test_an_unstyled_spaghetti_publishes_no_colour_on_any_member(self):
        """The cycle is this tier's way of telling strands apart, not a style anyone wrote.

        Test scenario:
            `spaghetti` wrote its per-member colour into the caller's own `**opts` bucket, which
            `portable_encodings` lifts **unfiltered** because every key there was written by the caller.
            So three unstyled members published `{'color': '#1f77b4'}`, `{'color': '#ff7f0e'}` and
            `{'color': '#2ca02c'}` as three different caller intents, and a figure carried to another tier
            repainted itself in this tier's cycle.
        """
        published = _published(_drawn(lambda m: m.spaghetti(_collection(), levels=4)))
        assert published == [{}, {}, {}], (
            f"an unstyled spaghetti published {published}; the per-member cycle is derived, not asked for"
        )

    def test_a_spaghetti_the_caller_coloured_still_publishes_that_colour(self):
        """The other direction: an explicit colour is the caller's and must survive.

        Test scenario:
            The cycle is disabled the moment `color=` is passed, so every member carries the one colour the
            caller wrote — and that one **is** an ask, so it must still reach `encodings`.
        """
        published = _published(
            _drawn(lambda m: m.spaghetti(_collection(2), levels=4, color="#cc4444"))
        )
        assert published == [{"color": "#cc4444"}, {"color": "#cc4444"}], (
            f"an explicitly coloured spaghetti published {published}"
        )

    def test_the_cycle_still_reaches_the_drawn_elements(self):
        """Moving the colour out of `opts` must not move it out of the picture.

        Test scenario:
            The fix is about attribution, not about rendering: the members still have to be told apart on
            screen. Read back off the live HoloViews elements rather than off the description, because the
            description is exactly what changed.
        """
        import holoviews as hv

        drawn = InteractiveMap()
        try:
            drawn.spaghetti(_collection(), levels=4)
            colours = [
                hv.Store.lookup_options("bokeh", layer, "style").kwargs.get("color")
                for layer in drawn.layers
            ]
        finally:
            drawn.close()
        assert len(set(colours)) == 3, (
            f"the members must stay distinguishable, got {colours}"
        )

    def test_the_tier_bucket_is_never_lifted_whatever_it_holds(self):
        """The bucket's whole job, asked of the lift directly.

        Test scenario:
            A value in the tier bucket drives a declared channel and carries a portable constant, so the
            only reason it publishes nothing is that the lift does not read that bucket at all. Asked here
            rather than only through `spaghetti`, so the next builder to use it inherits the guarantee.
        """
        invented = Symbology(props={TIER_BUCKET: {"size": 99.0, "alpha": 0.25}})
        assert portable_encodings(invented) == {}, (
            f"{TIER_BUCKET!r} was lifted; it holds style the tier invented, which no caller asked for"
        )

    def test_a_derived_colour_the_figure_cannot_carry_still_reaches_the_engine(
        self, monkeypatch
    ):
        """A cycle spelled as an RGB triple is held beside the layer rather than lost between the buckets.

        Args:
            monkeypatch: Stands an RGB-tuple cycle in for the shipped hex one. It is the only way to reach
                this path: every colour the tier cycles through today is a string, and a string travels.

        Test scenario:
            The derived bucket is described like any other style, so a value the shared travel rule refuses
            — a tuple is refused precisely because JSON reads it back as a list — is described as nothing
            and has to be held beside the layer instead. Only the held half carries the colour: the
            description keeps `None` whether or not the hold happens, so a drawer reading the description
            alone paints every member `color=None` and the three stop being distinguishable, which is the
            one thing the cycle exists for. Read off the live elements for that reason.
        """
        import holoviews as hv

        monkeypatch.setattr(InteractiveMap, "_SPAGHETTI_COLORS", UNTRAVELLED_CYCLE)
        drawn = InteractiveMap()
        try:
            drawn.spaghetti(_collection(), levels=4)
            reached = [
                hv.Store.lookup_options("bokeh", layer, "style").kwargs.get("color")
                for layer in drawn.layers
            ]
        finally:
            drawn.close()
        assert reached == list(UNTRAVELLED_CYCLE), (
            f"the members were drawn {reached}; a derived colour that cannot be described has to be "
            "held beside the layer, or the drawer is handed the nothing the description carries"
        )


class TestTheSpellingOfANumberDoesNotDecideWhatIsPublished:
    """`type(value) is type(default)` made `size=6` publish and `size=6.0` not (review R2-M4)."""

    @pytest.mark.parametrize("spelling", [6, 6.0])
    def test_the_tiers_own_default_is_read_as_a_default_however_it_is_spelled(
        self, spelling
    ):
        """An `int` and a `float` of the same value are one ask, not two.

        Args:
            spelling: `6` or `6.0` — this tier's own default marker size.

        Test scenario:
            The comparison was type-strict and this tier records what it was handed rather than coercing
            it, so `points(size=6)` published `{'size': 6}` and `points(size=6.0)` published nothing — two
            portable figures from one intent, in the field `to_backend()` will read.
        """
        published = _published(_drawn(lambda m: m.points(_points(), size=spelling)))[0]
        assert published == {}, (
            f"points(size={spelling!r}) published {published}; 6 and 6.0 are one ask"
        )

    @pytest.mark.parametrize("spelling", [7, 7.0])
    def test_a_value_the_tier_does_not_default_is_published_however_it_is_spelled(
        self, spelling
    ):
        """The other direction, so the check above cannot pass by publishing nothing ever.

        Args:
            spelling: `7` or `7.0` — a size no builder on this tier defaults.
        """
        published = _published(_drawn(lambda m: m.points(_points(), size=spelling)))[0]
        assert published.get("size") == 7, (
            f"points(size={spelling!r}) published {published}"
        )
