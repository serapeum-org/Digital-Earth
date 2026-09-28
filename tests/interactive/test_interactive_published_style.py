"""What this tier publishes as the caller's own style, builder by builder (review R2-H2/H3/M4/M5, #334).

`Symbology.encodings` is the one style reading another tier — and `to_backend()` at order 33 — can act on,
so what lands there has to be what a caller *asked for* rather than what this tier's builder resolved.
`tests/interactive/test_interactive_style_fold.py` checks the lift against hand-written props and
`test_interactive_asked_style.py` checks what the builders *record*; these check what a real draw
**publishes**, which is the end of that chain and the only part another tier sees.

Both directions are asked. An unstyled layer publishes nothing, and an ask is published — including an ask
whose value is exactly this tier's own default, which is what #334 changed. The lift used to read the
resolved options against a table of measured defaults (``UNASKED_STYLE``), so `points(size=6.0)` was
indistinguishable from `points(features)` and published no size at all; the table is gone, the builders
record the ask, and what used to be this module's documented loss is now one of its checks.
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


def _categories():
    """Return a small polygon frame whose one column is an unordered attribute.

    Returns:
        A two-polygon GeoDataFrame in EPSG:4326, with a `cover` column of distinct strings.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon

    return gpd.GeoDataFrame(
        {"cover": ["forest", "water"]},
        geometry=[
            Polygon([(4, 52), (5, 52), (5, 53), (4, 53)]),
            Polygon([(5, 52), (6, 52), (6, 53), (5, 53)]),
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
    "lines": lambda m: m.lines(_lines()),
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
    "field": lambda m: m.field(_dem()),
    "rgb": lambda m: m.rgb(_dem(), bands=(1, 1, 1)),
    "quadmesh": lambda m: m.quadmesh(_dem()),
    "contours": lambda m: m.contours(_dem(), levels=LEVELS),
    "filled_contours": lambda m: m.contours(_dem(), levels=LEVELS, filled=True),
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
        lambda m: m.field(_dem(), alpha=0.9),
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
    """Return every layer's published **constant** channels, in draw order.

    Constants only, because a constant is what a *style keyword* lifts to and this module is about style: the
    question every probe here asks is whether a value the builder resolved was published as though the caller
    had asked for it. A **field** binding is a different statement — `field(dem)` colours by the band it draws
    whether or not anybody styled it — and since order 24 the colour-driven builders publish one, so folding
    the two together would have this module's central check answer a question it was not asked. It would also
    raise: `Encoding.resolve()` needs the field's values for a binding, and there are none here.
    :func:`_bound_fields` is the field half, and :class:`TestTheColouredBuildersPublishWhatColoursThem` is
    where it is held to a table.

    Args:
        figure: A tier's `figure_spec`.

    Returns:
        One `{channel: value}` dict per layer, holding the channels bound to a constant.
    """
    return [
        {
            channel: symbology.encoding(channel).resolve()
            for channel in sorted(symbology.encodings)
            if symbology.encoding(channel).is_constant
        }
        for symbology in _symbologies(figure)
    ]


def _bound_fields(figure) -> list:
    """Return every layer's field-bound channels, in draw order.

    Args:
        figure: A tier's `figure_spec`.

    Returns:
        One `{channel: field name}` dict per layer — what the layer says its colour, size or opacity *varies
        with*, as opposed to what it was styled with.
    """
    return [
        {
            channel: symbology.encoding(channel).field
            for channel in sorted(symbology.encodings)
            if not symbology.encoding(channel).is_constant
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


#: What each builder in :data:`UNSTYLED` binds to a **field**, per layer it draws (order 24, #261).
#:
#: A field binding is not style — it is what the layer *is*. `field(dem)` colours by the band it draws and
#: `choropleth(gdf, "pop")` by the column, whether or not anybody styled either, which is why these are
#: published for an unstyled layer while the constants beside them are not. They are also what a colour key
#: hangs on: :meth:`~digitalearth.interactive.decoration.DecorationMixin.colorbar` attaches a
#: :class:`~digitalearth.base.spec.encoding.Guide` to this encoding, so a builder that publishes none refuses
#: a key — which makes this table the list of builders whose layers can carry one.
#:
#: Written out per builder rather than derived, so a builder that gains or loses a binding is a failing row
#: rather than a silent change. Every layer one call draws binds the same channels: `spaghetti` colours each
#: member one flat colour from its cycle, so all of its members bind nothing.
#:
#: `'Band_1'` is what `examples/data/acc4000.tif` calls its only band — measured, and the same name the drawn
#: `hv.Image`'s value dimension carries, which is the agreement
#: :func:`~digitalearth.interactive.raster.coloured_by` exists to keep.
BOUND_FIELDS: Dict[str, Dict[str, str]] = {
    "points": {},
    "lines": {},
    "polygons": {},
    "choropleth": {"color": "pop"},
    "labels": {},
    "hexbin": {},
    "kde": {},
    "graph": {},
    "flow": {},
    "trimesh": {},
    "vectorfield": {},
    "streamlines": {},
    "barbs": {},
    "field": {"color": "Band_1"},
    "rgb": {},
    "quadmesh": {"color": "Band_1"},
    "contours": {"color": "Band_1"},
    "filled_contours": {"color": "Band_1"},
    "large_image": {"color": "Band_1"},
    "spaghetti": {},
    "rasterize": {},
    "datashade": {},
    "trajectory": {},
    "tiles": {},
    "coastlines": {},
    "graticule": {},
    "text": {},
}


class TestTheColouredBuildersPublishWhatColoursThem:
    """The binding half of the same question, which order 24 needs and #334 did not answer."""

    @pytest.mark.parametrize("builder", sorted(UNSTYLED))
    def test_a_builder_binds_exactly_the_fields_its_row_names(self, builder):
        """A colour key hangs on a binding, so which builders publish one is part of the contract.

        Args:
            builder: The builder under test.

        Test scenario:
            Before order 24 no builder on this tier published a colour binding at all: a classified layer
            recorded HoloViews' `color_levels` beside a `color` naming the value dimension, and the lift
            refused that colour as a constant — correctly — so the layer published nothing a `Guide` could be
            attached to and the tier's colour key had to be a toggle on whichever layer was added last. The
            table is read in **both** directions: a builder that stops binding fails here as loudly as one
            that starts.
        """
        bound = _bound_fields(_drawn(UNSTYLED[builder]))
        expected = [BOUND_FIELDS[builder]] * len(bound)
        assert bound == expected, (
            f"an unstyled {builder} bound {bound}; BOUND_FIELDS says {expected}"
        )

    def test_a_classified_fill_publishes_the_scale_it_was_cut_with(self):
        """The `Scale` is the binding's other half, and the reason a swatch cannot disagree with the fill.

        Test scenario:
            `last_breaks` published the same edges as a plain list beside the layer, which is parity
            maintained by hand — `base/spec/legend.py`'s docstring names this tier for exactly that. On the
            encoding the edges are the scale the picture was drawn from, so
            `LegendSpec.from_scale` derives rows from the same object rather than from a second computation.
        """
        from digitalearth.base.spec import LegendSpec

        figure = _drawn(
            lambda m: m.choropleth(_polygons(), "pop", scheme="quantiles", k=2)
        )
        symbology = figure.layers.get(figure.layers.ids[-1]).symbology
        scale = symbology.encoding("color").scale
        assert scale is not None and tuple(scale.breaks) == (1.0, 5.0, 9.0), scale
        derived = LegendSpec.from_scale(scale, colors=["#440154", "#fde725"])
        assert [entry.value for entry in derived.entries] == [(1.0, 5.0), (5.0, 9.0)], (
            derived.entries
        )

    def test_a_categorical_fill_publishes_the_colours_it_drew_with(self):
        """And the categorical arm, whose scale carries the colours themselves.

        Test scenario:
            A categorical `Scale` needs no colours passed to `from_scale` — it holds them — so the swatch a
            legend shows is the colour the polygon was filled with by construction. The categories are the
            string form of the values, which is what this tier colours by (`_categorical_style` relabels the
            column) and what lets the scale be written into a figure at all.
        """
        from digitalearth.base.spec import LegendSpec

        figure = _drawn(
            lambda m: m.choropleth(_categories(), "cover", scheme="categorical")
        )
        symbology = figure.layers.get(figure.layers.ids[-1]).symbology
        scale = symbology.encoding("color").scale
        assert scale is not None and scale.is_categorical, scale
        rows = LegendSpec.from_scale(scale)
        assert [entry.label for entry in rows.entries] == ["forest", "water"], rows
        assert all(entry.color for entry in rows.entries), rows.entries


class TestAnExplicitAskIsStillPublished:
    """The other direction: an ask has to reach `encodings`, whatever value it carries."""

    @pytest.mark.parametrize("case", sorted(BORROWED))
    def test_a_value_another_builder_defaults_to_is_still_this_callers_ask(self, case):
        """A number is only a default for the builder that defaults it.

        Args:
            case: The borrowed-default case under test.

        Test scenario:
            A table keyed by style keyword pooled every builder's defaults under one name, so an explicit
            ask that happened to equal a *sibling's* default was dropped; keying it by builder answered that,
            and left the last case, which the record answered. Kept exactly as the table had to answer them,
            because "somebody else's default is still my ask" is a property of the answer and not of how it
            is reached.
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


class TestTheTiersOwnDefaultIsPublishedWhenItWasAskedFor:
    """The loss the subtractive table took, and the check that replaced it (#334)."""

    def test_a_caller_asking_for_their_own_builders_default_publishes_it(self):
        """`points(size=6.0)` publishes 6.0, where the table published nothing.

        Test scenario:
            The exact case this module used to pin as an accepted cost: the figure recorded the resolved
            value and never the fact that a keyword was passed, so an ask for the tier's own default was
            read as no ask, the channel published nothing, and a figure carried to another tier was drawn at
            *that* tier's default instead. Executed before the change, this published `{}`.
        """
        published = _published(_drawn(lambda m: m.points(_points(), size=6.0)))[0]
        assert published == {"size": 6.0}, (
            f"points(size=6.0) published {published}; the value a caller names is theirs even when the "
            "builder would have used it anyway"
        )

    def test_the_raster_builders_own_alpha_default_is_published_when_it_was_asked_for(
        self,
    ):
        """The same, for the other builder that resolves a channel parameter of its own.

        Test scenario:
            `field`'s `alpha` is the second and last row the deleted table held, and the two rows were the
            whole of this tier's exposure — so both are asked, rather than one being taken as evidence for
            the other.
        """
        published = _published(_drawn(lambda m: m.field(_dem(), alpha=1.0)))[0]
        assert published == {"opacity": 1.0}, (
            f"field(alpha=1.0) published {published}; 1.0 is what an unstyled field draws at, and asking "
            "for it is still asking"
        )

    @pytest.mark.parametrize("spelling", [6, 6.0])
    def test_the_value_published_is_the_value_written(self, spelling):
        """An `int` and a `float` of one number are one ask, and both are published (review R2-M4).

        Args:
            spelling: `6` or `6.0` — this tier's own default marker size, written both ways.

        Test scenario:
            The comparison that used to decide this was type-strict, and this tier records what it was
            handed rather than coercing it, so `points(size=6)` published `{'size': 6}` while
            `points(size=6.0)` published nothing — two portable figures from one intent. Nothing is compared
            any more: what the caller wrote is what is published, in the spelling they wrote it.
        """
        published = _published(_drawn(lambda m: m.points(_points(), size=spelling)))[0]
        assert published == {"size": spelling}, (
            f"points(size={spelling!r}) published {published}"
        )

    def test_a_description_that_records_no_ask_publishes_nothing_it_holds(self):
        """The record is what decides, asked of the lift directly rather than through a builder.

        Test scenario:
            A `Symbology` written by hand — or read out of a figure saved before the record existed — holds
            resolved style and no record. It must publish nothing rather than publish whatever it holds,
            which is the fail-closed half of the change: the missing rows of the deleted table failed the
            other way and published a tier default as caller intent.
        """
        held = Symbology(props={"via": "geometry", "common": {"size": 9.0}})
        assert portable_encodings(held) == {}, (
            "a description that records no ask published something it merely holds"
        )
