"""The static tier's half of order 23: the public toggle, reorder, replace, read-back and removal.

The shared probes are :mod:`tests.base.layer_management`; this module supplies the adapter and the checks
that are this tier's own — `add_layer`, which is where a caller's own matplotlib artist joins the figure,
and the painting order the reorder has to reach.

**Why the engine reading is matplotlib's own painting rule, not `Axes._children`.** matplotlib draws the
axes' artists `sorted(children, key=attrgetter("zorder"))` — a *stable* sort, so the list order decides only
between artists of one z-order. Reading `Axes._children` alone therefore reports the order the artists were
*added*, which is the painted order only while every layer carries the same z-order; a raster (``AxesImage``,
z-order 0) under a point layer (``PathCollection``, z-order 1) is the ordinary composition where the two part
company. So :func:`paint_order` applies matplotlib's rule — the same stable sort — and that is what the
shared probe reads (review H5: the probe agreed with a list the engine does not honour, and passed while
`move_layer` left the picture untouched). Reading `Renderer.drawn` instead would report a dict the tier keeps
beside the axes, which a move could reorder while the figure came out unchanged.
"""

import matplotlib

matplotlib.use("Agg")

from dataclasses import replace as with_fields

import pytest
from matplotlib.text import Text
from pyramids.feature import FeatureCollection

from digitalearth.base.capabilities import CapabilityError
from digitalearth.static import Map
from digitalearth.static.renderer import _painted
from tests.base.layer_management import (
    BOTTOM,
    TOP,
    LayerManagementConformance,
    LayerManagementContract,
)


def _owners(tier):
    """Return the layer id each artist belongs to, keyed by `id()` of the artist.

    Args:
        tier: The map.

    Returns:
        Artist identity to layer id.
    """
    return {
        id(artist): layer_id
        for layer_id, drawn in tier._renderer.drawn.items()
        for artist in drawn.artists
    }


def layer_artists(tier):
    """Return the artists the axes holds for the tier's layers, in the order the axes lists them.

    Args:
        tier: The map.

    Returns:
        The artists, with everything no layer owns — the axes' patch, its spines, a colorbar — left out.
    """
    owner = _owners(tier)
    return tuple(artist for artist in _painted(tier.ax) or () if id(artist) in owner)


def paint_order(tier):
    """Return the layer ids in the order matplotlib will paint them, bottom first.

    matplotlib's own rule, applied to the axes' own list: a stable sort by z-order, so the list order
    breaks ties and nothing else. This is what a move has to reach — the module docstring says why reading
    the unsorted list instead is a proxy the engine does not honour.

    Args:
        tier: The map.

    Returns:
        The ids, bottom first, each named once (a layer owning several artists is painted where its lowest
        one is).
    """
    owner = _owners(tier)
    ordered = []
    for artist in sorted(layer_artists(tier), key=lambda held: held.get_zorder()):
        layer_id = owner[id(artist)]
        if layer_id not in ordered:
            ordered.append(layer_id)
    return tuple(ordered)


class StaticContract(LayerManagementContract):
    """The matplotlib tier's adapter for the shared layer-management probes."""

    backend = "matplotlib"
    #: A caller's own PyVista object: registered, drawn from no data, and nothing this tier can draw.
    undrawable_kind = "custom:pyvista"

    def make(self):
        """Return an empty map whose display CRS matches the probes' coordinates.

        Returns:
            The map, which owns its own figure.
        """
        return Map(crs=4326)

    def draw_two(self, tier):
        """Draw two text labels, which are one kind and so share one band.

        Text rather than two rasters: a cleopatra glyph clears the axes unless a render opts into
        composing, so a second raster would take the first one's image off and the probes would be reading
        a tier that had drawn one layer.

        Args:
            tier: The map.

        Returns:
            The two ids, bottom first.
        """
        tier.text(0.0, 0.0, "lower", name=BOTTOM)
        tier.text(1.0, 1.0, "upper", name=TOP)
        return BOTTOM, TOP

    def engine_order(self, tier):
        """Return the layer ids in the order the axes will paint them.

        :func:`paint_order`, so the reading is matplotlib's own stable-sort-by-z-order rather than the
        addition order the axes' list happens to be in — the two agree only while every layer carries the
        same z-order, which is exactly the case the shared probe's one-kind `draw_two` builds.

        Args:
            tier: The map.

        Returns:
            The ids, bottom first, with an artist no layer owns — the axes' own spines, its patch — left
            out.
        """
        return paint_order(tier)

    def engine_objects(self, tier):
        """Return every artist the axes paints, in painting order.

        Args:
            tier: The map.

        Returns:
            The artists themselves, so a removal is held to having taken the very artist it drew off the
            axes rather than only out of the renderer's record.
        """
        return tuple(_painted(tier.ax) or ())

    def engine_holds(self, tier, layer_id):
        """Return the first artist the axes paints for one layer.

        The axes rather than the renderer's record: the record is what a redraw is decided from, so reading
        it back would be asking the decision about itself. A matplotlib artist is a stable object, so its
        identity is what tells a layer drawn again from one merely re-described.

        Args:
            tier: The map.
            layer_id: The layer to look up.

        Returns:
            The artist, or `None` when nothing the layer owns is on the axes.
        """
        owner = _owners(tier)
        for artist in _painted(tier.ax) or ():
            if owner.get(id(artist)) == layer_id:
                return artist
        return None


class TestStaticLayerManagement(LayerManagementConformance):
    """The static tier, signing the contract the 3-D tier already passes."""

    contract = StaticContract()


@pytest.fixture
def mixed(dataset):
    """Yield a map holding a raster field under a point layer — one band, two kinds, two z-orders.

    The shared `draw_two` is two layers of **one** kind by design, so both carry one z-order and a move
    between them is expressible by the artist list alone. This is the composition the shared probe cannot
    build and review H5 was found in: `AxesImage` (z-order 0) below `PathCollection` (z-order 1), both in
    the `data` band, so a move between them has to cross a z-order to reach the picture.

    Args:
        dataset: The session's pyramids raster.

    Yields:
        The map, closed on the way out.
    """
    built = Map(crs=dataset.epsg)
    built.field(dataset, name="grid")
    built.points(FeatureCollection.read_file("tests/data/points.geojson"), name="obs")
    yield built
    built.close()


class TestAMoveAcrossTwoZOrders:
    """`move_layer` between layers whose artists carry different z-orders (review H5).

    Every probe here reads :func:`paint_order` — matplotlib's own stable sort — so a reorder that only
    rearranged `Axes._children` fails them, which is what the record-reading probe could not do.
    """

    def test_the_two_kinds_start_out_on_different_z_orders(self, mixed):
        """The premise: without it the rest of this class proves nothing the one-kind probe cannot.

        Args:
            mixed: A raster field under a point layer.
        """
        zorders = sorted({artist.get_zorder() for artist in layer_artists(mixed)})
        assert len(zorders) == 2, f"both layers painted at one z-order: {zorders}"

    def test_the_move_reaches_what_matplotlib_paints(self, mixed):
        """The finding itself: the picture, not the record, has to follow the move.

        Args:
            mixed: A raster field under a point layer.

        Test scenario:
            The points are drawn over the raster, then moved to the bottom. Before the fix the axes' list
            moved and the painted order did not, so the figure kept drawing the points on top while
            `layer_ids` said they were underneath.
        """
        assert paint_order(mixed) == ("grid", "obs"), (
            f"the points were not painted over the raster to begin with: {paint_order(mixed)}"
        )
        mixed.move_layer("obs", 0)
        assert paint_order(mixed) == ("obs", "grid"), (
            f"the description says {mixed.layer_ids} and the axes paints {paint_order(mixed)}"
        )

    def test_the_description_and_the_painted_order_agree_after_a_move(self, mixed):
        """The two readings are built from different structures, and a move is what makes them agree.

        Args:
            mixed: A raster field under a point layer.
        """
        mixed.move_layer("grid", -1)
        assert list(paint_order(mixed)) == mixed.layer_ids, (
            f"painted {paint_order(mixed)}, described {mixed.layer_ids}"
        )

    def test_the_layers_keep_the_z_orders_they_held_between_them(self, mixed):
        """A move deals the layers' own z-orders back out; it does not invent a new scale.

        Args:
            mixed: A raster field under a point layer.

        Test scenario:
            The multiset is what places the layers against everything the reorder must not touch — a
            basemap far below, a coastline far above, a colorbar's axes. Preserving it is what keeps a move
            inside the band's own slice of the z stack.
        """
        before = sorted(artist.get_zorder() for artist in layer_artists(mixed))
        mixed.move_layer("obs", 0)
        assert (
            sorted(artist.get_zorder() for artist in layer_artists(mixed)) == before
        ), (
            f"the z-orders went from {before} to "
            f"{sorted(artist.get_zorder() for artist in layer_artists(mixed))}"
        )

    def test_moving_the_layer_back_paints_the_original_order_again(self, mixed):
        """A move is undoable, which a rewritten z-order scale would not be.

        Args:
            mixed: A raster field under a point layer.
        """
        mixed.move_layer("obs", 0)
        mixed.move_layer("obs", -1)
        assert paint_order(mixed) == ("grid", "obs"), (
            f"the round trip left the axes painting {paint_order(mixed)}"
        )

    def test_the_engine_reading_is_the_painted_order_not_the_addition_order(
        self, dataset
    ):
        """The probe's own semantics, pinned where the two readings disagree with no move involved.

        Args:
            dataset: The session's pyramids raster.

        Test scenario:
            A caller's explicit `zorder=` on the raster puts it above the points it was added before, so
            `Axes._children` reads `('grid', 'obs')` while matplotlib paints `('obs', 'grid')`. A probe
            reading the list would report the first and measure nothing.
        """
        built = Map(crs=dataset.epsg)
        built.field(dataset, name="grid", zorder=5)
        built.points(
            FeatureCollection.read_file("tests/data/points.geojson"), name="obs"
        )
        added = tuple(
            dict.fromkeys(_owners(built)[id(artist)] for artist in layer_artists(built))
        )
        try:
            assert added == ("grid", "obs"), f"the axes listed them as {added}"
            assert StaticContract().engine_order(built) == ("obs", "grid"), (
                f"the probe reported {StaticContract().engine_order(built)}"
            )
        finally:
            built.close()


@pytest.fixture
def blank():
    """Yield an empty map, closed on the way out.

    Yields:
        A `Map` in EPSG:4326.
    """
    built = Map(crs=4326)
    yield built
    built.close()


class TestAnArtistTheCallerBuiltThemselves:
    """`add_layer` — the Core spelling for the custom layer this tier held privately until order 23."""

    def test_the_artist_is_described_as_a_custom_layer(self, blank):
        """A caller's own artist is addressable like any other layer.

        Args:
            blank: The map under test.
        """
        blank.add_layer(blank.ax.add_artist(Text(0.0, 0.0, "mine")), name="mine")
        assert blank.get_layer("mine").kind == "custom:matplotlib", (
            f"the artist was filed as {blank.get_layer('mine').kind!r}"
        )

    def test_the_map_comes_back_so_calls_chain(self, blank):
        """Every Core builder returns the map; `add_layer` is one.

        Args:
            blank: The map under test.
        """
        artist = blank.ax.add_artist(Text(0.0, 0.0, "mine"))
        assert blank.add_layer(artist) is blank, "add_layer must return the map"

    def test_an_unnamed_artist_is_numbered_by_what_it_is(self, blank):
        """A generated id counts the kind, and a colon cannot sit in the middle of one.

        Args:
            blank: The map under test.
        """
        blank.add_layer(blank.ax.add_artist(Text(0.0, 0.0, "mine")))
        assert blank.layer_ids == ["custom-1"], blank.layer_ids

    def test_the_band_the_caller_names_is_where_the_layer_is_drawn(self, blank):
        """`custom:matplotlib` names the engine and says nothing about what the artist draws.

        Args:
            blank: The map under test.

        Test scenario:
            The kind's own band is `data`, so a caller who hands in a backdrop has no way to say so
            without this keyword — and the figure would describe a ground cover among the data.
        """
        blank.text(0.0, 0.0, "label", name="label")
        blank.add_layer(
            blank.ax.add_artist(Text(2.0, 2.0, "under")), name="under", band="underlay"
        )
        assert blank.layer_ids == ["under", "label"], blank.layer_ids

    def test_an_artist_handed_in_hidden_is_described_hidden(self, blank):
        """The visibility is read off the object, as the web tier reads it off a MapLibre layer.

        Args:
            blank: The map under test.

        Test scenario:
            Review L10 on the web tier: a layer built with `layout={"visibility": "none"}` was registered
            visible, so the figure described a layer the page did not draw. The same keystroke here is an
            artist built with `visible=False`.
        """
        hidden = Text(0.0, 0.0, "mine")
        hidden.set_visible(False)
        blank.add_layer(blank.ax.add_artist(hidden), name="mine")
        assert blank.figure_spec.layers.is_visible("mine") is False, (
            "an artist handed in hidden was described visible"
        )

    def test_a_band_the_tier_does_not_have_is_refused(self, blank):
        """A misspelled band would put the layer somewhere no band describes.

        Args:
            blank: The map under test.
        """
        artist = blank.ax.add_artist(Text(0.0, 0.0, "mine"))
        with pytest.raises(ValueError):
            blank.add_layer(artist, band="on-top-of-everything")


@pytest.fixture
def labelled(blank):
    """Yield a map holding one text layer named `ams`, the id every refusal below names.

    Args:
        blank: An empty map.

    Returns:
        The map, with the layer drawn.
    """
    blank.text(4.9, 52.4, "Amsterdam", name="ams")
    return blank


class TestWhatReplaceLayerRefusesAndSaysItRefuses:
    """`replace_layer`'s two refusals, and the docstring a caller writes their `except` clause from."""

    def test_the_refusal_class_the_docstring_promises_is_the_one_raised(self, labelled):
        """A `Raises:` block naming the wrong class is an `except` clause the exception walks straight past.

        Args:
            labelled: A map with one text layer.

        Test scenario:
            Review M1: the gate became `Capabilities.require`, which raises `CapabilityError`, while the
            docstring still promised `KeyError` for a kind the tier cannot draw. The two readings are built
            differently on purpose — one by running the call, one by reading what it says about itself.
        """
        replacement = with_fields(labelled.get_layer("ams"), kind="custom:pyvista")
        with pytest.raises(CapabilityError) as refusal:
            labelled.replace_layer(replacement)
        raised = type(refusal.value).__name__
        assert raised in (Map.replace_layer.__doc__ or ""), (
            f"replace_layer raises {raised} and its docstring does not name it"
        )

    def test_a_replacement_that_is_not_a_spec_is_refused_by_type(self, labelled):
        """Review L1: an id passed where the description belongs was answered with "no layer None".

        Args:
            labelled: A map with one text layer.

        Test scenario:
            `getattr(layer, "id", None)` made the id lookup fail first, so the message named an id the
            caller never passed and said nothing about the argument that was actually wrong.
        """
        with pytest.raises(ValueError, match=r"needs a LayerSpec.*got str"):
            labelled.replace_layer("ams")

    def test_the_refusal_names_what_was_passed_rather_than_a_missing_id(self, labelled):
        """The message a caller reads has to point at the argument, not at the layer list.

        Args:
            labelled: A map with one text layer.
        """
        with pytest.raises(ValueError) as refusal:
            labelled.replace_layer(42)
        assert "no layer" not in str(refusal.value), (
            f"the refusal blamed a missing layer: {refusal.value}"
        )
