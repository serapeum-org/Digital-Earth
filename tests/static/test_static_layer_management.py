"""The static tier's half of order 23: the public toggle, reorder, replace, read-back and removal.

The shared probes are :mod:`tests.base.layer_management`; this module supplies the adapter and the checks
that are this tier's own — `add_layer`, which is where a caller's own matplotlib artist joins the figure,
and the painting order the reorder has to reach.

**Why the engine reading is the axes' artist list.** matplotlib draws the artists of one z-order in the
order they were added, and `Axes._children` is that order — so reordering the list reorders the picture
(`digitalearth.static.renderer._painted` says so, and the rollback has always relied on it). Reading
`Renderer.drawn` instead would report a dict the tier keeps beside the axes, which a move could reorder
while the figure came out unchanged.
"""

import matplotlib

matplotlib.use("Agg")

import pytest
from matplotlib.text import Text

from digitalearth.static import Map
from digitalearth.static.renderer import _painted
from tests.base.layer_management import (
    BOTTOM,
    TOP,
    LayerManagementConformance,
    LayerManagementContract,
)


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

    @staticmethod
    def _owners(tier):
        """Return the id of the artist each layer owns, keyed by `id()` of the artist.

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

    def engine_order(self, tier):
        """Return the layer ids in the order the axes paints them.

        Args:
            tier: The map.

        Returns:
            The ids, bottom first, with an artist no layer owns — the axes' own spines, its patch — left
            out.
        """
        owner = self._owners(tier)
        painted = []
        for artist in _painted(tier.ax) or ():
            layer_id = owner.get(id(artist))
            if layer_id is not None and layer_id not in painted:
                painted.append(layer_id)
        return tuple(painted)

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
        owner = self._owners(tier)
        for artist in _painted(tier.ax) or ():
            if owner.get(id(artist)) == layer_id:
                return artist
        return None


class TestStaticLayerManagement(LayerManagementConformance):
    """The static tier, signing the contract the 3-D tier already passes."""

    contract = StaticContract()


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
