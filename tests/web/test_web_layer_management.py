"""The web tier's half of order 23: the public toggle, reorder and replace over its own renderer.

`get_layer` and `remove_layer` were already here — the identity half landed with the seam — so what order 23
adds on this tier is `set_visible`, `move_layer` and `replace_layer`, and the change path all three take.

The shared probes are :mod:`tests.base.layer_management`; this module supplies the adapter and the checks that
are this tier's own: what happens to a queue entry **no layer id addresses**, and what happens to the four
kinds that used to be nothing but such an entry.

**The last gap.** `basemap`, `terrain`, `point_cloud` and `model` queued a closure and recorded no layer, so
the three methods above — which address a layer by its id — could not reach them at all: they were invisible
to the very feature order 23 added. Each records a layer now, and `TestTheKindsThatUsedToRecordNoLayer` below
holds that to the page rather than to the description, because recording a layer nothing draws differently is
the drift this seam exists to remove.

**Why the engine reading is the queue.** `_build_map_widget` replays `WebMap._queued` into the widget, entry by
entry, and MapLibre draws a layer over the ones added before it — so the queue's order is the page's order.
Until order 23 only the builders and `remove_layer` wrote that list and `Renderer.apply` reconciled the
renderer's record beside it, so a change reached a record and never the page (review M1).
"""

import pytest

pytest.importorskip("maplibre", reason="the web tier needs the web environment")

from digitalearth.web import WebMap
from digitalearth.web.base import _Described
from tests.base.layer_management import (
    BOTTOM,
    TOP,
    LayerManagementConformance,
    LayerManagementContract,
)


class WebContract(LayerManagementContract):
    """The web tier's adapter for the shared layer-management probes."""

    backend = "web"
    #: A caller's own PyVista object: registered, drawn from no data, and nothing this tier can draw.
    undrawable_kind = "custom:pyvista"

    def make(self):
        """Return an empty map.

        Returns:
            The map.
        """
        return WebMap()

    def draw_two(self, tier):
        """Draw two text labels, which are one kind and so share one band.

        Args:
            tier: The map.

        Returns:
            The two ids, bottom first.
        """
        tier.text(0.0, 0.0, "lower", name=BOTTOM)
        tier.text(1.0, 1.0, "upper", name=TOP)
        return BOTTOM, TOP

    def engine_order(self, tier):
        """Return the layer ids in the order the widget adds them.

        Args:
            tier: The map.

        Returns:
            The ids, bottom first, read off the MapLibre layers `layers` resolves the queue into — so a queue
            entry the record cannot resolve is absent here, as it is from the page.
        """
        return tuple(
            drawn.id for drawn in tier.layers if getattr(drawn, "id", None) is not None
        )

    def engine_objects(self, tier):
        """Return every MapLibre layer the widget would be given.

        Args:
            tier: The map.

        Returns:
            The layer objects themselves, so a removal is held to having taken the very layer it built out of
            the queue rather than only out of the renderer's record.
        """
        return tuple(tier.layers)

    def engine_holds(self, tier, layer_id):
        """Return the MapLibre layer the queue resolves for one id.

        The queue rather than the renderer's record: the record is what decides whether the queue is
        re-arranged, so a probe reading it back would be asking the decision about itself.

        Args:
            tier: The map.
            layer_id: The layer to look up.

        Returns:
            The MapLibre layer, or `None` when the page would add none for it.
        """
        for drawn in tier.layers:
            if getattr(drawn, "id", None) == layer_id:
                return drawn
        return None


class TestWebLayerManagement(LayerManagementConformance):
    """The web tier, signing the contract the 3-D tier already passes."""

    contract = WebContract()


@pytest.fixture
def mixed():
    """Yield a map holding a queued entry no layer id addresses, beneath two described labels.

    A caller's own `apply(widget)` callable, handed straight to `add_underlay`, is the entry: it is the shape
    the tier's own builders queued for `basemap`, `terrain`, `point_cloud` and `model` until each started
    recording a layer, and it is still what a caller who wires their own MapLibre calls gets. The underlay
    band, because that is where `tiles()` files its own.

    Yields:
        The map.
    """

    def draw(widget):
        """Stand in for a builder's own MapLibre call.

        Args:
            widget: The widget the entry would be applied to.
        """

    built = WebMap()
    built.add_underlay(draw)
    built.text(0.0, 0.0, "lower", name=BOTTOM)
    built.text(1.0, 1.0, "upper", name=TOP)
    yield built
    built.close()


class TestAQueueEntryNoLayerIdAddresses:
    """The tier's unconverted builders queue a closure; a reorder must leave it exactly where it is."""

    @staticmethod
    def _shape(tier):
        """Return the queue as a comparable shape: a layer id, or `None` for a closure.

        Args:
            tier: The map.

        Returns:
            One entry per queued item, in order.
        """
        return [
            entry.layer_id if isinstance(entry, _Described) else None
            for entry in tier._queued
        ]

    def test_a_reorder_moves_the_described_layers_and_not_the_closure(self, mixed):
        """A closure is not addressed by an id, so nothing about it is being reordered.

        Args:
            mixed: The map under test.
        """
        mixed.move_layer(TOP, 0)
        assert self._shape(mixed) == [None, TOP, BOTTOM], self._shape(mixed)

    def test_the_band_counts_still_describe_the_queue(self, mixed):
        """The queue is addressed by counts from its two ends, so a reorder has to keep them true.

        Args:
            mixed: The map under test.

        Test scenario:
            `add_underlay` inserts by `_underlay_count` and `add_overlay` from the back. A reorder that left
            a count pointing at the wrong slot would put the *next* basemap above the data it belongs under —
            silently, and only on the next call.
        """
        mixed.move_layer(TOP, 0)
        counts = (
            mixed._underlay_count,
            mixed._reference_count,
            mixed._overlay_count,
        )
        assert counts == (1, 0, 2), counts

    def test_a_refused_change_leaves_the_band_counts_alone(self, mixed):
        """The counts roll back with the queue, or the next layer lands in the wrong band.

        Args:
            mixed: The map under test.
        """
        from digitalearth.base.capabilities import CapabilityError
        from digitalearth.base.spec import LayerSpec

        held = (mixed._underlay_count, mixed._reference_count, mixed._overlay_count)
        described = LayerSpec(TOP, "custom:pyvista", band="overlay")
        with pytest.raises(CapabilityError):
            mixed.replace_layer(described)
        counts = (
            mixed._underlay_count,
            mixed._reference_count,
            mixed._overlay_count,
        )
        assert counts == held, counts

    def test_an_arrangement_that_stops_part_way_puts_the_queue_back(
        self, mixed, monkeypatch
    ):
        """The queue and its counts go back together, or a count addresses a slot that is not there.

        Args:
            mixed: The map under test.
            monkeypatch: Stands a failure into the arrangement itself.

        Test scenario:
            The arrangement writes the list and then the three counts, so an interruption between them leaves
            the map addressing its bands by counts that no longer describe the queue — and the next
            `add_underlay` puts a basemap above the data. Provoked rather than waited for: nothing in the
            reconcile touches the queue, so this restore is unreachable from a refused *figure* and would be
            dead code with only that check behind it.
        """
        held = list(mixed._queued)

        def refuse(_after):
            """Stand in for an arrangement that empties the queue and then fails.

            Args:
                _after: The figure being arranged, unread.

            Raises:
                RuntimeError: always.
            """
            mixed._queued.clear()
            raise RuntimeError("stopped part-way")

        monkeypatch.setattr(mixed._renderer, "_arrange", refuse)
        with pytest.raises(RuntimeError):
            mixed.move_layer(TOP, 0)
        assert mixed._queued == held, self._shape(mixed)

    def test_a_removed_layer_gives_its_slot_up_without_moving_the_closure(self, mixed):
        """`remove_layer` is the tier's own path, and the arrangement has to agree with it.

        Args:
            mixed: The map under test.

        Test scenario:
            The removal and the reconcile are two ways to take a layer out of the queue — `remove_layer`
            filters the list itself — so this pins that a later reorder reads the queue the removal left and
            not a stale count.
        """
        mixed.remove_layer(BOTTOM)
        mixed.move_layer(TOP, 0)
        assert self._shape(mixed) == [None, TOP], self._shape(mixed)


#: The four kinds that queued a closure and recorded no layer until this gap was closed, resolved to a builder
#: call that takes a `name=` — so two of one kind can be built and reordered against each other inside the one
#: band their kind declares, which is the only move `LayerTree.move` allows.
#:
#: They are the same four `WEB_UNDRAWN_KINDS` in `tests/base/test_renderer_conformance.py` named, and the same
#: four `DRAWN_BUT_NOT_DESCRIBED` in `tests/web/test_web_capabilities.py` held a builder call for. Both lists
#: emptied when these were converted; this one is what holds the conversion to the *page*.
CONVERTED = {
    "basemap": lambda tier, name: tier.tiles(
        f"https://tiles.example/{name}/{{z}}/{{x}}/{{y}}.png", name=name
    ),
    "terrain": lambda tier, name: tier.terrain_tiles(name=name),
    "point_cloud": lambda tier, name: tier.point_cloud(
        [(4.9, 52.4, 10.0), (5.0, 52.2, 20.0)], name=name
    ),
    "model": lambda tier, name: tier.gltf(
        "https://example.invalid/model.glb", 4.9, 52.4, name=name
    ),
}


def _page_calls(tier):
    """Return what building the page asks of MapLibre, in order.

    The widget's own record, not a stand-in: `MapWidget.add_call` queues every call it will send, so this is
    the page as the viewer would get it.

    Args:
        tier: The map.

    Returns:
        A list of `(method, args)` pairs, in the order the build makes them.
    """
    return list(tier._build_map_widget()._message_queue)


def _page_layer_ids(tier):
    """Return the ids the built page draws, in the order it draws them.

    Each of the three routes is read in its own terms, because none of them is the others: a style layer is an
    ``addLayer`` whose ``layout.visibility`` is not ``none``; terrain is a ``setTerrain`` naming the source
    derived from the layer's id, and a hidden one is **not called at all**; and a deck.gl layer is an entry of
    the one ``addDeckOverlay`` whose own ``visible`` property is not off.

    Args:
        tier: The map.

    Returns:
        The layer ids the viewer would see, bottom first. The deck overlay is one call, so the ids it carries
        follow the style layers whatever the queue says — within it they keep the queue's order.
    """
    drawn = []
    for method, args in _page_calls(tier):
        if method == "addLayer":
            layer = args[0]
            if (layer.get("layout") or {}).get("visibility") != "none":
                drawn.append(layer["id"])
        elif method == "setTerrain":
            drawn.append(args[0]["source"].removesuffix("-src"))
        elif method == "addDeckOverlay":
            drawn.extend(layer["id"] for layer in args[0] if layer.get("visible", True))
    return drawn


@pytest.fixture(params=sorted(CONVERTED))
def converted(request):
    """Yield the kind under test and a map holding two layers of it, named `BOTTOM` and `TOP`.

    Two, so a reorder has somewhere to go inside the band the kind declares — a basemap cannot be dragged over
    the data, which is what `LayerTree.move` refuses and what would make a single layer untestable here.

    Args:
        request: pytest's request, carrying the kind under test.

    Yields:
        The kind, and the map.
    """
    build = CONVERTED[request.param]
    built = WebMap()
    build(built, BOTTOM)
    build(built, TOP)
    yield request.param, built
    built.close()


class TestTheKindsThatUsedToRecordNoLayer:
    """`basemap`, `terrain`, `point_cloud` and `model`, held to the page rather than to the record."""

    def test_the_builder_records_a_layer_under_the_name_it_was_given(self, converted):
        """A layer id is what all three methods address, and these builders minted none.

        Args:
            converted: The kind under test and a map holding two of it.
        """
        kind, tier = converted
        assert tier.layer_ids == [BOTTOM, TOP], tier.layer_ids
        kinds = [layer.kind for layer in tier.figure_spec.layers]
        assert kinds == [kind, kind], kinds

    def test_the_page_draws_both_of_them(self, converted):
        """The description has to be what the page is built from, or hiding one would change nothing.

        Args:
            converted: The kind under test and a map holding two of it.
        """
        _, tier = converted
        assert _page_layer_ids(tier) == [BOTTOM, TOP], _page_layer_ids(tier)

    def test_hiding_one_takes_it_off_the_page(self, converted):
        """`set_visible` reaches the viewer, not only the renderer's record.

        Args:
            converted: The kind under test and a map holding two of it.
        """
        _, tier = converted
        tier.set_visible(BOTTOM, False)
        assert _page_layer_ids(tier) == [TOP], _page_layer_ids(tier)

    def test_showing_it_again_puts_it_back(self, converted):
        """The other direction, so a fix that only ever hid could not pass.

        Args:
            converted: The kind under test and a map holding two of it.
        """
        _, tier = converted
        tier.set_visible(BOTTOM, False)
        tier.set_visible(BOTTOM, True)
        assert _page_layer_ids(tier) == [BOTTOM, TOP], _page_layer_ids(tier)

    def test_the_renderer_reports_the_visibility_it_was_asked_for(self, converted):
        """Every tier's renderer answers `is_visible`, and these routes had no answer to give at all.

        Args:
            converted: The kind under test and a map holding two of it.
        """
        _, tier = converted
        tier.set_visible(BOTTOM, False)
        assert tier._renderer.is_visible(BOTTOM) is False
        assert tier._renderer.is_visible(TOP) is True

    def test_moving_one_changes_what_the_page_draws_last(self, converted):
        """`move_layer` is the reorder, and what MapLibre adds last it draws on top.

        Args:
            converted: The kind under test and a map holding two of it.
        """
        _, tier = converted
        tier.move_layer(BOTTOM, 1)
        assert _page_layer_ids(tier) == [TOP, BOTTOM], _page_layer_ids(tier)

    def test_removing_one_takes_it_off_the_page(self, converted):
        """`remove_layer` was already public; it had nothing to address on these four.

        Args:
            converted: The kind under test and a map holding two of it.
        """
        _, tier = converted
        tier.remove_layer(BOTTOM)
        assert _page_layer_ids(tier) == [TOP], _page_layer_ids(tier)

    def test_replacing_one_keeps_its_id_and_its_place(self, converted):
        """A replacement is drawn again from the new description, in the slot the old one held.

        Args:
            converted: The kind under test and a map holding two of it.

        Test scenario:
            `group` is the portable "restyle that must reach the engine" the shared probes use: every tier's
            redraw guard compares `(symbology, filter, group)`, and it needs no knowledge of which props this
            tier's drawers read.
        """
        from dataclasses import replace as with_fields

        _, tier = converted
        tier.replace_layer(with_fields(tier.get_layer(BOTTOM), group="regrouped"))
        assert tier.get_layer(BOTTOM).group == "regrouped"
        assert _page_layer_ids(tier) == [BOTTOM, TOP], _page_layer_ids(tier)


class TestTheDeckOverlayIsOneCall:
    """deck.gl owns one overlay per page, and two kinds now reach it from their own queue slots."""

    def test_a_described_cloud_and_an_undescribed_scatter_share_one_overlay(self):
        """A second ``addDeckOverlay`` replaces the first, so every deck layer has to go in one call.

        Test scenario:
            `point_cloud` and `model` are described and join the overlay from their queue slots;
            `deck_scatter`, `deck_polygons` and `tiles_3d` record nothing and accumulate into
            `WebMap._deck_layers`. There used to be one accumulator and one closure, so one call was
            automatic. The page composes the overlay from the queue now, and a composition that left either
            source out — or issued a call per source — would drop layers silently.
        """
        import geopandas as gpd
        from shapely.geometry import Point

        frame = gpd.GeoDataFrame(
            geometry=[Point(4.9, 52.4), Point(5.0, 52.2)], crs=4326
        )
        tier = WebMap()
        tier.deck_scatter(frame)
        tier.point_cloud([(4.9, 52.4, 10.0)], name="cloud")
        tier.gltf("https://example.invalid/model.glb", 4.9, 52.4, name="statue")
        overlays = [
            args[0] for method, args in _page_calls(tier) if method == "addDeckOverlay"
        ]
        assert len(overlays) == 1, (
            f"{len(overlays)} deck overlays; all but the last are lost"
        )
        ids = [layer.get("id") for layer in overlays[0]]
        assert ids[1:] == ["cloud", "statue"], ids
        tier.close()

    def test_reordering_a_described_cloud_reorders_the_overlay(self):
        """The overlay is composed from the queue, so `move_layer` reaches deck.gl's draw order too."""
        tier = WebMap()
        tier.point_cloud([(4.9, 52.4, 10.0)], name="lower")
        tier.point_cloud([(5.0, 52.2, 20.0)], name="upper")
        tier.move_layer("lower", 1)
        assert _page_layer_ids(tier) == ["upper", "lower"], _page_layer_ids(tier)
        tier.close()
