"""The web tier's half of order 23: the public toggle, reorder and replace over its own renderer.

`get_layer` and `remove_layer` were already here — the identity half landed with the seam — so what order 23
adds on this tier is `set_visible`, `move_layer` and `replace_layer`, and the change path all three take.

The shared probes are :mod:`tests.base.layer_management`; this module supplies the adapter and the checks that
are this tier's own: what happens to a queue entry **no layer id addresses**, which is where the tier still
keeps its basemap, terrain, point-cloud and model layers.

**Why the engine reading is the queue.** `_build_map_widget` replays `WebMap._queued` into the widget, entry by
entry, and MapLibre draws a layer over the ones added before it — so the queue's order is the page's order.
Until order 23 only the builders and `remove_layer` wrote that list and `Renderer.apply` reconciled the
renderer's record beside it, so a change reached a record and never the page (review M1).
"""

import pytest

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

    A caller's own `apply(widget)` callable stands in for the entries the tier's unconverted builders queue —
    the basemap, terrain, point-cloud and model layers named in the renderer contract's `UNDRAWN_KINDS`. It is
    put in the underlay band, which is where `tiles()` queues its own.

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
        from digitalearth.base.spec import LayerSpec

        held = (mixed._underlay_count, mixed._reference_count, mixed._overlay_count)
        with pytest.raises(KeyError):
            mixed.replace_layer(LayerSpec(TOP, "custom:pyvista", band="overlay"))
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
