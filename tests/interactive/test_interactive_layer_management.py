"""The interactive tier's half of order 23: the public toggle, reorder, replace, read-back and removal.

The shared probes are :mod:`tests.base.layer_management`; this module supplies the adapter and the checks
that are this tier's own — chiefly what happens to a **custom** layer, the one kind this tier keeps outside
its renderer and so cannot draw again from a description.

**Why the engine reading is the overlay list.** `render()` composes
:attr:`~digitalearth.interactive.base.InteractiveMapBase.layers` with `*`, bottom first, so that list is what
a viewer sees. Until order 23 only the builders wrote it and `Renderer.apply` reconciled the renderer's own
record beside it — so a change reached a record and never the picture (review M1). The probes read the list.
"""

import pytest

from digitalearth.interactive import InteractiveMap

pytest.importorskip(
    "geoviews", reason="the interactive tier needs the interactive environment"
)
from tests.base.layer_management import (
    BOTTOM,
    TOP,
    LayerManagementConformance,
    LayerManagementContract,
)


class InteractiveContract(LayerManagementContract):
    """The interactive tier's adapter for the shared layer-management probes."""

    backend = "interactive"
    #: A caller's own PyVista object: registered, drawn from no data, and nothing this tier can draw.
    undrawable_kind = "custom:pyvista"

    def make(self):
        """Return an empty map.

        Returns:
            The map. Constructing one touches no engine; the builders lazy-import HoloViz.
        """
        return InteractiveMap()

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
        """Return the layer ids in the order `render()` overlays their elements.

        Args:
            tier: The map.

        Returns:
            The ids, bottom first, read by matching each overlaid element against what the renderer drew —
            so this answers from the list the overlay is composed from rather than from the record.
        """
        owner = {
            id(drawn.element): layer_id
            for layer_id, drawn in tier._renderer.drawn.items()
        }
        return tuple(
            owner[id(element)] for element in tier.layers if id(element) in owner
        )

    def engine_objects(self, tier):
        """Return every element `render()` would overlay.

        Args:
            tier: The map.

        Returns:
            The elements themselves, so a removal is held to having taken the very element it drew out of
            the overlay rather than only out of the renderer's record — which `Renderer.remove` clears
            whether or not the picture follows.
        """
        return tuple(tier.layers)

    def engine_holds(self, tier, layer_id):
        """Return the element the overlay holds for one layer.

        The overlay rather than the renderer's record: the record is what decides whether the overlay is
        re-arranged, so a probe reading it back would be asking the decision about itself. `layers` is kept
        in the tree's order — each element is placed at its layer's index — so the index is the key.

        Args:
            tier: The map.
            layer_id: The layer to look up.

        Returns:
            The element, or `None` when the map overlays none for it.
        """
        ids = tier.layer_ids
        if layer_id not in ids:
            return None
        return tier.layers[ids.index(layer_id)]


class TestInteractiveLayerManagement(LayerManagementConformance):
    """The interactive tier, signing the contract the 3-D tier already passes."""

    contract = InteractiveContract()


@pytest.fixture
def held():
    """Yield a map holding two of the caller's own objects, engine-free.

    A string stands in for a HoloViews element: the registry does not inspect what it is handed, which is
    what lets these checks run without the `interactive` extra — and a custom layer is exactly the case the
    renderer has no drawer for, so no engine would be reached even with one installed.

    Yields:
        The map, with `lower` and `upper` registered.
    """
    built = InteractiveMap()
    built.add_layer("lower-element", name=BOTTOM)
    built.add_layer("upper-element", name=TOP)
    yield built
    built.close()


class TestACallersOwnElement:
    """`custom:holoviews` — described, addressable, and the one kind no description can rebuild."""

    def test_removing_it_takes_it_out_of_the_overlay(self, held):
        """The element goes with the description, or `render()` keeps composing it.

        Args:
            held: The map under test.

        Test scenario:
            A custom layer is not in the renderer's record — the tier keeps the caller's object outside it —
            so a removal that only cleared the record would leave the element overlaid under no id at all.
        """
        held.remove_layer(BOTTOM)
        assert held.layers == ["upper-element"], held.layers

    def test_moving_it_reorders_the_overlay(self, held):
        """A move has to reach the list `render()` composes, custom layer or not.

        Args:
            held: The map under test.
        """
        held.move_layer(TOP, 0)
        assert held.layers == ["upper-element", "lower-element"], held.layers

    def test_replacing_its_description_leaves_the_element_where_it_is(self, held):
        """There is nothing to draw again: what a figure keeps of a custom layer is its description.

        Args:
            held: The map under test.

        Test scenario:
            The element is the caller's and the tier holds no recipe for it, so a replacement re-describes
            the layer and must not drop the object — which is what rebuilding the overlay from the
            renderer's record alone would have done.
        """
        from dataclasses import replace as with_fields

        held.replace_layer(with_fields(held.get_layer(BOTTOM), label="Lower"))
        assert held.layers == ["lower-element", "upper-element"], held.layers

    def test_hiding_it_is_described_although_no_element_can_say_so(self, held):
        """Honest half-measure: the description records it, and the overlay cannot express it.

        Args:
            held: The map under test.

        Test scenario:
            `set_visible` writes a HoloViews option onto the element the renderer recorded, and a custom
            layer has no record — so the figure says hidden and `render()` still overlays it. Pinned rather
            than left implied, because the alternative reading is that the tier drops the layer.
        """
        held.set_visible(BOTTOM, False)
        assert (held.figure_spec.layers.is_visible(BOTTOM), held.layers) == (
            False,
            ["lower-element", "upper-element"],
        ), held.layers

    def test_the_removed_layer_stops_being_the_one_the_toggles_act_on(self, held):
        """`colorbar()` and `legend()` act on the layer added last; a removed one is not it.

        Args:
            held: The map under test.

        Test scenario:
            The pointer is resolved by looking the id up in the tree, so leaving it on a removed layer made
            the next `colorbar()` answer `ValueError: tuple.index(x): x not in tuple` — review N8, found on
            the `_forget_layer` path and reachable the same way from here.
        """
        held.remove_layer(TOP)
        assert held._last_layer_index("colorbar") == 0, held._last_layer_id
