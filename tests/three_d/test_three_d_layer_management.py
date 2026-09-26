"""The 3-D tier under the shared layer-management probes — the tier order 23 had nothing to build on.

`Scene3D` answered to all six methods before this order; the other three tiers did not, which is why it is
the implementation they were written against. Subscribing it here is not coverage of new code — it is what
makes the probes a statement about the **package** rather than about the three tiers that changed, so a
divergence between the reference and its copies fails here rather than being discovered by a caller.

Subscribing it closed a divergence rather than recording one. `replace_layer` here used to reach its drawer
table part-way through the reconcile and answer with `KeyError`, where the other three check the new kind
against the tier's own `Capabilities` and refuse with `CapabilityError` before anything is built; this tier now
does the same, so all four refuse from what they declare and
:attr:`~tests.base.layer_management.LayerManagementContract.refuses_by_declaration` is left at its default.

**The engine has no draw order to read.** VTK composites its actors by depth, not by the order they were
added, so `move_layer` on this tier records draw order and does not apply it — the scene's own docstring says
so. :meth:`ThreeDContract.engine_order` answers `None` and the reorder probe skips, loudly.
"""

import importlib.util

import numpy as np
import pytest

from tests.base.layer_management import (
    BOTTOM,
    TOP,
    LayerManagementConformance,
    LayerManagementContract,
)

needs_pyvista = pytest.mark.skipif(
    importlib.util.find_spec("pyvista") is None,
    reason="the 3-D tier needs the viz3d environment",
)


class ThreeDContract(LayerManagementContract):
    """The 3-D tier's adapter for the shared layer-management probes."""

    backend = "3d"
    #: A string at a coordinate: registered, drawn from no data, and a 2-D tier's layer throughout.
    undrawable_kind = "text"

    def make(self):
        """Return an off-screen scene.

        Returns:
            The scene.
        """
        from digitalearth.three_d import Scene3D

        return Scene3D(off_screen=True)

    def draw_two(self, tier):
        """Draw two terrain surfaces, which are one kind and so share one band.

        Args:
            tier: The scene.

        Returns:
            The two ids, bottom first.
        """
        from digitalearth.base.sources import get_source

        dem = get_source(np.add.outer(np.arange(4.0), np.arange(5.0)))
        tier.terrain(dem, name=BOTTOM)
        tier.terrain(dem, name=TOP)
        return BOTTOM, TOP

    def engine_order(self, tier):
        """Report that there is no draw order to read.

        Args:
            tier: The scene.

        Returns:
            `None` — VTK composites by depth, so the order actors were added in does not decide what is in
            front, and a probe reading one would be reading something the tier deliberately does not apply.
        """
        return None

    def engine_holds(self, tier, layer_id):
        """Return the actor the plotter holds for one layer.

        Args:
            tier: The scene.
            layer_id: The layer to look up.

        Returns:
            The actor, or `None` when nothing was drawn for it.
        """
        if layer_id not in tier.layer_ids:
            return None
        return tier.actor_of(layer_id)

    def engine_objects(self, tier):
        """Return every actor the plotter renders.

        Args:
            tier: The scene.

        Returns:
            The actors themselves — a layer's mesh and the scalar bar beside it — so a removal is held to
            having taken the very actor it drew off the plotter.
        """
        return tuple(tier.plotter.renderer.actors.values())


@needs_pyvista
class TestThreeDLayerManagement(LayerManagementConformance):
    """The 3-D tier, which is the implementation the probes were written from."""

    contract = ThreeDContract()
