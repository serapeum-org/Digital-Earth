"""``to_backend(..., backend="3d")`` — the dispatcher routes to ``Scene3D.from_figure`` (U-6, order 33).

Runs in the 3-D environment. ``Scene3D.from_figure``'s own round trip is covered by ``test_seam3d``; this
pins that :func:`digitalearth.api.to_backend` reaches it, so the public entry point drives the 3-D tier.
"""

import pytest

pytest.importorskip("pyvista")  # the 3-D tier engine; skip in envs without it

from digitalearth import api  # noqa: E402
from digitalearth.three_d import Scene3D  # noqa: E402

DEM_PATH = "examples/data/acc4000.tif"


class TestThreeDDispatch:
    """The public dispatcher builds a 3-D scene from a figure."""

    def test_to_backend_builds_a_scene3d_with_the_figure_s_layers(self):
        """``to_backend(figure, "3d")`` returns a ``Scene3D`` carrying the described layers.

        Test scenario:
            A terrain scene describes itself; ``to_backend`` rebuilds it through the 3-D tier's
            ``from_figure`` (already round-trip-tested in ``test_seam3d``), so the dispatch — not the replay —
            is what this pins.
        """
        source = Scene3D(off_screen=True)
        try:
            source.terrain(DEM_PATH)
            figure = source.figure_spec
            layer_ids = source.layer_ids
        finally:
            source.close()

        replayed = api.to_backend(figure, backend="3d", off_screen=True)
        try:
            assert isinstance(replayed, Scene3D)
            assert replayed.layer_ids == layer_ids, (
                f"replayed {replayed.layer_ids} from {layer_ids}"
            )
        finally:
            replayed.close()
