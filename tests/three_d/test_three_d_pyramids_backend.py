"""The ``pyramids.plot_backends`` entry point on the PyVista tier.

The 3-D tier serves a still ``plot(backend="3d")`` and deliberately refuses an ``animate`` call —
its ``record`` needs a caller-supplied ``update`` callable and its ``orbit`` sweeps the camera over
a static scene, so neither can be driven from a plot call's forwarded keywords. The refusal is
pinned engine-free in ``tests/test_pyramids_backend.py``; what needs an engine, and is pinned here,
is that the still render really does come back as a scene. Runs in the ``viz3d`` pixi env.
"""

import pytest

from digitalearth.pyramids_backend import three_d

pytest.importorskip("pyvista")


class TestStillPlotThroughTheSeam:
    """A still ``plot(backend="3d")`` renders through ``quickmap`` as before."""

    def test_a_still_plot_returns_a_scene(self):
        """The facade's 0-based first band renders, and the tier's own object comes back.

        Test scenario:
            ``band=0`` is what ``Dataset.plot`` forwards for the first band. Before the shift it
            reached the reader as index ``-1`` and raised ``band -1 is out of range for a 1-band
            dataset`` — the defect reported in #501, here on the tier it was first reported against.
        """
        from pyramids.dataset import Dataset

        from digitalearth.three_d import Scene3D

        ds = Dataset.read_file("examples/data/acc4000.tif")
        out = three_d(ds, backend="3d", band=0)
        assert isinstance(out, Scene3D), f"expected a Scene3D, got {type(out)}"

    def test_the_refusal_names_verbs_this_tier_actually_has(self):
        """The two verbs the refusal sends a caller to exist on the real ``Scene3D``.

        Test scenario:
            The refusal is pinned engine-free elsewhere; what needs the engine, and is checked here,
            is that ``record`` and ``orbit`` are real methods. A refusal that points at verbs this
            tier does not have would be a dead end that no engine-free test could catch — and the
            previous version of this test only repeated the dev-env case, since the refusal imports
            nothing (round-1 L7).
        """
        from digitalearth.three_d import Scene3D

        try:
            three_d("a-dataset", backend="3d", animate=True)
        except ValueError as exc:
            message = str(exc)
        for verb in ("record", "orbit"):
            assert verb in message, f"the refusal should name {verb}: {message}"
            assert callable(getattr(Scene3D, verb, None)), (
                f"the refusal names Scene3D.{verb}, which this tier must actually have"
            )
