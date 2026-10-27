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

    def test_an_animation_is_refused_with_the_engine_installed_too(self):
        """The animation refusal holds in the real environment, not only without the engine.

        Test scenario:
            A by-name refusal that only fires because an import failed would be an accident rather
            than a decision. With PyVista present the refusal must still be the ``ValueError`` that
            names the two verbs to use directly.
        """
        with pytest.raises(ValueError, match="Scene3D.record"):
            three_d("a-dataset", backend="3d", animate=True)
