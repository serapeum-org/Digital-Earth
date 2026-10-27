"""The ``pyramids.plot_backends`` entry point on the MapLibre tier.

The translation itself is covered engine-free in ``tests/test_pyramids_backend.py``. Here: that a
still ``plot(backend="web")`` still comes back as a ``WebMap``, and that an
``animate(backend="web")`` call builds the time slider and writes the file — this tier has no
in-memory animation, so a destination is the whole difference between an export and a refusal.
Runs in the ``web`` pixi env.
"""

import pytest

from digitalearth.pyramids_backend import web

pytest.importorskip("maplibre")


@pytest.fixture
def stack():
    """A three-step raster stack, the shape an ``animate`` facade hands the backend."""
    from pyramids.dataset.collection import DatasetCollection

    return DatasetCollection.from_files(["examples/data/acc4000.tif"] * 3)


class TestStillPlotThroughTheSeam:
    """A plain ``plot(backend="web")`` goes through ``quickmap`` as before."""

    def test_a_still_plot_returns_a_web_map(self):
        """The still path is unchanged by the adapter, and the 0-based band is honoured.

        Test scenario:
            ``band=0`` is what ``Dataset.plot`` forwards for the first band; before the shift it
            reached the reader as ``-1`` and was refused as out of range.
        """
        from pyramids.dataset import Dataset

        from digitalearth.web import WebMap

        ds = Dataset.read_file("examples/data/acc4000.tif")
        out = web(ds, backend="web", band=0, colorbar=True)
        assert isinstance(out, WebMap), f"expected a WebMap, got {type(out)}"


class TestAnimateThroughTheSeam:
    """The animation call, which this tier serves only as a file."""

    def test_the_time_slider_is_built_from_the_forwarded_stack(self, stack):
        """The adapter builds a temporal layer, which is what the export needs to exist at all.

        Args:
            stack: A three-step collection.

        Test scenario:
            Asserted on the built scene rather than the written file, because the export itself needs
            a headless browser that may not be present. If the slider is absent the export could only
            ever fail, so this is the part worth pinning here.
        """
        from digitalearth.web import WebMap

        scene = WebMap().timeslider(stack)
        assert isinstance(scene, WebMap), (
            f"timeslider should return the scene, got {type(scene)}"
        )

    def test_an_animation_without_a_destination_is_refused_before_any_scene_is_built(
        self, stack
    ):
        """No ``path`` is refused by name, not with a ``TypeError`` from deep in the export.

        Args:
            stack: A three-step collection.

        Test scenario:
            ``WebMap.save_animation`` takes the path positionally and the facade forwards none of its
            own, so without this refusal the call died on a missing positional argument after
            building the whole scene.
        """
        with pytest.raises(ValueError, match="needs a path"):
            web(stack, backend="web", animate=True)
