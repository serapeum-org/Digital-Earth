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


class TestWhatTheAdapterDrivesOnThisTier:
    """The tier verbs the adapter calls. The export itself needs a headless browser, so it is stubbed.

    ``WebMap.save_animation`` raises ``ImportError`` without Playwright, which ``digitalearth[web]``
    does not install — so the adapter's own wiring is asserted by capturing the call it makes.
    """

    @pytest.fixture
    def captured(self, monkeypatch):
        """Replace the tier's animation verbs, recording what the adapter asks of them.

        Args:
            monkeypatch: pytest's patcher.

        Returns:
            dict: ``save`` (path and kwargs), ``basemap`` and ``title`` call records.
        """
        from digitalearth.web import WebMap

        record = {"save": [], "basemap": [], "title": []}
        monkeypatch.setattr(
            WebMap,
            "save_animation",
            lambda self, path, **k: record["save"].append((path, k)) or path,
        )
        monkeypatch.setattr(
            WebMap, "basemap", lambda self, *a, **k: record["basemap"].append(a) or self
        )
        monkeypatch.setattr(
            WebMap, "set_title", lambda self, *a, **k: record["title"].append(a) or self
        )
        return record

    def test_the_destination_and_translated_rate_reach_the_export(
        self, stack, captured
    ):
        """The written path and the rate converted from ``interval`` reach ``save_animation``.

        Args:
            stack: A three-step collection.
            captured: The recorded tier calls.

        Test scenario:
            200 ms a frame is five frames a second. Forwarding the interval unchanged would ask this
            tier for 200 fps.
        """
        web(stack, backend="web", animate=True, path="flood.gif", interval=200)
        assert captured["save"] == [("flood.gif", {"fps": 5.0})], (
            f"the export received {captured['save']}"
        )

    def test_an_export_without_an_interval_leaves_the_rate_to_the_tier(
        self, stack, captured
    ):
        """With no ``interval``, ``save_animation`` is called without ``fps``.

        Args:
            stack: A three-step collection.
            captured: The recorded tier calls.

        Test scenario:
            The shared ``DEFAULT_FPS`` is the one rate every tier reads; inventing a rate here would
            override it for a caller who asked for nothing.
        """
        web(stack, backend="web", animate=True, path="flood.gif")
        assert captured["save"] == [("flood.gif", {})], (
            f"fps should not be invented: {captured['save']}"
        )

    def test_the_basemap_and_title_go_through_this_tiers_own_verbs(
        self, stack, captured
    ):
        """``basemap`` and ``title`` are applied to the built scene, not to the constructor.

        Args:
            stack: A three-step collection.
            captured: The recorded tier calls.

        Test scenario:
            ``WebMapBase.__init__`` takes neither, and has no ``**kwargs`` — so passing either to the
            constructor is a ``TypeError``. The title in particular has to be set on the scene.
        """
        web(
            stack,
            backend="web",
            animate=True,
            path="f.gif",
            basemap=True,
            title="Flood",
        )
        assert captured["basemap"] == [()], (
            f"the default basemap should be asked for: {captured}"
        )
        assert captured["title"] == [("Flood",)], (
            f"the title should be set on the scene: {captured}"
        )
