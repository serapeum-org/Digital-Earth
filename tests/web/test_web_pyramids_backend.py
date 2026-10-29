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

    def test_the_time_slider_is_built_through_the_adapter(self, stack, monkeypatch):
        """The adapter really builds this tier's temporal layer, with the data it was handed.

        Args:
            stack: A three-step collection.
            monkeypatch: pytest's patcher.

        Test scenario:
            Previously this called ``WebMap().timeslider(stack)`` directly, which exercised the tier
            and not the adapter — so it could not have caught the adapter handing ``timeslider`` a
            keyword it has no parameter for (round-1 M6). The export is stubbed because it needs a
            headless browser, but the slider is built for real.
        """
        from digitalearth.web import WebMap

        built = []
        real = WebMap.timeslider

        def spy(self, features, **kwargs):
            built.append(kwargs)
            return real(self, features, **kwargs)

        monkeypatch.setattr(WebMap, "timeslider", spy)
        monkeypatch.setattr(WebMap, "save_animation", lambda self, path, **k: path)
        web(stack, backend="web", animate=True, path="out.gif")
        assert len(built) == 1, (
            f"the slider should be built once, got {len(built)} times"
        )

    @pytest.mark.parametrize("asked", [True, False])
    def test_a_colorbar_on_an_animation_is_applied_not_refused(
        self, stack, asked, monkeypatch
    ):
        """``colorbar`` works on an animation, as it already did on a still plot.

        Args:
            stack: A three-step collection.
            asked: Whether a key was asked for.
            monkeypatch: pytest's patcher.

        Test scenario:
            ``colorbar`` is in every facade's forwarded set and the still web path honours it, but
            this tier's slider has no such parameter — so round 1 refused it, turning a ``TypeError``
            into a nicer ``ValueError`` without making the documented call work, and a test pinned
            that refusal as intended. The scene owns a ``colorbar()`` verb, so it is applied there
            (round-2 H2). A falsy value needs no call: this tier shows no key unless asked.
        """
        from digitalearth.web import WebMap

        keyed = []
        monkeypatch.setattr(WebMap, "save_animation", lambda self, path, **k: path)
        monkeypatch.setattr(
            WebMap, "colorbar", lambda self, *a, **k: keyed.append(True) or self
        )
        web(stack, backend="web", animate=True, path="out.gif", colorbar=asked)
        assert keyed == ([True] if asked else []), (
            f"colorbar={asked} should {'add' if asked else 'not add'} a key, got {keyed}"
        )

    def test_a_still_plots_title_reaches_the_scene_not_the_builder(self):
        """``title`` is set on the finished scene, where the raster builder would have refused it.

        Test scenario:
            ``quickmap(ds, backend="web", title="DEM")`` raises ``RasterMixin.field() got an
            unexpected keyword argument 'title'. Did you mean 'tiles'?`` — executed. Four of the six
            facades forward a title, so routing it through ``set_title`` is what makes them work
            (round-1 H2).
        """
        from pyramids.dataset import Dataset

        from digitalearth.web import WebMap

        ds = Dataset.read_file("examples/data/acc4000.tif")
        scene = web(ds, backend="web", title="DEM")
        assert isinstance(scene, WebMap), f"expected a WebMap, got {type(scene)}"

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


class TestConstructionOptionsOnTheRealScene:
    """The web scene's construction-time options, reached through the plugin."""

    def test_zoom_and_height_change_the_built_scene(self):
        """`zoom` and `height` reach the constructor and change the scene they build.

        Test scenario:
            Both are construction-only on this tier — there is no setter for either on the finished
            `WebMap` — so a caller going through `plot(backend="web")` could not set them at all.
            Asserted against a default-built scene so the change is visible rather than assumed.
        """
        from pyramids.dataset import Dataset

        ds = Dataset.read_file("examples/data/acc4000.tif")
        plain = web(ds, backend="web", band=0)
        tuned = web(ds, backend="web", band=0, zoom=6, height=700)
        assert (plain.zoom, plain.height) != (tuned.zoom, tuned.height), (
            "the options must change the scene, not merely be accepted"
        )
        assert (tuned.zoom, tuned.height) == (6, 700), (
            f"expected zoom=6 height=700, got {tuned.zoom} {tuned.height}"
        )

    def test_an_unknown_keyword_is_still_refused_by_name(self):
        """A keyword neither the constructor nor the builder declares is still named.

        Test scenario:
            Widening what the constructor may take must not turn the seam into a sink that accepts
            anything — that is the silent-drop failure this module exists to prevent.
        """
        from pyramids.dataset import Dataset

        ds = Dataset.read_file("examples/data/acc4000.tif")
        with pytest.raises(
            Exception, match="(?i)unexpected keyword|not correct|invalid"
        ):
            web(ds, backend="web", band=0, definitely_not_a_real_option=1)
