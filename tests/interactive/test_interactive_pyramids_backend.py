"""The ``pyramids.plot_backends`` entry point actually animating on the HoloViz tier.

``tests/test_pyramids_backend.py`` covers the translation between pyramids' vocabulary and this
package's without an engine. This file covers the half that needs one: that
``Dataset.animate(backend="interactive")`` — which forwards ``animate=True`` and a frame
``interval`` — really builds a time cube and hands back a player, rather than a still map or a
``TypeError`` from a keyword the tier does not take. Runs in the ``interactive`` pixi env.
"""

import pytest

from digitalearth.pyramids_backend import interactive

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")
pn = pytest.importorskip("panel")


@pytest.fixture
def stack():
    """A three-step raster stack, the shape an ``animate`` facade hands the backend."""
    from pyramids.dataset.collection import DatasetCollection

    return DatasetCollection.from_files(["examples/data/acc4000.tif"] * 3)


class TestAnimateThroughTheSeam:
    """What the facade's animation call produces on this tier."""

    def test_an_animation_call_returns_a_player_over_every_step(self, stack):
        """``animate=True`` builds the time cube and returns a Panel player spanning the steps.

        Args:
            stack: A three-step collection.

        Test scenario:
            This is the path a bare ``quickmap`` call could not serve: ``quickmap`` draws one still
            field and has no ``animate`` keyword, so the forwarded ``animate=True`` reached cleopatra
            and was refused by listing *its* accepted keywords. The adapter has to build the cube and
            drive the tier's own verb instead.
        """
        app = interactive(stack, backend="interactive", animate=True, interval=250)
        assert isinstance(app, pn.viewable.Viewable), (
            f"expected a Panel layout, got {type(app)}"
        )
        players = app.select(pn.widgets.DiscretePlayer)
        assert players, "an animation must come back with a player widget"
        assert len(players[0].options) == 3, (
            f"the player should span the stack's three steps, got {len(players[0].options)}"
        )

    def test_the_forwarded_interval_sets_the_playback_rate(self, stack):
        """A 250 ms frame interval plays at four frames a second.

        Args:
            stack: A three-step collection.

        Test scenario:
            The facades describe speed as milliseconds per frame and every digitalearth verb takes a
            rate, so forwarding the number unchanged would play a 250 ms animation at 250 fps. The
            player's own interval is read back here, rather than trusting the conversion in isolation.
        """
        app = interactive(stack, backend="interactive", animate=True, interval=250)
        player = app.select(pn.widgets.DiscretePlayer)[0]
        assert player.interval == pytest.approx(250, rel=0.01), (
            f"the player should advance every 250 ms, got {player.interval}"
        )

    def test_a_destination_writes_the_animation_instead_of_returning_a_player(
        self, stack, tmp_path
    ):
        """With a ``path``, the animation is exported rather than handed back as a widget.

        Args:
            stack: A three-step collection.
            tmp_path: pytest's temporary directory.

        Test scenario:
            ``path`` is one of the "further keywords" the contract says are forwarded verbatim, and it
            is the only way a caller asks for a file rather than a live player.
        """
        out = tmp_path / "flood.gif"
        interactive(
            stack, backend="interactive", animate=True, path=str(out), interval=200
        )
        assert out.exists(), f"no animation written to {out}"
        assert out.stat().st_size > 1000, (
            f"the written animation is implausibly small: {out.stat().st_size} bytes"
        )

    def test_the_facades_first_band_animates_rather_than_failing(self, stack):
        """A forwarded 0-based ``band=0`` animates, where unshifted it read as band ``-1``.

        Args:
            stack: A three-step collection.

        Test scenario:
            The defect reported in #501, on the animation path: the facade's first band is ``0`` and
            this package's is ``1``, so the index had to be shifted or every single-band stack was
            refused as out of range.
        """
        app = interactive(stack, backend="interactive", animate=True, band=0)
        assert isinstance(app, pn.viewable.Viewable), (
            f"band=0 should animate, got {type(app)}"
        )


class TestStillPlotThroughTheSeam:
    """A plain ``plot(backend="interactive")`` still goes through ``quickmap``."""

    def test_a_still_plot_returns_an_interactive_map(self):
        """The still path is unchanged by the adapter — it delegates to ``quickmap``.

        Test scenario:
            The adapter must not have turned the ordinary still render into something else while
            adding the animation path; this is the call pyramids has always been able to make.
        """
        from pyramids.dataset import Dataset

        from digitalearth.interactive import InteractiveMap

        ds = Dataset.read_file("examples/data/acc4000.tif")
        out = interactive(ds, backend="interactive", band=0, title="DEM")
        assert isinstance(out, InteractiveMap), (
            f"expected an InteractiveMap, got {type(out)}"
        )


class TestWhatTheAdapterDrivesOnThisTier:
    """The tier verbs the adapter calls, stubbed so the wiring is asserted rather than the export."""

    def test_a_forwarded_basemap_is_applied_through_the_tiers_own_verb(
        self, stack, monkeypatch
    ):
        """``basemap=True`` reaches ``InteractiveMap.basemap``, not the constructor.

        Args:
            stack: A three-step collection.
            monkeypatch: pytest's patcher.

        Test scenario:
            This tier's constructor takes only the frame (``crs``/``title``); a basemap is a verb on
            the built scene, exactly as ``_quickmap_interactive`` applies it. Passing it to the
            constructor would raise ``TypeError``, so the route matters.
        """
        from digitalearth.interactive import InteractiveMap

        asked = []
        monkeypatch.setattr(
            InteractiveMap,
            "basemap",
            lambda self, *a, **k: asked.append((a, k)) or self,
        )
        interactive(stack, backend="interactive", animate=True, basemap=True)
        assert asked == [((), {})], (
            f"the default basemap should be asked for once, got {asked}"
        )

    def test_a_named_basemap_is_passed_on_rather_than_defaulted(
        self, stack, monkeypatch
    ):
        """A basemap *source* is forwarded to the verb, instead of being read as "the default".

        Args:
            stack: A three-step collection.
            monkeypatch: pytest's patcher.

        Test scenario:
            ``basemap`` is ``bool | str | Any`` upstream, so ``True`` and a provider name travel the
            same keyword and must not be conflated.
        """
        from digitalearth.interactive import InteractiveMap

        asked = []
        monkeypatch.setattr(
            InteractiveMap,
            "basemap",
            lambda self, *a, **k: asked.append((a, k)) or self,
        )
        interactive(
            stack, backend="interactive", animate=True, basemap="CartoDB.Positron"
        )
        assert asked == [(("CartoDB.Positron",), {})], (
            f"the source should be passed on, got {asked}"
        )

    def test_an_export_without_an_interval_leaves_the_rate_to_the_tier(
        self, stack, monkeypatch
    ):
        """With no ``interval``, ``save_animation`` is called without ``fps``.

        Args:
            stack: A three-step collection.
            monkeypatch: pytest's patcher.

        Test scenario:
            The tier's default rate is shared across backends (``DEFAULT_FPS``), so passing a
            computed rate when the caller named none would silently override that one default with
            this module's own guess.
        """
        from digitalearth.interactive import InteractiveMap

        calls = []
        monkeypatch.setattr(
            InteractiveMap,
            "save_animation",
            lambda self, path, **k: calls.append((path, k)) or path,
        )
        interactive(stack, backend="interactive", animate=True, path="out.gif")
        assert calls == [("out.gif", {})], f"fps should not be invented, got {calls}"
