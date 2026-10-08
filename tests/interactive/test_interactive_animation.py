"""DI.11 — animation export (Player playback + GIF / scrubber export of a time cube).

Builds a 3-step timecube, then asserts ``play`` is a Panel layout with a Player and ``save_animation``
writes a non-trivial GIF (matplotlib backend) and a self-contained scrubber HTML. MP4 is skipped when
ffmpeg is absent (logged via the skip reason). Runs in the ``interactive`` pixi env.
"""

import shutil

import pytest

from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")
pn = pytest.importorskip("panel")


@pytest.fixture
def cube_map() -> InteractiveMap:
    """A map carrying a 3-step timecube."""
    from pyramids.dataset.collection import DatasetCollection

    dc = DatasetCollection.from_files(["examples/data/acc4000.tif"] * 3)
    return InteractiveMap().timecube(dc)


class TestPlay:
    """``play`` — Panel Player bound to the time kdim."""

    def test_play_is_panel_with_player(self, cube_map):
        app = cube_map.play(fps=5)
        assert isinstance(app, pn.viewable.Viewable), f"not a Panel layout: {type(app)}"
        players = app.select(pn.widgets.DiscretePlayer)
        assert players, "play() must include a DiscretePlayer widget"
        assert len(players[0].options) == 3, "the Player must span the three frames"

    def test_play_a_named_temporal_layer(self):
        """IN-12 — with two time cubes, ``play(layer=...)`` animates the chosen one, not just the first."""
        from pyramids.dataset.collection import DatasetCollection

        dc = DatasetCollection.from_files(["examples/data/acc4000.tif"] * 3)
        m = InteractiveMap().timecube(dc, name="rain").timecube(dc, name="temp")
        app = m.play(layer="temp")
        assert isinstance(app, pn.layout.Column)

    def test_play_an_unknown_layer_lists_the_animatable_ones(self):
        """A layer id that is not an animatable temporal layer is refused, naming the ones that are."""
        from pyramids.dataset.collection import DatasetCollection

        dc = DatasetCollection.from_files(["examples/data/acc4000.tif"] * 3)
        m = InteractiveMap().timecube(dc, name="rain")
        with pytest.raises(ValueError, match="not an animatable temporal layer"):
            m.play(layer="nope")

    def test_play_without_timecube_raises(self):
        interactiveMap = InteractiveMap()
        with pytest.raises(ValueError, match="no time cube"):
            interactiveMap.play()


class TestSaveAnimation:
    """``save_animation`` — GIF / scrubber export."""

    def test_gif_export(self, cube_map, tmp_path):
        out = tmp_path / "anim.gif"
        assert cube_map.save_animation(str(out), fps=4) == out
        assert out.stat().st_size > 1_000, "the GIF should be a non-trivial file"

    def test_scrubber_html_export(self, cube_map, tmp_path):
        out = tmp_path / "anim.html"
        cube_map.save_animation(str(out))
        assert out.stat().st_size > 1_000, "the scrubber HTML should be self-contained"

    @pytest.mark.skipif(
        shutil.which("ffmpeg") is None, reason="ffmpeg not installed (logged skip)"
    )
    def test_mp4_export_when_ffmpeg_present(self, cube_map, tmp_path):
        out = tmp_path / "anim.mp4"
        cube_map.save_animation(str(out), fps=4)
        assert out.stat().st_size > 1_000

    def test_save_without_timecube_raises(self, tmp_path):
        destination = str(tmp_path / "x.gif")
        interactiveMap = InteractiveMap()
        with pytest.raises(ValueError, match="no time cube"):
            interactiveMap.save_animation(destination)


class TestFrames:
    """``frames`` registers any element sequence as one animatable temporal layer (IN-12 #436)."""

    @staticmethod
    def _point_frames():
        """Return three frames of a moving point — a temporal VECTOR layer, not a raster cube.

        Returns:
            A ``{step: hv.Points}`` mapping.
        """
        return {step: hv.Points([(step, step)]) for step in range(3)}

    def test_frames_from_mapping_is_animatable(self):
        """A frames() mapping registers one animatable temporal layer keyed by its frames (IN-12)."""
        m = InteractiveMap().frames(self._point_frames(), dimension="t")
        animatable = m._animatable_layers()
        assert len(animatable) == 1, (
            f"frames() must register one animatable layer, got {animatable}"
        )

    def test_frames_from_sequence_enumerates(self):
        """A plain sequence of elements is enumerated 0, 1, 2 … and is animatable (IN-12)."""
        m = InteractiveMap().frames([hv.Points([(0, 0)]), hv.Points([(1, 1)])])
        assert len(m._animatable_layers()) == 1, (
            "an enumerated frame sequence must animate"
        )

    def test_frames_vector_layer_plays_end_to_end(self):
        """play() drives a frames() vector layer exactly like a time cube — the point of IN-12."""
        m = InteractiveMap().frames(self._point_frames(), dimension="t")
        app = m.play(fps=4)
        players = app.select(pn.widgets.DiscretePlayer)
        assert players and len(players[0].options) == 3, (
            "a frames() vector layer must play its three frames"
        )

    def test_frames_save_animation_scrubber(self, tmp_path):
        """A frames() layer exports to the client-side scrubber HTML (no server, no ffmpeg).

        Args:
            tmp_path: pytest temp dir.
        """
        out = tmp_path / "moving.html"
        result = InteractiveMap().frames(self._point_frames()).save_animation(str(out))
        assert result == out and out.exists(), f"scrubber HTML not written: {result}"

    def test_frames_empty_raises(self):
        """An empty frame set has nothing to animate and is refused (IN-12)."""
        with pytest.raises(ValueError, match="at least one frame"):
            InteractiveMap().frames({})
