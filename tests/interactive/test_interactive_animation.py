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


@pytest.fixture()
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

    def test_play_without_timecube_raises(self):
        with pytest.raises(ValueError, match="no time cube"):
            InteractiveMap().play()


class TestSaveAnimation:
    """``save_animation`` — GIF / scrubber export."""

    def test_gif_export(self, cube_map, tmp_path):
        out = tmp_path / "anim.gif"
        assert cube_map.save_animation(str(out), fps=4) == str(out)
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
        with pytest.raises(ValueError, match="no time cube"):
            InteractiveMap().save_animation(str(tmp_path / "x.gif"))
