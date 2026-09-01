"""Tests for digitalearth.animation.save_animation — render once, encode twice.

The encoders themselves are cleopatra's (and ffmpeg's); what Digital-Earth owns is the wiring: which path is
written, at what frame rate, and — the substantive decision — that the intermediate video is written at full
chroma when a GIF is going to be derived from it, since a subsampled source permanently caps the GIF's colour.
"""
from typing import Any, Dict, List

import numpy as np
import pytest
from matplotlib.animation import FuncAnimation

from digitalearth import animation as de_animation
from digitalearth.animation import FULL_CHROMA_PIX_FMT, save_animation
from digitalearth.scene import Map


@pytest.fixture
def calls(monkeypatch) -> Dict[str, List[Any]]:
    """Record the cleopatra calls instead of invoking ffmpeg."""
    recorded: Dict[str, List[Any]] = {"save": [], "gif": []}
    monkeypatch.setattr(de_animation, "_cleopatra_save_animation",
                        lambda anim, path, **kw: recorded["save"].append((path, kw)))
    monkeypatch.setattr(de_animation, "gif_from_video",
                        lambda src, path, **kw: recorded["gif"].append((src, path, kw)))
    return recorded


@pytest.fixture
def anim() -> FuncAnimation:
    """A trivial two-frame animation; nothing here renders it."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    return FuncAnimation(fig, lambda i: ax.plot([i], [i]), frames=2, interval=100, blit=False)


class TestDirectSave:
    """Without gif=, this is a thin pass-through."""

    def test_writes_the_requested_path(self, anim, calls):
        assert save_animation(anim, "clip.mp4", fps=8) == "clip.mp4"
        assert calls["save"][0][0] == "clip.mp4"

    def test_passes_the_frame_rate_through(self, anim, calls):
        save_animation(anim, "clip.mp4", fps=8)
        assert calls["save"][0][1]["fps"] == 8

    def test_falls_back_to_the_default_rate(self, anim, calls):
        save_animation(anim, "clip.mp4")
        assert calls["save"][0][1]["fps"] == int(de_animation.DEFAULT_FPS)

    def test_does_not_force_a_pixel_format(self, anim, calls):
        """No GIF is being derived, so leave cleopatra's playback-friendly default alone."""
        save_animation(anim, "clip.mp4")
        assert "pix_fmt" not in calls["save"][0][1]

    def test_no_gif_is_derived(self, anim, calls):
        save_animation(anim, "clip.mp4")
        assert calls["gif"] == []

    def test_forwards_encoder_options(self, anim, calls):
        save_animation(anim, "clip.mp4", crf=20, codec="libx264")
        assert calls["save"][0][1]["crf"] == 20, "crf should be forwarded"
        assert calls["save"][0][1]["codec"] == "libx264", "codec should be forwarded"


class TestDerivedGif:
    """With gif=, the frames are drawn once and read back off the written video."""

    def test_returns_both_paths(self, anim, calls):
        assert save_animation(anim, "clip.mp4", gif="clip.gif") == ("clip.mp4", "clip.gif")

    def test_derives_the_gif_from_the_written_video(self, anim, calls):
        save_animation(anim, "clip.mp4", gif="clip.gif")
        src, out, _ = calls["gif"][0]
        assert src == "clip.mp4", f"the GIF should be derived from the video, got {src}"
        assert out == "clip.gif", f"the GIF should be written to clip.gif, got {out}"

    def test_the_animation_is_rendered_only_once(self, anim, calls):
        save_animation(anim, "clip.mp4", gif="clip.gif")
        assert len(calls["save"]) == 1

    def test_the_intermediate_is_written_at_full_chroma(self, anim, calls):
        """The point of the wrapper: a yuv420p source would cap the GIF's colour before quantisation."""
        save_animation(anim, "clip.mp4", gif="clip.gif")
        assert calls["save"][0][1]["pix_fmt"] == FULL_CHROMA_PIX_FMT

    def test_an_explicit_pixel_format_still_wins(self, anim, calls):
        save_animation(anim, "clip.mp4", gif="clip.gif", pix_fmt="yuv420p")
        assert calls["save"][0][1]["pix_fmt"] == "yuv420p"

    def test_the_gif_inherits_the_video_frame_rate(self, anim, calls):
        save_animation(anim, "clip.mp4", fps=9, gif="clip.gif")
        assert calls["gif"][0][2]["fps"] == 9.0

    def test_gif_options_override_the_inherited_rate(self, anim, calls):
        save_animation(anim, "clip.mp4", fps=9, gif="clip.gif", gif_options={"fps": 4, "max_colors": 64})
        assert calls["gif"][0][2]["fps"] == 4, "gif_options should override the inherited rate"
        assert calls["gif"][0][2]["max_colors"] == 64, "gif_options should be forwarded"

    @pytest.mark.parametrize("fps", [2.5, 7.4, 0.6, 12])
    def test_the_video_and_the_gif_share_one_frame_rate(self, anim, calls, fps):
        """Rounding only the video left the two files of one animation playing at different speeds."""
        save_animation(anim, "clip.mp4", fps=fps, gif="clip.gif")
        video_fps, gif_fps = calls["save"][0][1]["fps"], calls["gif"][0][2]["fps"]
        assert video_fps == gif_fps, f"video is {video_fps} fps but the GIF is {gif_fps} fps"

    def test_deriving_a_gif_from_a_gif_is_refused(self, anim, calls):
        with pytest.raises(ValueError, match="already a GIF"):
            save_animation(anim, "clip.gif", gif="other.gif")

    @pytest.mark.parametrize("path", ["clip.png", "clip", "clip.txt"])
    def test_a_non_video_source_path_is_refused(self, anim, calls, path):
        """Only a .gif path was rejected, so any other non-video reached ffmpeg and failed there."""
        with pytest.raises(ValueError, match="must be one of"):
            save_animation(anim, path, gif="clip.gif")

    @pytest.mark.parametrize("path", ["clip.webp", "clip.gif"])
    def test_a_pillow_written_source_is_refused(self, anim, calls, path):
        """Pillow ignores pix_fmt, so the full-chroma intermediate cannot be delivered for these."""
        with pytest.raises(ValueError, match="already a GIF|must be one of"):
            save_animation(anim, path, gif="clip.gif")

    def test_a_differing_gif_options_rate_warns_but_is_honoured(self, anim, calls):
        """gif_from_video types fps as a float, so a fractional rate is meaningful and must not be coerced —
        but two files of one animation playing at different speeds is worth saying out loud."""
        with pytest.warns(RuntimeWarning, match="different speeds"):
            save_animation(anim, "clip.mp4", fps=12, gif="clip.gif", gif_options={"fps": 4.5})
        assert calls["gif"][0][2]["fps"] == 4.5, "the caller's rate should reach gif_from_video unchanged"

    def test_a_matching_gif_options_rate_is_silent(self, anim, calls):
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            save_animation(anim, "clip.mp4", fps=12, gif="clip.gif", gif_options={"fps": 12})
        assert not [w for w in caught if "different speeds" in str(w.message)]

    def test_a_non_gif_derived_path_is_refused(self, anim, calls):
        with pytest.raises(ValueError, match="must end in"):
            save_animation(anim, "clip.mp4", gif="clip.webm")


class TestFrameRate:
    """The rate is rounded once, and a rate no encoder can use is refused rather than silently zeroed."""

    @pytest.mark.parametrize(
        "fps, expected",
        [(2.5, 2), (2.4, 2), (7.6, 8), (0.6, 1), (12, 12)],
    )
    def test_fractional_rates_round_to_a_whole_number(self, anim, calls, fps, expected):
        save_animation(anim, "clip.mp4", fps=fps)
        assert calls["save"][0][1]["fps"] == expected

    @pytest.mark.parametrize("fps", [0.5, 0.4, 0.0, -3])
    def test_a_rate_below_one_is_refused(self, anim, calls, fps):
        """fps=0.5 used to round to 0, which no encoder accepts — it must fail loudly, not silently."""
        with pytest.raises(ValueError, match="frames per second"):
            save_animation(anim, "clip.mp4", fps=fps)

    @pytest.mark.parametrize("fps", [float("nan"), float("inf")])
    def test_a_non_finite_rate_is_refused(self, anim, calls, fps):
        with pytest.raises(ValueError, match="finite"):
            save_animation(anim, "clip.mp4", fps=fps)

    def test_a_non_numeric_rate_is_refused(self, anim, calls):
        with pytest.raises(ValueError, match="must be a number"):
            save_animation(anim, "clip.mp4", fps="fast")


class TestMapIntegration:
    """Map.animate / Map.rotate feed the saver."""

    def test_saving_without_an_animation_is_refused(self, dataset):
        m = Map(crs=dataset.epsg)
        with pytest.raises(RuntimeError, match="no animation"):
            m.save_animation("clip.mp4")

    def test_the_scenes_own_frame_rate_is_used(self, dataset, calls):
        m = Map(crs=dataset.epsg)
        m.animate([dataset], fps=5.0)
        m.save_animation("clip.mp4")
        assert calls["save"][0][1]["fps"] == 5

    def test_an_explicit_rate_overrides_the_scenes(self, dataset, calls):
        m = Map(crs=dataset.epsg)
        m.animate([dataset], fps=5.0)
        m.save_animation("clip.mp4", fps=20)
        assert calls["save"][0][1]["fps"] == 20

    def test_rotate_records_its_rate_too(self, dataset, calls):
        m = Map(crs=dataset.epsg, globe=True)
        m.rotate(dataset, n_frames=2, fps=6.0)
        m.save_animation("clip.mp4")
        assert calls["save"][0][1]["fps"] == 6

    def test_map_can_derive_a_gif(self, dataset, calls):
        m = Map(crs=dataset.epsg)
        m.animate([dataset], fps=4.0)
        assert m.save_animation("clip.mp4", gif="clip.gif") == ("clip.mp4", "clip.gif")


def test_a_real_gif_is_written_end_to_end(tmp_path):
    """One unmocked pass through the direct path — Pillow only, no ffmpeg needed."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_axis_off()
    frames = [np.zeros((4, 4)), np.ones((4, 4))]
    anim = FuncAnimation(fig, lambda i: ax.imshow(frames[i]), frames=2, interval=200, blit=False)
    out = tmp_path / "clip.gif"
    assert save_animation(anim, str(out), fps=2) == str(out)
    assert out.exists(), f"{out} was not written"
    assert out.stat().st_size > 0, f"{out} is empty"
