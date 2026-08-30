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
        assert calls["save"][0][1]["crf"] == 20 and calls["save"][0][1]["codec"] == "libx264"


class TestDerivedGif:
    """With gif=, the frames are drawn once and read back off the written video."""

    def test_returns_both_paths(self, anim, calls):
        assert save_animation(anim, "clip.mp4", gif="clip.gif") == ("clip.mp4", "clip.gif")

    def test_derives_the_gif_from_the_written_video(self, anim, calls):
        save_animation(anim, "clip.mp4", gif="clip.gif")
        src, out, _ = calls["gif"][0]
        assert src == "clip.mp4" and out == "clip.gif"

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
        assert calls["gif"][0][2]["fps"] == 4 and calls["gif"][0][2]["max_colors"] == 64

    def test_deriving_a_gif_from_a_gif_is_refused(self, anim, calls):
        with pytest.raises(ValueError, match="already a GIF"):
            save_animation(anim, "clip.gif", gif="other.gif")

    def test_a_non_gif_derived_path_is_refused(self, anim, calls):
        with pytest.raises(ValueError, match="must end in"):
            save_animation(anim, "clip.mp4", gif="clip.webm")


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
    assert out.exists() and out.stat().st_size > 0
