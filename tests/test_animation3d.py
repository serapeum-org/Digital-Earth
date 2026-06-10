"""Tests for the D3.6 animation/delivery helpers (digitalearth.three_d.AnimationMixin).

Gated on the optional ``3d`` extra (pyvista + imageio). Covers orbit fly-throughs to GIF/MP4, the frame-by-frame
``animate`` driver, the movie-vs-gif writer dispatch, and the trame jupyter-backend switch.
"""
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pytest.importorskip("imageio")

from digitalearth.sources import get_source
from digitalearth.three_d import Scene3D
from digitalearth.three_d.animation import _MOVIE_SUFFIXES, _open_writer


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def _terrain_scene():
    """A small terrain scene to animate."""
    dem = np.add.outer(np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10))
    scene = Scene3D(off_screen=True)
    scene.terrain(get_source(dem), z_exaggeration=3.0)
    return scene


def test_orbit_writes_a_gif(tmp_path):
    """orbit() sweeps the camera and writes a non-empty GIF."""
    scene = _terrain_scene()
    out = scene.orbit(str(tmp_path / "spin.gif"), n_frames=6)
    assert (tmp_path / "spin.gif").stat().st_size > 0
    assert out.endswith("spin.gif")
    scene.close()


def test_orbit_writes_an_mp4(tmp_path):
    """A video suffix routes orbit() through open_movie and writes a non-empty MP4."""
    scene = _terrain_scene()
    scene.orbit(str(tmp_path / "spin.mp4"), n_frames=6, framerate=10)
    assert (tmp_path / "spin.mp4").stat().st_size > 0
    scene.close()


def test_animate_drives_frames_via_callback(tmp_path):
    """animate() calls the update callback once per frame and writes a non-empty GIF."""
    scene = _terrain_scene()
    calls = []

    def grow(s, factor):
        calls.append(factor)
        s.layers[0][0].points[:, 2] *= factor

    scene.animate([1.05, 1.05, 1.05], str(tmp_path / "grow.gif"), grow)
    assert calls == [1.05, 1.05, 1.05]
    assert (tmp_path / "grow.gif").stat().st_size > 0
    scene.close()


def test_writer_dispatch_movie_vs_gif():
    """_open_writer routes video suffixes to open_movie and everything else to open_gif."""
    assert ".mp4" in _MOVIE_SUFFIXES

    class _Spy:
        mwriter = object()  # satisfy the post-open writer guard added in _open_writer

        def __init__(self):
            self.opened = None

        def open_movie(self, path, framerate):
            self.opened = ("movie", path)

        def open_gif(self, path, fps):
            self.opened = ("gif", path)

    movie, gif = _Spy(), _Spy()
    _open_writer(movie, "x.mp4", 10)
    _open_writer(gif, "x.gif", 10)
    assert movie.opened[0] == "movie"
    assert gif.opened[0] == "gif"


def test_jupyter_switches_backend(monkeypatch):
    """jupyter() calls pyvista.set_jupyter_backend with the requested backend."""
    captured = {}
    monkeypatch.setattr(pv, "set_jupyter_backend", lambda b: captured.setdefault("backend", b))
    scene = Scene3D(off_screen=True)
    scene.jupyter("static")
    assert captured["backend"] == "static"
    scene.close()


def test_finalize_frames_noop_without_writer():
    """_finalize_frames is a safe no-op when no GIF/MP4 writer was opened (defensive path)."""
    from digitalearth.three_d.animation import _finalize_frames

    scene = Scene3D(off_screen=True)  # never opened a writer -> plotter has no mwriter
    _finalize_frames(scene.plotter)  # must not raise
    scene.close()


def test_animate_finalizes_writer_even_when_update_raises(tmp_path):
    """If the update callback raises mid-animation, the frame writer is still finalized (try/finally)."""
    scene = _terrain_scene()
    out = tmp_path / "boom.gif"

    def boom(s, frame):
        s.plotter.write_frame()
        raise RuntimeError("frame blew up")

    with pytest.raises(RuntimeError, match="blew up"):
        scene.animate([1, 2], str(out), boom)
    # finally-block ran: the writer was flushed/closed (no lingering open mwriter)
    assert getattr(scene.plotter, "mwriter", None) is None or scene.plotter.mwriter.closed
    scene.close()
