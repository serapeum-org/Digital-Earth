"""Tests for the D3.6 animation/delivery helpers (digitalearth.three_d.AnimationMixin).

Gated on the optional ``3d`` extra (pyvista + imageio). Covers orbit fly-throughs to GIF/MP4, the frame-by-frame
``animate`` driver, the movie-vs-gif writer dispatch, and the trame jupyter-backend switch.
"""

import inspect

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pytest.importorskip("imageio")

from digitalearth.base.sources import get_source
from digitalearth.three_d import Scene3D, animation
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
    monkeypatch.setattr(
        pv, "set_jupyter_backend", lambda b: captured.setdefault("backend", b)
    )
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
    assert (
        getattr(scene.plotter, "mwriter", None) is None or scene.plotter.mwriter.closed
    )
    scene.close()


def _record_pyvista_calls(monkeypatch, scene):
    """Record what orbit() hands to each pyvista call, rendering nothing.

    The frame writer is stubbed out too: with `orbit_on_path` replaced, no frame is ever appended, and closing
    an empty GIF writer is an error rather than the point of these tests.
    """
    seen = {}

    def fake_generate(**kwargs):
        seen["generate_orbital_path"] = kwargs
        return pv.PolyData()

    def fake_orbit(orbital_path, **kwargs):
        seen["orbit_on_path"] = kwargs

    monkeypatch.setattr(animation, "_open_writer", lambda *a, **k: None)
    monkeypatch.setattr(animation, "_finalize_frames", lambda *a, **k: None)
    monkeypatch.setattr(scene.plotter, "generate_orbital_path", fake_generate)
    monkeypatch.setattr(scene.plotter, "orbit_on_path", fake_orbit)
    return seen


def test_orbit_forwards_the_path_shape_to_the_generator(monkeypatch, tmp_path):
    """factor, shift and viewup reach generate_orbital_path, which is the call that shapes the orbit."""
    scene = _terrain_scene()
    seen = _record_pyvista_calls(monkeypatch, scene)
    scene.orbit(
        str(tmp_path / "spin.gif"),
        n_frames=6,
        factor=0.7,
        shift=2.4,
        viewup=(0.0, 0.0, 1.0),
    )
    generated = seen["generate_orbital_path"]
    assert generated["factor"] == 0.7, (
        f"factor must reach the generator, got {generated}"
    )
    assert generated["shift"] == 2.4, f"shift must reach the generator, got {generated}"
    assert generated["viewup"] == (0.0, 0.0, 1.0), (
        f"viewup must reach the generator, got {generated}"
    )
    assert generated["n_points"] == 6, (
        f"n_frames is the generator's n_points, got {generated}"
    )
    scene.close()


def test_orbit_path_defaults_match_pyvistas_own(monkeypatch, tmp_path):
    """Calling orbit() with no shaping arguments reproduces pyvista's defaults, so old clips are unchanged.

    The defaults are asserted against `generate_orbital_path`'s own signature rather than hard-coded, so this
    fails if pyvista changes them out from under us instead of silently drifting.
    """
    scene = _terrain_scene()
    seen = _record_pyvista_calls(monkeypatch, scene)
    scene.orbit(str(tmp_path / "spin.gif"), n_frames=6)
    generated = seen["generate_orbital_path"]
    upstream = inspect.signature(pv.Plotter.generate_orbital_path).parameters
    for name in ("factor", "shift", "viewup"):
        assert generated[name] == upstream[name].default, (
            f"orbit()'s {name} default ({generated[name]!r}) must match pyvista's "
            f"({upstream[name].default!r}) so existing fly-throughs render identically"
        )
    scene.close()


def test_orbit_gives_the_camera_the_paths_up_vector(monkeypatch, tmp_path):
    """viewup reaches both calls: a path built around one up vector and flown with another rolls the horizon."""
    scene = _terrain_scene()
    seen = _record_pyvista_calls(monkeypatch, scene)
    scene.orbit(str(tmp_path / "spin.gif"), n_frames=6, viewup=(0.0, 1.0, 0.0))
    assert seen["generate_orbital_path"]["viewup"] == (0.0, 1.0, 0.0), (
        "the path needs the up vector"
    )
    assert seen["orbit_on_path"]["viewup"] == (0.0, 1.0, 0.0), (
        "the camera needs the same up vector"
    )
    scene.close()


def test_orbit_kwargs_still_reach_orbit_on_path(monkeypatch, tmp_path):
    """Everything else still passes through to orbit_on_path, and does not leak into the generator.

    The two calls take different arguments — `step` is orbit_on_path's, `factor` is the generator's — so
    sending one bag to both is what made `factor` unreachable in the first place.
    """
    scene = _terrain_scene()
    seen = _record_pyvista_calls(monkeypatch, scene)
    scene.orbit(str(tmp_path / "spin.gif"), n_frames=6, step=0.25)
    assert seen["orbit_on_path"]["step"] == 0.25, (
        f"step belongs to orbit_on_path, got {seen['orbit_on_path']}"
    )
    assert "step" not in seen["generate_orbital_path"], (
        "step must not leak into the generator"
    )
    assert "factor" not in seen["orbit_on_path"], (
        "factor must not leak into orbit_on_path, which rejects it"
    )
    scene.close()


def test_orbit_writes_a_gif_with_a_shaped_path(tmp_path):
    """A shaped orbit renders for real, not just in the argument plumbing.

    The stubbed tests above prove the arguments arrive; this proves pyvista accepts them together and still
    writes frames, which is what the stubs cannot show.
    """
    scene = _terrain_scene()
    out = scene.orbit(
        str(tmp_path / "close.gif"),
        n_frames=6,
        factor=0.9,
        shift=1.6,
        viewup=(0.0, 0.0, 1.0),
    )
    assert (tmp_path / "close.gif").stat().st_size > 0, (
        "a shaped orbit must still write frames"
    )
    assert out.endswith("close.gif")
    scene.close()
