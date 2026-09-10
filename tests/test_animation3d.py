"""Tests for the D3.6 animation/delivery helpers (digitalearth.three_d.AnimationMixin).

Gated on the optional ``3d`` extra (pyvista + imageio). Covers orbit fly-throughs to GIF/MP4, the frame-by-frame
``animate`` driver, the movie-vs-gif writer dispatch, and the trame jupyter-backend switch.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pytest.importorskip("imageio")

from digitalearth.base.sources import get_source
from digitalearth.three_d import Scene3D, animation
from digitalearth.three_d.animation import _finite_number, _open_writer, _up_vector


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

    The frame writer is stubbed out too. Not because closing an empty one fails — it does not, as
    `test_orbit_finalizes_writer_when_an_unknown_kwarg_raises` relies on — but because these tests are about
    which arguments reach which call, and opening a writer for that is pure cost.
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
    assert np.array_equal(generated["viewup"], (0.0, 0.0, 1.0)), (
        f"viewup must reach the generator, got {generated}"
    )
    assert generated["n_points"] == 6, (
        f"n_frames is the generator's n_points, got {generated}"
    )
    scene.close()


def test_orbit_path_defaults_are_the_documented_ones(monkeypatch, tmp_path):
    """Calling orbit() with no shaping arguments sends the documented defaults, so old clips are unchanged.

    The values are asserted as *our* contract, not against `inspect.signature(pv.Plotter...)`. They were
    chosen to equal pyvista's at the time of writing, but gating on upstream's would mean a pyvista release
    that changed a default turned an unrelated dependency bump into red CI — and the only way back to green,
    copying the new value, would change our users' fly-throughs. That is precisely what this test exists to
    prevent, so ours are pinned here and following upstream stays a deliberate decision.
    """
    scene = _terrain_scene()
    seen = _record_pyvista_calls(monkeypatch, scene)
    scene.orbit(str(tmp_path / "spin.gif"), n_frames=6)
    generated = seen["generate_orbital_path"]
    assert generated["factor"] == 3.0, (
        f"the documented factor default is 3.0, got {generated['factor']!r}"
    )
    assert generated["shift"] == 0.0, (
        f"the documented shift default is 0.0, got {generated['shift']!r}"
    )
    assert generated["viewup"] is None, (
        f"the documented viewup default is None, got {generated['viewup']!r}"
    )
    scene.close()


def test_orbit_gives_the_camera_the_paths_up_vector(monkeypatch, tmp_path):
    """viewup reaches both calls: a path built around one up vector and flown with another rolls the horizon."""
    scene = _terrain_scene()
    seen = _record_pyvista_calls(monkeypatch, scene)
    scene.orbit(str(tmp_path / "spin.gif"), n_frames=6, viewup=(0.0, 1.0, 0.0))
    assert np.array_equal(seen["generate_orbital_path"]["viewup"], (0.0, 1.0, 0.0)), (
        "the path needs the up vector"
    )
    assert np.array_equal(seen["orbit_on_path"]["viewup"], (0.0, 1.0, 0.0)), (
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


def _orbit_path_points(scene, monkeypatch, tmp_path, **shape):
    """Return the points of the path `orbit()` hands the camera for `shape`, captured from orbit_on_path.

    Going through `orbit()` is the whole point: measuring `generate_orbital_path` directly would assert
    pyvista's own geometry and stay green even with the forwarding deleted.
    """
    captured = {}

    def capture(orbital_path, **kwargs):
        captured["path"] = orbital_path

    monkeypatch.setattr(animation, "_open_writer", lambda *a, **k: None)
    monkeypatch.setattr(animation, "_finalize_frames", lambda *a, **k: None)
    monkeypatch.setattr(scene.plotter, "orbit_on_path", capture)
    scene.orbit(str(tmp_path / "measure.gif"), n_frames=8, **shape)
    return np.asarray(captured["path"].points)


def _radius_and_height(points):
    """Return the mean horizontal radius and mean height of a captured orbital path."""
    radius = float(
        np.hypot(
            points[:, 0] - points[:, 0].mean(), points[:, 1] - points[:, 1].mean()
        ).mean()
    )
    return radius, float(points[:, 2].mean())


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
        shift=8.0,
        viewup=(0.0, 0.0, 1.0),
    )
    assert (tmp_path / "close.gif").stat().st_size > 0, (
        "a shaped orbit must still write frames"
    )
    assert out.endswith("close.gif")
    scene.close()


def test_factor_and_shift_actually_move_the_orbit(monkeypatch, tmp_path):
    """factor scales the orbit's radius and shift raises it, measured on the path orbit() hands the camera.

    Asserting only that a shaped orbit writes a non-empty file would pass with the feature removed, since the
    unshaped orbit writes one too — and so would measuring `generate_orbital_path` directly. The path is
    captured from `orbit_on_path`, so deleting the forwarding fails this test.
    """
    scene = _terrain_scene()
    base_radius, base_height = _radius_and_height(
        _orbit_path_points(scene, monkeypatch, tmp_path)
    )
    closer_radius, _ = _radius_and_height(
        _orbit_path_points(scene, monkeypatch, tmp_path, factor=0.9)
    )
    wider_radius, _ = _radius_and_height(
        _orbit_path_points(scene, monkeypatch, tmp_path, factor=6.0)
    )
    _, lifted_height = _radius_and_height(
        _orbit_path_points(scene, monkeypatch, tmp_path, shift=8.0)
    )
    assert closer_radius < base_radius, (
        f"factor below the default must close the orbit in: {closer_radius} !< {base_radius}"
    )
    assert wider_radius > base_radius, (
        f"factor above the default must widen the orbit: {wider_radius} !> {base_radius}"
    )
    assert lifted_height == pytest.approx(base_height + 8.0), (
        f"shift must raise the orbit by its own value: {lifted_height} != {base_height} + 8.0"
    )
    scene.close()


def test_shift_moves_along_viewup_not_along_z(monkeypatch, tmp_path):
    """shift offsets the orbit along `viewup`, which is only the z axis while viewup is z-aligned.

    pyvista computes `center += np.array(viewup) * shift`, so the two arguments interact. The docstring says
    so; this pins it through `orbit()`, because a y-up scene shifted "upwards" moves along y.
    """
    scene = _terrain_scene()
    shifted = _orbit_path_points(
        scene, monkeypatch, tmp_path, viewup=(0.0, 1.0, 0.0), shift=2.0
    )
    unshifted = _orbit_path_points(
        scene, monkeypatch, tmp_path, viewup=(0.0, 1.0, 0.0), shift=0.0
    )
    moved = shifted.mean(axis=0) - unshifted.mean(axis=0)
    assert moved[1] == pytest.approx(2.0), (
        f"a y-up shift must move along y, got {moved}"
    )
    assert moved[2] == pytest.approx(0.0, abs=1e-6), (
        f"a y-up shift must not move along z, got {moved}"
    )
    scene.close()


def test_orbit_refuses_threaded_rather_than_writing_nothing(tmp_path):
    """threaded=True is rejected: it would return a path for a file that never gets written.

    `orbit_on_path(threaded=True)` returns before the render thread has appended a frame, so orbit's
    `finally` closes the writer first and no file is produced — silently, with the path handed back as if it
    had been. Refusing is the honest answer.
    """
    scene = _terrain_scene()
    out = tmp_path / "threaded.gif"
    with pytest.raises(ValueError, match="threaded"):
        scene.orbit(str(out), n_frames=6, threaded=True)
    assert not out.exists(), "nothing should be written for a rejected orbit"
    scene.close()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"factor": 0.0}, "positive, finite factor"),
        ({"factor": -2.0}, "positive, finite factor"),
        ({"factor": float("nan")}, "finite factor"),
        ({"factor": float("inf")}, "finite factor"),
        ({"factor": 1 + 2j}, "numeric factor"),
        ({"shift": float("nan")}, "finite shift"),
        ({"shift": "up"}, "numeric shift"),
        ({"n_frames": 1}, "at least 3"),
        ({"n_frames": 3.5}, "whole number of frames"),
        ({"viewup": ["a", "b", "c"]}, "numeric viewup"),
        ({"viewup": (0.0, 1.0)}, "3-component viewup"),
        ({"viewup": (0.0, 0.0, 0.0)}, "zero viewup"),
        ({"viewup": (0.0, 0.0, float("nan"))}, "finite viewup"),
    ],
    ids=[
        "zero-factor",
        "negative-factor",
        "nan-factor",
        "inf-factor",
        "complex-factor",
        "nan-shift",
        "string-shift",
        "too-few-frames",
        "fractional-frames",
        "string-viewup",
        "short-viewup",
        "zero-viewup",
        "nan-viewup",
    ],
)
def test_orbit_rejects_degenerate_shapes(tmp_path, kwargs, message):
    """Degenerate path arguments raise ValueError naming the argument, instead of rendering nonsense.

    Args:
        tmp_path: Destination for the file that must not be written.
        kwargs: The single degenerate argument under test.
        message: Fragment the error must contain, so the message names the offending argument.

    Test scenario:
        Before these arguments existed none of these values was reachable. A zero or negative factor gives a
        degenerate or mirrored path, fewer than three frames is not a circle, and a two-component viewup fails
        deep inside numpy with a message naming neither viewup nor orbit.
    """
    scene = _terrain_scene()
    with pytest.raises(ValueError, match=message):
        scene.orbit(str(tmp_path / "bad.gif"), **kwargs)
    scene.close()


def test_orbit_finalizes_writer_when_an_unknown_kwarg_raises(tmp_path):
    """An unrecognised orbit_kwarg raises TypeError, and the frame writer is still finalized (try/finally).

    Test scenario:
        `orbit_on_path` takes no `**kwargs`, so anything it does not name — the docstring calls out `factor`
        and `shift`, which are the named arguments here instead — raises TypeError from inside the try block.
        The writer must still be flushed and closed, or the plotter is left holding an open GIF writer and the
        next render appends to a half-written file. `animate` has this covered; `orbit` did not, and the new
        keyword surface makes it the likelier way in.
    """
    scene = _terrain_scene()
    out = tmp_path / "boom.gif"
    with pytest.raises(TypeError, match="not_a_real_orbit_argument"):
        scene.orbit(str(out), n_frames=4, not_a_real_orbit_argument=1.0)
    assert (
        getattr(scene.plotter, "mwriter", None) is None or scene.plotter.mwriter.closed
    ), "the frame writer must be closed even when orbit_on_path rejects a keyword"
    scene.close()


def test_jupyter_round_trips_the_process_global_backend():
    """jupyter() really does change PyVista's process-wide backend, and it can be put back.

    The doctest on the method shows the same round trip, but a doctest shares its interpreter with every other
    one in `doctests-3d`. Asserting it here, where the restore is in a fixture-free `finally` that pytest
    cannot skip, is what makes the claim safe to make.
    """
    scene = _terrain_scene()
    previous = pv.global_theme.jupyter_backend
    try:
        scene.jupyter("static")
        assert pv.global_theme.jupyter_backend == "static", (
            f"jupyter() must set the backend, got {pv.global_theme.jupyter_backend!r}"
        )
    finally:
        # close first: a raising restore would otherwise leak the window as well as the backend
        scene.close()
        pv.set_jupyter_backend(previous)
    assert pv.global_theme.jupyter_backend == previous, (
        f"the backend must round-trip, got {pv.global_theme.jupyter_backend!r} not {previous!r}"
    )


def test_orbit_accepts_a_numpy_viewup(monkeypatch, tmp_path):
    """A numpy array is a valid viewup, which is why the annotation is not `Sequence[float]`.

    `np.ndarray` is not a `typing.Sequence`, so annotating it that way puts every caller who spells a vector
    the natural way into the repo's mypy arg-type baseline. It works at runtime, and the length check has to
    cope with it too.
    """
    scene = _terrain_scene()
    seen = _record_pyvista_calls(monkeypatch, scene)
    scene.orbit(
        str(tmp_path / "spin.gif"), n_frames=6, viewup=np.array([0.0, 0.0, 1.0])
    )
    passed = seen["generate_orbital_path"]["viewup"]
    assert np.array_equal(passed, np.array([0.0, 0.0, 1.0])), (
        f"a numpy viewup must reach the generator unchanged, got {passed!r}"
    )
    scene.close()


@pytest.mark.parametrize(
    "bad",
    [np.zeros((3, 1)), np.float64(1.0)],
    ids=["column-vector", "zero-dimensional"],
)
def test_orbit_rejects_a_wrongly_shaped_numpy_viewup(tmp_path, bad):
    """A numpy viewup of the wrong shape is caught here, not deep inside pyvista.

    Widening the annotation to accept arrays admits shapes a length check cannot judge: `(3, 1)` has `len` 3
    and then dies in pyvista with a broadcast error naming neither viewup nor orbit, and a 0-d array has no
    `len` at all.
    """
    scene = _terrain_scene()
    with pytest.raises(ValueError, match="3-component viewup"):
        scene.orbit(str(tmp_path / "bad.gif"), n_frames=4, viewup=bad)
    scene.close()


@pytest.mark.parametrize("falsy", [False, 0], ids=["false", "zero"])
def test_orbit_allows_a_falsy_threaded(tmp_path, falsy):
    """Only a truthy `threaded` is refused, which is the same test pyvista itself applies.

    Args:
        tmp_path: Destination for the GIF.
        falsy: A `threaded` value that means "render synchronously".

    Test scenario:
        The guard reads `orbit_kwargs.get("threaded")` for its truthiness rather than comparing identity,
        because `orbit_on_path` does `if threaded:` too. Passing the argument explicitly off must therefore
        behave exactly like not passing it, not trip the guard.
    """
    scene = _terrain_scene()
    out = scene.orbit(str(tmp_path / "sync.gif"), n_frames=4, threaded=falsy)
    assert (tmp_path / "sync.gif").stat().st_size > 0, (
        f"threaded={falsy!r} means synchronous, so the file must still be written"
    )
    assert out.endswith("sync.gif")
    scene.close()


class _RecordingWriterPlotter:
    """Plotter stand-in that records which writer was opened and with what frame rate."""

    mwriter = object()  # satisfies the post-open guard in _open_writer

    def __init__(self):
        self.opened = None

    def open_movie(self, path, framerate):
        """Record a movie writer opening, as `pyvista.Plotter.open_movie` would."""
        self.opened = ("movie", path, framerate)

    def open_gif(self, path, fps):
        """Record a GIF writer opening, as `pyvista.Plotter.open_gif` would."""
        self.opened = ("gif", path, fps)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("clip.mp4", "movie"),
        ("clip.mov", "movie"),
        ("clip.avi", "movie"),
        ("clip.m4v", "movie"),
        ("clip.MP4", "movie"),
        ("clip.Mov", "movie"),
        ("clip.gif", "gif"),
        ("clip.GIF", "gif"),
        ("clip", "gif"),
        ("clip.mp4.gif", "gif"),
    ],
    ids=[
        "mp4",
        "mov",
        "avi",
        "m4v",
        "upper-mp4",
        "mixed-mov",
        "gif",
        "upper-gif",
        "no-suffix",
        "movie-suffix-in-stem",
    ],
)
def test_writer_dispatch_covers_every_movie_suffix(name, expected):
    """Every suffix in `_MOVIE_SUFFIXES` routes to open_movie, in any case; everything else opens a GIF.

    Args:
        name: File name whose suffix decides the writer.
        expected: The writer `_open_writer` should open.

    Test scenario:
        The pre-existing dispatch test covered `.mp4` and `.gif` only, so dropping `.m4v` from the tuple, or
        the `.lower()` that makes `clip.MP4` a movie, would both have gone unnoticed. `clip.mp4.gif` pins that
        the decision is made on the real suffix rather than a substring.
    """
    plotter = _RecordingWriterPlotter()
    _open_writer(plotter, name, 10)
    assert plotter.opened[0] == expected, (
        f"{name!r} should open a {expected} writer, got {plotter.opened[0]!r}"
    )


@pytest.mark.parametrize(
    "name, keyword",
    [("clip.mp4", "framerate"), ("clip.gif", "fps")],
    ids=["movie-framerate", "gif-fps"],
)
def test_writer_dispatch_forwards_the_frame_rate(name, keyword):
    """The frame rate reaches the writer, under whichever keyword that writer names it.

    Args:
        name: File name selecting the movie or GIF writer.
        keyword: The keyword pyvista names it with, used only in the failure message.

    Test scenario:
        `open_movie` takes `framerate` and `open_gif` takes `fps`. Only the dispatch was asserted before, so
        passing the wrong value — or the default — to either would not have failed a test.
    """
    plotter = _RecordingWriterPlotter()
    _open_writer(plotter, name, 24)
    assert plotter.opened[2] == 24, (
        f"{name!r} must forward 24 as {keyword}, got {plotter.opened[2]!r}"
    )


def test_orbit_forwards_the_validated_vector_not_the_original(monkeypatch, tmp_path):
    """pyvista receives the float array the guard checked, never the caller's original object.

    Args:
        monkeypatch: Installs the recording stubs.
        tmp_path: Destination for the notional GIF.

    Test scenario:
        pyvista does `np.array(viewup) * shift` on whatever it is handed. Validating a converted copy and
        passing the original lets a list of numeric strings die inside pyvista with a ufunc error naming
        neither viewup nor orbit, so the check has to be on what actually reaches the call.
    """
    scene = _terrain_scene()
    seen = _record_pyvista_calls(monkeypatch, scene)
    scene.orbit(str(tmp_path / "spin.gif"), n_frames=4, viewup=["0", "0", "1"])
    passed = seen["generate_orbital_path"]["viewup"]
    assert passed.dtype == np.dtype(float), (
        f"the forwarded viewup must be the validated float array, got {passed!r}"
    )
    assert np.array_equal(passed, np.array([0.0, 0.0, 1.0])), (
        f"the forwarded viewup must equal the converted vector, got {passed!r}"
    )
    assert np.array_equal(seen["orbit_on_path"]["viewup"], passed), (
        "the camera must get the same validated vector as the path"
    )
    scene.close()


@pytest.mark.parametrize(
    "masked",
    [
        np.ma.array([0.0, 0.0, 1.0], mask=[True, False, False]),
        np.ma.array([5.0, 0.0, 1.0], mask=[True, False, False]),
        np.ma.array([0.0, 0.0, 1.0], mask=[True, True, True]),
    ],
    ids=["masked-zero", "masked-value", "fully-masked"],
)
def test_orbit_refuses_a_masked_viewup(tmp_path, masked):
    """A masked up vector is refused rather than read from under its mask.

    Args:
        tmp_path: Destination for the file that must not be written.
        masked: An up vector with at least one component masked out.

    Test scenario:
        `np.asarray` discards a mask and returns the underlying data, so accepting one would make the orbit
        depend on a value the caller declared missing — `5.0` under a mask tilts the plane, `nan` under one
        raises, and a fully masked vector names no up direction at all yet would clear the zero-vector check.
    """
    scene = _terrain_scene()
    with pytest.raises(ValueError, match="masked viewup"):
        scene.orbit(str(tmp_path / "masked.gif"), n_frames=4, viewup=masked)
    scene.close()


def test_up_vector_and_finite_number_reject_without_a_scene():
    """The two validation helpers are pure, and are exercised here without building a VTK scene.

    Test scenario:
        Every other test of these guards goes through `orbit()`, which costs a render window for a check that
        touches none. These pin the helpers directly, including that a valid vector round-trips to a float
        array.
    """
    assert _finite_number("0.9", "factor") == pytest.approx(0.9), (
        "a numeric string is accepted, as the docstring says"
    )
    assert np.array_equal(_up_vector([0, 0, 1]), np.array([0.0, 0.0, 1.0])), (
        "a valid vector converts to a float array"
    )
    assert _up_vector(None) is None, "None passes through untouched"
    with pytest.raises(ValueError, match="numeric factor"):
        _finite_number(1 + 2j, "factor")
    with pytest.raises(ValueError, match="finite factor"):
        _finite_number(float("inf"), "factor")
    with pytest.raises(ValueError, match="3-component viewup"):
        _up_vector([1.0, 2.0])
