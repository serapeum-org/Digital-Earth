"""Scene3D shows a static image inline in a notebook, via ``_repr_png_`` (#558).

``ds.plot(var, backend="3d")`` returns a ``Scene3D``; without a rich repr a notebook shows only its text repr,
and ``show()`` is a live widget that does not survive ``nbconvert``. ``_repr_png_`` returns a rendered PNG of the
off-screen scene, so a plain ``plot(backend="3d")`` embeds a static image that survives a static export. Gated on
the ``3d`` extra.
"""

# ruff: noqa: E402
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.base.sources import get_source
from digitalearth.three_d import Scene3D

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless so the repr never opens a window."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def _dem():
    """A small ramped DEM Source."""
    return get_source(np.add.outer(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8)))


def test_repr_png_returns_a_png_of_the_rendered_scene():
    """``_repr_png_`` returns PNG bytes of a drawn scene, so a notebook embeds a static image.

    Test scenario:
        A scene with a terrain layer renders to a non-trivial PNG whose bytes start with the PNG signature —
        the inline image a plain ``plot(backend="3d")`` cell shows.
    """
    scene = Scene3D(off_screen=True)
    scene.terrain(_dem())
    png = scene._repr_png_()
    try:
        assert isinstance(png, bytes), type(png)
        assert png.startswith(_PNG_MAGIC), png[:8]
        assert len(png) > 1000, f"expected a real frame, got {len(png)} bytes"
    finally:
        scene.close()


def test_repr_png_decodes_to_the_scene_frame_size():
    """The embedded PNG decodes to an image the size of the scene's window (not a stub).

    Test scenario:
        A scene built at a fixed window size produces a PNG that decodes to that height and width, confirming
        the repr renders the actual frame rather than a placeholder.
    """
    from io import BytesIO

    from PIL import Image

    scene = Scene3D(off_screen=True, window_size=(240, 180))
    scene.terrain(_dem())
    try:
        image = Image.open(BytesIO(scene._repr_png_()))
        assert image.format == "PNG", image.format
        assert image.size == (240, 180), image.size
    finally:
        scene.close()


def test_repr_png_renders_an_empty_scene_without_raising():
    """An undrawn scene still reprs to a PNG (the background frame), so a bare cell never errors.

    Test scenario:
        A scene with no layers returns a valid PNG rather than raising, so displaying a freshly built scene is
        always safe.
    """
    scene = Scene3D(off_screen=True)
    try:
        png = scene._repr_png_()
        assert isinstance(png, bytes) and png.startswith(_PNG_MAGIC), png[:8]
    finally:
        scene.close()


def test_repr_png_degrades_to_none_when_rendering_fails(monkeypatch):
    """A repr must never raise: if the frame cannot be rendered, ``_repr_png_`` returns ``None`` (L2).

    Test scenario:
        A render failure (here a headless on-screen plotter, simulated by a raising ``screenshot``) makes
        ``_repr_png_`` fall back to ``None`` so a notebook shows the text repr instead of a traceback.
    """
    scene = Scene3D(off_screen=True)
    scene.terrain(_dem())
    try:
        monkeypatch.setattr(
            type(scene),
            "screenshot",
            lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("no display")),
        )
        assert scene._repr_png_() is None
    finally:
        scene.close()
