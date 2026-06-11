"""Tests for digitalearth.three_d.Scene3D — the PyVista shared-plotter host.

Gated on the optional ``3d`` extra: when pyvista is not installed these are skipped, so the default suite stays
green; install ``digitalearth[3d]`` (or run the ``viz3d`` pixi env) to exercise them for real.
"""
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.three_d import Scene3D


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test so no window opens in CI."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def _dem_grid(n: int = 12) -> "pv.ImageData":
    """Build a small ramped DEM as a warped ImageData (a non-flat surface to render)."""
    arr = np.add.outer(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n))
    grid = pv.ImageData(dimensions=(n, n, 1))
    grid.point_data["z"] = arr.ravel(order="F")
    return grid.warp_by_scalar("z", factor=3.0)


def test_scene_starts_empty():
    """A fresh Scene3D owns a plotter and no layers."""
    scene = Scene3D(off_screen=True)
    assert scene.plotter is not None
    assert scene.layers == []
    scene.close()


def test_add_mesh_registers_one_layer():
    """add_mesh adds the actor to the plotter and registers exactly one (mesh, actor) layer."""
    scene = Scene3D(off_screen=True)
    actor = scene.add_mesh(_dem_grid(), scalars="z", cmap="terrain")
    assert actor is not None
    assert len(scene.layers) == 1
    mesh, registered = scene.layers[0]
    assert registered is actor
    scene.close()


def test_two_layers_compose_on_one_plotter():
    """Two meshes stack on the same plotter and both register as layers."""
    scene = Scene3D(off_screen=True)
    scene.add_mesh(_dem_grid(), scalars="z")
    scene.add_mesh(pv.Sphere(radius=0.5, center=(5, 5, 5)))
    assert len(scene.layers) == 2
    scene.close()


def test_screenshot_is_a_nonempty_rgb_frame():
    """Off-screen screenshot returns a real (H, W, 3) RGB array with content."""
    scene = Scene3D(off_screen=True)
    scene.add_mesh(_dem_grid(), scalars="z", cmap="terrain")
    img = scene.screenshot()
    assert img.ndim == 3 and img.shape[-1] == 3
    assert bool(img.any())
    scene.close()


def test_screenshot_writes_png(tmp_path):
    """screenshot(path=...) writes a PNG file to disk."""
    scene = Scene3D(off_screen=True)
    scene.add_mesh(_dem_grid(), scalars="z")
    out = tmp_path / "frame.png"
    scene.screenshot(path=str(out))
    assert out.exists() and out.stat().st_size > 0
    scene.close()


def test_save_dispatches_png_vs_html(tmp_path):
    """save() writes a PNG for image paths and a self-contained HTML page for .html paths."""
    scene = Scene3D(off_screen=True)
    scene.add_mesh(_dem_grid(), scalars="z")
    png, html = tmp_path / "s.png", tmp_path / "s.html"
    png_result = scene.save(str(png))
    html_result = scene.save(str(html))
    assert png.exists() and png.stat().st_size > 0
    assert html.exists() and html.stat().st_size > 0
    # save() returns the RGB frame for a raster path and None for the HTML path
    assert png_result is not None and png_result.ndim == 3
    assert html_result is None
    scene.close()


def test_add_volume_registers_layer():
    """add_volume() ray-casts a scalar field and registers it as a layer."""
    scene = Scene3D(off_screen=True)
    grid = pv.ImageData(dimensions=(6, 6, 6))
    grid.cell_data["v"] = np.linspace(0.0, 1.0, 5 * 5 * 5)
    actor = scene.add_volume(grid, cmap="viridis")
    assert actor is not None and len(scene.layers) == 1
    scene.close()


def test_show_off_screen_does_not_block():
    """show() renders a frame off-screen and returns without opening a blocking window."""
    scene = Scene3D(off_screen=True)
    scene.add_mesh(_dem_grid())
    scene.show()  # must return (off-screen); no assertion beyond "did not hang/raise"
    scene.close()


def test_context_manager_closes_plotter():
    """The context manager closes the plotter on exit and does not suppress exceptions."""
    with Scene3D(off_screen=True) as scene:
        scene.add_mesh(_dem_grid(), scalars="z")
        assert len(scene.layers) == 1
    # after exit the plotter is closed; re-closing is a no-op (must not raise)
    scene.close()

    with pytest.raises(ValueError):
        with Scene3D(off_screen=True):
            raise ValueError("propagates")
