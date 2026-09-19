"""The 3-D tier builds its plotter, and imports PyVista, only when something is drawn (DE-47, #290).

A scene was a render window from its first line: `Scene3DBase.__init__` built a `pyvista.Plotter`, and five modules
imported `pyvista` at load. The renderer seam splits describing a scene from drawing it, and describing should need
neither.
"""

import subprocess
import sys

import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.three_d import Scene3D  # noqa: E402


@pytest.fixture
def plotters_built(monkeypatch):
    """Count the `pyvista.Plotter` objects built while a test runs.

    Args:
        monkeypatch: pytest's monkeypatch, used to wrap the constructor.

    Returns:
        A list that gains one entry per plotter built.
    """
    built = []
    real = pv.Plotter

    def counting(*args, **kwargs):
        plotter = real(*args, **kwargs)
        built.append(plotter)
        return plotter

    monkeypatch.setattr(pv, "Plotter", counting)
    return built


class TestThePlotterIsBuiltOnFirstUse:
    """`Scene3DBase.plotter`."""

    def test_building_a_scene_builds_no_plotter(self, plotters_built):
        """A scene that has drawn nothing has no render window yet."""
        Scene3D(off_screen=True)
        assert plotters_built == [], (
            f"{len(plotters_built)} plotter(s) built by the constructor"
        )

    def test_drawing_builds_exactly_one_plotter(self, plotters_built):
        """The first layer builds the plotter, and later layers reuse it."""
        scene = Scene3D(off_screen=True)
        scene.add_mesh(pv.Sphere())
        scene.add_mesh(pv.Cube())
        count = len(plotters_built)
        scene.close()
        assert count == 1, f"expected one plotter, built {count}"

    def test_the_plotter_takes_the_scene_settings(self):
        """The window size and extra keywords given to the scene reach the plotter built later."""
        scene = Scene3D(off_screen=True, window_size=(320, 240), shape=(1, 2))
        settings = (list(scene.plotter.window_size), tuple(scene.plotter.shape))
        scene.close()
        assert settings == ([320, 240], (1, 2)), settings

    def test_a_scene_given_no_theme_uses_the_house_theme(self):
        """The default theme is applied when the plotter is built, not dropped by building it later."""
        scene = Scene3D(off_screen=True)
        background = scene.plotter.background_color.hex_rgb
        scene.close()
        assert background == "#ffffff", background

    def test_closing_a_scene_that_never_drew_builds_no_plotter(self, plotters_built):
        """Closing has nothing to free when no render window was opened, and does not open one to close it."""
        scene = Scene3D(off_screen=True)
        scene.close()
        assert plotters_built == [], f"close() built {len(plotters_built)} plotter(s)"

    def test_a_plotter_assigned_by_hand_is_the_one_used(self):
        """Assigning `plotter` replaces what the scene would build, which is how a test swaps in a stand-in."""
        scene = Scene3D(off_screen=True)
        stand_in = object()
        scene.plotter = stand_in
        assert scene.plotter is stand_in, scene.plotter


class TestImportingTheTierLoadsNoEngine:
    """Module-level imports of the 3-D tier."""

    def test_importing_the_scene_module_does_not_import_pyvista(self):
        """A fresh interpreter can import `Scene3D` without loading PyVista or VTK.

        Test scenario:
            `pyvista` was imported at module level in `base.py`, `point_cloud.py`, `terrain.py`, `vector.py` and
            `volume.py`, so describing a scene needed the whole VTK stack in memory.
        """
        code = (
            "import sys\n"
            "import digitalearth.three_d.scene3d\n"
            "print(sorted(name for name in ('pyvista', 'vtkmodules') if name in sys.modules))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "[]", result.stdout + result.stderr
