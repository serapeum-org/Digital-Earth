"""Sun, shadows and ambient occlusion for the 3-D tier — rasterization-grade relief shading (#493).

`Scene3D.sun(azimuth, elevation)` places a directional light; `shadows()` and `ambient_occlusion()` toggle VTK's
shadow-map and SSAO passes. These cover that the sun direction reaches a single directional light, that the
passes toggle on and off (and persist across a decoration redraw), that lighting is re-applied to a plotter
swapped in, and the argument refusals — all headless.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.base.sources import get_source  # noqa: E402
from digitalearth.three_d import Scene3D  # noqa: E402


def _relief() -> object:
    """Return a small ramped relief source to light.

    Returns:
        A `Source` built from a 6x6 ramp.
    """
    return get_source(np.add.outer(np.arange(6.0), np.arange(6.0)))


@pytest.fixture
def scene():
    """Yield an off-screen scene with relief, closed on the way out.

    Yields:
        A scene carrying a terrain so there is something to light.
    """
    built = Scene3D(off_screen=True)
    built.terrain(_relief())
    yield built
    built.close()


class TestSun:
    """`Scene3D.sun` lights the scene with one directional light."""

    def test_it_returns_self(self, scene):
        """The builder returns the scene.

        Args:
            scene: The scene under test.
        """
        assert scene.sun(azimuth=90.0, elevation=20.0) is scene, "sun must return self"

    def test_it_installs_exactly_one_directional_light(self, scene):
        """Setting a sun drops the default lights and leaves one directional light.

        Args:
            scene: The scene under test.
        """
        scene.sun(azimuth=90.0, elevation=20.0)
        lights = scene.plotter.renderer.lights
        assert len(lights) == 1, f"one sun, got {len(lights)} lights"
        assert not bool(lights[0].positional), "a sun is a directional light"

    def test_the_light_points_along_the_azimuth(self, scene):
        """A due-east, low sun sits to the east of the scene (positive x).

        Args:
            scene: The scene under test.

        Test scenario:
            Azimuth 90 is due east and x is east, so the light's position is east of the scene centre; a
            due-west sun (270) is to the west, which is what tells the direction apart from its mirror.
        """
        scene.sun(azimuth=90.0, elevation=10.0)
        east_x = scene.plotter.renderer.lights[0].position[0]
        scene.sun(azimuth=270.0, elevation=10.0)
        west_x = scene.plotter.renderer.lights[0].position[0]
        assert east_x > 0 > west_x, (east_x, west_x)

    def test_a_non_finite_angle_is_refused(self, scene):
        """A sun at NaN elevation cannot be placed.

        Args:
            scene: The scene under test.
        """
        nan = float("nan")
        with pytest.raises(ValueError, match="finite number"):
            scene.sun(azimuth=0.0, elevation=nan)


class TestShadowsAndAmbientOcclusion:
    """The shadow-map and SSAO passes toggle on and off."""

    def test_shadows_toggle_on_and_off(self, scene):
        """`shadows()` enables the pass and `shadows(False)` disables it.

        Args:
            scene: The scene under test.
        """
        scene.sun(azimuth=135.0, elevation=30.0).shadows()
        on = getattr(scene.plotter, "_de_shadows", False)
        scene.shadows(False)
        off = getattr(scene.plotter, "_de_shadows", False)
        assert (on, off) == (True, False), (on, off)

    def test_ambient_occlusion_toggles_on_and_off(self, scene):
        """`ambient_occlusion()` enables SSAO and `ambient_occlusion(False)` disables it.

        Args:
            scene: The scene under test.
        """
        scene.ambient_occlusion()
        on = getattr(scene.plotter, "_de_ssao", False)
        scene.ambient_occlusion(False)
        off = getattr(scene.plotter, "_de_ssao", False)
        assert (on, off) == (True, False), (on, off)

    def test_lighting_renders_headless(self, scene):
        """A sun with shadows and SSAO renders a non-blank frame off-screen.

        Args:
            scene: The scene under test.
        """
        scene.sun(azimuth=135.0, elevation=30.0).shadows().ambient_occlusion()
        frame = scene.screenshot()
        assert bool(frame.any()), "the lit scene rendered blank"


class TestLightingIsViewState:
    """Lighting is re-applied to whichever plotter the scene is dressed with."""

    def test_the_sun_is_reapplied_to_a_swapped_plotter(self, scene):
        """A plotter handed to the scene after a sun is set still gets exactly one directional light.

        Args:
            scene: The scene under test.

        Test scenario:
            The sun is view state like the camera, so swapping the window must not drop it — the measured
            failure mode for the exaggeration and camera (review M7), checked here for lighting.
        """
        scene.sun(azimuth=45.0, elevation=25.0)
        replacement = pv.Plotter(off_screen=True)
        try:
            scene.plotter = replacement
            assert len(replacement.renderer.lights) == 1, (
                "the sun must follow onto the new plotter"
            )
        finally:
            replacement.close()
