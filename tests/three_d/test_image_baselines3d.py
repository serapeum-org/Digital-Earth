"""3-D image baselines: one rendered scene per builder, compared with a committed image (DE-47, #290).

Every test here renders a small, deterministic scene at a fixed size from the isometric camera. With
`--image3d-compare` (the `test-3d-images` task, run in CI's Linux 3-D job) each is compared with its baseline in
`tests/baseline3d`; without it the scene is only rendered. See `tests/three_d/_image3d.py`.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.base.sources import get_source  # noqa: E402
from digitalearth.three_d import Scene3D  # noqa: E402
from tests.three_d._image3d import (  # noqa: E402
    ERROR_THRESHOLD,
    WINDOW_SIZE,
    check_against_baseline,
    render,
)


def _dem(bump: float = 0.0) -> np.ndarray:
    """Return a 24x32 elevation grid with one 50 m Gaussian bump, optionally raising its centre cell.

    Args:
        bump: Metres added to the centre cell.

    Returns:
        The elevation array.
    """
    y, x = np.mgrid[0:24, 0:32]
    z = 50.0 * np.exp(-((x - 16) ** 2 + (y - 12) ** 2) / 60.0)
    z[12, 16] += bump
    return z


def _cube() -> np.ndarray:
    """Return a 16^3 cube holding one Gaussian blob, peaking at 1.0 in the middle.

    Returns:
        The cube, shaped `(z, y, x)`.
    """
    zz, yy, xx = np.mgrid[0:16, 0:16, 0:16]
    return np.exp(-((xx - 8) ** 2 + (yy - 8) ** 2 + (zz - 8) ** 2) / 20.0)


def _shot(build) -> np.ndarray:
    """Build a scene with `build`, render it, and close it.

    Args:
        build: A callable that adds layers to a `Scene3D`.

    Returns:
        The rendered image.
    """
    scene = Scene3D(off_screen=True, window_size=WINDOW_SIZE)
    try:
        build(scene)
        return render(scene)
    finally:
        scene.close()


@pytest.mark.image3d
class TestTheBuilderBaselines:
    """One baseline per 3-D builder."""

    def test_terrain(self, request):
        """A DEM drawn as a surface with the terrain colormap."""
        image = _shot(lambda scene: scene.terrain(get_source(_dem()), cmap="terrain"))
        check_against_baseline(request.config, image, "terrain")

    def test_terrain_exaggerated(self, request):
        """The same DEM with a vertical exaggeration of 3, applied as a view scale."""
        image = _shot(
            lambda scene: scene.terrain(
                get_source(_dem()), cmap="terrain", z_exaggeration=3.0
            )
        )
        check_against_baseline(request.config, image, "terrain_exaggerated")

    def test_point_cloud(self, request):
        """200 seeded random points coloured by their height."""
        points = np.random.default_rng(7).uniform(0, 10, size=(200, 3))
        image = _shot(lambda scene: scene.point_cloud(points, values=points[:, 2]))
        check_against_baseline(request.config, image, "point_cloud")

    def test_extruded_polygons(self, request):
        """Three unit squares extruded, and coloured, by a height column."""
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import box

        polygons = gpd.GeoDataFrame(
            {"h": [1.0, 2.0, 3.0]},
            geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1), box(4, 0, 5, 1)],
            crs=3857,
        )
        image = _shot(
            lambda scene: scene.extruded_polygons(polygons, height="h", column="h")
        )
        check_against_baseline(request.config, image, "extruded_polygons")

    def test_vectors(self, request):
        """A 5x5 rotational vector field drawn as arrow glyphs."""
        gy, gx = np.mgrid[0:5, 0:5]
        points = np.c_[gx.ravel(), gy.ravel(), np.zeros(25)]
        vectors = np.c_[-(gy.ravel() - 2), gx.ravel() - 2, np.zeros(25)] * 0.2
        image = _shot(lambda scene: scene.vectors(points, vectors))
        check_against_baseline(request.config, image, "vectors")

    def test_volume(self, request):
        """A Gaussian blob ray-cast as a volume."""
        image = _shot(lambda scene: scene.volume(_cube()))
        check_against_baseline(request.config, image, "volume")

    def test_isosurface(self, request):
        """The surface where the blob reaches 0.5."""
        image = _shot(lambda scene: scene.isosurface(_cube(), isosurfaces=[0.5]))
        check_against_baseline(request.config, image, "isosurface")

    def test_globe(self, request):
        """A global 10-degree field draped on a sphere, without coastlines so nothing is downloaded."""
        from pyramids.base.georeference import GeoReference
        from pyramids.dataset import Dataset

        lat, lon = np.mgrid[85:-90:-10, -175:180:10]
        field = np.cos(np.radians(lat)) * np.sin(np.radians(lon))
        dataset = Dataset.from_array(
            field.astype("float32"),
            geo_ref=GeoReference(
                top_left_corner=(-180.0, 90.0), cell_size=10.0, epsg=4326
            ),
        )
        image = _shot(lambda scene: scene.globe(dataset, coastlines=False))
        check_against_baseline(request.config, image, "globe")


class TestTheComparisonThreshold:
    """The threshold tells a real change from none, measured on this machine rather than against a baseline."""

    def test_the_same_scene_rendered_twice_is_within_the_threshold(self):
        """Two renders of one scene compare as the same picture."""
        first = _shot(lambda scene: scene.terrain(get_source(_dem()), cmap="terrain"))
        second = _shot(lambda scene: scene.terrain(get_source(_dem()), cmap="terrain"))
        error = pv.compare_images(first, second)
        assert error <= ERROR_THRESHOLD, f"error {error} for an unchanged scene"

    def test_raising_one_cell_fails_the_comparison(self):
        """A 5 m change to one cell of the terrain is caught by the threshold.

        Test scenario:
            The comparison exists to notice geometry moving; if one cell can change without crossing the
            threshold, the baselines protect nothing.
        """
        before = _shot(lambda scene: scene.terrain(get_source(_dem()), cmap="terrain"))
        after = _shot(
            lambda scene: scene.terrain(get_source(_dem(bump=5.0)), cmap="terrain")
        )
        error = pv.compare_images(before, after)
        assert error > ERROR_THRESHOLD, f"a one-cell change scored only {error}"
