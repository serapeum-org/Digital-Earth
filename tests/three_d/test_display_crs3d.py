"""The 3-D tier's display CRS: layers in different CRSs share one scene (TD-5a, #291).

The tier declared it had no display CRS and placed every layer's coordinates as given, so a DEM in UTM and a point
in lon/lat were drawn some 435,000 scene units apart, without a warning.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from pyramids.dataset import Dataset  # noqa: E402
from pyramids.feature import FeatureCollection  # noqa: E402

from digitalearth.base.sources import get_source  # noqa: E402
from digitalearth.base.spec.bounds import same_crs  # noqa: E402
from digitalearth.three_d import Scene3D  # noqa: E402

DEM_PATH = "examples/data/acc4000.tif"


@pytest.fixture(scope="module")
def dem() -> Dataset:
    """Return `acc4000.tif`, a DEM in EPSG:32618.

    Returns:
        The dataset.
    """
    return Dataset.read_file(DEM_PATH)


def _centre_point(dem: Dataset, crs) -> FeatureCollection:
    """Return one point at the DEM's centre, expressed in `crs`.

    Args:
        dem: The DEM whose centre is used.
        crs: The CRS to express the point in.

    Returns:
        A one-point `FeatureCollection` with a `v` column.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    xmin, ymin, xmax, ymax = dem.bbox
    frame = gpd.GeoDataFrame(
        {"v": [1.0]},
        geometry=[Point((xmin + xmax) / 2, (ymin + ymax) / 2)],
        crs=dem.epsg,
    )
    return FeatureCollection(frame.to_crs(crs))


@pytest.fixture
def scene():
    """Yield an off-screen `Scene3D`, closed on the way out.

    Yields:
        The scene.
    """
    built = Scene3D(off_screen=True)
    yield built
    built.close()


class TestLayersShareOneCrs:
    """Placing each layer in the scene's CRS."""

    def test_a_point_in_another_crs_lands_on_the_dem(self, scene, dem):
        """A lon/lat point at the DEM's centre is drawn inside the DEM, not 435,000 units away.

        Test scenario:
            The reproduction from #291: a UTM DEM, then a point given in EPSG:4326. Before, the point's x was -75.
        """
        scene.terrain(dem)
        scene.point_cloud(_centre_point(dem, 4326), value_column="v")
        terrain_mesh, cloud = scene.layers[0][0], scene.layers[1][0]
        xmin, xmax = terrain_mesh.bounds[0], terrain_mesh.bounds[1]
        x = cloud.bounds[0]
        assert xmin <= x <= xmax, (
            f"the point is at x={x}, outside the DEM's {xmin}..{xmax}"
        )

    def test_a_scene_takes_the_crs_of_its_first_layer(self, scene, dem):
        """With no `crs=`, the first layer that declares a CRS decides the scene's."""
        scene.terrain(dem)
        assert same_crs(scene.display_crs, dem.epsg), scene.display_crs

    def test_a_scene_in_one_crs_reprojects_nothing(self, scene, dem, mocker):
        """Layers that share a CRS are drawn as they are, with no warp.

        Test scenario:
            The display CRS must not cost anything in the common case, or move a pixel of an existing render.
        """
        warp = mocker.patch("digitalearth.three_d.base.reproject")
        scene.terrain(dem)
        scene.point_cloud(_centre_point(dem, dem.epsg), value_column="v")
        assert warp.call_count == 0, f"reproject was called {warp.call_count} time(s)"

    def test_a_scene_given_a_crs_reprojects_the_dem_into_it(self, dem):
        """`Scene3D(crs=4326)` draws the UTM DEM in degrees."""
        scene = Scene3D(off_screen=True, crs=4326)
        try:
            scene.terrain(dem)
            xmin, xmax = scene.layers[0][0].bounds[:2]
        finally:
            scene.close()
        assert -76.0 < xmin < xmax < -74.0, (
            f"the DEM spans x {xmin}..{xmax}, not longitudes"
        )

    def test_data_that_declares_no_crs_is_placed_as_given(self):
        """A bare array has no CRS to warp from, so its coordinates are used as they are."""
        scene = Scene3D(off_screen=True, crs=4326)
        try:
            scene.point_cloud(np.array([[1.0e6, 2.0e6, 0.0]]))
            x = scene.layers[0][0].bounds[0]
        finally:
            scene.close()
        assert x == pytest.approx(1.0e6), x

    def test_a_source_in_another_crs_is_refused(self, dem):
        """An extracted `Source` cannot be warped, so a mismatch is refused rather than drawn in the wrong place."""
        source = get_source(dem)
        scene = Scene3D(off_screen=True, crs=4326)
        try:
            with pytest.raises(ValueError, match="cannot be reprojected"):
                scene.terrain(source)
        finally:
            scene.close()


class TestTheElevationFactor:
    """`terrain` converts metre elevations for a geographic scene."""

    def test_the_factor_follows_the_scene_crs(self):
        """A DEM with no CRS in a lon/lat scene has its metres converted to degrees, as a lon/lat DEM does.

        Test scenario:
            The factor was read from each layer's own CRS, so a CRS-less DEM kept metre heights over degree
            coordinates — a needle.
        """
        y, x = np.mgrid[0:4, 0:5]
        elevation = 100.0 * (x + y).astype(float)
        scene = Scene3D(off_screen=True, crs=4326)
        try:
            scene.terrain(get_source(elevation))
            ztop = scene.layers[0][0].bounds[5]
        finally:
            scene.close()
        assert ztop == pytest.approx(700.0 / 111_320.0), ztop


class TestTheGlobe:
    """`globe` draws in EPSG:4326."""

    @pytest.fixture
    def field(self) -> Dataset:
        """Return a global 10-degree field in EPSG:4326.

        Returns:
            The dataset.
        """
        from pyramids.base.georeference import GeoReference

        lat, lon = np.mgrid[85:-90:-10, -175:180:10]
        values = np.cos(np.radians(lat)) * np.sin(np.radians(lon))
        return Dataset.from_array(
            values.astype("float32"),
            geo_ref=GeoReference(
                top_left_corner=(-180.0, 90.0), cell_size=10.0, epsg=4326
            ),
        )

    def test_a_globe_declares_epsg_4326(self, scene, field):
        """A globe sets the scene's CRS to the one it is drawn in."""
        scene.globe(field, coastlines=False)
        assert same_crs(scene.display_crs, 4326), scene.display_crs

    def test_a_globe_in_a_scene_drawn_in_another_crs_is_refused(
        self, scene, dem, field
    ):
        """A sphere cannot share a scene with a UTM surface; the message names both CRSs."""
        scene.terrain(dem)
        with pytest.raises(ValueError, match=r"globe\(\) draws in EPSG:4326.*32618"):
            scene.globe(field, coastlines=False)
