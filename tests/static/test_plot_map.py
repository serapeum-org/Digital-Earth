"""A drawn layer is reprojected to the display CRS, and the caller's own object is left as it was.

This file held the tests for ``StaticGlyph.plot``. That class is gone, and with it every test of its own
plumbing — its bare-ndarray entry point, its id-labelled point overlay, its ``no_data_value`` guards. One
property those tests measured is not StaticGlyph's at all, though, and had no guard anywhere else: reprojection
is **non-mutating**. ``StaticGlyph.plotCatchment`` reprojected three GeoDataFrames to the points' CRS and
``tests/static/test_plot_catchment.py`` checked that the caller's frames came back unchanged. ``Map`` does the
same thing to every layer it draws, against its own display CRS, so the property moves here rather than being
deleted with the class that used to carry it.

Both tests draw in EPSG:4326 from data stored in EPSG:32618, and each asserts the reprojection **happened**
before asserting that the input survived it. Without that first assertion a ``Map`` that quietly skipped
reprojection would satisfy the rest, and the guard would pass while measuring nothing.
"""

from pathlib import Path

import pytest
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

from digitalearth.static import Map

#: The repository root, which the data paths below hang from.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: A point layer stored in EPSG:32618 (UTM 18N), i.e. in metres.
POINTS = REPO_ROOT / "tests" / "data" / "points.geojson"

#: A raster stored in the same projected CRS.
RASTER = REPO_ROOT / "examples" / "data" / "acc4000.tif"

#: The CRS both files are stored in.
STORED_EPSG = 32618

#: The display CRS both tests draw in — degrees, so a reprojected coordinate is unmistakable.
DISPLAY_EPSG = 4326

#: Longitudes and latitudes are inside this box; the stored eastings/northings (~4.4e5) are not.
DEGREE_LIMIT = 180.0


class TestDrawingLeavesTheCallersDataAlone:
    """A layer is reprojected on its way to the axes, not in the object the caller still holds."""

    def test_a_vector_layer_is_reprojected_without_touching_the_caller(self):
        """``Map.points`` draws in the display CRS and hands the caller's collection back untouched.

        Test scenario:
            The same file is read twice, independently: one copy is drawn on a 4326 ``Map``, the other is
            never passed to anything. The drawn offsets must be degrees — proof the draw reprojected — while
            the drawn copy must still agree with the untouched one on its CRS and its extent. Reprojecting in
            place would move the caller's geometries under them, which is what the deleted catchment test
            guarded against.
        """
        drawn = FeatureCollection.read_file(str(POINTS))
        untouched = FeatureCollection.read_file(str(POINTS))

        collection = Map(crs=DISPLAY_EPSG).points(drawn, size=40)

        offsets = collection.get_offsets()
        assert abs(offsets).max() < DEGREE_LIMIT, (
            f"the draw did not reproject to degrees: offsets reach {abs(offsets).max()}"
        )
        assert drawn.epsg == untouched.epsg, (
            f"the draw reprojected the caller's collection in place: {drawn.epsg}"
        )
        assert tuple(drawn.total_bounds) == pytest.approx(
            tuple(untouched.total_bounds)
        ), f"the draw moved the caller's geometries: {tuple(drawn.total_bounds)}"

    def test_a_raster_layer_is_reprojected_without_touching_the_caller(self):
        """``Map.field`` warps a copy to the display CRS and leaves the caller's ``Dataset`` alone.

        Test scenario:
            The raster side of the same property, which nothing measured. A 4326 ``Map`` must frame the field
            in degrees while the ``Dataset`` the caller passed keeps its own EPSG, geotransform and shape — a
            warp applied in place would silently resample the object they go on to read or write.
        """
        drawn = Dataset.read_file(str(RASTER))
        untouched = Dataset.read_file(str(RASTER))

        scene = Map(crs=DISPLAY_EPSG)
        scene.field(drawn)

        left, right = scene.ax.get_xlim()
        assert max(abs(left), abs(right)) < DEGREE_LIMIT, (
            f"the field was not framed in degrees: x limits are ({left}, {right})"
        )
        assert drawn.epsg == untouched.epsg, (
            f"the draw reprojected the caller's Dataset in place: {drawn.epsg}"
        )
        assert drawn.geotransform == pytest.approx(untouched.geotransform), (
            f"the draw rewrote the caller's geotransform: {drawn.geotransform}"
        )
        assert drawn.shape == untouched.shape, (
            f"the draw resampled the caller's Dataset: {drawn.shape}"
        )

    def test_the_fixtures_are_stored_in_a_crs_the_display_crs_is_not(self):
        """The two tests above are only meaningful if the data does not already arrive in degrees.

        Test scenario:
            Re-projecting 4326 data to 4326 is a no-op, and both guards would pass against data that was
            never in a projected CRS at all. This pins the premise: the files on disk are in metres.
        """
        stored = {
            "points": FeatureCollection.read_file(str(POINTS)).epsg,
            "raster": Dataset.read_file(str(RASTER)).epsg,
        }
        assert stored == {"points": STORED_EPSG, "raster": STORED_EPSG}, (
            f"the fixtures are no longer stored in EPSG:{STORED_EPSG}: {stored}"
        )
