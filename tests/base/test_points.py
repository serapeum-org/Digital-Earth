"""`PointArrays` — one geometry-to-arrays path instead of thirteen (DE-18, #278).

22 extraction lines across 9 files turned geometry into coordinate arrays, in three different spellings, with
inconsistent dtype handling and inconsistent treatment of non-finite coordinates. These cover the type that
replaces them, seeded from the one tier that had already built it locally.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from digitalearth.base.points import PointArrays


class TestBuilding:
    """The coordinate read, and the dtype every call site used to pick for itself."""

    def test_coordinates_are_always_float64(self):
        """Integer input comes back as floats, whatever the caller passed.

        Test scenario:
            The copies disagreed here: some forced `dtype=float`, some `dtype="float64"`, some forced
            nothing. A site that got an integer array and passed it to a reprojection got integer division
            behaviour it never asked for.
        """
        points = PointArrays.of([0, 1, 2], [3, 4, 5])
        assert points.x.dtype == np.float64, f"x must be float64, got {points.x.dtype}"
        assert points.y.dtype == np.float64, f"y must be float64, got {points.y.dtype}"

    def test_z_defaults_to_zeros_rather_than_being_absent(self):
        """A 2-D input still has a z array, so a 3-D consumer needs no branch.

        Test scenario:
            The 3-D tier wrote `np.zeros_like(x)` itself when the geometry had no z. Making z always present
            is what lets `as_columns()` be unconditional.
        """
        points = PointArrays.of([0.0, 1.0], [2.0, 3.0])
        assert points.z.tolist() == [0.0, 0.0], "z must default to zeros"

    def test_misaligned_arrays_are_refused(self):
        """x, y and z must be the same length.

        Test scenario:
            They are index-aligned with each other and with any attribute column filtered alongside them. A
            mismatch draws points at another point's coordinates, silently.
        """
        with pytest.raises(ValueError, match="the same length"):
            PointArrays(np.array([0.0, 1.0]), np.array([0.0]), np.array([0.0]))

    def test_the_length_is_the_point_count(self):
        """`len()` answers how many points there are.

        Test scenario:
            Call sites check emptiness before tessellating or binning; `len(points)` should be the obvious
            way rather than reaching into `.x`.
        """
        assert len(PointArrays.of([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])) == 3


class TestFromFeatures:
    """Reading a pyramids collection or a geopandas frame."""

    def test_point_geometry_is_read_directly(self):
        """The common case: point geometry becomes x/y arrays.

        Test scenario:
            What the majority of the replaced sites did, in four different spellings.
        """
        gdf = gpd.GeoDataFrame(geometry=[Point(0, 1), Point(2, 3)], crs="EPSG:4326")
        points = PointArrays.from_features(gdf)
        assert points.x.tolist() == [0.0, 2.0], "x must come from the geometry"
        assert points.y.tolist() == [1.0, 3.0], "and y with it"

    def test_three_dimensional_geometry_keeps_its_z(self):
        """A 3-D geometry's z is read rather than zeroed.

        Test scenario:
            The 3-D tier checked `geom.has_z.all()` itself to decide. Getting that wrong flattens a point
            cloud onto a plane, which renders — just wrongly.
        """
        gdf = gpd.GeoDataFrame(
            geometry=[Point(0, 1, 5), Point(2, 3, 7)], crs="EPSG:4326"
        )
        assert PointArrays.from_features(gdf).z.tolist() == [5.0, 7.0], (
            "a 3-D geometry's z must survive"
        )

    def test_non_point_geometry_falls_back_to_centroids(self):
        """A polygon collection is read at its centres.

        Test scenario:
            `base/sources/extractors.py` did this and most other sites did not — so the same collection
            produced coordinates on one path and an exception on another.
        """
        square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        gdf = gpd.GeoDataFrame(geometry=[square], crs="EPSG:4326")
        points = PointArrays.from_features(gdf)
        assert (points.x.tolist(), points.y.tolist()) == ([1.0], [1.0]), (
            "a polygon must be read at its centroid"
        )

    def test_the_centroid_fallback_can_be_refused(self):
        """A caller whose maths needs real points can say so.

        Test scenario:
            A centroid is a reasonable stand-in for a label anchor and a poor one for a point cloud. The
            default keeps the extractor's behaviour; `centroids=False` is for the sites that should refuse.
        """
        square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        gdf = gpd.GeoDataFrame(geometry=[square], crs="EPSG:4326")
        with pytest.raises(ValueError, match="not all points"):
            PointArrays.from_features(gdf, centroids=False)

    def test_a_non_point_geometry_is_checked_before_its_coordinates_are_touched(self):
        """The geometry-type check comes first, because reading `.x` on a polygon raises.

        Test scenario:
            geopandas raises `ValueError: x attribute access only provided for Point geometries` rather than
            answering, so even `hasattr(geom, "x")` raises instead of returning False. Guarding on `.x`
            turned the centroid fallback into a crash.
        """
        gdf = gpd.GeoDataFrame(
            geometry=[Polygon([(0, 0), (1, 0), (1, 1)])], crs="EPSG:4326"
        )
        assert len(PointArrays.from_features(gdf)) == 1, (
            "a polygon must reach the centroid fallback, not raise"
        )

    def test_something_with_no_geometry_is_refused_by_name(self):
        """A non-geospatial input says what was wrong with it.

        Test scenario:
            Otherwise the failure is an AttributeError from inside the extraction, naming a column rather
            than the argument.
        """
        with pytest.raises(TypeError, match="needs vector data with a geometry"):
            PointArrays.from_features([1, 2, 3])


class TestFiniteFiltering:
    """The part that actually bites, and the reason the static tier built this first."""

    def test_non_finite_points_are_dropped(self):
        """A point that reprojected to infinity is removed.

        Test scenario:
            The far side of a clipped or globe display CRS reprojects to inf/nan. Passed on, they fail
            somewhere else entirely — inside a tessellator or matplotlib — with a message about neither the
            points nor the projection.
        """
        points, _ = PointArrays.of([0.0, float("inf"), 2.0], [1.0, 2.0, 3.0]).finite()
        assert points.x.tolist() == [0.0, 2.0], "the non-finite point must be dropped"

    def test_a_non_finite_y_or_z_drops_the_point_too(self):
        """All three coordinates are checked, not just x.

        Test scenario:
            A point with a finite x and an infinite y is just as unusable, and a z-aware consumer needs the
            third checked as well.
        """
        assert len(PointArrays.of([0.0], [float("nan")], [1.0]).finite()[0]) == 0
        assert len(PointArrays.of([0.0], [1.0], [float("inf")]).finite()[0]) == 0

    def test_aligned_arrays_are_filtered_by_the_same_mask(self):
        """A value column stays in step with the points that survive.

        Test scenario:
            This is what makes the helper worth having: filtering the points and forgetting the column
            shifts every remaining value onto the wrong point — a wrong map that still renders.
        """
        points = PointArrays.of([0.0, float("inf"), 2.0], [1.0, 2.0, 3.0])
        kept, (values,) = points.finite(np.array([10.0, 20.0, 30.0]))
        assert values.tolist() == [10.0, 30.0], (
            "the value column must follow the points"
        )
        assert len(kept) == len(values), "and stay the same length as them"

    def test_a_none_column_passes_through_as_none(self):
        """An optional column needs no branch at the call site.

        Test scenario:
            Every migrated site has an optional value column. Making the caller guard on it is how the
            copies grew their own shapes.
        """
        _, (values,) = PointArrays.of([0.0], [1.0]).finite(None)
        assert values is None, "a None column must stay None"

    def test_filtering_returns_a_new_value(self):
        """The original is unchanged.

        Test scenario:
            A caller may need both the full set and the drawable subset — the globe tier reports how many
            points it skipped.
        """
        points = PointArrays.of([0.0, float("inf")], [1.0, 2.0])
        points.finite()
        assert len(points) == 2, "the original must keep every point"


class TestShapes:
    """The two forms consumers ask for."""

    def test_as_columns_is_the_n_by_3_array_the_3d_tiers_stack(self):
        """`as_columns()` replaces a hand-written `np.column_stack`.

        Test scenario:
            Both PyVista and deck.gl want (N, 3). The 3-D tier built it by stacking three arrays it had just
            extracted separately.
        """
        stacked = PointArrays.of([0.0, 1.0], [2.0, 3.0], [4.0, 5.0]).as_columns()
        assert stacked.shape == (2, 3), f"expected (2, 3), got {stacked.shape}"
        assert stacked[1].tolist() == [1.0, 3.0, 5.0], (
            "a row must be one point's x, y, z"
        )

    def test_as_xy_lets_a_2d_consumer_ignore_z(self):
        """The many 2-D sites never have to know z exists.

        Test scenario:
            Most consumers are 2-D. Making them unpack three arrays to use two would be a worse API than the
            two lines it replaced.
        """
        x, y = PointArrays.of([0.0, 1.0], [2.0, 3.0]).as_xy()
        assert (x.tolist(), y.tolist()) == ([0.0, 1.0], [2.0, 3.0])
