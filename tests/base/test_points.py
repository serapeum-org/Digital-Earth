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
        gdf = gpd.GeoDataFrame(geometry=[square], crs="EPSG:3857")
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
        gdf = gpd.GeoDataFrame(geometry=[square], crs="EPSG:3857")
        with pytest.raises(ValueError, match="not all points"):
            PointArrays.from_features(gdf, centroids=False)

    @pytest.mark.parametrize("centroids", [True, False])
    def test_a_missing_geometry_reads_as_nan_rather_than_raising(self, centroids):
        """A `null` geometry among points is absent data, not data of the wrong shape.

        Test scenario:
            `main` drew such a frame: geopandas answers `NaN` from `.x` for a null geometry and the finite
            mask dropped the row. Testing the whole series with `(geom_type == "Point").all()` classified it
            as "not all points" — a null's `geom_type` is `NaN` — so `centroids=False` refused a frame that
            had been rendering, at `InteractiveMap.trimesh` and at the 3-D point cloud. A GeoJSON feature
            with a `null` geometry is ordinary, so this is reachable straight from user data.

            Both values of `centroids` are asserted because the refusal is what broke, and the fallback runs
            the same classification one line above it.
        """
        gdf = gpd.GeoDataFrame(
            geometry=[Point(0, 0), None, Point(2, 2)], crs="EPSG:3857"
        )
        points = PointArrays.from_features(gdf, centroids=centroids)
        assert len(points) == 3, "the missing row stays, so the arrays still align with the frame"
        assert np.isnan(points.x[1]), "and reads as NaN, the way geopandas answers for it"
        kept, _ = points.finite()
        assert len(kept) == 2, "leaving `finite` to drop it, as every caller does"

    def test_a_real_non_point_geometry_is_still_refused_when_centroids_is_off(self):
        """Relaxing the null case does not relax the case the guard is for.

        Test scenario:
            The fix narrows the classification to geometry that is present; a polygon sitting beside a null
            must still be refused, or the guard has been removed rather than corrected.
        """
        gdf = gpd.GeoDataFrame(
            geometry=[Point(0, 0), None, Polygon([(0, 0), (1, 0), (1, 1)])],
            crs="EPSG:3857",
        )
        with pytest.raises(ValueError, match="not all points"):
            PointArrays.from_features(gdf, centroids=False)

    @pytest.mark.parametrize("centroids", [True, False])
    def test_a_frame_with_no_geometry_present_reads_as_all_nan(self, centroids):
        """A frame whose every geometry is `null` reads as NaN throughout, rather than raising.

        Args:
            centroids: The fallback setting, which must not matter when nothing is present.

        Test scenario:
            The boundary of judging only the geometry that is there: with none present the classification runs
            over an empty series, which is vacuously all points, so neither setting refuses the frame and
            `finite` is left with nothing to draw.
        """
        gdf = gpd.GeoDataFrame(geometry=[None, None], crs="EPSG:3857")
        points = PointArrays.from_features(gdf, centroids=centroids)
        assert len(points) == 2, "every row is kept, so the arrays still align with the frame"
        assert np.isnan(points.x).all() and np.isnan(points.y).all(), f"got x={points.x}, y={points.y}"
        assert len(points.finite()[0]) == 0, "and no point survives the finite mask"

    def test_a_non_point_geometry_is_checked_before_its_coordinates_are_touched(self):
        """The geometry-type check comes first, because reading `.x` on a polygon raises.

        Test scenario:
            geopandas raises `ValueError: x attribute access only provided for Point geometries` rather than
            answering, so even `hasattr(geom, "x")` raises instead of returning False. Guarding on `.x`
            turned the centroid fallback into a crash.
        """
        gdf = gpd.GeoDataFrame(
            geometry=[Polygon([(0, 0), (1, 0), (1, 1)])], crs="EPSG:3857"
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

    def test_a_non_finite_y_drops_the_point_too(self):
        """Both position coordinates are checked, not just x.

        Test scenario:
            A point with a finite x and an infinite y is just as unplaceable. `z` is deliberately *not* in
            the default mask — see `test_an_unknown_elevation_does_not_drop_a_drawable_2d_point`, which is
            the behaviour this test used to contradict.
        """
        assert len(PointArrays.of([0.0], [float("nan")], [1.0]).finite()[0]) == 0
        assert len(PointArrays.of([0.0], [1.0], [float("inf")]).finite()[0]) == 1, (
            "an unusable elevation must not remove a point a 2-D consumer can place"
        )

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

    def test_an_unknown_elevation_does_not_drop_a_drawable_2d_point(self):
        """A NaN z must not remove a point whose x and y are perfectly good.

        Test scenario:
            The replaced code masked on x and y only — z was never read at the interactive tessellation site.
            Reading z in `from_features` and folding it into the mask silently changed the topology of a 2-D
            Delaunay mesh: a 4-node mesh became 3 nodes because one point's elevation was unknown. 3-D point
            geometry is ordinary in GeoJSON, so this is reachable from `InteractiveMap.trimesh`.
        """
        points = PointArrays.of([0.0, 1.0], [2.0, 3.0], [9.0, float("nan")])
        assert len(points.finite()[0]) == 2, "a 2-D consumer must keep both points"

    def test_a_three_dimensional_consumer_can_ask_for_z_to_count(self):
        """Filtering on z is available, just not the default.

        Test scenario:
            A point cloud genuinely cannot place a point with no elevation. The distinction is the caller's
            to make, which is why it is a parameter rather than a fixed rule.
        """
        points = PointArrays.of([0.0, 1.0], [2.0, 3.0], [9.0, float("nan")])
        assert len(points.finite(dims="xyz")[0]) == 1, (
            "opting in must drop the NaN-z point"
        )

    def test_a_dims_string_naming_something_else_is_refused(self):
        """A typo'd axis name is caught rather than silently filtering nothing.

        Test scenario:
            `dims="xu"` would otherwise mask on x alone and quietly ignore the rest, which reads as working.
        """
        with pytest.raises(ValueError, match="names"):
            PointArrays.of([0.0], [1.0]).finite(dims="xu")

    @pytest.mark.parametrize("dims", ["", "xx", "yxy"])
    def test_an_empty_or_repeating_dims_string_is_refused(self, dims):
        """`dims=""` filters nothing and `dims="xx"` names an axis twice; both are typos, not requests.

        Args:
            dims: The malformed axis string under test.

        Test scenario:
            The guard refused only letters outside x/y/z. An empty string produced an all-True mask, so every
            point survived — the silently-working typo the guard exists to stop — and a repeated letter
            passed where the caller almost certainly meant a different axis.
        """
        with pytest.raises(ValueError, match="at most once, and at least one"):
            PointArrays.of([0.0, float("nan")], [1.0, 2.0]).finite(dims=dims)

    @pytest.mark.parametrize("dims, kept_x", [("y", [1.0, 2.0]), ("zx", [0.0, 2.0]), ("zyx", [2.0])])
    def test_a_single_or_reordered_dims_string_is_accepted(self, dims, kept_x):
        """Any non-empty, non-repeating choice of axes filters on exactly those axes, in any order.

        Args:
            dims: The axis string under test.
            kept_x: The x of the points expected to survive.

        Test scenario:
            The guard for `""` and repeated letters must refuse only those. One that compared against the two
            documented spellings, `"xy"` and `"xyz"`, would pass every refusal test and still reject a caller
            filtering on `y` alone or naming the axes in another order.
        """
        points = PointArrays.of([0.0, 1.0, 2.0], [float("nan"), 1.0, 2.0], [2.0, float("nan"), 2.0])
        kept, _ = points.finite(dims=dims)
        assert kept.x.tolist() == kept_x, f"dims={dims!r} must keep x={kept_x}, got {kept.x.tolist()}"

    def test_two_equal_readings_compare_equal(self):
        """Comparison works at all, which it did not before round 1.

        Test scenario:
            The dataclass-generated `__eq__` compared the arrays with `==`, yielding an array, so any
            instance holding more than one point raised "The truth value of an array with more than one
            element is ambiguous" on comparison.
        """
        assert PointArrays.of([0.0, 1.0], [2.0, 3.0]) == PointArrays.of(
            [0.0, 1.0], [2.0, 3.0]
        )

    def test_different_points_do_not_compare_equal(self):
        """The comparison is a real one, not a constant True.

        Test scenario:
            Guards the boundary of the test above — an `__eq__` that always answered True would satisfy it.
        """
        assert PointArrays.of([0.0], [1.0]) != PointArrays.of([9.0], [1.0])

    def test_comparing_with_something_else_defers(self):
        """A non-PointArrays gets `NotImplemented`, so Python can try the other side.

        Test scenario:
            Returning False outright would stop a future type from defining equality with this one.
        """
        assert PointArrays.of([0.0], [1.0]) != "not points"

    def test_points_are_deliberately_unhashable(self):
        """Bulk data does not key a cache, unlike `Bounds` and `Selection`.

        Test scenario:
            Stated rather than accidental: the other value types in this refactor hash on purpose, so the
            difference needs to be a decision the tests record.
        """
        with pytest.raises(TypeError):
            hash(PointArrays.of([0.0], [1.0]))


class TestTheReadOnlyGuaranteeReachesSource:
    """What `PointArrays`' frozen arrays mean for every vector `Source`."""

    def test_a_vector_source_hands_out_read_only_coordinates(self):
        """`Source.x.values` for a FeatureCollection refuses an in-place write.

        Test scenario:
            `_from_feature` passes `PointArrays`' frozen arrays straight into the `Source`. Under pandas 3 the
            geometry's `.x` was already read-only through copy-on-write; under pandas 2, which `pyproject.toml`
            still admits, it was writable, so the guarantee is now unconditional. `Source.x` documents it, and
            this pins it so a change to either side is noticed.
        """
        from pyramids.feature import FeatureCollection

        from digitalearth.base.sources import get_source

        gdf = gpd.GeoDataFrame({"v": [1.0, 2.0]}, geometry=[Point(0, 1), Point(2, 3)], crs="EPSG:4326")
        source = get_source(FeatureCollection(gdf))
        with pytest.raises(ValueError, match="read-only"):
            source.x.values[0] = 9.0
        assert np.array(source.x.values).flags.writeable, "and the documented copy is writable"


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
