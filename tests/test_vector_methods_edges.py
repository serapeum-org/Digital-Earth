"""Gap-filling tests for the new Map vector plots.

Complements the per-plot test files (``test_voronoi``/``test_cartogram``/``test_quadtree``/``test_kde``/
``test_sankey``/``test_scale_legend``/``test_scheme``) with the scenarios they did not cover: quadtree aggregation
variants + ``nmin``, Multi-geometry expansion (cartogram/sankey), empty-FeatureCollection and clipped/globe-CRS
edge cases, the value→size alignment of ``scatter(scale=...)``, and direct unit tests of the private helpers.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, box

from digitalearth.scene import Map

ORTHO = "+proj=ortho +lat_0=0 +lon_0=0 +datum=WGS84"  # clips the far hemisphere -> non-finite coords


def _fc(gdf):
    """Wrap a GeoDataFrame in a pyramids FeatureCollection."""
    from pyramids.feature import FeatureCollection

    return FeatureCollection(gdf)


@pytest.fixture
def value_points():
    """Four points with known values [1, 2, 3, 4] in a projected CRS (EPSG:32618)."""
    pts = [Point(0, 0), Point(10, 0), Point(0, 10), Point(10, 10)]
    return _fc(gpd.GeoDataFrame({"v": [1.0, 2.0, 3.0, 4.0]}, geometry=pts, crs="EPSG:32618"))


@pytest.fixture
def far_side_points():
    """Points where two reproject to non-finite coords under an orthographic centred at (0, 0)."""
    pts = [Point(0, 0), Point(30, 10), Point(180, 0), Point(150, -20)]  # 180/150 lon = far side
    return _fc(gpd.GeoDataFrame({"v": [1.0, 2.0, 3.0, 4.0]}, geometry=pts, crs="EPSG:4326"))


def _empty_points():
    return _fc(gpd.GeoDataFrame({"v": []}, geometry=[], crs="EPSG:32618"))


class TestQuadtreeAggregation:
    """Map.quadtree per-cell reducer selection (a single cell with nmax>=n isolates the reducer)."""

    @pytest.mark.parametrize(
        "agg, expected",
        [("sum", 10.0), ("mean", 2.5), ("min", 1.0), ("max", 4.0), ("median", 2.5), ("count", 4.0)],
    )
    def test_named_agg(self, value_points, agg, expected):
        """A single cell carries ``agg`` applied to all four values.

        Args:
            agg: Reducer name under test.
            expected: The reducer applied to ``[1, 2, 3, 4]``.

        Test scenario:
            ``nmax=100`` keeps the whole bbox as one cell, so ``pc.get_array()[0]`` equals the reducer over
            every value — isolating the per-cell aggregation from the spatial split.
        """
        pc = Map(crs=value_points.epsg).quadtree(value_points, column="v", agg=agg, nmax=100)
        arr = np.asarray(pc.get_array())
        assert arr.size == 1, f"expected one cell, got {arr.size}"
        assert float(arr[0]) == pytest.approx(expected), f"{agg} -> {arr[0]}, expected {expected}"

    def test_callable_agg(self, value_points):
        """A callable ``agg`` is applied to the per-cell value array (here counting elements)."""
        pc = Map(crs=value_points.epsg).quadtree(value_points, column="v", agg=lambda a: a.size, nmax=100)
        assert float(np.asarray(pc.get_array())[0]) == pytest.approx(4.0)

    def test_density_count_without_column(self, value_points):
        """``column=None`` colours each cell by its point count."""
        pc = Map(crs=value_points.epsg).quadtree(value_points, nmax=100)
        assert float(np.asarray(pc.get_array())[0]) == pytest.approx(4.0)


class TestQuadtreeCells:
    """Map._quadtree_cells partitioning invariants (pure geometry, no rendering)."""

    def test_respects_nmax(self):
        """No emitted cell exceeds nmax points on splittable (non-coincident) input."""
        rng = np.random.default_rng(1337)
        xs, ys = rng.random(200), rng.random(200)
        cells = Map._quadtree_cells(xs, ys, agg_fn=lambda idx: float(len(idx)), nmax=10, nmin=0)
        assert cells, "expected at least one cell"
        assert all(val <= 10 for *_b, val in cells), "a cell exceeded nmax"

    def test_nmin_drops_sparse_cells(self):
        """A lone far point (count 1) is dropped when nmin=2, the dense cluster (count 4) is kept."""
        xs = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
        ys = np.zeros(5)
        kept = Map._quadtree_cells(xs, ys, agg_fn=lambda idx: float(len(idx)), nmax=2, nmin=2)
        assert sorted(c[4] for c in kept) == [4.0], f"unexpected kept counts: {[c[4] for c in kept]}"

    def test_coincident_points_terminate(self):
        """All-coincident points cannot recurse forever — one cell is emitted with the full count."""
        xs = np.zeros(5)
        ys = np.zeros(5)
        kept = Map._quadtree_cells(xs, ys, agg_fn=lambda idx: float(len(idx)), nmax=1, nmin=0)
        assert len(kept) == 1 and kept[0][4] == 5.0, f"expected one count-5 cell, got {kept}"


class TestCartogramMultiPolygon:
    """Map.cartogram with MultiPolygon input expands to one drawn polygon per part."""

    def test_multipolygon_expands(self):
        """A single MultiPolygon feature (2 parts) yields 2 filled polygons sharing the row's value."""
        mp = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])
        fc = _fc(gpd.GeoDataFrame({"v": [5.0]}, geometry=[mp], crs="EPSG:32618"))
        pc = Map(crs=fc.epsg).cartogram(fc, scale="v", column="v")
        assert len(pc.get_paths()) == 2, f"expected 2 parts, got {len(pc.get_paths())}"


class TestSankeyMultiLineString:
    """Map.sankey with MultiLineString input expands to one drawn path per part."""

    def test_multilinestring_expands(self):
        """A single MultiLineString feature (2 parts) yields 2 line segments."""
        mls = MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])
        fc = _fc(gpd.GeoDataFrame({"flow": [1.0], "w": [2.0]}, geometry=[mls], crs="EPSG:32618"))
        lc = Map(crs=fc.epsg).sankey(fc, column="flow", scale="w")
        assert len(lc.get_segments()) == 2, f"expected 2 segments, got {len(lc.get_segments())}"


class TestScatterScaleAlignment:
    """Map.scatter(scale=...) maps marker size to the right point (positional alignment)."""

    def test_sizes_align_with_scale_column(self):
        """Per-point marker areas rank-match the scale column in row order."""
        pts = [Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0)]
        s = [4.0, 1.0, 3.0, 2.0]
        fc = _fc(gpd.GeoDataFrame({"s": s}, geometry=pts, crs="EPSG:32618"))
        pc = Map(crs=fc.epsg).scatter(fc, scale="s", size_limits=(10, 200))
        sizes = np.asarray(pc.get_sizes())
        assert np.argsort(sizes).tolist() == np.argsort(s).tolist(), "sizes not aligned to scale column order"


class TestEmptyFeatureCollection:
    """Every new Map method raises a clear ValueError on an empty FeatureCollection (review L1)."""

    @pytest.mark.parametrize("method, kwargs", [
        ("voronoi", {}),
        ("quadtree", {}),
        ("kde", {}),
    ])
    def test_point_methods_reject_empty(self, method, kwargs):
        """Point-input methods reject an empty collection up front."""
        with pytest.raises(ValueError, match="empty"):
            getattr(Map(crs=32618), method)(_empty_points(), **kwargs)

    def test_cartogram_rejects_empty(self):
        """cartogram rejects an empty polygon collection."""
        empty = _fc(gpd.GeoDataFrame({"v": []}, geometry=[], crs="EPSG:32618"))
        with pytest.raises(ValueError, match="empty"):
            Map(crs=32618).cartogram(empty, scale="v")

    def test_sankey_rejects_empty(self):
        """sankey rejects an empty line collection."""
        empty = _fc(gpd.GeoDataFrame({"w": []}, geometry=[], crs="EPSG:32618"))
        with pytest.raises(ValueError, match="empty"):
            Map(crs=32618).sankey(empty, scale="w")


class TestGlobeNonFinite:
    """Point methods drop non-finite reprojected coords on a clipped/globe CRS instead of crashing (M1)."""

    def test_kde_drops_far_side(self, far_side_points):
        """kde renders from the near-side points only, no GEOS/ValueError."""
        m = Map(crs=ORTHO)
        m.kde(far_side_points)
        assert len(m.layers) == 1

    def test_quadtree_drops_far_side(self, far_side_points):
        """quadtree bins the near-side points only, no inf bbox."""
        m = Map(crs=ORTHO)
        pc = m.quadtree(far_side_points, column="v", nmax=1)
        assert len(pc.get_paths()) >= 1

    @pytest.mark.parametrize("method, kwargs", [
        ("voronoi", {"column": "v"}),
        ("quadtree", {"column": "v"}),
        ("kde", {}),
    ])
    def test_all_far_side_raises(self, method, kwargs):
        """If every point is on the far side (all non-finite), a clear error is raised, not an opaque GEOS one."""
        pts = [Point(180, 0), Point(170, -10)]  # both far side of ortho@(0,0)
        fc = _fc(gpd.GeoDataFrame({"v": [1.0, 2.0]}, geometry=pts, crs="EPSG:4326"))
        with pytest.raises(ValueError, match="finite"):
            getattr(Map(crs=ORTHO), method)(fc, **kwargs)


class TestApiWrappersNoColumn:
    """api.* wrappers take the no-colorbar branch when ``column`` is omitted (still produce a layer)."""

    def test_api_voronoi_no_column(self, value_points):
        """api.voronoi without a column draws outline cells and skips the colorbar."""
        from digitalearth import api

        m = api.voronoi(value_points, crs=value_points.epsg)
        assert len(m.layers) == 1, "expected one voronoi layer"

    def test_api_cartogram_no_column(self):
        """api.cartogram without a column draws scaled outlines and skips the colorbar."""
        from digitalearth import api

        polys = _fc(
            gpd.GeoDataFrame({"v": [1.0, 2.0]}, geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)], crs="EPSG:32618")
        )
        m = api.cartogram(polys, scale="v", crs=polys.epsg)
        assert len(m.layers) == 1, "expected one cartogram layer"

    def test_api_sankey_no_column(self):
        """api.sankey without a column draws uniform-colour flows and skips the colorbar."""
        from digitalearth import api

        lines = _fc(
            gpd.GeoDataFrame(
                {"w": [1.0, 2.0]},
                geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
                crs="EPSG:32618",
            )
        )
        m = api.sankey(lines, scale="w", crs=lines.epsg)
        assert len(m.layers) == 1, "expected one sankey layer"


class TestVectorHelpers:
    """Direct unit tests for the private helpers backing the new methods."""

    def test_polygons_of(self):
        """_polygons_of returns Polygon parts and ignores non-polygonal/None input."""
        assert Map._polygons_of(None) == []
        assert Map._polygons_of(Point(0, 0)) == []
        poly = box(0, 0, 1, 1)
        assert Map._polygons_of(poly) == [poly]
        mp = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])
        assert len(Map._polygons_of(mp)) == 2

    def test_clip_geometry_none(self):
        """_clip_geometry passes ``None`` through unchanged."""
        assert Map(crs=32618)._clip_geometry(None) is None

    def test_clip_geometry_shapely_passthrough(self):
        """A shapely geometry is assumed to be in the display CRS and returned as-is."""
        g = box(0, 0, 1, 1)
        assert Map(crs=32618)._clip_geometry(g) is g

    def test_clip_geometry_gdf_unioned(self):
        """A GeoDataFrame is reprojected and unioned into a single geometry."""
        gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)], crs="EPSG:32618")
        out = Map(crs=32618)._clip_geometry(gdf)
        assert out.geom_type in ("Polygon", "MultiPolygon", "GeometryCollection")

    def test_clip_path_from_polygon(self):
        """_clip_path turns a polygon boundary into a non-empty matplotlib Path."""
        from matplotlib.path import Path as MplPath

        path = Map(crs=32618)._clip_path(box(0, 0, 1, 1))
        assert isinstance(path, MplPath)
        assert len(path.vertices) >= 4

    def test_clip_path_none(self):
        """_clip_path returns ``None`` when there is no boundary."""
        assert Map(crs=32618)._clip_path(None) is None

    def test_finite_point_xy_drops_nonfinite(self):
        """_finite_point_xy drops non-finite points and the matching values together."""
        gdf = gpd.GeoDataFrame(
            {"v": [1.0, 2.0, 3.0]},
            geometry=[Point(0, 0), Point(np.inf, 0), Point(2, 2)],
            crs="EPSG:32618",
        )
        xs, ys, vals = Map._finite_point_xy(gdf.geometry, np.array([1.0, 2.0, 3.0]))
        assert xs.tolist() == [0.0, 2.0]
        assert ys.tolist() == [0.0, 2.0]
        assert vals.tolist() == [1.0, 3.0]

    @pytest.mark.parametrize("limits, expected_first, expected_last", [((0.2, 1.0), 0.2, 1.0)])
    def test_scale_factors_endpoints(self, limits, expected_first, expected_last):
        """_scale_factors maps the value range onto the limit endpoints."""
        f = Map._scale_factors(np.array([0.0, 5.0, 10.0]), limits)
        assert f[0] == pytest.approx(expected_first)
        assert f[-1] == pytest.approx(expected_last)
