"""Tests for the Map vector/polygon helpers added in PA-4: _vector_input and _polygon_layer."""
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from pyramids.feature import FeatureCollection  # noqa: E402

from digitalearth.scene import Map  # noqa: E402


@pytest.fixture
def points_fc():
    """The point fixture as a pyramids FeatureCollection (EPSG:32618, numeric 'fid' column)."""
    return FeatureCollection.read_file("tests/data/points.geojson")


@pytest.fixture
def polygons_fc(points_fc):
    """A polygon FeatureCollection: each fixture point buffered into a small projected polygon."""
    gdf = points_fc.copy()
    gdf["geometry"] = gdf.geometry.buffer(500.0)
    return FeatureCollection(gdf)


class TestVectorInput:
    """Tests for Map._vector_input."""

    def test_reprojects_to_display_crs(self, points_fc):
        """_vector_input returns the features reprojected to the Map's display CRS.

        Test scenario:
            With a display CRS different from the source, the returned GeoDataFrame reports the Map CRS.
        """
        m = Map(crs=4326)
        gdf = m._vector_input(points_fc)
        assert gdf.crs.to_epsg() == 4326, f"expected EPSG:4326, got {gdf.crs}"

    def test_no_geom_check_passes_any_geometry(self, polygons_fc):
        """_vector_input without geom_types accepts any geometry.

        Test scenario:
            Omitting geom_types skips validation, so polygons pass through unchallenged.
        """
        m = Map(crs=polygons_fc.epsg)
        gdf = m._vector_input(polygons_fc)
        assert len(gdf) == len(polygons_fc), "all features should be returned"

    def test_empty_collection_raises_with_name(self, points_fc):
        """_vector_input rejects an empty collection, naming the caller.

        Test scenario:
            An empty FeatureCollection raises ValueError containing '<name> got an empty FeatureCollection'.
        """
        empty = FeatureCollection(points_fc.iloc[0:0].copy())
        m = Map(crs=points_fc.epsg)
        with pytest.raises(ValueError, match="probe got an empty FeatureCollection") as exc:
            m._vector_input(empty, name="probe")
        assert "empty" in str(exc.value), f"unexpected message: {exc.value}"

    def test_wrong_geometry_raises_with_label(self, polygons_fc):
        """_vector_input rejects geometry outside geom_types, using the human label.

        Test scenario:
            Polygons given to a Point-only check raise '<name> requires a FeatureCollection of point geometries'.
        """
        m = Map(crs=polygons_fc.epsg)
        with pytest.raises(ValueError, match="probe requires a FeatureCollection of point geometries"):
            m._vector_input(polygons_fc, geom_types=("Point",), name="probe", geom_label="point")

    def test_matching_geometry_passes(self, points_fc):
        """_vector_input accepts geometry that matches geom_types.

        Test scenario:
            Points given to a Point-only check return the reprojected frame without raising.
        """
        m = Map(crs=points_fc.epsg)
        gdf = m._vector_input(points_fc, geom_types=("Point",), name="probe", geom_label="point")
        assert (gdf.geometry.geom_type == "Point").all(), "all geometries should be points"

    def test_label_defaults_to_joined_types(self, points_fc):
        """_vector_input falls back to the joined geom_types when geom_label is omitted.

        Test scenario:
            Without geom_label, the error lists the allowed types joined by ' / '.
        """
        m = Map(crs=points_fc.epsg)
        with pytest.raises(ValueError, match="LineString / MultiLineString"):
            m._vector_input(points_fc, geom_types=("LineString", "MultiLineString"), name="probe")


class TestExtentOf:
    """Tests for Map._extent_of (PA-9)."""

    def test_bbox_order_from_arrays(self):
        """_extent_of returns [xmin, ymin, xmax, ymax] from 1-D coordinate arrays.

        Test scenario:
            Unsorted x/y arrays still yield the correct min/max bbox in cleopatra order.
        """
        x = np.array([2.0, 0.0, 1.0])
        y = np.array([5.0, 9.0, 7.0])
        assert Map._extent_of(x, y) == [0.0, 5.0, 2.0, 9.0], "bbox order/values incorrect"

    def test_extent_delegates_to_extent_of(self):
        """_extent(ds) matches _extent_of(ds.x, ds.y).

        Test scenario:
            The dataset-based _extent is just _extent_of over the dataset's coordinate vectors.
        """
        from pyramids.dataset import Dataset

        ds = Dataset.read_file("examples/data/acc4000.tif")
        m = Map(crs=ds.epsg)
        assert m._extent(ds) == Map._extent_of(ds.x, ds.y), "_extent should delegate to _extent_of"


class TestPolygonLayer:
    """Tests for Map._polygon_layer."""

    @pytest.fixture
    def squares(self):
        """Two unit-square polygon rings as (N, 2) vertex arrays."""
        a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        b = a + 2.0
        return [a, b]

    def test_fill_mode_registers_one_layer(self, squares):
        """_polygon_layer with values draws a filled layer and registers exactly one Scene layer.

        Test scenario:
            Passing per-polygon values produces one PolyCollection layer with all paths drawn.
        """
        m = Map(crs=4326)
        pc = m._polygon_layer(squares, np.array([1.0, 2.0]))
        assert len(m.layers) == 1, f"expected one layer, got {len(m.layers)}"
        assert len(pc.get_paths()) == 2, f"expected 2 paths, got {len(pc.get_paths())}"

    def test_outline_mode_registers_one_layer(self, squares):
        """_polygon_layer without values draws outline-only and registers exactly one Scene layer.

        Test scenario:
            Omitting values produces a single outline PolyCollection layer.
        """
        m = Map(crs=4326)
        pc = m._polygon_layer(squares)
        assert len(m.layers) == 1, f"expected one layer, got {len(m.layers)}"
        assert len(pc.get_paths()) == 2, f"expected 2 paths, got {len(pc.get_paths())}"

    def test_respects_explicit_add_colorbar(self, squares):
        """_polygon_layer honours an explicit add_colorbar instead of forcing the default.

        Test scenario:
            Passing add_colorbar=True is forwarded to the glyph (a colorbar axes is added to the figure).
        """
        m = Map(crs=4326)
        before = len(m.fig.axes)
        m._polygon_layer(squares, np.array([1.0, 2.0]), add_colorbar=True)
        assert len(m.fig.axes) > before, "explicit add_colorbar=True should add a colorbar axes"
