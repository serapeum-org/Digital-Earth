"""``Map.hexbin`` — points aggregated onto a hexagonal lattice (ST-22).

cleopatra 0.38's ``HexbinGlyph`` bins a point cloud and colours each cell by a count, or by the ``reduce`` of
a per-point value. The static tier wires it the way it wires its other point aggregate, ``quadtree``: the
points are reprojected through pyramids and binned in the display CRS, and the filled cells are a
``choropleth`` layer coloured by the aggregate. Expected aggregates are computed from the inputs in the
test.
"""

import geopandas as gpd
import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from pyramids.feature import FeatureCollection
from shapely.geometry import LineString, Point

from digitalearth.static import Map, projections

#: Three coincident observations at the origin and one far away, so the origin's bin aggregates three.
XY = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (8.0, 8.0)]
DEPTH = [2.0, 4.0, 6.0, 100.0]


@pytest.fixture
def wells() -> FeatureCollection:
    """Four wells in lon/lat with a depth column.

    Returns:
        A pyramids `FeatureCollection`.
    """
    gdf = gpd.GeoDataFrame(
        {"depth": DEPTH}, geometry=[Point(x, y) for x, y in XY], crs="EPSG:4326"
    )
    return FeatureCollection(gdf)


@pytest.fixture
def cloud() -> FeatureCollection:
    """Five hundred scattered points in lon/lat, with a value column.

    Returns:
        A pyramids `FeatureCollection`.
    """
    rng = np.random.default_rng(7)
    x, y = rng.uniform(0, 10, 500), rng.uniform(40, 50, 500)
    gdf = gpd.GeoDataFrame(
        {"v": rng.uniform(0, 1, 500)},
        geometry=[Point(a, b) for a, b in zip(x, y)],
        crs="EPSG:4326",
    )
    return FeatureCollection(gdf)


class TestCounting:
    """With no column, each cell is coloured by how many points fell in it."""

    def test_it_returns_the_cells(self, wells):
        """The public return is the drawn ``PolyCollection``.

        Args:
            wells: The wells.
        """
        artist = Map(crs=4326).hexbin(wells, gridsize=4)
        assert isinstance(artist, PolyCollection), type(artist)

    def test_every_point_is_counted_once(self, cloud):
        """The cell counts add up to the number of points.

        Args:
            cloud: The scattered points.
        """
        artist = Map(crs=4326).hexbin(cloud, gridsize=10)
        assert int(np.asarray(artist.get_array()).sum()) == 500, artist.get_array()

    def test_coincident_points_share_a_cell(self, wells):
        """The three wells at the origin are one cell holding three.

        Args:
            wells: The wells.
        """
        artist = Map(crs=4326).hexbin(wells, gridsize=4)
        assert sorted(np.asarray(artist.get_array()).tolist()) == [1.0, 3.0]

    def test_an_empty_cell_is_not_drawn(self, cloud):
        """A count map blanks the cells no point fell in rather than tinting the whole window with zeros.

        Args:
            cloud: The scattered points.
        """
        artist = Map(crs=4326).hexbin(cloud, gridsize=40)
        assert np.asarray(artist.get_array()).min() >= 1, artist.get_array()

    def test_min_count_drops_sparse_cells(self, wells):
        """``min_count=2`` leaves out the far well's single-point cell.

        Args:
            wells: The wells.
        """
        artist = Map(crs=4326).hexbin(wells, gridsize=4, min_count=2)
        assert np.asarray(artist.get_array()).tolist() == [3.0], artist.get_array()

    def test_a_finer_lattice_has_more_cells(self, cloud):
        """``gridsize`` reaches the lattice.

        Args:
            cloud: The scattered points.
        """
        coarse = Map(crs=4326).hexbin(cloud, gridsize=5)
        fine = Map(crs=4326).hexbin(cloud, gridsize=30)
        # One value per drawn cell: hexbin draws one hexagon path at many offsets, so paths do not count cells.
        assert len(fine.get_array()) > len(coarse.get_array()), (
            len(coarse.get_array()),
            len(fine.get_array()),
        )


class TestAggregating:
    """With a column, each cell is coloured by the ``reduce`` of its points' values."""

    @pytest.mark.parametrize(
        "reduce, origin",
        [("mean", 4.0), ("sum", 12.0), ("max", 6.0), ("count", 3.0)],
    )
    def test_the_origin_cell_holds_the_reduced_value(self, wells, reduce, origin):
        """The origin's three depths, 2, 4 and 6, reduce to the expected aggregate.

        Args:
            wells: The wells.
            reduce: The named reducer.
            origin: Its aggregate of 2, 4 and 6.
        """
        artist = Map(crs=4326).hexbin(wells, "depth", reduce=reduce, gridsize=4)
        assert origin in np.asarray(artist.get_array()).tolist(), artist.get_array()

    def test_a_reducer_of_the_callers_own_is_used(self, wells):
        """A callable reduces too; the range (``numpy.ptp``) of 2, 4 and 6 is 4.

        Args:
            wells: The wells.
        """
        artist = Map(crs=4326).hexbin(wells, "depth", reduce=np.ptp, gridsize=4)
        assert 4.0 in np.asarray(artist.get_array()).tolist(), artist.get_array()

    def test_a_scheme_classifies_the_cells(self, cloud):
        """``scheme=``/``k=`` cut the aggregate into classes, as on the other builders.

        Args:
            cloud: The scattered points.
        """
        from matplotlib.colors import BoundaryNorm

        artist = Map(crs=4326).hexbin(cloud, "v", gridsize=8, scheme="quantiles", k=4)
        assert isinstance(artist.norm, BoundaryNorm), type(artist.norm)


class TestTheDescription:
    """The layer is a described, addressable choropleth of the aggregate."""

    def test_the_layer_is_a_choropleth_drawn_by_hexbin(self, wells):
        """The kind is the registered ``choropleth``; ``via`` says how it was drawn.

        Args:
            wells: The wells.
        """
        canvas = Map(crs=4326)
        canvas.hexbin(wells, "depth", reduce="sum", gridsize=4, name="h")
        layer = canvas.figure_spec.layers.get("h")
        props = layer.symbology.props
        assert (layer.kind, props["via"]) == ("choropleth", "hexbin"), (
            layer.kind,
            props,
        )
        assert (props["column"], props["reduce"], props["gridsize"]) == (
            "depth",
            "sum",
            4,
        ), props

    def test_a_callable_reducer_is_held_rather_than_described(self, wells):
        """A function has no spelling in a figure, so the description names no reducer.

        Args:
            wells: The wells.
        """
        canvas = Map(crs=4326)
        canvas.hexbin(wells, "depth", reduce=np.ptp, gridsize=4, name="h")
        assert canvas.figure_spec.layers.get("h").symbology.props["reduce"] is None

    def test_a_count_map_is_coloured_by_the_count(self, wells):
        """The colour encoding names the quantity: ``count`` without a column, the column with one.

        Args:
            wells: The wells.
        """
        canvas = Map(crs=4326)
        canvas.hexbin(wells, gridsize=4, name="n")
        canvas.hexbin(wells, "depth", gridsize=4, name="d")
        fields = [
            canvas.figure_spec.layers.get(i).symbology.encoding("color").field
            for i in ("n", "d")
        ]
        assert fields == ["count", "depth"], fields

    def test_non_point_geometry_is_refused(self):
        """Lines are not points; nothing is described."""
        lines = FeatureCollection(
            gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:4326")
        )
        canvas = Map(crs=4326)
        with pytest.raises(ValueError, match="point"):
            canvas.hexbin(lines)
        assert canvas.layer_ids == [], canvas.layer_ids

    def test_points_behind_the_globe_are_skipped(self, wells):
        """A globe that cannot show the wells skips the layer, as every vector builder does.

        Args:
            wells: The wells.
        """
        canvas = Map(crs=projections.orthographic(lon=180, lat=0), globe=True)
        assert canvas.hexbin(wells) is None
        assert canvas.layer_ids == [], canvas.layer_ids
