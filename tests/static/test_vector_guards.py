"""Guards on the vector builders: a helpful missing-column error (#379) and a refused colour norm (#380).

The fixtures build pyramids ``FeatureCollection``s from geopandas/shapely input (the vector shape the suite
already uses), so no raw GDAL is involved. Every ``pytest.raises`` block holds a single call — the builder
under test — with the ``Map`` and any ``Normalize``/``ColorScaling`` built above it, so a failing fixture
cannot pass the refusal check (``tests/base/test_refusal_blocks.py``).
"""

import geopandas as gpd
import matplotlib
import pytest
from matplotlib.colors import LogNorm
from pyramids.feature import FeatureCollection
from shapely.geometry import LineString, Point, Polygon

from digitalearth.static import Map

matplotlib.use("Agg")


@pytest.fixture
def points_fc():
    """Two points carrying one numeric column, ``pop``."""
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"pop": [1.0, 5.0]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326"
        )
    )


@pytest.fixture
def bare_points_fc():
    """Two points carrying no attribute column at all."""
    return FeatureCollection(
        gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326")
    )


@pytest.fixture
def lines_fc():
    """One line carrying one numeric column, ``flow``."""
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"flow": [3.0]}, geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:4326"
        )
    )


@pytest.fixture
def polygons_fc():
    """Two polygons carrying one numeric column, ``pop``."""
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"pop": [1.0, 9.0]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1)]),
                Polygon([(2, 2), (3, 2), (3, 3)]),
            ],
            crs="EPSG:4326",
        )
    )


@pytest.fixture
def text_polygons_fc():
    """Two polygons whose only attribute column, ``label``, holds text."""
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"label": ["a", "b"]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1)]),
                Polygon([(2, 2), (3, 2), (3, 3)]),
            ],
            crs="EPSG:4326",
        )
    )


@pytest.fixture
def text_points_fc():
    """Two points whose only attribute column, ``label``, holds text."""
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"label": ["a", "b"]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326"
        )
    )


class TestMissingColumnIsNamed:
    """#379 — a column a builder cannot find is refused by name, listing the columns that exist."""

    def test_points_size_column(self, bare_points_fc):
        """points(size_column=) names the call, the keyword and the available columns."""
        canvas = Map(crs=4326)
        with pytest.raises(KeyError, match=r"Map\.points\(\): size_column='nope'"):
            canvas.points(bare_points_fc, size_column="nope")

    def test_choropleth_column(self, polygons_fc):
        """choropleth(column) is named in the refusal."""
        canvas = Map(crs=4326)
        with pytest.raises(KeyError, match=r"choropleth\(\): column='nope'"):
            canvas.choropleth(polygons_fc, "nope")

    def test_voronoi_column(self, points_fc):
        """voronoi(column=) is named in the refusal."""
        canvas = Map(crs=4326)
        with pytest.raises(KeyError, match=r"voronoi\(\): column='nope'"):
            canvas.voronoi(points_fc, column="nope")

    def test_cartogram_scale(self, polygons_fc):
        """cartogram's sizing column (``scale``) is named when missing."""
        canvas = Map(crs=4326)
        with pytest.raises(KeyError, match=r"cartogram\(\): scale='nope'"):
            canvas.cartogram(polygons_fc, "nope")

    def test_cartogram_colour_column(self, polygons_fc):
        """cartogram's colour ``column`` is named when missing, even with a valid scale."""
        canvas = Map(crs=4326)
        with pytest.raises(KeyError, match=r"cartogram\(\): column='nope'"):
            canvas.cartogram(polygons_fc, "pop", column="nope")

    def test_quadtree_column(self, points_fc):
        """quadtree(column=) is named in the refusal."""
        canvas = Map(crs=4326)
        with pytest.raises(KeyError, match=r"quadtree\(\): column='nope'"):
            canvas.quadtree(points_fc, column="nope")

    def test_sankey_column(self, lines_fc):
        """sankey(column=) is named in the refusal."""
        canvas = Map(crs=4326)
        with pytest.raises(KeyError, match=r"sankey\(\): column='nope'"):
            canvas.sankey(lines_fc, column="nope")

    def test_sankey_scale(self, lines_fc):
        """sankey's width column (``scale``) is named when missing."""
        canvas = Map(crs=4326)
        with pytest.raises(KeyError, match=r"sankey\(\): scale='nope'"):
            canvas.sankey(lines_fc, scale="nope")

    def test_message_lists_available_columns(self, lines_fc):
        """The refusal lists the columns the features do carry, so the typo is easy to correct."""
        canvas = Map(crs=4326)
        with pytest.raises(KeyError, match=r"available: \['flow'\]"):
            canvas.sankey(lines_fc, column="nope")


class TestNonNumericColumnIsNamed:
    """#379 (L2) — a scale/column that exists but holds text is refused by name, not deep in numpy."""

    def test_cartogram_scale_non_numeric(self, text_polygons_fc):
        """cartogram's sizing column holding text is named, not reported as a numpy conversion error."""
        canvas = Map(crs=4326)
        with pytest.raises(
            ValueError,
            match=r"cartogram\(\) needs scale='label' to name a column of numbers",
        ):
            canvas.cartogram(text_polygons_fc, "label")

    def test_quadtree_column_non_numeric(self, text_points_fc):
        """quadtree's aggregate column holding text is named, not reported as a numpy conversion error."""
        canvas = Map(crs=4326)
        with pytest.raises(
            ValueError,
            match=r"quadtree\(\) needs column='label' to name a column of numbers",
        ):
            canvas.quadtree(text_points_fc, column="label")


class TestColourNormWithoutValuesRefused:
    """#380 — a Normalize under color= is refused when the call has no values to scale, honoured otherwise."""

    def test_lines_without_column(self, lines_fc):
        """lines(color=Normalize) with no column is refused, naming color=."""
        canvas = Map(crs=4326)
        norm = LogNorm(vmin=1, vmax=10)
        with pytest.raises(ValueError, match=r"Map\.lines\(\): color= is a LogNorm"):
            canvas.lines(lines_fc, color=norm)

    def test_sankey_without_column(self, lines_fc):
        """sankey(color=Normalize) with no column is refused."""
        canvas = Map(crs=4326)
        norm = LogNorm(vmin=1, vmax=10)
        with pytest.raises(ValueError, match=r"sankey\(\): color= is a LogNorm"):
            canvas.sankey(lines_fc, color=norm)

    def test_polygons_always(self, polygons_fc):
        """polygons(color=Normalize) is refused — a uniform fill has no values to scale."""
        canvas = Map(crs=4326)
        norm = LogNorm(vmin=1, vmax=10)
        with pytest.raises(ValueError, match=r"polygons\(\): color= is a LogNorm"):
            canvas.polygons(polygons_fc, color=norm)

    def test_points_without_values(self, bare_points_fc):
        """points(color=Normalize) is refused when the features carry no numeric value."""
        canvas = Map(crs=4326)
        norm = LogNorm(vmin=1, vmax=10)
        with pytest.raises(ValueError, match=r"Map\.points\(\): color= is a LogNorm"):
            canvas.points(bare_points_fc, color=norm)

    def test_lines_with_column_honoured(self, lines_fc):
        """With a column, a Normalize color is honoured — the artist carries that norm, not a refusal."""
        artist = (
            Map(crs=4326)
            .lines(lines_fc, column="flow", color=LogNorm(vmin=1, vmax=10))
            .artist()
        )
        assert artist.norm.__class__.__name__ == "LogNorm"

    def test_points_with_values_honoured(self, points_fc):
        """points colours by its own numeric value, so a Normalize is honoured when one is present."""
        artist = (
            Map(crs=4326).points(points_fc, color=LogNorm(vmin=1, vmax=10)).artist()
        )
        assert artist.norm.__class__.__name__ == "LogNorm"

    def test_constant_colour_unaffected(self, lines_fc):
        """A plain colour string is still a constant colour, not refused."""
        artist = Map(crs=4326).lines(lines_fc, color="steelblue").artist()
        assert artist is not None

    def test_colorscaling_lines_refused(self, lines_fc):
        """A cleopatra ColorScaling (the docstring-equivalent of Normalize) is refused like a Normalize."""
        from cleopatra.glyphs.base.glyph import ColorScaling

        canvas = Map(crs=4326)
        scaling = ColorScaling()
        with pytest.raises(
            ValueError, match=r"Map\.lines\(\): color= is a ColorScaling"
        ):
            canvas.lines(lines_fc, color=scaling)

    def test_colorscaling_polygons_refused(self, polygons_fc):
        """polygons refuses a ColorScaling too — its uniform fill has nothing to scale."""
        from cleopatra.glyphs.base.glyph import ColorScaling

        canvas = Map(crs=4326)
        scaling = ColorScaling()
        with pytest.raises(ValueError, match=r"polygons\(\): color= is a ColorScaling"):
            canvas.polygons(polygons_fc, color=scaling)
