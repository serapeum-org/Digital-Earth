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


@pytest.fixture
def text_lines_fc():
    """One line whose only attribute column, ``label``, holds text."""
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"label": ["a"]},
            geometry=[LineString([(0, 0), (1, 1)])],
            crs="EPSG:4326",
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

    def test_points_column(self, points_fc):
        """points(column=) names ``Map.points()`` when the colour column is absent (#332 draw path).

        Args:
            points_fc: Two points carrying only the ``pop`` column.

        Test scenario:
            The ``column=`` keyword picks the values to colour by; a name the features do not carry is
            refused by name in ``draw_scatter`` rather than failing later inside numpy.
        """
        canvas = Map(crs=4326)
        with pytest.raises(KeyError, match=r"Map\.points\(\): column='nope'"):
            canvas.points(points_fc, column="nope")

    def test_polygons_column(self, polygons_fc):
        """polygons(column=) routes to the fill recipe, so a missing column is named ``choropleth()`` (#332).

        Args:
            polygons_fc: Two polygons carrying only the ``pop`` column.

        Test scenario:
            A filled ``polygons(column=)`` layer is drawn by the choropleth recipe, so its column guard is
            the choropleth one and names that call — the entry point differs, the refusal does not.
        """
        canvas = Map(crs=4326)
        with pytest.raises(KeyError, match=r"choropleth\(\): column='nope'"):
            canvas.polygons(polygons_fc, column="nope")

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

    def test_sankey_scale_non_numeric(self, text_lines_fc):
        """sankey's width scale holding text is named, not left to numpy's bare conversion error."""
        canvas = Map(crs=4326)
        with pytest.raises(
            ValueError,
            match=r"sankey\(\) needs scale='label' to name a column of numbers",
        ):
            canvas.sankey(text_lines_fc, scale="label")

    def test_sankey_column_non_numeric(self, text_lines_fc):
        """sankey's colour column holding text is named, not left to numpy's bare ufunc error."""
        canvas = Map(crs=4326)
        with pytest.raises(
            ValueError,
            match=r"sankey\(\) needs column='label' to name a column of numbers",
        ):
            canvas.sankey(text_lines_fc, column="label")

    def test_points_column_non_numeric_without_categorical_scheme(self, text_points_fc):
        """points(column=<text>) with no categorical scheme is named, not a numpy conversion error (#332 L4).

        Args:
            text_points_fc: Two points whose only column, ``label``, holds strings.

        Test scenario:
            The ``column=`` keyword routes a text column straight into the continuous ramp, which coerced to
            float and died in cleopatra as "could not convert string to float". The refusal names the call,
            the column and the ``scheme="categorical"`` way out instead.
        """
        canvas = Map(crs=4326)
        with pytest.raises(
            ValueError,
            match=r"Map\.points\(\) needs column='label' to name a column of numbers",
        ):
            canvas.points(text_points_fc, column="label")

    def test_points_datetime_column_is_steered_to_numbers_not_categorical(self):
        """A datetime column on a continuous ramp is refused toward numbers, not categorical (#332 L1).

        Test scenario:
            A date ramp is conceptually valid, but cleopatra cannot scale a ``datetime64`` column, and
            ``scheme="categorical"`` — one colour per timestamp — is almost never what a date ramp wants.
            The message steers the caller to an ordinal/epoch column instead.
        """
        import pandas as pd

        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"when": pd.to_datetime(["2020-01-01", "2020-06-01"])},
                geometry=[Point(0, 0), Point(1, 1)],
                crs="EPSG:4326",
            )
        )
        canvas = Map(crs=4326)
        with pytest.raises(
            ValueError, match=r"date/time values a colour ramp cannot scale"
        ):
            canvas.points(fc, column="when")

    def test_choropleth_column_non_numeric_without_categorical_scheme(
        self, text_polygons_fc
    ):
        """choropleth(<text column>) with no categorical scheme is named, not a numpy error (#332 L4).

        Args:
            text_polygons_fc: Two polygons whose only column, ``label``, holds strings.

        Test scenario:
            The continuous choropleth ramp over a string column crashed the same way ``points`` did; it now
            refuses by name, consistently with ``points`` and with the ``scheme="categorical"`` remedy.
        """
        canvas = Map(crs=4326)
        with pytest.raises(
            ValueError,
            match=r"choropleth\(\) needs column='label' to name a column of numbers",
        ):
            canvas.choropleth(text_polygons_fc, "label")


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

    def test_polygons_refusal_points_at_column(self, polygons_fc):
        """The no-column refusal now points at ``column=``, not at ``choropleth`` (#332 message).

        Args:
            polygons_fc: Two polygons carrying a numeric ``pop`` column.

        Test scenario:
            ``polygons`` grew a ``column=`` keyword, so the way to colour by an attribute is that keyword
            on the same builder; the refusal names it rather than redirecting to ``choropleth(column=)``.
        """
        canvas = Map(crs=4326)
        norm = LogNorm(vmin=1, vmax=10)
        with pytest.raises(ValueError, match=r"pass column= to colour polygons"):
            canvas.polygons(polygons_fc, color=norm)

    def test_polygons_with_column_honours_a_norm(self, polygons_fc):
        """With a ``column``, a ``Normalize`` colour is honoured, as on the other filled builders (#332).

        Args:
            polygons_fc: Two polygons carrying a numeric ``pop`` column.

        Test scenario:
            A filled ``polygons(column=)`` layer has values to scale, so the colour scaling that an
            outline layer refuses is carried on the artist here instead.
        """
        artist = (
            Map(crs=4326)
            .polygons(polygons_fc, column="pop", color=LogNorm(vmin=1, vmax=10))
            .artist()
        )
        assert artist.norm.__class__.__name__ == "LogNorm", (
            f"a filled polygons layer must keep the norm, got {artist.norm.__class__.__name__}"
        )

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


class TestSchemeWithoutColumnRefused:
    """#332 — a ``scheme`` with no ``column`` to classify is refused, not silently acted on.

    Before this, static ``points`` silently classified an auto-detected field and static ``polygons``
    silently ignored the scheme — two different wrong answers to one call, where interactive ``lines`` /
    ``points`` / ``polygons`` already refuse it (#331). The message is the interactive tier's, in the static
    tier's own spelling of the caller.
    """

    def test_points_scheme_without_column(self, points_fc):
        """``points(scheme=)`` with no ``column`` is refused, naming the call and ``column=``.

        Args:
            points_fc: Two points carrying a numeric ``pop`` column that the call does not name.
        """
        canvas = Map(crs=4326)
        with pytest.raises(
            ValueError,
            match=r"Map\.points\(\): scheme= classifies column=, which was not given",
        ):
            canvas.points(points_fc, scheme="quantiles")

    def test_polygons_scheme_without_column(self, polygons_fc):
        """``polygons(scheme=)`` with no ``column`` is refused rather than drawing plain outlines.

        Args:
            polygons_fc: Two polygons carrying a numeric ``pop`` column that the call does not name.
        """
        canvas = Map(crs=4326)
        with pytest.raises(
            ValueError,
            match=r"polygons\(\): scheme= classifies column=, which was not given",
        ):
            canvas.polygons(polygons_fc, scheme="quantiles")

    def test_points_scheme_with_column_is_not_refused(self, points_fc):
        """The guard does not over-reach: ``scheme`` paired with a ``column`` still classifies.

        Args:
            points_fc: Two points carrying a numeric ``pop`` column.
        """
        artist = (
            Map(crs=4326).points(points_fc, column="pop", scheme="quantiles").artist()
        )
        assert artist is not None, (
            "points(column=, scheme=) must still draw a classified layer"
        )

    def test_polygons_without_scheme_is_not_refused(self, polygons_fc):
        """A plain outline ``polygons`` (no ``scheme``, no ``column``) is unaffected by the guard.

        Args:
            polygons_fc: Two polygons; neither ``scheme`` nor ``column`` is given.
        """
        artist = Map(crs=4326).polygons(polygons_fc).artist()
        assert artist is not None, "plain polygons outlines must still draw"


class TestCategoricalStringColumnDraws:
    """#332 — a categorical ``points`` layer coloured by a non-numeric column draws, like ``choropleth``.

    A string/object category column used to crash ``draw_scatter`` with numpy's "could not convert string to
    float", because the scatter path force-coerced the colour values to float before cleopatra saw them.
    ``choropleth``/``polygons`` read the column untyped and classify a nominal column fine; ``points`` now
    does too.
    """

    def test_string_category_column_does_not_crash(self, text_points_fc):
        """``points(column=<string>, scheme="categorical")`` draws an artist instead of raising.

        Args:
            text_points_fc: Two points whose only column, ``label``, holds the strings ``"a"``/``"b"``.
        """
        artist = (
            Map(crs=4326)
            .points(text_points_fc, column="label", scheme="categorical")
            .artist()
        )
        assert artist is not None, "a categorical string column must draw a point layer"

    def test_string_category_column_is_keyed_by_a_swatch_legend(self, text_points_fc):
        """The categorical scatter builds the glyph's own swatch legend, one swatch per distinct label.

        Args:
            text_points_fc: Two points whose only column, ``label``, holds the strings ``"a"``/``"b"``.

        Test scenario:
            A real categorical render keys itself by a disjoint swatch legend (``category_legend``) rather
            than a colorbar over opaque class codes — the same key ``choropleth(scheme="categorical")``
            draws — so the labels read back are the caller's categories.
        """
        canvas = Map(crs=4326)
        canvas.points(text_points_fc, column="label", scheme="categorical")
        legend = canvas.layers[-1][0].category_legend
        labels = [text.get_text() for text in legend.get_texts()]
        assert labels == ["a", "b"], (
            f"the swatch legend keyed {labels!r}, not the categories ['a', 'b']"
        )

    def test_string_category_column_records_no_numeric_edges(self, text_points_fc):
        """A categorical scatter records no numeric ``last_breaks`` — codes are not categories.

        Args:
            text_points_fc: Two points whose only column, ``label``, holds the strings ``"a"``/``"b"``.
        """
        canvas = Map(crs=4326)
        canvas.points(text_points_fc, column="label", scheme="categorical")
        assert canvas.last_breaks is None, (
            f"a categorical point column recorded {canvas.last_breaks!r}, which are codes not categories"
        )
