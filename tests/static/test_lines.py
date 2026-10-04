"""``Map.lines`` — plain line geometry on the static tier (ST-11, #226).

Rivers, roads, tracks and reach networks are a basic map layer, and the static tier's only line renderer was
``sankey``, a flow map under a flow name with flow defaults. ``lines`` is the Core contract's name for it,
with the keywords the web tier's ``lines`` takes. Expected values are built from the inputs in the test, not
read back through the code under test.
"""

import math

import geopandas as gpd
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LogNorm, Normalize, to_rgba
from pyramids.feature import FeatureCollection
from shapely.geometry import LineString, MultiLineString, Point

from digitalearth.base.contract import PENDING
from digitalearth.static import Map

#: Web Mercator's sphere radius, in metres (EPSG:3857).
MERCATOR_R = 6378137.0


@pytest.fixture
def rivers() -> FeatureCollection:
    """Three reaches in lon/lat, one of them a two-part MultiLineString, with a discharge column.

    Returns:
        A pyramids `FeatureCollection`.
    """
    gdf = gpd.GeoDataFrame(
        {"discharge": [10.0, 40.0, 25.0], "order": [1, 3, 2]},
        geometry=[
            LineString([(0.0, 0.0), (1.0, 1.0)]),
            MultiLineString([[(1.0, 1.0), (2.0, 1.5)], [(2.0, 1.5), (3.0, 1.0)]]),
            LineString([(0.0, 2.0), (1.0, 2.5), (2.0, 2.0)]),
        ],
        crs="EPSG:4326",
    )
    return FeatureCollection(gdf)


class TestDrawingLines:
    """The builder draws one path per line part, in the display CRS."""

    def test_it_returns_a_line_collection(self, rivers):
        """The public return is the drawn ``LineCollection``.

        Args:
            rivers: The reaches.
        """
        artist = Map(crs=4326).lines(rivers)
        assert isinstance(artist, LineCollection), type(artist)

    def test_a_multilinestring_draws_one_path_per_part(self, rivers):
        """Three features, one of them in two parts: four paths.

        Args:
            rivers: The reaches.
        """
        artist = Map(crs=4326).lines(rivers)
        assert len(artist.get_paths()) == 4, len(artist.get_paths())

    def test_the_lines_are_reprojected_into_the_display_crs(self, rivers):
        """On Web Mercator the first reach ends at the Mercator position of (1, 1).

        Args:
            rivers: The reaches.
        """
        artist = Map(crs=3857).lines(rivers)
        end = artist.get_paths()[0].vertices[-1]
        expected = (
            MERCATOR_R * math.radians(1.0),
            MERCATOR_R * math.log(math.tan(math.pi / 4 + math.radians(1.0) / 2)),
        )
        assert end == pytest.approx(expected, rel=1e-6), end

    def test_the_layer_is_described_as_lines(self, rivers):
        """The figure records the registered ``lines`` kind and the column it was coloured by.

        Args:
            rivers: The reaches.
        """
        canvas = Map(crs=4326)
        canvas.lines(rivers, column="discharge", name="rivers")
        layer = canvas.figure_spec.layers.get("rivers")
        assert layer.kind == "lines", layer.kind
        assert layer.symbology.props["column"] == "discharge", layer.symbology.props

    def test_non_line_geometry_is_refused(self):
        """Points are not lines; the refusal names the geometry it wanted.

        Test scenario:
            The same guard ``sankey`` uses, so the two builders reject the same inputs the same way.
        """
        points = FeatureCollection(
            gpd.GeoDataFrame(geometry=[Point(0.0, 0.0)], crs="EPSG:4326")
        )
        canvas = Map(crs=4326)
        with pytest.raises(ValueError, match="line"):
            canvas.lines(points)
        assert canvas.layer_ids == [], canvas.layer_ids


class TestColour:
    """A constant colour, or a colour per feature from a column."""

    def test_a_column_colours_each_path_by_its_feature(self, rivers):
        """Each part carries its feature's value, the MultiLineString's twice.

        Args:
            rivers: The reaches.
        """
        artist = Map(crs=4326).lines(rivers, column="discharge")
        assert list(artist.get_array()) == [10.0, 40.0, 40.0, 25.0], artist.get_array()

    def test_a_scheme_classifies_the_column(self, rivers):
        """``scheme=`` and ``k=`` cut the values into classes, as on the other builders.

        Args:
            rivers: The reaches.
        """
        artist = Map(crs=4326).lines(
            rivers, column="discharge", scheme="equal_interval", k=3
        )
        assert isinstance(artist.norm, BoundaryNorm), type(artist.norm)

    def test_a_constant_colour_paints_every_line(self, rivers):
        """``color=`` a colour is the line colour, the web tier's spelling.

        Args:
            rivers: The reaches.
        """
        artist = Map(crs=4326).lines(rivers, color="steelblue")
        colours = {tuple(rgba) for rgba in artist.get_colors()}
        assert colours == {to_rgba("steelblue")}, colours

    def test_a_cmap_reaches_the_ramp(self, rivers):
        """``cmap=`` names the ramp the column is coloured through.

        Args:
            rivers: The reaches.
        """
        artist = Map(crs=4326).lines(rivers, column="discharge", cmap="plasma")
        assert artist.get_cmap().name == "plasma", artist.get_cmap().name

    def test_a_built_norm_under_color_scales_the_column(self, rivers):
        """``color=`` a ``Normalize`` is the colour *scaling*, not a constant line colour.

        Args:
            rivers: The reaches.

        Test scenario:
            ``color=`` is overloaded on this builder: a matplotlib colour paints every line, and a built
            ``ColorScaling``/``Normalize`` is instead how ``column`` maps onto the ramp — the escape hatch
            every other static builder offers. Discharge spans 10 to 40, so a ``LogNorm`` and a linear one
            place the middle reach at visibly different points on the ramp; the object the caller built
            must arrive on the collection untouched rather than be rebuilt or read as a colour name.
        """
        caller_norm = LogNorm(vmin=10.0, vmax=40.0)
        artist = Map(crs=4326).lines(rivers, column="discharge", color=caller_norm)
        assert artist.norm is caller_norm, (
            f"the caller's own norm must reach the collection, got {artist.norm!r}"
        )
        assert artist.norm(25.0) != pytest.approx(Normalize(10.0, 40.0)(25.0)), (
            "a LogNorm that scaled like a linear one would make this pin vacuous"
        )


class TestWidthAndOpacity:
    """``width=`` takes a number or a column; ``opacity=`` fades the layer."""

    def test_a_number_is_every_line_s_width(self, rivers):
        """A scalar width is the width of every path, in points.

        Args:
            rivers: The reaches.
        """
        artist = Map(crs=4326).lines(rivers, width=3.5)
        assert set(np.round(artist.get_linewidths(), 6)) == {3.5}, (
            artist.get_linewidths()
        )

    def test_a_column_scales_the_widths_with_its_values(self, rivers):
        """Wider for a larger value: the paths' widths rank as the column does.

        Args:
            rivers: The reaches.
        """
        artist = Map(crs=4326).lines(rivers, width="order")
        widths = np.asarray(artist.get_linewidths())
        order = np.repeat([1, 3, 2], [1, 2, 1])
        assert list(np.argsort(widths, kind="stable")) == list(
            np.argsort(order, kind="stable")
        ), (widths, order)

    def test_opacity_fades_the_layer(self, rivers):
        """``opacity=`` is the layer's alpha.

        Args:
            rivers: The reaches.
        """
        artist = Map(crs=4326).lines(rivers, opacity=0.4)
        assert artist.get_alpha() == pytest.approx(0.4), artist.get_alpha()

    def test_a_layer_built_hidden_draws_hidden(self, rivers):
        """``visible=False`` builds the layer hidden and describes it hidden.

        Args:
            rivers: The reaches.
        """
        canvas = Map(crs=4326)
        artist = canvas.lines(rivers, name="r", visible=False)
        assert artist.get_visible() is False, "the artist must be hidden"
        assert canvas.figure_spec.layers.get("r").visible is False, (
            "and so must the description"
        )


class TestOffTheLimb:
    """Lines the display CRS cannot place are a skipped layer, as on every other vector builder."""

    @pytest.mark.parametrize("builder", ["lines", "sankey"])
    def test_lines_behind_the_globe_are_skipped(self, rivers, builder):
        """A globe centred on the antimeridian cannot show reaches near (0, 0); nothing is drawn or described.

        Args:
            rivers: The reaches, all within a few degrees of the origin.
            builder: The line builder under test.
        """
        from digitalearth.static import projections

        canvas = Map(crs=projections.orthographic(lon=180, lat=0), globe=True)
        drawn = getattr(canvas, builder)(rivers)
        assert drawn is None, drawn
        assert canvas.layer_ids == [], canvas.layer_ids


class TestTheContract:
    """The Core ``lines`` name is no longer pending on the static tier."""

    def test_lines_is_not_pending_on_matplotlib(self):
        """The contract's PENDING table stops excusing the tier."""
        assert "lines" not in PENDING["matplotlib"], dict(PENDING["matplotlib"])
