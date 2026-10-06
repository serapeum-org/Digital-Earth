"""WB-22 — clustering polish: style a cluster by an aggregated value, and give it a legend.

``cluster`` drew every bubble one flat colour and sized it by point count: no ``clusterProperties``, so a
cluster could not reflect a feature value, and no key, so a reader could not tell what a colour meant. This
closes both:

* **styling by value** — ``color_by=`` aggregates a feature property across each cluster (``aggregate="sum"``
  or ``"max"``) through MapLibre ``clusterProperties``, and colours the bubble by that aggregate with the same
  classified ``step``/``interpolate`` paint ``choropleth`` builds (reusing ``_color_expr``, not a second copy);
* **a legend** — a classified cluster is keyable exactly as a choropleth is, so ``legend()`` / ``colorbar()``
  draw its key from the colours it actually rendered. A plain cluster carries no classification and so has no
  key, which stays an honest refusal.

The aggregate is computed in the browser per cluster, while the class breaks are derived from the
single-feature column's own spread, so the legend shows the breaks the paint was compiled from. That is exact
only for ``max``: for ``aggregate="sum"`` the legend breaks describe single-feature values, not cluster sums,
so a dense cluster clamps to the top colour while its legend label reads lower than the quantity it encodes.
"""

import pytest

from digitalearth.web import WebMap

pytest.importorskip("maplibre")


@pytest.fixture
def points_gdf():
    """Return five points in lon/lat with a numeric ``value`` column.

    Returns:
        A GeoDataFrame in EPSG:4326.
    """
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    geoms = [Point(x, x) for x in range(5)]
    return gpd.GeoDataFrame(
        {"value": [0.0, 1.0, 2.0, 3.0, 4.0]}, geometry=geoms, crs=4326
    )


class TestStylingByValue:
    """``color_by=`` drives ``clusterProperties`` and a data-driven bubble colour."""

    def test_cluster_properties_are_recorded_for_a_sum(self, points_gdf):
        """A ``sum`` aggregate records the MapLibre ``+`` accumulator over the named column.

        Args:
            points_gdf: The point fixture.

        Test scenario:
            The recorded description is what the drawer wires onto the source, so the accumulator it carries
            is what the browser aggregates with — it has to name the ``+`` operator over ``["get", column]``.
        """
        m = WebMap().cluster(points_gdf, color_by="value", aggregate="sum")
        props = m.get_layer(m.layer_ids[0]).symbology.props
        # `Symbology` canonicalises list props to tuples on storage (as choropleth's paint expressions are);
        # they serialise to JSON arrays for MapLibre.
        assert props["cluster_properties"] == {"value_sum": ("+", ("get", "value"))}

    def test_the_max_aggregate_uses_the_max_operator(self, points_gdf):
        """A ``max`` aggregate records the ``max`` accumulator rather than the ``+`` one.

        Args:
            points_gdf: The point fixture.
        """
        m = WebMap().cluster(points_gdf, color_by="value", aggregate="max")
        props = m.get_layer(m.layer_ids[0]).symbology.props
        assert props["cluster_properties"] == {"value_max": ("max", ("get", "value"))}

    def test_the_bubble_colour_reads_the_aggregated_property(self, points_gdf):
        """The bubble's colour expression reads the cluster's aggregated property, not the flat colour.

        Args:
            points_gdf: The point fixture.
        """
        m = WebMap().cluster(points_gdf, color_by="value", aggregate="sum")
        expression = m.get_layer(m.layer_ids[0]).symbology.props["cluster_color"]
        assert isinstance(expression, tuple)
        assert "value_sum" in repr(expression)

    def test_the_drawn_source_carries_cluster_properties(self, points_gdf):
        """The drawer wires ``clusterProperties`` onto the GeoJSON source, so the browser aggregates.

        Args:
            points_gdf: The point fixture.

        Test scenario:
            Recording the accumulator is only half the job; the proof it reaches MapLibre is that the drawn
            source carries it, which is what the widget's ``add_source`` serialises.
        """
        m = WebMap().cluster(points_gdf, color_by="value", aggregate="sum")
        source = m._renderer.drawn[m.layer_ids[0]].source_spec
        assert source.cluster_properties == {"value_sum": ("+", ("get", "value"))}

    def test_a_plain_cluster_records_no_cluster_properties(self, points_gdf):
        """Without ``color_by`` nothing is aggregated, so no ``clusterProperties`` is recorded.

        Args:
            points_gdf: The point fixture.
        """
        m = WebMap().cluster(points_gdf)
        assert "cluster_properties" not in m.get_layer(m.layer_ids[0]).symbology.props

    def test_an_unknown_aggregate_is_refused(self, points_gdf):
        """Only the accumulators MapLibre has are offered; a typo reads as one rather than building nothing.

        Args:
            points_gdf: The point fixture.
        """
        m = WebMap()
        with pytest.raises(ValueError, match="aggregate"):
            m.cluster(points_gdf, color_by="value", aggregate="median")

    def test_an_unknown_column_is_refused(self, points_gdf):
        """``color_by`` naming no attribute is a mistake, and surfaces as the missing-column error.

        Args:
            points_gdf: The point fixture.
        """
        m = WebMap()
        with pytest.raises(KeyError, match="column 'nope' not found"):
            m.cluster(points_gdf, color_by="nope")


class TestTheClusterLegend:
    """A classified cluster is keyable the way a choropleth is; a plain one is not."""

    def test_a_classified_cluster_draws_a_key(self, points_gdf):
        """``legend()`` draws the cluster's key from the colours it rendered.

        Args:
            points_gdf: The point fixture.
        """
        m = WebMap().cluster(points_gdf, color_by="value", aggregate="sum").legend()
        assert "legend" in m._panels

    def test_a_plain_cluster_has_nothing_to_key(self, points_gdf):
        """A cluster with no classification refuses a key rather than drawing an empty box.

        Args:
            points_gdf: The point fixture.
        """
        m = WebMap().cluster(points_gdf)
        with pytest.raises(ValueError, match="nothing to describe"):
            m.legend()

    def test_the_classification_is_filed_under_the_cluster(self, points_gdf):
        """The breaks the paint was compiled from are filed under the cluster, for its key to read.

        Args:
            points_gdf: The point fixture.
        """
        m = WebMap().cluster(points_gdf, color_by="value", aggregate="max")
        filed = m._legends[m.layer_ids[0]]
        assert filed["column"] == "value_max"


class TestTheChainingContract:
    """``cluster`` returns the map whether or not it classifies, so builder calls chain."""

    def test_a_plain_cluster_returns_self(self, points_gdf):
        """The unclassified path returns the map.

        Args:
            points_gdf: The point fixture.
        """
        m = WebMap()
        assert m.cluster(points_gdf) is m

    def test_a_classified_cluster_returns_self(self, points_gdf):
        """The classified path returns the map too.

        Args:
            points_gdf: The point fixture.
        """
        m = WebMap()
        assert m.cluster(points_gdf, color_by="value") is m
