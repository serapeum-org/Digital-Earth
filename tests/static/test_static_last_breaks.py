"""The static tier's ``last_breaks``, read off the artist the engine built.

``last_breaks`` is the one classification reading every 2-D tier publishes on the live map, and
``tests/base/test_map_conformance.py`` holds the tiers to one answer for one call
(``test_the_same_classification_gives_the_same_class_edges_on_every_tier``). The static tier published
nothing: measured before this, simulating the Core rename that lets it sign the suite,
``AttributeError: 'Map' object has no attribute 'last_breaks'``.

The edges are **not** recomputed here. cleopatra classifies, and what it built is a ``BoundaryNorm`` on the
artist — so the record is read from the mappable the drawer handed back, which is the only reading that
cannot disagree with the picture. Each check below asserts the record against that norm *and* against an
independently known value, so a record that agreed with a wrong classification would still fail.
"""

import geopandas as gpd
import pytest
from pyramids.feature import FeatureCollection
from shapely.geometry import Point, Polygon

from digitalearth.static import Map

#: The column every fixture below classifies.
COLUMN = "pop"

#: The two values a quantile cut into two classes has to answer with, and the edges it must produce: the
#: minimum, the break between the classes, and the maximum. Small enough to check by hand, which is the point
#: — the assertion is not "whatever the norm says", it is "these three numbers, and the norm agrees".
VALUES = (1.0, 9.0)
EXPECTED_EDGES = (1.0, 5.0, 9.0)

#: The scheme and class count the graduated checks ask for.
SCHEME = "quantiles"
CLASSES = 2


def _polygons(values):
    """Return one unit square per value, carrying it in :data:`COLUMN`.

    Args:
        values: The column's values, one per square.

    Returns:
        A pyramids ``FeatureCollection`` in EPSG:4326.
    """
    return FeatureCollection(
        gpd.GeoDataFrame(
            {COLUMN: list(values)},
            geometry=[
                Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)])
                for i in range(len(values))
            ],
            crs=4326,
        )
    )


def _points(values):
    """Return one point per value, carrying it in :data:`COLUMN`.

    Args:
        values: The column's values, one per point.

    Returns:
        A pyramids ``FeatureCollection`` in EPSG:4326.
    """
    return FeatureCollection(
        gpd.GeoDataFrame(
            {COLUMN: list(values)},
            geometry=[Point(4.0 + i / 10, 52.0 + i / 10) for i in range(len(values))],
            crs=4326,
        )
    )


@pytest.fixture
def drawn():
    """Yield an empty static map in EPSG:4326, closed on the way out.

    Yields:
        The tier's ``Map``.
    """
    scene = Map(crs=4326)
    yield scene
    scene.close()


class TestAGraduatedLayerRecordsTheEdgesItWasDrawnWith:
    """The promise the conformance suite asks of every 2-D tier, kept by this one."""

    def test_a_fresh_map_has_classified_nothing(self, drawn):
        """Nothing has been drawn, so there is no most-recent classification to report.

        Args:
            drawn: The map under test.
        """
        assert drawn.last_breaks is None, (
            f"a map with no layers reports {drawn.last_breaks!r} as its last classification"
        )

    def test_a_graduated_choropleth_records_the_class_edges(self, drawn):
        """Two values, two quantile classes, three edges — and the record is the numbers.

        Args:
            drawn: The map under test.

        Test scenario:
            The edges are known by hand (:data:`EXPECTED_EDGES`) rather than copied from the norm, so the
            check fails both for a record that is missing and for one that agrees with a classification
            nobody asked for.
        """
        drawn.choropleth(_polygons(VALUES), COLUMN, scheme=SCHEME, k=CLASSES)
        assert tuple(drawn.last_breaks) == EXPECTED_EDGES, (
            f"a {SCHEME} cut of {VALUES} into {CLASSES} classes recorded {drawn.last_breaks!r}"
        )

    def test_the_record_is_the_edges_the_engine_was_given(self, drawn):
        """And it is read off the artist, not computed beside it.

        Args:
            drawn: The map under test.

        Test scenario:
            The defect this tier has repeatedly had is a value recorded and never handed to the renderer, or
            recomputed so that the record and the picture can drift. The mappable's ``BoundaryNorm`` is what
            matplotlib colours the polygons through, so comparing the record against *it* asks whether the
            two agree — two different readings of one classification, not one value against itself.
        """
        drawn.choropleth(_polygons(VALUES), COLUMN, scheme=SCHEME, k=CLASSES)
        artist = drawn.artist()
        coloured_through = tuple(float(edge) for edge in artist.norm.boundaries)
        assert tuple(drawn.last_breaks) == coloured_through, (
            f"the map records {drawn.last_breaks!r} and the polygons are coloured through "
            f"{coloured_through!r}"
        )

    def test_a_graduated_point_layer_records_them_too(self, drawn):
        """The classification is the layer's, not the builder's — so every classifying builder records.

        Args:
            drawn: The map under test.

        Test scenario:
            `choropleth` declares `scheme=`; `points` takes it through `**opts`. A record written into one
            builder would have left the other silent, which is the "fixed the instance, described the class
            as fixed" shape. Two points valued 1 and 9, cut the same way, give the same three edges. The
            column is named: a `scheme` with no `column` to classify is now refused (#332), so the supported
            way to classify a point layer is the column that holds the values.
        """
        drawn.points(_points(VALUES), column=COLUMN, scheme=SCHEME, k=CLASSES)
        assert tuple(drawn.last_breaks) == EXPECTED_EDGES, (
            f"a graduated point layer recorded {drawn.last_breaks!r}"
        )


class TestTheColumnKeywordClassifiesTheSameWayChoroplethDoes:
    """``points(column=)`` and ``polygons(column=)`` gained the Core colour vocabulary (#332).

    ``points`` and ``polygons`` now name the colour ``column`` and its ``scheme``/``k`` directly, rather than
    reaching the classify group through ``**opts`` or deferring to ``choropleth``. A classified layer built
    through either keyword must record the same class edges a choropleth of the same values does.
    """

    def test_a_point_layer_coloured_by_a_column_has_no_class_edges(self, drawn):
        """A continuous ``column`` colours along a ramp, so there is nothing classified to record.

        Args:
            drawn: The map under test.
        """
        drawn.points(_points(VALUES), column=COLUMN)
        assert drawn.last_breaks is None, (
            f"an unclassified point layer recorded {drawn.last_breaks!r} as its class edges"
        )

    def test_a_graduated_point_column_records_the_choropleth_edges(self, drawn):
        """``points(column=, scheme=)`` cuts the named column into the same edges a choropleth does.

        Args:
            drawn: The map under test.

        Test scenario:
            Two values cut into two quantile classes give the three hand-known edges, the same answer
            :data:`EXPECTED_EDGES` holds for the choropleth of the same values.
        """
        drawn.points(_points(VALUES), column=COLUMN, scheme=SCHEME, k=CLASSES)
        assert tuple(drawn.last_breaks) == EXPECTED_EDGES, (
            f"a graduated point column recorded {drawn.last_breaks!r}, not {EXPECTED_EDGES}"
        )

    def test_a_polygon_column_fill_matches_the_choropleth_of_the_same_values(
        self, drawn
    ):
        """``polygons(column=, scheme=)`` is the choropleth recipe, so it records identical edges.

        Args:
            drawn: The map under test.

        Test scenario:
            The filled ``polygons(column=)`` layer and a ``choropleth`` of the same values and scheme must
            record the one classification — the builders differ, the edges do not.
        """
        drawn.polygons(_polygons(VALUES), column=COLUMN, scheme=SCHEME, k=CLASSES)
        through_polygons = tuple(drawn.last_breaks)
        reference = Map(crs=4326)
        reference.choropleth(_polygons(VALUES), COLUMN, scheme=SCHEME, k=CLASSES)
        through_choropleth = tuple(reference.last_breaks)
        reference.close()
        assert through_polygons == EXPECTED_EDGES, (
            f"a graduated polygons column recorded {through_polygons}, not {EXPECTED_EDGES}"
        )
        assert through_polygons == through_choropleth, (
            f"polygons recorded {through_polygons}, choropleth recorded {through_choropleth}"
        )

    def test_a_categorical_point_column_records_no_numeric_edges(self, drawn):
        """A categorical ``points`` column records ``None``, matching the categorical choropleth.

        Args:
            drawn: The map under test.

        Test scenario:
            A categorical scatter's ``BoundaryNorm`` bins class *codes* — ``[-0.5, 0.5, 1.5]`` for two
            categories — which are not the caller's categories, so publishing them as class edges would be
            publishing a wrong answer. ``points`` records ``None`` exactly as
            ``choropleth(..., scheme="categorical")`` does. Numeric category labels, so the reading is
            isolated from the non-numeric-column handling.
        """
        drawn.points(_points((0.0, 1.0, 0.0)), column=COLUMN, scheme="categorical")
        assert drawn.last_breaks is None, (
            f"a categorical point column recorded {drawn.last_breaks!r}, which are codes not categories"
        )

    def test_a_polygon_column_fill_with_no_scheme_has_no_class_edges(self, drawn):
        """A continuous fill classifies nothing, matching the continuous choropleth.

        Args:
            drawn: The map under test.
        """
        drawn.polygons(_polygons(VALUES), column=COLUMN)
        assert drawn.last_breaks is None, (
            f"a continuous polygons fill recorded {drawn.last_breaks!r} as its class edges"
        )

    def test_a_categorical_polygon_fill_records_no_numeric_edges(self, drawn):
        """A categorical ``polygons`` fill records no numeric edges, matching the categorical choropleth.

        Args:
            drawn: The map under test.

        Test scenario:
            The categorical path bins class *codes*, which are not the caller's categories, so the filled
            ``polygons`` layer records ``None`` exactly as ``choropleth(..., scheme="categorical")`` does.
        """
        drawn.polygons(_polygons(("a", "b", "a")), column=COLUMN, scheme="categorical")
        assert drawn.last_breaks is None, (
            f"a categorical polygons fill recorded {drawn.last_breaks!r}, which are codes not categories"
        )


class TestALayerWithNoClassificationRecordsNone:
    """The complement, so the record cannot pass by always holding numbers."""

    def test_a_continuous_ramp_has_no_classes(self, drawn):
        """`scheme=None` classifies nothing, so the previous classification must not stand.

        Args:
            drawn: The map under test.

        Test scenario:
            Drawn *after* a graduated layer, because that is the case a record can get wrong: the edges of a
            layer that is no longer the most recent one would be reported as this one's.
        """
        drawn.choropleth(_polygons(VALUES), COLUMN, scheme=SCHEME, k=CLASSES)
        drawn.choropleth(_polygons(VALUES), COLUMN)
        assert drawn.last_breaks is None, (
            f"a continuous ramp reported {drawn.last_breaks!r} as its class edges"
        )

    def test_a_categorical_fill_records_no_numeric_edges(self, drawn):
        """A categorical norm bins class *codes*, and codes are not the caller's categories.

        Args:
            drawn: The map under test.

        Test scenario:
            Measured: `choropleth(..., scheme="categorical")` over two categories builds a `BoundaryNorm`
            whose boundaries are `[-0.5, 0.5, 1.5]` — the codes cleopatra assigned, not `["a", "b"]`.
            Publishing those as class edges would be publishing a wrong answer, which is worse than
            publishing none; this tier's categorical key is the swatch legend the glyph draws.
        """
        drawn.choropleth(_polygons(("a", "b", "a")), COLUMN, scheme="categorical")
        assert drawn.last_breaks is None, (
            f"a categorical fill reported {drawn.last_breaks!r}, which are class codes and not categories"
        )


class TestALayerThatColoursNothingLeavesTheRecordStanding:
    """A graticule is not a classification, and must not clear one."""

    def test_a_graticule_drawn_after_a_classification_does_not_clear_it(self, drawn):
        """The reference layer has no mappable at all, so it has nothing to say about class edges.

        Args:
            drawn: The map under test.

        Test scenario:
            The conformance suite's seed figure draws a graticule *after* its choropleth, so a record
            cleared by every subsequent layer would report `None` for a map that had just classified. The
            graticule's drawer hands back no mappable (measured: `None`), which is what makes it silent
            here.
        """
        drawn.choropleth(_polygons(VALUES), COLUMN, scheme=SCHEME, k=CLASSES)
        drawn.graticule(lon_step=30.0, lat_step=30.0)
        assert tuple(drawn.last_breaks) == EXPECTED_EDGES, (
            f"a graticule drawn after a classification left {drawn.last_breaks!r}"
        )

    def test_a_text_label_drawn_after_a_classification_does_not_clear_it(self, drawn):
        """The same for a label, whose artist is a `Text` and carries no colour norm.

        Args:
            drawn: The map under test.
        """
        drawn.choropleth(_polygons(VALUES), COLUMN, scheme=SCHEME, k=CLASSES)
        drawn.text(4.9, 52.4, "Amsterdam")
        assert tuple(drawn.last_breaks) == EXPECTED_EDGES, (
            f"a text label drawn after a classification left {drawn.last_breaks!r}"
        )
