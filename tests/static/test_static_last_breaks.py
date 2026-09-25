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
        artist = drawn.choropleth(_polygons(VALUES), COLUMN, scheme=SCHEME, k=CLASSES)
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
            `choropleth` declares `scheme=`; `scatter` takes it through `**opts`. A record written into one
            builder would have left the other silent, which is the "fixed the instance, described the class
            as fixed" shape. Two points valued 1 and 9, cut the same way, give the same three edges.
        """
        drawn.scatter(_points(VALUES), scheme=SCHEME, k=CLASSES)
        assert tuple(drawn.last_breaks) == EXPECTED_EDGES, (
            f"a graduated scatter recorded {drawn.last_breaks!r}"
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
