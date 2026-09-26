"""The interactive tier spells the opacity and column channels the way the Core does (#332, order 27a).

Two divergences, both invisible to a name check because both spellings reached the engine and worked:

* ``opacity`` — what `CHANNELS`, the Core contract and the web and 3-D tiers call it — was HoloViews'
  ``alpha``, travelling through ``**opts``. The conformance suite records the disagreement in
  ``CROSS_TIER_STYLE``, which asked the web tier for ``opacity=`` and this tier for ``alpha=`` for one
  channel;
* ``column`` on ``points`` was ``value_column``, while ``polygons`` and ``choropleth`` on the same tier
  already called it ``column``. One tier, one concept, two names.

Each check reads what HoloViews was **handed** (`style_of`), not what the layer recorded: a channel written
into the description and never passed on is the shape of defect this repository has had repeatedly.
"""

import geopandas as gpd
import pytest
from pyramids.feature import FeatureCollection
from shapely.geometry import Point, Polygon

from digitalearth.interactive import InteractiveMap

gv = pytest.importorskip("geoviews")

#: The opacity the checks ask for. Not 1.0, which would pass for a builder that dropped the keyword.
OPACITY = 0.25

#: The column the fill or the point colour is driven by.
COLUMN = "pop"


def _points():
    """Return two point features carrying a numeric column.

    Returns:
        A pyramids ``FeatureCollection`` in EPSG:4326.
    """
    return FeatureCollection(
        gpd.GeoDataFrame(
            {COLUMN: [1.0, 9.0]},
            geometry=[Point(4.1, 52.9), Point(4.2, 52.8)],
            crs=4326,
        )
    )


def _polygons():
    """Return two polygon features carrying a numeric column.

    Returns:
        A pyramids ``FeatureCollection`` in EPSG:4326.
    """
    return FeatureCollection(
        gpd.GeoDataFrame(
            {COLUMN: [1.0, 9.0]},
            geometry=[
                Polygon([(4.0, 52.0), (4.1, 52.0), (4.1, 52.1), (4.0, 52.1)]),
                Polygon([(4.2, 52.0), (4.3, 52.0), (4.3, 52.1), (4.2, 52.1)]),
            ],
            crs=4326,
        )
    )


#: Every builder the opacity channel is declared on by the Core, with the data it draws. Parametrised from
#: one place so a builder that gains the channel later cannot be added without a reader noticing it is here.
OPACITY_BUILDERS = {
    "points": _points,
    "polygons": _polygons,
    "choropleth": _polygons,
}


def _drawn(builder: str, **style):
    """Draw one layer and return what HoloViews was handed for it.

    Args:
        builder: The builder to call.
        **style: The keywords to draw with.

    Returns:
        The layer's resolved common style options — the engine's own reading, which is the only place a
        channel can have an effect.
    """
    scene = InteractiveMap()
    data = OPACITY_BUILDERS[builder]()
    if builder == "choropleth":
        getattr(scene, builder)(data, COLUMN, **style)
    else:
        getattr(scene, builder)(data, **style)
    return scene.style_of(0)["common"]


class TestTheCoreSpellingReachesTheEngine:
    """``opacity=`` is the name; HoloViews' ``alpha`` is what it arrives as."""

    @pytest.mark.parametrize("builder", sorted(OPACITY_BUILDERS))
    def test_opacity_is_handed_to_holoviews_as_alpha(self, builder):
        """The channel reaches the engine under the option HoloViews reads.

        Args:
            builder: The builder under test.
        """
        assert _drawn(builder, opacity=OPACITY).get("alpha") == OPACITY, builder

    @pytest.mark.parametrize("builder", sorted(OPACITY_BUILDERS))
    def test_no_opacity_is_handed_over_when_none_was_asked_for(self, builder):
        """A caller who asks for no opacity must not have one invented for them.

        Args:
            builder: The builder under test.

        Test scenario:
            The complement that stops the check above passing for a builder that always sets an alpha.
            Measured: with nothing asked for, HoloViews is handed no `alpha` option at all.
        """
        assert "alpha" not in _drawn(builder), builder

    def test_column_colours_the_points_by_that_column(self):
        """``column=`` on ``points`` is the same name its sibling builders already use.

        Test scenario:
            Read as the colour dimension HoloViews was given, which is what `value_column` produced — so the
            two spellings are compared on the engine's own reading rather than on the record.
        """
        assert _drawn("points", column=COLUMN).get("color") == COLUMN
