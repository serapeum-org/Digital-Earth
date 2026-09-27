"""The static tier spells the opacity channel the way the Core does (#332, order 27a).

`CHANNELS` and the Core contract both call it ``opacity``; the web and 3-D tiers take it under that name.
This tier took matplotlib's ``alpha`` straight through ``**opts``, so a caller writing one script against two
tiers had to spell the same channel two ways — and `KEYWORD_SHORTFALLS` carried the gap as tracked divergence
rather than as a fix.

What is asserted below is that the Core spelling reaches the **artist**, not merely the description: an
opacity recorded and never handed to matplotlib is precisely the shape of defect this tier has had before (a
held colormap four builders never read, a ``visible=False`` that hid the element while the record said shown).
"""

import geopandas as gpd
import pytest
from pyramids.feature import FeatureCollection
from shapely.geometry import Polygon

from digitalearth.static import Map

#: The opacity the checks ask for. Not 1.0, which is matplotlib's own default and so would pass for a
#: builder that dropped the keyword entirely.
OPACITY = 0.25

#: The column the fill is driven by.
COLUMN = "pop"


@pytest.fixture
def polygons():
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


@pytest.fixture
def drawn():
    """Yield an empty map in EPSG:4326, closed on the way out.

    Yields:
        The tier's ``Map``.
    """
    scene = Map(crs=4326)
    yield scene
    scene.close()


class TestTheCoreSpellingReachesTheArtist:
    """``opacity=`` is the name; matplotlib's ``alpha`` is what the artist ends up holding."""

    def test_opacity_sets_the_artists_alpha(self, drawn, polygons):
        """The channel reaches the engine, which is the only place it can have an effect.

        Args:
            drawn: The map under test.
            polygons: The polygon fixture.
        """
        artist = drawn.choropleth(polygons, COLUMN, opacity=OPACITY)
        assert artist.get_alpha() == OPACITY, artist.get_alpha()

    def test_it_is_absent_by_default(self, drawn, polygons):
        """A caller who asks for no opacity must not have one invented for them.

        Args:
            drawn: The map under test.
            polygons: The polygon fixture.

        Test scenario:
            The complement that stops the check above passing for a builder that always sets an alpha.
            matplotlib's own "no alpha" is `None`, not `1.0`, and a layer left at `None` is what lets a
            colormap's own alpha channel through.
        """
        artist = drawn.choropleth(polygons, COLUMN)
        assert artist.get_alpha() is None, artist.get_alpha()
