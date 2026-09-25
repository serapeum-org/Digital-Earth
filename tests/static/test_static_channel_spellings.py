"""The static tier spells the opacity channel the way the Core does (#332, order 27a).

`CHANNELS` and the Core contract both call it ``opacity``; the web and 3-D tiers take it under that name.
This tier took matplotlib's ``alpha`` straight through ``**opts``, so a caller writing one script against two
tiers had to spell the same channel two ways — and `KEYWORD_SHORTFALLS` carried the gap as tracked divergence
rather than as a fix.

``alpha`` keeps working for one release and warns. What is asserted below is that the new spelling reaches
the **artist**, not merely the description: an opacity recorded and never handed to matplotlib is precisely
the shape of defect this tier has had before (a held colormap four builders never read, a ``visible=False``
that hid the element while the record said shown).
"""

import warnings

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


class TestTheEnginesSpellingStillWorks:
    """``alpha=`` is a promise to every script already written against this tier."""

    def test_alpha_reaches_the_artist_identically(self, polygons):
        """Both spellings must leave the artist holding the same value.

        Args:
            polygons: The polygon fixture.
        """
        under_core = Map(crs=4326)
        under_engine = Map(crs=4326)
        try:
            core = under_core.choropleth(polygons, COLUMN, opacity=OPACITY)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                engine = under_engine.choropleth(polygons, COLUMN, alpha=OPACITY)
            assert engine.get_alpha() == core.get_alpha(), (
                f"alpha= gave {engine.get_alpha()} and opacity= gave {core.get_alpha()}"
            )
        finally:
            under_core.close()
            under_engine.close()

    def test_alpha_warns_and_names_the_core_spelling(self, drawn, polygons):
        """The warning has to tell the caller what to write instead.

        Args:
            drawn: The map under test.
            polygons: The polygon fixture.
        """
        with pytest.warns(DeprecationWarning) as caught:
            drawn.choropleth(polygons, COLUMN, alpha=OPACITY)
        assert "use opacity= instead" in str(caught[0].message), caught[0].message

    def test_the_warning_points_at_the_callers_own_line(self, drawn, polygons):
        """A warning blaming a line inside the package is one nobody can act on.

        Args:
            drawn: The map under test.
            polygons: The polygon fixture.

        Test scenario:
            The builder is wrapped by `_skips_off_limb`, so the frame count from the resolver to the caller
            is one deeper than a plain method's. A hand-counted `stacklevel` that forgot the wrapper would
            blame `static/maps/vector.py`, and nothing but this would say so.
        """
        with pytest.warns(DeprecationWarning) as caught:
            drawn.choropleth(polygons, COLUMN, alpha=OPACITY)
        assert caught[0].filename == __file__, (
            f"{caught[0].filename}:{caught[0].lineno}"
        )

    def test_both_spellings_at_once_is_refused(self, drawn, polygons):
        """Two names for one channel is a caller error, reported as one.

        Args:
            drawn: The map under test.
            polygons: The polygon fixture.
        """
        with pytest.raises(TypeError) as refused:
            drawn.choropleth(polygons, COLUMN, opacity=OPACITY, alpha=0.9)
        assert "so pass only opacity=" in str(refused.value), refused.value
