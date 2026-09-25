"""The interactive tier spells the opacity and column channels the way the Core does (#332, order 27a).

Two divergences, both invisible to a name check because both spellings reached the engine and worked:

* ``opacity`` — what `CHANNELS`, the Core contract and the web and 3-D tiers call it — was HoloViews'
  ``alpha``, travelling through ``**opts``. The conformance suite records the disagreement in
  ``CROSS_TIER_STYLE``, which asked the web tier for ``opacity=`` and this tier for ``alpha=`` for one
  channel;
* ``column`` on ``points`` was ``value_column``, while ``polygons`` and ``choropleth`` on the same tier
  already called it ``column``. One tier, one concept, two names.

Both old spellings keep working for one release and warn. Each check reads what HoloViews was **handed**
(`style_of`), not what the layer recorded: a channel written into the description and never passed on is the
shape of defect this repository has had repeatedly.
"""

import warnings

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
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


class TestTheOldSpellingsStillWork:
    """Both are a promise to every script already written against this tier."""

    @pytest.mark.parametrize("builder", sorted(OPACITY_BUILDERS))
    def test_alpha_is_handed_over_identically(self, builder):
        """Both spellings must reach HoloViews as the same option and value.

        Args:
            builder: The builder under test.
        """
        assert _drawn(builder, alpha=OPACITY).get("alpha") == _drawn(
            builder, opacity=OPACITY
        ).get("alpha"), builder

    @pytest.mark.parametrize("builder", sorted(OPACITY_BUILDERS))
    def test_alpha_warns_and_names_the_core_spelling(self, builder):
        """The warning has to tell the caller what to write instead.

        Args:
            builder: The builder under test.
        """
        with pytest.warns(DeprecationWarning) as caught:
            _drawn_warning(builder, alpha=OPACITY)
        assert "use opacity= instead" in str(caught[0].message), caught[0].message

    @pytest.mark.parametrize("builder", sorted(OPACITY_BUILDERS))
    def test_the_alpha_warning_points_at_the_callers_own_line(self, builder):
        """A warning blaming a line inside the package is one nobody can act on.

        Args:
            builder: The builder under test.

        Test scenario:
            Every builder here is wrapped by `_skips_off_limb`, so the frame count from the resolver to the
            caller is one deeper than a plain method's. A hand-counted `stacklevel` that forgot the wrapper
            would blame `interactive/vector.py`, and nothing but this would say so.
        """
        with pytest.warns(DeprecationWarning) as caught:
            _drawn_warning(builder, alpha=OPACITY)
        assert caught[0].filename == __file__, (
            f"{caught[0].filename}:{caught[0].lineno}"
        )

    @pytest.mark.parametrize("builder", sorted(OPACITY_BUILDERS))
    def test_both_opacity_spellings_at_once_is_refused(self, builder):
        """Two names for one channel is a caller error, reported as one.

        Args:
            builder: The builder under test.
        """
        with pytest.raises(TypeError) as refused:
            _drawn_warning(builder, opacity=OPACITY, alpha=0.9)
        assert "so pass only opacity=" in str(refused.value), refused.value

    def test_value_column_is_handed_over_identically(self):
        """``value_column=`` keeps colouring the points by that column."""
        assert _drawn("points", value_column=COLUMN).get("color") == _drawn(
            "points", column=COLUMN
        ).get("color")

    def test_value_column_warns_and_names_the_core_spelling(self):
        """And says what to write instead."""
        with pytest.warns(DeprecationWarning) as caught:
            _drawn_warning("points", value_column=COLUMN)
        assert "use column= instead" in str(caught[0].message), caught[0].message

    def test_the_value_column_warning_points_at_the_callers_own_line(self):
        """The column rename is resolved inline, so it counts one frame fewer than the opacity one.

        Test scenario:
            Two `stacklevel` counts on one builder: `column` is resolved in `points()` itself and `opacity`
            through the shared `resolve_opacity`, which adds a frame. A single number copied to both would
            put one of the two warnings inside the package, and only a filename check says which.
        """
        with pytest.warns(DeprecationWarning) as caught:
            _drawn_warning("points", value_column=COLUMN)
        assert caught[0].filename == __file__, (
            f"{caught[0].filename}:{caught[0].lineno}"
        )

    def test_both_column_spellings_at_once_is_refused(self):
        """Two names for one column is a caller error, reported as one."""
        with pytest.raises(TypeError) as refused:
            _drawn_warning("points", column=COLUMN, value_column=COLUMN)
        assert "so pass only column=" in str(refused.value), refused.value


def _drawn_warning(builder: str, **style):
    """Draw one layer without swallowing its warnings, for the checks that are about them.

    Args:
        builder: The builder to call.
        **style: The keywords to draw with.

    Returns:
        The map, so a caller can read it; the point is that nothing filters the warning.
    """
    scene = InteractiveMap()
    data = OPACITY_BUILDERS[builder]()
    if builder == "choropleth":
        getattr(scene, builder)(data, COLUMN, **style)
    else:
        getattr(scene, builder)(data, **style)
    return scene
