"""The static tier's three Core spellings, and the three they were adopted from (order 27a).

`imshow`, `scatter` and `shapes` sat in ``PLANNED_RENAMES`` — agreed and unadopted — from the day the Core
contract was frozen. Order 27a adopts them: the tier answers to ``field``, ``points`` and ``polygons``, and
each old spelling is a live alias that warns and names its replacement.

`tests/test_contract_names.py` holds the *shape* of that — the name exists, the alias warns, the alias names
its replacement — off the class, without calling anything. What it cannot ask is whether the alias still
**draws**, which is the half a rename actually risks: an alias that forwarded to the wrong method, or to a
method whose recipe key had moved with it, would satisfy every check there and put a different picture on the
axes. So each pair below is drawn twice, once under each spelling, and the two are compared on what reached
matplotlib and on what the figure records.
"""

import warnings

import geopandas as gpd
import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference
from pyramids.feature import FeatureCollection
from shapely.geometry import Point, Polygon

from digitalearth.api import _STATIC_RASTER_KINDS, quickmap
from digitalearth.base.contract import alias_table
from digitalearth.static import Map
from digitalearth.static.maps.animation import _ANIMATION_KINDS, _KIND_METHODS

#: The three renames, as `(old spelling, Core spelling)`. Parametrised from one place so a fourth cannot be
#: added to the tier without a reader noticing it is untested here.
RENAMED = (("imshow", "field"), ("scatter", "points"), ("shapes", "polygons"))

#: The recipe key each Core name records under `symbology.props["via"]`. **Unchanged by the rename**, which is
#: the claim the aliases' own comments make: a figure written before the rename names `via="imshow"`, and the
#: drawer table is keyed by that, so it has to keep reading back into the same drawer.
VIA = {"field": "imshow", "points": "scatter", "polygons": "shapes"}

#: The raster ``kind=`` tokens `quickmap` accepts — ``"auto"`` plus one per module-level wrapper `api` builds
#: with `_method`. Written here because `api` exposes the wrappers and not the vocabulary, and the check
#: below is about the vocabulary being resolvable rather than about any one token.
QUICKMAP_KINDS = ("auto", "imshow", "contourf", "contour", "pcolormesh")


def _raster():
    """Return a small single-band raster in EPSG:4326.

    Returns:
        A pyramids ``Dataset``.
    """
    return Dataset.from_array(
        np.arange(400, dtype="float32").reshape(20, 20),
        geo_ref=GeoReference(geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326),
    )


def _point_features():
    """Return two point features in EPSG:4326.

    Returns:
        A pyramids ``FeatureCollection``.
    """
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"pop": [1.0, 9.0]},
            geometry=[Point(4.1, 52.9), Point(4.2, 52.8)],
            crs=4326,
        )
    )


def _polygon_features():
    """Return two polygon features in EPSG:4326.

    Returns:
        A pyramids ``FeatureCollection``.
    """
    return FeatureCollection(
        gpd.GeoDataFrame(
            {"pop": [1.0, 9.0]},
            geometry=[
                Polygon([(4.0, 52.0), (4.1, 52.0), (4.1, 52.1), (4.0, 52.1)]),
                Polygon([(4.2, 52.0), (4.3, 52.0), (4.3, 52.1), (4.2, 52.1)]),
            ],
            crs=4326,
        )
    )


#: What each Core name is called with, so one parametrised body can draw all three.
DATA = {"field": _raster, "points": _point_features, "polygons": _polygon_features}


def _drawn_by(spelling: str, core: str):
    """Draw one layer under one spelling and report what reached the engine and what was recorded.

    Args:
        spelling: The method to call — the Core name or the deprecated one.
        core: The Core name, which chooses the data.

    Returns:
        `(artist type name, layer kind, recipe key, count of artists on the axes)`. Deliberately not the
        artist itself: two maps draw two different objects, and what has to match is what kind of thing each
        one is and what the figure says about it.
    """
    scene = Map(crs=4326)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            artist = getattr(scene, spelling)(DATA[core]())
        figure = scene.figure_spec
        layer = figure.layers.get(figure.layers.ids[-1])
        on_axes = len(scene.ax.images) + len(scene.ax.collections)
        return (
            type(artist).__name__,
            layer.kind,
            layer.symbology.props.get("via"),
            on_axes,
        )
    finally:
        scene.close()


class TestTheTierAnswersToItsCoreSpelling:
    """The new names draw, and record what the drawer table is keyed by."""

    @pytest.mark.parametrize("core", sorted(VIA))
    def test_the_core_name_draws_and_records_its_recipe(self, core):
        """Each Core name puts something on the axes and files it under the recipe that drew it.

        Args:
            core: The Core name under test.
        """
        _, _, via, on_axes = _drawn_by(core, core)
        assert (via, on_axes) == (VIA[core], 1), (
            f"Map.{core}() recorded via={via!r} and left {on_axes} artists on the axes"
        )


class TestTheOldSpellingStillDrawsTheSamePicture:
    """The half a name check cannot reach: the alias has to forward to the method that draws."""

    @pytest.mark.parametrize(("old", "core"), RENAMED)
    def test_the_alias_draws_what_the_core_name_draws(self, old, core):
        """Both spellings must leave the same kind of artist on the axes and record the same layer.

        Args:
            old: The deprecated spelling.
            core: The Core name it forwards to.

        Test scenario:
            An alias that warned correctly and forwarded to the wrong method would pass every check in
            `tests/test_contract_names.py`, which reads the class and never calls it. Compared on the
            engine's side — the artist's type and how many artists landed on the axes — and on the figure's:
            the layer's kind and its recipe key.
        """
        assert _drawn_by(old, core) == _drawn_by(core, core), (
            f"Map.{old}() drew {_drawn_by(old, core)} and Map.{core}() drew {_drawn_by(core, core)}"
        )

    @pytest.mark.parametrize(("old", "core"), RENAMED)
    def test_the_alias_warns_and_names_the_core_spelling(self, old, core):
        """The warning has to tell the caller what to write instead.

        Args:
            old: The deprecated spelling.
            core: The Core name it forwards to.
        """
        scene = Map(crs=4326)
        try:
            with pytest.warns(DeprecationWarning) as caught:
                getattr(scene, old)(DATA[core]())
        finally:
            scene.close()
        said = str(caught[0].message)
        assert f"Map.{core}()" in said, f"Map.{old}() warned {said!r}"

    @pytest.mark.parametrize(("old", "core"), RENAMED)
    def test_the_warning_points_at_the_callers_own_line(self, old, core):
        """A warning that blames a line inside the package is one nobody can act on.

        Args:
            old: The deprecated spelling.
            core: The Core name it forwards to.
        """
        scene = Map(crs=4326)
        try:
            with pytest.warns(DeprecationWarning) as caught:
                getattr(scene, old)(DATA[core]())
        finally:
            scene.close()
        assert caught[0].filename == __file__, caught[0].filename


class TestNoInternalCallerReachesADeprecatedSpelling:
    """A rename has call sites, and the ones inside the package are the ones a caller cannot see.

    ``quickmap`` dispatches by input type and by a ``kind=`` token, and ``animate``/``rotate`` dispatch by the
    same kind vocabulary. Every one of those reached a method by name, so leaving any of them on the old
    spelling would warn a caller about a name they never wrote — the "fixed the instance, described the class
    as fixed" failure. The two tables that do the resolving are enumerated here rather than sampled, and the
    dispatch is then exercised for real.
    """

    @pytest.mark.parametrize("kind", sorted(set(_ANIMATION_KINDS)))
    def test_every_animation_kind_resolves_to_a_live_method(self, kind):
        """An animation ``kind`` must name a method that is not itself a deprecated alias.

        Args:
            kind: The kind token under test.
        """
        resolved, _ = _KIND_METHODS.get(kind, (kind, {}))
        assert hasattr(Map, resolved), (
            f"animation kind {kind!r} resolves to Map.{resolved}(), which does not exist"
        )
        assert resolved not in alias_table("matplotlib"), (
            f"animation kind {kind!r} resolves to Map.{resolved}(), which is a deprecated alias; map it in "
            "_KIND_METHODS so a caller is not warned about a spelling they never wrote"
        )

    @pytest.mark.parametrize("kind", sorted(QUICKMAP_KINDS))
    def test_every_quickmap_raster_kind_resolves_to_a_live_method(self, kind):
        """The same for the ``kind=`` vocabulary `quickmap` and its module-level wrappers accept.

        Args:
            kind: The kind token under test.
        """
        resolved, _ = _STATIC_RASTER_KINDS.get(kind, (kind, {}))
        assert hasattr(Map, resolved), (
            f"quickmap kind {kind!r} resolves to Map.{resolved}(), which does not exist"
        )
        assert resolved not in alias_table("matplotlib"), (
            f"quickmap kind {kind!r} resolves to Map.{resolved}(), which is a deprecated alias"
        )

    @pytest.mark.parametrize("core", sorted(VIA))
    def test_quickmap_draws_without_warning_the_caller(self, core):
        """And the dispatch itself, run for real: input type chooses the builder, and none is deprecated.

        Args:
            core: The Core name whose input type `quickmap` must route to.

        Test scenario:
            The table checks above read the resolution; this runs it. `quickmap` picks `polygons` for polygon
            input, `points` for anything else, and the ``kind`` method for a raster — three call sites that
            were `shapes`, `scatter` and `imshow`.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            quickmap(DATA[core](), colorbar=False).close()
        blamed = [
            str(record.message)
            for record in caught
            if issubclass(record.category, DeprecationWarning)
            and "Map." in str(record.message)
        ]
        assert blamed == [], f"quickmap on {core} input warned the caller: {blamed}"
