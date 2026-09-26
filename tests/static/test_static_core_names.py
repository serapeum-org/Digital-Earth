"""The static tier's three Core spellings, and the dispatch tables that have to reach them (order 27a).

``field``, ``points`` and ``polygons`` sat in ``PLANNED_RENAMES`` — agreed and unadopted — from the day the
Core contract was frozen. Order 27a adopts them, and the spellings they replaced are **deleted** rather than
aliased: nothing here is released, so there is no caller to keep a promise to.

`tests/test_contract_names.py` holds the *shape* of that — the name exists on the facade — off the class,
without calling anything. What it cannot ask is whether the name still **draws**, and whether the tables that
resolve a ``kind=`` token still land on a method that exists. Both are what a rename actually risks: a
dispatch entry left on a spelling that has gone now reaches nothing at all, and the only symptom is an
``AttributeError`` from inside the package. So the Core names are drawn here for real, and both dispatch
tables are enumerated rather than sampled.
"""

import geopandas as gpd
import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference
from pyramids.feature import FeatureCollection
from shapely.geometry import Point, Polygon

from digitalearth.api import _STATIC_RASTER_KINDS, quickmap
from digitalearth.static import Map
from digitalearth.static.maps.animation import _ANIMATION_KINDS, _KIND_METHODS

#: The recipe key each Core name records under `symbology.props["via"]`. **Unchanged by the rename**, which is
#: the claim `base/contract.py` makes for it: a figure written before the rename names `via="imshow"`, and the
#: drawer table is keyed by that, so it has to keep reading back into the same drawer.
VIA = {"field": "imshow", "points": "scatter", "polygons": "shapes"}

#: The raster ``kind=`` tokens `quickmap` accepts — ``"auto"`` plus matplotlib's own four render names, which
#: is the vocabulary the module-level wrappers in `api` inject. Written here because `api` exposes the
#: wrappers and not the vocabulary, and the check below is about every token being resolvable rather than
#: about any one of them.
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
    """Draw one layer and report what reached the engine and what was recorded.

    Args:
        spelling: The method to call.
        core: The Core name, which chooses the data.

    Returns:
        `(artist type name, layer kind, recipe key, count of artists on the axes)`. Deliberately not the
        artist itself: what has to be right is what kind of thing landed and what the figure says about it.
    """
    scene = Map(crs=4326)
    try:
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
    """The Core names draw, and record what the drawer table is keyed by."""

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

    @pytest.mark.parametrize("core", sorted(VIA))
    def test_the_spelling_the_core_name_replaced_is_gone(self, core):
        """The old spelling is deleted, not kept as a second name.

        Args:
            core: The Core name whose predecessor must be absent.

        Test scenario:
            A method left behind under both names would pass every other check here and go on teaching the
            vocabulary the contract exists to end. The predecessor is read out of :data:`VIA`, which records
            it as the recipe key, so the two cannot drift apart.
        """
        assert not hasattr(Map, VIA[core]), (
            f"Map still answers to {VIA[core]!r}, the spelling Map.{core}() replaced"
        )


class TestEveryDispatchTableReachesALiveMethod:
    """A rename has call sites, and the ones inside the package are the ones a caller cannot see.

    ``quickmap`` dispatches by input type and by a ``kind=`` token, and ``animate``/``rotate`` dispatch by the
    same kind vocabulary. Every one of those reaches a method by name, so a table entry left on a spelling
    that has been deleted raises an ``AttributeError`` from inside the package — the "fixed the instance,
    described the class as fixed" failure. The two tables that do the resolving are enumerated here rather
    than sampled, and the dispatch is then exercised for real.
    """

    @pytest.mark.parametrize("kind", sorted(set(_ANIMATION_KINDS)))
    def test_every_animation_kind_resolves_to_a_live_method(self, kind):
        """An animation ``kind`` must name a method the tier has.

        Args:
            kind: The kind token under test.
        """
        resolved, _ = _KIND_METHODS.get(kind, (kind, {}))
        assert hasattr(Map, resolved), (
            f"animation kind {kind!r} resolves to Map.{resolved}(), which does not exist; map it in "
            "_KIND_METHODS"
        )

    @pytest.mark.parametrize("kind", sorted(QUICKMAP_KINDS))
    def test_every_quickmap_raster_kind_resolves_to_a_live_method(self, kind):
        """The same for the ``kind=`` vocabulary `quickmap` and its module-level wrappers accept.

        Args:
            kind: The kind token under test.
        """
        resolved, _ = _STATIC_RASTER_KINDS.get(kind, (kind, {}))
        assert hasattr(Map, resolved), (
            f"quickmap kind {kind!r} resolves to Map.{resolved}(), which does not exist; map it in "
            "_STATIC_RASTER_KINDS"
        )

    @pytest.mark.parametrize("core", sorted(VIA))
    def test_quickmap_routes_each_input_type_to_its_builder(self, core):
        """And the dispatch itself, run for real: input type chooses the builder that draws.

        Args:
            core: The Core name whose input type `quickmap` must route to.

        Test scenario:
            The table checks above read the resolution; this runs it. `quickmap` picks `polygons` for polygon
            input, `points` for anything else, and the ``kind`` method for a raster — three call sites the
            rename moved. Measured on the recipe the drawn layer records, so routing to a *different* live
            builder fails rather than passing for having drawn something.
        """
        scene = quickmap(DATA[core](), colorbar=False)
        try:
            figure = scene.figure_spec
            layer = figure.layers.get(figure.layers.ids[-1])
            recorded = layer.symbology.props.get("via")
        finally:
            scene.close()
        assert recorded == VIA[core], (
            f"quickmap on {core} input drew via={recorded!r}, not {VIA[core]!r}"
        )
