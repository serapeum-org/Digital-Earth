"""``_CANONICAL_VIA`` / ``retarget_via`` on the static tier — the drift guards and the retarget behaviour.

The shared logic lives in :mod:`tests.base._retarget_checks`; this passes the static tier's ``_recipes`` /
``_CANONICAL_VIA`` / ``retarget_via`` and static-built figures to it. Runs in the core ``dev`` env (matplotlib).
"""

from pyramids.feature import FeatureCollection

from digitalearth.static import Map
from digitalearth.static.renderer import _CANONICAL_VIA, _recipes, retarget_via
from tests.base._retarget_checks import (
    assert_canonical_entries_are_recipes,
    assert_every_multi_recipe_kind_is_covered,
    assert_foreign_via_retargets,
    assert_known_recipe_passes_through,
    assert_table_only_covers_multi_recipe,
)


def test_each_canonical_via_is_a_recipe_the_tier_actually_draws():
    """Every ``_CANONICAL_VIA`` value is a recipe its kind is drawn by on this tier."""
    assert_canonical_entries_are_recipes(_recipes(), _CANONICAL_VIA)


def test_the_table_only_covers_kinds_drawn_more_than_one_way():
    """A single-recipe kind needs no ``_CANONICAL_VIA`` entry."""
    assert_table_only_covers_multi_recipe(_recipes(), _CANONICAL_VIA)


def test_every_multi_recipe_kind_has_a_canonical_entry():
    """Every multi-recipe kind is named in ``_CANONICAL_VIA`` (the static tier excludes none)."""
    assert_every_multi_recipe_kind_is_covered(_recipes(), _CANONICAL_VIA)


class TestRetargetVia:
    """``retarget_via`` rewrites a foreign recipe to this tier's own, and leaves a known one alone."""

    def test_a_figure_this_tier_described_passes_through_unchanged(self, dataset):
        """A same-tier figure's recipes are all known, so it is returned unchanged.

        Args:
            dataset: The raster fixture, drawn as a field.
        """
        source = Map(crs=dataset.epsg)
        source.field(dataset)
        assert_known_recipe_passes_through(source.figure_spec, retarget_via)

    def test_a_foreign_single_recipe_via_maps_to_the_sole_recipe(self, dataset):
        """A foreign ``raster`` recipe retargets to this tier's only raster recipe, ``imshow``.

        Args:
            dataset: The raster fixture; ``raster`` is drawn one way here.
        """
        source = Map(crs=dataset.epsg)
        source.field(dataset)
        layer_id = source.figure_spec.layers.ids[0]
        assert_foreign_via_retargets(
            source.figure_spec, layer_id, "image", "imshow", retarget_via
        )

    def test_a_foreign_multi_recipe_via_maps_through_the_canonical_table(self, points):
        """A foreign ``points`` recipe retargets through ``_CANONICAL_VIA`` to the plain ``scatter``.

        Args:
            points: The point-geometry fixture, drawn as points (drawn several ways here).
        """
        source = Map(crs=4326)
        source.points(FeatureCollection(points))
        layer_id = source.figure_spec.layers.ids[0]
        assert_foreign_via_retargets(
            source.figure_spec, layer_id, "geometry", "scatter", retarget_via
        )
