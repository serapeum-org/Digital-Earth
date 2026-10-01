"""``_CANONICAL_VIA`` / ``retarget_via`` / ``_HV_TYPE_FOR_KIND`` on the interactive tier.

The recipe-table drift guards and the retarget behaviour share their logic with the static tier via
:mod:`tests.base._retarget_checks`; this passes the interactive tier's objects and HoloViz-built figures to it,
and adds the tier-specific ``_HV_TYPE_FOR_KIND`` guard. Runs in the ``interactive`` env.
"""

import pytest

pytest.importorskip(
    "geoviews"
)  # the interactive tier engine; _recipes imports the HoloViz builders

from pyramids.feature import FeatureCollection  # noqa: E402

from digitalearth.interactive import InteractiveMap, vector  # noqa: E402
from digitalearth.interactive.renderer import (  # noqa: E402
    _CANONICAL_VIA,
    _recipes,
    retarget_via,
)
from digitalearth.interactive.vector import _HV_TYPE_FOR_KIND  # noqa: E402
from tests.base._retarget_checks import (  # noqa: E402
    assert_canonical_entries_are_recipes,
    assert_every_multi_recipe_kind_is_covered,
    assert_foreign_via_retargets,
    assert_known_recipe_passes_through,
    assert_table_only_covers_multi_recipe,
)

#: ``heatmap`` (hexbin vs kde) deliberately left out of ``_CANONICAL_VIA`` — the plain recipe is a judgement.
_ALLOWED_UNMAPPED = frozenset({"heatmap"})


def test_each_canonical_via_is_a_recipe_the_tier_actually_draws():
    """Every ``_CANONICAL_VIA`` value is a recipe its kind is drawn by on this tier."""
    assert_canonical_entries_are_recipes(_recipes(), _CANONICAL_VIA)


def test_the_table_only_covers_kinds_drawn_more_than_one_way():
    """A single-recipe kind needs no ``_CANONICAL_VIA`` entry."""
    assert_table_only_covers_multi_recipe(_recipes(), _CANONICAL_VIA)


def test_every_multi_recipe_kind_has_a_canonical_entry_or_is_deliberately_excluded():
    """Every multi-recipe kind is in ``_CANONICAL_VIA`` or the deliberate-exclusion set (``heatmap``)."""
    assert_every_multi_recipe_kind_is_covered(
        _recipes(), _CANONICAL_VIA, _ALLOWED_UNMAPPED
    )


def test_hv_type_covers_every_kind_draw_vector_serves():
    """Every kind whose recipe is ``draw_vector`` has a ``_HV_TYPE_FOR_KIND`` entry.

    ``draw_vector`` derives a foreign layer's HoloViews element type from its kind via this table; a kind it
    draws with no entry would raise a cryptic ``KeyError`` only on the cross-tier path, invisible to same-tier
    tests. Holds the table against the kinds ``_recipes`` routes to ``draw_vector``.
    """
    routed = {
        kind
        for kind, recipes in _recipes().items()
        if any(drawer is vector.draw_vector for drawer in recipes.values())
    }
    missing = routed - set(_HV_TYPE_FOR_KIND)
    assert not missing, (
        f"kinds drawn by draw_vector with no _HV_TYPE_FOR_KIND entry: {sorted(missing)}"
    )


class TestRetargetVia:
    """``retarget_via`` rewrites a foreign recipe to this tier's own, and leaves a known one alone."""

    def test_a_figure_this_tier_described_passes_through_unchanged(self, dataset):
        """A same-tier figure's recipes are all known, so it is returned unchanged.

        Args:
            dataset: The raster fixture, drawn as a field.
        """
        source = InteractiveMap(crs=4326)
        source.field(dataset)
        assert_known_recipe_passes_through(source.figure_spec, retarget_via)

    def test_a_foreign_single_recipe_via_maps_to_the_sole_recipe(self):
        """A foreign recipe on a one-recipe kind (``graticule``) retargets to that sole recipe."""
        source = InteractiveMap(crs=4326)
        source.graticule(lon_step=30.0, lat_step=30.0)
        layer_id = source.figure_spec.layers.ids[0]
        assert_foreign_via_retargets(
            source.figure_spec, layer_id, "not-a-recipe", "graticule", retarget_via
        )

    def test_a_foreign_multi_recipe_via_maps_through_the_canonical_table(self, points):
        """A foreign ``points`` recipe retargets through ``_CANONICAL_VIA`` to the plain ``geometry``.

        Args:
            points: The point-geometry fixture, drawn as points (drawn several ways here).
        """
        source = InteractiveMap(crs=4326)
        source.points(FeatureCollection(points))
        layer_id = source.figure_spec.layers.ids[0]
        assert_foreign_via_retargets(
            source.figure_spec, layer_id, "scatter", "geometry", retarget_via
        )
