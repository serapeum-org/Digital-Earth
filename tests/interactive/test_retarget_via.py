"""``_CANONICAL_VIA`` names a real, plain recipe for each ambiguous kind — the interactive-tier drift guard.

Runs in the ``interactive`` environment, because :func:`~digitalearth.interactive.renderer._recipes` imports the
HoloViz builders. See the static counterpart for what this guards (U-6's cross-tier ``retarget_via``).
"""

from digitalearth.interactive.renderer import _CANONICAL_VIA, _recipes


def test_each_canonical_via_is_a_recipe_the_tier_actually_draws():
    """Every ``_CANONICAL_VIA`` value is a recipe its kind is drawn by on this tier."""
    recipes = _recipes()
    for kind, via in _CANONICAL_VIA.items():
        assert kind in recipes, f"{kind!r} is not a kind this tier draws"
        assert via in recipes[kind], (
            f"{via!r} is not a {kind!r} recipe; this tier draws it by {sorted(recipes[kind])}"
        )


def test_the_table_only_covers_kinds_drawn_more_than_one_way():
    """A kind with a single recipe needs no entry — its retarget is unambiguous without the table."""
    recipes = _recipes()
    for kind in _CANONICAL_VIA:
        assert len(recipes[kind]) > 1, (
            f"{kind!r} has one recipe, so _CANONICAL_VIA should not name it"
        )
