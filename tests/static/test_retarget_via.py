"""``_CANONICAL_VIA`` names a real, plain recipe for each ambiguous kind — the static-tier drift guard.

The table is the only per-tier knowledge :func:`~digitalearth.static.renderer.retarget_via` adds for cross-tier
replay (U-6); a typo in it would silently retarget a foreign ``via`` to a recipe this tier does not have, or
cover a kind that needs no entry. These hold it to :func:`~digitalearth.static.renderer._recipes`.
"""

from digitalearth.static.renderer import _CANONICAL_VIA, _recipes


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
