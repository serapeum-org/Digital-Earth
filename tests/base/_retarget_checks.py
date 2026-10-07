"""Shared assertions for the per-tier ``retarget_via`` / ``_CANONICAL_VIA`` tests (U-6 cross-tier).

The static and interactive tiers each have a ``retarget_via`` and a ``_CANONICAL_VIA`` table with identical
contracts, so their test *logic* is the same — only the tier's ``_recipes`` / ``_CANONICAL_VIA`` / scenes
differ. These helpers hold that logic once; each tier's ``test_retarget_via`` passes in its own objects. Not a
test module itself (no ``test_`` names) and engine-free — it only manipulates a ``FigureSpec`` handed to it, so
both the matplotlib-env and HoloViz-env suites import it.
"""

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any


def forge_via(figure: Any, layer_id: str, via: Any) -> Any:
    """Return ``figure`` with one layer's recorded ``via`` replaced — a stand-in for a foreign tier's recipe.

    Args:
        figure: The figure to rewrite.
        layer_id: The layer whose ``via`` to forge.
        via: The recipe name to record, as another tier would.

    Returns:
        A new figure whose ``layer_id`` records ``via``; everything else is unchanged.
    """
    layer = figure.layers.get(layer_id)
    props = {**dict(layer.symbology.props), "via": via}
    forged = replace(layer, symbology=replace(layer.symbology, props=props))
    return replace(figure, layers=figure.layers.replace(forged))


def assert_canonical_entries_are_recipes(
    recipes: Mapping[str, Mapping[str, Any]], canonical: Mapping[str, str]
) -> None:
    """Every ``canonical`` value is a recipe its kind is actually drawn by on this tier."""
    for kind, via in canonical.items():
        assert kind in recipes, f"{kind!r} is not a kind this tier draws"
        assert via in recipes[kind], (
            f"{via!r} is not a {kind!r} recipe; this tier draws it by {sorted(recipes[kind])}"
        )


def assert_table_only_covers_multi_recipe(
    recipes: Mapping[str, Mapping[str, Any]], canonical: Mapping[str, str]
) -> None:
    """A kind with a single recipe needs no entry — its retarget is unambiguous without the table."""
    for kind in canonical:
        assert len(recipes[kind]) > 1, (
            f"{kind!r} has one recipe, so _CANONICAL_VIA should not name it"
        )


def assert_every_multi_recipe_kind_is_covered(
    recipes: Mapping[str, Mapping[str, Any]],
    canonical: Mapping[str, str],
    allowed: frozenset[str] = frozenset(),
) -> None:
    """Every kind drawn more than one way is in ``canonical`` or in the deliberate-exclusion ``allowed`` set.

    A new multi-recipe kind (or a new recipe making a single-recipe kind multi) without an entry would make
    ``retarget_via`` return the foreign ``via`` unchanged and be refused downstream, with no failing test.
    """
    multi = {kind for kind, drawers in recipes.items() if len(drawers) > 1}
    missing = multi - set(canonical) - allowed
    assert not missing, (
        f"multi-recipe kinds with no _CANONICAL_VIA entry and not excluded: {sorted(missing)}"
    )


def assert_known_recipe_passes_through(
    figure: Any, retarget_via: Callable[[Any], Any]
) -> None:
    """A figure whose recipes this tier already knows is returned as the same object, untouched."""
    assert retarget_via(figure) is figure, (
        "a known-recipe figure should be returned unchanged"
    )


def assert_foreign_via_retargets(
    figure: Any,
    layer_id: str,
    foreign_via: str,
    expected_via: str,
    retarget_via: Callable[[Any], Any],
) -> None:
    """A foreign ``via`` on ``layer_id`` retargets to this tier's ``expected_via`` for the kind."""
    forged = forge_via(figure, layer_id, foreign_via)
    out = retarget_via(forged)
    actual = out.layers.get(layer_id).symbology.props["via"]
    assert actual == expected_via, (
        f"foreign via {foreign_via!r} retargeted to {actual!r}, expected {expected_via!r}"
    )
