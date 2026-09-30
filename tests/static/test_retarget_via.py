"""``_CANONICAL_VIA`` names a real, plain recipe for each ambiguous kind — the static-tier drift guard.

The table is the only per-tier knowledge :func:`~digitalearth.static.renderer.retarget_via` adds for cross-tier
replay (U-6); a typo in it would silently retarget a foreign ``via`` to a recipe this tier does not have, or
cover a kind that needs no entry. These hold it to :func:`~digitalearth.static.renderer._recipes`.
"""

from dataclasses import replace

from digitalearth.static import Map
from digitalearth.static.renderer import _CANONICAL_VIA, _recipes, retarget_via


def _forge_via(figure, layer_id, via):
    """Return ``figure`` with one layer's recorded ``via`` replaced — a stand-in for a foreign tier's recipe.

    Args:
        figure: The figure to rewrite.
        layer_id: The layer whose ``via`` to forge.
        via: The recipe name to record, as another tier would.

    Returns:
        A new figure whose ``layer_id`` records ``via``; the source and everything else is unchanged.
    """
    layer = figure.layers.get(layer_id)
    props = {**dict(layer.symbology.props), "via": via}
    forged = replace(layer, symbology=replace(layer.symbology, props=props))
    return replace(figure, layers=figure.layers.replace(forged))


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


def test_every_multi_recipe_kind_has_a_canonical_entry():
    """Every kind drawn more than one way is named in ``_CANONICAL_VIA``, so cross-tier retarget cannot
    silently lose one. A new multi-recipe kind (or a new recipe making a single-recipe kind multi) without an
    entry would make :func:`retarget_via` return ``None``, keep the foreign ``via`` and be refused at dispatch
    — a regression this guard turns into a failing test. The static tier has no deliberate exclusions.
    """
    recipes = _recipes()
    multi = {kind for kind, drawers in recipes.items() if len(drawers) > 1}
    missing = multi - set(_CANONICAL_VIA)
    assert not missing, (
        f"multi-recipe kinds with no _CANONICAL_VIA entry: {sorted(missing)}"
    )


class TestRetargetVia:
    """``retarget_via`` rewrites a foreign recipe to this tier's own, and leaves a known one alone."""

    def test_a_figure_this_tier_described_passes_through_unchanged(self, dataset):
        """A figure whose recipes this tier already knows is returned as the same object, untouched.

        Args:
            dataset: The raster fixture, drawn as a field (recipe ``imshow``).

        Test scenario:
            A same-tier round trip must not be perturbed — every ``via`` is already known, so nothing is
            rewritten and the identical object comes back.
        """
        source = Map(crs=dataset.epsg)
        source.field(dataset)
        figure = source.figure_spec
        assert retarget_via(figure) is figure, (
            "a known-recipe figure should be returned unchanged"
        )

    def test_a_foreign_single_recipe_via_maps_to_the_sole_recipe(self, dataset):
        """A ``via`` from another tier on a one-recipe kind retargets to that kind's only recipe here.

        Args:
            dataset: The raster fixture; ``raster`` is drawn one way here (``imshow``).

        Test scenario:
            A raster the interactive tier recorded as ``image`` must draw here as ``imshow`` — the single
            recipe makes the retarget unambiguous, no table needed.
        """
        source = Map(crs=dataset.epsg)
        source.field(dataset)
        layer_id = source.figure_spec.layers.ids[0]
        forged = _forge_via(source.figure_spec, layer_id, "image")
        out = retarget_via(forged)
        assert out.layers.get(layer_id).symbology.props["via"] == "imshow", (
            "a foreign raster recipe should retarget to this tier's sole raster recipe"
        )

    def test_a_foreign_multi_recipe_via_maps_through_the_canonical_table(self, points):
        """A ``via`` from another tier on a multi-recipe kind retargets to the table's plain recipe.

        Args:
            points: The point-geometry fixture, drawn as points (recipe ``scatter``).

        Test scenario:
            ``points`` is drawn several ways here, so a foreign ``geometry`` (the interactive tier's) cannot
            be resolved by count alone; ``_CANONICAL_VIA`` names ``scatter`` as the plain one, and that is what
            it retargets to.
        """
        from pyramids.feature import FeatureCollection

        source = Map(crs=4326)
        source.points(FeatureCollection(points))
        layer_id = source.figure_spec.layers.ids[0]
        forged = _forge_via(source.figure_spec, layer_id, "geometry")
        out = retarget_via(forged)
        assert out.layers.get(layer_id).symbology.props["via"] == "scatter", (
            "a foreign multi-recipe via should retarget through _CANONICAL_VIA to the plain recipe"
        )
