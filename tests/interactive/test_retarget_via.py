"""``_CANONICAL_VIA`` names a real, plain recipe for each ambiguous kind — the interactive-tier drift guard.

Runs in the ``interactive`` environment, because :func:`~digitalearth.interactive.renderer._recipes` imports the
HoloViz builders. See the static counterpart for what this guards (U-6's cross-tier ``retarget_via``).
"""

from dataclasses import replace

from digitalearth.interactive import InteractiveMap
from digitalearth.interactive.renderer import _CANONICAL_VIA, _recipes, retarget_via


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


#: Multi-recipe kinds deliberately left out of ``_CANONICAL_VIA`` — the plain recipe is a genuine judgement, so
#: a foreign ``via`` for one is refused rather than guessed. ``heatmap`` is hexbin vs kde.
_ALLOWED_UNMAPPED = frozenset({"heatmap"})


def test_every_multi_recipe_kind_has_a_canonical_entry_or_is_deliberately_excluded():
    """Every kind drawn more than one way is either in ``_CANONICAL_VIA`` or in the deliberate-exclusion set,
    so a new multi-recipe kind cannot silently lose cross-tier support: without an entry :func:`retarget_via`
    would return ``None``, keep the foreign ``via`` and be refused at dispatch, with no failing test to catch
    it. A kind added to the exclusion set is an explicit choice, recorded here.
    """
    recipes = _recipes()
    multi = {kind for kind, drawers in recipes.items() if len(drawers) > 1}
    missing = multi - set(_CANONICAL_VIA) - _ALLOWED_UNMAPPED
    assert not missing, (
        f"multi-recipe kinds with no _CANONICAL_VIA entry and not excluded: {sorted(missing)}"
    )


class TestRetargetVia:
    """``retarget_via`` rewrites a foreign recipe to this tier's own, and leaves a known one alone."""

    def test_a_figure_this_tier_described_passes_through_unchanged(self, dataset):
        """A figure whose recipes this tier already knows is returned as the same object, untouched.

        Args:
            dataset: The raster fixture, drawn as a field (recipe ``image``).
        """
        source = InteractiveMap(crs=4326)
        source.field(dataset)
        figure = source.figure_spec
        assert retarget_via(figure) is figure, (
            "a known-recipe figure should be returned unchanged"
        )

    def test_a_foreign_single_recipe_via_maps_to_the_sole_recipe(self):
        """A ``via`` from another tier on a one-recipe kind retargets to that kind's only recipe here.

        Test scenario:
            ``graticule`` is drawn one way here, so any unknown recorded ``via`` retargets to ``graticule``
            by count alone — no table needed.
        """
        source = InteractiveMap(crs=4326)
        source.graticule(lon_step=30.0, lat_step=30.0)
        layer_id = source.figure_spec.layers.ids[0]
        forged = _forge_via(source.figure_spec, layer_id, "not-a-recipe")
        out = retarget_via(forged)
        assert out.layers.get(layer_id).symbology.props["via"] == "graticule", (
            "a foreign single-recipe via should retarget to this tier's sole recipe for the kind"
        )

    def test_a_foreign_multi_recipe_via_maps_through_the_canonical_table(self, points):
        """A ``via`` from another tier on a multi-recipe kind retargets to the table's plain recipe.

        Args:
            points: The point-geometry fixture, drawn as points (recipe ``geometry``).

        Test scenario:
            ``points`` is drawn several ways here (``geometry``/``datashade``), so a foreign ``scatter`` (the
            static tier's) cannot be resolved by count; ``_CANONICAL_VIA`` names ``geometry`` as the plain one.
        """
        from pyramids.feature import FeatureCollection

        source = InteractiveMap(crs=4326)
        source.points(FeatureCollection(points))
        layer_id = source.figure_spec.layers.ids[0]
        forged = _forge_via(source.figure_spec, layer_id, "scatter")
        out = retarget_via(forged)
        assert out.layers.get(layer_id).symbology.props["via"] == "geometry", (
            "a foreign multi-recipe via should retarget through _CANONICAL_VIA to the plain recipe"
        )
