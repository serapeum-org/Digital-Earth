"""``to_backend`` carries a figure described by one tier into the interactive tier — the U-6 headline (order 33).

One ``FigureSpec``, drawn on a different backend from the one that described it. The module needs both non-core
2-D engines to build the source maps, so it skips unless both import (it runs in the ``all`` env /
``test-backends`` job, and skips in the lean ``dev`` matrix and the single-backend jobs).

The **cross-tier target is the interactive tier**: its drawers fall back to their own defaults for the style a
portable figure does not carry, and the foreign recipe name (``via``) is retargeted to its own — see
:func:`digitalearth.interactive.renderer.retarget_via`. The web, matplotlib and 3-D tiers read a layer's style
from ``props`` their own builders write and a portable figure does not carry (``NO_PORTABLE_CHANNELS``), so a
figure made elsewhere cannot be drawn on them in this cut; they are same-tier only, covered by their own suites.
"""

import pytest

pytest.importorskip(
    "maplibre", reason="web tier engine (to build a web-described source)"
)
pytest.importorskip("geoviews", reason="interactive tier engine")

from digitalearth import to_backend  # noqa: E402

#: The three 2-D tiers a figure can be described *by* — each becomes a source carried into the interactive tier.
SOURCES = ("matplotlib", "interactive", "web")

#: The tier a figure described elsewhere can be drawn *on* in this cut. The web, matplotlib and 3-D tiers are
#: same-tier only (see the module docstring), so they are not cross-tier targets here.
TARGETS = ("interactive",)


def _build(tier: str, dataset):
    """Build a one-layer map of ``tier`` — a raster field — for the cross-tier carry.

    Args:
        tier: The source backend.
        dataset: The raster fixture drawn as a field.

    Returns:
        The built tier scene, with one ``raster`` layer.
    """
    if tier == "matplotlib":
        from digitalearth.static import Map

        scene = Map(crs=dataset.epsg)
    elif tier == "interactive":
        from digitalearth.interactive import InteractiveMap

        scene = InteractiveMap(crs=3857)
    else:
        from digitalearth.web import WebMap

        scene = WebMap(crs=4326)
    scene.field(dataset)
    return scene


def _close(scene) -> None:
    """Release a scene, tolerating a tier without ``close`` (the static figure is closed by conftest)."""
    try:
        scene.close()
    except (
        Exception
    ):  # pragma: no cover - a tier may have no close(), or refuse a second
        pass


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("source", SOURCES)
def test_a_field_described_by_one_tier_draws_on_another(source, target, dataset):
    """A raster field carried from ``source`` to ``target`` draws there as the same layer.

    Args:
        source: The tier that describes the figure.
        target: The tier ``to_backend`` draws it on.
        dataset: The raster fixture.

    Test scenario:
        The source is kept alive across the carry so its in-memory source stays readable (an ``object:``
        source is not stored, only replayed in process). ``to_backend`` retargets the source tier's recipe
        name to the target's, so the layer draws rather than being refused for an unknown ``via``; the layer's
        id and kind are what must survive (style a ``via`` cannot carry is out of scope, as ``to_backend``
        states).
    """
    origin = _build(source, dataset)
    try:
        replayed = to_backend(origin.figure_spec, backend=target)
        try:
            assert replayed.layer_ids == origin.layer_ids, (
                f"{source}->{target}: {replayed.layer_ids} from {origin.layer_ids}"
            )
            kinds_before = [origin.get_layer(i).kind for i in origin.layer_ids]
            kinds_after = [replayed.get_layer(i).kind for i in replayed.layer_ids]
            assert kinds_after == kinds_before, (
                f"{source}->{target}: kinds {kinds_after} from {kinds_before}"
            )
        finally:
            _close(replayed)
    finally:
        _close(origin)
