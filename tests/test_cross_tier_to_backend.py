"""``to_backend`` carries a figure described by one 2-D tier onto any other 2-D tier — the U-6 headline (order 33).

One ``FigureSpec``, drawn on a different backend from the one that described it. The module needs both non-core
2-D engines to build the source maps, so it skips unless both import (it runs in the ``all`` env /
``test-backends`` job, and skips in the lean ``dev`` matrix and the single-backend jobs).

Every layer carries a kind, a source and the portable half of its style; a tier's drawer fills the rest from
its own defaults, and the foreign recipe name (``via``) is retargeted to the drawing tier's own (see
:func:`digitalearth.static.renderer.retarget_via`). So a figure one 2-D tier described draws on any other 2-D
tier. The 3-D tier is not part of this matrix — it draws terrain/volumes, not the flat fields and vectors these
carry — and is covered same-tier by ``tests/three_d``.
"""

import pytest

pytest.importorskip("maplibre", reason="web tier engine")
pytest.importorskip("geoviews", reason="interactive tier engine")

from digitalearth import to_backend  # noqa: E402

#: The three 2-D tiers, each of which can both describe a figure and draw one another tier described.
TIERS = ("matplotlib", "interactive", "web")


def _new_scene(tier: str):
    """Return an empty map of ``tier`` in EPSG:4326 (the CRS every tier accepts).

    Args:
        tier: The backend to build.

    Returns:
        An empty tier scene.
    """
    if tier == "matplotlib":
        from digitalearth.static import Map

        return Map(crs=4326)
    if tier == "interactive":
        from digitalearth.interactive import InteractiveMap

        return InteractiveMap(crs=4326)
    from digitalearth.web import WebMap

    return WebMap(crs=4326)


def _build(tier: str, kind: str, dataset, features):
    """Build a one-layer map of ``tier`` holding a layer of ``kind``.

    Args:
        tier: The source backend.
        kind: ``"field"`` (a raster) or ``"points"`` (a vector point layer).
        dataset: The raster fixture, for ``"field"``.
        features: A pyramids ``FeatureCollection`` of points, for ``"points"``.

    Returns:
        The built tier scene, with one layer.
    """
    scene = _new_scene(tier)
    if kind == "field":
        scene.field(dataset)
    else:
        scene.points(features)
    return scene


def _close(scene) -> None:
    """Release a scene, tolerating a tier without ``close`` (the static figure is closed by conftest)."""
    try:
        scene.close()
    except (
        Exception
    ):  # pragma: no cover - a tier may have no close(), or refuse a second
        pass


@pytest.mark.parametrize("kind", ["field", "points"])
@pytest.mark.parametrize("target", TIERS)
@pytest.mark.parametrize("source", TIERS)
def test_a_layer_described_by_one_2d_tier_draws_on_another(
    source, target, kind, dataset, points
):
    """A ``kind`` layer carried from ``source`` to ``target`` draws there as the same layer.

    Args:
        source: The tier that describes the figure.
        target: The tier ``to_backend`` draws it on.
        kind: The layer kind under test (``field`` or ``points``).
        dataset: The raster fixture.
        points: The point-geometry fixture (a GeoDataFrame), wrapped in a ``FeatureCollection``.

    Test scenario:
        The full 2-D matrix, same-tier included. The source is kept alive across the carry so its in-memory
        source stays readable. ``to_backend`` retargets the recipe name and each tier's drawer fills the style
        a portable figure does not carry from its own defaults, so the layer draws rather than being refused;
        the layer's id and kind must survive.
    """
    from pyramids.feature import FeatureCollection

    features = FeatureCollection(points)
    origin = _build(source, kind, dataset, features)
    try:
        replayed = to_backend(origin.figure_spec, backend=target)
        try:
            assert replayed.layer_ids == origin.layer_ids, (
                f"{source}->{target} [{kind}]: {replayed.layer_ids} from {origin.layer_ids}"
            )
            kinds_before = [origin.get_layer(i).kind for i in origin.layer_ids]
            kinds_after = [replayed.get_layer(i).kind for i in replayed.layer_ids]
            assert kinds_after == kinds_before, (
                f"{source}->{target} [{kind}]: kinds {kinds_after} from {kinds_before}"
            )
        finally:
            _close(replayed)
    finally:
        _close(origin)
