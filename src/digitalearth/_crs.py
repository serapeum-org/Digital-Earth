"""Tiny CRS helper — the best-effort EPSG lookup shared by the extractors and the map projection code.

A leaf module (no pyramids/cleopatra/matplotlib import) so both :mod:`digitalearth.sources.extractors` and
:mod:`digitalearth.scene.map` can resolve a vector layer's EPSG the same way.
"""
from typing import Any, Optional


def source_epsg(features: Any, default: Optional[int] = None) -> Optional[int]:
    """Best-effort EPSG code of a ``FeatureCollection`` / ``GeoDataFrame``.

    Prefers the pyramids ``.epsg`` attribute, falls back to deriving a code from ``.crs`` (``crs.to_epsg()``),
    and finally returns ``default`` when neither yields a code.

    Args:
        features: A pyramids ``FeatureCollection`` or a geopandas ``GeoDataFrame``/``GeoSeries``.
        default: Value returned when no EPSG code can be resolved (e.g. ``None`` for "unknown", or ``4326``
            to assume lon/lat).

    Returns:
        The resolved EPSG integer, or ``default`` when none is available.
    """
    epsg = getattr(features, "epsg", None)
    if epsg is not None:
        return epsg
    crs = getattr(features, "crs", None)
    if crs is not None:
        code = crs.to_epsg()
        if code is not None:
            return code
    return default
