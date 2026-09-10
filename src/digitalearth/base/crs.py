"""Tiny CRS helper — the best-effort EPSG lookup, and the shared "nothing landed on the view" signal.

Both are engine-neutral: the lookup reads a CRS off whatever pyramids hands over, and the reprojection
guard is a string match on what GDAL says plus an exception type. Every backend warps to a display CRS,
so every backend can hit the same failure, and they answer it the same way.
"""

import re
from typing import Any, Optional

__all__ = ["source_epsg"]


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


#: GDAL's complaint when a warp cannot place the data in the target CRS. It fires as soon as too few
#: sample points survive to bound an output — its own threshold is ``failed > total - 10``, not all of
#: them — and by then it has already refused to compute those bounds. So *any* occurrence means there is
#: no output raster, whatever the counts say; a warp that does produce output never raises it, which is
#: why the counts are not read here.
_POINTS_FAILED = re.compile(r"Too many points \(\d+ out of \d+\) failed to transform")


class OffLimbError(RuntimeError):
    """The data lies outside the area the display CRS can represent.

    Raised in place of GDAL's opaque "Too many points ... failed to transform", which on an orthographic
    globe means the data sits behind the visible limb. Layer methods treat it as "there is nothing to draw
    here" and render an empty frame; it is a distinct type so that a caller can tell it apart from a real
    projection failure.
    """


def reproject(dataset: Any, crs: Any) -> Any:
    """Reproject ``dataset`` to ``crs``, reporting an empty view as :class:`OffLimbError`.

    Args:
        dataset: A pyramids ``Dataset`` (anything with ``to_crs``).
        crs: The display CRS to warp into.

    Returns:
        The reprojected dataset.

    Raises:
        OffLimbError: when the warp reports too few surviving sample points to bound an output, i.e. the
            data is outside the projection's visible area.
        RuntimeError: any other warp failure, re-raised as it came — a real projection error must not be
            mistaken for an empty view.
    """
    try:
        return dataset.to_crs(crs)
    except RuntimeError as error:
        if _POINTS_FAILED.search(str(error)):
            raise OffLimbError(
                f"the data lies outside what {crs!r} can show: too few sample points survive the warp "
                f"for it to produce any output"
            ) from error
        raise
