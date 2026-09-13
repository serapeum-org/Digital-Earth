"""Tiny CRS helpers — the code/definition lookups, the geographic test, and the "nothing landed" signal.

All of them are engine-neutral: the lookups read a CRS off whatever pyramids hands over, the geographic
test and the authority-code read ask **pyramids** to interpret it (never coordinate magnitudes, never
string parsing here), and the reprojection guard reads what the warp produced. Every backend warps to a
display CRS, so every backend can hit the same questions, and they answer them the same way.

The "nothing landed" signal has to be raised from two different readings, because the two data families
fail differently: a raster warp *raises* (GDAL refuses to bound an output), while a vector warp *succeeds*
and hands back non-finite coordinates. :func:`reproject` covers both, so :class:`OffLimbError` means the
same thing whichever kind of layer a backend was drawing.
"""

import re
from typing import Any, Optional

import numpy as np
from pyramids.base.crs import crs_from_user_input

__all__ = [
    "OffLimbError",
    "authority_code",
    "declared_crs",
    "is_geographic",
    "reproject",
    "source_epsg",
]


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


def declared_crs(data: Any) -> Any:
    """The CRS an input says its coordinates are in: its EPSG code, else its projection definition.

    The reader behind :attr:`~digitalearth.base.sources.Source.crs` when no caller names one. pyramids
    reports ``epsg is None`` for a projection carrying no authority code — exactly what a warp into an
    orthographic display CRS produces — so falling back to the definition keeps the answer able to say
    where the coordinates live instead of claiming the CRS is unknown.

    Args:
        data: A pyramids ``Dataset``/``NetCDF`` (duck-typed by ``epsg`` / ``crs``).

    Returns:
        The EPSG integer, the projection definition (WKT), or ``None`` when the input declares no CRS
        at all — the one case that genuinely means "unknown".

    Examples:
        - A coded raster answers with its code:
            ```python
            >>> from pyramids.dataset import Dataset
            >>> from digitalearth.base.crs import declared_crs
            >>> declared_crs(Dataset.read_file("examples/data/acc4000.tif"))
            32618

            ```
        - Anything declaring no CRS at all is unknown:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.crs import declared_crs
            >>> declared_crs(np.zeros((2, 2))) is None
            True

            ```
    """
    epsg = getattr(data, "epsg", None)
    if epsg is not None:
        return epsg
    return getattr(data, "crs", None) or None


def is_geographic(crs: Any) -> Optional[bool]:
    """Whether ``crs`` is a geographic (lon/lat) CRS — in **every** spelling the ``Source`` contract allows.

    The CRS is interpreted by pyramids (``crs_from_user_input``), the GIS engine, so each of the forms
    :attr:`~digitalearth.base.sources.Source.crs` may carry reads the same way: an EPSG ``int``, an
    ``"EPSG:<code>"`` string in either case, a proj4 string, or a WKT definition. Nothing is inferred from
    coordinate magnitudes and nothing is parsed out of the string here — a reader that only understood the
    ``int`` spelling answered "unknown" for a projected CRS written ``"EPSG:3857"``, which let
    :meth:`digitalearth.three_d.globe.GlobeMixin.globe` drape Web-Mercator metres on a sphere.

    An unreadable CRS is reported as "unknown" (``None``) rather than guessed at, so a caller can tell
    "definitely projected" apart from "no CRS to go on" and answer the two differently.

    Args:
        crs: A CRS in any form pyramids accepts, or ``None``/an unreadable value for "no CRS".

    Returns:
        ``True`` for a geographic CRS, ``False`` for a projected one, ``None`` when it cannot be resolved.

    Examples:
        - Every spelling of one code answers alike, and a projected CRS is known to be projected:
            ```python
            >>> from digitalearth.base.crs import is_geographic
            >>> is_geographic(4326), is_geographic("EPSG:4326"), is_geographic("epsg:4326")
            (True, True, True)
            >>> is_geographic(3857), is_geographic("EPSG:3857")
            (False, False)

            ```
        - A projection with no authority code is read from its definition rather than refused:
            ```python
            >>> from digitalearth.base.crs import is_geographic
            >>> is_geographic("+proj=ortho +lat_0=53 +lon_0=4")
            False
            >>> is_geographic("+proj=longlat +datum=WGS84 +no_defs")
            True

            ```
        - Anything that is not a readable CRS is "unknown", never "projected":
            ```python
            >>> from digitalearth.base.crs import is_geographic
            >>> is_geographic(None) is None, is_geographic("not-a-crs") is None
            (True, True)

            ```

    See Also:
        declared_crs: what to feed this for a pyramids input that has not been warped by the caller.
    """
    if crs is None or isinstance(crs, bool):
        return None
    try:
        return bool(crs_from_user_input(crs).is_geographic)
    # Any CRS-resolution failure means "unknown", never "projected".
    except Exception:  # noqa: BLE001
        return None


def authority_code(crs: Any) -> Optional[int]:
    """The EPSG code pyramids reads out of ``crs``, or ``None`` when the definition carries none.

    The companion of :func:`is_geographic` — the same "let pyramids interpret it" rule, asked of the
    narrower question. A CRS written out in full still names its authority (WKT ends on an
    ``AUTHORITY["EPSG", ...]`` / ``ID["EPSG", ...]`` block), so a reader that only matched the
    ``"EPSG:<code>"`` spelling answered "no code" for a CRS that plainly has one.

    Args:
        crs: A CRS in any form pyramids accepts, or ``None``/an unreadable value.

    Returns:
        The EPSG integer, or ``None`` when the CRS is unreadable or genuinely carries no authority code.

    Examples:
        - A definition is read down to its authority block, whichever spelling it is written in:
            ```python
            >>> from pyramids.base.crs import crs_from_user_input
            >>> from digitalearth.base.crs import authority_code
            >>> authority_code(crs_from_user_input(3857).to_wkt())
            3857

            ```
        - A projection that carries no code — and anything unreadable — answers ``None``:
            ```python
            >>> from digitalearth.base.crs import authority_code
            >>> authority_code("+proj=ortho +lat_0=53 +lon_0=4") is None
            True
            >>> authority_code("not-a-crs") is None
            True

            ```

    See Also:
        digitalearth.base.sources.Source.epsg: the reader this answers for a wrapped source.
    """
    try:
        return crs_from_user_input(crs).to_epsg()
    # Any CRS-resolution failure means "no code to name", never a guess.
    except Exception:  # noqa: BLE001
        return None


#: GDAL's complaint when a warp cannot place the data in the target CRS. It fires as soon as too few
#: sample points survive to bound an output — its own threshold is ``failed > total - 10``, not all of
#: them — and by then it has already refused to compute those bounds. So *any* occurrence means there is
#: no output raster, whatever the counts say; a warp that does produce output never raises it, which is
#: why the counts are not read here.
_POINTS_FAILED = re.compile(r"Too many points \(\d+ out of \d+\) failed to transform")


class OffLimbError(RuntimeError):
    """The data lies outside the area the display CRS can represent.

    Raised in place of GDAL's opaque "Too many points ... failed to transform", which on an orthographic
    globe means the data sits behind the visible limb — and, for a vector layer, in place of the ``inf``
    coordinates a geopandas warp hands back instead of complaining at all. Layer methods treat it as "there
    is nothing to draw here" and render an empty frame; it is a distinct type so that a caller can tell it
    apart from a real projection failure.

    Examples:
        - Callers rarely see it: the layer methods answer it by drawing nothing:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from digitalearth.static import Map, projections
            >>> ds = Dataset.from_array(
            ...     np.ones((20, 20), "float32"),
            ...     geo_ref=GeoReference(geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326),
            ... )
            >>> hidden = Map(crs=projections.orthographic(lon=-175, lat=15), globe=True)
            >>> hidden.imshow(ds) is None
            True
            >>> len(hidden.ax.images)
            0

            ```
        - The matplotlib backend re-exports the very same class, for a caller that wants to catch it:
            ```python
            >>> from digitalearth.base.crs import OffLimbError
            >>> from digitalearth.static import OffLimbError as FromBackend
            >>> OffLimbError is FromBackend
            True

            ```
    """


def _placed_nowhere(data: Any) -> bool:
    """Whether ``data``'s extent has no finite coordinate in it at all.

    Reads ``total_bounds``, which pyramids exposes on both families, and asks only the one question the
    off-limb signal turns on: did *anything* land somewhere nameable. A layer with even one finite bound
    has something to draw and is left alone, so a warp that hides part of the data is not reported as if
    it had hidden all of it.

    Args:
        data: A pyramids ``Dataset`` / ``FeatureCollection`` (duck-typed by ``total_bounds``), or anything
            without that attribute.

    Returns:
        ``True`` when the extent is readable and wholly non-finite, ``False`` otherwise — including for an
        input that reports no extent at all, which says nothing either way.
    """
    try:
        bounds = getattr(data, "total_bounds", None)
        values = None if bounds is None else np.asarray(bounds, dtype="float64")
    # An unreadable extent answers nothing; never invent an off-limb report.
    except Exception:  # noqa: BLE001
        return False
    if values is None:
        return False
    return bool(values.size) and not bool(np.isfinite(values).any())


def reproject(dataset: Any, crs: Any) -> Any:
    """Reproject ``dataset`` to ``crs``, reporting an empty view as :class:`OffLimbError`.

    The two data families fail differently and both have to reach the one signal:

    * A **raster** warp raises — GDAL refuses to bound an output once too few sample points survive — so
      that wording is matched and re-raised as :class:`OffLimbError`.
    * A **vector** warp succeeds and hands back ``inf`` coordinates for every geometry the projection
      cannot place. Nothing raises, so the extent is read afterwards: an input that had a finite extent
      and comes back with none landed nowhere, which is the same condition under a different disguise.
      A layer only partly hidden keeps a finite bound and is returned untouched, and an input that had no
      finite extent to begin with (an empty collection) is not blamed on the projection.

    Args:
        dataset: A pyramids ``Dataset`` or ``FeatureCollection`` (anything with ``to_crs``).
        crs: The display CRS to warp into.

    Returns:
        The reprojected dataset.

    Raises:
        OffLimbError: when the warp reports too few surviving sample points to bound an output, or leaves a
            vector layer with no finite coordinate — i.e. the data is outside the projection's visible area.
        RuntimeError: any other warp failure, re-raised as it came — a real projection error must not be
            mistaken for an empty view.

    Examples:
        - A dataset the projection can show comes back reprojected:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from digitalearth.base.crs import reproject
            >>> ds = Dataset.from_array(
            ...     np.ones((20, 20), "float32"),
            ...     geo_ref=GeoReference(geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326),
            ... )
            >>> reproject(ds, 3857).epsg
            3857

            ```
        - One it cannot is reported as an empty view rather than in GDAL's wording:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from digitalearth.base.crs import OffLimbError, reproject
            >>> from digitalearth.static import projections
            >>> ds = Dataset.from_array(
            ...     np.ones((20, 20), "float32"),
            ...     geo_ref=GeoReference(geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326),
            ... )
            >>> try:
            ...     reproject(ds, projections.orthographic(lon=-175, lat=15))
            ... except OffLimbError as error:
            ...     print(str(error).split(":")[-1].strip())
            too few sample points survive the warp for it to produce any output

            ```
        - A vector layer the projection cannot place reaches the same signal, even though its warp
          "succeeds" and quietly returns infinities:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from pyramids.feature import FeatureCollection
            >>> from digitalearth.base.crs import OffLimbError, reproject
            >>> points = FeatureCollection(
            ...     gpd.GeoDataFrame(geometry=[Point(4.0, 53.0), Point(4.1, 53.1)], crs=4326)
            ... )
            >>> try:
            ...     reproject(points, "+proj=ortho +lat_0=-40 +lon_0=105")
            ... except OffLimbError as error:
            ...     print(str(error).split(":")[-1].strip())
            no geometry came back with a finite coordinate

            ```
    """
    try:
        warped = dataset.to_crs(crs)
    except RuntimeError as error:
        if _POINTS_FAILED.search(str(error)):
            raise OffLimbError(
                f"the data lies outside what {crs!r} can show: too few sample points survive the warp "
                f"for it to produce any output"
            ) from error
        raise
    if _placed_nowhere(warped) and not _placed_nowhere(dataset):
        raise OffLimbError(
            f"the data lies outside what {crs!r} can show: no geometry came back with a finite coordinate"
        )
    return warped
