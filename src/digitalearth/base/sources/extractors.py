"""Extractors — turn a pyramids object (or raw numpy) into a :class:`Source`.

``extract`` dispatches on the input type to one of the ``_from_*`` builders. Each builder reads through the
**pyramids** API only (no ``xarray``/``rasterio``) and returns numpy arrays + coordinate vectors + CRS +
metadata wrapped in a :class:`Source`.

Band convention: the Digital-Earth API is **1-based** (``band=1`` is the first band, matching ``StaticGlyph``
and GDAL), while pyramids' ``read_array(band=)`` and the per-band metadata tuples are **0-based** — so we
read ``band - 1`` internally.

Two cross-cutting rules every builder follows:

* **Values are masked by pyramids, never by hand.** Every raster read goes through
  :func:`~digitalearth.base.arrays.read_masked_band` (``read_array(masked=True)``); comparing an unpacked
  value against the stored ``no_data_value`` sentinel missed every nodata cell of a CF-packed band.
* **``Source.crs`` is the CRS the coordinates are in, as given.** A caller that has already warped the data
  passes the CRS it warped to as ``crs=`` and it is stored verbatim; otherwise the input's own EPSG code is
  used, falling back to its projection definition when it has no authority code. ``None`` is reserved for a
  genuinely unknown CRS (a raw numpy array).
"""

from typing import Any, Mapping, Optional

import numpy as np
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from pyramids.dataset import Dataset

from digitalearth.base.arrays import read_masked_band
from digitalearth.base.crs import source_epsg
from digitalearth.base.sources.dimension import DimensionInfo
from digitalearth.base.sources.source import Source
from digitalearth.base.types import PlottableData, RasterLike


def get_stack(data: RasterLike, bands: Any, *, mask: bool = True) -> np.ndarray:
    """Read several raster bands into one band-last ``(rows, cols, n)`` ``float64`` stack.

    The multiband companion to :func:`extract` / ``get_source`` (which model a single band): it gives the
    composite renderers (``rgb_composite`` / ``hsv_composite``) a way to read a band stack through the sources
    layer instead of calling pyramids' ``read_array`` directly.

    Args:
        data: A pyramids ``Dataset`` (duck-typed by ``read_array`` / ``no_data_value``).
        bands: An ordered iterable of **1-based** band indices to stack (e.g. ``(1, 2, 3)``).
        mask: When ``True`` (default) each band's nodata cells are set to ``NaN`` (consistent with the
            single-band path); ``False`` returns the raw cast values.

    Returns:
        np.ndarray: a ``float64`` array of shape ``(rows, cols, len(bands))``.
    """
    layers = [
        read_masked_band(data, b)
        if mask
        else np.asarray(data.read_array(band=b - 1), dtype="float64")
        for b in bands
    ]
    return np.dstack(layers)


def extract(
    data: PlottableData,
    *,
    band: int = 1,
    variable: Optional[str] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    metadata: Optional[dict] = None,
    crs: Any = None,
) -> Source:
    """Build a :class:`Source` from any supported input.

    Args:
        data: A pyramids ``Dataset``, ``NetCDF``, ``DatasetCollection`` or ``FeatureCollection``, or a numpy
            array.
        band: 1-based band index for raster/collection inputs.
        variable: Variable name for ``NetCDF`` inputs (defaults to the first variable).
        x: Optional x coordinates for a raw numpy array (defaults to pixel indices).
        y: Optional y coordinates for a raw numpy array (defaults to pixel indices).
        metadata: Extra metadata merged into the resulting ``Source``.
        crs: The CRS ``data``'s coordinates are already in, stored on the ``Source`` verbatim. Pass it when
            the caller has warped the data: pyramids reports ``epsg is None`` for a projection with no
            authority code (an orthographic globe, say), so re-deriving the address from the warped dataset
            would lose it. ``None`` (the default) derives the CRS from the input.

    Returns:
        Source: the uniform wrapper the glyph wiring consumes.

    Raises:
        TypeError: if ``data`` is not a supported type.
    """
    # Local imports: keep optional/heavier pyramids submodules out of import time.
    from pyramids.dataset.collection import DatasetCollection
    from pyramids.feature import FeatureCollection
    from pyramids.netcdf import NetCDF

    # NetCDF subclasses Dataset, so it MUST be checked before the Dataset branch.
    if isinstance(data, NetCDF):
        return _from_netcdf(data, variable, metadata, crs=crs)
    if isinstance(data, Dataset):
        return _from_raster(data, band, metadata, crs=crs)
    if isinstance(data, DatasetCollection):
        return _from_collection(data, band, metadata, crs=crs)
    if isinstance(data, FeatureCollection):
        return _from_feature(data, metadata, crs=crs)
    if isinstance(data, np.ndarray):
        return _from_numpy(data, x, y, metadata, crs=crs)
    raise TypeError(f"cannot build a Source from {type(data).__name__}")


def _axis(values: Any, name: str, units: Optional[str] = None) -> DimensionInfo:
    """Wrap a coordinate/value array in a :class:`DimensionInfo`."""
    return DimensionInfo(np.asarray(values), name, units)


def _band_item(seq: Any, index: int, default: Any = None) -> Any:
    """Safely read ``seq[index]`` from a per-band list/tuple, tolerating ``None``/short sequences."""
    if not seq:
        return default
    try:
        return seq[index]
    except (IndexError, TypeError, KeyError):
        return default


def _attr(metadata: Any, key: str) -> Optional[str]:
    """Read a CF identity item (``standard_name``, ``units``, ...) from an attribute mapping.

    Case-insensitive, because the same attribute reaches us as ``standard_name`` from a CF file and as
    ``STANDARD_NAME`` from a raster whose writer upper-cased its GDAL metadata keys.

    Args:
        metadata: A mapping of attribute names to values, or anything else (answered with ``None``).
        key: The attribute name to look for, in any case.

    Returns:
        The attribute's value as a stripped string, or ``None`` when it is absent or empty.
    """
    if not isinstance(metadata, Mapping):
        return None
    wanted = key.lower()
    for name, value in metadata.items():
        if str(name).lower() == wanted and value is not None:
            text = str(value).strip()
            return text or None
    return None


def _identity(*mappings: Any) -> dict:
    """Collect the CF identity the style matcher reads, from the first mapping that declares it.

    Args:
        *mappings: Attribute mappings in precedence order (e.g. the band's metadata, then the dataset's).

    Returns:
        A ``{"standard_name": ...}`` dict, or an empty one when nothing declares a standard name — so a
        caller-supplied value is never overwritten with ``None``.
    """
    for mapping in mappings:
        found = _attr(mapping, "standard_name")
        if found is not None:
            return {"standard_name": found}
    return {}


def _raster_crs(ds: Any) -> Any:
    """The CRS a pyramids raster's coordinates are in: its EPSG code, else its projection definition.

    pyramids reports ``epsg is None`` for a projection carrying no authority code — exactly what a warp into
    an orthographic display CRS produces — so falling back to the WKT keeps the ``Source`` able to say where
    its coordinates live instead of claiming the CRS is unknown.

    Args:
        ds: A pyramids ``Dataset``/``NetCDF`` (duck-typed by ``epsg`` / ``crs``).

    Returns:
        The EPSG integer, the projection WKT, or ``None`` when the input declares no CRS at all.
    """
    epsg = getattr(ds, "epsg", None)
    if epsg is not None:
        return epsg
    return getattr(ds, "crs", None) or None


def _features_crs(fc: Any) -> Any:
    """The CRS a vector frame's coordinates are in: its EPSG code, else its projection definition.

    Args:
        fc: A pyramids ``FeatureCollection`` / geopandas ``GeoDataFrame``.

    Returns:
        The EPSG integer, the CRS's WKT, or ``None`` when the frame declares no CRS.
    """
    epsg = source_epsg(fc)
    if epsg is not None:
        return epsg
    crs = getattr(fc, "crs", None)
    if crs is None:
        return None
    to_wkt = getattr(crs, "to_wkt", None)
    return to_wkt() if callable(to_wkt) else crs


def _from_raster(
    ds: Dataset, band: int, metadata: Optional[dict], *, crs: Any = None
) -> Source:
    """Build a raster :class:`Source` from a pyramids ``Dataset`` (1-based ``band``).

    The band's identity is read alongside its values: ``units`` from ``band_units`` (falling back to a
    ``units`` attribute in the band's or the dataset's GDAL metadata), and CF ``standard_name`` from that
    same metadata — the key the Magics style matcher consults when a variable's short name is opaque.

    Args:
        ds: The pyramids ``Dataset`` to read.
        band: 1-based band index.
        metadata: Extra metadata merged over the derived keys (the caller wins).
        crs: The CRS the dataset's coordinates are already in, stored verbatim; ``None`` derives it.

    Returns:
        Source: the raster source.
    """
    idx = band - 1
    z = read_masked_band(ds, band)
    band_meta = _band_item(ds.band_meta_data, idx) or {}
    dataset_meta = getattr(ds, "meta_data", None)
    units = (
        _band_item(ds.band_units, idx)
        or _attr(band_meta, "units")
        or _attr(dataset_meta, "units")
    )
    return Source(
        z=_axis(z, "z", units),
        x=_axis(ds.x, "x"),  # 1-D cell-centre coords, length == columns
        y=_axis(ds.y, "y"),  # 1-D cell-centre coords, length == rows
        crs=crs if crs is not None else _raster_crs(ds),
        units=units,
        metadata={
            "kind": "raster",
            "variable": _band_item(ds.band_names, idx, ""),
            "band": band,
            **_identity(band_meta, dataset_meta),
            **(metadata or {}),
        },
    )


def _variable_attributes(nc: Any, variable: str) -> dict:
    """The CF attributes pyramids read off one NetCDF variable, as a plain mapping.

    ``NetCDF.meta_data`` is a ``NetCDFMetadata`` whose ``variables`` hold one ``VariableInfo`` per variable,
    carrying the file's own attribute list (``standard_name``, ``long_name``, …) plus a resolved ``unit``.
    That is the public per-variable accessor, so nothing here reaches for ``netCDF4`` or GDAL.

    Args:
        nc: A pyramids ``NetCDF`` container.
        variable: The variable's name.

    Returns:
        The variable's attributes, with the resolved ``unit`` folded in as ``units`` when the attribute list
        itself carries none. An empty dict when the container exposes no such metadata.
    """
    variables = getattr(getattr(nc, "meta_data", None), "variables", None)
    if variables is None:
        return {}
    infos = variables.values() if isinstance(variables, Mapping) else variables
    for info in infos:
        if getattr(info, "name", None) != variable:
            continue
        attributes = dict(getattr(info, "attributes", None) or {})
        unit = getattr(info, "unit", None)
        if unit and not _attr(attributes, "units"):
            attributes["units"] = unit
        return attributes
    return {}


def _from_netcdf(
    nc: Any, variable: Optional[str], metadata: Optional[dict], *, crs: Any = None
) -> Source:
    """Build a raster :class:`Source` from a pyramids ``NetCDF`` variable (defaults to the first).

    NetCDF is where CF metadata actually lives, so the variable's ``units`` and ``standard_name`` are read
    here and carried onto the ``Source``; without them the style matcher's units and standard-name steps
    could never fire on the one input type they were written for.

    Args:
        nc: The pyramids ``NetCDF`` container to read.
        variable: The variable to plot; ``None`` takes the first.
        metadata: Extra metadata merged over the derived keys (the caller wins).
        crs: The CRS the container's coordinates are already in, stored verbatim; ``None`` derives it.

    Returns:
        Source: the raster source.

    Raises:
        ValueError: when the container exposes no variables to plot.
    """
    names = list(nc.variable_names)
    if variable is None:
        if not names:
            raise ValueError("NetCDF has no variables to plot")
        variable = names[0]
    z = np.ma.filled(
        np.ma.asarray(nc.read_array(variable=variable, masked=True)).astype("float64"),
        np.nan,
    )
    attributes = _variable_attributes(nc, variable)
    units = _attr(attributes, "units")
    return Source(
        z=_axis(z, "z", units),
        x=_axis(nc.lon, "x"),
        y=_axis(nc.lat, "y"),
        crs=crs if crs is not None else _raster_crs(nc),
        units=units,
        metadata={
            "kind": "raster",
            "variable": variable,
            **_identity(attributes),
            **(metadata or {}),
        },
    )


def _from_collection(
    dc: Any, band: int, metadata: Optional[dict], member: int = 0, *, crs: Any = None
) -> Source:
    """Build a raster :class:`Source` from one member of a pyramids ``DatasetCollection``.

    ``member`` is a 0-based index into ``dc.datasets``; member/count are recorded in metadata so callers
    (e.g. spaghetti/timeseries wiring) can iterate the rest.

    Args:
        dc: The pyramids ``DatasetCollection``.
        band: 1-based band index read from the member.
        metadata: Extra metadata merged over the derived keys (the caller wins).
        member: 0-based index of the member to read.
        crs: The CRS the member's coordinates are already in, stored verbatim; ``None`` derives it.

    Returns:
        Source: the raster source, with the member's identity carried forward.
    """
    members = dc.datasets
    src = _from_raster(members[member], band, metadata, crs=crs)
    standard_name = src.metadata("standard_name")
    # augment the raster metadata with the collection position
    return Source(
        z=src.z,
        x=src.x,
        y=src.y,
        crs=src.crs,
        units=src.units,
        metadata={
            "kind": "raster",
            "variable": src.metadata("variable", ""),
            "member": member,
            "n_members": len(members),
            **({"standard_name": standard_name} if standard_name else {}),
            **(metadata or {}),
        },
    )


def _from_feature(fc: Any, metadata: Optional[dict], *, crs: Any = None) -> Source:
    """Build a vector (point) :class:`Source` from a pyramids ``FeatureCollection``.

    ``FeatureCollection`` is a GeoDataFrame subclass, so we read its geometry/CRS directly. ``z`` is the
    first numeric non-geometry column (or ``None`` when there is none). "Numeric" is decided by pandas, so
    a nullable ``Int64``/``Float64`` counts and a ``string``/``boolean`` column does not. Bools are
    excluded, as ``np.issubdtype`` had them; timedeltas are too, which ``np.issubdtype`` did not, because
    a timedelta column cannot be rendered — picking one only moved the failure downstream. Point
    coordinates come from the geometry; non-point geometries fall back to their centroid.
    """
    geom_name = fc.geometry.name
    geom = fc.geometry
    if (geom.geom_type == "Point").all():
        xs, ys = geom.x.to_numpy(), geom.y.to_numpy()
    else:
        cent = geom.centroid
        xs, ys = cent.x.to_numpy(), cent.y.to_numpy()

    # Classified with pandas, not np.issubdtype: the latter understands only numpy dtypes and *raises*
    # on a pandas extension dtype rather than answering False, so one nullable or string column anywhere
    # in the frame took down every vector render.
    #
    # Two dtypes answer differently from np.issubdtype, both deliberately. Bools: numeric to pandas, but
    # np.issubdtype said False and they were never a value column here. Timedeltas: np.timedelta64 sits
    # under np.signedinteger so np.issubdtype said True, but choosing one only moved the failure — the
    # values reach the renderer as datetime.timedelta objects and float() rejects them. Skipping the
    # column renders the frame instead of crashing on it.
    value_cols = [
        c
        for c in fc.columns
        if c != geom_name and is_numeric_dtype(fc[c]) and not is_bool_dtype(fc[c])
    ]
    column = value_cols[0] if value_cols else None
    z = _axis(fc[column].to_numpy(), "z") if column is not None else None

    return Source(
        z=z,
        x=_axis(xs, "x"),
        y=_axis(ys, "y"),
        crs=crs if crs is not None else _features_crs(fc),
        metadata={"kind": "vector", "variable": column or "", **(metadata or {})},
    )


def _from_numpy(
    arr: np.ndarray,
    x: Optional[np.ndarray],
    y: Optional[np.ndarray],
    metadata: Optional[dict],
    *,
    crs: Any = None,
) -> Source:
    """Build a raster :class:`Source` from a raw 2-D numpy array (pixel-index axes unless ``x``/``y`` given).

    A bare array carries no CRS, so ``crs`` stays ``None`` unless the caller names one — this is the case the
    contract's ``None`` is reserved for.
    """
    a = np.asarray(arr, dtype="float64")
    if a.ndim != 2:
        raise ValueError(f"numpy Source expects a 2-D array, got {a.ndim}-D")
    rows, cols = a.shape
    xs = np.asarray(x) if x is not None else np.arange(cols, dtype="float64")
    ys = np.asarray(y) if y is not None else np.arange(rows, dtype="float64")
    return Source(
        z=_axis(a, "z"),
        x=_axis(xs, "x"),
        y=_axis(ys, "y"),
        crs=crs,
        metadata={"kind": "raster", "variable": "", **(metadata or {})},
    )
