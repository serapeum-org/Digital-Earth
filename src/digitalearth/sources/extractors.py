"""Extractors — turn a pyramids object (or raw numpy) into a :class:`Source`.

``extract`` dispatches on the input type to one of the ``_from_*`` builders. Each builder reads through the
**pyramids** API only (no ``xarray``/``rasterio``) and returns numpy arrays + coordinate vectors + CRS +
metadata wrapped in a :class:`Source`.

Band convention: the Digital-Earth API is **1-based** (``band=1`` is the first band, matching ``StaticGlyph``
and GDAL), while pyramids' ``read_array(band=)`` and the per-band metadata tuples are **0-based** — so we
read ``band - 1`` internally.
"""
from typing import Any, Optional

import numpy as np
from pyramids.dataset import Dataset

from digitalearth._arrays import mask_nodata, read_masked_band
from digitalearth._crs import source_epsg
from digitalearth.sources.dimension import DimensionInfo
from digitalearth.sources.source import Source


def get_stack(data: Any, bands: Any, *, mask: bool = True) -> np.ndarray:
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
        read_masked_band(data, b) if mask else np.asarray(data.read_array(band=b - 1), dtype="float64")
        for b in bands
    ]
    return np.dstack(layers)


def extract(
    data: Any,
    *,
    band: int = 1,
    variable: Optional[str] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    metadata: Optional[dict] = None,
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
        return _from_netcdf(data, variable, metadata)
    if isinstance(data, Dataset):
        return _from_raster(data, band, metadata)
    if isinstance(data, DatasetCollection):
        return _from_collection(data, band, metadata)
    if isinstance(data, FeatureCollection):
        return _from_feature(data, metadata)
    if isinstance(data, np.ndarray):
        return _from_numpy(data, x, y, metadata)
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


def _from_raster(ds: Dataset, band: int, metadata: Optional[dict]) -> Source:
    """Build a raster :class:`Source` from a pyramids ``Dataset`` (1-based ``band``)."""
    idx = band - 1
    arr = ds.read_array(band=idx)
    z = mask_nodata(arr, _band_item(ds.no_data_value, idx))
    units = _band_item(ds.band_units, idx) or None
    return Source(
        z=_axis(z, "z", units),
        x=_axis(ds.x, "x"),  # 1-D cell-centre coords, length == columns
        y=_axis(ds.y, "y"),  # 1-D cell-centre coords, length == rows
        crs=ds.epsg,
        units=units,
        metadata={
            "kind": "raster",
            "variable": _band_item(ds.band_names, idx, ""),
            "band": band,
            **(metadata or {}),
        },
    )


def _from_netcdf(nc: Any, variable: Optional[str], metadata: Optional[dict]) -> Source:
    """Build a raster :class:`Source` from a pyramids ``NetCDF`` variable (defaults to the first)."""
    names = list(nc.variable_names)
    if variable is None:
        if not names:
            raise ValueError("NetCDF has no variables to plot")
        variable = names[0]
    arr = nc.read_array(variable=variable)
    nodata = nc.no_data_value
    if isinstance(nodata, (list, tuple)):
        nodata = nodata[0] if len(nodata) else None
    z = mask_nodata(arr, nodata)
    return Source(
        z=_axis(z, "z"),
        x=_axis(nc.lon, "x"),
        y=_axis(nc.lat, "y"),
        crs=nc.epsg,
        metadata={"kind": "raster", "variable": variable, **(metadata or {})},
    )


def _from_collection(
    dc: Any, band: int, metadata: Optional[dict], member: int = 0
) -> Source:
    """Build a raster :class:`Source` from one member of a pyramids ``DatasetCollection``.

    ``member`` is a 0-based index into ``dc.datasets``; member/count are recorded in metadata so callers
    (e.g. spaghetti/timeseries wiring) can iterate the rest.
    """
    members = dc.datasets
    src = _from_raster(members[member], band, metadata)
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
            **(metadata or {}),
        },
    )


def _from_feature(fc: Any, metadata: Optional[dict]) -> Source:
    """Build a vector (point) :class:`Source` from a pyramids ``FeatureCollection``.

    ``FeatureCollection`` is a GeoDataFrame subclass, so we read its geometry/CRS directly. ``z`` is the
    first numeric non-geometry column (or ``None`` when there is none). Point coordinates come from the
    geometry; non-point geometries fall back to their centroid.
    """
    geom_name = fc.geometry.name
    geom = fc.geometry
    if (geom.geom_type == "Point").all():
        xs, ys = geom.x.to_numpy(), geom.y.to_numpy()
    else:
        cent = geom.centroid
        xs, ys = cent.x.to_numpy(), cent.y.to_numpy()

    value_cols = [
        c
        for c in fc.columns
        if c != geom_name and np.issubdtype(fc[c].dtype, np.number)
    ]
    column = value_cols[0] if value_cols else None
    z = _axis(fc[column].to_numpy(), "z") if column is not None else None

    return Source(
        z=z,
        x=_axis(xs, "x"),
        y=_axis(ys, "y"),
        crs=source_epsg(fc),
        metadata={"kind": "vector", "variable": column or "", **(metadata or {})},
    )


def _from_numpy(
    arr: np.ndarray,
    x: Optional[np.ndarray],
    y: Optional[np.ndarray],
    metadata: Optional[dict],
) -> Source:
    """Build a raster :class:`Source` from a raw 2-D numpy array (pixel-index axes unless ``x``/``y`` given)."""
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
        crs=None,
        metadata={"kind": "raster", "variable": "", **(metadata or {})},
    )
