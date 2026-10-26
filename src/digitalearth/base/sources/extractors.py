"""Extractors — turn a pyramids object (or raw numpy) into a :class:`Source`.

``extract`` dispatches on the input type to one of the ``_from_*`` builders. Each builder reads through the
**pyramids** API only (no ``xarray``/``rasterio``) and returns numpy arrays + coordinate vectors + CRS +
metadata wrapped in a :class:`Source`.

Band convention: the Digital-Earth API is **1-based** (``band=1`` is the first band, matching GDAL), while
pyramids' ``read_array(band=)`` and the per-band metadata tuples are **0-based** — so we read ``band - 1``
internally.

Two cross-cutting rules every builder follows:

* **Values are masked by pyramids, never by hand.** Every raster read goes through
  :func:`~digitalearth.base.arrays.read_masked_band` (``read_array(masked=True)``); comparing an unpacked
  value against the stored ``no_data_value`` sentinel missed every nodata cell of a CF-packed band.
* **``Source.crs`` is the CRS the coordinates are in, as given.** A caller that has already warped the data
  passes the CRS it warped to as ``crs=`` and it is stored verbatim; otherwise the input's own EPSG code is
  used, falling back to its projection definition when it has no authority code. ``None`` is reserved for a
  genuinely unknown CRS (a raw numpy array).

:func:`require_drawable` is the other side of that dispatch: what ``extract`` *cannot* read, refused where the
caller can see it. A builder declares which of :data:`DRAWABLE_FAMILIES` it draws and gets a `TypeError`
naming the call, the type and what it takes — rather than the attribute error whichever internal reached the
wrong type first.
"""

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from pyramids.dataset import Dataset

from digitalearth.base.arrays import read_masked_band
from digitalearth.base.crs import declared_crs, source_epsg
from digitalearth.base.points import PointArrays
from digitalearth.base.sources.dimension import DimensionInfo
from digitalearth.base.sources.mesh import MESH_LOCATIONS, MeshSource
from digitalearth.base.sources.source import Source
from digitalearth.base.spec import DEFAULT_BAND, DataRef, Selection
from digitalearth.base.types import PlottableData, RasterLike

logger = logging.getLogger(__name__)

#: The families of data a builder may declare it draws, each with the phrase a refusal uses to say so. They
#: are the branches :func:`extract` dispatches on, named — a builder takes some of them and not others, and
#: until :func:`require_drawable` that was something a caller found out from whichever attribute error the
#: wrong type happened to trip first (`Map.field(ndarray)` leaked ``'numpy.ndarray' object has no attribute
#: 'to_crs'`` from inside a warp, #343).
DRAWABLE_FAMILIES: Mapping[str, str] = MappingProxyType(
    {
        "raster": "a pyramids Dataset (a NetCDF or a DatasetCollection counts as one)",
        "array": "a 2-D numpy array",
        "vector": "a pyramids FeatureCollection",
        "mesh": "a pyramids UgridDataset (an unstructured mesh)",
    }
)


def _is_raster(data: Any) -> bool:
    """Whether `data` is a pyramids raster :func:`extract` can read a band out of.

    Args:
        data: The candidate.

    Returns:
        `True` for a `Dataset` (a `NetCDF` is one) or a `DatasetCollection`. Matched by type rather than
        duck-typed, because that is how :func:`extract` itself chooses the branch.
    """
    from pyramids.dataset.collection import DatasetCollection

    return isinstance(data, (Dataset, DatasetCollection))


def _is_array(data: Any) -> bool:
    """Whether `data` is a bare numpy array.

    Args:
        data: The candidate.

    Returns:
        `True` for any `numpy.ndarray`, a masked array included. Whether it is 2-D is
        :func:`_from_numpy`'s question, so that refusal keeps naming the number of dimensions.
    """
    return isinstance(data, np.ndarray)


def _is_vector(data: Any) -> bool:
    """Whether `data` is a vector layer a builder can reproject and read geometry from.

    Args:
        data: The candidate.

    Returns:
        `True` for anything exposing the two members a vector render needs — the geometry column and the
        reprojection. Duck-typed rather than matched against `FeatureCollection` because a plain geopandas
        frame is what pyramids' own ``to_geodataframe()`` hands back, and because this module does not import
        geopandas: pyramids is the only GIS dependency.
    """
    return hasattr(data, "geometry") and hasattr(data, "to_crs")


#: How each family is recognised. Keyed by the same names as :data:`DRAWABLE_FAMILIES`, so a family with a
#: phrase and no test — or the reverse — is a `KeyError` here rather than a silent "no".
def _is_mesh(data: Any) -> bool:
    """Whether `data` is an unstructured mesh :func:`extract` can read a variable off.

    Args:
        data: The candidate.

    Returns:
        `True` for a pyramids `UgridDataset`. Matched by type, like the raster test, because that is how
        :func:`extract` chooses the branch — a mesh is told apart from a raster by *being* one, not by
        exposing a member a raster lacks.
    """
    from pyramids.netcdf.ugrid.dataset import UgridDataset

    return isinstance(data, UgridDataset)


_RECOGNISERS: Mapping[str, Callable[[Any], bool]] = MappingProxyType(
    {
        "raster": _is_raster,
        "array": _is_array,
        "vector": _is_vector,
        "mesh": _is_mesh,
    }
)


def require_drawable(
    data: Any, *, caller: str, accepts: Sequence[str] = ("raster",)
) -> None:
    """Refuse a builder's data argument by name, instead of leaking an attribute error from inside a warp.

    A public method handed the wrong type should say so, and say what it takes. Until this, a tier's builders
    found out by calling: `Map.field({})` reached ``GeoLayerBase._reproject``, which called ``.to_crs`` on
    whatever it was given, and the caller read ``AttributeError: 'dict' object has no attribute 'to_crs'`` —
    a method they never called, about an attribute they never heard of (#343). ``quickmap`` already refused in
    the right register (``TypeError: quickmap cannot draw a ndarray``); this is that register, for a builder.

    A **reference** always passes: a path, a URL or a :class:`~digitalearth.base.spec.dataref.DataRef` names
    data that is opened when the layer is drawn, so there is nothing here to inspect. What it opens to is the
    resolver's business, and an unreadable file raises in its own words.

    Args:
        data: What the caller handed the builder.
        caller: The public call, spelled as a caller writes it — ``"Map.field()"``.
        accepts: The families this builder draws, as :data:`DRAWABLE_FAMILIES` keys them. A raster field
            takes ``("raster", "array")``; a builder needing pyramids' cell geometry takes ``("raster",)``,
            because a bare array carries no geo-transform to read one from.

    Raises:
        TypeError: naming the caller, the type it was handed and what it takes — the three things a caller
            needs to fix the call.
        ValueError: for an `accepts` naming a family nobody declared, which is a defect in the builder rather
            than in the call, and so is refused in different words.

    Examples:
        - A raster field takes a grid, whether pyramids' or numpy's, and names the type it was handed:
            ```python
            >>> from digitalearth.base.sources import require_drawable
            >>> require_drawable({}, caller="Map.field()", accepts=("raster", "array"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            TypeError: Map.field() cannot draw a dict; it takes a pyramids Dataset ...

            ```
        - An array is a grid and not a set of features, so one builder takes the value and another refuses it:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import require_drawable
            >>> grid = np.zeros((2, 2))
            >>> print(require_drawable(grid, caller="Map.field()", accepts=("raster", "array")))
            None
            >>> require_drawable(grid, caller="Map.points()", accepts=("vector",))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            TypeError: Map.points() cannot draw a ndarray; it takes a pyramids FeatureCollection, ...

            ```
        - A path is not inspected here: it names data the layer opens when it is drawn:
            ```python
            >>> from digitalearth.base.sources import require_drawable
            >>> print(require_drawable("examples/data/acc4000.tif", caller="Map.field()"))
            None

            ```
    """
    unknown = sorted(set(accepts).difference(DRAWABLE_FAMILIES))
    if unknown or not accepts:
        raise ValueError(
            f"require_drawable(accepts=) must name families from {sorted(DRAWABLE_FAMILIES)}; "
            f"got {list(accepts)!r}"
        )
    # A reference is data by name; it is opened at the draw, and only then can anything be said about it.
    if isinstance(data, (str, os.PathLike, DataRef)):
        return
    if any(_RECOGNISERS[family](data) for family in accepts):
        return
    takes = ", ".join(DRAWABLE_FAMILIES[family] for family in accepts)
    raise TypeError(
        f"{caller} cannot draw a {type(data).__name__}; it takes {takes}, or a path or URL to one"
    )


def get_stack(data: RasterLike, bands: Any, *, mask: bool = True) -> np.ndarray:
    """Read several raster bands into one band-last ``(rows, cols, n)`` ``float64`` stack.

    The multiband companion to :func:`extract` / ``get_source`` (which model a single band): it gives the
    composite renderers (``rgb_composite`` / ``hsv_composite``) a way to read a band stack through the sources
    layer instead of calling pyramids' ``read_array`` directly.

    Args:
        data: A pyramids ``Dataset`` (duck-typed by ``read_array`` / ``no_data_value``).
        bands: Which bands to stack, in order — a :class:`~digitalearth.base.spec.selection.Selection`, or
            an ordered iterable of **1-based** indices (e.g. ``(1, 2, 3)``). Either way it is normalised
            through `Selection`, so the 1-based contract this docstring states is now enforced rather than
            assumed: a 0 raises here instead of reading the wrong band.
        mask: When ``True`` (default) each band's nodata cells are set to ``NaN`` (consistent with the
            single-band path); ``False`` returns the raw cast values.

    Returns:
        np.ndarray: a ``float64`` array of shape ``(rows, cols, len(bands))``.

    Raises:
        ValueError: if `bands` is empty, or names a band below 1.
    """
    wanted = bands.band if isinstance(bands, Selection) else Selection.of(bands).band
    layers = [
        read_masked_band(data, b)
        if mask
        else np.asarray(data.read_array(band=b - 1), dtype="float64")
        for b in wanted
    ]
    return np.dstack(layers)


def extract(
    data: PlottableData,
    *,
    band: int = DEFAULT_BAND,
    variable: str | None = None,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    metadata: dict | None = None,
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
            would lose it. ``None`` (the default) derives the CRS from the input. The callers expected to
            pass it are the tiers' display-CRS choke points — see ``get_source`` for the list.

    Returns:
        Source: the uniform wrapper the glyph wiring consumes.

    Raises:
        TypeError: if ``data`` is not a supported type.

    Examples:
        - A raster comes back as a 2-D grid with 1-D cell-centre axes and the band's identity:
            ```python
            >>> from pyramids.dataset import Dataset
            >>> from digitalearth.base.sources import extract
            >>> src = extract(Dataset.read_file("examples/data/acc4000.tif"))
            >>> src.z.values.shape
            (13, 14)
            >>> (src.x.values.size, src.y.values.size)
            (14, 13)
            >>> src.metadata("kind")
            'raster'
            >>> src.epsg
            32618

            ```
        - A vector collection comes back point-shaped: 1-D coordinates, and ``z`` is the first
            numeric non-geometry column (the variable it names is recorded in the metadata):
            ```python
            >>> from pyramids.feature import FeatureCollection
            >>> from digitalearth.base.sources import extract
            >>> src = extract(FeatureCollection.read_file("tests/data/points.geojson"))
            >>> src.metadata("kind")
            'vector'
            >>> src.metadata("variable")
            'fid'
            >>> src.x.values.ndim, src.z.values.shape == src.x.values.shape
            (1, True)

            ```
        - A raw numpy array is supported too, with pixel-index axes and a deliberately unknown
            CRS — and ``crs=`` is stored verbatim when the caller has already placed the data:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import extract
            >>> extract(np.zeros((2, 3))).x.values.tolist()
            [0.0, 1.0, 2.0]
            >>> extract(np.zeros((2, 3))).crs is None
            True
            >>> extract(np.zeros((2, 3)), crs="+proj=ortho +lat_0=53").epsg is None
            True

            ```
        - Anything else is refused by type name rather than half-extracted:
            ```python
            >>> from digitalearth.base.sources import extract
            >>> try:
            ...     extract("examples/data/acc4000.tif")
            ... except TypeError as error:
            ...     print(error)
            cannot build a Source from str

            ```
    """
    # Local imports: keep optional/heavier pyramids submodules out of import time.
    from pyramids.dataset.collection import DatasetCollection
    from pyramids.feature import FeatureCollection
    from pyramids.netcdf import NetCDF

    if _is_mesh(data):
        return _from_ugrid(data, variable, metadata, band=band, crs=crs)
    # NetCDF subclasses Dataset, so it MUST be checked before the Dataset branch.
    if isinstance(data, NetCDF):
        return _from_netcdf(data, variable, metadata, band=band, crs=crs)
    if isinstance(data, Dataset):
        return _from_raster(data, band, metadata, crs=crs)
    if isinstance(data, DatasetCollection):
        return _from_collection(data, band, metadata, crs=crs)
    if isinstance(data, FeatureCollection):
        return _from_feature(data, metadata, crs=crs)
    if isinstance(data, np.ndarray):
        return _from_numpy(data, x, y, metadata, crs=crs)
    raise TypeError(f"cannot build a Source from {type(data).__name__}")


def _axis(values: Any, name: str, units: str | None = None) -> DimensionInfo:
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


def _attr(metadata: Any, key: str) -> str | None:
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


#: CF identity attributes carried onto a ``Source``: ``standard_name`` drives the Magics style matcher, and
#: ``long_name`` is the most human name :func:`~digitalearth.base.captions.caption` reaches for first.
_IDENTITY_KEYS = ("standard_name", "long_name")


def _identity(*mappings: Any) -> dict:
    """Collect the CF identity keys a title/style reads, each from the first mapping that declares it.

    Args:
        *mappings: Attribute mappings in precedence order (e.g. the band's metadata, then the dataset's).

    Returns:
        A dict of the :data:`_IDENTITY_KEYS` any mapping declares (``standard_name``, ``long_name``), or an
        empty one when none do — so a caller-supplied value is never overwritten with ``None``.
    """
    found: dict[str, str] = {}
    for key in _IDENTITY_KEYS:
        for mapping in mappings:
            value = _attr(mapping, key)
            if value is not None:
                found[key] = value
                break
    return found


#: GRIB keys :func:`pyramids.grib.grib_band_metadata` decodes that are worth carrying onto a ``Source`` — the
#: message-native identity plus the forecast time/horizon. ``forecast_seconds`` is surfaced under the friendlier
#: name ``step``. ``band`` (the 1-based index) is dropped: the ``Source`` already records it.
_GRIB_META_KEYS = {
    "element": "element",
    "short_name": "short_name",
    "comment": "comment",
    "discipline": "discipline",
    "ref_time": "ref_time",
    "valid_time": "valid_time",
    "forecast_seconds": "step",
}


#: Per-dataset cache of pyramids' decoded GRIB band metadata. ``grib_band_metadata`` decodes *all* bands, so
#: reading an n-band GRIB stack band-by-band (as facet/quickpanels do) would call it once per band — quadratic.
#: Keyed weakly by the ``Dataset``, so a dataset that goes out of scope drops its entry without being pinned.
_GRIB_DECODE_CACHE: "WeakKeyDictionary[Dataset, list[dict[str, Any]]]" = (
    WeakKeyDictionary()
)


def _is_grib(band_meta: Any) -> bool:
    """Whether a band's GDAL metadata marks it as GRIB — a cheap ``GRIB_*`` key check, no decode."""
    return isinstance(band_meta, Mapping) and any(
        str(key).startswith("GRIB_") for key in band_meta
    )


def _decode_grib(ds: Dataset, *, refresh: bool = False) -> list[dict[str, Any]]:
    """Return pyramids' per-band GRIB metadata for ``ds``, decoding once and caching it per dataset.

    Args:
        ds: The pyramids ``Dataset`` to decode.
        refresh: When ``True``, ignore any cached decode and re-decode — used to drop a cache entry that a
            caller found stale (the band metadata was reassigned in place after the first decode).
    """
    cached = None if refresh else _GRIB_DECODE_CACHE.get(ds)
    if cached is None:
        from pyramids.grib import grib_band_metadata

        cached = grib_band_metadata(ds)
        _GRIB_DECODE_CACHE[ds] = cached
    return cached


def _grib_identity(ds: Dataset, band_meta: Any, idx: int) -> dict:
    """Decode a GRIB band's NWP metadata through pyramids, when the band carries any.

    Detection is cheap — a GRIB band's GDAL metadata carries ``GRIB_*`` keys (see :func:`_is_grib`) — so a
    non-GRIB raster never pays for the decode. The decoding itself is **pyramids'**
    (:func:`pyramids.grib.grib_band_metadata`), not reimplemented here: it parses the reference/valid times to
    timezone-aware datetimes and the forecast horizon to seconds, and is cached per dataset by
    :func:`_decode_grib`. Only the keys in :data:`_GRIB_META_KEYS` that are present (non-``None``) are returned,
    with ``forecast_seconds`` surfaced as ``step``, so :func:`~digitalearth.base.captions.caption` can stamp a
    ``valid_time`` and a panel can be keyed by ``step`` / ``element``.

    Args:
        ds: The pyramids ``Dataset`` the band belongs to.
        band_meta: The band's GDAL metadata mapping (already read by the caller), used only to detect GRIB.
        idx: 0-based band index into the decoded list.

    Returns:
        The decoded NWP metadata for the band, or an empty dict when the band is not GRIB.
    """
    if not _is_grib(band_meta):
        return {}
    entry = _band_item(_decode_grib(ds), idx) or {}
    # Guard against a stale cache: `element`/`short_name` in the decoded entry come from the same GDAL keys the
    # live `band_meta` carries, so a disagreement means the band metadata was reassigned in place after the
    # first decode. Re-decode once to drop the stale entry, keeping identity and units consistent.
    if (entry.get("element"), entry.get("short_name")) != (
        band_meta.get("GRIB_ELEMENT"),
        band_meta.get("GRIB_SHORT_NAME"),
    ):
        entry = _band_item(_decode_grib(ds, refresh=True), idx) or {}
    return {
        out: entry[key]
        for key, out in _GRIB_META_KEYS.items()
        if entry.get(key) is not None
    }


def _unbracket(units: str | None) -> str | None:
    """Strip the ``[...]`` GDAL wraps a GRIB unit in (``"[C]"`` → ``"C"``); other spellings pass through."""
    if not units:
        return units
    stripped = units.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1].strip()
    return stripped or None


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
    ds: Dataset, band: int, metadata: dict | None, *, crs: Any = None
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
    grib = _grib_identity(ds, band_meta, idx)
    # Gate the GRIB_UNIT fallback on GRIB *detection*, not on the decoded identity being non-empty: a band that
    # decodes to no identity/time keys still declares its unit, and GRIB_UNIT is not matched by the `units` step.
    grib_unit = (
        _unbracket(_attr(band_meta, "GRIB_UNIT")) if _is_grib(band_meta) else None
    )
    units = (
        _band_item(ds.band_units, idx)
        or _attr(band_meta, "units")
        or _attr(dataset_meta, "units")
        or grib_unit
    )
    identity = _identity(band_meta, dataset_meta)
    # pyramids auto-names a GRIB band "Band_N", so the human name has to come from the message itself: its
    # comment (the description), else the short name / element. Surfaced as long_name, which caption reads first.
    if grib and "long_name" not in identity:
        grib_name = grib.get("comment") or grib.get("short_name") or grib.get("element")
        if grib_name:
            identity = {"long_name": grib_name, **identity}
    return Source(
        z=_axis(z, "z", units),
        x=_axis(ds.x, "x"),  # 1-D cell-centre coords, length == columns
        y=_axis(ds.y, "y"),  # 1-D cell-centre coords, length == rows
        crs=crs if crs is not None else declared_crs(ds),
        units=units,
        metadata={
            "kind": "raster",
            "variable": _band_item(ds.band_names, idx, ""),
            "band": band,
            **grib,
            **identity,
            **(metadata or {}),
        },
    )


def _variable_attributes(nc: Any, variable: str | None) -> dict:
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


def _netcdf_slice(z: np.ndarray, named: str, band: int) -> np.ndarray:
    """Take the 1-based ``band`` slice off a band-first NetCDF read.

    Args:
        z: The 3-D ``(slices, rows, columns)`` array pyramids read.
        named: How the variable reads in a message — ``"variable 't2m'"``, or ``"pinned variable"``.
        band: 1-based index of the slice to draw.

    Returns:
        np.ndarray: the 2-D slice ``band`` names.

    Raises:
        ValueError: when ``band`` names a slice the variable does not carry. Silently drawing slice 0 for
            an out-of-range index would show the wrong field without saying so.

    Examples:
        - The band is 1-based, like the raster path's, so ``band=2`` is the second slice:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources.extractors import _netcdf_slice
            >>> stack = np.stack([np.full((2, 2), float(step)) for step in range(3)])
            >>> _netcdf_slice(stack, "variable 't2m'", 2).tolist()
            [[1.0, 1.0], [1.0, 1.0]]

            ```
        - A slice the variable does not carry is named rather than rounded down to the first:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources.extractors import _netcdf_slice
            >>> stack = np.stack([np.zeros((2, 2)), np.ones((2, 2))])
            >>> try:
            ...     _netcdf_slice(stack, "variable 't2m'", 4)
            ... except ValueError as error:
            ...     print(error)
            band=4 is out of range for the variable 't2m', which carries 2 slice(s)

            ```
    """
    slices = int(z.shape[0])
    if not 1 <= band <= slices:
        raise ValueError(
            f"band={band} is out of range for the {named}, which carries {slices} slice(s)"
        )
    if slices > 1 and band == DEFAULT_BAND:
        logger.warning(
            "The %s has %d slices; the Source draws band=%d (pass band= to choose another)",
            named,
            slices,
            DEFAULT_BAND,
        )
    taken: np.ndarray = z[band - 1]
    return taken


def _from_netcdf(
    nc: Any,
    variable: str | None,
    metadata: dict | None,
    *,
    band: int = DEFAULT_BAND,
    crs: Any = None,
) -> Source:
    """Build a raster :class:`Source` from a pyramids ``NetCDF`` variable (defaults to the first).

    NetCDF is where CF metadata actually lives, so the variable's ``units`` and ``standard_name`` are read
    here and carried onto the ``Source``; without them the style matcher's units and standard-name steps
    could never fire on the one input type they were written for.

    Two shapes of input arrive here. A **container** exposes ``variable_names`` and is read by name. A
    **pinned variable subset** — what ``NetCDF.get_variable(name)`` hands back — reports no names of its
    own, because it *is* the variable; it is read without one. Refusing it as "no variables to plot" was
    the defect #499 reported.

    A subset still carries its identity, and this reads it back: the variable's **name** off the subset
    and its **CF attributes** off the subset's own attributes. Both matter beyond tidiness — the style
    matcher picks a colormap from the name, so losing it silently restyled the map.

    Args:
        nc: The pyramids ``NetCDF`` container to read, or a pinned ``Variable`` subset of one.
        variable: The variable to plot; ``None`` takes the first, or the pinned one.
        metadata: Extra metadata merged over the derived keys (the caller wins).
        band: 1-based slice to draw from a variable carrying a band/time dimension.
        crs: The CRS the container's coordinates are already in, stored verbatim; ``None`` derives it.

    Returns:
        Source: the raster source.

    Raises:
        ValueError: when the container exposes no variables to plot, or when ``band`` names a slice the
            variable does not carry.
    """
    from pyramids.netcdf import Variable

    # A pinned subset reports no variable_names of its own, so it is read without naming one.
    pinned = isinstance(nc, Variable)
    if variable is None and not pinned:
        names = list(nc.variable_names)
        if not names:
            raise ValueError("NetCDF has no variables to plot")
        variable = names[0]
    if pinned and variable is None:
        # The subset knows which variable it is. Recovering the name is what keeps the identity the style
        # matcher reads (`metadata("variable")`), the interactive colorbar label, and the slice messages —
        # all of which went blank when the name was dropped here (review H1).
        variable = getattr(nc, "_source_var_name", None)
    named = "pinned variable" if variable is None else f"variable {variable!r}"
    # Read by name only from a container: a subset *is* the variable and refuses to be asked for one.
    by_name = {} if pinned else {"variable": variable}
    z = np.ma.filled(
        np.ma.asarray(nc.read_array(masked=True, **by_name)).astype("float64"),
        np.nan,
    )
    # pyramids 0.71 reads a NetCDF variable band-first — a single-variable grid comes back as ``(1, H, W)``
    # where earlier it was ``(H, W)``. A ``Source`` wraps one 2-D field, so take one slice off a leading
    # band axis: the one ``band`` names, on the same 1-based index the raster path's ``band`` uses.
    if z.ndim == 3:
        z = _netcdf_slice(z, named, band)
    attributes = _variable_attributes(nc, variable)
    if pinned and not attributes:
        # A subset's metadata holds no per-variable entry for the variable it already is, so the CF
        # attributes are read off the subset itself; without them the units and standard-name steps of the
        # style matcher cannot fire on the one input type they were written for.
        attributes = dict(getattr(nc, "attrs", None) or {})
    units = _attr(attributes, "units")
    return Source(
        z=_axis(z, "z", units),
        x=_axis(nc.lon, "x"),
        y=_axis(nc.lat, "y"),
        crs=crs if crs is not None else declared_crs(nc),
        units=units,
        metadata={
            "kind": "raster",
            # Left out rather than recorded as None in the one case the name is unrecoverable, so a reader
            # asking for it gets their own fallback.
            **({} if variable is None else {"variable": variable}),
            **_identity(attributes),
            **(metadata or {}),
        },
    )


def _fan_values(values: np.ndarray, mesh: Any, triangles: np.ndarray) -> np.ndarray:
    """Spread one value per mesh face over the fan triangles that face was cut into.

    A face with ``N`` nodes becomes ``N - 2`` triangles, so a face-located variable carries fewer values
    than the triangulation has triangles. Repeating each face's value across its own triangles is what
    keeps a quad flat-shaded as one colour instead of being silently misaligned onto other faces.

    Args:
        values: One value per mesh face.
        mesh: The pyramids ``Mesh2d`` whose connectivity was triangulated.
        triangles: The fan triangulation, used to check the spread result lines up with it.

    Returns:
        np.ndarray: one value per triangle.

    Raises:
        ValueError: when the repeated values do not match the triangle count, which means the mesh's
            connectivity and its fan triangulation disagree — colouring it would be a guess.
    """
    per_face = np.asarray(mesh.face_node_connectivity.nodes_per_element()) - 2
    # A face with fewer than three valid nodes yields no triangles and pyramids skips it, so clip rather
    # than repeat a negative count.
    spread = np.repeat(values, np.clip(per_face, 0, None))
    if spread.size != triangles.shape[0]:
        raise ValueError(
            f"the mesh has {triangles.shape[0]} triangles but its face values spread to "
            f"{spread.size}; connectivity and triangulation disagree"
        )
    return spread


def _mesh_step(
    values: np.ndarray, named: str, band: int, expected: int
) -> np.ndarray:
    """Reduce a temporal or layered mesh variable to the single step ``band`` names.

    ``UgridDataset.masked_values`` keeps a variable's full shape, and UGRID model output normally carries a
    leading time axis — so ``(n_steps, n_elements)`` is the common case, not the exception. A surface render
    colours one value per element, so the step is taken here: letting a 2-D array through reached the
    fan-spread as a bare numpy broadcast error for face data, and built a ``MeshSource`` with a 2-D ``z``
    for node data, against that class's own documented contract (review H4).

    Args:
        values: The variable as read — 1-D, or ``(n_steps, n_elements)``.
        named: How the variable reads in a message.
        band: 1-based index of the step to draw.
        expected: How many elements the mesh has at this variable's location.

    Returns:
        np.ndarray: one value per element.

    Raises:
        ValueError: when the shape cannot be reduced to one value per element, or when ``band`` names a
            step the variable does not carry.
    """
    if values.ndim == 1:
        if values.size != expected:
            raise ValueError(
                f"{named} holds {values.size} values but its mesh has {expected} at that location"
            )
        return values
    if values.ndim != 2 or int(values.shape[-1]) != expected:
        raise ValueError(
            f"{named} has shape {tuple(int(size) for size in values.shape)}, which a mesh render cannot "
            f"reduce to one value per element ({expected} expected on the last axis)"
        )
    steps = int(values.shape[0])
    if not 1 <= band <= steps:
        raise ValueError(
            f"band={band} is out of range for {named}, which carries {steps} step(s)"
        )
    if steps > 1 and band == DEFAULT_BAND:
        logger.warning(
            "The %s has %d steps; the Source draws band=%d (pass band= to choose another)",
            named,
            steps,
            DEFAULT_BAND,
        )
    taken: np.ndarray = values[band - 1]
    return taken


def _from_ugrid(
    ug: Any,
    variable: str | None,
    metadata: dict | None,
    *,
    band: int = DEFAULT_BAND,
    crs: Any = None,
) -> MeshSource:
    """Build a :class:`MeshSource` from a pyramids ``UgridDataset`` variable (defaults to the first).

    The geometry comes from pyramids rather than being rebuilt here: ``Mesh2d.fan_triangles`` cuts every
    face — triangle, quad or larger polygon — into triangles with pure numpy, which is both the one shape
    a triangular renderer takes and the reason this stays inside ``base/``'s engine-neutral rule.
    Re-triangulating the nodes instead (a Delaunay pass) would invent faces the mesh does not have and
    mis-assign face-located values onto them.

    Args:
        ug: The pyramids ``UgridDataset`` to read.
        variable: The data variable to draw; ``None`` takes the first.
        metadata: Extra metadata merged over the derived keys (the caller wins).
        band: 1-based step to draw from a variable carrying a time (or layer) axis.
        crs: The CRS the mesh's coordinates are already in, stored verbatim; ``None`` derives it.

    Returns:
        MeshSource: the mesh source, with values on the nodes or on the triangles.

    Raises:
        ValueError: when the mesh carries no data variables, when ``variable`` names one it does not
            carry, when the variable sits on a location a surface render cannot colour, or when its shape
            cannot be reduced to one value per element (including a ``band`` it does not carry).
    """
    carried = list(ug.data_variable_names)
    if variable is None:
        if not carried:
            raise ValueError("UgridDataset has no data variables to plot")
        variable = carried[0]
    elif variable not in carried:
        raise ValueError(
            f"variable={variable!r} is not in this UgridDataset, which carries {carried}"
        )
    held = ug.data_vars[variable]
    location = getattr(held, "location", None)
    if location not in MESH_LOCATIONS:
        raise ValueError(
            f"variable={variable!r} sits on {location!r}; a mesh render colours "
            f"{' or '.join(MESH_LOCATIONS)} values"
        )
    mesh = ug.mesh
    triangles = np.asarray(mesh.fan_triangles)
    values = np.ma.filled(
        np.ma.asarray(ug.masked_values(variable)).astype("float64"), np.nan
    )
    expected = int(mesh.n_face) if location == "face" else int(mesh.n_node)
    values = _mesh_step(values, f"variable {variable!r}", band, expected)
    if location == "face":
        values = _fan_values(values, mesh, triangles)
    units = getattr(held, "units", None)
    standard_name = getattr(held, "standard_name", None)
    return MeshSource(
        _axis(values, "z", units),
        _axis(mesh.node_x, "x"),
        _axis(mesh.node_y, "y"),
        triangles=triangles,
        location=location,
        crs=crs if crs is not None else declared_crs(ug),
        units=units,
        metadata={
            "kind": "mesh",
            "variable": variable,
            "location": location,
            "n_faces": int(mesh.n_face),
            "n_nodes": int(mesh.n_node),
            **({} if standard_name is None else {"standard_name": standard_name}),
            **(metadata or {}),
        },
    )


def _from_collection(
    dc: Any, band: int, metadata: dict | None, member: int = 0, *, crs: Any = None
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
    # Pass the collection position *through* _from_raster rather than rebuilding the Source afterwards: the
    # rebuild hand-listed a subset of keys and silently dropped everything else _from_raster derives (the GRIB
    # message metadata and its long_name). _from_raster merges this dict last (the caller's own values within
    # it still win), so member/n_members are recorded without losing the member's identity.
    position = {"member": member, "n_members": len(members), **(metadata or {})}
    return _from_raster(members[member], band, position, crs=crs)


def _from_feature(fc: Any, metadata: dict | None, *, crs: Any = None) -> Source:
    """Build a vector (point) :class:`Source` from a pyramids ``FeatureCollection``.

    ``FeatureCollection`` is a GeoDataFrame subclass, so we read its geometry/CRS directly. ``z`` is the
    first numeric non-geometry column (or ``None`` when there is none). "Numeric" is decided by pandas, so
    a nullable ``Int64``/``Float64`` counts and a ``string``/``boolean`` column does not. Bools are
    excluded, as ``np.issubdtype`` had them; timedeltas are too, which ``np.issubdtype`` did not, because
    a timedelta column cannot be rendered — picking one only moved the failure downstream. Point
    coordinates come from the geometry; non-point geometries fall back to their centroid.
    """
    geom_name = fc.geometry.name
    # The point-or-centroid choice and the coordinate read are PointArrays' — this was the copy the other
    # sites' spellings were measured against.
    points = PointArrays.from_features(fc, centroids=True)
    xs, ys = points.x, points.y

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
    x: np.ndarray | None,
    y: np.ndarray | None,
    metadata: dict | None,
    *,
    crs: Any = None,
) -> Source:
    """Build a raster :class:`Source` from a raw 2-D numpy array (pixel-index axes unless ``x``/``y`` given).

    A bare array carries no CRS, so ``crs`` stays ``None`` unless the caller names one — this is the case the
    contract's ``None`` is reserved for.

    **A masked array's mask is honoured**, as the module's first cross-cutting rule requires of every other
    input: the masked cells come back as ``NaN``, which is the nodata spelling the renderers already read.
    ``np.asarray`` alone hands back the *data* buffer and drops the mask, so ``np.ma.masked_equal(grid,
    -9999)`` extracted with its ``-9999`` sentinels intact and was drawn as if they were values — the very
    defect :func:`~digitalearth.base.arrays.read_masked_band` exists to stop for a pyramids raster (#343).
    A masked array is how a bare array says "nodata": there is no sidecar to carry a ``no_data_value``.

    Args:
        arr: The 2-D grid of values, plain or masked.
        x: Optional x coordinates; ``None`` takes the column indices.
        y: Optional y coordinates; ``None`` takes the row indices.
        metadata: Extra metadata merged over the derived keys (the caller wins).
        crs: The CRS the coordinates are already in, stored verbatim; ``None`` says it is unknown.

    Returns:
        Source: the raster source.

    Raises:
        ValueError: when `arr` is not 2-D.
    """
    a = (
        np.ma.filled(arr.astype("float64"), np.nan)
        if np.ma.isMaskedArray(arr)
        else np.asarray(arr, dtype="float64")
    )
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
