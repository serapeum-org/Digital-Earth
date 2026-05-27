"""Source — the uniform wrapper the glyph wiring consumes.

A ``Source`` is the single representation every Digital-Earth plot method reads from: it carries the data
array (``z``), the coordinate axes (``x``/``y``), the CRS, and free-form metadata, regardless of whether
the input was a pyramids raster, NetCDF variable, dataset collection, feature collection or raw numpy.

This module is a **leaf**: it imports nothing from :mod:`digitalearth.sources.extractors`, so that
``extractors`` and the package ``__init__`` can both import ``Source`` without creating an import cycle.
"""
from typing import Any, Optional

from digitalearth.sources.dimension import DimensionInfo


class Source:
    """Uniform container for plottable data extracted from a pyramids object (or numpy array).

    Attributes are exposed read-only via properties. For raster-like sources ``z`` is a 2-D grid and
    ``x``/``y`` are 1-D axis vectors of length ``columns``/``rows``; for vector (point) sources ``z`` is a
    1-D value array (or ``None``) aligned with the 1-D ``x``/``y`` point coordinates.

    Args:
        z: The data dimension (a :class:`DimensionInfo`), or ``None`` for geometry-only vector sources.
        x: The x / longitude axis (a :class:`DimensionInfo`).
        y: The y / latitude axis (a :class:`DimensionInfo`).
        crs: The CRS as an EPSG integer or WKT string (whatever pyramids reported), or ``None``.
        metadata: Free-form metadata dict (e.g. ``variable``, ``kind``, ``time``, ``member``).
        units: Unit string for the data values, or ``None``.
    """

    def __init__(
        self,
        z: Optional[DimensionInfo],
        x: DimensionInfo,
        y: DimensionInfo,
        crs: Any = None,
        metadata: Optional[dict] = None,
        units: Optional[str] = None,
    ):
        self._z, self._x, self._y = z, x, y
        self._crs, self._meta, self._units = crs, metadata or {}, units

    @property
    def z(self) -> Optional[DimensionInfo]:
        """The data dimension (2-D grid for rasters, 1-D values for points), or ``None``."""
        return self._z

    @property
    def x(self) -> DimensionInfo:
        """The x / longitude dimension."""
        return self._x

    @property
    def y(self) -> DimensionInfo:
        """The y / latitude dimension."""
        return self._y

    @property
    def crs(self) -> Any:
        """The CRS as reported by pyramids (EPSG int or WKT str), or ``None``."""
        return self._crs

    @property
    def units(self) -> Optional[str]:
        """Unit string for the data values, or ``None``."""
        return self._units

    def metadata(self, key: str, default: Any = None) -> Any:
        """Return a metadata value.

        Args:
            key: Metadata key to look up.
            default: Value returned when ``key`` is absent.

        Returns:
            The stored value, or ``default`` if the key is not present.
        """
        return self._meta.get(key, default)
