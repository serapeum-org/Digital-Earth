"""Source — the uniform wrapper the glyph wiring consumes.

A ``Source`` is the single representation every Digital-Earth plot method reads from: it carries the data
array (``z``), the coordinate axes (``x``/``y``), the CRS, and free-form metadata, regardless of whether
the input was a pyramids raster, NetCDF variable, dataset collection, feature collection or raw numpy.

This module is a **leaf**: it imports nothing from :mod:`digitalearth.base.sources.extractors`, so that
``extractors`` and the package ``__init__`` can both import ``Source`` without creating an import cycle.
"""

from typing import Any, Optional

from digitalearth.base.sources.dimension import DimensionInfo


class Source:
    """Uniform container for plottable data extracted from a pyramids object (or numpy array).

    Attributes are exposed read-only via properties. For raster-like sources ``z`` is a 2-D grid and
    ``x``/``y`` are 1-D axis vectors of length ``columns``/``rows``; for vector (point) sources ``z`` is a
    1-D value array (or ``None``) aligned with the 1-D ``x``/``y`` point coordinates.

    Args:
        z: The data dimension (a :class:`DimensionInfo`), or ``None`` for geometry-only vector sources.
        x: The x / longitude axis (a :class:`DimensionInfo`).
        y: The y / latitude axis (a :class:`DimensionInfo`).
        crs: The CRS the coordinates in ``x``/``y`` are expressed in, **as given** — see :attr:`crs` for
            the contract.
        metadata: Free-form metadata dict (e.g. ``variable``, ``kind``, ``time``, ``member``).
        units: Unit string for the data values, or ``None``.

    Examples:
        - Build a raster Source by hand and read its grid + axes:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import Source, DimensionInfo
            >>> z = DimensionInfo(np.arange(6.0).reshape(2, 3), "z", "mm")
            >>> src = Source(z, DimensionInfo(np.array([0.0, 1.0, 2.0]), "x"),
            ...              DimensionInfo(np.array([0.0, 1.0]), "y"), crs=4326,
            ...              metadata={"variable": "rain"}, units="mm")
            >>> src.z.values.shape
            (2, 3)
            >>> src.crs
            4326
            >>> src.epsg
            4326
            >>> src.metadata("variable")
            'rain'

            ```
        - Missing metadata keys fall back to the supplied default:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import Source, DimensionInfo
            >>> src = Source(None, DimensionInfo(np.array([0.0]), "x"),
            ...              DimensionInfo(np.array([0.0]), "y"))
            >>> src.metadata("variable", "unknown")
            'unknown'
            >>> src.units is None
            True

            ```
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
        """The CRS the coordinates in :attr:`x` / :attr:`y` are expressed in, as given.

        The contract, so a reader can trust the answer:

        * It is the address of **these** coordinates, not of the file they came from. A caller that warps
          data into a display CRS before wrapping it passes that CRS to ``get_source(..., crs=...)``, and it
          is stored verbatim — an EPSG ``int``, or a proj4/WKT ``str`` for a projection with no authority
          code (an orthographic globe, say, where pyramids reports ``epsg is None``).
        * Without such a caller it is derived from the input: its EPSG code when it has one, else its
          projection definition.
        * ``None`` means the CRS is genuinely **unknown** — a raw numpy array, or an input declaring no CRS
          at all. It never means "there was a CRS but no code for it"; that case yields the definition.

        Use :attr:`epsg` for the separate code-or-``None`` question.
        """
        return self._crs

    @property
    def epsg(self) -> Optional[int]:
        """The EPSG code of :attr:`crs`, or ``None`` when it has none.

        The narrower of the two CRS questions: :attr:`crs` always says where the coordinates are, while this
        answers only "is there an authority code for it". A projection defined by proj4/WKT alone gives
        ``None`` here and still gives its definition from :attr:`crs`.

        Returns:
            The EPSG integer, or ``None`` for an unknown or code-less CRS.

        Examples:
            - An EPSG-coded source answers both questions the same way:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import Source, DimensionInfo
                >>> axis = DimensionInfo(np.array([0.0]), "x")
                >>> Source(None, axis, axis, crs="EPSG:3857").epsg
                3857

                ```
            - A projection with no authority code keeps its definition but has no code:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import Source, DimensionInfo
                >>> axis = DimensionInfo(np.array([0.0]), "x")
                >>> src = Source(None, axis, axis, crs="+proj=ortho +lat_0=53 +lon_0=4")
                >>> src.epsg is None, src.crs.startswith("+proj=ortho")
                (True, True)

                ```
        """
        crs = self._crs
        if isinstance(crs, bool) or crs is None:
            return None
        if isinstance(crs, int):
            return crs
        text = str(crs).strip()
        if text.lower().startswith("epsg:"):
            text = text.split(":", 1)[1].strip()
        return int(text) if text.isdigit() else None

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
