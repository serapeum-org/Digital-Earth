"""Source — the uniform wrapper the glyph wiring consumes.

A ``Source`` is the single representation every Digital-Earth plot method reads from: it carries the data
array (``z``), the coordinate axes (``x``/``y``), the CRS, and free-form metadata, regardless of whether
the input was a pyramids raster, NetCDF variable, dataset collection, feature collection or raw numpy.

This module is a **leaf**: it imports nothing from :mod:`digitalearth.base.sources.extractors`, so that
``extractors`` and the package ``__init__`` can both import ``Source`` without creating an import cycle.
"""

from typing import Any, Optional

from digitalearth.base.crs import authority_code
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
        """The x / longitude dimension.

        Returns:
            The :class:`DimensionInfo` holding the x coordinates: one per column for a raster, one per point
            for a vector source.

        Note:
            For a **vector** source the coordinate array is read-only: it comes from
            :class:`~digitalearth.base.points.PointArrays`, which freezes its arrays so one reading cannot be
            changed under another holder of it. Copy it (``np.array(source.x.values)``) before modifying in
            place. This is not new under pandas 3, whose copy-on-write already made the geometry's ``.x``
            read-only; it is new, and unconditional, for an install on pandas 2. A raster source builds its
            axes fresh, and they are writable. So are a plain array's pixel-index axes — but coordinates the
            caller passed as `x` / `y` are wrapped without a copy, sharing the caller's buffer and keeping
            its writability.

        Examples:
            - A plain array's x holds one pixel index per column, and can be modified in place:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import get_source
                >>> src = get_source(np.zeros((2, 3)))
                >>> src.x.values.tolist(), src.x.values.flags.writeable
                ([0.0, 1.0, 2.0], True)

                ```
            - A vector source's x is read-only, so it is copied before being changed:
                ```python
                >>> import geopandas as gpd
                >>> import numpy as np
                >>> from pyramids.feature import FeatureCollection
                >>> from shapely.geometry import Point
                >>> from digitalearth.base.sources import get_source
                >>> frame = gpd.GeoDataFrame(geometry=[Point(0, 1), Point(2, 3)], crs="EPSG:4326")
                >>> src = get_source(FeatureCollection(frame))
                >>> src.x.values.flags.writeable
                False
                >>> shifted = np.array(src.x.values)
                >>> shifted += 10.0
                >>> shifted.tolist(), src.x.values.tolist()
                ([10.0, 12.0], [0.0, 2.0])

                ```
        """
        return self._x

    @property
    def y(self) -> DimensionInfo:
        """The y / latitude dimension.

        Returns:
            The :class:`DimensionInfo` holding the y coordinates: one per row for a raster, one per point for
            a vector source.

        Note:
            Read-only for a vector source, for the reason given on :attr:`x`.

        Examples:
            - A north-up raster's y runs from the top row's cell centre down:
                ```python
                >>> import numpy as np
                >>> from pyramids.base.georeference import GeoReference
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.base.sources import get_source
                >>> grid = Dataset.from_array(
                ...     np.zeros((2, 3)),
                ...     geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
                ... )
                >>> get_source(grid).y.values.tolist()
                [1.5, 0.5]

                ```
            - A vector source's y is read-only, like its x:
                ```python
                >>> import geopandas as gpd
                >>> from pyramids.feature import FeatureCollection
                >>> from shapely.geometry import Point
                >>> from digitalearth.base.sources import get_source
                >>> frame = gpd.GeoDataFrame(geometry=[Point(0, 1), Point(2, 3)], crs="EPSG:4326")
                >>> src = get_source(FeatureCollection(frame))
                >>> src.y.values.tolist(), src.y.values.flags.writeable
                ([1.0, 3.0], False)

                ```
        """
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

        Returns:
            The CRS exactly as it was supplied or derived: an EPSG ``int``, an ``"EPSG:<code>"`` /
            proj4 / WKT ``str``, or ``None`` when the CRS is genuinely unknown.

        Examples:
            - A caller that warps into a display CRS stores that CRS verbatim, code and all:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import Source, DimensionInfo
                >>> axis = DimensionInfo(np.array([0.0]), "x")
                >>> Source(None, axis, axis, crs=3857).crs
                3857

                ```
            - A projection with no authority code keeps its **definition** here — this is the case
              ``None`` is never used for, because the coordinates do have an address:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import Source, DimensionInfo
                >>> axis = DimensionInfo(np.array([0.0]), "x")
                >>> src = Source(None, axis, axis, crs="+proj=ortho +lat_0=53 +lon_0=4")
                >>> src.crs
                '+proj=ortho +lat_0=53 +lon_0=4'
                >>> src.epsg is None
                True

                ```
            - ``None`` is reserved for a genuinely unknown CRS — a raw numpy array declares none:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import get_source
                >>> get_source(np.zeros((2, 3))).crs is None
                True

                ```

        See Also:
            epsg: the narrower question — the authority code, or ``None`` when there is none.
        """
        return self._crs

    @property
    def epsg(self) -> Optional[int]:
        """The EPSG code of :attr:`crs`, or ``None`` when it has none.

        The narrower of the two CRS questions: :attr:`crs` always says where the coordinates are, while this
        answers only "is there an authority code for it". ``None`` therefore means the CRS genuinely names
        no authority — not merely that it was written out rather than abbreviated.

        The distinction matters because :attr:`crs` is allowed to hold a full definition: a caller that warps
        into a display CRS stores what pyramids reports, and a warp into a *coded* CRS reports its WKT, which
        ends on the authority block (``AUTHORITY["EPSG", ...]`` in WKT1, ``ID["EPSG", ...]`` in WKT2). The
        code spellings are read here directly and anything longer is handed to
        :func:`~digitalearth.base.crs.authority_code`, so pyramids does the reading — the same rule
        :func:`~digitalearth.base.crs.is_geographic` follows, and for the same reason: a reader that only
        matched ``"EPSG:<code>"`` reported "no code" for a CRS that plainly carries one.

        Returns:
            The EPSG integer, or ``None`` for an unknown CRS or one that names no authority.

        Examples:
            - An EPSG-coded source answers both questions the same way:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import Source, DimensionInfo
                >>> axis = DimensionInfo(np.array([0.0]), "x")
                >>> Source(None, axis, axis, crs="EPSG:3857").epsg
                3857

                ```
            - A CRS written out as a definition is read down to the authority block it carries:
                ```python
                >>> import numpy as np
                >>> from pyramids.base.crs import crs_from_user_input
                >>> from digitalearth.base.sources import Source, DimensionInfo
                >>> axis = DimensionInfo(np.array([0.0]), "x")
                >>> Source(None, axis, axis, crs=crs_from_user_input(3857).to_wkt()).epsg
                3857

                ```
            - A projection that names no authority keeps its definition and has no code:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import Source, DimensionInfo
                >>> axis = DimensionInfo(np.array([0.0]), "x")
                >>> src = Source(None, axis, axis, crs="+proj=ortho +lat_0=53 +lon_0=4")
                >>> src.epsg is None, src.crs.startswith("+proj=ortho")
                (True, True)

                ```

        See Also:
            crs: the wider question — where the coordinates are, in whatever spelling was supplied.
        """
        crs = self._crs
        if isinstance(crs, bool) or crs is None:
            return None
        if isinstance(crs, int):
            return crs
        text = str(crs).strip()
        if text.lower().startswith("epsg:"):
            text = text.split(":", 1)[1].strip()
        if text.isdigit():
            return int(text)
        return authority_code(crs)

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
