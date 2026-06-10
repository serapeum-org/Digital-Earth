"""Structural types for the pyramids objects Digital-Earth consumes.

These ``Protocol`` classes document the *duck-typed surface* Digital-Earth actually uses from a pyramids raster
(``Dataset``) and vector layer (``FeatureCollection`` / geopandas ``GeoDataFrame``) — just enough to type the
public entry points without importing the upstream classes (Digital-Earth treats pyramids/cleopatra as
third-party). They are structural, so a real pyramids object satisfies them without subclassing anything here;
the goal is clearer signatures at the boundary in place of bare ``Any``.

Inside the plotting mixins, ``Any`` is still used deliberately for free-form ``**kwargs`` / ``opts`` forwarded
straight to cleopatra — typing those adds noise without safety.
"""
from typing import Any, Protocol, Union, runtime_checkable

import numpy as np


@runtime_checkable
class RasterLike(Protocol):
    """The raster surface Digital-Earth reads: CRS, reprojection, band arrays and cell coordinates.

    Any object exposing these members (notably a pyramids ``Dataset``) is accepted by the raster code paths.
    """

    @property
    def epsg(self) -> Any:
        """EPSG code of the raster's CRS."""
        ...

    @property
    def no_data_value(self) -> Any:
        """Per-band nodata sentinels (indexable by 0-based band)."""
        ...

    @property
    def x(self) -> Any:
        """1-D cell-centre x / longitude coordinates."""
        ...

    @property
    def y(self) -> Any:
        """1-D cell-centre y / latitude coordinates."""
        ...

    def to_crs(self, crs: Any) -> "RasterLike":
        """Reproject the raster to ``crs`` and return the reprojected raster."""
        ...

    def read_array(self, band: int = ...) -> np.ndarray:
        """Read a (0-based) band as a numpy array."""
        ...


@runtime_checkable
class VectorLike(Protocol):
    """The vector surface Digital-Earth reads: CRS, reprojection, geometry and attribute columns.

    Any object exposing these members (a pyramids ``FeatureCollection`` or geopandas ``GeoDataFrame``) is
    accepted by the vector code paths.
    """

    @property
    def epsg(self) -> Any:
        """EPSG code of the layer's CRS (may be ``None``)."""
        ...

    @property
    def crs(self) -> Any:
        """The layer's CRS object (may be ``None``)."""
        ...

    @property
    def geometry(self) -> Any:
        """The geometry column (a geopandas ``GeoSeries``)."""
        ...

    def to_crs(self, crs: Any) -> "VectorLike":
        """Reproject the layer to ``crs`` and return the reprojected layer."""
        ...

    def __len__(self) -> int:
        """Number of features in the layer."""
        ...


#: Anything the extractor / quick API can turn into something plottable: a single-band raster
#: (:class:`RasterLike`), a vector layer (:class:`VectorLike`), or a raw numpy array.
#:
#: This is a **documentation-grade approximation of the common case**, not the exhaustive accepted set.
#: ``get_source`` / ``extract`` also dispatch on a pyramids ``NetCDF`` (a ``Dataset`` subclass that exposes
#: ``lon``/``lat`` rather than ``x``/``y``, so it does *not* structurally satisfy :class:`RasterLike`) and a
#: ``DatasetCollection`` (one member is extracted). Those are matched by ``isinstance`` inside ``extract``,
#: not by this union; the annotation is a hint, never a runtime gate.
PlottableData = Union[RasterLike, VectorLike, np.ndarray]
