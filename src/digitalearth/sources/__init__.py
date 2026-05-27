"""sources — the uniform ``Source`` abstraction over pyramids inputs.

Public surface::

    from digitalearth.sources import Source, DimensionInfo, get_source
    src = get_source(dataset)          # pyramids Dataset/NetCDF/DatasetCollection/FeatureCollection or numpy
    src.z.values, src.x.values, src.crs

This package is a thin, pyramids-only extraction layer (no ``xarray``/``rasterio``). ``Source`` lives in its
own leaf module (:mod:`digitalearth.sources.source`) so importing it here and from :mod:`.extractors` does
not create a cycle.
"""
from typing import Any, Optional

import numpy as np

from digitalearth.sources.dimension import DimensionInfo
from digitalearth.sources.extractors import extract
from digitalearth.sources.source import Source

__all__ = ["Source", "DimensionInfo", "get_source"]


def get_source(
    data: Any,
    *,
    band: int = 1,
    variable: Optional[str] = None,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    metadata: Optional[dict] = None,
) -> Source:
    """Build a :class:`Source` from any supported input (dispatch entry point).

    Args:
        data: A pyramids ``Dataset``/``NetCDF``/``DatasetCollection``/``FeatureCollection`` or numpy array.
        band: 1-based band index for raster/collection inputs.
        variable: Variable name for ``NetCDF`` inputs (defaults to the first variable).
        x: Optional x coordinates for a raw numpy array (defaults to pixel indices).
        y: Optional y coordinates for a raw numpy array (defaults to pixel indices).
        metadata: Extra metadata merged into the resulting ``Source``.

    Returns:
        Source: the uniform wrapper the glyph wiring consumes.
    """
    return extract(data, band=band, variable=variable, x=x, y=y, metadata=metadata)
