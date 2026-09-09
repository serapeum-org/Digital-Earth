"""sources — the uniform ``Source`` abstraction over pyramids inputs.

Public surface::

    from digitalearth.base.sources import Source, DimensionInfo, get_source
    src = get_source(dataset)          # pyramids Dataset/NetCDF/DatasetCollection/FeatureCollection or numpy
    src.z.values, src.x.values, src.crs

This package is a thin, pyramids-only extraction layer (no ``xarray``/``rasterio``). ``Source`` lives in its
own leaf module (:mod:`digitalearth.base.sources.source`) so importing it here and from :mod:`.extractors` does
not create a cycle.
"""

from typing import Optional

import numpy as np

from digitalearth.base.sources.dimension import DimensionInfo
from digitalearth.base.sources.extractors import extract, get_stack
from digitalearth.base.sources.source import Source
from digitalearth.base.types import PlottableData

__all__ = ["Source", "DimensionInfo", "get_source", "get_stack"]


def get_source(
    data: PlottableData,
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

    Examples:
        - Wrap a raw 2-D numpy array (pixel-index axes, no CRS):
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import get_source
            >>> src = get_source(np.arange(12.0).reshape(3, 4))
            >>> src.z.values.shape
            (3, 4)
            >>> src.x.values.tolist()
            [0.0, 1.0, 2.0, 3.0]
            >>> src.crs is None
            True

            ```
        - Supply explicit coordinates for a numpy array:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import get_source
            >>> src = get_source(np.zeros((2, 2)), x=np.array([10.0, 20.0]),
            ...                  y=np.array([5.0, 6.0]))
            >>> src.x.values.tolist()
            [10.0, 20.0]
            >>> src.metadata("kind")
            'raster'

            ```
    """
    return extract(data, band=band, variable=variable, x=x, y=y, metadata=metadata)
