"""Shared array helpers — the small numpy chores duplicated across the wiring modules.

Three operations recurred verbatim across :mod:`digitalearth.sources.extractors`, :mod:`digitalearth.scene.map`,
:mod:`digitalearth.charts`, :mod:`digitalearth.series` and :mod:`digitalearth.temporal`, with subtly different
nodata-masking rules. They live here once so every caller masks the same way.

Masking uses an **exact** comparison against the nodata sentinel (``arr == nodata``): a nodata value is a sentinel
read straight from the dataset, so it is reproduced exactly in the array, and an exact test cannot accidentally null
legitimate values that merely sit close to the sentinel (which ``np.isclose`` could). This is pure numpy — no
pyramids/cleopatra import — so the module stays a leaf consumable from anywhere.
"""
from typing import Any, Callable, Dict, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

#: NaN-aware spatial/array reducers keyed by name — the single source consumed by the temporal time-series
#: reducer and the quadtree per-cell aggregator (which adds its own ``"count"`` on top). Each maps a name to a
#: callable taking a 1-D array and returning a scalar.
NAN_REDUCERS: Dict[str, Callable[..., Any]] = {
    "mean": np.nanmean,
    "sum": np.nansum,
    "median": np.nanmedian,
    "min": np.nanmin,
    "max": np.nanmax,
    "std": np.nanstd,
}


def fig_of(ax: Optional[Axes]) -> Optional[Figure]:
    """Return the figure that owns ``ax``, or ``None`` when ``ax`` is ``None``.

    Args:
        ax: A matplotlib axes, or ``None``.

    Returns:
        The owning :class:`~matplotlib.figure.Figure`, or ``None`` when ``ax`` is ``None``.

    Examples:
        - An axes reports the figure that owns it:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from digitalearth._arrays import fig_of
            >>> fig, ax = plt.subplots()
            >>> fig_of(ax) is fig
            True

            ```
        - ``None`` short-circuits to ``None``:
            ```python
            >>> from digitalearth._arrays import fig_of
            >>> fig_of(None) is None
            True

            ```
    """
    return ax.get_figure() if ax is not None else None


def mask_nodata(arr: Any, nodata: Optional[float]) -> np.ndarray:
    """Return ``arr`` as ``float64`` with cells equal to ``nodata`` replaced by ``NaN``.

    Args:
        arr: Any array-like of values.
        nodata: The nodata sentinel to null out, or ``None`` to leave every value untouched.

    Returns:
        A ``float64`` copy of ``arr`` with exact ``nodata`` matches set to ``NaN``.

    Examples:
        - The sentinel becomes ``NaN``; everything else is preserved:
            ```python
            >>> import numpy as np
            >>> from digitalearth._arrays import mask_nodata
            >>> mask_nodata(np.array([1.0, -9999.0, 3.0]), -9999.0).tolist()
            [1.0, nan, 3.0]

            ```
        - ``None`` nodata is a no-op (just a float cast):
            ```python
            >>> import numpy as np
            >>> from digitalearth._arrays import mask_nodata
            >>> mask_nodata(np.array([1, 2, 3]), None).tolist()
            [1.0, 2.0, 3.0]

            ```
    """
    a = np.asarray(arr, dtype="float64")
    if nodata is None:
        return a
    return np.where(a == nodata, np.nan, a)


def finite(arr: Any) -> np.ndarray:
    """Return the flattened, finite (non-``NaN``/non-``inf``) values of ``arr`` as a 1-D ``float64`` array.

    Args:
        arr: Any array-like; flattened in row-major order before filtering.

    Returns:
        A 1-D ``float64`` array of the finite values (empty when every value is non-finite).

    Examples:
        - ``NaN`` and ``inf`` are dropped and the result is flattened:
            ```python
            >>> import numpy as np
            >>> from digitalearth._arrays import finite
            >>> finite(np.array([[1.0, np.nan], [np.inf, 4.0]])).tolist()
            [1.0, 4.0]

            ```
        - An all-non-finite input collapses to an empty array:
            ```python
            >>> import numpy as np
            >>> from digitalearth._arrays import finite
            >>> int(finite(np.array([np.nan, -np.inf])).size)
            0

            ```
    """
    a = np.asarray(arr, dtype="float64").ravel()
    return a[np.isfinite(a)]


def _band_nodata(dataset: Any, index: int) -> Optional[float]:
    """Safely read the 0-based band ``index`` nodata from a dataset's ``no_data_value`` tuple.

    Args:
        dataset: An object exposing a ``no_data_value`` sequence (e.g. a pyramids ``Dataset``).
        index: 0-based band index into ``no_data_value``.

    Returns:
        The sentinel at ``index``, or ``None`` when it is missing, out of range, or unset.

    Examples:
        - Read the sentinel for a specific band:
            ```python
            >>> from types import SimpleNamespace
            >>> from digitalearth._arrays import _band_nodata
            >>> _band_nodata(SimpleNamespace(no_data_value=(-1.0, -2.0)), 1)
            -2.0

            ```
        - An out-of-range index returns ``None`` instead of raising:
            ```python
            >>> from types import SimpleNamespace
            >>> from digitalearth._arrays import _band_nodata
            >>> _band_nodata(SimpleNamespace(no_data_value=(-1.0,)), 5) is None
            True

            ```
    """
    ndv = getattr(dataset, "no_data_value", None)
    if not ndv:
        return None
    try:
        return ndv[index]
    except (IndexError, TypeError, KeyError):
        return None


def read_masked_band(dataset: Any, band: int = 1) -> np.ndarray:
    """Read a 1-based ``band`` of a pyramids ``Dataset`` as ``float64`` with its nodata cells set to ``NaN``.

    Args:
        dataset: A pyramids ``Dataset`` (duck-typed by ``read_array`` / ``no_data_value``).
        band: 1-based band index (the first band is ``1``); read internally as ``band - 1``.

    Returns:
        The band as a ``float64`` array (same shape as stored) with nodata cells replaced by ``NaN``.

    Examples:
        - Read band 1 (1-based) and null its nodata cells:
            ```python
            >>> import numpy as np
            >>> from types import SimpleNamespace
            >>> from digitalearth._arrays import read_masked_band
            >>> ds = SimpleNamespace(no_data_value=(-1.0,),
            ...                      read_array=lambda band=0: np.array([[5.0, -1.0]]))
            >>> read_masked_band(ds, band=1).tolist()
            [[5.0, nan]]

            ```
    """
    idx = band - 1
    return mask_nodata(dataset.read_array(band=idx), _band_nodata(dataset, idx))
