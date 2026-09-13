"""Shared array helpers — the small numpy chores duplicated across the wiring modules.

These operations recurred verbatim across :mod:`digitalearth.base.sources.extractors`,
:mod:`digitalearth.static.charts`, :mod:`digitalearth.static.temporal` and
:mod:`digitalearth.static.textured_globe`, with subtly different nodata-masking rules. They live here once so
every caller masks the same way. (``static.map`` and ``static.series`` were consumers before the backend
restructure split ``fig_of`` out into :mod:`digitalearth.static.figures`; they no longer import from here.)

:func:`read_masked_band` is how a raster's nodata is masked — the one way, for every caller. It asks pyramids
for the mask instead of rebuilding it: ``read_array(band=..., masked=True)``, then the mask filled with
``NaN``. That is the only correct reading since pyramids 0.62.0, whose ``read_array`` unpacks CF-packed bands
(``scale_factor``/``add_offset``) to physical units by default while ``no_data_value`` stays a **stored**
sentinel — upstream's own docstring says to compare the sentinel against an ``unpack=False`` read or let
``masked=True`` build the mask, never to match it against physical values. Comparing the two by hand silently
missed every nodata cell of a packed band (a stored ``-9999`` reads as ``-98.49`` at ``scale=0.01,
offset=1.5``). Going through pyramids also honours the band's GDAL **mask/alpha band**, which the hand-rolled
comparison never saw.

Masking used to have a second tier for callers holding raw values and a sentinel with no dataset behind them —
``mask_nodata``, plus the ``_band_nodata`` sentinel reader it was paired with. Routing every raster read
through pyramids left both without a caller, so they were removed rather than kept as a second, wronger way to
mask.

:func:`ring_runs` is here for the same reason: both globe tiers — the 2-D projected disc in
:mod:`digitalearth.static.projections` and the 3-D sphere in :mod:`digitalearth.static.textured_globe` — clip a
closed ring to what the camera sees, and both split it into visible runs first. Doing that on a boolean mask
keeps it engine-neutral, and doing it once keeps the seam handling the same in both.

The one matplotlib chore that used to sit alongside these (``fig_of``) is not here: when the old flat
``digitalearth._arrays`` was split it went to :mod:`digitalearth.static.figures`, since a figure lookup belongs to
the matplotlib backend rather than to the engine-neutral shared layer.
"""

from typing import Any, Callable, Dict, List

import numpy as np

__all__ = ["NAN_REDUCERS", "finite", "read_masked_band", "ring_runs"]


def ring_runs(visible: Any) -> List[np.ndarray]:
    """Split a closed ring's visibility mask into runs of consecutive visible vertex indices.

    A ring is circular, so the stretch that runs off the end of the array carries on at its start. Splitting the
    array as a straight line would cut that stretch in two at the seam, and a caller closing each run along a
    horizon would then add a spur from wherever the data happens to begin out to the edge. Here the run that
    wraps is returned whole, its indices continuing past the end (``[8, 9, 0, 1]``), and the runs come back in
    the order they occur round the ring, so each one's neighbours on either side are hidden.

    Pass the ring *without* the repeated closing vertex a GeoJSON-style ring ends with; that repeat would
    otherwise sit in the middle of the wrapped run.

    Args:
        visible: One boolean per vertex, ``True`` where the vertex can be seen.

    Returns:
        One integer index array per run of visible vertices, in ring order — one run covering every index when
        the whole ring is visible, and an empty list when none of it is.

    Examples:
        - The visible stretch that crosses the seam comes back as one run, not two:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.arrays import ring_runs
            >>> [run.tolist() for run in ring_runs([True, True, False, False, True])]
            [[4, 0, 1]]

            ```
        - Separate stretches stay separate, in the order they occur round the ring:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.arrays import ring_runs
            >>> [run.tolist() for run in ring_runs([False, True, False, True, True, False])]
            [[1], [3, 4]]

            ```
        - A wholly visible ring is one run; a wholly hidden one has none:
            ```python
            >>> from digitalearth.base.arrays import ring_runs
            >>> [run.tolist() for run in ring_runs([True, True, True])], ring_runs([False, False])
            ([[0, 1, 2]], [])

            ```
    """
    mask = np.asarray(visible, dtype=bool).ravel()
    if not mask.any():
        return []
    count = mask.size
    if mask.all():
        return [np.arange(count)]
    # start the scan on a hidden vertex, so no run can straddle the array's end
    offset = int(np.flatnonzero(~mask)[0])
    shown = np.flatnonzero(np.roll(mask, -offset))
    runs = np.split(shown, np.flatnonzero(np.diff(shown) != 1) + 1)
    return [(run + offset) % count for run in runs]


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
            >>> from digitalearth.base.arrays import finite
            >>> finite(np.array([[1.0, np.nan], [np.inf, 4.0]])).tolist()
            [1.0, 4.0]

            ```
        - An all-non-finite input collapses to an empty array:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.arrays import finite
            >>> int(finite(np.array([np.nan, -np.inf])).size)
            0

            ```
    """
    a = np.asarray(arr, dtype="float64").ravel()
    return a[np.isfinite(a)]


def read_masked_band(dataset: Any, band: int = 1) -> np.ndarray:
    """Read a 1-based ``band`` of a pyramids ``Dataset`` as ``float64`` with its masked cells set to ``NaN``.

    The mask comes from pyramids (``read_array(band=..., masked=True)``), not from comparing values against
    ``no_data_value`` here. Two reasons, both of which the hand-rolled comparison got wrong:

    * ``read_array`` unpacks a CF-packed band (``scale_factor``/``add_offset``) to physical units, while
      ``no_data_value`` stays a **stored** sentinel — so the two are different numbers and no cell ever
      matched. Letting pyramids build the mask is what its own docstring prescribes.
    * A band's GDAL **mask/alpha band** marks cells that carry no sentinel at all; ``masked=True`` folds it in.

    Args:
        dataset: A pyramids ``Dataset`` (duck-typed by ``read_array(band=..., masked=True)``).
        band: 1-based band index (the first band is ``1``); read internally as ``band - 1``.

    Returns:
        The band as a ``float64`` array (same shape as stored) with every masked cell replaced by ``NaN``.

    Examples:
        - A packed band's nodata cell is masked even though its physical value is nothing like the sentinel:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from digitalearth.base.arrays import read_masked_band
            >>> ds = Dataset.from_array(
            ...     np.array([[100, 200], [300, -9999]], dtype="int16"),
            ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
            ...     no_data_value=-9999,
            ... )
            >>> ds.scale, ds.offset = [0.01], [1.5]
            >>> read_masked_band(ds, band=1).tolist()
            [[2.5, 3.5], [4.5, nan]]

            ```
    """
    idx = band - 1
    values = np.ma.asarray(dataset.read_array(band=idx, masked=True)).astype("float64")
    return np.ma.filled(values, np.nan)
