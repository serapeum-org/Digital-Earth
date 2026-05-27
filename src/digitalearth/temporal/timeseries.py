"""TimeSeries — reduce a DatasetCollection to a per-time-step series and plot it as a line."""
from typing import Any, Optional, Sequence

import numpy as np
from cleopatra.line_glyph import LineGlyph
from matplotlib.axes import Axes


class TimeSeries:
    """A 1-D time series formed by reducing each member of a ``DatasetCollection`` to one value.

    Each member (time step) is reduced over space by a numpy reducer (default ``nanmean``), giving one
    value per step; :meth:`plot` draws the resulting series via ``cleopatra.LineGlyph``.

    Args:
        collection: A pyramids ``DatasetCollection`` whose members are ordered time steps.
        band: 1-based band read from each member.
        reducer: Spatial reducer applied to each member's array (``"mean"``, ``"sum"``, ``"min"``, ``"max"``).

    Attributes:
        collection: The source collection.
        band: The band index used.
        reducer: The configured reducer name.

    Examples:
        - Reduce a 3-step collection to one spatial-mean value per step:
            ```python
            >>> from pyramids.dataset.collection import DatasetCollection
            >>> from digitalearth.temporal import TimeSeries
            >>> dc = DatasetCollection.from_files(["examples/data/acc4000.tif"] * 3)
            >>> ts = TimeSeries(dc, reducer="mean")
            >>> ts.values().shape
            (3,)

            ```
        - Identical members give a constant series:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset.collection import DatasetCollection
            >>> from digitalearth.temporal import TimeSeries
            >>> dc = DatasetCollection.from_files(["examples/data/acc4000.tif"] * 3)
            >>> vals = TimeSeries(dc).values()
            >>> bool(np.allclose(vals, vals[0]))
            True

            ```
    """

    _REDUCERS = {"mean": np.nanmean, "sum": np.nansum, "min": np.nanmin, "max": np.nanmax}

    def __init__(self, collection: Any, band: int = 1, reducer: str = "mean"):
        if reducer not in self._REDUCERS:
            raise ValueError(f"unknown reducer {reducer!r}; choose from {sorted(self._REDUCERS)}")
        self.collection = collection
        self.band = band
        self.reducer = reducer

    def values(self) -> np.ndarray:
        """Return the reduced value of each member (one per time step).

        Returns:
            1-D array of length ``len(collection.datasets)``.
        """
        func = self._REDUCERS[self.reducer]
        out = []
        for member in self.collection.datasets:
            arr = member.read_array(band=self.band - 1).astype("float64")
            nodata = member.no_data_value[self.band - 1]
            if nodata is not None:
                arr = np.where(np.isclose(arr, nodata, rtol=1e-3), np.nan, arr)
            out.append(func(arr))
        return np.asarray(out)

    def plot(self, times: Optional[Sequence] = None, ax: Optional[Axes] = None, **kwargs) -> Any:
        """Plot the series as a line.

        Args:
            times: Optional x values (defaults to ``range(n_steps)``).
            ax: Axes to draw on (created when ``None``).
            **kwargs: Forwarded to ``LineGlyph.line``.

        Returns:
            The ``LineGlyph.line`` result ``(fig, ax, ...)``.
        """
        y = self.values()
        x = np.arange(len(y)) if times is None else np.asarray(times)
        glyph = LineGlyph(x, y, ax=ax, fig=ax.get_figure() if ax is not None else None)
        return glyph.line(ax=ax, **kwargs)
