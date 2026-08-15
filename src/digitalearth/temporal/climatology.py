"""Climatology — group a time series by a periodic label and plot the group means + spread plume."""
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from cleopatra.glyphs.primitives.line_glyph import LineGlyph
from matplotlib.axes import Axes

from digitalearth import series
from digitalearth.temporal.timeseries import TimeSeries


class Climatology:
    """Group a ``DatasetCollection`` time series by a periodic label and summarise each group.

    For each group label (e.g. month-of-year) the per-step values are aggregated to a mean, with the
    group min/max forming a spread plume. :meth:`plot` draws the mean line and the plume.

    Args:
        collection: A pyramids ``DatasetCollection`` of ordered time steps.
        labels: One group label per member (same length/order as ``collection.datasets``).
        band: 1-based band read from each member.
        reducer: Spatial reducer for each member (passed to :class:`TimeSeries`).

    Attributes:
        labels: The per-member group labels.

    Examples:
        - Group a 6-step collection by a 3-season cycle into three group means:
            ```python
            >>> from pyramids.dataset.collection import DatasetCollection
            >>> from digitalearth.temporal import Climatology
            >>> dc = DatasetCollection.from_files(["examples/data/acc4000.tif"] * 6)
            >>> clim = Climatology(dc, ["djf", "jja", "son", "djf", "jja", "son"])
            >>> groups, mean, low, high = clim.climatology()
            >>> groups
            ['djf', 'jja', 'son']
            >>> mean.shape
            (3,)

            ```
    """

    def __init__(self, collection: Any, labels: Sequence, band: int = 1, reducer: str = "mean"):
        self._series = TimeSeries(collection, band=band, reducer=reducer)
        self.labels = list(labels)
        if len(self.labels) != len(collection.datasets):
            raise ValueError("labels length must match the number of collection members")

    def climatology(self) -> Tuple[List, np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate the series by group label.

        Returns:
            ``(groups, mean, low, high)``: sorted unique group labels and per-group mean / min / max of the
            reduced values.
        """
        values = self._series.values()
        labels = np.asarray(self.labels)
        groups = sorted(set(self.labels))
        mean = np.array([np.nanmean(values[labels == g]) for g in groups])
        low = np.array([np.nanmin(values[labels == g]) for g in groups])
        high = np.array([np.nanmax(values[labels == g]) for g in groups])
        return groups, mean, low, high

    def plot(self, ax: Optional[Axes] = None, **kwargs) -> Any:
        """Plot the climatology mean line with a min/max spread plume.

        Args:
            ax: Axes to draw on (created when ``None``).
            **kwargs: Forwarded to ``LineGlyph.line`` for the mean line.

        Returns:
            The ``LineGlyph.line`` result ``(fig, ax, ...)`` for the mean line.
        """
        groups, mean, low, high = self.climatology()
        x = np.arange(len(groups))
        glyph = LineGlyph(x, mean, ax=ax, fig=ax.get_figure() if ax is not None else None)
        line = glyph.line(ax=ax, **kwargs)
        plot_ax = line[1] if isinstance(line, tuple) else ax
        series.envelope(x, low, high, ax=plot_ax, alpha=0.25)
        return line
