"""Engine-neutral time reductions — the aggregation half of the temporal products.

The temporal products (``TimeSeries``, ``Climatology``) do two separable things: they **reduce** a
``DatasetCollection`` over time, and they **plot** the result. Only the plotting needs a renderer; the
reduction is pure ``numpy`` + pyramids. It lived inside the static backend (``static/temporal/``) because that
is where the products were first written, but the reduction is useful to any backend — the interactive and web
tiers have their own temporal modules that could compute the same series — so it belongs in ``base``.

This module holds that reduction. It imports no renderer (``cleopatra``/``matplotlib``/``pyvista``/…), so it
passes ``base``'s engine-neutral guard (:mod:`tests.test_base_is_engine_neutral`); each backend's temporal
module consumes these functions and adds its own drawing.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from digitalearth.base.arrays import NAN_REDUCERS, read_masked_band
from digitalearth.base.spec import DEFAULT_BAND

#: The spatial reducers a time series may be formed with — the NaN-skipping numpy reducers, named. Shared so a
#: backend validates a ``reducer=`` against the same set the reduction uses.
SERIES_REDUCERS = {name: NAN_REDUCERS[name] for name in ("mean", "sum", "min", "max")}


def series_values(
    collection: Any, band: int = DEFAULT_BAND, reducer: str = "mean"
) -> np.ndarray:
    """Reduce each member of a ``DatasetCollection`` over space to one value per time step.

    Args:
        collection: A pyramids ``DatasetCollection`` whose members are ordered time steps.
        band: 1-based band read from each member.
        reducer: The spatial reducer applied to each member's masked array — one of
            :data:`SERIES_REDUCERS` (``"mean"``, ``"sum"``, ``"min"``, ``"max"``), NaN-skipping.

    Returns:
        A 1-D ``float`` array of length ``len(collection.datasets)`` — the reduced value of each step, in order.

    Raises:
        ValueError: if ``reducer`` is not one of :data:`SERIES_REDUCERS`.

    Examples:
        - Reduce a 3-step collection to one spatial-mean value per step:
            ```python
            >>> from pyramids.dataset.collection import DatasetCollection
            >>> from digitalearth.base.temporal import series_values
            >>> dc = DatasetCollection.from_files(["examples/data/acc4000.tif"] * 3)
            >>> series_values(dc, reducer="mean").shape
            (3,)

            ```
        - An unknown reducer is refused by name:
            ```python
            >>> from pyramids.dataset.collection import DatasetCollection
            >>> from digitalearth.base.temporal import series_values
            >>> dc = DatasetCollection.from_files(["examples/data/acc4000.tif"])
            >>> series_values(dc, reducer="median")
            Traceback (most recent call last):
                ...
            ValueError: unknown reducer 'median'; choose from ['max', 'mean', 'min', 'sum']

            ```
    """
    if reducer not in SERIES_REDUCERS:
        raise ValueError(
            f"unknown reducer {reducer!r}; choose from {sorted(SERIES_REDUCERS)}"
        )
    func = SERIES_REDUCERS[reducer]
    return np.asarray(
        [func(read_masked_band(member, band)) for member in collection.datasets]
    )


def group_by_label(
    values: np.ndarray, labels: Sequence
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Group a 1-D series by a periodic label into per-group mean / min / max (a climatology).

    Args:
        values: The 1-D series — one value per time step, e.g. from :func:`series_values`.
        labels: One group label per step (same length and order as ``values``), e.g. month-of-year.

    Returns:
        ``(groups, mean, low, high)``: the sorted unique labels, and the per-group ``nanmean`` / ``nanmin`` /
        ``nanmax`` of the values whose label matches, aligned with ``groups``.

    Raises:
        ValueError: if ``labels`` and ``values`` differ in length.

    Examples:
        - Group a 6-step series by a 3-season cycle into three group means:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.temporal import group_by_label
            >>> values = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
            >>> groups, mean, low, high = group_by_label(values, ["djf", "jja", "son"] * 2)
            >>> groups
            ['djf', 'jja', 'son']
            >>> mean.tolist()
            [3.0, 4.0, 5.0]
            >>> low.tolist(), high.tolist()
            ([1.0, 2.0, 3.0], [5.0, 6.0, 7.0])

            ```
    """
    values = np.asarray(values)
    if len(labels) != len(values):
        raise ValueError(
            f"labels has {len(labels)} entries but the series has {len(values)} values"
        )
    label_arr = np.asarray(labels)
    groups = sorted(set(label_arr.tolist()))
    mean = np.array([np.nanmean(values[label_arr == group]) for group in groups])
    low = np.array([np.nanmin(values[label_arr == group]) for group in groups])
    high = np.array([np.nanmax(values[label_arr == group]) for group in groups])
    return groups, mean, low, high
