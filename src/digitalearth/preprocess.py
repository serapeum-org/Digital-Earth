"""preprocess — longitude/cyclic helpers for global fields.

Two small operations from earthkit-plots' pipeline that prepare global gridded data for plotting:

- :func:`wrap_longitude` rolls a 0-360 dataset to -180..180 (delegated to **pyramids**
  ``Dataset.wrap_longitude`` — the CRS-aware operation belongs there, not here).
- :func:`add_cyclic_column` appends the first data column so a global contour/pcolormesh has no seam at the
  antimeridian. This is pure array bookkeeping (numpy), so it lives in the digitalearth wiring.
"""
from typing import Any, Tuple

import numpy as np


def wrap_longitude(dataset: Any) -> Any:
    """Roll a 0-360 longitude dataset to -180..180 via pyramids ``wrap_longitude``.

    Args:
        dataset: A pyramids ``Dataset`` whose x/longitude runs 0..360.

    Returns:
        A new ``Dataset`` with longitudes in -180..180.
    """
    return dataset.wrap_longitude()


def add_cyclic_column(z: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Append the first column of a global field to close the antimeridian seam.

    Args:
        z: 2-D data grid ``(rows, columns)``.
        x: 1-D longitude coordinates of length ``columns``.

    Returns:
        ``(z2, x2)`` where ``z2`` has one extra column (a copy of column 0) and ``x2`` has one extra
        longitude one grid step past the last (so a global contour wraps cleanly).

    Examples:
        - Closing a 2x3 global field adds one wrap-around column:
            ```python
            >>> import numpy as np
            >>> from digitalearth.preprocess import add_cyclic_column
            >>> z = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> x = np.array([0.0, 120.0, 240.0])
            >>> z2, x2 = add_cyclic_column(z, x)
            >>> z2.shape
            (2, 4)
            >>> x2.tolist()
            [0.0, 120.0, 240.0, 360.0]
            >>> z2[:, -1].tolist()
            [1.0, 4.0]

            ```
    """
    z2 = np.concatenate([z, z[:, :1]], axis=1)
    step = (x[1] - x[0]) if len(x) > 1 else 360.0
    x2 = np.concatenate([x, [x[-1] + step]])
    return z2, x2
