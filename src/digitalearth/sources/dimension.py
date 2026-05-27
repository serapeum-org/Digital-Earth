"""DimensionInfo — a labelled coordinate/value axis used by :class:`~digitalearth.sources.source.Source`."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DimensionInfo:
    """A named array of values with optional units.

    Wraps one axis (``x``/``y``) or the data array (``z``) of a
    :class:`~digitalearth.sources.source.Source` together with a human label and
    optional unit string, mirroring earthkit-plots' ``DimensionInfo``.

    Attributes:
        values: The underlying numpy array (1-D coordinate vector or 2-D data grid).
        name: Short label for the dimension, e.g. ``"x"``, ``"y"`` or ``"z"``.
        units: Unit string for the values, or ``None`` when unknown.

    Examples:
        - Wrap a coordinate vector and read its label/units:
            ```python
            >>> import numpy as np
            >>> from digitalearth.sources.dimension import DimensionInfo
            >>> dim = DimensionInfo(np.array([0.0, 1.0, 2.0]), "x", "m")
            >>> dim.name
            'x'
            >>> dim.units
            'm'
            >>> dim.values.tolist()
            [0.0, 1.0, 2.0]

            ```
        - Units default to None when omitted:
            ```python
            >>> import numpy as np
            >>> from digitalearth.sources.dimension import DimensionInfo
            >>> dim = DimensionInfo(np.array([1, 2]), "z")
            >>> dim.units is None
            True

            ```
    """

    values: np.ndarray
    name: str = ""
    units: Optional[str] = None
