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
    """

    values: np.ndarray
    name: str = ""
    units: Optional[str] = None
