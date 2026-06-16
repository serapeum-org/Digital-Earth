try:
    from importlib.metadata import PackageNotFoundError  # type: ignore
    from importlib.metadata import version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError  # type: ignore
    from importlib_metadata import version


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

# documentation format
__author__ = "Mostafa Farrag"
__email__ = "moah.farag@gmail.com"
__docformat__ = "restructuredtext"

# Let users know if they're missing any of our hard dependencies
hard_dependencies = ()  # ("numpy", "pandas", "gdal")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)
        print(e)

if missing_dependencies:
    raise ImportError("Missing required dependencies {0}".format(missing_dependencies))


__doc__ = """
digitalearth - visualization package
"""

from digitalearth.api import quickmap, quickplot  # noqa: E402
from digitalearth.batch import Batch  # noqa: E402
from digitalearth.browser import gallery  # noqa: E402
from digitalearth.charts import bar, histogram, line, scatter, statistics  # noqa: E402
from digitalearth.plugins import load_plugins  # noqa: E402
from digitalearth.scene import Map, Scene, grid, projections, shared_colorbar  # noqa: E402
from digitalearth.series import (  # noqa: E402
    boxplot,
    envelope,
    multiboxplot,
    quantile_band,
    stripes,
)
from digitalearth.sources import DimensionInfo, Source, get_source  # noqa: E402
from digitalearth.temporal import Climatology, TimeSeries  # noqa: E402

__all__ = [
    # one-call API + composition
    "quickplot", "quickmap", "Map", "Scene", "grid", "shared_colorbar", "projections",
    # data view
    "get_source", "Source", "DimensionInfo",
    # charts and statistical series
    "line", "bar", "histogram", "scatter", "statistics",
    "envelope", "quantile_band", "boxplot", "multiboxplot", "stripes",
    # temporal products
    "TimeSeries", "Climatology",
    # operational tier
    "Batch", "gallery", "load_plugins",
]
