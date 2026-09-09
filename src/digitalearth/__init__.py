from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

# documentation format
__author__ = "Mostafa Farrag"
__email__ = "moah.farag@gmail.com"
__docformat__ = "restructuredtext"

# Let users know if they're missing any of our hard dependencies
hard_dependencies: tuple[str, ...] = ()  # ("numpy", "pandas", "gdal")
missing_dependencies: list[str] = []

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
from digitalearth.base.sources import DimensionInfo, Source, get_source  # noqa: E402
from digitalearth.ops.batch import Batch  # noqa: E402
from digitalearth.ops.browser import gallery  # noqa: E402
from digitalearth.ops.plugins import load_plugins  # noqa: E402
from digitalearth.static import (  # noqa: E402
    Map,
    Scene,
    TexturedGlobe,
    grid,
    projections,
    shared_colorbar,
)
from digitalearth.static.charts import (  # noqa: E402
    bar,
    bar_by,
    histogram,
    line,
    line_by,
    scatter,
    statistics,
)
from digitalearth.static.geostatistics import (  # noqa: E402
    hotspot_map,
    kriging_map,
    lisa_map,
)
from digitalearth.static.series import (  # noqa: E402
    boxplot,
    envelope,
    multiboxplot,
    quantile_band,
    stripes,
)
from digitalearth.static.temporal import Climatology, TimeSeries  # noqa: E402

__all__ = [
    # one-call API + composition
    "quickplot",
    "quickmap",
    "Map",
    "Scene",
    "TexturedGlobe",
    "grid",
    "shared_colorbar",
    "projections",
    # data view
    "get_source",
    "Source",
    "DimensionInfo",
    # charts and statistical series
    "line",
    "bar",
    "bar_by",
    "line_by",
    "histogram",
    "scatter",
    "statistics",
    "envelope",
    "quantile_band",
    "boxplot",
    "multiboxplot",
    "stripes",
    # temporal products
    "TimeSeries",
    "Climatology",
    # geostatistics visualization (composes geostatista outputs)
    "lisa_map",
    "hotspot_map",
    "kriging_map",
    # operational tier
    "Batch",
    "gallery",
    "load_plugins",
]

# --- back-compat: submodules that moved in the backend restructure -------------------------------------------
#
# Before the restructure these twelve names were bound as attributes of `digitalearth` -- some because
# `__init__` imported from them, the rest as a side effect of those imports -- so `digitalearth.charts` and
# `from digitalearth import series` both worked. Moving them under base/, static/ and ops/ silently unbound
# every one: attribute access raised a bare AttributeError and the `from` form an ImportError, neither
# mentioning where the module went.
#
# PEP 562 module __getattr__ restores them as deprecated aliases that say where to go. It covers the attribute
# forms only; `import digitalearth.charts` and `from digitalearth.charts import histogram` still fail, because
# those need a real module on disk -- `digitalearth.scene` is the one such shim we ship.
_MOVED_SUBMODULES = {
    "animation": "digitalearth.static.animation",
    "autostyle": "digitalearth.base.autostyle",
    "batch": "digitalearth.ops.batch",
    "browser": "digitalearth.ops.browser",
    "charts": "digitalearth.static.charts",
    "cli": "digitalearth.ops.cli",
    "geostatistics": "digitalearth.static.geostatistics",
    "plugins": "digitalearth.ops.plugins",
    "scene": "digitalearth.scene",
    "series": "digitalearth.static.series",
    "sources": "digitalearth.base.sources",
    "temporal": "digitalearth.static.temporal",
}


def __getattr__(name: str):
    """Resolve a submodule that moved in the backend restructure, with a :class:`DeprecationWarning`.

    Args:
        name: The attribute being looked up on the ``digitalearth`` package.

    Returns:
        The module at its new location.

    Raises:
        AttributeError: for any name that is neither a real attribute nor a moved submodule.
    """
    target = _MOVED_SUBMODULES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    import warnings

    warnings.warn(
        f"digitalearth.{name} moved to {target} in the backend restructure; import it from there. "
        "This alias will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    module = importlib.import_module(target)
    globals()[name] = module  # cache, so the warning fires once per process
    return module


def __dir__() -> list:
    """Include the moved-submodule aliases so tab-completion and ``dir()`` still find them."""
    return sorted(set(globals()) | set(_MOVED_SUBMODULES))
