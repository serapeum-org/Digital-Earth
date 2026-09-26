# `importlib.metadata` is stdlib from 3.8 on and `requires-python` floors at 3.11, so the
# `importlib_metadata` backport this used to fall back to can never be reached: read the version straight
# from the installed distribution's metadata.
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    # Running from a source tree that was never installed (e.g. pytest's `pythonpath = ["src"]`): there is
    # no distribution to read a version off, so report "unknown" rather than failing the import.
    __version__ = "unknown"

__author__ = "Mostafa Farrag"
__email__ = "moah.farag@gmail.com"
#: Docstring dialect the API reference is rendered from.
__docformat__ = "restructuredtext"

#: Modules whose absence should be reported as one collected ImportError rather than as whichever
#: `from ... import` happened to run first. Currently empty (the commented-out names show the intended
#: shape), so the loop below is a no-op: every dependency is declared in `pyproject.toml` and installed
#: with the package. The annotation is what tells mypy the element type — an empty literal gives it
#: nothing to infer from, and this is a module-level name other code may read.
hard_dependencies: tuple[str, ...] = ()  # ("numpy", "pandas", "gdal")
#: The subset of `hard_dependencies` that failed to import, collected by the loop below.
missing_dependencies: list[str] = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)
        print(e)

if missing_dependencies:
    raise ImportError("Missing required dependencies {0}".format(missing_dependencies))


# Assigned rather than written as a module docstring at the top of the file: the imports above have to run
# first (the version lookup and the dependency check), and a docstring is only a docstring in first position.
__doc__ = """digitalearth — geospatial visualization built on pyramids and cleopatra.

Reads data through pyramids and renders it through one subpackage per backend: `static` (matplotlib, the
default), `interactive` (HoloViz/Bokeh), `three_d` (PyVista) and `web` (MapLibre + deck.gl), over the
engine-neutral `base`. `quickmap`/`quickplot` are the one-call entry points; `Map` is the composable scene.
"""

from digitalearth.base.registry import register_classifier as _register_classifier


def _cleopatra_classify(values, scheme, k):
    """Cut class edges with cleopatra's classifier.

    The implementation behind `base/`'s classifier seam. It lives here rather than in `base/` because `base/`
    may not import a renderer, and cleopatra is one; the import is inside the call so importing this package
    does not pull it in.

    Args:
        values: The data to classify.
        scheme: Scheme name.
        k: Number of classes.

    Returns:
        Whatever ``cleopatra.styling.styles.classify`` returns — an ``(edges, _)`` pair.
    """
    from cleopatra.styling.styles import classify

    return classify(values, scheme, k)


_register_classifier(_cleopatra_classify)

# --- the public names, resolved on first use ----------------------------------------------------------------
#
# These were imported eagerly, which made `import digitalearth` — and so importing *anything* under it, including
# the engine-neutral `digitalearth.base.spec` — pull in matplotlib.pyplot and cleopatra through `static`. That made
# the design's test for a figure description unrunnable: `FigureSpec.from_dict(fig.to_dict())` has to round-trip
# with no renderer imported (#283), and it could not, because the package root had imported one before the spec
# module was reached.
#
# Each name is imported from its home the first time it is touched and cached in the module globals, so
# `from digitalearth import Map`, `digitalearth.Map`, `from digitalearth import *` and `dir()` all behave as before;
# only the moment the import happens moves. The classifier registration above stays eager: it imports no renderer.
#: Each home module, with the public names imported from it. Grouped by module so each path is written once;
#: :data:`_LAZY_EXPORTS` is the name -> module index `__getattr__` looks names up in.
_LAZY_MODULES = {
    "digitalearth.api": ("quickmap", "quickplot"),
    "digitalearth.base.sources": ("DimensionInfo", "Source", "get_source"),
    "digitalearth.ops.batch": ("Batch",),
    "digitalearth.ops.browser": ("gallery",),
    "digitalearth.ops.plugins": ("load_plugins",),
    "digitalearth.static": (
        "Map",
        "Scene",
        "TexturedGlobe",
        "grid",
        "projections",
        "shared_colorbar",
    ),
    "digitalearth.static.charts": (
        "bar",
        "bar_by",
        "histogram",
        "line",
        "line_by",
        "scatter",
        "statistics",
    ),
    "digitalearth.static.series": (
        "boxplot",
        "envelope",
        "multiboxplot",
        "quantile_band",
        "stripes",
    ),
    "digitalearth.static.temporal": ("Climatology", "TimeSeries"),
}
#: Public name -> the module it is imported from: :data:`_LAZY_MODULES` turned inside out.
_LAZY_EXPORTS = {
    name: module for module, names in _LAZY_MODULES.items() for name in names
}

#: Subpackages the eager imports used to leave bound on the package, so `import digitalearth` followed by
#: `digitalearth.static.Map` kept working. The optional backends were never bound this way and still are not:
#: resolving `digitalearth.three_d` on attribute access would turn `hasattr` into a `ModuleNotFoundError` without
#: the `3d` extra, because `hasattr` only swallows `AttributeError`.
_LAZY_SUBPACKAGES = ("api", "base", "ops", "static")

# A type checker does not run `__getattr__` — it reads imports. Without these, mypy and IDEs saw every public name
# as the `Any` an attribute hook returns: no completion, no go-to-definition, and `quickmap(backend=5)` passed. They
# are the same names `_LAZY_MODULES` resolves at runtime, where `TYPE_CHECKING` is False and none of this runs, so
# importing the package still imports no renderer. A test holds the two lists to one another.
if TYPE_CHECKING:
    # The redundant aliases mark these as re-exports: they are not in `__all__`, which star-import reads.
    from digitalearth import api as api
    from digitalearth import base as base
    from digitalearth import ops as ops
    from digitalearth import static as static
    from digitalearth.api import quickmap, quickplot
    from digitalearth.base.sources import DimensionInfo, Source, get_source
    from digitalearth.ops.batch import Batch
    from digitalearth.ops.browser import gallery
    from digitalearth.ops.plugins import load_plugins
    from digitalearth.static import (
        Map,
        Scene,
        TexturedGlobe,
        grid,
        projections,
        shared_colorbar,
    )
    from digitalearth.static.charts import (
        bar,
        bar_by,
        histogram,
        line,
        line_by,
        scatter,
        statistics,
    )
    from digitalearth.static.series import (
        boxplot,
        envelope,
        multiboxplot,
        quantile_band,
        stripes,
    )
    from digitalearth.static.temporal import Climatology, TimeSeries

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
    # operational tier
    "Batch",
    "gallery",
    "load_plugins",
]

# --- removed: the geostatistics presets, which move upstream --------------------------------------------
#
# `lisa_map`/`hotspot_map`/`kriging_map` and `digitalearth.static.geostatistics` drew another package's
# output and nothing else, so the drawing moves to the package that produces the labels
# (serapeum-org/geostatista#61). They are removed outright rather than deprecated: a forwarding shim would
# keep the wrong dependency direction documented as supported. But a bare "module 'digitalearth' has no
# attribute 'lisa_map'" sends a reader looking for a typo, so the removed names still fail with a message
# naming the replacement. The standard prefix is kept, so `hasattr` is still False and code matching on the
# usual wording still matches.
_REMOVED_GEOSTATISTICS = ("geostatistics", "hotspot_map", "kriging_map", "lisa_map")


def __getattr__(name: str) -> Any:
    """Resolve an attribute the package does not bind yet.

    That is a public name, a subpackage, or a removed geostatistics name. A public name from `__all__` is
    imported from its home module on first use and cached, as is one of the subpackages the package used to
    bind eagerly — so importing the package imports no renderer until something asks for one. The removed
    names are tested next and always raise: the ``geostatistics`` submodule and the
    ``lisa_map``/``hotspot_map``/``kriging_map`` presets went upstream to geostatista, so there is nothing here
    to forward them to and the error says where they went.

    Args:
        name: The attribute being looked up on the ``digitalearth`` package.

    Returns:
        The public object or subpackage. The removed names never reach this path.

    Raises:
        AttributeError: for one of the removed geostatistics names, with a message naming its replacement; and
            for any other name that is neither a real attribute nor a lazily resolved name or subpackage.
    """
    if name in _LAZY_EXPORTS:
        import importlib

        value = getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
        globals()[name] = value  # cache, so the import runs once per process
        return value
    if name in _LAZY_SUBPACKAGES:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _REMOVED_GEOSTATISTICS:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}: the geostatistics presets were removed and "
            "move to geostatista, which produces the labels they colour (see "
            "serapeum-org/geostatista#61). Until it ships them, draw the result directly with "
            "digitalearth.Map().choropleth(..., scheme='categorical') and your own colour mapping."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    """List the package's attributes, including the ones it has not resolved yet.

    Returns:
        The bound module globals and the lazily resolved public names and subpackages, as one sorted list — so
        tab-completion and `dir()` find a name before its first use imports it.
    """
    return sorted(set(globals()) | set(_LAZY_EXPORTS) | set(_LAZY_SUBPACKAGES))
