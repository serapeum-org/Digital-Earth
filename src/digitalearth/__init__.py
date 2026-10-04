# `importlib.metadata` is stdlib from 3.8 on and `requires-python` floors at 3.11, so the
# `importlib_metadata` backport this used to fall back to can never be reached: read the version straight
# from the installed distribution's metadata.
import logging
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Callable

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

from digitalearth.base.registry import (
    register_classifier as _register_classifier,
)
from digitalearth.base.registry import (
    resolvers as _resolvers,
)


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

# --- installed plugins, registered at import (RP.11) --------------------------------------------------------
#
# `digitalearth.ops.plugins.load_plugins` discovers and imports the entry points in the `digitalearth.styles`
# and `digitalearth.sources` groups, but nothing registered what it returned — a shipped source adapter or
# style library was loaded and then dropped. This is the wiring that connects the two: each loaded object is
# registered into the matching `base` registry, per a small per-group contract.
#
# * `digitalearth.sources`: the entry point loads to a `(scheme, resolver)` pair, registered through
#   `base.registry.register_resolver`, so `resolve_uri("<scheme>:...")` reaches the plugin's reader.
# * `digitalearth.styles`: the entry point loads to a per-variable style mapping (the `variables.yml` shape),
#   merged into the autostyle library through `base.autostyle.register_style_library`.
#
# It runs at **import**, next to the classifier registration above and for the same reason: the package is set
# up once, deterministically, the moment it is imported, so a plugin is active without a caller remembering to
# turn it on. The alternative — loading lazily on first registry use — would force `base` to reach up into
# `ops.plugins`, an upward-layer import `base` does not otherwise make, for no saving in the ordinary case:
# the package ships no plugins of its own, so with none installed this costs two empty `entry_points()` lookups
# and imports no renderer (a `styles` plugin only pulls in `base.autostyle` when one is actually installed).
_plugin_logger = logging.getLogger(__name__)


def _register_source_plugin(name: str, loaded: Any) -> None:
    """Register one ``digitalearth.sources`` plugin: a ``(scheme, resolver)`` pair.

    Warns (M3) when the plugin's scheme is one of the built-in resolvers the package registers itself
    (:data:`_BUILTIN_RESOLVER_SCHEMES`, the ``file``/``object`` readers). The override still takes effect —
    ``register_resolver`` is last-wins, which :func:`~digitalearth.base.registry.temporary_resolver` relies on —
    but it no longer happens silently, so an installed dependency re-pointing the ``file:`` reader for every
    consumer is at least visible. Shadowing *another plugin's* scheme stays quiet: that is ordinary last-wins.

    Args:
        name: The entry-point name of the plugin, for the shadow warning.
        loaded: What the entry point loaded to — a ``(scheme, resolver)`` pair. A malformed object (not such a
            pair) raises here and is skipped by :func:`register_plugins`.
    """
    scheme, resolver = loaded
    from digitalearth.base.registry import register_resolver

    if scheme in _BUILTIN_RESOLVER_SCHEMES:
        _plugin_logger.warning(
            "digitalearth.sources plugin %r registers scheme %r, shadowing the built-in resolver of the same "
            "name; the plugin wins (last-wins), replacing the built-in %r reader for this process",
            name,
            scheme,
            scheme,
        )
    register_resolver(scheme, resolver)


def _register_style_plugin(name: str, loaded: Any) -> None:
    """Register one ``digitalearth.styles`` plugin: a per-variable style mapping.

    Warns (M3) when the plugin redefines a bundled style group (one already shipped in
    ``base.autostyle``'s library, e.g. ``default``). As with the source resolvers the merge still happens —
    :func:`~digitalearth.base.autostyle.load_library` is last-wins — but a plugin silently re-colouring the
    fallback palette for every consumer is made visible. Adding a *new* group, or shadowing another plugin's,
    stays quiet.

    Args:
        name: The entry-point name of the plugin, for the shadow warning.
        loaded: What the entry point loaded to — a mapping of style-group name to a parameter dict. A
            non-mapping raises in :func:`digitalearth.base.autostyle.register_style_library` and is skipped by
            :func:`register_plugins`.
    """
    from digitalearth.base.autostyle import _bundled_library, register_style_library

    if isinstance(loaded, Mapping):
        shadowed = sorted(set(loaded) & set(_bundled_library()))
        if shadowed:
            _plugin_logger.warning(
                "digitalearth.styles plugin %r redefines bundled style group(s) %s, shadowing the built-in "
                "group(s) of the same name; the plugin wins (last-wins) for this process",
                name,
                shadowed,
            )
    register_style_library(loaded)


#: How each entry-point group's loaded object is registered. Keyed by the same public group strings
#: :data:`digitalearth.ops.plugins.GROUPS` publishes; a test holds the two sets equal so a new group cannot be
#: discovered and then silently dropped for want of a registrar here.
_PLUGIN_REGISTRARS: dict[str, Callable[[str, Any], None]] = {
    "digitalearth.sources": _register_source_plugin,
    "digitalearth.styles": _register_style_plugin,
}


def register_plugins(group: str, loaded: dict[str, Any]) -> None:
    """Register into its registry every plugin object ``load_plugins`` returned for one group.

    The seam :func:`load_installed_plugins` drives, exposed on its own so a test can hand it a fake set (built
    with ``load_plugins(group, eps=[...])``) without installing a real package. A plugin whose object is
    malformed for its group is logged and skipped, so one bad plugin cannot stop the rest — the same tolerance
    :func:`digitalearth.ops.plugins.load_plugins` already gives a plugin whose import raises. A group with no
    registrar is a no-op. A plugin that overrides a built-in name (scheme or style group) is registered but
    warns first (M3), whereas overriding another plugin's name is ordinary last-wins and stays quiet.

    Args:
        group: The entry-point group the objects came from (one of
            :data:`digitalearth.ops.plugins.GROUPS`).
        loaded: The ``{name: loaded object}`` mapping :func:`digitalearth.ops.plugins.load_plugins` returned.
    """
    registrar = _PLUGIN_REGISTRARS.get(group)
    if registrar is None:
        return
    for name, obj in loaded.items():
        try:
            registrar(name, obj)
        except (TypeError, ValueError) as exc:
            # Only the shapes a plugin malformed *for its group* produces (a bad object failing the contract
            # check — a non-(scheme, resolver) pair, a non-mapping style library) are tolerated, so one bad
            # plugin cannot abort the healthy ones (L4). An unexpected error from a registrar is a genuine
            # wiring bug, not a bad plugin, and is left to propagate rather than hidden at WARNING during import.
            # This narrowness is deliberate, and the opposite of load_installed_plugins' broad discovery catch:
            # here the exception shapes are our own contract check's and thus known; there they are a corrupt
            # environment's and open-ended. See the comment at that catch (L5) for the full contrast.
            _plugin_logger.warning(
                "skipping plugin %r in group %r: %s", name, group, exc
            )


def load_installed_plugins() -> None:
    """Discover, load and register every installed ``digitalearth.*`` plugin, for both groups.

    Called once at package import. Idempotent — re-running it re-registers the same objects, which the
    registries absorb as ordinary duplicates (last-wins), so it is safe to call again.

    Discovery is itself best-effort (M1). ``load_plugins`` drives ``iter_plugins``, whose first step is
    ``importlib.metadata.entry_points()`` — and a corrupt or partial environment (malformed distribution
    metadata, a broken ``RECORD``) can make *that* raise, before any single plugin is even reached. Because
    this runs at ``import digitalearth``, an unguarded failure there would abort the import of a package the
    whole stack depends on. So enumeration of each group is wrapped: a failure logs a WARNING and leaves the
    package importable with that group's plugins simply absent — the same best-effort stance ``load_plugins``
    already takes for a single plugin whose ``load()`` raises.

    See Also:
        register_plugins: Registers the objects for one group; this calls it for each of
            :data:`digitalearth.ops.plugins.GROUPS`.
    """
    from digitalearth.ops.plugins import GROUPS, load_plugins

    for group in GROUPS:
        # This catch is broad on purpose (L5), and the deliberate opposite of register_plugins' narrow
        # (TypeError, ValueError) catch, because the two guards protect different things:
        #   * Here the guarded call is third-party-metadata enumeration. entry_points() reads arbitrary
        #     installed distributions, whose corruption modes (malformed METADATA, a broken RECORD, a bad
        #     encoding) are open-ended and raised by importlib/stdlib internals -- there is no narrow exception
        #     tuple that means "the environment is corrupt". A corrupt env is not a bug in our code, and
        #     narrowing would let a real enumeration failure abort `import digitalearth`, the exact thing this
        #     guard exists to prevent. The accepted tradeoff -- a coding bug in the tiny iter_plugins/
        #     load_plugins loop would also be logged here rather than raised -- is the right call at the import
        #     boundary, where keeping the package importable outweighs surfacing a bug in two doctest-covered
        #     functions.
        #   * register_plugins narrows to (TypeError, ValueError) because those are exactly the shapes its own
        #     contract check raises for a malformed plugin object; anything else there is our wiring bug and
        #     must surface (L4). Discovery cannot make that assumption, so it stays broad.
        # The explicit public API takes the opposite stance again: load_plugins/iter_plugins propagate an
        # enumeration failure to a direct caller (N2). This import-time path is the sole place it is tolerated.
        try:
            loaded = load_plugins(group)
        except Exception as exc:  # corrupt/partial env: keep the package importable without this group's plugins
            _plugin_logger.warning(
                "could not enumerate %r plugins (entry-point discovery failed); continuing without them: %s",
                group,
                exc,
            )
            continue
        register_plugins(group, loaded)


#: Resolver schemes the package registers itself — the built-in ``file``/``object`` readers — captured before
#: any plugin loads (``base.registry`` registers them at its own import, which line 49 above has already
#: triggered). A ``digitalearth.sources`` plugin that reuses one of these shadows a built-in, which
#: :func:`_register_source_plugin` warns about (M3); a plugin reusing another plugin's scheme does not, because
#: that is ordinary last-wins. Snapshotting once keeps "built-in" fixed even though ``load_installed_plugins``
#: may run again.
_BUILTIN_RESOLVER_SCHEMES: frozenset[str] = frozenset(_resolvers())

load_installed_plugins()

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
    "digitalearth.api": ("quickmap", "quickplot", "to_backend"),
    "digitalearth.base.sources": ("DimensionInfo", "Source", "get_source"),
    "digitalearth.ops.batch": ("Batch",),
    "digitalearth.ops.browser": ("gallery",),
    "digitalearth.ops.plugins": ("load_plugins",),
    "digitalearth.static": (
        "Map",
        "Scene",
        "TexturedGlobe",
        "facet",
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
    from digitalearth.api import quickmap, quickplot, to_backend
    from digitalearth.base.sources import DimensionInfo, Source, get_source
    from digitalearth.ops.batch import Batch
    from digitalearth.ops.browser import gallery
    from digitalearth.ops.plugins import load_plugins
    from digitalearth.static import (
        Map,
        Scene,
        TexturedGlobe,
        facet,
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
    "to_backend",
    "Map",
    "Scene",
    "TexturedGlobe",
    "facet",
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
