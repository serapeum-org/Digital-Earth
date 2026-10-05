"""autostyle — map a Source's metadata (variable/units) to cleopatra Style parameters.

The *mechanics* of styling (colormaps, norms, levels) live in cleopatra; the *domain mapping* — "a variable
called ``t2m`` should use a temperature colormap" — is a geospatial concern and lives here, driven by a
per-variable YAML library under ``autostyle/library/``.

Two layers cooperate. The richer :func:`~digitalearth.base.autostyle.magics.magics_style` (RP.10) ports ECMWF
Magics' operational identity matching — name → CF ``standard_name`` → units — and resolves canonical
colormap *and* contour levels for common meteorological fields; :func:`auto_style` consults it first and
falls back to the lighter substring library (``variables.yml``) for everything else.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

from digitalearth.base.autostyle.magics import load_magics_library, magics_style
from digitalearth.base.sources.source import Source

__all__ = [
    "auto_style",
    "load_library",
    "load_magics_library",
    "magics_style",
    "register_style_library",
    "temporary_style_library",
]

_LIBRARY_DIR = Path(__file__).parent / "library"

#: Per-variable style groups contributed by installed ``digitalearth.styles`` plugins, merged into the library
#: after the bundled YAML so a plugin can add a group or override one. Populated by
#: :func:`register_style_library`, which the package's plugin wiring
#: (:func:`digitalearth.load_installed_plugins`) calls for each loaded ``styles`` plugin; empty until then.
_PLUGIN_LIBRARY: Dict[str, dict] = {}


@lru_cache(maxsize=1)
def _bundled_library() -> Dict[str, dict]:
    """Load and merge every bundled ``*.yml`` file in the style library (cached).

    The bundled files never change during a run, so this is cached; the plugin-contributed groups are merged
    on top of it in :func:`load_library`, which is why they are not cached here.

    Returns:
        Mapping of style-group name (e.g. ``"temperature"``, ``"default"``) to its parameter dict.
    """
    library: Dict[str, dict] = {}
    for path in sorted(_LIBRARY_DIR.glob("*.yml")):
        if path.name == "magics.yml":
            continue  # the Magics operational library is loaded separately via load_magics_library()
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        library.update(loaded)
    return library


def load_library() -> Dict[str, dict]:
    """Return the bundled style groups merged with any ``digitalearth.styles`` plugin has contributed.

    The bundled ``*.yml`` groups are cached; the plugin-contributed groups (:data:`_PLUGIN_LIBRARY`, filled by
    :func:`register_style_library`) are merged on top, last-wins, so a plugin can add a group or override a
    bundled one — the same policy the bundled files already merge under.

    A **fresh top-level mapping** is returned each call, so a caller may add, drop or replace whole groups
    without disturbing the lru-cached bundled groups or another caller's result. The group dicts *inside* it
    are **not** copied — they are the cached bundled objects (and a plugin's own objects) shared by reference —
    so a returned group must be treated as read-only and never mutated in place. The sole in-tree caller,
    :func:`auto_style`, only reads, so this is safe; keeping the copy shallow also lets a ``digitalearth.styles``
    plugin contribute a live, not-deep-copyable style value (a matplotlib ``Colormap``, a ``ColorScale``, a
    callable) — an earlier unconditional deep copy of the merged library raised on such a value, disabling all
    styling the moment any such plugin was installed.

    Returns:
        Mapping of style-group name (e.g. ``"temperature"``, ``"default"``) to its parameter dict.

    Examples:
        - The shipped library defines a default and several variable groups:
            ```python
            >>> from digitalearth.base.autostyle import load_library
            >>> lib = load_library()
            >>> lib["default"]["cmap"]
            'viridis'
            >>> "temperature" in lib
            True

            ```

    See Also:
        register_style_library: How a ``digitalearth.styles`` plugin adds groups to what this returns.
        load_magics_library: The richer ECMWF Magics operational library (loaded separately).
    """
    return {**_bundled_library(), **_PLUGIN_LIBRARY}


def register_style_library(library: Mapping[str, dict]) -> None:
    """Merge a ``digitalearth.styles`` plugin's per-variable style groups into the autostyle library.

    This is the ``styles`` half of the plugin contract (RP.11): a ``digitalearth.styles`` entry point loads to
    a mapping of style-group name to a parameter dict — the same shape as a bundled ``variables.yml`` group
    (an optional ``match`` list plus cleopatra style parameters such as ``cmap``/``levels``) — and the wiring
    hands it here. The groups join what :func:`load_library` returns, so :func:`auto_style` resolves against
    them beside the bundled ones. A group name a plugin reuses overwrites, last-wins, exactly as the bundled
    YAML files already merge.

    Args:
        library: The plugin's style groups, name -> parameter dict.

    Raises:
        TypeError: if `library` is not a mapping. A ``styles`` plugin that loads to anything else does not
            match the contract, and the wiring skips it rather than corrupting the library.

    Examples:
        - A plugin's group extends the library. A plain call is **permanent** — the group stays for the life of
          the process — so this shows it inside :func:`temporary_style_library`, the scoped form to reach for
          anywhere the extension should not outlive the code that made it:
            ```python
            >>> from digitalearth.base.autostyle import load_library, temporary_style_library
            >>> group = {"ocean_heat": {"match": ["ohc"], "cmap": "inferno"}}
            >>> with temporary_style_library(group):
            ...     load_library()["ocean_heat"]["cmap"]
            'inferno'
            >>> "ocean_heat" in load_library()
            False

            ```
    """
    if not isinstance(library, Mapping):
        raise TypeError(
            f"a digitalearth.styles plugin must load to a mapping of style groups; got "
            f"{type(library).__name__}"
        )
    _PLUGIN_LIBRARY.update(library)


@contextmanager
def temporary_style_library(library: Mapping[str, dict]) -> Iterator[None]:
    """Merge plugin style groups for the duration of a block, then restore the library as it was.

    The plugin style table is process-global and :func:`register_style_library` has no un-register, so a test
    or an example that adds a group changes what every later :func:`auto_style` sees. This is the scoped form,
    so demonstrating the registry does not leave a group behind in the session that ran the demonstration.

    Args:
        library: The style groups to merge for the block.

    Yields:
        Nothing; the groups are in effect inside the block.

    Examples:
        - The group resolves inside the block and is gone afterwards:
            ```python
            >>> from digitalearth.base.autostyle import load_library, temporary_style_library
            >>> with temporary_style_library({"scratch_var": {"cmap": "magma"}}):
            ...     "scratch_var" in load_library()
            True
            >>> "scratch_var" in load_library()
            False

            ```
    """
    previous = dict(_PLUGIN_LIBRARY)
    register_style_library(library)
    try:
        yield
    finally:
        _PLUGIN_LIBRARY.clear()
        _PLUGIN_LIBRARY.update(previous)


def auto_style(source: Source) -> Dict[str, Any]:
    """Resolve cleopatra style parameters for a :class:`~digitalearth.base.sources.source.Source`.

    The richer ECMWF Magics identity matching (:func:`~digitalearth.base.autostyle.magics.magics_style`) is tried
    first — on the source's variable name, then CF ``standard_name``, then ``units`` — and when it recognises
    the field it contributes a canonical colormap *and* contour ``levels``. Otherwise the source's variable
    name is matched (case-insensitive substring) against the lighter ``variables.yml`` groups, the first
    match overriding the ``default`` group. The returned dict is suitable to pass as ``ArrayGlyph`` options
    (e.g. ``cmap``, ``levels``), plus an optional ``units`` hint.

    **Units fall back to the source's own** when neither library named any (ST-18, through
    :func:`_with_source_units`). The library's answer is canonical and still wins where it has one — mean
    sea-level pressure is styled in hPa whatever the file says — but it has an opinion about a few hundred
    variables and none about the rest, and a band carrying its own units was previously styled with no
    units at all, leaving every reader of this dict (a colorbar's label, an animation's label, a swatch
    legend's heading) nothing to say about a quantity the data had named itself.

    Args:
        source: The data source whose ``metadata("variable")`` / ``units`` drive the lookup.

    Returns:
        A style-parameter dict (always includes ``cmap``); ``match`` keys are stripped. ``units`` is
        present when the library named one *or* the source carries one — and absent altogether when
        neither does, so ``"units" in style`` stays the question "does anything know what these values are
        measured in".

    Examples:
        - A temperature-like variable selects the temperature colormap:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import Source, DimensionInfo
            >>> from digitalearth.base.autostyle import auto_style
            >>> src = Source(DimensionInfo(np.zeros((2, 2)), "z"), DimensionInfo(np.array([0.0]), "x"),
            ...              DimensionInfo(np.array([0.0]), "y"), metadata={"variable": "t2m"})
            >>> auto_style(src)["cmap"]
            'coolwarm'

            ```
        - An unrecognised variable falls back to the default colormap:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import Source, DimensionInfo
            >>> from digitalearth.base.autostyle import auto_style
            >>> src = Source(DimensionInfo(np.zeros((2, 2)), "z"), DimensionInfo(np.array([0.0]), "x"),
            ...              DimensionInfo(np.array([0.0]), "y"), metadata={"variable": "mystery"})
            >>> auto_style(src)["cmap"]
            'viridis'

            ```
        - An operational field also picks up canonical Magics contour levels and units:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import Source, DimensionInfo
            >>> from digitalearth.base.autostyle import auto_style
            >>> src = Source(DimensionInfo(np.zeros((2, 2)), "z"), DimensionInfo(np.array([0.0]), "x"),
            ...              DimensionInfo(np.array([0.0]), "y"), metadata={"variable": "msl"})
            >>> style = auto_style(src)
            >>> style["units"], style["levels"][0]
            ('hPa', 960)

            ```
        - A variable the libraries do not recognise keeps the units its source declared, and one that
          declares none leaves the key out:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import Source, DimensionInfo
            >>> from digitalearth.base.autostyle import auto_style
            >>> def source_of(units):
            ...     return Source(
            ...         DimensionInfo(np.zeros((2, 2)), "z"),
            ...         DimensionInfo(np.array([0.0]), "x"),
            ...         DimensionInfo(np.array([0.0]), "y"),
            ...         metadata={"variable": "widget_flux"},
            ...         units=units,
            ...     )
            >>> auto_style(source_of("widgets/s"))["units"]
            'widgets/s'
            >>> "units" in auto_style(source_of(None))
            False

            ```

    See Also:
        magics_style: The ECMWF Magics identity matcher consulted first.
        load_library: The lighter ``variables.yml`` fallback library.
    """
    library = load_library()
    variable_raw = str(source.metadata("variable") or "")
    style: Dict[str, Any] = dict(library.get("default", {}))

    # First, the richer ECMWF Magics identity matching (name -> standard_name -> units): when it recognises
    # the field, it supplies a canonical colormap *and* contour levels and we are done.
    magics = magics_style(variable_raw, source.metadata("standard_name"), source.units)
    if magics is not None:
        style.update(magics)
        return _with_source_units(style, source)

    # Otherwise fall back to the lighter variables.yml substring library (case-insensitive, first match wins).
    variable = variable_raw.lower()
    for name, params in library.items():
        if name == "default":
            continue
        patterns = params.get("match", [])
        if isinstance(patterns, str):
            patterns = [patterns]
        if any(p.lower() in variable for p in patterns):
            style.update({k: v for k, v in params.items() if k != "match"})
            break
    return _with_source_units(style, source)


def _with_source_units(style: Dict[str, Any], source: Source) -> Dict[str, Any]:
    """Fill a resolved style's ``units`` from the source itself when the library named none (ST-18).

    The library's answer is **canonical** and wins where it has one: a field matched as mean sea-level
    pressure is styled in hPa, with contour levels to match, and a file saying ``Pa`` is the very case that
    canonical answer exists for. But the library has an opinion about a few hundred variables and none at
    all about the rest, and "no opinion" is not the same as "no units" — a GeoTIFF band carrying
    ``widgets/s`` was styled with the default colormap and *no* units, so every reader of this dict (a
    colorbar's label, an animation's label, a swatch legend's heading) had nothing to say about a quantity
    the data had named itself.

    Args:
        style: The style resolved from the library, modified in place and returned.
        source: The source the style was resolved for.

    Returns:
        The same dict. It gains a ``units`` key only when the library left one out *and* the source carries
        one: a source naming no units adds no key at all, so ``"units" in style`` stays the question "does
        anything know what these values are measured in".

    Examples:
        - A variable the library does not recognise keeps the units its source declared:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import DimensionInfo, Source
            >>> from digitalearth.base.autostyle import auto_style
            >>> def src(variable, units):
            ...     return Source(
            ...         DimensionInfo(np.zeros((2, 2)), "z"),
            ...         DimensionInfo(np.array([0.0, 1.0]), "x"),
            ...         DimensionInfo(np.array([0.0, 1.0]), "y"),
            ...         metadata={"variable": variable},
            ...         units=units,
            ...     )
            >>> auto_style(src("widget_flux", "widgets/s"))["units"]
            'widgets/s'

            ```
        - A recognised variable keeps the library's canonical units instead, and an unlabelled source
          leaves the key out:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import DimensionInfo, Source
            >>> from digitalearth.base.autostyle import auto_style
            >>> def src(variable, units):
            ...     return Source(
            ...         DimensionInfo(np.zeros((2, 2)), "z"),
            ...         DimensionInfo(np.array([0.0, 1.0]), "x"),
            ...         DimensionInfo(np.array([0.0, 1.0]), "y"),
            ...         metadata={"variable": variable},
            ...         units=units,
            ...     )
            >>> auto_style(src("msl", "Pa"))["units"]
            'hPa'
            >>> "units" in auto_style(src("widget_flux", None))
            False

            ```
        - A source declaring an empty string names no units either, so no key is added:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import DimensionInfo, Source
            >>> from digitalearth.base.autostyle import auto_style
            >>> blank = Source(
            ...     DimensionInfo(np.zeros((2, 2)), "z"),
            ...     DimensionInfo(np.array([0.0, 1.0]), "x"),
            ...     DimensionInfo(np.array([0.0, 1.0]), "y"),
            ...     metadata={"variable": "widget_flux"},
            ...     units="",
            ... )
            >>> "units" in auto_style(blank)
            False

            ```
    """
    if style.get("units") is None and source.units:
        style["units"] = source.units
    return style
