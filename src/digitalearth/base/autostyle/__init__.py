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
    bundled one — the same policy the bundled files already merge under. A fresh dict is returned each call, so
    a caller cannot edit the cached bundled groups by accident.

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

    Args:
        source: The data source whose ``metadata("variable")`` / ``units`` drive the lookup.

    Returns:
        A style-parameter dict (always includes ``cmap``); ``match`` keys are stripped.

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
        return style

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
    return style
