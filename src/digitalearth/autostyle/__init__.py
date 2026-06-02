"""autostyle — map a Source's metadata (variable/units) to cleopatra Style parameters.

The *mechanics* of styling (colormaps, norms, levels) live in cleopatra; the *domain mapping* — "a variable
called ``t2m`` should use a temperature colormap" — is a geospatial concern and lives here, driven by a
per-variable YAML library under ``autostyle/library/``.

Two layers cooperate. The richer :func:`~digitalearth.autostyle.magics.magics_style` (RP.10) ports ECMWF
Magics' operational identity matching — name → CF ``standard_name`` → units — and resolves canonical
colormap *and* contour levels for common meteorological fields; :func:`auto_style` consults it first and
falls back to the lighter substring library (``variables.yml``) for everything else.
"""
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

from digitalearth.autostyle.magics import load_magics_library, magics_style
from digitalearth.sources.source import Source

__all__ = ["auto_style", "load_library", "load_magics_library", "magics_style"]

_LIBRARY_DIR = Path(__file__).parent / "library"


@lru_cache(maxsize=1)
def load_library() -> Dict[str, dict]:
    """Load and merge every ``*.yml`` file in the style library (cached).

    Returns:
        Mapping of style-group name (e.g. ``"temperature"``, ``"default"``) to its parameter dict.

    Examples:
        - The shipped library defines a default and several variable groups:
            ```python
            >>> from digitalearth.autostyle import load_library
            >>> lib = load_library()
            >>> lib["default"]["cmap"]
            'viridis'
            >>> "temperature" in lib
            True

            ```
    """
    library: Dict[str, dict] = {}
    for path in sorted(_LIBRARY_DIR.glob("*.yml")):
        if path.name == "magics.yml":
            continue  # the Magics operational library is loaded separately via load_magics_library()
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        library.update(loaded)
    return library


def auto_style(source: Source) -> Dict[str, Any]:
    """Resolve cleopatra style parameters for a :class:`~digitalearth.sources.source.Source`.

    The richer ECMWF Magics identity matching (:func:`~digitalearth.autostyle.magics.magics_style`) is tried
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
            >>> from digitalearth.sources import Source, DimensionInfo
            >>> from digitalearth.autostyle import auto_style
            >>> src = Source(DimensionInfo(np.zeros((2, 2)), "z"), DimensionInfo(np.array([0.0]), "x"),
            ...              DimensionInfo(np.array([0.0]), "y"), metadata={"variable": "t2m"})
            >>> auto_style(src)["cmap"]
            'coolwarm'

            ```
        - An unrecognised variable falls back to the default colormap:
            ```python
            >>> import numpy as np
            >>> from digitalearth.sources import Source, DimensionInfo
            >>> from digitalearth.autostyle import auto_style
            >>> src = Source(DimensionInfo(np.zeros((2, 2)), "z"), DimensionInfo(np.array([0.0]), "x"),
            ...              DimensionInfo(np.array([0.0]), "y"), metadata={"variable": "mystery"})
            >>> auto_style(src)["cmap"]
            'viridis'

            ```
        - An operational field also picks up canonical Magics contour levels and units:
            ```python
            >>> import numpy as np
            >>> from digitalearth.sources import Source, DimensionInfo
            >>> from digitalearth.autostyle import auto_style
            >>> src = Source(DimensionInfo(np.zeros((2, 2)), "z"), DimensionInfo(np.array([0.0]), "x"),
            ...              DimensionInfo(np.array([0.0]), "y"), metadata={"variable": "msl"})
            >>> style = auto_style(src)
            >>> style["units"], style["levels"][0]
            ('hPa', 960)

            ```
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
