"""magics — ECMWF Magics-style identity matching: a field's identity -> a canonical operational style.

ECMWF's **Magics** is the operational plotting engine behind ECMWF weather charts: each meteorological field
(2 m temperature, mean-sea-level pressure, total precipitation, …) has a *canonical* look — a fixed
colormap, contour interval and unit — so that the same field is always drawn the same way. earthkit-plots
ports that idea as ``styles/magics.py`` plus an automatic identity matcher. This module ports the
**mechanism**: resolve a field's identity (name, then CF ``standard_name``, then units) to a canonical
style, backed by an **extensible** library (``library/magics.yml``) that ships a *representative subset* of
common fields — not the full ECMWF Magics catalogue.

The split mirrors the rest of the package. Applying a colormap / contour levels is generic matplotlib and
lives in cleopatra; the *domain knowledge* — "``msl`` is mean-sea-level pressure, drawn every 4 hPa between
960 and 1052 hPa" — is geospatial and lives here, as data in the YAML library rather than as code.
"""
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

__all__ = ["magics_style", "load_magics_library"]

_MAGICS_LIBRARY = Path(__file__).parent / "library" / "magics.yml"

# Keys that exist only to *match* a field's identity; they are stripped from the returned style so the
# caller receives just the renderable parameters (cmap, levels, units, magics_name).
_MATCH_KEYS = ("match", "standard_name", "match_units")


def _as_list(value: Any) -> List[Any]:
    """Coerce ``None`` / a scalar / a sequence into a list (so callers can iterate uniformly)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _style_of(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a field entry with the match-only keys removed — i.e. just the renderable style."""
    return {k: v for k, v in params.items() if k not in _MATCH_KEYS}


@lru_cache(maxsize=1)
def load_magics_library() -> Dict[str, dict]:
    """Load the Magics operational style library from ``library/magics.yml`` (cached).

    Returns:
        Mapping of field-group name (e.g. ``"temperature_2m"``, ``"mean_sea_level_pressure"``) to its entry,
        where each entry carries both the match keys and the canonical style.

    Examples:
        - The shipped library covers common operational fields:
            ```python
            >>> from digitalearth.autostyle.magics import load_magics_library
            >>> lib = load_magics_library()
            >>> "mean_sea_level_pressure" in lib
            True
            >>> lib["temperature_2m"]["magics_name"]
            't2m'

            ```
    """
    return yaml.safe_load(_MAGICS_LIBRARY.read_text(encoding="utf-8")) or {}


def magics_style(
    name: Optional[str] = None,
    standard_name: Optional[str] = None,
    units: Optional[str] = None,
    *,
    library: Optional[Dict[str, dict]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve a canonical Magics-style for a field, by identity matching.

    Matching is tried in the order Magics itself prefers — **name first**, then CF ``standard_name``, then
    ``units`` as a deliberately narrow last resort — and the first hit wins:

    1. ``name`` — case-insensitive *substring* against each entry's ``match`` aliases (so ``"2t_daily_mean"``
       matches the ``"2t"`` alias).
    2. ``standard_name`` — exact (case-insensitive) against each entry's ``standard_name`` list.
    3. ``units`` — exact (case-insensitive) against each entry's ``match_units`` list (only distinctive units
       are listed in the library, to avoid e.g. plain ``"m"`` colliding with elevation).

    Args:
        name: The field's variable/short name (e.g. ``"t2m"``, ``"msl"``).
        standard_name: The field's CF ``standard_name`` (e.g. ``"air_temperature"``), if known.
        units: The field's units (e.g. ``"hPa"``), if known.
        library: Override library (defaults to the shipped one); mainly for testing.

    Returns:
        The canonical style dict (``cmap``, ``levels``, ``units``, ``magics_name``) with the match-only keys
        stripped, or ``None`` when nothing matches.

    Examples:
        - Match by name and read the canonical contour interval:
            ```python
            >>> from digitalearth.autostyle.magics import magics_style
            >>> style = magics_style("t2m")
            >>> style["cmap"], style["magics_name"]
            ('coolwarm', 't2m')
            >>> magics_style("2t_daily_mean")["magics_name"]
            't2m'

            ```
        - Match by CF standard_name when the short name is unknown:
            ```python
            >>> from digitalearth.autostyle.magics import magics_style
            >>> magics_style(standard_name="air_pressure_at_mean_sea_level")["units"]
            'hPa'

            ```
        - An unrecognised field returns ``None`` (the caller falls back to a default style):
            ```python
            >>> from digitalearth.autostyle.magics import magics_style
            >>> magics_style("mystery_field") is None
            True

            ```
    """
    lib = library if library is not None else load_magics_library()
    name_l = str(name or "").lower()
    sname_l = str(standard_name or "").lower()
    units_l = str(units or "").lower()

    # 1) by name — case-insensitive substring against each entry's aliases (the primary Magics key).
    if name_l:
        for params in lib.values():
            if any(str(pat).lower() in name_l for pat in _as_list(params.get("match"))):
                return _style_of(params)
    # 2) by CF standard_name — exact, case-insensitive.
    if sname_l:
        for params in lib.values():
            if sname_l in [str(s).lower() for s in _as_list(params.get("standard_name"))]:
                return _style_of(params)
    # 3) by units — exact, case-insensitive; narrow last-resort fallback.
    if units_l:
        for params in lib.values():
            if units_l in [str(u).lower() for u in _as_list(params.get("match_units"))]:
                return _style_of(params)
    return None
