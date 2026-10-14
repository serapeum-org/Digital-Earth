"""Unit conversion for display — convert values between common geophysical units.

earthkit-plots converts a field's values at plot time from the units its metadata names (kelvin shown as °C,
pascals as hPa); digitalearth styled and labelled from units but never converted the numbers. This is the
conversion core: pure, engine-neutral arithmetic over NumPy arrays, with no renderer and no GIS import, so it
sits in ``base`` and any tier (or a caller) can convert before drawing and label with the target units.

It is **display-time** conversion — it transforms values a caller already holds; it does not read or rewrite a
dataset on disk (that would be a ``pyramids`` concern). Only affine conversions within one physical *family*
are defined (temperature, pressure, precipitation/length, speed, fraction); a cross-family request (°C → Pa)
is refused rather than guessed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["can_convert", "convert", "convert_array", "normalize_units"]

#: Spelling variants folded to one canonical key before lookup. Symbols and CF-style unit strings reach the
#: same entry as the word.
_ALIASES = {
    "k": "kelvin",
    "degk": "kelvin",
    "°k": "kelvin",
    "degc": "celsius",
    "°c": "celsius",
    "degree_celsius": "celsius",
    "degrees_celsius": "celsius",
    "degf": "fahrenheit",
    "°f": "fahrenheit",
    "degree_fahrenheit": "fahrenheit",
    "hpa": "hpa",
    "mbar": "hpa",
    "millibar": "hpa",
    "pa": "pa",
    "pascal": "pa",
    "kpa": "kpa",
    "bar": "bar",
    "m": "m",
    "metre": "m",
    "meter": "m",
    "mm": "mm",
    "millimetre": "mm",
    "millimeter": "mm",
    "cm": "cm",
    "m/s": "m/s",
    "m s-1": "m/s",
    "ms-1": "m/s",
    "km/h": "km/h",
    "kph": "km/h",
    "knot": "knot",
    "knots": "knot",
    "kt": "knot",
    "mph": "mph",
    "fraction": "fraction",
    "1": "fraction",
    "percent": "percent",
    "%": "percent",
}

#: Canonical unit → (family, scale, offset), where ``value_in_base = value * scale + offset`` and the base of
#: each family is the unit whose scale is ``1`` and offset ``0`` (kelvin, pascal, metre, m/s, fraction).
_UNITS = {
    # temperature, base kelvin
    "kelvin": ("temperature", 1.0, 0.0),
    "celsius": ("temperature", 1.0, 273.15),
    "fahrenheit": ("temperature", 5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
    # pressure, base pascal
    "pa": ("pressure", 1.0, 0.0),
    "hpa": ("pressure", 100.0, 0.0),
    "kpa": ("pressure", 1000.0, 0.0),
    "bar": ("pressure", 100000.0, 0.0),
    # length / precipitation, base metre
    "m": ("length", 1.0, 0.0),
    "cm": ("length", 0.01, 0.0),
    "mm": ("length", 0.001, 0.0),
    # speed, base metre/second
    "m/s": ("speed", 1.0, 0.0),
    "km/h": ("speed", 1.0 / 3.6, 0.0),
    "knot": ("speed", 0.514444, 0.0),
    "mph": ("speed", 0.44704, 0.0),
    # dimensionless fraction, base fraction
    "fraction": ("fraction", 1.0, 0.0),
    "percent": ("fraction", 0.01, 0.0),
}


def normalize_units(units: str) -> str:
    """Fold a unit string to its canonical key (lower-cased, aliases resolved).

    Args:
        units: A unit string as a file or a caller spells it (``"K"``, ``"deg C"``, ``"m s-1"``).

    Returns:
        The canonical key used by the registry, or the lower-cased input when nothing matches (so an
        unknown unit is reported with the spelling the caller used, not a mangled one).

    Examples:
        ```python
        >>> from digitalearth.base.units import normalize_units
        >>> normalize_units("K"), normalize_units("deg C"), normalize_units("m s-1")
        ('kelvin', 'celsius', 'm/s')

        ```
    """
    collapsed = " ".join(units.strip().lower().split())
    if collapsed in _ALIASES:
        return _ALIASES[collapsed]
    despaced = collapsed.replace("deg ", "deg").replace(" ", "")
    return _ALIASES.get(despaced, collapsed)


def _entry(units: str) -> tuple[str, float, float]:
    """Resolve a unit string to its ``(family, scale, offset)``, raising on an unknown one."""
    canonical = normalize_units(units)
    try:
        return _UNITS[canonical]
    except KeyError:
        raise ValueError(
            f"unknown unit {units!r}; convertible units are {sorted(_UNITS)}"
        ) from None


def can_convert(source: str, target: str) -> bool:
    """Return whether values in ``source`` units can be converted to ``target`` units.

    True only when both units are known *and* share a physical family; a cross-family pair (°C → Pa) or an
    unknown unit is ``False`` rather than an error, so a caller can test before committing to a conversion.

    Args:
        source: The units the values are in.
        target: The units to convert to.

    Returns:
        Whether :func:`convert` / :func:`convert_array` would succeed for this pair.

    Examples:
        ```python
        >>> from digitalearth.base.units import can_convert
        >>> can_convert("K", "celsius"), can_convert("K", "hPa"), can_convert("K", "widgets")
        (True, False, False)

        ```
    """
    try:
        return _entry(source)[0] == _entry(target)[0]
    except ValueError:
        return False


def convert(value: float, source: str, target: str) -> float:
    """Convert one scalar value from ``source`` units to ``target`` units.

    Args:
        value: The value to convert.
        source: The units ``value`` is in.
        target: The units to convert to.

    Returns:
        ``value`` expressed in ``target`` units.

    Raises:
        ValueError: If either unit is unknown, or the two are in different physical families.

    Examples:
        ```python
        >>> from digitalearth.base.units import convert
        >>> round(convert(300.0, "K", "celsius"), 2)
        26.85
        >>> convert(1000.0, "hPa", "Pa")
        100000.0

        ```
    """
    src_family, src_scale, src_offset = _entry(source)
    dst_family, dst_scale, dst_offset = _entry(target)
    if src_family != dst_family:
        raise ValueError(
            f"cannot convert {source!r} to {target!r}: {src_family} and {dst_family} are different "
            "physical quantities"
        )
    in_base = value * src_scale + src_offset
    return (in_base - dst_offset) / dst_scale


def convert_array(
    values: np.ndarray, source: str, target: str
) -> tuple[np.ndarray, str]:
    """Convert an array of values from ``source`` to ``target`` units, preserving non-finite cells.

    The array counterpart of :func:`convert`, for converting a whole field before it is coloured. ``NaN``
    (nodata, already masked) stays ``NaN`` — the affine transform would otherwise turn it into a number.
    Same-unit conversion returns the values unchanged (a float copy).

    Args:
        values: The field values, nodata already masked to ``NaN``.
        source: The units ``values`` are in.
        target: The units to convert to.

    Returns:
        A ``(converted, target_units)`` pair — the converted float array and the canonical target unit
        string, ready to pass to a colorbar label.

    Raises:
        ValueError: If either unit is unknown, or the two are in different physical families.

    Examples:
        ```python
        >>> import numpy as np
        >>> from digitalearth.base.units import convert_array
        >>> out, units = convert_array(np.array([273.15, np.nan, 373.15]), "K", "celsius")
        >>> units, round(float(out[0]), 2), bool(np.isnan(out[1])), round(float(out[2]), 2)
        ('celsius', 0.0, True, 100.0)

        ```
    """
    src_family, src_scale, src_offset = _entry(source)
    dst_family, dst_scale, dst_offset = _entry(target)
    if src_family != dst_family:
        raise ValueError(
            f"cannot convert {source!r} to {target!r}: {src_family} and {dst_family} are different "
            "physical quantities"
        )
    array = np.asarray(values, dtype=float)
    if normalize_units(source) == normalize_units(target):
        # Identical units: return a plain float copy rather than the affine round-trip, which for an offset
        # family (celsius/fahrenheit) would drift — x + 273.15 - 273.15 != x in float64.
        return array.copy(), normalize_units(target)
    in_base = array * src_scale + src_offset
    converted = (in_base - dst_offset) / dst_scale
    return converted, normalize_units(target)
