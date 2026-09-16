"""The rules every spec type's `to_dict`/`from_dict` pair shares.

A figure description is only as serialisable as its least serialisable part, so the checks live here once rather
than being re-spelled — slightly differently — on each type. Three rules:

* **An unknown key is refused, not dropped.** A newer writer's field silently lost on read is data loss that
  surfaces far from its cause. `DataRef.from_dict` already worked this way; the other types follow it.
* **A value with no JSON form is refused where it is written**, naming the field. A spec that holds a live
  object — a dataset, an artist, a `datetime` — cannot cross the seam the design says nothing live may cross, and
  finding that out inside `json.dumps` names neither the type nor the field.
* **A CRS is written in a spelling that reads back.** An EPSG integer or a string passes through; a CRS *object*
  is written as ``"EPSG:<code>"``, or as WKT when it carries no authority code.
"""

from math import isfinite
from typing import Any, Dict, Iterable, Mapping

import numpy as np

__all__ = ["crs_to_json", "finite_number", "refuse_unknown", "require", "to_json_value"]


def refuse_unknown(owner: str, data: Any, known: Iterable[str]) -> None:
    """Refuse a mapping that is not one, or that carries a key the reading type does not know.

    Args:
        owner: The type doing the reading, for the message — ``"LayerSpec"``.
        data: What `from_dict` was given.
        known: Every key the type writes.

    Raises:
        TypeError: if `data` is not a mapping.
        ValueError: if it has a key outside `known`. The message lists both sets, sorted, so the caller can see
            whether it is a typo or a field from a newer version.
    """
    if not isinstance(data, Mapping):
        raise TypeError(f"{owner}.from_dict needs a mapping; got {type(data).__name__}")
    allowed = set(known)
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(
            f"{owner}.from_dict got unknown keys {unknown}; known keys are {sorted(allowed)}"
        )


def require(owner: str, data: Mapping[str, Any], key: str) -> Any:
    """Return a key `from_dict` cannot do without.

    Args:
        owner: The type doing the reading, for the message.
        data: The mapping being read.
        key: The required key.

    Returns:
        The value stored under `key`.

    Raises:
        ValueError: if the key is absent, naming it rather than surfacing as a bare `KeyError`.
    """
    if key not in data:
        raise ValueError(f"{owner}.from_dict needs {key!r}; got keys {sorted(data)}")
    return data[key]


def finite_number(owner: str, name: str, value: Any) -> float:
    """Return `value` as a finite float, refusing a boolean, a non-number or a non-finite one.

    Args:
        owner: The type being built, for the message.
        name: The field being checked.
        value: The candidate.

    Returns:
        The value as a Python float.

    Raises:
        ValueError: for a bool (which is an int in Python, and would read `True` as `1.0`), a non-number, `nan`
            or an infinity.
    """
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{owner} needs {name} as a number; got {value!r}")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{owner} needs a finite {name}; got {value!r}")
    return number


def to_json_value(value: Any, where: str) -> Any:
    """Return `value` in the form `json.dumps` writes and `json.loads` reads back.

    Args:
        value: A field's value.
        where: The field, for the message — ``"Symbology.props['levels']"``.

    Returns:
        The value with tuples and arrays as lists, mappings as dicts and numpy scalars as Python numbers. JSON has
        no tuple, so a tuple comes back from a round trip as a list: equality survives a round trip for values
        that are already JSON-native, which is what a figure description should hold.

    Raises:
        TypeError: for anything else — a dataset, a `datetime`, a set, a mapping with a non-string key — naming
            `where`. A set is refused rather than listed because it has no order to write down. `nan` and the
            infinities are refused too: JSON has no spelling for them, and the ``NaN``/``Infinity`` tokens Python's
            `json` writes by default are rejected by strict readers — ``json.dumps(allow_nan=False)``, and
            JavaScript's ``JSON.parse``, which is what reads an exported page.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        return _finite(value, where)
    if isinstance(value, np.generic):
        native = value.item()
        if isinstance(native, (bool, int, float, str)):
            return _finite(native, where)
    if isinstance(value, np.ndarray):
        return [
            to_json_value(item, f"{where}[{index}]")
            for index, item in enumerate(value.tolist())
        ]
    if isinstance(value, (list, tuple)):
        return [
            to_json_value(item, f"{where}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{where} has the non-string key {key!r}; a JSON object's keys are strings"
                )
            out[key] = to_json_value(item, f"{where}[{key!r}]")
        return out
    raise TypeError(
        f"{where} holds a {type(value).__name__}, which has no JSON form. A figure description holds plain "
        "values — numbers, strings, lists and dicts — not live objects"
    )


def _finite(value: Any, where: str) -> Any:
    """Return a number unchanged, refusing one JSON cannot spell.

    Args:
        value: A number, or a boolean or string passed through from a numpy scalar.
        where: The field, for the message.

    Returns:
        `value` unchanged.

    Raises:
        TypeError: for `nan`, `inf` or `-inf`.
    """
    if isinstance(value, float) and not isfinite(value):
        raise TypeError(
            f"{where} is {value!r}, which has no JSON form; strict JSON readers refuse NaN and Infinity"
        )
    return value


def crs_to_json(crs: Any, where: str) -> Any:
    """Return a CRS in a spelling that survives a JSON round trip and still names the same system.

    Args:
        crs: The CRS as the spec holds it — ``None``, an EPSG integer, a string, or a CRS object.
        where: The field, for the message.

    Returns:
        ``None``, an integer or a string unchanged. A CRS object becomes ``"EPSG:<code>"`` when pyramids reads an
        authority code out of it, and its WKT otherwise. Either reads back through the same pyramids parser that
        read the object, so the system is kept even though the Python type is not.

    Raises:
        TypeError: for a boolean — `True` is an `int` and would be written as EPSG code 1 — or for an object
            pyramids cannot read as a CRS.
    """
    if crs is None or isinstance(crs, str):
        return crs
    if isinstance(crs, bool):
        raise TypeError(
            f"{where} is {crs!r}, which names no coordinate reference system"
        )
    if isinstance(crs, (int, np.integer)):
        return int(crs)
    # Imported here: resolving a CRS object is pyramids' job, and a spec holding only EPSG ints or strings should
    # not pay for importing it.
    from digitalearth.base.crs import authority_code

    code = authority_code(crs)
    if code is not None:
        return f"EPSG:{code}"
    to_wkt = getattr(crs, "to_wkt", None)
    if callable(to_wkt):
        return str(to_wkt())
    raise TypeError(
        f"{where} holds a {type(crs).__name__} that is not a readable CRS; store an EPSG integer or a CRS string"
    )
