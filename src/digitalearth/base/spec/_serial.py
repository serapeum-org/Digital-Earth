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
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np

__all__ = [
    "as_list",
    "as_mapping",
    "crs_to_json",
    "finite_number",
    "refuse_unknown",
    "require",
    "to_json_value",
]


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

    Examples:
        - A mapping that uses only known keys passes, and nothing is returned:
            ```python
            >>> from digitalearth.base.spec._serial import refuse_unknown
            >>> print(refuse_unknown("Camera", {"position": [0, 1, 0]}, ("position", "view_up")))
            None

            ```
        - An unknown key is named beside the keys that exist:
            ```python
            >>> from digitalearth.base.spec._serial import refuse_unknown
            >>> refuse_unknown("Camera", {"position": [0, 1, 0], "roll": 3}, ("view_up", "position"))
            Traceback (most recent call last):
                ...
            ValueError: Camera.from_dict got unknown keys ['roll']; known keys are ['position', 'view_up']

            ```
        - Something that is not a mapping at all is a `TypeError`:
            ```python
            >>> from digitalearth.base.spec._serial import refuse_unknown
            >>> refuse_unknown("Camera", [0, 1, 0], ("position",))
            Traceback (most recent call last):
                ...
            TypeError: Camera.from_dict needs a mapping; got list

            ```
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

    Examples:
        - A present key returns its value:
            ```python
            >>> from digitalearth.base.spec._serial import require
            >>> require("Viewport", {"crs": 4326, "globe": True}, "crs")
            4326

            ```
        - A missing key is named, with the keys that were there:
            ```python
            >>> from digitalearth.base.spec._serial import require
            >>> require("Viewport", {"globe": True}, "crs")
            Traceback (most recent call last):
                ...
            ValueError: Viewport.from_dict needs 'crs'; got keys ['globe']

            ```
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
        The value as a Python float. A Python or numpy integer is accepted and converted.

    Raises:
        ValueError: for a bool (which is an int in Python, and would read `True` as `1.0`), a non-number — a
            numeric string included — `nan` or an infinity.

    Examples:
        - An integer, Python or numpy, comes back as a float:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.spec._serial import finite_number
            >>> finite_number("Camera", "view_angle", 45), finite_number("Camera", "view_angle", np.int64(45))
            (45.0, 45.0)

            ```
        - A boolean is refused rather than read as `1.0`:
            ```python
            >>> from digitalearth.base.spec._serial import finite_number
            >>> finite_number("Camera", "view_angle", True)
            Traceback (most recent call last):
                ...
            ValueError: Camera needs view_angle as a number; got True

            ```
        - A non-finite value is refused, naming the field:
            ```python
            >>> from digitalearth.base.spec._serial import finite_number
            >>> finite_number("Camera", "distance", float("inf"))
            Traceback (most recent call last):
                ...
            ValueError: Camera needs a finite distance; got inf

            ```
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
        The value with tuples and arrays as lists (nested arrays as nested lists), mappings as dicts and numpy
        scalars as the Python `bool`, `int` or `float` they hold. JSON has no tuple, so a tuple comes back from a
        round trip as a list: equality survives a round trip for values that are already JSON-native, which is
        what a figure description should hold.

    Raises:
        TypeError: for anything else — a dataset, a `datetime`, a set, a mapping with a non-string key — naming
            `where`, down to the item inside a list or dict. A set is refused rather than listed because it has
            no order to write down. `nan` and the infinities are refused too: JSON has no spelling for them. The
            `NaN`/`Infinity` tokens Python's `json.dumps` writes by default are not JSON — `json.dumps` itself
            refuses them with `allow_nan=False`, and JavaScript's `JSON.parse`, which is what reads an exported
            page, raises a `SyntaxError` on them.

    Examples:
        - Tuples, arrays and numpy scalars come back as the JSON-native types:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.spec._serial import to_json_value
            >>> to_json_value({"levels": (1, 2), "grid": np.array([[1, 2], [3, 4]]), "k": np.int64(5)}, "props")
            {'levels': [1, 2], 'grid': [[1, 2], [3, 4]], 'k': 5}

            ```
        - A live object is refused, naming where it sits:
            ```python
            >>> from datetime import datetime
            >>> from digitalearth.base.spec._serial import to_json_value
            >>> to_json_value({"when": [datetime(2024, 1, 1)]}, "Symbology.props")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            TypeError: Symbology.props['when'][0] holds a datetime, which has no JSON form. A figure description ...

            ```
        - `nan` is refused rather than written as a token strict readers reject:
            ```python
            >>> from digitalearth.base.spec._serial import to_json_value
            >>> to_json_value([1.0, float("nan")], "Scale.breaks")
            Traceback (most recent call last):
                ...
            TypeError: Scale.breaks[1] is nan, which has no JSON form; strict JSON readers refuse NaN and Infinity

            ```
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
        `value` unchanged — only a float is inspected, so a boolean, an int or a string always passes.

    Raises:
        TypeError: for a float that is `nan`, `inf` or `-inf`.
    """
    if isinstance(value, float) and not isfinite(value):
        raise TypeError(
            f"{where} is {value!r}, which has no JSON form; strict JSON readers refuse NaN and Infinity"
        )
    return value


def as_list(owner: str, key: str, value: Any) -> Tuple[Any, ...]:
    """Return a stored list as a tuple, refusing a value of another shape by the key it was stored under.

    Args:
        owner: The type reading, for the message — ``"Camera"``.
        key: The key the value was stored under.
        value: The stored value.

    Returns:
        The items as a tuple. A tuple is accepted as well as a list, so a dict built in Python reads too.

    Raises:
        TypeError: for anything else, naming the key and what was found. `tuple(5)` would raise
            ``'int' object is not iterable``, which names neither the type being read nor the field.

    Examples:
        - A stored list reads back as a tuple; a scalar names the key:
            ```python
            >>> from digitalearth.base.spec._serial import as_list
            >>> as_list("Camera", "position", [0, -10, 5])
            (0, -10, 5)
            >>> as_list("Camera", "position", 5)
            Traceback (most recent call last):
                ...
            TypeError: Camera.from_dict needs 'position' as a list; got int 5

            ```
    """
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise TypeError(
        f"{owner}.from_dict needs {key!r} as a list; got {type(value).__name__} {value!r}"
    )


def as_mapping(owner: str, key: str, value: Any) -> Dict[str, Any]:
    """Return a stored mapping as a dict, refusing a value of another shape by the key it was stored under.

    Args:
        owner: The type reading, for the message.
        key: The key the value was stored under.
        value: The stored value.

    Returns:
        A fresh dict of the mapping.

    Raises:
        TypeError: for anything that is not a mapping, naming the key and what was found.

    Examples:
        - A list where a mapping belongs names the key:
            ```python
            >>> from digitalearth.base.spec._serial import as_mapping
            >>> as_mapping("FigureSpec", "sources", ["dem.tif"])
            Traceback (most recent call last):
                ...
            TypeError: FigureSpec.from_dict needs 'sources' as a mapping; got list ['dem.tif']

            ```
    """
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        f"{owner}.from_dict needs {key!r} as a mapping; got {type(value).__name__} {value!r}"
    )


def crs_to_json(crs: Any, where: str) -> Any:
    """Return a CRS in a spelling that survives a JSON round trip and still names the same system.

    Args:
        crs: The CRS as the spec holds it — ``None``, an EPSG integer, a string, or a CRS object.
        where: The field, for the message.

    Returns:
        `None` or a string unchanged — a string is not checked, so one pyramids cannot read is written as
        given — and an integer (a numpy integer included) as a Python `int`. A CRS object becomes
        `"EPSG:<code>"` when pyramids reads an authority code out of it, and its WKT otherwise. Either reads
        back through the same pyramids parser that read the object, so the system is kept even though the
        Python type is not.

    Raises:
        TypeError: for a boolean — `True` is an `int` and would be written as EPSG code 1 — or for any other
            object pyramids cannot read as a CRS, a float among them.

    Examples:
        - An EPSG integer and a string pass through:
            ```python
            >>> from digitalearth.base.spec._serial import crs_to_json
            >>> crs_to_json(4326, "Viewport.crs"), crs_to_json("+proj=ortho +lat_0=30", "Viewport.crs")
            (4326, '+proj=ortho +lat_0=30')

            ```
        - A CRS object is written by its authority code, or as WKT when it has none:
            ```python
            >>> from pyramids.base.crs import crs_from_user_input
            >>> from digitalearth.base.spec._serial import crs_to_json
            >>> crs_to_json(crs_from_user_input(3857), "Bounds.crs")
            'EPSG:3857'
            >>> crs_to_json(crs_from_user_input("+proj=ortho +lat_0=30 +lon_0=10"), "Bounds.crs")[:7]
            'PROJCRS'

            ```
        - A boolean names no reference system:
            ```python
            >>> from digitalearth.base.spec._serial import crs_to_json
            >>> crs_to_json(True, "Bounds.crs")
            Traceback (most recent call last):
                ...
            TypeError: Bounds.crs is True, which names no coordinate reference system

            ```
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
