"""The rules every spec type's `to_dict`/`from_dict` pair shares.

A figure description is only as serialisable as its least serialisable part, so the checks live here once rather
than being re-spelled — slightly differently — on each type. Three rules:

* **An unknown key is refused, not dropped.** A newer writer's field silently lost on read is data loss that
  surfaces far from its cause. `DataRef.from_dict` already worked this way; the other types follow it.
* **A value with no JSON form is refused where it is written**, naming the field. A spec that holds a live
  object — a dataset, an artist, a `datetime` — cannot cross the seam the design says nothing live may cross, and
  finding that out inside `json.dumps` names neither the type nor the field.
* **A CRS is written in a spelling that reads back.** An EPSG integer or a string passes through; a CRS *object*
  is written as `"EPSG:<code>"` when its own definition carries an EPSG code, and as WKT otherwise — an ESRI code
  included. `Bounds` and `Viewport` hold a CRS object in that spelling from the moment they are built.
"""

from math import isfinite
from numbers import Real
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    NoReturn,
    Optional,
    Tuple,
    TypeVar,
)

import numpy as np

T = TypeVar("T")

__all__ = [
    "FrozenDict",
    "frozen_value",
    "as_list",
    "as_mapping",
    "crs_to_json",
    "finite_number",
    "plain_text",
    "positive_number",
    "read_entry",
    "refuse_unknown",
    "require",
    "to_json_value",
    "true_or_false",
]


class FrozenDict(Dict[str, Any]):
    """A `dict` that refuses every change after it is built — how the spec types hold a mapping.

    A read-only `mappingproxy` view froze the mapping too, but it cannot be copied: `pickle`, `copy.deepcopy` and
    `dataclasses.asdict` all raised `cannot pickle 'mappingproxy' object`. This is a real `dict` whose mutators all
    raise: `pickle`, `copy.deepcopy` and `dataclasses.asdict` rebuild it as a `FrozenDict`, `json` writes it, and
    `==` compares it with a plain dict. Like a `dict`, it does not hash.

    Examples:
        - It reads as a dict and refuses a change:
            ```python
            >>> from digitalearth.base.spec._serial import FrozenDict
            >>> frozen = FrozenDict({"levels": (1, 2)})
            >>> frozen == {"levels": (1, 2)}
            True
            >>> frozen["levels"] = (3,)
            Traceback (most recent call last):
                ...
            TypeError: FrozenDict is read-only; build a new value instead

            ```
        - A copy is still read-only:
            ```python
            >>> import copy
            >>> from digitalearth.base.spec._serial import FrozenDict
            >>> clone = copy.deepcopy(FrozenDict({"levels": (1, 2)}))
            >>> clone
            {'levels': (1, 2)}
            >>> clone.pop("levels")
            Traceback (most recent call last):
                ...
            TypeError: FrozenDict is read-only; build a new value instead

            ```
    """

    __slots__ = ()

    def _refuse(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Refuse a change.

        Args:
            *args: Whatever the mutator was given.
            **kwargs: Likewise.

        Raises:
            TypeError: always.
        """
        raise TypeError(
            f"{type(self).__name__} is read-only; build a new value instead"
        )

    __setitem__ = _refuse  # type: ignore[assignment]
    __delitem__ = _refuse  # type: ignore[assignment]
    __ior__ = _refuse  # type: ignore[assignment]
    update = _refuse  # type: ignore[assignment]
    setdefault = _refuse  # type: ignore[assignment]
    pop = _refuse  # type: ignore[assignment]
    popitem = _refuse  # type: ignore[assignment]
    clear = _refuse  # type: ignore[assignment]

    def __reduce__(self) -> Tuple[Any, Tuple[Dict[str, Any]]]:
        """Pickle and copy by rebuilding from a plain dict.

        Returns:
            ``(type(self), (dict(self),))``. The default reduction for a `dict` subclass refills the new object
            item by item through `__setitem__`, which this refuses.
        """
        return type(self), (dict(self),)


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


def true_or_false(value: Any) -> Optional[bool]:
    """Return `value` as a Python bool when it is a boolean, numpy's `bool_` included, and ``None`` otherwise.

    Args:
        value: A flag a caller passed.

    Returns:
        ``True`` or ``False`` for a Python or a numpy boolean, and ``None`` for anything else, which the caller
        refuses in its own words. A numpy boolean is what a comparison on an array gives back, so a flag computed
        with numpy has to be accepted wherever the same flag typed in is. It comes back as Python's, because
        `json.dumps` cannot write numpy's.

    Examples:
        - Both kinds of boolean come back as Python's; a truthy number is not a boolean:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.spec._serial import true_or_false
            >>> true_or_false(np.bool_(True)), true_or_false(False), true_or_false(1)
            (True, False, None)

            ```
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def plain_text(value: Optional[str]) -> Optional[str]:
    """Return a string field as a Python `str`, or ``None`` unchanged.

    Args:
        value: A string the spec holds — any `str` subclass, `numpy.str_` among them — or ``None``.

    Returns:
        The same text as a plain `str`. `np.unique` over a column gives `numpy.str_`, which `json` writes but YAML,
        TOML and msgpack writers refuse, so the typed string fields a `to_dict` writes go through this.

    Examples:
        - A numpy string comes back as Python's:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.spec._serial import plain_text
            >>> type(plain_text(np.str_("dem"))).__name__, plain_text(None)
            ('str', None)

            ```
    """
    return None if value is None else str(value)


def positive_number(value: Any) -> Optional[float]:
    """Return `value` as a float when it is a positive finite real number, numpy's included, and ``None`` otherwise.

    Args:
        value: A size or a ratio a caller passed.

    Returns:
        The value as a Python float, or ``None`` — for a boolean, a string, anything that is not a real number,
        zero, a negative or a non-finite one — which the caller refuses in its own words.

    Examples:
        - A numpy integer or float is accepted and converted; a boolean and a zero are not:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.spec._serial import positive_number
            >>> positive_number(np.int64(8)), positive_number(np.float32(2.5))
            (8.0, 2.5)
            >>> positive_number(True), positive_number(0), positive_number("2")
            (None, None, None)

            ```
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if isfinite(number) and number > 0 else None


#: What `_json_scalar` answers for a value that is not a JSON scalar, so `None` can be a scalar it returns.
_NOT_A_SCALAR = object()


def _json_scalar(value: Any, where: str) -> Any:
    """Return a scalar in the form `json` writes, or `_NOT_A_SCALAR` for anything that is not one.

    Args:
        value: A field's value. A numpy time has already been refused by the caller.
        where: The field, for the message.

    Returns:
        `None` or a boolean unchanged; a string — a `numpy.str_` included — as `str`; a finite number, Python's or
        numpy's, as a Python number; `_NOT_A_SCALAR` otherwise.

    Raises:
        ValueError: for a non-finite number.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        # `numpy.str_` is a `str`, so it was returned as numpy — as `np.float64` was before it was written as a float.
        return str(value)
    if isinstance(value, (int, float)) and not isinstance(value, np.generic):
        # `np.float64` subclasses `float`, so without the second test it came back as numpy, not as the Python float
        # the dict promises — `json` copes, but YAML, TOML and msgpack writers do not.
        return _finite(value, where)
    if isinstance(value, np.generic):
        native = value.item()
        if isinstance(native, (bool, int, float, str)):
            return _finite(native, where)
    return _NOT_A_SCALAR


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
    if isinstance(value, (np.datetime64, np.timedelta64)) or (
        isinstance(value, np.ndarray) and value.dtype.kind in "Mm"
    ):
        # Refused by type, before `.item()`: at nanosecond precision `.item()` returns an *int*, because a Python
        # datetime cannot hold nanoseconds, and that int would be written as though it were a plain number.
        raise TypeError(
            f"{where} holds a {type(value).__name__}, which has no JSON form ({value.dtype}); store a time as an "
            "ISO 8601 string"
        )
    scalar = _json_scalar(value, where)
    if scalar is not _NOT_A_SCALAR:
        return scalar
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
            out[str(key)] = to_json_value(item, f"{where}[{key!r}]")
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


def frozen_value(value: Any) -> Any:
    """Return `value` with every list and tuple in it, however nested, as a tuple.

    Args:
        value: A free-form value a spec holds — a `Symbology` property, an `Encoding` constant, a `Selection` axis,
            a `Scale` category.

    Returns:
        The value with sequences as tuples and the values inside a dict frozen the same way (the dict itself is a
        fresh dict). A numpy array becomes nested tuples of its elements, and a 0-d array its one element: an array
        compares element-wise, so a spec holding one could neither be compared nor hashed. Anything else — a
        scalar, a string — is returned as it is.

        This is the canonical form the constructors store. JSON has no tuple, so a tuple written by `to_dict`
        reads back as a list; storing both spellings as a tuple is what makes ``from_dict(to_dict(x)) == x`` and
        keeps `x` hashable after the round trip. It also means a caller's list is copied, so appending to it later
        no longer changes the spec that was built from it.

    Examples:
        - Lists become tuples, inside dicts too:
            ```python
            >>> from digitalearth.base.spec._serial import frozen_value
            >>> frozen_value([1, [2, 3]]), frozen_value({"levels": [1, 2]})
            ((1, (2, 3)), {'levels': (1, 2)})

            ```
        - An array becomes tuples of its elements, which compare equal to the lists a round trip reads:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.spec._serial import frozen_value
            >>> frozen_value(np.array([[1, 2], [3, 4]])) == ((1, 2), (3, 4))
            True

            ```
    """
    if isinstance(value, (list, tuple)):
        return tuple(frozen_value(item) for item in value)
    if isinstance(value, np.ndarray):
        # The elements stay numpy scalars, so a time array is still refused by type when it is written.
        return (
            value[()]
            if value.ndim == 0
            else tuple(frozen_value(item) for item in value)
        )
    if isinstance(value, dict):
        return {key: frozen_value(item) for key, item in value.items()}
    return value


def hashable_value(value: Any) -> Any:
    """Return `value` in a form that hashes, with every mapping in it as its sorted items.

    The vocabulary freezes a list to a tuple so a spec holding one still hashes, but it leaves a mapping a
    mapping — and every tier records at least one: MapLibre's `paint`, the resolved HoloViews style, a tile
    preset. So almost every real `Symbology` raised `unhashable type: 'dict'`, while `Bounds`, `Scale`,
    `Selection` and `Encoding` all hashed. :meth:`~digitalearth.base.spec.FigureSpec.__hash__` met the same
    wall with its `sources` and answered it this way; this is that answer, reaching wherever a mapping is.

    Args:
        value: A stored property, or any part of one.

    Returns:
        The value with each mapping as a tuple of its ``(key, value)`` pairs in key order, applied inside
        sequences and mappings alike. Anything else is returned as it is — a value that is unhashable for its
        own reasons still raises when it is hashed, which is the honest outcome.

        The order is the keys' own where they compare, and their `repr`\\ s where they do not. A mapping's
        keys need not be orderable against each other — `{1: 'a', 'b': 2}` hashes perfectly well — so sorting
        them was refusing values that this helper exists to accept (review L6). Either way the order is a
        function of the keys alone, never of the insertion order, so two spellings of one mapping agree.

    Examples:
        - Two mappings written in a different order hash alike, because the items are ordered by key:
            ```python
            >>> from digitalearth.base.spec._serial import hashable_value
            >>> hashable_value({"b": 1, "a": 2}) == hashable_value({"a": 2, "b": 1})
            True

            ```
        - It reaches a mapping held inside a sequence:
            ```python
            >>> from digitalearth.base.spec._serial import hashable_value
            >>> hash(hashable_value(({"at": 0.0}, {"at": 1.0}))) is not None
            True

            ```
        - Keys of two types are ordered by `repr` rather than refused:
            ```python
            >>> from digitalearth.base.spec._serial import hashable_value
            >>> hash(hashable_value({1: "a", "b": 2})) is not None
            True

            ```
    """
    if isinstance(value, Mapping):
        items = [(key, hashable_value(item)) for key, item in value.items()]
        try:
            return tuple(sorted(items))
        except TypeError:
            # Only the keys are ever compared — a mapping's keys are unique, so the second half of a pair is
            # never reached — and two key types need not be ordered against each other. `repr` gives them one
            # total order that still depends on nothing but the keys.
            return tuple(sorted(items, key=lambda item: repr(item[0])))
    if isinstance(value, tuple):
        return tuple(hashable_value(item) for item in value)
    return value


def thawed_value(value: Any) -> Any:
    """Return `value` with every tuple in it, however nested, as a list.

    The inverse of :func:`frozen_value`, for a renderer handing a stored property to an engine that reads the
    two spellings as two different requests. HoloViews is the one that forced it: it reads a tuple of
    dimensions as a ``(name, label)`` pair, so the `("fid",)` a symbology stores is refused where the
    `["fid"]` it was built from is accepted.

    Args:
        value: A stored property, or any part of one.

    Returns:
        The value with every tuple as a list and the values inside a dict thawed the same way (the dict
        itself is a fresh dict). Anything else — a scalar, a string, an object — is returned as it is.

    Examples:
        - Tuples become lists, inside dicts too:
            ```python
            >>> from digitalearth.base.spec._serial import thawed_value
            >>> thawed_value((1, (2, 3))), thawed_value({"levels": (1, 2)})
            ([1, [2, 3]], {'levels': [1, 2]})

            ```
        - It undoes a freeze, which is the round trip a drawer depends on:
            ```python
            >>> from digitalearth.base.spec._serial import frozen_value, thawed_value
            >>> thawed_value(frozen_value({"cmap": ["#f00", "#00f"]}))
            {'cmap': ['#f00', '#00f']}

            ```
    """
    if isinstance(value, tuple):
        return [thawed_value(item) for item in value]
    if isinstance(value, dict):
        return {key: thawed_value(item) for key, item in value.items()}
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


def read_entry(owner: str, key: str, read: Callable[[Any], T], value: Any) -> T:
    """Read one stored part with its own `from_dict`, naming where it sits if that refuses it.

    Args:
        owner: The type reading the enclosing dict — ``"FigureSpec"``.
        key: Where the part sits in it — ``"sources['a']"``, ``"panels[1]"``, ``"viewport"``.
        read: The part's reader, such as `DataRef.from_dict`.
        value: The stored part.

    Returns:
        What `read` returns.

    Raises:
        TypeError: as `read` raised it, with ``"<owner>.from_dict <key>: "`` in front of its message. Nested reads
            each add their own step, so the message is the path from the outermost dict to the broken part.
        ValueError: likewise. An error of a subclass of either is raised as `read` raised it, without the step,
            since a subclass's constructor may not take a plain message.

    Examples:
        - A broken source names the figure field it was stored under:
            ```python
            >>> from digitalearth.base.spec import DataRef
            >>> from digitalearth.base.spec._serial import read_entry
            >>> read_entry("FigureSpec", "sources['b']", DataRef.from_dict, None)
            Traceback (most recent call last):
                ...
            TypeError: FigureSpec.from_dict sources['b']: DataRef.from_dict needs a mapping; got NoneType

            ```
    """
    try:
        return read(value)
    except (TypeError, ValueError) as error:
        if type(error) not in (TypeError, ValueError):
            # A subclass may not take a plain message; it is raised as it was, still naming its own type.
            raise
        raise type(error)(f"{owner}.from_dict {key}: {error}") from error


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
        `None` unchanged, a string (a numpy string included) as a Python `str` — not checked, so one pyramids
        cannot read is written as given — and an integer (a numpy integer included) as a Python `int`. A CRS object
        becomes `"EPSG:<code>"` when its own definition carries an EPSG code, and its WKT otherwise — an ESRI code
        included. Either reads back through the same pyramids parser that read the object, so the system is kept
        even though the Python type is not.

        The code is read off the definition, never identified against the PROJ database. Identification is
        slow — tens of milliseconds for a CRS with no code, on every call — and at its default confidence it
        guesses: a UTM zone on the International ellipsoid comes back as EPSG:23031, which is ED50, another
        datum, so a stored figure would read back in a CRS it was not built in.

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
        - A CRS object is written by its EPSG code, or as WKT when its definition carries none:
            ```python
            >>> from pyramids.base.crs import crs_from_user_input
            >>> from digitalearth.base.spec._serial import crs_to_json
            >>> crs_to_json(crs_from_user_input(3857), "Bounds.crs")
            'EPSG:3857'
            >>> crs_to_json(crs_from_user_input("+proj=ortho +lat_0=30 +lon_0=10"), "Bounds.crs")[:7]
            'PROJCRS'

            ```
        - A CRS with no code of its own is not written as the code PROJ would guess for it:
            ```python
            >>> from pyramids.base.crs import crs_from_user_input
            >>> from digitalearth.base.spec._serial import crs_to_json
            >>> crs_to_json(crs_from_user_input("+proj=utm +zone=31 +ellps=intl"), "Bounds.crs")[:7]
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
        return plain_text(crs)
    if isinstance(crs, bool):
        raise TypeError(
            f"{where} is {crs!r}, which names no coordinate reference system"
        )
    if isinstance(crs, (int, np.integer)):
        return int(crs)
    # Imported here: reading a CRS object is pyramids' job, and a spec holding only EPSG ints or strings should
    # not pay for importing it.
    from pyramids.base.crs import crs_from_user_input

    try:
        parsed = crs_from_user_input(crs)
    # Whatever pyramids cannot read as a CRS — a float, a list, an arbitrary object — has no written form.
    except Exception:  # noqa: BLE001
        parsed = None
    definition = getattr(parsed, "to_json_dict", None)
    if not callable(definition):
        raise TypeError(
            f"{where} holds a {type(crs).__name__} that is not a readable CRS; store an EPSG integer or a CRS "
            "string"
        )
    identifier = definition().get("id") or {}
    if identifier.get("authority") == "EPSG" and isinstance(
        identifier.get("code"), int
    ):
        return f"EPSG:{identifier['code']}"
    return str(parsed.to_wkt())
