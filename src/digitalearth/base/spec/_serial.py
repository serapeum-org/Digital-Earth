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
    List,
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
    "MAX_TRAVELLING_DEPTH",
    "MAX_TRAVELLING_ELEMENTS",
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
    "travels_in_a_figure",
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
        numpy's, as a plain Python `int` or `float`; `_NOT_A_SCALAR` otherwise.

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
        #
        # Through `int()`/`float()`, for that same argument one step further (`R2-M2`). A subclass came back live:
        # a units library's quantity, and — the case that reads worst — an `int`-valued `enum` member, sitting
        # inside the plain mapping `Symbology.to_dict` promises. `str` was flattened here from the beginning; its
        # number siblings were not, and the round trip hid it because `json` flattens a number on its own. A
        # flattening that moved the *value* is caught by `_written_back_equal`, which compares what came out.
        plain = int(value) if isinstance(value, int) else float(value)
        return _finite(plain, where)
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


#: What :func:`_written_back_equal` tells the writer it is asking about. The refusal's message is thrown away —
#: only the yes or no is used — but `to_json_value` names a field in every message it raises, and every caller
#: of the oracle is asking on behalf of a layer's recorded properties.
_ASKING_FOR: str = "Symbology.props"


def _written_back_equal(value: Any) -> bool:
    """Whether the figure writer takes `value` **and** writes something equal to it.

    The oracle is the writer itself — :func:`to_json_value`, which `Symbology.to_dict` applies — so the answer
    cannot drift from what a figure actually accepts. It is what refuses `nan` and the infinities, a `datetime`,
    a set, a mapping with a non-string key, and every live object.

    Acceptance alone is not enough, and one scalar proves it. The writer flattens a scalar subclass to its plain
    counterpart, which is almost always the *same value* — ``np.float64(2.5)`` writes ``2.5``, a `str` subclass
    writes its characters. A `str`-valued `enum.Enum` member is the exception: flattening goes through `str`,
    which on a mixin enumeration is `Enum.__str__`, so ``Linestyle.SOLID`` writes ``'Linestyle.SOLID'`` where its
    value ``'solid'`` belongs — a layer that reloads with a linestyle no engine has heard of. Comparing what came
    out against what went in catches that by measurement rather than by naming `enum`, so any scalar whose
    flattening moves the value is refused the same way.

    Args:
        value: A scalar a layer is about to record.

    Returns:
        `True` when the writer would take it and give back an equal value.
    """
    try:
        written = to_json_value(value, _ASKING_FOR)
    except (TypeError, ValueError):
        return False
    return bool(written == value)


#: How many values one description may carry for a single keyword, counting every value at every depth —
#: the list itself, each item, and each item's items. The bound is stated rather than implied, because it is
#: the second of the two reasons a container might not travel and the type gate used to hide it: refusing
#: every container refused the big ones as a side effect, and widening the gate would have let a per-pixel
#: `alpha` through with nothing left to stop it.
#:
#: Measured on the full record path (freeze, hash, write, `json.dumps`) for a flat list of floats:
#: 256 values ~0.6 ms / 1.6 KB, 1,000 ~2.2 ms / 6.7 KB, 10,000 ~31 ms / 77 KB, 1,000,000 ~3.0 s / 9.4 MB.
#: Every style container a caller really writes sits far below the bound — a 256-entry palette, a colour
#: ramp, a classifier's edges, a list of columns — and the thing the bound exists to refuse, a per-pixel
#: `alpha` on a 1000 x 1000 raster, is a thousand times above it. There is no realistic value in between,
#: which is why one flat number is enough and why it is a round one.
MAX_TRAVELLING_ELEMENTS: int = 1000

#: How many containers deep one described value may nest. A second bound, and a much smaller number, because
#: it answers a different question: :data:`MAX_TRAVELLING_ELEMENTS` bounds what a figure is willing to *carry*,
#: and this bounds what the record path can *reach*.
#:
#: The path is recursive where the gate is not — `frozen_value`, `hashable_value`, `to_json_value` and
#: `thawed_value` each walk the value with the interpreter's stack — so counting depth against the element
#: budget promised a depth the writers cannot survive (`R2-M1`). Measured here, on a fresh 1,000-frame
#: recursion limit, as the shallowest nesting each one raises `RecursionError` at: `hashable_value` 497,
#: `frozen_value` 499, `to_json_value` 995, `json.dumps` of its output 995. The gate admitted 999.
#:
#: 32, not "a little under 497", because 497 is not a property of this package. It moves with
#: `sys.setrecursionlimit` and with how many frames the caller has already spent before the builder is
#: reached, so a bound close to it would hold on the machine that measured it and nowhere else. 32 costs
#: about 64 frames of whatever stack is left, which any caller has, and it is an order of magnitude above
#: every real style value: a palette nests 1 deep, a dash pattern 2, a dict of per-class colour lists 3.
#:
#: Making the four writers iterative instead was the other way to make the two agree, and it loses: it is
#: four rewrites on the hot record path, it trades a stated bound for an unstated one (whatever stack is
#: left at the call site), and it would leave the gate promising a depth that still cannot be written down.
#: Hardening `to_json_value` against its own recursion remains worth doing for callers that reach it
#: directly — that is #335, and it is not this.
MAX_TRAVELLING_DEPTH: int = 32


def _travelling_budget(value: Any, budget: int) -> int:
    """Return what is left of `budget` after `value`, or ``-1`` when `value` cannot travel.

    The walk behind :func:`travels_in_a_figure`. It counts as it goes, so a container too large to describe
    is refused after :data:`MAX_TRAVELLING_ELEMENTS` values rather than after all of them: the gate costs
    the same whether a caller passes a 3-element list or a million-element one.

    The walk carries its own stack rather than the interpreter's, and that is not a style choice (`R-M1`).
    `MAX_TRAVELLING_ELEMENTS` is 1,000 and CPython's default recursion limit is 1,000, so a recursive walk
    raised `RecursionError` before the budget could refuse a deeply nested container — and a builder that
    raises loses the value outright, where the tier's promise is to *hold* what this refuses beside the
    layer.

    **Depth is counted separately, and much lower** (:data:`MAX_TRAVELLING_DEPTH`, `R2-M1`). Counting it
    against the element budget only moved the raise: the gate accepted a container nested 999 deep, and
    `frozen_value` — recursive, like every other step of the record path — raised on it at 499. A value
    this admits has to be one the tier can then *record*, so the bound that decides has to be one the
    recursive half survives. It also ends a cycle sooner than the element budget did: a container holding
    itself is over the depth bound after `MAX_TRAVELLING_DEPTH` visits, so no cycle memory is needed.

    Args:
        value: The value, or a part of one.
        budget: How many values may still be counted.

    Returns:
        The budget left, or ``-1`` for a value the description cannot carry — one the writer refuses or
        writes back as a different value, a tuple or an array (which JSON reads back as a list), a mapping
        with a non-string key, a live object, or a container that exhausts either bound, whether by holding
        too many values, by nesting too deep, or by holding itself.
    """
    pending: List[Tuple[Any, int]] = [(value, 0)]
    while pending:
        if budget <= 0:
            return -1
        item, depth = pending.pop()
        # `np.generic` is every numpy *scalar* and no array, so it admits the whole family at once rather
        # than naming its members. Enumerating them is what let `np.bool_` fall through while `np.float64`
        # travelled (#329): a numpy float and int register as `numbers.Real` and a numpy str subclasses
        # `str`, but a numpy bool is neither, so it missed a gate it belonged in. Membership is not the
        # decision — the writer still is, and it refuses `datetime64`, `complex128` and `timedelta64`.
        if item is None or isinstance(item, (bool, str, Real, np.generic)):
            if not _written_back_equal(item):
                return -1
            budget -= 1
            continue
        # `list` and `dict` **exactly**, not `isinstance`: those two are what the round trip returns as
        # themselves, and a subclass is re-typed by it exactly as a tuple is. That is not a technicality —
        # an `xyzservices.TileProvider` *is* a dict, of plain strings, one of which is the caller's API key,
        # and `isinstance` wrote it into the figure. A dict subclass is an engine object wearing a dict; the
        # type check that refuses a tuple is what keeps it, and the key in it, out.
        if depth >= MAX_TRAVELLING_DEPTH:
            # Below the scalar branch, so the bound is read as "how deep a *container* may sit": the leaves
            # of the deepest admitted container are plain values the recursive writers reach in one frame.
            return -1
        if type(item) is list:
            budget -= 1
            pending.extend((entry, depth + 1) for entry in item)
            continue
        if type(item) is dict:
            budget -= 1
            for key, entry in item.items():
                if not isinstance(key, str):
                    return -1
                pending.append((entry, depth + 1))
            continue
        return -1
    return budget


def travels_in_a_figure(value: Any) -> bool:
    """Whether a figure's description carries `value`, or the tier must hold it beside the layer.

    The one rule the tiers share, so that "what can a figure carry?" has a single answer rather than one per
    backend (#322) — which tier asks it, and how, is at the foot of this docstring. The rule is two-sided,
    and it is stated that way because it is enforced that way:

    * **A scalar travels when the round trip gives back an equal value.** Its class need not survive, and for
      a subclass it does not: the trip flattens every one of them to the plain counterpart the writer spells.
    * **A container travels when the round trip gives back an equal value of the same class**, at every depth.

    So: a string, a boolean, a finite number or `None`, and a `list` or `dict` built out of those — up to
    :data:`MAX_TRAVELLING_ELEMENTS` values, nested no more than :data:`MAX_TRAVELLING_DEPTH` containers
    deep. That is what a reader on another machine can act on, and — the second bound's whole job — what
    the recursive record path behind this can then write down without raising (`R2-M1`).

    The asymmetry is not an oversight, and it is the half a single-sentence "an equal value of the same kind"
    got wrong (`R-H1`, `R-M5`). A container's class is load-bearing twice over — flattening a tuple changes the
    *value* into one matplotlib refuses, and flattening a `dict` subclass copies its contents, the API key
    included, into the figure. A scalar carries nothing but itself, so flattening it moves no payload and,
    where the value survives, costs the drawer nothing: ``np.float64(2.5)`` is handed on as ``2.5``, which every
    engine takes in its place. Where the value does *not* survive the flattening the scalar is refused too —
    see :func:`_written_back_equal` for the `str`-valued enumeration that makes the point.

    It is **narrower than the writer**, and the boundary is measured rather than argued. Taking the whole
    record-to-draw path — `frozen_value` into the spec, `to_json_value` out to JSON, back in, `thawed_value`
    at the drawer — here is each kind, and what the trip does to it:

    * **A list or a dict of plain values comes back as itself.** ``['#ff0000', '#00ff00']``, ``[4, 4]``,
      ``[0.0, 0.5, 1.0]``, ``['fid']`` and ``{'a': 1}`` each return equal, and as the same type, through both
      the freeze and the JSON trip. So a graduated choropleth's ``color_levels`` and a categorical palette
      travel, and a figure reloaded elsewhere still redraws in the colours it was built with (#330).
    * **A tuple does not.** JSON has no tuple, so a matplotlib dash pattern written as ``(0, (5, 5))`` comes
      back ``[0, [5, 5]]``, which matplotlib refuses outright (``ValueError: Unrecognized linestyle``).
      Describing it would trade a layer that redraws with the engine's defaults for one that cannot redraw at
      all. The loss is inherent to the **tuple**, not to the container: a tuple nested inside a list re-types
      the same way, so ``[1, (2, 3)]`` is held too.
    * **An array does not either**, and for a second reason: it is re-typed like a tuple *and* it is
      unbounded. An array is the layer's data; a description is not the place to copy it to.
    * **An engine object has no JSON form at all** — a ``Normalize``, a ``FontProperties``, a ``Colormap``, a
      Datashader reduction — so a figure holding one could not be written down. The writer is the oracle for
      every scalar and every leaf, so that refusal is not re-spelled here.
    * **A `list` or `dict` *subclass* does not travel either**, and for the same reason a tuple does not: the
      trip returns a plain `list` or `dict`, which is a different type. It is the subclasses that make this
      matter rather than the principle — an ``xyzservices.TileProvider`` *is* a dict, of plain strings, one
      of which is the caller's API key, and a figure is not a place to write a credential.
    * **A *scalar* subclass does travel**, and comes back as its plain counterpart — flattened by the writer
      itself, not by `json` on the way out (`R2-M2`). ``np.float64(2.5)`` returns ``2.5``, ``np.bool_(True)``
      returns ``True``, a `str` subclass returns its characters as `str` and an `int`-valued `enum` member
      returns the number it stands for rather than the member. The value
      is the same one, and it is the one the drawer would have been handed anyway, so refusing it would hold a
      numpy flag out of a figure for a change no engine can observe (#329). Nothing else rides along: unlike a
      container subclass, a scalar has no contents for the trip to copy. The exception is the one scalar whose
      *value* moves — a `str`-valued `enum.Enum` member, flattened through `Enum.__str__` — and that is refused
      on the same measurement rather than by name.

    The other half of the rule is the asking tier's to keep, and it is two-sided:

    * A value this refuses must still reach the drawer, or the engine silently draws its own default in its
      place. The tier holds what is refused under the layer's id and merges it back before drawing.
    * A value this accepts is stored frozen — `Symbology` turns every list into a tuple so a spec still
      hashes — so the tier **thaws the described half at its read boundary**. Because only a `list` ever
      travels, thawing is the exact inverse of that freeze; a genuine tuple is in the held half, which is
      merged on afterwards and never thawed.

    **Which tiers ask, and which do not.** Measured: the two that hand a caller's keywords straight to their
    engine ask it per value and keep both halves — the static tier in
    :func:`~digitalearth.static.scene.described_opts` (held in ``Scene._layer_opts``) and the interactive
    tier in :func:`~digitalearth.interactive.base.describe` (held in ``InteractiveMapBase._layer_held``).
    The web and 3-D tiers take named, typed parameters and resolve them into their engine's own spelling
    before recording, so they have no caller keyword to split and no held half of their own; the web drawer
    thaws what it reads all the same (``web/renderer.py``), and the 3-D tier calls no thaw at all.

    **Three** of the four also reach this rule *indirectly*, through
    :func:`~digitalearth.base.spec.style.portable_constants`, which asks it of any value before lifting it
    onto a channel: the two above, and the web tier through its own ``portable_encodings``. The 3-D tier
    reaches it by **no** path — measured, nothing under ``three_d/`` names this function,
    ``portable_constants``, ``asked_constants`` or ``portable_encodings`` — so that tier publishes no
    channel and asks this rule of nothing. "Every tier" was the reading this paragraph was rewritten to
    stop, and the sentence that followed the rewrite put it back (`R2-N1`).

    Args:
        value: The caller's value for one keyword.

    Returns:
        `True` when the layer's description carries it; `False` when the tier must hold it beside the layer.

    Examples:
        - A scalar travels, and so does a list or a dict of scalars:
            ```python
            >>> from digitalearth.base.spec._serial import travels_in_a_figure
            >>> travels_in_a_figure(0.25), travels_in_a_figure("solid"), travels_in_a_figure(None)
            (True, True, True)
            >>> travels_in_a_figure(["#ff0000", "#00ff00"]), travels_in_a_figure({"a": 1})
            (True, True)

            ```
        - A tuple does not, wherever it sits, because JSON reads it back as a list:
            ```python
            >>> from digitalearth.base.spec._serial import travels_in_a_figure
            >>> travels_in_a_figure((0, (5, 5))), travels_in_a_figure([1, (2, 3)])
            (False, False)

            ```
        - A scalar the writer itself refuses does not travel, inside a container or out of one:
            ```python
            >>> from digitalearth.base.spec._serial import travels_in_a_figure
            >>> travels_in_a_figure(float("nan")), travels_in_a_figure([1.0, float("nan")])
            (False, False)

            ```
        - A numpy scalar travels wherever its Python counterpart does, and the array around it does not:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.spec._serial import travels_in_a_figure
            >>> travels_in_a_figure(np.bool_(True)), travels_in_a_figure(np.float64(2.5))
            (True, True)
            >>> travels_in_a_figure(np.array([1.0, 2.0]))
            False

            ```
        - A container is bounded by the values in it, however they are nested:
            ```python
            >>> from digitalearth.base.spec._serial import MAX_TRAVELLING_ELEMENTS, travels_in_a_figure
            >>> travels_in_a_figure(list(range(MAX_TRAVELLING_ELEMENTS - 1)))
            True
            >>> travels_in_a_figure(list(range(MAX_TRAVELLING_ELEMENTS)))
            False

            ```
        - And by how deeply they are nested, which is the bound the record path sets:
            ```python
            >>> from digitalearth.base.spec._serial import MAX_TRAVELLING_DEPTH, travels_in_a_figure
            >>> deep = "leaf"
            >>> for _ in range(MAX_TRAVELLING_DEPTH):
            ...     deep = [deep]
            >>> travels_in_a_figure(deep), travels_in_a_figure([deep])
            (True, False)

            ```
    """
    return _travelling_budget(value, MAX_TRAVELLING_ELEMENTS) >= 0


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


#: Heads a mapping's hash surrogate, so a mapping never hashes as the tuple of pairs it is turned into.
#: `Symbology(props={"p": {"x": 1}})` and `Symbology(props={"p": (("x", 1),)})` are unequal properties that
#: landed on one hash without it (review N1). A fresh object rather than a string or a tuple: a property can
#: hold any value this module could spell, and colliding with one is what the tag exists to prevent.
_MAPPING_TAG: Any = object()


def hashable_value(value: Any) -> Any:
    """Return `value` in a form that hashes, with every mapping in it reduced to its items.

    The vocabulary freezes a list to a tuple so a spec holding one still hashes, but it leaves a mapping a
    mapping — and every tier records at least one: MapLibre's `paint`, the resolved HoloViews style, a tile
    preset. So almost every real `Symbology` raised `unhashable type: 'dict'`, while `Bounds`, `Scale`,
    `Selection` and `Encoding` all hashed.

    A mapping becomes a **frozenset** of its ``(key, value)`` pairs, applied inside a mapping and inside a
    **tuple**. A frozenset is order-independent by construction, which is what makes two spellings of one
    mapping agree: the keys need neither to be orderable against each other — ``{1: 'a', 'b': 2}`` hashes
    perfectly well — nor to have distinct reprs. It carries a tag, so a mapping never hashes as the plain
    collection of pairs that a caller might have written instead.

    Args:
        value: A stored property, or any part of one.

    Returns:
        The value with each mapping as a tagged frozenset of its items, reached inside mappings and tuples
        alike. Anything else is returned as it is — a value that is unhashable for its own reasons still
        raises when it is hashed, which is the honest outcome.

    Examples:
        - Two mappings written in a different order reduce to the same value, so they hash alike:
            ```python
            >>> from digitalearth.base.spec._serial import hashable_value
            >>> hashable_value({"b": 1, "a": 2}) == hashable_value({"a": 2, "b": 1})
            True

            ```
        - Keys that cannot be ordered against each other are no obstacle, because nothing is ordered:
            ```python
            >>> from digitalearth.base.spec._serial import hashable_value
            >>> hash(hashable_value({1: "a", "b": 2})) is not None
            True

            ```
        - A mapping does not hash as the pairs it is made of:
            ```python
            >>> from digitalearth.base.spec._serial import hashable_value
            >>> hashable_value({"x": 1}) == hashable_value((("x", 1),))
            False

            ```
    """
    if isinstance(value, Mapping):
        # A frozenset rather than a sorted tuple: it is order-independent by construction, so it needs the
        # keys neither to be orderable against each other nor to have distinct reprs. Sorting by `repr` as a
        # fallback quietly required the second — a stable sort keeps insertion order for keys that share one,
        # so two mappings that compared equal hashed differently, which is the invariant this exists to hold.
        return (
            _MAPPING_TAG,
            frozenset((key, hashable_value(item)) for key, item in value.items()),
        )
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
