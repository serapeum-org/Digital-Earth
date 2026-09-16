"""Which slice of a dataset a layer draws — one value instead of a scalar plus per-tier extras.

Every builder takes ``band: int = 1``. That one integer is being asked to answer several different questions,
and where a tier needs more it adds a parameter beside it rather than extending the answer:

* a **composite** needs three bands at once, so `rgb_composite` takes a separate ``bands`` sequence
* the **3-D** tier needs a vertical level
* **temporal** work needs a frame sequence, which each of ``animate`` / ``timecube`` / ``timeslider`` handles
  its own way
* the **large-data** paths need an overview level and a read budget

So "which slice" is currently a scalar, plus whatever each method decided to put next to it. `Selection` is
that question as one value: built once, passed down, and narrowed with :meth:`Selection.with_band` rather than
rebuilt.

The load-bearing change is that `band` is a **tuple**. A composite is not a special case of a different
parameter; it is a selection of three bands.

This type does not resolve anything — it says *what* to read, and :class:`~digitalearth.base.spec.DataRef`
says *where* from. Materialising both is the data tier's job (`DE-16`, Wave 2).
"""

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from digitalearth.base.spec._serial import as_list, refuse_unknown, to_json_value

__all__ = ["DEFAULT_BAND", "Selection"]

#: The band a builder reads when the caller names none. 1-based, matching every tier's public ``band=``.
DEFAULT_BAND: int = 1


def _as_bands(band: Any) -> Tuple[Any, ...]:
    """Normalise a band argument to a tuple, leaving validation to :class:`Selection`.

    Args:
        band: A single band, or any iterable of them — a list, a tuple, a numpy array or a generator.

    Returns:
        The bands as a tuple. Anything that is not an iterable of bands is wrapped as a one-tuple, so the
        real complaint comes from the one place that knows what a band must be rather than from ``tuple()``.
        Testing for ``Iterable`` rather than ``Sequence`` is deliberate: a numpy array of indices and a
        generator expression are both natural ways to name bands, and ``Sequence`` excludes both.
    """
    if isinstance(band, Iterable) and not isinstance(band, (str, bytes)):
        return tuple(band)
    return (band,)


@dataclass(frozen=True)
class Selection:
    """A slice of a dataset: which bands, and which position along every other axis.

    Attributes:
        band: 1-based band indices, as a tuple. A single band is a one-tuple; a composite is three. Ordered,
            because an RGB composite's order is the channel order.
        time: Time step or timestamp, or ``None`` for a dataset with no time axis.
        level: Vertical level, for a 3-D or atmospheric dataset.
        member: Ensemble member.
        overview: Overview/LOD level to read, where the source has a pyramid. ``None`` reads full resolution.
        budget: Greatest number of cells or features the reader may return, for a large-data path. ``None``
            means unbounded.

    Raises:
        ValueError: if `band` is empty, or any band index is below 1. Bands are 1-based everywhere in this
            package's public surface — a 0 is almost always a caller who expected 0-based indexing, and
            silently reading band 1 instead would draw the wrong data with no error.

    Examples:
        - A scalar band is normalised to a one-tuple, so one code path serves both:
            ```python
            >>> from digitalearth.base.spec import Selection
            >>> Selection.of(2).band
            (2,)

            ```
        - A composite is a selection of three bands, not a different parameter:
            ```python
            >>> from digitalearth.base.spec import Selection
            >>> rgb = Selection.of((3, 2, 1))
            >>> rgb.band, rgb.is_composite
            ((3, 2, 1), True)

            ```
        - Narrowing to one band keeps everything else:
            ```python
            >>> from digitalearth.base.spec import Selection
            >>> stack = Selection.of((1, 2, 3), level=850, member=4)
            >>> narrowed = stack.with_band(2)
            >>> narrowed.band, narrowed.level, narrowed.member
            ((2,), 850, 4)

            ```
    """

    band: Tuple[int, ...] = (DEFAULT_BAND,)
    time: Any = None
    level: Any = None
    member: Any = None
    overview: Optional[int] = None
    budget: Optional[int] = None

    def __post_init__(self) -> None:
        """Refuse a selection that names no readable band.

        Raises:
            ValueError: for an empty band tuple or a non-positive index.
        """
        # A caller reaching the constructor directly can pass a list; stored as given it would leave the
        # selection unhashable, so coerce before the guards read it.
        object.__setattr__(self, "band", tuple(self.band))
        if not self.band:
            raise ValueError("Selection needs at least one band")
        bands: List[int] = []
        for index in self.band:
            # bool is an int in Python, and True would read as band 1 — almost never what a caller meant.
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise ValueError(
                    f"Selection needs whole 1-based band numbers; got {index!r}"
                )
            if index < 1:
                raise ValueError(
                    f"Selection bands are 1-based; got {index}. Band 0 is usually a caller expecting "
                    "0-based indexing, which would silently draw the wrong band"
                )
            # Accept any Integral but store a Python int: np.int64 is a whole 1-based band number by every
            # reasonable reading, and GDAL's SWIG binding rejects it outright, so coercing here is what makes
            # `bands=np.array([3, 2, 1])` work at all rather than failing deep inside the reader.
            bands.append(int(index))
        object.__setattr__(self, "band", tuple(bands))

    @classmethod
    def of(
        cls, band: Union[int, Iterable[int]] = DEFAULT_BAND, **rest: Any
    ) -> "Selection":
        """Build a selection, accepting a band as either a scalar or a sequence.

        The one constructor both spellings go through, so a builder that takes ``band=2`` and one that takes
        ``bands=(3, 2, 1)`` end up with the same kind of value.

        Args:
            band: A single 1-based band, or a sequence of them.
            **rest: Any other axis — `time`, `level`, `member`, `overview`, `budget`.

        Returns:
            The selection.

        Raises:
            ValueError: if the bands are empty or not 1-based.

        Examples:
            - Both spellings produce the same shape:
                ```python
                >>> from digitalearth.base.spec import Selection
                >>> Selection.of(1) == Selection.of([1])
                True

                ```
        """
        bands = _as_bands(band)
        return cls(band=bands, **rest)

    @property
    def is_composite(self) -> bool:
        """Whether this selection names more than one band.

        Returns:
            ``True`` for a multi-band selection — what a composite renderer needs to know.

        Examples:
            - Three bands are a composite; one is not:
                ```python
                >>> from digitalearth.base.spec import Selection
                >>> Selection.of((3, 2, 1)).is_composite
                True
                >>> Selection.of(1).is_composite
                False

                ```
            - Which is how a builder picks its path without inspecting the tuple itself:
                ```python
                >>> from digitalearth.base.spec import Selection
                >>> sel = Selection.of((4, 3, 2))
                >>> len(sel.band) if sel.is_composite else 1
                3

                ```
        """
        return len(self.band) > 1

    @property
    def first_band(self) -> int:
        """The single band a scalar-band renderer should read.

        Returns:
            The first band in the selection. A composite narrowed to one channel goes through
            :meth:`with_band` instead; this is for the many builders that only ever draw one.

        Examples:
            - The band a single-band builder should read:
                ```python
                >>> from digitalearth.base.spec import Selection
                >>> Selection.of(3).first_band
                3

                ```
            - For a composite it is the first channel, in the order the caller wrote them:
                ```python
                >>> from digitalearth.base.spec import Selection
                >>> Selection.of((7, 2, 1)).first_band
                7

                ```
        """
        return self.band[0]

    def with_band(self, band: Union[int, Iterable[int]]) -> "Selection":
        """Return a copy naming different bands, keeping every other axis.

        Args:
            band: The replacement band or bands.

        Returns:
            A new selection. The original is unchanged — this is how a composite iterates its channels without
            losing the level, member or budget that were chosen once.

        Raises:
            ValueError: if the replacement is empty or not 1-based.

        Examples:
            - The other axes survive the narrowing:
                ```python
                >>> from digitalearth.base.spec import Selection
                >>> Selection.of(1, time="2024-01").with_band(7).time
                '2024-01'

                ```
        """
        bands = _as_bands(band)
        return replace(self, band=bands)

    def frames(self) -> Tuple["Selection", ...]:
        """Return one single-band selection per band, in order.

        Returns:
            One selection per band — for a single-band selection, one equal to this one; for a composite, the
            per-channel reads in channel order, each still carrying the shared axes. The entries are fresh
            values rather than ``self``, so a caller may hold both without either aliasing the other.

        Examples:
            - An RGB composite becomes its three channel reads, in order:
                ```python
                >>> from digitalearth.base.spec import Selection
                >>> [s.first_band for s in Selection.of((3, 2, 1)).frames()]
                [3, 2, 1]

                ```
        """
        return tuple(self.with_band(index) for index in self.band)

    # ------------------------------------------------------------------ serialisation

    def to_dict(self) -> Dict[str, Any]:
        """Return the plain-dict form a figure stores.

        Returns:
            ``band`` as a list, plus each other axis that is set. An unset axis is omitted rather than written as
            ``None``, so the common single-band selection is one key.

        Raises:
            TypeError: if an axis other than `band` holds a value with no JSON form — a `datetime`, a set or `nan`,
                say. Store a `datetime` as its ISO string instead.

        Examples:
            - The default selection is one key:
                ```python
                >>> from digitalearth.base.spec import Selection
                >>> Selection().to_dict()
                {'band': [1]}

                ```
            - A slice through time and level keeps both:
                ```python
                >>> from digitalearth.base.spec import Selection
                >>> Selection.of(2, time="2024-01", level=850).to_dict()
                {'band': [2], 'time': '2024-01', 'level': 850}

                ```
            - A live `datetime` has no JSON form and is refused where it is written:
                ```python
                >>> from datetime import datetime
                >>> from digitalearth.base.spec import Selection
                >>> Selection.of(1, time=datetime(2024, 1, 1)).to_dict()  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                TypeError: Selection.time holds a datetime, which has no JSON form. ...

                ```
        """
        out: Dict[str, Any] = {"band": list(self.band)}
        for name in ("time", "level", "member", "overview", "budget"):
            value = getattr(self, name)
            if value is not None:
                out[name] = to_json_value(value, f"Selection.{name}")
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Selection":
        """Rebuild a selection from its dict form.

        Args:
            data: A mapping as produced by :meth:`to_dict`. A missing ``band`` means the default band.

        Returns:
            The selection, validated as the constructor validates it.

        Raises:
            TypeError: if `data` is not a mapping, or `band` is not a list — a bare number is refused by name,
                since `to_dict` always writes a list.
            ValueError: for an unknown key, or a band the constructor refuses.

        Examples:
            - A stored composite reads back as a band tuple:
                ```python
                >>> from digitalearth.base.spec import Selection
                >>> Selection.from_dict({"band": [3, 2, 1]}).band
                (3, 2, 1)

                ```
            - Without a `band` key the default band is read, and the other axes come back as stored:
                ```python
                >>> from digitalearth.base.spec import Selection
                >>> selection = Selection.from_dict({"time": "2024-01-01", "level": 850})
                >>> selection.band, selection.time, selection.level
                ((1,), '2024-01-01', 850)

                ```
        """
        refuse_unknown(
            "Selection", data, ("band", "time", "level", "member", "overview", "budget")
        )
        return cls(
            band=as_list("Selection", "band", data.get("band", (DEFAULT_BAND,))),
            time=data.get("time"),
            level=data.get("level"),
            member=data.get("member"),
            overview=data.get("overview"),
            budget=data.get("budget"),
        )
