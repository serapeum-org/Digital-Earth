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
from typing import Any, Optional, Sequence, Tuple, Union

__all__ = ["DEFAULT_BAND", "Selection"]

#: The band a builder reads when the caller names none. 1-based, matching every tier's public ``band=``.
DEFAULT_BAND: int = 1


def _as_bands(band: Any) -> Tuple[Any, ...]:
    """Normalise a band argument to a tuple, leaving validation to :class:`Selection`.

    Args:
        band: A single band, or any sequence of them.

    Returns:
        The bands as a tuple. A non-sequence — including a ``bool``, which is an ``int`` in Python and would
        otherwise fail obscurely inside ``tuple()`` — is wrapped as a one-tuple so the real complaint comes
        from the one place that knows what a band must be.
    """
    if isinstance(band, Sequence) and not isinstance(band, (str, bytes)):
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
        if not self.band:
            raise ValueError("Selection needs at least one band")
        for index in self.band:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError(
                    f"Selection needs whole 1-based band numbers; got {index!r}"
                )
            if index < 1:
                raise ValueError(
                    f"Selection bands are 1-based; got {index}. Band 0 is usually a caller expecting "
                    "0-based indexing, which would silently draw the wrong band"
                )

    @classmethod
    def of(
        cls, band: Union[int, Sequence[int]] = DEFAULT_BAND, **rest: Any
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
        """
        return len(self.band) > 1

    @property
    def first_band(self) -> int:
        """The single band a scalar-band renderer should read.

        Returns:
            The first band in the selection. A composite narrowed to one channel goes through
            :meth:`with_band` instead; this is for the many builders that only ever draw one.
        """
        return self.band[0]

    def with_band(self, band: Union[int, Sequence[int]]) -> "Selection":
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
            One selection per band. For a single-band selection that is just ``(self,)``; for a composite it
            is the per-channel reads, in channel order, each still carrying the shared axes.

        Examples:
            - An RGB composite becomes its three channel reads, in order:
                ```python
                >>> from digitalearth.base.spec import Selection
                >>> [s.first_band for s in Selection.of((3, 2, 1)).frames()]
                [3, 2, 1]

                ```
        """
        return tuple(self.with_band(index) for index in self.band)
