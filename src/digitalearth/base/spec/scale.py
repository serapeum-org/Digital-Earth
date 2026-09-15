"""How a value becomes a colour — the one rule, instead of one per tier.

Deciding what colour a number gets was spread across the package: the limits were derived in several places,
the "a constant band has no range" guard was written out five times, and three tiers each called cleopatra's
classifier with their own error handling. The copies agreed by luck, and drifted where they did not.

`Scale` holds the three parts of that decision together — what **kind** of scale it is, what **domain** it
covers, and the **colours** it maps onto — and can be *frozen*: a scale derived once and reused across frames
is what stops an animation's colours flickering, and what lets a legend show the colours that were actually
drawn rather than a second guess at them.

What it absorbs:

| Was | Where |
|---|---|
| the constant-band widening rule, `hi = lo + 1.0` | `base/stretch.py`, `static/textured_globe.py`, `web/bigdata.py`, `web/raster.py`, `web/vector.py` |
| `classify(values, scheme, k)` plus its own `try`/`except` | `interactive/vector.py`, `three_d/base.py`, `web/vector.py` |
| the stack colour range | :mod:`digitalearth.base.clim` — one way to *construct* frozen limits, not a parallel mechanism |

Engine-neutral, and the classifier is the part that took thought. The arithmetic lives in cleopatra, which
`base/` may not import **at all** — not even lazily, since `tests/test_base_is_engine_neutral.py` reads the
source rather than the imports. So this module declares a seam,
:func:`digitalearth.base.registry.register_classifier`, and :mod:`digitalearth` fills it at package import.
The policy stays here — the default class count, the error that names the scheme and `k`, turning edges
into classes — and only the arithmetic is injected.
"""

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from digitalearth.base.arrays import finite
from digitalearth.base.registry import get_classifier

__all__ = ["DEFAULT_CLASS_COUNT", "Scale"]

#: Classes a graduated scheme cuts when the caller names a scheme but not a count. Five is the cartographic
#: convention and what every tier already defaulted to; sharing it is what stops a sixth appearing.
DEFAULT_CLASS_COUNT: int = 5


@dataclass(frozen=True)
class Scale:
    """A resolved mapping from values to colours, and the domain it covers.

    Attributes:
        vmin: Lower limit of the domain.
        vmax: Upper limit. Always strictly greater than `vmin` — see :meth:`from_values` for why.
        scheme: Classification scheme name (``"quantiles"``, ``"equal_interval"``, …), or ``None`` for a
            continuous ramp.
        breaks: Class edges when `scheme` is set; empty for a continuous ramp. ``k`` classes give ``k + 1``
            edges, so a legend can label each class by the range it covers.
        categories: The distinct values a categorical scale colours, in the order their colours were assigned.
        missing: Colour for a value the scale cannot place — nodata, or a category it never saw.

    Raises:
        ValueError: if `vmin`/`vmax` are not finite, or `vmax` is not greater than `vmin`. Build through
            :meth:`from_values` rather than widening by hand.

    Examples:
        - Derive a continuous scale from data, and read back the limits it settled on:
            ```python
            >>> from digitalearth.base.spec import Scale
            >>> scale = Scale.from_values([1.0, 5.0, 9.0])
            >>> scale.vmin, scale.vmax
            (1.0, 9.0)
            >>> scale.is_classified
            False

            ```
        - A constant band is widened rather than dividing by zero:
            ```python
            >>> from digitalearth.base.spec import Scale
            >>> Scale.from_values([7.0, 7.0, 7.0]).as_limits()
            (7.0, 8.0)

            ```
        - Nothing finite falls back to the unit range, so a builder still has limits to hand a colormap:
            ```python
            >>> from digitalearth.base.spec import Scale
            >>> Scale.from_values([]).as_limits()
            (0.0, 1.0)

            ```
    """

    vmin: float
    vmax: float
    scheme: Optional[str] = None
    breaks: Tuple[float, ...] = ()
    categories: Tuple[Any, ...] = ()
    missing: Optional[str] = None
    _colors: Tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        """Refuse a domain nothing can be normalised against.

        Raises:
            ValueError: for a non-finite or degenerate domain, or for class edges that bound no class.
        """
        for name, value in (("vmin", self.vmin), ("vmax", self.vmax)):
            if not isfinite(value):
                raise ValueError(f"Scale needs a finite domain; got {name}={value!r}")
        if self.vmax <= self.vmin:
            raise ValueError(
                f"Scale needs vmax > vmin; got vmin={self.vmin}, vmax={self.vmax}. "
                "Build with Scale.from_values(), which widens a constant domain for you"
            )
        if self.breaks and len(self.breaks) < 2:
            raise ValueError(
                f"Scale needs at least two class edges to bound one class; got {self.breaks}. "
                "An empty breaks tuple is how a continuous ramp says it cuts no classes"
            )

    # ------------------------------------------------------------------ builders

    @classmethod
    def from_values(
        cls,
        values: Any,
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        scheme: Optional[str] = None,
        k: int = DEFAULT_CLASS_COUNT,
        missing: Optional[str] = None,
    ) -> "Scale":
        """Derive a scale from data, applying the one widening rule every tier used to write out.

        Two degenerate cases are handled here rather than at five call sites:

        * **nothing finite** — the domain falls back to ``(0.0, 1.0)``. A builder still needs limits to hand a
          colormap, and a stack whose every frame is off the view has no range to report.
        * **a constant domain** — ``vmax`` becomes ``vmin + 1``. Normalising against a zero-width range
          divides by zero; the five copies of this rule all chose ``+ 1``, so that is what is kept.

        Args:
            values: The data to measure. Non-finite values are ignored.
            vmin: Explicit lower limit, overriding the measured one.
            vmax: Explicit upper limit.
            scheme: Classification scheme to cut classes with, or ``None`` for a continuous ramp.
            k: Number of classes when `scheme` is set.
            missing: Colour for values the scale cannot place.

        Returns:
            The resolved scale.

        Raises:
            ValueError: if `scheme` is unknown to the classifier, or `k` is not a usable class count. The
                message names the scheme and the count, since a classifier error alone says neither.

        Examples:
            - An explicit limit wins over the measured one:
                ```python
                >>> from digitalearth.base.spec import Scale
                >>> Scale.from_values([1.0, 2.0, 3.0], vmax=10.0).as_limits()
                (1.0, 10.0)

                ```
        """
        lo, hi = cls._limits(values, vmin, vmax)
        breaks: Tuple[float, ...] = ()
        if scheme is not None:
            breaks = cls._breaks(values, scheme, k)
        return cls(lo, hi, scheme=scheme, breaks=breaks, missing=missing)

    @classmethod
    def from_limits(
        cls, vmin: float, vmax: float, *, missing: Optional[str] = None
    ) -> "Scale":
        """Build from limits that were derived elsewhere — a frozen stack range, or a caller's `clim=`.

        Args:
            vmin: Lower limit.
            vmax: Upper limit.
            missing: Colour for values the scale cannot place.

        Returns:
            The scale, with the same widening rule applied so a degenerate pair cannot get through.

        Examples:
            - The frozen range of a time stack becomes a scale every frame shares:
                ```python
                >>> from digitalearth.base.spec import Scale
                >>> Scale.from_limits(2.0, 9.0).as_limits()
                (2.0, 9.0)

                ```
            - A degenerate pair is widened here too, rather than raising on the caller:
                ```python
                >>> from digitalearth.base.spec import Scale
                >>> Scale.from_limits(4.0, 4.0).as_limits()
                (4.0, 5.0)

                ```
        """
        lo, hi = float(vmin), float(vmax)
        if not (isfinite(lo) and isfinite(hi)):
            raise ValueError(f"Scale needs finite limits; got ({vmin!r}, {vmax!r})")
        if hi <= lo:
            hi = lo + 1.0
        return cls(lo, hi, missing=missing)

    @classmethod
    def categorical(
        cls,
        categories: Sequence[Any],
        colors: Sequence[str],
        *,
        missing: Optional[str] = None,
    ) -> "Scale":
        """Build a scale over unordered categories rather than a numeric range.

        Args:
            categories: The distinct values, in the order their colours were assigned.
            colors: One colour per category.
            missing: Colour for a value that is not one of `categories`.

        Returns:
            The categorical scale. Its numeric domain is the category *index* range, which is what a renderer
            needs to place a swatch; `vmin`/`vmax` are not meaningful as data values.

        Raises:
            ValueError: if there are no categories, or the counts do not match.

        Examples:
            - Three categories and their colours:
                ```python
                >>> from digitalearth.base.spec import Scale
                >>> scale = Scale.categorical(["a", "b", "c"], ["#f00", "#0f0", "#00f"])
                >>> scale.is_categorical, scale.color_for("b")
                (True, '#0f0')

                ```
        """
        cats = list(categories)
        cols = list(colors)
        if not cats:
            raise ValueError("a categorical Scale needs at least one category")
        if len(cats) != len(cols):
            raise ValueError(
                f"a categorical Scale needs one colour per category; got {len(cats)} categories "
                f"and {len(cols)} colours"
            )
        return cls(
            0.0,
            float(max(len(cats) - 1, 1)),
            categories=tuple(cats),
            missing=missing,
            _colors=tuple(cols),
        )

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _limits(
        values: Any, vmin: Optional[float], vmax: Optional[float]
    ) -> Tuple[float, float]:
        """Measure the domain, honouring explicit limits and widening a constant one.

        Args:
            values: The data to measure.
            vmin: Explicit lower limit, or ``None`` to measure.
            vmax: Explicit upper limit, or ``None`` to measure.

        Returns:
            A ``(lo, hi)`` pair with ``hi > lo`` guaranteed.
        """
        measured = finite(values)
        data_lo, data_hi = (
            (float(measured.min()), float(measured.max()))
            if measured.size
            else (0.0, 1.0)
        )
        lo = data_lo if vmin is None else float(vmin)
        hi = data_hi if vmax is None else float(vmax)
        if hi <= lo:
            # The rule all five copies chose: a constant domain is widened by one rather than divided by.
            hi = lo + 1.0
        return lo, hi

    @staticmethod
    def _breaks(values: Any, scheme: str, k: int) -> Tuple[float, ...]:
        """Cut class edges with the registered classifier, blaming the caller's arguments when it refuses.

        Args:
            values: The data to classify.
            scheme: Scheme name.
            k: Number of classes.

        Returns:
            The class edges, ``k + 1`` of them for ``k`` classes.

        Raises:
            ValueError: when the classifier refuses — an unknown scheme, a class count below one, or a
                constant column that cannot be cut. Three tiers each wrote their own version of this
                message; the classifier's own error names none of the three arguments, so they are
                named here.
            RuntimeError: if no classifier has been registered — see
                :func:`~digitalearth.base.registry.register_classifier`.
        """
        classify = get_classifier()
        numbers = np.asarray(values, dtype="float64")
        try:
            edges, _ = classify(numbers, scheme, k)
        except ValueError as err:
            raise ValueError(
                f"scheme={scheme!r} with k={k} cannot classify this data: {err}"
            ) from err
        return tuple(float(e) for e in edges)

    # ------------------------------------------------------------------ readers

    @property
    def is_classified(self) -> bool:
        """Whether this scale cuts classes rather than running a continuous ramp.

        Returns:
            ``True`` when a scheme produced class edges.
        """
        return bool(self.breaks)

    @property
    def is_categorical(self) -> bool:
        """Whether this scale colours unordered categories rather than a numeric range.

        Returns:
            ``True`` when categories were supplied.
        """
        return bool(self.categories)

    def as_limits(self) -> Tuple[float, float]:
        """Return the ``(vmin, vmax)`` pair a renderer's normaliser takes.

        Returns:
            The domain, with ``vmax > vmin`` guaranteed.
        """
        return self.vmin, self.vmax

    def color_for(self, category: Any) -> Optional[str]:
        """Return the colour assigned to one category.

        Args:
            category: The category to look up.

        Returns:
            Its colour, or :attr:`missing` when the scale never saw it.

        Examples:
            - An unseen category gets the missing colour, not a wrong one:
                ```python
                >>> from digitalearth.base.spec import Scale
                >>> scale = Scale.categorical(["a"], ["#f00"], missing="#ccc")
                >>> scale.color_for("zzz")
                '#ccc'

                ```
        """
        if category in self.categories:
            return self._colors[self.categories.index(category)]
        return self.missing

    def class_of(self, value: float) -> Optional[int]:
        """Return the 0-based class a value falls in, for a classified scale.

        Args:
            value: The value to place.

        Returns:
            Its class index, or ``None`` when the scale is not classified or the value is not finite. A value
            outside the edges is clamped to the first or last class rather than dropped, matching what a
            renderer does with a value outside its domain.

        Examples:
            - Three classes cut from ten values, and where 5 lands:
                ```python
                >>> from digitalearth.base.spec import Scale
                >>> scale = Scale.from_values(list(range(10)), scheme="equal_interval", k=3)
                >>> scale.class_of(5.0) in (0, 1, 2)
                True

                ```
        """
        if not self.is_classified or not isfinite(value):
            return None
        upper = list(self.breaks)[1:]
        for index, edge in enumerate(upper):
            if value <= edge:
                return index
        return len(upper) - 1

    def class_ranges(self) -> List[Tuple[float, float]]:
        """Return each class as a ``(low, high)`` pair, which is what a legend row shows.

        Returns:
            One pair per class, empty when the scale is not classified.

        Examples:
            - Two classes give two ranges:
                ```python
                >>> from digitalearth.base.spec import Scale
                >>> scale = Scale.from_values([0.0, 1.0, 2.0, 3.0], scheme="equal_interval", k=2)
                >>> len(scale.class_ranges())
                2

                ```
        """
        edges = list(self.breaks)
        return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    def freeze(self) -> "Scale":
        """Return a scale whose domain will not be re-derived from later data.

        `Scale` is already immutable, so this returns ``self``; it exists to say at a call site that the range
        is deliberately fixed — the thing a time sequence needs so its colours do not move between frames.

        Returns:
            This scale.

        Examples:
            - Freezing is explicit at the call site:
                ```python
                >>> from digitalearth.base.spec import Scale
                >>> Scale.from_limits(0.0, 1.0).freeze().as_limits()
                (0.0, 1.0)

                ```
        """
        return self
