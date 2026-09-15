"""What a renderer is asking for — the region, at the resolution, within the budget.

Three of the four backend studies invented this type independently: the interactive tier called it
`ViewRequest`, the 3-D tier `BuildContext`, and the web tier folded it into its source resolver's request.
That is the usual sign a type is missing rather than optional.

It is what makes :class:`~digitalearth.base.spec.dataref.DataRef` useful. A reference says *where* data is and
a :class:`~digitalearth.base.spec.selection.Selection` says *which slice*; neither says how much of it to read
or how finely. Without that third question a view can only ever be materialised one way — which is why the
one existing windowed-overview path (`interactive/raster.py`) had to drive its own loop, sizing the canvas and
choosing the overview at the call site.

The design doc lists this in Wave 6 beside `RenderTarget`. It comes forward here because
:meth:`~digitalearth.base.sources.view.SourceView.reread` has no signature without it; `RenderTarget`, which
decides *which* request a given output wants, stays there.
"""

from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Integral
from typing import Any, Optional, Tuple

from digitalearth.base.spec.bounds import Bounds

__all__ = ["ViewRequest"]


@dataclass(frozen=True)
class ViewRequest:
    """A request for a region of data at a resolution, bounded by a budget.

    Attributes:
        bounds: The region wanted, in its own CRS — a reader reprojects if it must. ``None`` asks for
            whatever the source covers.
        width: Target width in pixels, or ``None``.
        height: Target height in pixels, or ``None``.
        pixel_ratio: Device pixel ratio, so a HiDPI canvas can ask for the pixels it will actually draw.
        budget: Greatest number of cells the reader may return. This is the honest limit — `width`/`height`
            say what is wanted, `budget` says what the process can afford, and a reader that cannot serve
            both must respect this one.

    Raises:
        ValueError: if a dimension or the budget is not a positive finite number. A zero-width request reads
            nothing and a negative one is a computed value that went wrong upstream; both are worth catching
            where they are written rather than inside a reader.

    Examples:
        - A request for a region at a canvas size:
            ```python
            >>> from digitalearth.base.spec import Bounds, ViewRequest
            >>> req = ViewRequest(bounds=Bounds(0.0, 0.0, 10.0, 10.0, crs=4326), width=800, height=600)
            >>> req.pixels
            480000

            ```
        - A budget alone, with no canvas — what a decimating reader needs:
            ```python
            >>> from digitalearth.base.spec import ViewRequest
            >>> ViewRequest(budget=1_000_000).side()
            1000

            ```
    """

    bounds: Optional[Bounds] = None
    width: Optional[int] = None
    height: Optional[int] = None
    pixel_ratio: float = 1.0
    budget: Optional[int] = None

    def __post_init__(self) -> None:
        """Refuse a request that asks for nothing, or for a negative amount of something.

        Raises:
            ValueError: for a non-positive or non-finite dimension, ratio or budget.
        """
        for name in ("width", "height", "budget"):
            value = getattr(self, name)
            if value is None:
                continue
            # Integral, not int, and coerced — exactly as Selection does, and for the same reason: canvas
            # sizes and cell budgets arrive from numpy arithmetic, and these two types are meant to be used
            # in the same call. Refusing np.int64 here while Selection accepts it is one vocabulary
            # disagreeing with itself.
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(
                    f"ViewRequest {name} must be a whole number of pixels or cells; got {value!r}"
                )
            if value <= 0:
                raise ValueError(f"ViewRequest {name} must be positive; got {value}")
            object.__setattr__(self, name, int(value))
        if not isfinite(self.pixel_ratio) or self.pixel_ratio <= 0:
            raise ValueError(
                f"ViewRequest pixel_ratio must be a positive number; got {self.pixel_ratio!r}"
            )

    @property
    def pixels(self) -> Optional[int]:
        """How many device pixels the request covers, or ``None`` when it names no canvas.

        Returns:
            ``width * height``, scaled by :attr:`pixel_ratio` — the count a reader compares against
            :attr:`budget`.

        Examples:
            - A retina canvas asks for the pixels it will actually draw:
                ```python
                >>> from digitalearth.base.spec import ViewRequest
                >>> ViewRequest(width=800, height=600).pixels
                480000
                >>> ViewRequest(width=800, height=600, pixel_ratio=2.0).pixels
                1920000

                ```
        """
        if self.width is None or self.height is None:
            return None
        return int(self.width * self.height * self.pixel_ratio * self.pixel_ratio)

    def side(self, floor: int = 64) -> int:
        """Return the square side a decimating reader should aim for.

        Args:
            floor: Smallest side to return. A budget small enough to ask for a handful of cells produces an
                image nothing can be read from, so there is a lower bound.

        Returns:
            The larger of `floor` and the square root of the effective cell allowance — the canvas sizing
            `interactive/raster.py` does by hand as ``max(64, int(sqrt(max_pixels)))``. With neither a
            budget nor a canvas, `floor` is the answer.

        Examples:
            - The budget sets the side, and a tiny one still floors:
                ```python
                >>> from digitalearth.base.spec import ViewRequest
                >>> ViewRequest(budget=1_000_000).side(), ViewRequest(budget=4).side()
                (1000, 64)

                ```
        """
        allowance = self.budget if self.budget is not None else self.pixels
        if allowance is None:
            return floor
        return max(floor, int(sqrt(allowance)))

    def within_budget(self, cells: int) -> bool:
        """Whether a read of `cells` cells is affordable.

        Args:
            cells: How many cells the read would return.

        Returns:
            ``True`` when there is no budget, or the read fits inside it.

        Examples:
            - No budget affords anything:
                ```python
                >>> from digitalearth.base.spec import ViewRequest
                >>> ViewRequest().within_budget(10 ** 9)
                True
                >>> ViewRequest(budget=100).within_budget(101)
                False

                ```
        """
        return self.budget is None or cells <= self.budget

    def as_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Return the requested region as the bbox tuple a pyramids reader takes.

        Returns:
            ``(xmin, ymin, xmax, ymax)``, or ``None`` when the request names no region. The ordering comes
            from :meth:`~digitalearth.base.spec.bounds.Bounds.as_bbox` rather than being written out here.

        Examples:
            - A region in bbox order, and no region at all:
                ```python
                >>> from digitalearth.base.spec import Bounds, ViewRequest
                >>> ViewRequest(bounds=Bounds(0.0, 1.0, 2.0, 3.0, crs=4326)).as_bbox()
                (0.0, 1.0, 2.0, 3.0)
                >>> ViewRequest(budget=10).as_bbox() is None
                True

                ```
        """
        if self.bounds is None:
            return None
        xmin, ymin, xmax, ymax = self.bounds.as_bbox()
        return (xmin, ymin, xmax, ymax)

    def crs(self) -> Any:
        """Return the CRS the requested region is expressed in.

        Returns:
            The bounds' CRS, or ``None`` when the request names no region.

        Examples:
            - The CRS travels with the region:
                ```python
                >>> from digitalearth.base.spec import Bounds, ViewRequest
                >>> ViewRequest(bounds=Bounds(0.0, 0.0, 1.0, 1.0, crs=3857)).crs()
                3857

                ```
        """
        return None if self.bounds is None else self.bounds.crs
