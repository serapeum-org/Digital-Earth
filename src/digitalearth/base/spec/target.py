"""Where a figure is going, and so how much of each source it may read.

The read budget used to belong to whoever made the call. `interactive/raster.py`'s `large_image` sets its own
default, `max_pixels=4_000_000`; the interactive tier's `big_data_threshold` is a per-map attribute; an exported
page and a live window get the same budget. The design's rule is the other way round — **the budget belongs to the
target, never to the layer**: a page shared as HTML carries its pixels inline (#189) and should be lighter than a
window a reader pans around.

:class:`RenderTarget` is that target, and :meth:`RenderTarget.view_request` is the one place a
:class:`~digitalearth.base.spec.viewrequest.ViewRequest` is made from a view: the region from the view, the canvas and
the budget from the target.
"""

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Union

from digitalearth.base.spec._serial import positive_number, refuse_unknown
from digitalearth.base.spec.bounds import Bounds
from digitalearth.base.spec.viewport import Camera, Viewport
from digitalearth.base.spec.viewrequest import ViewRequest

__all__ = ["DEFAULT_BUDGETS", "RenderTarget", "TARGET_KINDS"]

#: The outputs a figure can be rendered to.
TARGET_KINDS = ("window", "html", "image", "batch")

#: Cells a target reads per source when it names no budget of its own. ``window`` is the budget the package already
#: uses for a live view — ``large_image``'s ``max_pixels``. ``html`` is a quarter of that, because a page embeds its
#: pixels rather than reading them on demand. ``image`` and ``batch`` write a still once and take the window's budget;
#: a still with a named canvas is sized from the canvas anyway, within the budget.
DEFAULT_BUDGETS: Mapping[str, int] = MappingProxyType(
    {"window": 4_000_000, "html": 1_000_000, "image": 4_000_000, "batch": 4_000_000}
)


def _positive_whole(name: str, value: Any) -> Optional[int]:
    """Return `value` as a positive int, or ``None`` when unset.

    Args:
        name: The field, for the message.
        value: The candidate.

    Returns:
        The value as a Python int, or ``None``.

    Raises:
        ValueError: for a boolean, a non-integer (a whole float such as `2.0` included) or a value below one.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"RenderTarget {name} must be a whole number; got {value!r}")
    if value < 1:
        raise ValueError(f"RenderTarget {name} must be positive; got {value}")
    return int(value)


@dataclass(frozen=True)
class RenderTarget:
    """An output a figure is rendered to: its kind, its canvas and its read budget.

    Attributes:
        kind: One of :data:`TARGET_KINDS`.
        width: Canvas width in CSS pixels, or ``None``.
        height: Canvas height in CSS pixels, or ``None``.
        pixel_ratio: Device pixels per CSS pixel — ``2.0`` on a HiDPI display.
        budget: Cells the target may read per source. ``None`` takes :data:`DEFAULT_BUDGETS` for the kind.

    Raises:
        ValueError: for an unknown kind (naming the kinds that exist), a non-positive or non-integer canvas or
            budget, or a pixel ratio that is a boolean, not a number, not finite or not positive.

    Examples:
        - A page is lighter than a window by default:
            ```python
            >>> from digitalearth.base.spec import RenderTarget
            >>> RenderTarget("window").effective_budget, RenderTarget("html").effective_budget
            (4000000, 1000000)

            ```
        - A whole-number pixel ratio is stored as a float:
            ```python
            >>> from digitalearth.base.spec import RenderTarget
            >>> RenderTarget("image", width=1200, height=800, pixel_ratio=2).pixel_ratio
            2.0

            ```
        - An unknown kind is refused, naming the kinds that exist:
            ```python
            >>> from digitalearth.base.spec import RenderTarget
            >>> RenderTarget("pdf")
            Traceback (most recent call last):
                ...
            ValueError: RenderTarget kind must be one of ['window', 'html', 'image', 'batch']; got 'pdf'

            ```
    """

    kind: str = "window"
    width: Optional[int] = None
    height: Optional[int] = None
    pixel_ratio: float = 1.0
    budget: Optional[int] = None

    def __post_init__(self) -> None:
        """Refuse a target nothing could be rendered to.

        Raises:
            ValueError: as described on the class.
        """
        if self.kind not in TARGET_KINDS:
            raise ValueError(
                f"RenderTarget kind must be one of {list(TARGET_KINDS)}; got {self.kind!r}"
            )
        for name in ("width", "height", "budget"):
            object.__setattr__(self, name, _positive_whole(name, getattr(self, name)))
        ratio = positive_number(self.pixel_ratio)
        if ratio is None:
            raise ValueError(
                f"RenderTarget pixel_ratio must be a positive number; got {self.pixel_ratio!r}"
            )
        object.__setattr__(self, "pixel_ratio", ratio)

    @property
    def effective_budget(self) -> int:
        """The cells this target reads per source.

        Returns:
            `budget` when set, otherwise the kind's entry in :data:`DEFAULT_BUDGETS`.

        Examples:
            - An explicit budget wins over the kind's default:
                ```python
                >>> from digitalearth.base.spec import RenderTarget
                >>> RenderTarget("html", budget=250_000).effective_budget
                250000

                ```
            - With no budget of its own, each kind reads its entry in `DEFAULT_BUDGETS`:
                ```python
                >>> from digitalearth.base.spec import TARGET_KINDS, RenderTarget
                >>> {kind: RenderTarget(kind).effective_budget for kind in TARGET_KINDS}
                {'window': 4000000, 'html': 1000000, 'image': 4000000, 'batch': 4000000}

                ```
        """
        return self.budget if self.budget is not None else DEFAULT_BUDGETS[self.kind]

    def view_request(
        self,
        view: Optional[Union[Viewport, Camera]] = None,
        *,
        bounds: Optional[Bounds] = None,
    ) -> ViewRequest:
        """Return the request a reader answers for this view on this target.

        The region comes from the view and the canvas and budget from the target — the only way this package makes a
        :class:`~digitalearth.base.spec.viewrequest.ViewRequest` from a view, so a layer never chooses its own budget.

        Args:
            view: The view being rendered. A `Viewport` supplies its framed region; a `Camera` supplies none, since a
                3-D view has no rectangle to read.
            bounds: A region to read instead of the view's. When a `Viewport` is given, the region is reprojected into
                the view's CRS, which is the CRS the reader is asked in; with a `Camera` or no view it is used as
                given.

        Returns:
            The request: region, canvas, pixel ratio and :attr:`effective_budget`. With a canvas named, the
            request's :meth:`~digitalearth.base.spec.viewrequest.ViewRequest.side` sizes a read to the canvas,
            within the budget. The region is `None` when neither the view nor `bounds` supplies one — an unframed
            `Viewport` included — and a request with no region carries no CRS either: the reader returns the source
            in its own CRS, and the renderer reprojects it into the view's.

        Raises:
            ValueError: if `view` is neither a `Viewport`, a `Camera` nor `None`, if `bounds` is not a `Bounds`, or
                if `bounds` cannot be reprojected into the viewport's CRS.

        Examples:
            - The view's region, the target's canvas and budget:
                ```python
                >>> from digitalearth.base.spec import Bounds, RenderTarget, Viewport
                >>> view = Viewport(4326, bounds=Bounds(0.0, 0.0, 10.0, 5.0, crs=4326))
                >>> request = RenderTarget("html", width=800, height=400).view_request(view)
                >>> request.as_bbox(), request.width, request.budget
                ((0.0, 0.0, 10.0, 5.0), 800, 1000000)

                ```
            - A region given in another CRS is asked for in the view's:
                ```python
                >>> from digitalearth.base.spec import Bounds, RenderTarget, Viewport
                >>> request = RenderTarget().view_request(Viewport(3857), bounds=Bounds(0.0, 0.0, 1.0, 1.0, crs=4326))
                >>> request.crs, [round(edge) for edge in request.as_bbox()]
                (3857, [0, 0, 111319, 111325])

                ```
            - A 3-D camera has no rectangle, so only the canvas and the budget are set:
                ```python
                >>> from digitalearth.base.spec import Camera, RenderTarget
                >>> request = RenderTarget("image", width=640, height=480).view_request(Camera((0.0, -10.0, 5.0)))
                >>> request.bounds, request.width, request.budget
                (None, 640, 4000000)

                ```
        """
        if view is not None and not isinstance(view, (Viewport, Camera)):
            raise ValueError(
                f"RenderTarget.view_request needs a Viewport, a Camera or None; got {type(view).__name__}"
            )
        if bounds is not None and not isinstance(bounds, Bounds):
            raise ValueError(
                f"RenderTarget.view_request needs bounds as a Bounds; got {type(bounds).__name__}"
            )
        region: Optional[Bounds] = None
        if isinstance(view, Viewport):
            region = view.framed(bounds).bounds if bounds is not None else view.bounds
        elif bounds is not None:
            region = bounds
        return ViewRequest(
            bounds=region,
            width=self.width,
            height=self.height,
            pixel_ratio=self.pixel_ratio,
            budget=self.effective_budget,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the plain-dict form a figure stores.

        Returns:
            ``kind`` and ``pixel_ratio``, plus the canvas and the budget when set. An unset budget is written as
            unset, not as the kind's default, so a stored target keeps following the default if the default changes.

        Examples:
            - A retina image target:
                ```python
                >>> from digitalearth.base.spec import RenderTarget
                >>> RenderTarget("image", width=1200, height=800, pixel_ratio=2.0).to_dict()
                {'kind': 'image', 'width': 1200, 'height': 800, 'pixel_ratio': 2.0}

                ```
            - The pixel ratio is written even at its default, and an unset budget stays unset:
                ```python
                >>> from digitalearth.base.spec import RenderTarget
                >>> RenderTarget("html").to_dict()
                {'kind': 'html', 'pixel_ratio': 1.0}

                ```
        """
        out: Dict[str, Any] = {"kind": self.kind}
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        out["pixel_ratio"] = self.pixel_ratio
        if self.budget is not None:
            out["budget"] = self.budget
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RenderTarget":
        """Rebuild a target from its dict form.

        Args:
            data: A mapping as produced by :meth:`to_dict`. A missing ``kind`` means ``"window"``.

        Returns:
            The target, validated as the constructor validates it.

        Raises:
            TypeError: if `data` is not a mapping.
            ValueError: for an unknown key, or a target the constructor refuses.

        Examples:
            - A stored page target reads back with its budget:
                ```python
                >>> from digitalearth.base.spec import RenderTarget
                >>> RenderTarget.from_dict({"kind": "html", "budget": 500000}).effective_budget
                500000

                ```
            - A key this version does not know is refused rather than dropped:
                ```python
                >>> from digitalearth.base.spec import RenderTarget
                >>> RenderTarget.from_dict({"kind": "html", "dpi": 2})  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                ValueError: RenderTarget.from_dict got unknown keys ['dpi']; known keys are ['budget', 'height', ...]

                ```
        """
        refuse_unknown(
            "RenderTarget", data, ("kind", "width", "height", "pixel_ratio", "budget")
        )
        return cls(
            kind=data.get("kind", "window"),
            width=data.get("width"),
            height=data.get("height"),
            pixel_ratio=data.get("pixel_ratio", 1.0),
            budget=data.get("budget"),
        )
