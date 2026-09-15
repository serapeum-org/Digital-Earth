"""A `Source` that remembers the address it was read from — so it can be read again.

A `Source` carries the materialised data and forgets where it came from. That single omission is what blocks
dynamic tiling (#189), level of detail on the 3-D tier (#207) and point clouds too large to hold (#206): the
only way to get different pixels is to start again from whatever object the caller originally passed.

It is already being worked around. `interactive/raster.py`'s `large_image` drives its own viewport loop —
sizing the canvas, choosing the overview, calling pyramids' `read_part`/`preview` — and nothing else can reuse
any of it, because what it produces is a plain `Source` with no memory of the read.

`SourceView` is that memory: the same data, plus the :class:`~digitalearth.base.spec.dataref.DataRef` and
:class:`~digitalearth.base.spec.selection.Selection` that produced it. With those,
:meth:`SourceView.reread` can ask for the same slice at another resolution or over another window.

It **subclasses** `Source` rather than replacing it. Every existing consumer across the four tiers takes a
`Source` and keeps working unchanged; the address is additive. Replacing `Source` outright would have meant
touching every reader in the package to gain a capability none of them uses yet.
"""

from typing import Any, Optional

from digitalearth.base.sources.dimension import DimensionInfo
from digitalearth.base.sources.source import Source
from digitalearth.base.spec.dataref import DataRef
from digitalearth.base.spec.selection import Selection
from digitalearth.base.spec.viewrequest import ViewRequest

__all__ = ["SourceView"]


class SourceView(Source):
    """A materialised view of data, carrying the reference and slice it was read from.

    Args:
        z: The data dimension, as :class:`~digitalearth.base.sources.source.Source` takes it.
        x: The x / longitude axis.
        y: The y / latitude axis.
        crs: The CRS the coordinates are expressed in.
        metadata: Free-form metadata.
        units: Unit string for the values.
        ref: Where the data came from. ``None`` for a view built from an object the caller held, which can
            still be read but not re-read.
        selection: Which slice was read.
        request: The request this view answered, when it answered one. Kept so a caller can tell what it
            asked for from what it got — a reader may return fewer cells than the budget allowed.

    Examples:
        - A view knows its own address:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import DimensionInfo
            >>> from digitalearth.base.sources.view import SourceView
            >>> from digitalearth.base.spec import DataRef, Selection
            >>> axis = DimensionInfo(np.array([0.0, 1.0]), "x")
            >>> view = SourceView(None, axis, axis, crs=4326,
            ...                   ref=DataRef("data/dem.tif"), selection=Selection.of(2))
            >>> view.ref.uri, view.selection.first_band
            ('data/dem.tif', 2)

            ```
        - And is still an ordinary `Source`, so every existing consumer takes it:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import DimensionInfo, Source
            >>> from digitalearth.base.sources.view import SourceView
            >>> axis = DimensionInfo(np.array([0.0]), "x")
            >>> isinstance(SourceView(None, axis, axis, crs=3857), Source)
            True

            ```
    """

    def __init__(
        self,
        z: Optional[DimensionInfo],
        x: DimensionInfo,
        y: DimensionInfo,
        crs: Any = None,
        metadata: Optional[dict] = None,
        units: Optional[str] = None,
        *,
        ref: Optional[DataRef] = None,
        selection: Optional[Selection] = None,
        request: Optional[ViewRequest] = None,
    ):
        super().__init__(z, x, y, crs, metadata, units)
        self._ref = ref
        self._selection = selection if selection is not None else Selection()
        self._request = request

    @property
    def ref(self) -> Optional[DataRef]:
        """Where this view's data came from, or ``None`` when it was built from a held object.

        Returns:
            The reference. ``None`` is not a defect — a caller may pass a `Dataset` they built in a notebook,
            and that view is perfectly usable; it just cannot be re-read from an address it never had.
        """
        return self._ref

    @property
    def selection(self) -> Selection:
        """Which slice of the source this view holds.

        Returns:
            The selection. Defaults to the default band rather than ``None``, so a caller never has to guard
            before narrowing it.
        """
        return self._selection

    @property
    def request(self) -> Optional[ViewRequest]:
        """The request this view answered, or ``None`` when it was not read against one.

        Returns:
            The request, kept so a caller can compare what it asked for with what it got.
        """
        return self._request

    @property
    def rereadable(self) -> bool:
        """Whether this view can be read again.

        Returns:
            ``True`` when the view carries a reference. A view built from a held object answers ``False``,
            and :meth:`reread` says so rather than failing deep inside a reader.

        Examples:
            - A view with no reference knows it cannot be re-read:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import DimensionInfo
                >>> from digitalearth.base.sources.view import SourceView
                >>> axis = DimensionInfo(np.array([0.0]), "x")
                >>> SourceView(None, axis, axis).rereadable
                False

                ```
        """
        return self._ref is not None

    def reread(self, request: ViewRequest) -> "SourceView":
        """Read the same slice again, for a different region or resolution.

        This is what the address is for. The view re-opens its own
        :class:`~digitalearth.base.spec.dataref.DataRef`, applies its own
        :class:`~digitalearth.base.spec.selection.Selection`, and asks the reader for what `request` wants —
        the loop `interactive/raster.py` currently drives by hand at one call site.

        Args:
            request: The region, resolution and budget wanted.

        Returns:
            A new view of the same slice, carrying the same reference and selection plus the request it
            answered.

        Raises:
            RuntimeError: if this view has no reference to read from — see :attr:`rereadable`. Raised here,
                naming the view, rather than surfacing as an `AttributeError` from inside a reader.

        Examples:
            - A view built from a held object refuses, and says why:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import DimensionInfo
                >>> from digitalearth.base.sources.view import SourceView
                >>> from digitalearth.base.spec import ViewRequest
                >>> axis = DimensionInfo(np.array([0.0]), "x")
                >>> SourceView(None, axis, axis).reread(ViewRequest(budget=100))
                Traceback (most recent call last):
                    ...
                RuntimeError: this SourceView carries no DataRef, so it cannot be re-read. It was built from an object the caller held rather than from an address; register it with DataRef.to_object() to make it re-readable

                ```
        """
        if self._ref is None:
            raise RuntimeError(
                "this SourceView carries no DataRef, so it cannot be re-read. It was built from an object "
                "the caller held rather than from an address; register it with DataRef.to_object() to make "
                "it re-readable"
            )
        data = self._ref.open()
        return self.of(
            data,
            ref=self._ref,
            selection=self._selection,
            request=request,
            crs=self.crs,
        )

    @classmethod
    def of(
        cls,
        data: Any,
        *,
        ref: Optional[DataRef] = None,
        selection: Optional[Selection] = None,
        request: Optional[ViewRequest] = None,
        crs: Any = None,
    ) -> "SourceView":
        """Materialise a view from an opened object, honouring the request where the reader supports it.

        Args:
            data: The opened data — a pyramids ``Dataset``, ``FeatureCollection``, or anything
                :func:`~digitalearth.base.sources.get_source` accepts.
            ref: The reference `data` was opened from, recorded so the view can be re-read.
            selection: Which slice to read. Defaults to the default band.
            request: The region/resolution/budget wanted. Applied through pyramids' windowed read when the
                object exposes one, and otherwise recorded but not enforced — a reader that cannot window is
                not a reason to refuse the read.
            crs: The CRS to record for the coordinates, passed through to the extractor.

        Returns:
            The view.

        Raises:
            Exception: whatever the extractor raises for data it cannot read.
        """
        from digitalearth.base.sources import get_source

        picked = selection if selection is not None else Selection()
        windowed = cls._windowed(data, picked, request)
        source = get_source(windowed, band=picked.first_band, crs=crs)
        return cls(
            source.z,
            source.x,
            source.y,
            source.crs,
            {"variable": source.metadata("variable")}
            if source.metadata("variable")
            else None,
            source.units,
            ref=ref,
            selection=picked,
            request=request,
        )

    @staticmethod
    def _windowed(
        data: Any, selection: Selection, request: Optional[ViewRequest]
    ) -> Any:
        """Narrow `data` to the requested window, when the reader can do it.

        Args:
            data: The opened object.
            selection: Which slice is wanted.
            request: What was asked for, or ``None``.

        Returns:
            The narrowed object, or `data` unchanged when there is nothing to narrow or no reader support.
            pyramids owns the windowing (``read_part``); this only decides whether to ask for it, which is
            the half `interactive/raster.py` does inline.
        """
        if request is None or not hasattr(data, "read_part"):
            return data
        bbox = request.as_bbox()
        if bbox is None:
            return data
        side = request.side()
        return data.read_part(
            bbox=bbox,
            dst_width=side,
            dst_height=side,
            bbox_crs=request.crs(),
            band=selection.first_band - 1,  # pyramids' windowed read is 0-based
        )
