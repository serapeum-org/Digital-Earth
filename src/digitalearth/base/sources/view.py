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

from typing import Any, Optional, Tuple

import numpy as np

from digitalearth.base.sources.dimension import DimensionInfo
from digitalearth.base.sources.source import Source
from digitalearth.base.spec.bounds import Bounds
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

        Examples:
            - The address a view was read from:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import DimensionInfo
                >>> from digitalearth.base.sources.view import SourceView
                >>> from digitalearth.base.spec import DataRef
                >>> axis = DimensionInfo(np.array([0.0]), "x")
                >>> SourceView(None, axis, axis, ref=DataRef("dem.tif")).ref.uri
                'dem.tif'

                ```
        """
        return self._ref

    @property
    def selection(self) -> Selection:
        """Which slice of the source this view holds.

        Returns:
            The selection. Defaults to the default band rather than ``None``, so a caller never has to guard
            before narrowing it.

        Examples:
            - A view with no selection still answers with one:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import DimensionInfo
                >>> from digitalearth.base.sources.view import SourceView
                >>> axis = DimensionInfo(np.array([0.0]), "x")
                >>> SourceView(None, axis, axis).selection.first_band
                1

                ```
        """
        return self._selection

    @property
    def request(self) -> Optional[ViewRequest]:
        """The request this view answered, or ``None`` when it was not read against one.

        Returns:
            The request, kept so a caller can compare what it asked for with what it got.

        Examples:
            - A view not read against a request says so:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import DimensionInfo
                >>> from digitalearth.base.sources.view import SourceView
                >>> axis = DimensionInfo(np.array([0.0]), "x")
                >>> SourceView(None, axis, axis).request is None
                True

                ```
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
                naming the view, rather than surfacing as an `AttributeError` from inside a reader. Build the
                view from a `DataRef`, or register the object with
                :meth:`~digitalearth.base.spec.dataref.DataRef.to_object`, to make it re-readable.

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
                RuntimeError: this SourceView has no DataRef, so it cannot be re-read

                ```
        """
        if self._ref is None:
            raise RuntimeError(
                "this SourceView has no DataRef, so it cannot be re-read"
            )
        data = self._ref.open()
        again = self.of(
            data,
            ref=self._ref,
            selection=self._selection,
            request=request,
            crs=self.crs,
        )
        # A windowed read returns a bare array, which carries no band name — so the variable, units and kind
        # would be lost by re-reading the very same slice at a different resolution. They describe the data,
        # not the window, so they are carried over where the new read supplied nothing.
        for key, value in self._meta.items():
            # `setdefault` is not enough: the extractor writes an *empty* variable name for a bare array, so
            # the key exists and would keep winning over the real one.
            if not again._meta.get(key):
                again._meta[key] = value
        if again.units is None:
            again._units = self.units
        return again

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
                object exposes one — including a budget with no region, which windows against the source's
                own extent. Otherwise recorded but not enforced: a reader that cannot window is not a reason
                to refuse the read.
            crs: The CRS to record for the coordinates, passed through to the extractor.

        Returns:
            The view.

        Raises:
            ValueError: if `selection` names more than a band. Only the band is honoured, so a view that
                stored `time`, `level`, `member` or `overview` would report a slice it does not hold; a
                composite selection is refused for the same reason.
            Exception: whatever the extractor raises for data it cannot read.

        Examples:
            - A plain array becomes a view with no address, which is a legitimate state:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources.view import SourceView
                >>> view = SourceView.of(np.arange(6.0).reshape(2, 3), crs=4326)
                >>> view.z.values.shape, view.rereadable
                ((2, 3), False)

                ```
        """
        from digitalearth.base.sources import get_source

        picked = selection if selection is not None else Selection()
        # Only the band is honoured today: the extractor takes `band=` and nothing else, and the overview is
        # chosen by read_part from the requested size. Storing a selection whose other axes were silently
        # dropped would let a caller read `.selection` back and believe the view holds a slice it does not.
        ignored = [
            name
            for name in ("time", "level", "member", "overview")
            if getattr(picked, name) is not None
        ]
        if ignored:
            raise ValueError(
                f"SourceView cannot yet honour {ignored} on a Selection — only the band is applied, and a "
                "view that stored the rest would report a slice it does not hold. Narrow the data before "
                "building the view, or track the gap in the data tier"
            )
        if picked.is_composite:
            raise ValueError(
                f"SourceView reads one band; got {len(picked.band)}. Use Selection.frames() and build a "
                "view per channel"
            )
        windowed, xs, ys, window_crs = cls._windowed(data, picked, request)
        source = get_source(
            windowed,
            band=picked.first_band,
            x=xs,
            y=ys,
            crs=window_crs if window_crs is not None else crs,
        )
        return cls(
            source.z,
            source.x,
            source.y,
            source.crs,
            # The extractor's metadata is passed through whole. Rebuilding it as {"variable": ...} dropped
            # `kind` and `standard_name`, and autostyle reads standard_name for the ECMWF-Magics match — so a
            # view silently lost the identity match this wave's auto_cmap consolidation exists to keep.
            dict(source._meta),
            source.units,
            ref=ref,
            selection=picked,
            request=request,
        )

    @staticmethod
    def _shape(request: ViewRequest) -> Tuple[int, int]:
        """Return the ``(width, height)`` in cells a windowed read should produce.

        Args:
            request: What was asked for.

        Returns:
            The caller's canvas when they named both dimensions **and it fits the budget** — reading a square
            for a request that says ``800x600`` distorts the aspect ratio of the very canvas those fields
            exist to describe. Otherwise a square of :meth:`~digitalearth.base.spec.viewrequest.ViewRequest.side`,
            which is what a budget alone can say.

            A canvas over budget is scaled down keeping its aspect ratio, because the budget is the limit the
            process can actually afford and the canvas is only what would look best.
        """
        width, height = request.width, request.height
        if width is None or height is None:
            if request.budget is None:
                side = request.side()
            else:
                # The budget wins over `side()`'s readability floor. Its own docstring calls it the limit "a
                # reader that cannot serve both must respect", and a floor of 64 turned a budget of 16 into
                # 4,096 cells. A caller who sets a budget that small has asked for a thumbnail.
                side = max(1, int(request.budget**0.5))
            return side, side
        if request.within_budget(width * height):
            return width, height
        scale = (request.budget / (width * height)) ** 0.5  # type: ignore[operator]
        return max(1, int(width * scale)), max(1, int(height * scale))

    @classmethod
    def _windowed(
        cls, data: Any, selection: Selection, request: Optional[ViewRequest]
    ) -> Tuple[Any, Optional[Any], Optional[Any], Optional[Any]]:
        """Narrow `data` to the requested window, and say where the window's cells are.

        Args:
            data: The opened object.
            selection: Which slice is wanted.
            request: What was asked for, or ``None``.

        Returns:
            A ``(data, x, y, crs)`` tuple. Unwindowed, that is `data` unchanged and three ``None``s — there
            is nothing to narrow, or the reader has no windowed read.

            Windowed, it is the **array** ``read_part`` returns plus the coordinates of its cell centres and
            the CRS they are in. The coordinates have to be computed here because ``read_part`` is an array
            reader, not a dataset reader: pyramids' own docstring says *"Pixel values only — no transform,
            bounds, or CRS is attached"*. Without them the extractor falls back to ``np.arange`` axes and the
            view claims a projected CRS while holding pixel indices.

        Raises:
            Exception: whatever pyramids raises for a window it cannot read.
        """
        nothing = (data, None, None, None)
        if request is None or not hasattr(data, "read_part"):
            return nothing
        bbox = request.as_bbox()
        if bbox is None:
            # A budget with no region still has to be honoured: the object *can* window, so skipping here
            # returned the whole raster and blew the budget silently. The source's own bbox is the region.
            source_bbox = getattr(data, "bbox", None)
            if source_bbox is None or request.budget is None:
                return nothing
            window = Bounds.from_bbox(
                list(source_bbox), crs=getattr(data, "epsg", None)
            )
        else:
            # read_part returns data in the *dataset's* CRS, so a bbox given in another one is converted
            # first and the window described in the CRS its cells are actually measured in.
            window = Bounds.from_bbox(list(bbox), crs=request.crs)
            target = getattr(data, "epsg", None) or getattr(data, "crs", None)
            if request.crs is not None and target is not None:
                window = window.to_crs(target)
        # Snap the window outward to whole source pixels *before* reading. read_part does this internally
        # (floor/ceil through world_to_pixel) and returns a buffer spanning the snapped window — so labelling
        # the result from the requested bbox misplaces every cell by up to one source pixel per edge, worst
        # on exactly the zoomed-out tiles this exists for. Aligning the request makes the snap a no-op, so
        # the window asked for and the window read are the same rectangle.
        window = cls._aligned(window, data)
        width, height = cls._shape(request)
        array = np.asarray(
            data.read_part(
                bbox=window.as_bbox(),
                dst_width=width,
                dst_height=height,
                bbox_crs=window.crs,
                band=selection.first_band - 1,  # pyramids' windowed read is 0-based
            )
        )
        rows, columns = array.shape[-2], array.shape[-1]
        xmin, ymin, xmax, ymax = window.as_bbox()
        # Cell centres, not edges: an extractor's axes name where each cell *is*, and a half-cell offset is
        # the difference between a raster drawn correctly and one drawn half a pixel adrift.
        xs = xmin + (np.arange(columns) + 0.5) * (xmax - xmin) / columns
        ys = ymax - (np.arange(rows) + 0.5) * (ymax - ymin) / rows
        return array, xs, ys, window.crs

    @staticmethod
    def _aligned(window: Bounds, data: Any) -> Bounds:
        """Grow `window` outward to the source's pixel edges, so a read of it needs no snapping.

        Args:
            window: The window wanted, in the source's CRS.
            data: The object being read, for its geotransform.

        Returns:
            The smallest pixel-aligned window containing `window`, or `window` unchanged when the source
            exposes no geotransform to align against.

            This is the same outward snap ``read_part`` performs internally. Doing it here rather than
            reproducing its arithmetic afterwards is what keeps the labels honest: the caller's bbox and the
            rectangle actually read become one rectangle, so there is no second window to get wrong.

            The long-term answer is for the reader to return the window's transform alongside the array —
            an upstream change, not one to make here.
        """
        transform = getattr(data, "geotransform", None)
        if not transform:
            return window
        origin_x, pixel_w, _, origin_y, _, pixel_h = transform[:6]
        if not pixel_w or not pixel_h:
            return window
        height = abs(pixel_h)
        xmin, ymin, xmax, ymax = window.as_bbox()
        left = origin_x + np.floor((xmin - origin_x) / pixel_w) * pixel_w
        right = origin_x + np.ceil((xmax - origin_x) / pixel_w) * pixel_w
        # The geotransform's y step is negative for a north-up raster, so rows are measured down from the
        # origin; align against that and convert back.
        top = origin_y - np.floor((origin_y - ymax) / height) * height
        bottom = origin_y - np.ceil((origin_y - ymin) / height) * height
        return Bounds(float(left), float(bottom), float(right), float(top), window.crs)
