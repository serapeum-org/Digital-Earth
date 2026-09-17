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

from dataclasses import replace
from typing import Any, Optional, Tuple, cast

import numpy as np

from digitalearth.base.sources.dimension import DimensionInfo
from digitalearth.base.sources.source import Source
from digitalearth.base.spec.bounds import Bounds
from digitalearth.base.spec.dataref import DataRef
from digitalearth.base.spec.selection import Selection
from digitalearth.base.spec.viewrequest import ViewRequest

__all__ = ["DESCRIBING_KEYS", "SourceView"]

#: Metadata keys that describe the data rather than the read, and so survive a re-read of the same slice. A
#: windowed read comes back as a bare array, for which the extractor writes only `kind="raster"` and an empty
#: `variable` placeholder, so the band's name and `standard_name` would otherwise be lost. Everything *not*
#: listed here belongs to the read that produced it and is never filled in from an earlier one.
DESCRIBING_KEYS = ("variable", "kind", "standard_name")


def _off_source(error: BaseException) -> bool:
    """Whether a failed windowed read means the window missed the data.

    Args:
        error: What the reader raised.

    Returns:
        `True` for pyramids' out-of-bounds report, matched by name so this module needs no import of it — the
        exception moved package once already, and a viewport that pans off the edge must not depend on where
        it lives.
    """
    if "OutOfBounds" in type(error).__name__:
        return True
    return "out of bounds" in str(error).lower()


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
            answered. Its metadata is the new read's, with each key in :data:`DESCRIBING_KEYS` the new read
            did not supply filled in from this view — see :meth:`_carried` — and its `units` are this view's
            when the new read found none.

        Raises:
            RuntimeError: if this view has no reference to read from — see :attr:`rereadable`. Raised here,
                naming the view, rather than surfacing as an `AttributeError` from inside a reader. Build the
                view from a `DataRef`, or register the object with
                :meth:`~digitalearth.base.spec.dataref.DataRef.to_object`, to make it re-readable.
            KeyError: if the reference no longer resolves — an `object:` id not registered in this process,
                or a scheme no resolver handles. Raised by
                :meth:`~digitalearth.base.spec.dataref.DataRef.open`.
            ValueError: if this view's selection is one :meth:`of` refuses, or the source is a rotated raster,
                which no window can be labelled for.

        Examples:
            - Re-reading within a budget keeps the address, and the band name the bare windowed array lacks:
                ```python
                >>> import numpy as np
                >>> from pyramids.base.georeference import GeoReference
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.base.sources.view import SourceView
                >>> from digitalearth.base.spec import DataRef, ViewRequest
                >>> dem = Dataset.from_array(
                ...     np.arange(16.0).reshape(4, 4),
                ...     geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
                ... )
                >>> ref = DataRef.to_object(dem, name="doc-reread")
                >>> view = SourceView.of(ref.open(), ref=ref)
                >>> view.z.values.shape, view.metadata("variable"), view.metadata("band")
                ((4, 4), 'Band_1', 1)
                >>> coarse = view.reread(ViewRequest(budget=4))
                >>> coarse.z.values.shape, coarse.ref.uri, coarse.request.budget
                ((2, 2), 'object:doc-reread', 4)
                >>> coarse.x.values.tolist(), coarse.y.values.tolist()
                ([1.0, 3.0], [3.0, 1.0])

                ```
            - Only the describing keys are carried; `band` belongs to the first read, so it is not:
                ```python
                >>> import numpy as np
                >>> from pyramids.base.georeference import GeoReference
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.base.sources.view import DESCRIBING_KEYS, SourceView
                >>> from digitalearth.base.spec import DataRef, ViewRequest
                >>> dem = Dataset.from_array(
                ...     np.arange(16.0).reshape(4, 4),
                ...     geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
                ... )
                >>> ref = DataRef.to_object(dem, name="doc-reread-keys")
                >>> coarse = SourceView.of(ref.open(), ref=ref).reread(ViewRequest(budget=4))
                >>> DESCRIBING_KEYS
                ('variable', 'kind', 'standard_name')
                >>> coarse.metadata("variable"), coarse.metadata("kind"), coarse.metadata("band")
                ('Band_1', 'raster', None)

                ```
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
        # A windowed read returns a bare array, which carries no band name — so the variable, units and
        # standard_name would be lost by re-reading the very same slice at a different resolution. They
        # describe the data, not the window, so they are carried over where the new read supplied nothing.
        # The result is built as a new view from public properties rather than by writing into `again`'s
        # private state.
        return SourceView(
            again.z,
            again.x,
            again.y,
            again.crs,
            self._carried(self._meta, again._meta),
            again.units if again.units is not None else self.units,
            ref=again.ref,
            selection=again.selection,
            request=again.request,
        )

    @staticmethod
    def _carried(previous: dict, fresh: dict) -> dict:
        """Return a re-read's metadata, with what describes the data filled in from the previous read.

        Args:
            previous: The metadata of the view being re-read.
            fresh: The metadata the new read produced.

        Returns:
            A copy of `fresh` — neither argument is modified — plus each key in :data:`DESCRIBING_KEYS` that
            `previous` holds and the new read did not supply. Every other key is the new read's alone.

            Two rules replace a truthiness test that over-reached. First, only a **named** set of keys is
            carried: the ones that say what the data *is*, which a window cannot change. A key describing
            the read itself is not the old read's to supply. Second, "did not supply" means *absent*, with
            one named exception — the extractor writes ``variable=""`` for a bare array, which is a
            placeholder, not an answer. Testing ``not fresh.get(key)`` instead replaced every falsy value,
            and falsy values carry meaning: the collection extractor writes a **0-based** ``member``, so a
            re-read of member ``0`` would have reported whatever member the previous read held.
        """
        merged = dict(fresh)
        for key in DESCRIBING_KEYS:
            if key not in previous:
                continue
            placeholder = key == "variable" and merged.get(key) == ""
            if key not in merged or placeholder:
                merged[key] = previous[key]
        return merged

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
            selection: Which slice to read. Defaults to the default band. Only its band and its `budget`
                are honoured; the axes it may not name are listed under Raises.
            request: The region/resolution/budget wanted. Applied through pyramids' windowed read when the
                object exposes one (`read_part`) and the request names a region, a budget, or a full canvas
                (both `width` and `height`) — a size with no region windows against the source's own `bbox`.
                A request naming none of those, or one given for an object that cannot window, is recorded
                but not enforced: a reader that cannot window is not a reason to refuse the read. A `budget`
                on `selection` is folded in first, the smaller of the two winning — see :meth:`_budgeted` —
                and the view records the request as it stands after that.
            crs: The CRS to record for the coordinates, passed through to the extractor. A windowed read
                records the window's CRS instead, when the window has one.

        Returns:
            The view, carrying `ref`, the selection, and the request as recorded under `request` above.

        Raises:
            ValueError: if `selection` names a `time`, `level`, `member` or `overview`, or more than one
                band. Nothing but the band and the budget is honoured, so a view that stored the rest would
                report a slice it does not hold; a composite selection is refused for the same reason. Also
                if `request` would window a rotated raster — see :meth:`_refuse_rotated`.
            Exception: whatever the extractor raises for data it cannot read, or pyramids for a window it
                cannot read.

        Examples:
            - A plain array becomes a view with no address, which is a legitimate state:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources.view import SourceView
                >>> view = SourceView.of(np.arange(6.0).reshape(2, 3), crs=4326)
                >>> view.z.values.shape, view.rereadable
                ((2, 3), False)

                ```
            - A budget with no region windows the raster's own extent, labelled with real cell centres:
                ```python
                >>> import numpy as np
                >>> from pyramids.base.georeference import GeoReference
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.base.sources.view import SourceView
                >>> from digitalearth.base.spec import ViewRequest
                >>> dem = Dataset.from_array(
                ...     np.arange(16.0).reshape(4, 4),
                ...     geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
                ... )
                >>> view = SourceView.of(dem, request=ViewRequest(budget=4))
                >>> view.z.values.shape, view.crs
                ((2, 2), 4326)
                >>> view.x.values.tolist(), view.y.values.tolist()
                ([1.0, 3.0], [3.0, 1.0])

                ```
            - A budget on the selection is applied too, and the tighter of the two budgets wins:
                ```python
                >>> import numpy as np
                >>> from pyramids.base.georeference import GeoReference
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.base.sources.view import SourceView
                >>> from digitalearth.base.spec import Selection, ViewRequest
                >>> dem = Dataset.from_array(
                ...     np.arange(16.0).reshape(4, 4),
                ...     geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
                ... )
                >>> view = SourceView.of(
                ...     dem, selection=Selection.of(1, budget=4), request=ViewRequest(budget=100)
                ... )
                >>> view.request.budget, view.selection.budget, view.z.values.shape
                (4, 4, (2, 2))

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
        request = cls._budgeted(request, picked.budget)
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
    def _budgeted(
        request: Optional[ViewRequest], budget: Optional[int]
    ) -> Optional[ViewRequest]:
        """Fold a selection's cell budget into the request, so it is applied rather than only stored.

        Args:
            request: What was asked for, or ``None``.
            budget: The budget the selection names, or ``None``.

        Returns:
            `request` unchanged when there is no selection budget, or the request's own budget is already
            the tighter of the two. Otherwise the request with the selection's budget — a fresh
            ``ViewRequest(budget=...)`` when there was no request at all.

            Both are limits, so the smaller wins. The guard in :meth:`of` refuses `time`, `level`, `member`
            and `overview` because a view that stored them would report a slice it does not hold; `budget`
            was neither refused nor applied, so ``.selection.budget`` read back as a limit that had never
            been respected. Honouring it is possible where the others are not, because the request already
            carries the same limit.
        """
        if budget is None:
            return request
        if request is None:
            return ViewRequest(budget=budget)
        if request.budget is not None and request.budget <= budget:
            return request
        return cast(ViewRequest, replace(request, budget=budget))

    @classmethod
    def _shape(cls, request: ViewRequest) -> Tuple[int, int]:
        """Return the ``(width, height)`` in cells a windowed read should produce.

        Args:
            request: What was asked for.

        Returns:
            The caller's canvas when they named both dimensions **and it fits the budget** — reading a square
            for a request that says ``800x600`` distorts the aspect ratio of the very canvas those fields
            exist to describe. A request missing either dimension gets a square, which is what a budget
            alone can say: `int(sqrt(budget))` cells on a side (at least one) when it has a budget, and
            :meth:`~digitalearth.base.spec.viewrequest.ViewRequest.side`, with its floor of 64, when it has
            none.

            The canvas is measured in **device** pixels, so a request carrying ``pixel_ratio=2.0`` reads the
            1600x1200 it will draw rather than the 800x600 it is laid out at. That is the whole purpose of
            :attr:`~digitalearth.base.spec.viewrequest.ViewRequest.pixels`, which
            :meth:`~digitalearth.base.spec.viewrequest.ViewRequest.side` already folds the ratio into; reading
            the CSS canvas instead returned a quarter of the detail on a retina display, and did it silently,
            because the result is *under* the budget the caller set.

            The count compared against the budget is the product of the two integers returned — the number of
            cells that will actually be read, which is ``pixels`` up to the rounding of a fractional device
            pixel.

            A canvas over budget is scaled down by :meth:`_fitted`. The budget is the **guarantee**: the
            product of the two numbers returned never exceeds it. The aspect ratio is best effort, held to
            the nearest whole cell, because cells do not come in fractions.
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
        columns = max(1, int(width * request.pixel_ratio))
        rows = max(1, int(height * request.pixel_ratio))
        if request.within_budget(columns * rows):
            return columns, rows
        return cls._fitted(columns, rows, request.budget)  # type: ignore[arg-type]

    @staticmethod
    def _fitted(width: int, height: int, budget: int) -> Tuple[int, int]:
        """Shrink a canvas until it fits `budget` cells, keeping its shape as nearly as whole cells allow.

        Args:
            width: The canvas width in cells.
            height: The canvas height in cells.
            budget: The greatest number of cells that may be returned.

        Returns:
            A ``(width, height)`` whose product is **always** ``<= budget``, and which is within one cell
            per axis of the exactly-scaled canvas whenever both axes survive the scaling.

            Truncating each axis can only go under the scale factor, so the product of the two truncated
            axes is under the budget by construction. What is *not* safe is lifting a truncated axis back
            to one afterwards, which is what this replaces: for an elongated canvas the short axis rounds to
            zero, and restoring it multiplies the long axis straight back through the limit. Measured on the
            old spelling, ``10000x1`` under a budget of ``100`` returned ``1000x1`` — ten times the limit the
            budget exists to impose, on exactly the shapes it matters for (a cross-section strip, a profile).

            A canvas that elongated gets its long axis capped at the budget and its short axis set to one,
            which is the most detail the limit can buy.
        """
        scale = (budget / (width * height)) ** 0.5
        columns, rows = int(width * scale), int(height * scale)
        if columns and rows:
            return columns, rows
        if width >= height:
            return max(1, min(width, budget)), 1
        return 1, max(1, min(height, budget))

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
            A ``(data, x, y, crs)`` tuple. Unwindowed, that is `data` unchanged and three ``None``s: when
            there is no request, the object has no `read_part`, or the request names no region and either
            names no usable size (no budget, and not both of width and height) or meets an object with no
            `bbox` to window the size against.

            Windowed, the window is first grown to whole source pixels by :meth:`_aligned`, and the result is
            the **array** ``read_part`` returns for it plus the coordinates of its cell centres and the CRS
            they are in. The coordinates have to be computed here because ``read_part`` is an array
            reader, not a dataset reader: pyramids' own docstring says *"Pixel values only — no transform,
            bounds, or CRS is attached"*. Without them the extractor falls back to ``np.arange`` axes and the
            view claims a projected CRS while holding pixel indices.

        Raises:
            ValueError: if the source is a rotated raster — see :meth:`_refuse_rotated`.
            Exception: whatever pyramids raises for a window it cannot read.
        """
        nothing = (data, None, None, None)
        if request is None or not hasattr(data, "read_part"):
            return nothing
        bbox = request.as_bbox()
        if bbox is None:
            # A size with no region still has to be honoured: the object *can* window, so skipping here
            # returned the whole raster — blowing a budget silently, or handing a caller who asked for a 4x3
            # canvas all 13x14 cells. The source's own bbox is the region.
            source_bbox = getattr(data, "bbox", None)
            # A full canvas counts; half of one does not. `_shape` cannot use a lone width, so windowing on
            # one read a 64x64 floor square that honoured nothing the caller named.
            canvas = request.width is not None and request.height is not None
            if source_bbox is None or (request.budget is None and not canvas):
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
        cls._refuse_rotated(data)
        window = cls._aligned(window, data)
        width, height = cls._native(window, data, *cls._shape(request))
        band = selection.first_band
        try:
            array = cls._read_window(data, window, width, height, band)
        # A viewport panned off the data is a place with nothing in it, not a failed read: the canvas comes
        # back empty, as it does for a window over a hole in the raster, and the map keeps panning.
        except Exception as error:  # noqa: BLE001
            if not _off_source(error):
                raise
            array = np.full((height, width), np.nan)
        rows, columns = array.shape[-2], array.shape[-1]
        xs, ys = cls._axes(window, rows, columns, data)
        return array, xs, ys, window.crs

    @staticmethod
    def _missing(data: Any, band: int) -> Optional[float]:
        """Return the value a windowed read of this band uses for "no data", in the units it returns.

        Args:
            data: The raster being read.
            band: The 1-based band.

        Returns:
            The sentinel as the read returns it — scaled and offset when the band is packed, since a windowed
            read unpacks its values — or `None` when the band declares none.

        Note:
            A native-resolution read masks through pyramids (`read_array(masked=True)`), which is the reliable
            way and the one :func:`~digitalearth.base.arrays.read_masked_band` uses. A decimated read cannot:
            pyramids refuses `masked=True` together with `out_shape=`, so until it does not
            (serapeum-org/pyramids#1156) a viewport read compares against the sentinel itself.
        """
        sentinels = getattr(data, "no_data_value", None)
        if not sentinels or band - 1 >= len(sentinels):
            return None
        sentinel = sentinels[band - 1]
        if sentinel is None:
            return None
        scales = getattr(data, "scale", None) or [1.0]
        offsets = getattr(data, "offset", None) or [0.0]
        index = band - 1 if band - 1 < len(scales) else 0
        scale = scales[index] if scales[index] is not None else 1.0
        offset = (offsets[index] if index < len(offsets) else 0.0) or 0.0
        return float(sentinel) * float(scale) + float(offset)

    @classmethod
    def _unmasked(cls, array: np.ndarray, data: Any, band: int) -> np.ndarray:
        """Return a windowed read as floats, with its nodata cells set to `NaN`.

        Args:
            array: What the windowed read returned.
            data: The raster it was read from.
            band: The 1-based band.

        Returns:
            A `float64` array. Without this a nodata cell reaches the colour range as its sentinel —
            -3.4e38 on `acc4000.tif` — which flattens every real value onto one end of the ramp.
        """
        values = np.asarray(array, dtype="float64")
        sentinel = cls._missing(data, band)
        if sentinel is not None:
            values = np.where(values == sentinel, np.nan, values)
        return values

    @classmethod
    def _read_window(
        cls, data: Any, window: Bounds, width: int, height: int, band: int
    ) -> np.ndarray:
        """Read one window of a raster as floats, with its missing cells as `NaN`.

        Args:
            data: The raster to read.
            window: The region, snapped to source pixels and in the source's CRS.
            width: Cells across the canvas.
            height: Cells down it.
            band: The 1-based band.

        Returns:
            A `float64` array of `(height, width)`.

        Note:
            **Two reads, one meaning.** At full resolution the window is read with `masked=True`, so pyramids
            decides what is missing — the only reliable answer, and the one a packed band needs, since its
            nodata cells unpack to ordinary-looking numbers. Decimated, that is not available (pyramids
            refuses `masked=True` with `out_shape=`, serapeum-org/pyramids#1156), so the read is compared
            against the sentinel instead. The difference is visible: on `acc4000.tif` the resampler carries
            values across the nodata boundary, so a decimated frame has fewer missing cells than the raster
            does. Reading at full resolution when the window allows it is what keeps a zoomed-in frame honest.
        """
        native = cls._native_cells(window, data)
        if native is not None and (width, height) == native:
            values = np.ma.asarray(
                data.read_array(band=band - 1, bbox=window.as_bbox(), masked=True)
            ).astype("float64")
            return np.ma.filled(values, np.nan)
        return cls._unmasked(
            data.read_part(
                bbox=window.as_bbox(),
                dst_width=width,
                dst_height=height,
                bbox_crs=window.crs,
                band=band - 1,  # pyramids' windowed read is 0-based
            ),
            data,
            band,
        )

    @staticmethod
    def _native(window: Bounds, data: Any, width: int, height: int) -> Tuple[int, int]:
        """Return the canvas, never asking for more cells than the window holds at full resolution.

        Args:
            window: The region being read, in the source's CRS.
            data: The raster being read.
            width: The canvas width the budget allowed.
            height: The canvas height it allowed.

        Returns:
            `(width, height)`, each capped at the number of source cells the window spans. Reading 100x100
            cells from a 13x14 raster is interpolation dressed as data: it costs more, says no more, and is a
            picture of the resampler rather than of the raster.
        """
        cells = SourceView._native_cells(window, data)
        if cells is None:
            return width, height
        return min(width, cells[0]), min(height, cells[1])

    @staticmethod
    def _native_cells(window: Bounds, data: Any) -> Optional[Tuple[int, int]]:
        """Return how many source cells a window spans, across and down.

        Args:
            window: The region, in the source's CRS.
            data: The raster being read.

        Returns:
            `(across, down)`, or `None` when the source does not say what a cell measures — a reader with no
            geotransform, which is windowed exactly as it was asked.
        """
        cell = getattr(data, "cell_size", None)
        if not cell:
            return None
        xmin, ymin, xmax, ymax = window.as_bbox()
        across = max(1, int(round((xmax - xmin) / float(cell))))
        down = max(1, int(round((ymax - ymin) / float(cell))))
        return across, down

    @classmethod
    def _aligned(cls, window: Bounds, data: Any) -> Bounds:
        """Grow `window` outward to the source's pixel edges, so a read of it needs no snapping.

        Args:
            window: The window wanted, in the source's CRS.
            data: The object being read, for its geotransform.

        Returns:
            The smallest pixel-aligned window containing `window`, or `window` itself when :meth:`_grid`
            finds no grid to align against (no geotransform, or a zero step).

            On an axis-aligned grid this is the same outward snap ``read_part`` performs internally — the
            floor and ceiling of the corners' pixel coordinates. Doing it here rather than reproducing its
            arithmetic afterwards is what keeps the labels honest: the caller's bbox and the rectangle
            actually read become one rectangle, so there is no second window to get wrong. A rotated grid is
            snapped as if unrotated, because :meth:`_grid` drops the rotation terms.

            The long-term answer is for the reader to report the window it actually read alongside the
            array, which is https://github.com/serapeum-org/pyramids/issues/1149.
        """
        transform = cls._grid(data)
        if transform is None:
            return window
        origin_x, pixel_w, origin_y, pixel_h = transform
        xmin, ymin, xmax, ymax = window.as_bbox()
        left, right = cls._snapped(xmin, xmax, origin_x, pixel_w)
        bottom, top = cls._snapped(ymin, ymax, origin_y, pixel_h)
        return Bounds(left, bottom, right, top, window.crs)

    @staticmethod
    def _refuse_rotated(data: Any) -> None:
        """Refuse a windowed read of a raster whose geotransform carries rotation.

        Args:
            data: The object about to be windowed.

        Raises:
            ValueError: if `geotransform[2]` or `geotransform[4]` is non-zero.

            A rotated raster's cells do not lie along one x axis and one y axis, so no pair of 1-D
            coordinate arrays can label a window of it — and the axis-aligned bbox `read_part` takes does
            not describe the rectangle of cells it returns. Windowing one anyway came back silently wrong:
            on geotransform `(0, 1, 0.5, 8, 0, -1)` a 4x4 read labelled `x = [1.5, 2.5, 3.5, 4.5]` held
            `[-9999.0, 30.0, 30.5, 31.17]` — nodata and interpolated values under clean labels. Refused
            here, like the selection axes :meth:`of` cannot honour, rather than drawn wrong.
        """
        transform = getattr(data, "geotransform", None)
        if transform and len(transform) >= 6 and (transform[2] or transform[4]):
            raise ValueError(
                f"SourceView cannot window a rotated raster (geotransform {tuple(transform)}): its cells do "
                "not lie along one x and one y axis, so a window of it has no coordinates to label. Read it "
                "without a request instead"
            )

    @staticmethod
    def _grid(data: Any) -> Optional[Tuple[float, float, float, float]]:
        """Return the source's ``(origin_x, step_x, origin_y, step_y)``, or ``None`` if it has no grid.

        Args:
            data: The object being read.

        Returns:
            The four geotransform entries that place a cell on an axis-aligned grid. The rotation terms
            (`geotransform[2]` and `geotransform[4]`) are not part of the answer: a windowed read refuses a
            rotated raster first, in :meth:`_refuse_rotated`, so by the time this is consulted they are zero.
            ``None`` when there is no geotransform, it has fewer than six entries, or a step is zero and the
            grid degenerate.
        """
        transform = getattr(data, "geotransform", None)
        if not transform or len(transform) < 6:
            return None
        origin_x, pixel_w, _, origin_y, _, pixel_h = transform[:6]
        if not pixel_w or not pixel_h:
            return None
        return float(origin_x), float(pixel_w), float(origin_y), float(pixel_h)

    @staticmethod
    def _snapped(
        low: float, high: float, origin: float, step: float
    ) -> Tuple[float, float]:
        """Widen one axis of a window out to the cell edges that enclose it.

        Args:
            low: The lower world coordinate of the window on this axis.
            high: The upper world coordinate.
            origin: The geotransform's origin for this axis.
            step: The geotransform's cell step, which may be negative.

        Returns:
            The ``(low, high)`` pair of enclosing cell edges, in ascending world order.

            The floor and the ceiling are taken in **pixel** space and only then converted back, because a
            negative step reverses which end of the window each one belongs to. Flooring the low world
            coordinate directly — the obvious spelling — snaps *inward* on such an axis, which reads as a
            correct fix and silently crops.
        """
        first = (low - origin) / step
        second = (high - origin) / step
        edges = (
            origin + float(np.floor(min(first, second))) * step,
            origin + float(np.ceil(max(first, second))) * step,
        )
        return min(edges), max(edges)

    @classmethod
    def _axes(
        cls, window: Bounds, rows: int, columns: int, data: Any
    ) -> Tuple[Any, Any]:
        """Return the cell-centre coordinates of a windowed read, in the order its rows and columns come.

        Args:
            window: The pixel-aligned window that was read.
            rows: How many rows came back.
            columns: How many columns.
            data: The object that was read, for its geotransform.

        Returns:
            An ``(x, y)`` pair of centre coordinates.

            Cell centres, not edges: an extractor's axes name where each cell *is*, and a half-cell offset
            is the difference between a raster drawn correctly and one drawn half a pixel adrift.

            The **direction** of each axis comes from the geotransform's step signs, because ``read_part``
            returns rows and columns in storage order. A raster stored south-up (``step_y > 0``) therefore
            hands back ascending y, and one stored east-left (``step_x < 0``) descending x; labelling
            either the usual way round renders it mirrored, silently, since a flipped raster draws
            perfectly happily. With no geotransform to consult the north-up, west-left convention is
            assumed, which is the transform pyramids builds from a `GeoReference` corner and cell size.

        Note:
            Only this windowed path derives the y direction. pyramids' own ``Dataset.y`` measures rows
            downward from ``geotransform[3]`` whatever the sign of ``geotransform[5]``, so a south-up source
            read through any other extractor is labelled outside its own extent (a raster spanning y 0..8
            reports ``y = [-0.5, -1.5, ...]``), and its ``bbox`` comes back with ``ymin > ymax``. That is
            https://github.com/serapeum-org/pyramids/issues/1148; this method only avoids adding a second,
            differently-wrong answer.
        """
        xmin, ymin, xmax, ymax = window.as_bbox()
        grid = cls._grid(data)
        east_left = grid is not None and grid[1] < 0
        south_up = grid is not None and grid[3] > 0
        x_from, x_to = (xmax, xmin) if east_left else (xmin, xmax)
        y_from, y_to = (ymin, ymax) if south_up else (ymax, ymin)
        xs = x_from + (np.arange(columns) + 0.5) * (x_to - x_from) / columns
        ys = y_from + (np.arange(rows) + 0.5) * (y_to - y_from) / rows
        return xs, ys
