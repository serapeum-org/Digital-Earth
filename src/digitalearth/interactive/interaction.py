"""InteractionMixin — tap-to-inspect, hover & draw/edit AOI for ``InteractiveMap``.

Owns ``on_tap`` / ``tap_profile`` / ``hover`` (DI.7) and ``draw`` / ``drawn_geometry`` / ``aoi_crop`` /
``cross_filter`` (DI.8). These are the interactive-only payoff — click a pixel to pull a time series,
draw an area-of-interest and get the geometry **back in pyramids' CRS** to drive a crop, brush a
selection across panels.

**Server note:** HoloViews streams that round-trip to Python (``Tap``, ``BoxEdit``, ``PolyDraw``,
linked selections) only sync with a **live kernel/server**; in a saved static HTML they degrade to
plain hover. Each method documents this rather than implying full interactivity in an exported file.
All CRS work (crop, the un-projection of drawn geometry) goes through pyramids.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Self

from digitalearth.base.crs import reproject
from digitalearth.base.spec import DEFAULT_BAND
from digitalearth.interactive.base import _require_holoviz

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.interactive.base import InteractiveMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


class InteractionMixin(_MixinBase):
    """Interactivity builders (DI.7 + DI.8): tap-to-inspect, rich hover, draw-AOI, linked selection.

    A capability mixin of :class:`~digitalearth.interactive.map.InteractiveMap`: it is only ever composed into that
    map class, never instantiated or subclassed on its own. Its methods reach the element registry, the display CRS
    and the render/save lifecycle — and the sibling mixins' methods — through ``self``, and only the composition
    supplies those.

    The ``if TYPE_CHECKING`` base declared above the class is what records that contract for a type checker: it
    resolves each ``self.<attr>`` against :class:`~digitalearth.interactive.base.InteractiveMapBase`, the state
    ``InteractiveMap`` inherits. At runtime that base is plain ``object``, so composing this mixin leaves the
    ``InteractiveMap`` MRO exactly what it was before the annotation.

    See Also:
        digitalearth.interactive.map.InteractiveMap: the composition that supplies the state these methods use.
        digitalearth.interactive.base.InteractiveMapBase: the typing-only base declared above the class.
    """

    def hover(
        self,
        *,
        tooltips: list | None = None,
        formatters: dict | None = None,
        layer: int | None = None,
    ) -> Self:
        """Configure the Bokeh hover tooltips on a registered layer (DI.7).

        Args:
            tooltips: Bokeh ``HoverTool`` tooltips spec (e.g. ``[("value", "@value"), ("x", "$x")]``);
                ``None`` keeps the default value+coord readout.
            formatters: Optional Bokeh tooltip ``formatters`` mapping.
            layer: Index into :attr:`layers` — draw order, bottom first — of the layer to configure.
                ``None`` (the default) configures the layer the caller added last, which is not
                ``layers[-1]`` whenever that layer is drawn beneath an overlay such as coastlines; an
                underlay added after data does not take it over (see
                :meth:`~digitalearth.interactive.base.InteractiveMapBase._last_layer_index`, which is that
                rule. The colour keys read it too, but tighten it: a layer whose colour varies with nothing
                cannot take a key over, and a hover has no such condition — any layer can be hovered).

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when there is no layer to configure.
        """
        from bokeh.models import HoverTool

        gv, hv = _require_holoviz()
        if not self.layers:
            raise ValueError(
                "hover() needs at least one layer — add a builder call first"
            )
        index = self._last_layer_index("hover") if layer is None else layer
        tool = HoverTool(tooltips=tooltips, formatters=formatters or {})
        self.layers[index] = self.layers[index].opts(tools=[tool], backend="bokeh")
        return self

    def on_tap(self, callback: Callable, *, source: Any = None) -> Any:
        """Wire a ``Tap`` stream to ``callback`` and return the resulting ``DynamicMap`` (DI.7).

        On click the callback receives ``(x, y)`` in the display CRS and returns a HoloViews element
        (e.g. a side panel). Needs a live kernel/server to round-trip — in static HTML the map renders
        without the tap response.

        Args:
            callback: ``callback(x, y) -> hv element`` invoked on each tap.
            source: The element the tap listens on; defaults to the layer the caller added last — not the
                one drawn on top, which is a different layer whenever an overlay sits over it.

        Returns:
            The ``hv.DynamicMap`` driven by the tap stream.

        Raises:
            ValueError: when there is no source layer.
        """
        gv, hv = _require_holoviz()
        from holoviews import streams

        src = source
        if src is None:
            if self._last_layer_id is None:
                raise ValueError(
                    "on_tap() needs a source layer — add a builder call or pass source="
                )
            src = self.layers[self._last_layer_index("on_tap")]
        tap = streams.Tap(source=src, x=0.0, y=0.0)
        return hv.DynamicMap(lambda x, y: callback(x, y), streams=[tap])

    def tap_profile(
        self, collection: Any, *, source: Any = None, band: int = DEFAULT_BAND
    ) -> Any:
        """Click a cell to pull a time series from a ``DatasetCollection`` at that point (DI.7).

        Returns a ``DynamicMap`` whose ``Tap`` callback reads each member's value at the clicked cell
        (via pyramids ``map_to_array_coordinates``) and plots the series — the most-requested
        interactive feature over a static map. Needs a live kernel/server.

        Args:
            collection: A pyramids ``DatasetCollection`` whose members are time steps.
            source: The map layer the tap listens on; defaults to the layer the caller added last, as
                :meth:`on_tap` reads it.
            band: 1-based band sampled at the clicked cell.

        Returns:
            The ``hv.DynamicMap`` (a ``Curve`` time series per tap).

        Raises:
            ValueError: when there is no source layer.
        """
        import numpy as np

        gv, hv = _require_holoviz()
        if band < 1:
            raise ValueError(f"band is 1-based; got {band}")
        members = collection.datasets

        def _profile(x: float, y: float) -> Any:
            # pyramids' map_to_array_coordinates wants a GeoDataFrame/FeatureCollection; build the
            # single clicked point through pyramids (no shapely/geopandas import here, DX.3).
            from pyramids.feature.geometry import point_collection

            point = point_collection([(x, y)], crs=self.crs)
            series = []
            for member in members:
                ds = (
                    reproject(member, self.crs)
                    if self._needs_reproject(member)
                    else member
                )
                row, col = np.asarray(ds.map_to_array_coordinates(point))[0]
                arr = np.asarray(
                    ds.read_array(band=band - 1)
                )  # Dataset.read_array is 0-based
                row_i, col_i = int(row), int(col)
                inside = 0 <= row_i < arr.shape[0] and 0 <= col_i < arr.shape[1]
                series.append(float(arr[row_i, col_i]) if inside else np.nan)
            return hv.Curve(list(enumerate(series)), "t", "value")

        return self.on_tap(_profile, source=source)

    def _source_layer(self, source: Any, method: str) -> Any:
        """Resolve the element a stream listens on: the caller's, else the layer added last (IN-8).

        Args:
            source: The element the caller passed, or ``None`` to default to the last layer.
            method: The public method asking, named in the refusal.

        Returns:
            The element to bind the stream to.

        Raises:
            ValueError: when ``source`` is ``None`` and the map has no layer yet.
        """
        if source is not None:
            return source
        if self._last_layer_id is None:
            raise ValueError(
                f"{method}() needs a source layer — add a builder call or pass source="
            )
        return self.layers[self._last_layer_index(method)]

    def on_select(
        self, callback: Callable, *, kind: str = "box", source: Any = None
    ) -> Any:
        """Wire a selection stream to ``callback`` and return the resulting ``DynamicMap`` (DI.8 / IN-8).

        The selection half of :meth:`on_tap`: a box, a lasso or an index selection on the map hands its
        result back to Python, so brushing a region returns data rather than only highlighting it. Needs a
        live kernel/server to round-trip — in static HTML the selection tool draws but fires no callback.

        Args:
            callback: Invoked with the stream's own keyword on each selection — ``bounds`` (a
                ``(x0, y0, x1, y1)`` tuple) for ``"box"``, ``geometry`` (an ``(N, 2)`` array) for
                ``"lasso"``, ``index`` (a list of selected row indices) for ``"index"``. It returns a
                HoloViews element (e.g. a side panel of the selected rows).
            kind: ``"box"`` (``BoundsXY``), ``"lasso"`` (``Lasso``) or ``"index"`` (``Selection1D``).
            source: The element the stream listens on; defaults to the layer the caller added last.

        Returns:
            The ``hv.DynamicMap`` driven by the selection stream.

        Raises:
            ValueError: for an unknown ``kind``, or when there is no source layer.
        """
        _require_holoviz()
        from holoviews import streams

        src = self._source_layer(source, "on_select")
        makers = {
            "box": streams.BoundsXY,
            "lasso": streams.Lasso,
            "index": streams.Selection1D,
        }
        if kind not in makers:
            raise ValueError(
                f"unknown selection kind {kind!r}; choose 'box'/'lasso'/'index'"
            )
        import holoviews as hv

        stream = makers[kind](source=src)
        return hv.DynamicMap(callback, streams=[stream])

    def on_reset(self, callback: Callable, *, source: Any = None) -> Any:
        """Wire a ``PlotReset`` stream to ``callback`` and return the ``DynamicMap`` (DI.8 / IN-8).

        Fires when the viewer clicks Bokeh's reset tool, which is how a map restores its own derived state
        (a recomputed overview, a cleared selection) rather than leaving it stale. Needs a live kernel.

        Args:
            callback: Invoked with ``resetting`` (a bool) on each reset; returns a HoloViews element.
            source: The element the stream listens on; defaults to the layer the caller added last.

        Returns:
            The ``hv.DynamicMap`` driven by the reset stream.

        Raises:
            ValueError: when there is no source layer.
        """
        _require_holoviz()
        import holoviews as hv
        from holoviews import streams

        src = self._source_layer(source, "on_reset")
        return hv.DynamicMap(callback, streams=[streams.PlotReset(source=src)])

    def draw(
        self,
        kind: str = "box",
        *,
        num_objects: int | None = None,
        name: str | None = None,
        visible: bool = True,
    ) -> Self:
        """Add a draw/edit tool so the user can sketch an area-of-interest (DI.8).

        Wraps a HoloViews draw stream around a fresh annotation layer: ``"box"`` → ``BoxEdit``,
        ``"poly"`` → ``PolyDraw``, ``"point"`` → ``PointDraw``, ``"freehand"`` → ``FreehandDraw``. Read
        the result back via :attr:`drawn_geometry`. Needs a live kernel/server — static HTML captures
        nothing.

        Args:
            kind: ``"box"`` / ``"poly"`` / ``"point"`` / ``"freehand"``.
            num_objects: Max number of shapes (``None`` = unlimited).
            name: The caller's own name for the annotation layer, used as its id and its label;
                ``None`` (default) generates one, and a name already on the map is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the annotation layer is drawn. ``False`` builds it hidden **and**
                describes it hidden (#327).

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: for an unknown ``kind``.
        """
        gv, hv = _require_holoviz()
        from holoviews import streams

        crs = gv.util.process_crs(self.crs)
        if kind == "box":
            layer = gv.Rectangles([], crs=crs)
            stream = streams.BoxEdit(source=layer, num_objects=num_objects or 0)
        elif kind == "poly":
            layer = gv.Polygons([], crs=crs)
            stream = streams.PolyDraw(source=layer, num_objects=num_objects or 0)
        elif kind == "point":
            layer = gv.Points([], crs=crs)
            stream = streams.PointDraw(source=layer, num_objects=num_objects or 0)
        elif kind == "freehand":
            layer = gv.Path([], crs=crs)
            stream = streams.FreehandDraw(source=layer, num_objects=num_objects or 0)
        else:
            raise ValueError(
                f"unknown draw kind {kind!r}; choose 'box'/'poly'/'point'/'freehand'"
            )
        # Appended, not overwritten (IN-11): a map can carry several draw tools at once, and a second
        # `draw()` no longer silently replaces the first tool's binding. `_draw_stream` tracks the most
        # recent so the single-tool `drawn_geometry` read keeps meaning what it did.
        self._draw_streams.append(stream)
        self._draw_stream = stream
        self.add_layer(
            layer,
            name=name,
            visible=visible,
            kind="custom:holoviews",
        )
        return self

    @property
    def drawn_geometry(self) -> Any:
        """The geometry drawn via :meth:`draw`, as a display-CRS bbox/dict (or ``None``).

        For a box this is ``(xmin, ymin, xmax, ymax)`` in the display CRS — ready to feed
        ``Dataset.crop`` (pyramids). Returns ``None`` before anything is drawn. Needs a live kernel
        for the stream to have captured anything.

        Returns:
            The drawn geometry, or ``None``.
        """
        stream = self._draw_stream
        if stream is None or not getattr(stream, "data", None):
            return None
        data = stream.data
        if {"x0", "y0", "x1", "y1"} <= set(data):  # BoxEdit
            if not data["x0"]:
                return None
            return (data["x0"][0], data["y0"][0], data["x1"][0], data["y1"][0])
        return data

    @property
    def drawn_geometries(self) -> list:
        """Every draw tool's captured geometry, in the order the tools were added (IN-11).

        A map can carry several :meth:`draw` tools now; :attr:`drawn_geometry` reads the most recent, and
        this reads them all. Each entry is what that one tool captured — a ``(xmin, ymin, xmax, ymax)`` bbox
        for a box tool, the raw stream ``data`` dict otherwise — with a tool that has captured nothing
        skipped. Empty before anything is drawn, or with no live kernel.

        Returns:
            One captured geometry per draw tool that has captured something, oldest first.
        """
        captured = []
        for stream in self._draw_streams:
            data = getattr(stream, "data", None)
            if not data:
                continue
            if {"x0", "y0", "x1", "y1"} <= set(data):
                if not data["x0"]:
                    continue
                captured.append(
                    (data["x0"][0], data["y0"][0], data["x1"][0], data["y1"][0])
                )
            else:
                captured.append(data)
        return captured

    def annotate(self, element: Any = None, **kwargs: Any) -> Any:
        """Open HoloViews' annotator — a draw tool paired with an editable attribute table (DI.8 / IN-11).

        Where :meth:`draw` attaches a bare draw stream, this composes GeoViews' geographic adaptation of
        ``holoviews.annotate``: the drawn geometry gets an editable table of its vertices and any annotation
        columns, and the edits round-trip to Python through the annotator's ``annotated``/``selected``
        objects. Needs a live kernel/server. The annotation itself is not reprojected — build the element in
        the display CRS (the mixed-CRS guard on :meth:`add_layer` is the check for a layer, and this is the
        same contract).

        Args:
            element: The GeoViews element to annotate — ``gv.Points``/``gv.Path``/``gv.Polygons`` in the
                display CRS. ``None`` annotates an empty ``gv.Points`` layer, i.e. "let me draw points and
                edit their table".
            **kwargs: Forwarded to ``holoviews.annotate`` — ``annotations`` (the attribute columns),
                ``num_objects``, ``vertex_annotations``, ``table_transforms``, ….

        Returns:
            The ``holoviews.annotate`` **instance** it was applied through: ``.annotated`` is the edited
            element and ``.selected`` the picked rows, both live, and displaying the instance shows the
            draw-tool-plus-table layout. A fresh instance per call, so two annotators do not share state.
        """
        gv, _ = _require_holoviz()
        from holoviews import annotate as _annotate

        target = (
            element
            if element is not None
            else gv.Points([], crs=gv.util.process_crs(self.crs))
        )
        annotator = _annotate.instance()
        annotator(target, **kwargs)
        return annotator

    def aoi_crop(self, dataset: Any) -> Any:
        """Crop ``dataset`` (in pyramids) to the drawn box and return the clipped raster (DI.8).

        Reads the drawn bbox (display CRS) and calls ``Dataset.crop`` — the crop itself is pyramids',
        this method only reads the drawn coordinates.

        Args:
            dataset: A pyramids ``Dataset`` to crop.

        Returns:
            The cropped ``Dataset``.

        Raises:
            ValueError: when nothing has been drawn yet.
        """
        bbox = self.drawn_geometry
        if not isinstance(bbox, tuple):
            raise ValueError(
                "aoi_crop() needs a drawn box — call draw('box') and draw a region first"
            )
        return dataset.crop(bbox=bbox, epsg=self.crs)

    def cross_filter(
        self,
        *panels: Any,
        selection_mode: str = "union",
        cross_filter_mode: str = "intersect",
        index_cols: list | None = None,
        selected_color: str | None = None,
        unselected_color: str | None = None,
        unselected_alpha: float = 0.1,
        show_regions: bool = True,
    ) -> Any:
        """Link selections across ``panels`` so brushing one cross-filters the others (DI.8 / IN-5).

        Wraps ``holoviews.selection.link_selections`` and — the point of IN-5 — exposes the parameters that
        decide *how* a brush selects and lets the result be read back into Python: ``selection_expr`` on the
        returned linker is the live predicate, and ``selection_param(element)`` hands back the filtered data.
        So brushing an interactive map returns data, the same payoff :meth:`aoi_crop` already gives a drawn
        box. Needs a live kernel/server.

        Args:
            *panels: HoloViews elements/overlays to link (defaults to this map's render when empty).
            selection_mode: How a new box combines with the current selection *on one element* —
                ``"union"`` (default), ``"intersect"``, ``"overwrite"`` or ``"inverse"``.
            cross_filter_mode: How selections combine *across* elements — ``"intersect"`` (default) or
                ``"overwrite"``.
            index_cols: Columns that identify a row across elements, so a selection in one table highlights
                the same records in another; ``None`` links by the geometry/position instead.
            selected_color: Colour drawn for selected glyphs; ``None`` keeps HoloViews' default.
            unselected_color: Colour drawn for unselected glyphs; ``None`` keeps the default.
            unselected_alpha: Opacity of the unselected glyphs, so the selection stands out.
            show_regions: Draw the selection geometry (the box/lasso) over the data.

        Returns:
            The ``link_selections`` instance applied to the panels. Read ``.selection_expr`` for the live
            predicate, or ``.selection_param(element)`` for the filtered data.
        """
        gv, hv = _require_holoviz()
        from holoviews.selection import link_selections

        targets = list(panels) if panels else [self.render()]
        combined = targets[0]
        for panel in targets[1:]:
            combined = combined + panel
        options: dict[str, Any] = {
            "selection_mode": selection_mode,
            "cross_filter_mode": cross_filter_mode,
            "unselected_alpha": unselected_alpha,
            "show_regions": show_regions,
        }
        if index_cols is not None:
            options["index_cols"] = index_cols
        if selected_color is not None:
            options["selected_color"] = selected_color
        if unselected_color is not None:
            options["unselected_color"] = unselected_color
        linker = link_selections.instance(**options)
        linker(combined)
        return linker
