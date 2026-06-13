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

from typing import Any, Callable, Optional

from digitalearth.interactive.base import _require_holoviz


class InteractionMixin:
    """Interactivity builders (DI.7 + DI.8): tap-to-inspect, rich hover, draw-AOI, linked selection."""

    def hover(
        self,
        *,
        tooltips: Optional[list] = None,
        formatters: Optional[dict] = None,
        layer: int = -1,
    ) -> "InteractionMixin":
        """Configure the Bokeh hover tooltips on a registered layer (DI.7).

        Args:
            tooltips: Bokeh ``HoverTool`` tooltips spec (e.g. ``[("value", "@value"), ("x", "$x")]``);
                ``None`` keeps the default value+coord readout.
            formatters: Optional Bokeh tooltip ``formatters`` mapping.
            layer: Index of the layer to configure (default the most recent).

        Returns:
            This map (chainable).

        Raises:
            ValueError: when there is no layer to configure.
        """
        from bokeh.models import HoverTool

        gv, hv = _require_holoviz()
        if not self.layers:
            raise ValueError(
                "hover() needs at least one layer — add a builder call first"
            )
        tool = HoverTool(tooltips=tooltips, formatters=formatters or {})
        self.layers[layer] = self.layers[layer].opts(tools=[tool], backend="bokeh")
        return self

    def on_tap(self, callback: Callable, *, source: Any = None) -> Any:
        """Wire a ``Tap`` stream to ``callback`` and return the resulting ``DynamicMap`` (DI.7).

        On click the callback receives ``(x, y)`` in the display CRS and returns a HoloViews element
        (e.g. a side panel). Needs a live kernel/server to round-trip — in static HTML the map renders
        without the tap response.

        Args:
            callback: ``callback(x, y) -> hv element`` invoked on each tap.
            source: The element the tap listens on; defaults to the most recent layer.

        Returns:
            The ``hv.DynamicMap`` driven by the tap stream.

        Raises:
            ValueError: when there is no source layer.
        """
        gv, hv = _require_holoviz()
        from holoviews import streams

        src = (
            source if source is not None else (self.layers[-1] if self.layers else None)
        )
        if src is None:
            raise ValueError(
                "on_tap() needs a source layer — add a builder call or pass source="
            )
        tap = streams.Tap(source=src, x=0.0, y=0.0)
        return hv.DynamicMap(lambda x, y: callback(x, y), streams=[tap])

    def tap_profile(self, collection: Any, *, source: Any = None, band: int = 1) -> Any:
        """Click a cell to pull a time series from a ``DatasetCollection`` at that point (DI.7).

        Returns a ``DynamicMap`` whose ``Tap`` callback reads each member's value at the clicked cell
        (via pyramids ``map_to_array_coordinates``) and plots the series — the most-requested
        interactive feature over a static map. Needs a live kernel/server.

        Args:
            collection: A pyramids ``DatasetCollection`` whose members are time steps.
            source: The map layer the tap listens on; defaults to the most recent.
            band: 1-based band sampled at the clicked cell.

        Returns:
            The ``hv.DynamicMap`` (a ``Curve`` time series per tap).

        Raises:
            ValueError: when there is no source layer.
        """
        import numpy as np

        gv, hv = _require_holoviz()
        members = collection.datasets

        def _profile(x: float, y: float) -> Any:
            # pyramids' map_to_array_coordinates wants a GeoDataFrame/FeatureCollection; build the
            # single clicked point through pyramids (no shapely/geopandas import here, DX.3).
            from pyramids.feature.geometry import point_collection

            point = point_collection([(x, y)], crs=self.crs)
            series = []
            for member in members:
                ds = (
                    member.to_crs(self.crs) if self._needs_reproject(member) else member
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

    def draw(
        self, kind: str = "box", *, num_objects: Optional[int] = None
    ) -> "InteractionMixin":
        """Add a draw/edit tool so the user can sketch an area-of-interest (DI.8).

        Wraps a HoloViews draw stream around a fresh annotation layer: ``"box"`` → ``BoxEdit``,
        ``"poly"`` → ``PolyDraw``, ``"point"`` → ``PointDraw``, ``"freehand"`` → ``FreehandDraw``. Read
        the result back via :attr:`drawn_geometry`. Needs a live kernel/server — static HTML captures
        nothing.

        Args:
            kind: ``"box"`` / ``"poly"`` / ``"point"`` / ``"freehand"``.
            num_objects: Max number of shapes (``None`` = unlimited).

        Returns:
            This map (chainable).

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
        self._draw_stream = stream
        self.add_element(layer)
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
        stream = getattr(self, "_draw_stream", None)
        if stream is None or not getattr(stream, "data", None):
            return None
        data = stream.data
        if {"x0", "y0", "x1", "y1"} <= set(data):  # BoxEdit
            if not data["x0"]:
                return None
            return (data["x0"][0], data["y0"][0], data["x1"][0], data["y1"][0])
        return data

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

    def cross_filter(self, *panels: Any) -> Any:
        """Link selections across ``panels`` so brushing one cross-filters the others (DI.8).

        Wraps ``holoviews.selection.link_selections``; ``selection_expr``/``filter`` on the returned
        object pull the selected rows back into Python. Needs a live kernel/server.

        Args:
            *panels: HoloViews elements/overlays to link (defaults to this map's render when empty).

        Returns:
            The ``link_selections`` instance applied to the panels.
        """
        gv, hv = _require_holoviz()
        from holoviews.selection import link_selections

        targets = list(panels) if panels else [self.render()]
        combined = targets[0]
        for panel in targets[1:]:
            combined = combined + panel
        linker = link_selections.instance()
        linker(combined)
        return linker
