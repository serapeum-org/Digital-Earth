"""RasterMixin — raster builders for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``image`` / ``rgb`` / ``quadmesh`` / ``contours`` / ``filled_contours`` / ``spaghetti`` (DI.1a);
``large_image`` viewport loading lands later (DI.14).

Every builder funnels through ``self._to_display_source`` (reproject in **pyramids**, option A), then
emits a **plain HoloViews** element (``hv.Image``/``hv.RGB``/``hv.QuadMesh``) whose coordinates are
already in the display CRS — deliberately *not* ``gv.Image``, whose default PlateCarree ``crs`` would
re-project already-projected coordinates at render time. NoData arrives as a masked array from pyramids
and renders transparent (``NaN``).

**Naming note** — the interactive builders use HoloViews-idiomatic names that differ from the static
``Map``: ``image`` (static ``imshow``), ``rgb`` (static ``rgb_composite``), ``contours``/
``filled_contours`` (static ``contour``/``contourf``). ``spaghetti``/``quadmesh`` match. The divergence
is intentional (this tier reads as HoloViews to its users); the static↔interactive mapping is documented
in the tier plan's feature-parity matrix.
"""

from typing import Any, Optional, Sequence, Tuple

from digitalearth.interactive.base import _masked_to_nan, _require_holoviz


class RasterMixin:
    """Raster builders (DI.1a): colour-mapped fields, composites and ensemble spaghetti."""

    def _image_element(
        self, data: Any, *, band: int = 1, vname: Optional[str] = None
    ) -> Any:
        """Build a display-CRS ``hv.Image`` from ``data`` (the shared I1 recipe).

        Args:
            data: A pyramids ``Dataset`` / ``NetCDF`` / ``Source`` (anything the extractor accepts).
            band: 1-based band to read.
            vname: Value-dimension name; defaults to the source's variable/z name.

        Returns:
            holoviews.Image: the raster as a plain HoloViews image in the display CRS.
        """
        gv, hv = _require_holoviz()
        src = self._to_display_source(data, band=band)
        arr = _masked_to_nan(src.z.values)
        name = vname or self._vdim_name(src)
        return self._raster_element(src.x.values, src.y.values, arr, name)

    def image(
        self,
        data: Any,
        *,
        band: int = 1,
        cmap: Optional[str] = None,
        clim: Optional[Tuple[float, float]] = None,
        alpha: float = 1.0,
        colorbar: bool = True,
        **opts: Any,
    ) -> "RasterMixin":
        """Add a colour-mapped raster layer with hover readout (interactive ``imshow``).

        Args:
            data: A pyramids ``Dataset`` / ``NetCDF`` / ``Source``; reprojected to the display CRS
                through pyramids when needed.
            band: 1-based band to render.
            cmap: Colormap name; ``None`` (default) resolves it from the variable via
                ``autostyle.auto_style`` (DI.12) — the same lookup the static ``Map`` uses.
            clim: Optional ``(vmin, vmax)`` colour limits; ``None`` auto-scales.
            alpha: Layer opacity in ``[0, 1]``.
            colorbar: Whether to draw a colorbar.
            **opts: Extra HoloViews style options applied to the element.

        Examples:
            - Render a DEM as a pan/zoom raster with explicit colour limits:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().image(dem, cmap="terrain", clim=(0, 60))  # doctest: +SKIP
                >>> len(m.layers)                                               # doctest: +SKIP
                1

                ```

        Returns:
            This map (chainable).
        """
        src = self._to_display_source(data, band=band)
        arr = _masked_to_nan(src.z.values)
        element = self._raster_element(
            src.x.values, src.y.values, arr, self._vdim_name(src)
        )
        element = self._styled(
            element,
            common={
                "cmap": self._auto_cmap(src, cmap),
                "clim": clim,
                "alpha": alpha,
                "colorbar": colorbar,
                **opts,
            },
            bokeh={"tools": ["hover"]},
        )
        return self.add_element(element)

    def rgb(
        self, data: Any, *, bands: Sequence[int] = (1, 2, 3), **opts: Any
    ) -> "RasterMixin":
        """Add a true-colour composite from three raster bands (2–98 % percentile stretch).

        Args:
            data: A pyramids multiband ``Dataset``; reprojected to the display CRS first.
            bands: The three 1-based band indices composing ``(R, G, B)``.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            This map (chainable).

        Examples:
            - Compose three bands of a satellite stack into true colour:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> stack = Dataset.read_file("sentinel.tif")                   # doctest: +SKIP
                >>> m = InteractiveMap().rgb(stack, bands=(4, 3, 2))            # doctest: +SKIP
                >>> [d.name for d in m.layers[0].vdims]                         # doctest: +SKIP
                ['R', 'G', 'B']

                ```

        Raises:
            ValueError: when ``bands`` does not name exactly three bands.
        """
        import numpy as np

        from digitalearth.sources import get_stack

        gv, hv = _require_holoviz()
        if len(bands) != 3:
            raise ValueError(f"rgb() needs exactly three bands, got {tuple(bands)!r}")
        # Reproject once into a local handle, then feed both the coordinate extraction (get_source,
        # via _to_display_source) and the band stack (get_stack) from it — get_stack needs the same
        # already-reprojected dataset, so a single warp here keeps them consistent (H1).
        if hasattr(data, "to_crs") and self._needs_reproject(data):
            data = data.to_crs(self.crs)
        src = self._to_display_source(data, band=bands[0])
        stack = get_stack(data, bands)
        channels = []
        for index in range(3):
            channel = stack[:, :, index]
            with np.errstate(
                invalid="ignore"
            ):  # an all-nodata channel -> NaN bounds (renders transparent)
                low, high = np.nanpercentile(channel, (2.0, 98.0))
            span = high - low
            # Guard a zero / NaN span (constant or all-nodata channel) so the stretch never divides
            # by zero/NaN for the finite pixels; genuinely-nodata pixels stay NaN (transparent).
            scale = span if np.isfinite(span) and span > 0 else 1.0
            channels.append(np.clip((channel - np.nan_to_num(low)) / scale, 0.0, 1.0))
        element = hv.RGB(
            (src.x.values, src.y.values, *channels),
            kdims=["x", "y"],
            vdims=["R", "G", "B"],
        )
        return self.add_element(self._styled(element, common=opts or None))

    def quadmesh(
        self, data: Any, *, band: int = 1, cmap: Optional[str] = None, **opts: Any
    ) -> "RasterMixin":
        """Add a quadrilateral-mesh raster layer (handles non-uniform / curvilinear coordinates).

        Unlike :meth:`image` (regular grid), a ``QuadMesh`` draws each cell from its coordinate
        arrays, so irregularly spaced or 2-D (curvilinear) coordinates render without resampling.

        Args:
            data: A pyramids ``Dataset`` / ``NetCDF`` / ``Source``; reprojected through pyramids.
            band: 1-based band to render.
            cmap: Colormap name; ``None`` (default) resolves it from the variable via
                ``autostyle.auto_style`` (DI.12) — consistent with :meth:`image`.
            **opts: Extra HoloViews style options applied to the element.

        Examples:
            - Draw an irregular grid without resampling to axis-aligned:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> grid = Dataset.read_file("examples/data/acc4000.tif")       # doctest: +SKIP
                >>> InteractiveMap().quadmesh(grid).save("mesh.html")           # doctest: +SKIP
                'mesh.html'

                ```

        Returns:
            This map (chainable).
        """
        gv, hv = _require_holoviz()
        src = self._to_display_source(data, band=band)
        arr = _masked_to_nan(src.z.values)
        name = self._vdim_name(src)
        element = hv.QuadMesh(
            (src.x.values, src.y.values, arr), kdims=["x", "y"], vdims=[name]
        )
        element = self._styled(
            element,
            common={"cmap": self._auto_cmap(src, cmap), **opts},
            bokeh={"tools": ["hover"]},
        )
        return self.add_element(element)

    def contours(
        self, data: Any, *, band: int = 1, levels: Any = None, **opts: Any
    ) -> "RasterMixin":
        """Add line contours of a raster band.

        Args:
            data: A pyramids ``Dataset`` / ``NetCDF`` / ``Source``; reprojected through pyramids.
            band: 1-based band to contour.
            levels: Contour levels — an int (count) or explicit sequence; ``None`` uses 10.
            **opts: Extra HoloViews style options applied to the element.

        Examples:
            - Contour a raster at five levels:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().contours(dem, levels=5)                # doctest: +SKIP
                >>> len(m.layers)                                               # doctest: +SKIP
                1

                ```

        Returns:
            This map (chainable).
        """
        return self._contour_layer(data, band=band, levels=levels, filled=False, **opts)

    def filled_contours(
        self, data: Any, *, band: int = 1, levels: Any = None, **opts: Any
    ) -> "RasterMixin":
        """Add filled contour bands of a raster band.

        Args:
            data: A pyramids ``Dataset`` / ``NetCDF`` / ``Source``; reprojected through pyramids.
            band: 1-based band to contour.
            levels: Contour levels — an int (count) or explicit sequence; ``None`` uses 10.
            **opts: Extra HoloViews style options applied to the element.

        Examples:
            - Fill the bands between contour levels:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> InteractiveMap().filled_contours(dem, levels=5).save("b.html")  # doctest: +SKIP
                'b.html'

                ```

        Returns:
            This map (chainable).
        """
        return self._contour_layer(data, band=band, levels=levels, filled=True, **opts)

    def _contour_layer(
        self, data: Any, *, band: int, levels: Any, filled: bool, **opts: Any
    ) -> "RasterMixin":
        """Shared contour recipe: I1 image → ``holoviews.operation.contours`` → styled layer."""
        gv, hv = _require_holoviz()
        from holoviews.operation import contours as contour_op

        element = contour_op(
            self._image_element(data, band=band),
            levels=10 if levels is None else levels,
            filled=filled,
        )
        element = self._styled(element, common=opts or None, bokeh={"tools": ["hover"]})
        return self.add_element(element)

    #: Colour cycle used to distinguish ensemble members in :meth:`spaghetti` (Category10-ish).
    _SPAGHETTI_COLORS = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )

    def spaghetti(
        self, collection: Any, *, band: int = 1, **opts: Any
    ) -> "RasterMixin":
        """Overlay each member of a ``DatasetCollection`` as line contours (ensemble spaghetti).

        Each member gets a distinct colour from a cycling palette so the strands are
        distinguishable, unless the caller passes an explicit ``color``/``cmap`` in ``opts``.

        Args:
            collection: A pyramids ``DatasetCollection`` whose members share a grid.
            band: 1-based band contoured in every member.
            **opts: Extra HoloViews style options applied to each member's contour element. An
                explicit ``color`` or ``cmap`` here disables the per-member colour cycle.

        Examples:
            - Overlay a three-member ensemble as spaghetti contours:
                ```python
                >>> from pyramids.dataset.collection import DatasetCollection   # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dc = DatasetCollection.from_files(["examples/data/acc4000.tif"] * 3)  # doctest: +SKIP
                >>> m = InteractiveMap().spaghetti(dc, levels=4)                # doctest: +SKIP
                >>> len(m.layers)                                               # doctest: +SKIP
                3

                ```

        Returns:
            This map (chainable) — one contour layer registered per member.
        """
        cycle_colour = "color" not in opts and "cmap" not in opts
        for index, member in enumerate(collection.datasets):
            member_opts = dict(opts)
            if cycle_colour:
                member_opts["color"] = self._SPAGHETTI_COLORS[
                    index % len(self._SPAGHETTI_COLORS)
                ]
            self.contours(member, band=band, **member_opts)
        return self

    def large_image(
        self,
        dataset: Any,
        *,
        band: int = 1,
        max_pixels: int = 4_000_000,
        dynamic: bool = True,
        cmap: Optional[str] = None,
        **opts: Any,
    ) -> "RasterMixin":
        """Add a large raster / COG by loading only the viewport at a decimated overview (DI.14).

        The raster analogue of the vector Datashader path: instead of materialising a multi-GB raster,
        it reads only the visible window at a suitable overview via **pyramids** and re-reads on pan/
        zoom. All windowing/overview selection is pyramids' (``read_part`` / ``preview`` —
        cloud-native partial reads for remote COGs over ``/vsicurl/`` come for free); this method only
        drives the viewport loop and styles the result.

        Args:
            dataset: A pyramids ``Dataset`` (ideally a COG with overviews). Must expose ``read_part``.
            band: 1-based band to read.
            max_pixels: Pixel budget per rendered frame; the canvas is sized to stay under it.
            dynamic: Re-read the viewport on pan/zoom via a ``RangeXY`` stream (needs a live server);
                ``False`` renders one decimated ``preview`` frame (deterministic — what tests assert).
            cmap: Colormap; ``None`` resolves from the variable via autostyle.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            This map (chainable).

        Raises:
            AttributeError: when ``dataset`` lacks the pyramids COG/overview read surface
                (``read_part``/``preview``) — file a pyramids issue rather than reaching around it.
        """
        import numpy as np

        gv, hv = _require_holoviz()
        if band < 1:
            raise ValueError(f"band is 1-based; got {band!r}")
        if not hasattr(dataset, "read_part") or not hasattr(dataset, "preview"):
            raise AttributeError(
                "large_image needs pyramids' COG/overview read surface (Dataset.read_part / "
                ".preview); upgrade pyramids or use image() for a small raster"
            )
        ds = dataset.to_crs(self.crs) if self._needs_reproject(dataset) else dataset
        side = max(64, int(np.sqrt(max_pixels)))
        read_band = band - 1  # pyramids preview/read_part are 0-based (like Dataset.read_array)

        def _frame(x_range: Any = None, y_range: Any = None) -> Any:
            if (
                x_range is None or y_range is None
            ):  # initial / static frame: a cheap overview preview
                arr = _masked_to_nan(
                    np.asarray(ds.preview(max_size=side, band=read_band))
                )
                bounds = ds.bbox if hasattr(ds, "bbox") else None
            else:
                bbox = (x_range[0], y_range[0], x_range[1], y_range[1])
                arr = _masked_to_nan(
                    np.asarray(
                        ds.read_part(
                            bbox=bbox,
                            dst_width=side,
                            dst_height=side,
                            bbox_crs=self.crs,
                            band=read_band,
                        )
                    )
                )
                bounds = (bbox[0], bbox[1], bbox[2], bbox[3])
            image = hv.Image(arr, bounds=bounds) if bounds else hv.Image(arr)
            return image.opts(cmap=cmap or "viridis", colorbar=True, **opts)

        if not dynamic:
            return self.add_element(_frame())
        from holoviews.streams import RangeXY

        dmap = hv.DynamicMap(_frame, streams=[RangeXY()])
        return self.add_element(dmap)
