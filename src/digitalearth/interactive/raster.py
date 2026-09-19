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

from typing import TYPE_CHECKING, Any, Dict, Optional, Self, Sequence, Tuple

from digitalearth.base.crs import reproject
from digitalearth.base.sources.view import SourceView
from digitalearth.base.spec import (
    Bounds,
    DataRef,
    RenderTarget,
    Selection,
    Viewport,
)
from digitalearth.base.stretch import (
    DEFAULT_COMPOSITE_BANDS,
    ChannelLimits,
    require_three_bands,
    stretch_to_unit,
)
from digitalearth.interactive.base import (
    _masked_to_nan,
    _require_holoviz,
    _skips_off_limb,
)

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.interactive.base import InteractiveMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


class RasterMixin(_MixinBase):
    """Raster builders (DI.1a): colour-mapped fields, composites and ensemble spaghetti."""

    def _image_from_source(self, src: Any, *, vname: Optional[str] = None) -> Any:
        """Build the I1 image from an already display-CRS :class:`Source`.

        Takes an already-reprojected source rather than the raw data, so a builder that also needs the
        source itself — for the autostyle ``cmap``/``levels``/``units`` lookup (#230) — reprojects once
        instead of twice.

        Args:
            src: The display-CRS source.
            vname: Value-dimension name; defaults to the source's variable/z name.

        Returns:
            holoviews.Image: the raster as a plain HoloViews image in the display CRS.

            A view that read a window is placed on the rectangle it read, rather than on one derived from
            its axes: a zoom that lands on a single cell leaves an axis with no spacing to derive from,
            and HoloViews answers that with `nan` bounds and a raised frame (#300).
        """
        gv, hv = _require_holoviz()
        arr = _masked_to_nan(src.z.values)
        name = vname or self._vdim_name(src)
        window = getattr(src, "window", None)
        return self._raster_element(
            src.x.values,
            src.y.values,
            arr,
            name,
            bounds=None if window is None else window.as_bbox(),
        )

    @_skips_off_limb
    def image(
        self,
        data: Any,
        *,
        band: int = 1,
        cmap: Optional[str] = None,
        clim: Optional[Tuple[float, float]] = None,
        alpha: float = 1.0,
        colorbar: bool = True,
        clabel: Optional[str] = None,
        **opts: Any,
    ) -> Self:
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
            clabel: Colorbar label; ``None`` (default) takes the variable's ``units`` from
                ``autostyle.auto_style`` (#230) and leaves the colorbar unlabelled when it knows none.
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
        element = self._styled(
            self._image_from_source(src),
            common={
                "cmap": self._auto_cmap(src, cmap),
                "clim": clim,
                "alpha": alpha,
                "colorbar": colorbar,
                "clabel": self._auto_clabel(src, clabel),
                **opts,
            },
            bokeh={"tools": ["hover"]},
        )
        return self.add_element(element)

    @_skips_off_limb
    def rgb(
        self,
        data: Any,
        *,
        bands: Sequence[int] = DEFAULT_COMPOSITE_BANDS,
        limits: Optional[ChannelLimits] = None,
        **opts: Any,
    ) -> Self:
        """Add a true-colour composite from three raster bands (2–98 % percentile stretch).

        Args:
            data: A pyramids multiband ``Dataset``; reprojected to the display CRS first.
            bands: The three 1-based band indices composing ``(R, G, B)``.
            limits: Optional frozen ``(lo, hi)`` stretch bounds, one pair per channel — skips the per-call
                percentile scan, so a sequence of frames can share one black and white point.
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
            ValueError: when ``bands`` does not name exactly three bands, or ``limits`` is given without
                one ``(lo, hi)`` pair per channel.
        """
        from digitalearth.base.sources import get_stack

        gv, hv = _require_holoviz()
        require_three_bands("rgb", bands)
        # Reproject once into a local handle, then feed both the coordinate extraction (get_source,
        # via _to_display_source) and the band stack (get_stack) from it — get_stack needs the same
        # already-reprojected dataset, so a single warp here keeps them consistent (H1).
        if hasattr(data, "to_crs") and self._needs_reproject(data):
            data = reproject(data, self.crs)
        src = self._to_display_source(data, band=bands[0])
        stack = get_stack(data, bands)
        # One shared stretch for every backend (base/stretch.py). Passing `limits` holds it fixed across a
        # sequence of frames, the same way the matplotlib tier freezes an animation.
        stretched = stretch_to_unit(stack, limits)
        channels = [stretched[:, :, index] for index in range(3)]
        element = hv.RGB(
            (src.x.values, src.y.values, *channels),
            kdims=["x", "y"],
            vdims=["R", "G", "B"],
        )
        return self.add_element(self._styled(element, common=opts or None))

    @_skips_off_limb
    def quadmesh(
        self,
        data: Any,
        *,
        band: int = 1,
        cmap: Optional[str] = None,
        clabel: Optional[str] = None,
        **opts: Any,
    ) -> Self:
        """Add a quadrilateral-mesh raster layer (handles non-uniform / curvilinear coordinates).

        Unlike :meth:`image` (regular grid), a ``QuadMesh`` draws each cell from its coordinate
        arrays, so irregularly spaced or 2-D (curvilinear) coordinates render without resampling.

        Args:
            data: A pyramids ``Dataset`` / ``NetCDF`` / ``Source``; reprojected through pyramids.
            band: 1-based band to render.
            cmap: Colormap name; ``None`` (default) resolves it from the variable via
                ``autostyle.auto_style`` (DI.12) — consistent with :meth:`image`.
            clabel: Colorbar label; ``None`` (default) takes the variable's ``units`` from
                ``autostyle.auto_style`` (#230), as :meth:`image` does.
            **opts: Extra HoloViews style options applied to the element.

        Examples:
            - Draw an irregular grid without resampling to axis-aligned:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> grid = Dataset.read_file("examples/data/acc4000.tif")       # doctest: +SKIP
                >>> InteractiveMap().quadmesh(grid).save("mesh.html").name      # doctest: +SKIP
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
            common={
                "cmap": self._auto_cmap(src, cmap),
                "clabel": self._auto_clabel(src, clabel),
                **opts,
            },
            bokeh={"tools": ["hover"]},
        )
        return self.add_element(element)

    @_skips_off_limb
    def contours(
        self, data: Any, *, band: int = 1, levels: Any = None, **opts: Any
    ) -> Self:
        """Add line contours of a raster band.

        Args:
            data: A pyramids ``Dataset`` / ``NetCDF`` / ``Source``; reprojected through pyramids.
            band: 1-based band to contour.
            levels: Contour levels — an int (count) or explicit sequence; ``None`` takes the
                variable's canonical levels from ``autostyle.auto_style`` (#230), falling back to 10.
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

    @_skips_off_limb
    def filled_contours(
        self, data: Any, *, band: int = 1, levels: Any = None, **opts: Any
    ) -> Self:
        """Add filled contour bands of a raster band.

        Args:
            data: A pyramids ``Dataset`` / ``NetCDF`` / ``Source``; reprojected through pyramids.
            band: 1-based band to contour.
            levels: Contour levels — an int (count) or explicit sequence; ``None`` takes the
                variable's canonical levels from ``autostyle.auto_style`` (#230), falling back to 10.
            **opts: Extra HoloViews style options applied to the element.

        Examples:
            - Fill the bands between contour levels:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().filled_contours(dem, levels=5)         # doctest: +SKIP
                >>> m.save("b.html").name                                      # doctest: +SKIP
                'b.html'

                ```

        Returns:
            This map (chainable).
        """
        return self._contour_layer(data, band=band, levels=levels, filled=True, **opts)

    def _contour_layer(
        self, data: Any, *, band: int, levels: Any, filled: bool, **opts: Any
    ) -> Self:
        """Shared contour recipe: I1 image → ``holoviews.operation.contours`` → styled layer.

        A caller's ``levels`` always wins; ``None`` consults ``autostyle.auto_style`` for the variable's
        canonical contour levels (#230) before falling back to the tier's 10.
        """
        gv, hv = _require_holoviz()
        from holoviews.operation import contours as contour_op

        src = self._to_display_source(data, band=band)
        resolved = self._auto_levels(src, levels)
        element = contour_op(
            self._image_from_source(src),
            levels=10 if resolved is None else resolved,
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

    def spaghetti(self, collection: Any, *, band: int = 1, **opts: Any) -> Self:
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

    @_skips_off_limb
    def large_image(
        self,
        dataset: Any,
        *,
        band: int = 1,
        max_pixels: int = 4_000_000,
        dynamic: bool = True,
        cmap: Optional[str] = None,
        **opts: Any,
    ) -> Self:
        """Add a large raster / COG by loading only the viewport at a decimated overview (DI.14).

        The raster analogue of the vector Datashader path: instead of materialising a multi-GB raster,
        it reads only the visible window at a suitable overview via **pyramids** and re-reads on pan/
        zoom. Cloud-native partial reads for remote COGs over ``/vsicurl/`` come for free.

        **Every frame is a read through the data tier** (#297): the window, the canvas and the budget are a
        :class:`~digitalearth.base.spec.RenderTarget` request, and
        :class:`~digitalearth.base.sources.view.SourceView` answers it — snapping the window to source pixel
        edges, never asking for more cells than the window holds, masking what is missing, and drawing an
        empty frame for a window off the data. A raster in another CRS is read through pyramids'
        ``Dataset.warped_view``, so each frame warps only its own window rather than the whole raster once.

        Args:
            dataset: A pyramids ``Dataset`` (ideally a COG with overviews). Must expose ``read_part``.
            band: 1-based band to read.
            max_pixels: Cell budget per rendered frame. The canvas follows the map's own width and height
                within it, so a wide map reads a wide window rather than a square one.
            dynamic: Re-read the viewport on pan/zoom via a ``RangeXY`` stream (needs a live server);
                ``False`` renders one frame of the whole raster within the budget (deterministic — what
                tests assert).
            cmap: Colormap; ``None`` (default) resolves from the band's variable name via
                ``autostyle.auto_style`` (#249), with ``"viridis"`` behind the lookup as the fallback.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            This map (chainable).

        Raises:
            ValueError: when ``band`` is below 1 (the tier's band numbering is 1-based, like GDAL).
            AttributeError: when ``dataset`` lacks the pyramids COG/overview read surface
                (``read_part``/``preview``) — file a pyramids issue rather than reaching around it.

        Examples:
            - ``dynamic=False`` reads one frame of the whole raster — deterministic, and the one
              form that works with no live server behind it. A raster already under the budget comes back
              whole, so the budget only ever *caps* what is materialised:
                ```python
                >>> from pyramids.dataset import Dataset                       # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")       # doctest: +SKIP
                >>> (dem.rows, dem.columns)                                    # doctest: +SKIP
                (13, 14)
                >>> m = InteractiveMap().large_image(dem, dynamic=False)       # doctest: +SKIP
                >>> m.layers[0].dimension_values(2, flat=False).shape          # doctest: +SKIP
                (13, 14)

                ```
            - The default ``dynamic=True`` registers a viewport-driven layer instead, carrying the
              one stream that re-reads the window from the axes ranges on every pan/zoom:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> m = InteractiveMap().large_image(dem)                      # doctest: +SKIP
                >>> len(m.layers[0].streams)                                   # doctest: +SKIP
                1
                >>> sorted(m.layers[0].streams[0].contents)                    # doctest: +SKIP
                ['x_range', 'y_range']

                ```
            - A raster without pyramids' windowed-read surface is refused by name, rather than
              being quietly read whole:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> try:                                                       # doctest: +SKIP
                ...     InteractiveMap().large_image(object())
                ... except AttributeError as error:
                ...     print(str(error).split(" (")[0])
                large_image needs pyramids' COG/overview read surface

                ```
        """
        gv, hv = _require_holoviz()
        if band < 1:
            raise ValueError(f"band is 1-based; got {band!r}")
        if not hasattr(dataset, "read_part") or not hasattr(dataset, "preview"):
            raise AttributeError(
                "large_image needs pyramids' COG/overview read surface (Dataset.read_part / "
                ".preview); upgrade pyramids or use image() for a small raster"
            )
        ds = self._windowable(dataset)
        # Resolved once, from the band's name only: reading the array to build a full Source would
        # defeat the whole point of a windowed reader.
        cmap = self._auto_cmap_for_band(ds, band, cmap)
        target = RenderTarget(
            "window", width=self.width, height=self.height, budget=max_pixels
        )
        view = SourceView.of(
            ds,
            ref=DataRef.to_object(ds),
            selection=Selection.of(band),
            request=target.view_request(Viewport(self.crs)),
        )

        # Filled in below with the `DynamicMap` the frames belong to, so each frame's style is filed
        # against the layer a caller holds rather than against whichever layer happened to register last
        # (review H3). It stays `None` for the static path, where the frame *is* the layer.
        owner: Dict[str, Any] = {}

        def _frame(x_range: Any = None, y_range: Any = None) -> Any:
            """Read the window the viewport asks for, and draw it.

            Args:
                x_range: The viewport's x range, or `None` for the first frame.
                y_range: Its y range, or `None`.

            Returns:
                The styled element for that window.
            """
            if x_range is None or y_range is None:
                # The first frame is the whole raster within the budget: a canvas with no region windows
                # the source itself, which is what `preview` used to approximate.
                shown = view
            else:
                shown = view.reread(
                    target.view_request(
                        Viewport(self.crs),
                        bounds=Bounds(
                            x_range[0], y_range[0], x_range[1], y_range[1], crs=self.crs
                        ),
                    )
                )
            return self._styled(
                self._image_from_source(shown),
                common={"cmap": cmap, "colorbar": True, **opts},
                bokeh={"tools": ["hover"]},
                owner=owner.get("layer"),
            )

        if not dynamic:
            return self.add_element(_frame())
        from holoviews.streams import RangeXY

        dmap = hv.DynamicMap(_frame, streams=[RangeXY()])
        owner["layer"] = dmap
        return self.add_element(dmap)

    def _windowable(self, dataset: Any) -> Any:
        """Return the raster to read windows from, warped lazily when the display CRS asks for it.

        Args:
            dataset: The raster the caller gave.

        Returns:
            The dataset itself when it is already in the display CRS; otherwise pyramids' lazily warped view
            of it (`Dataset.warped_view`), so each frame warps only the window it reads rather than the whole
            raster once, up front, at full resolution. A pyramids without `warped_view` falls back to warping
            the dataset, which is what this did before.
        """
        if not self._needs_reproject(dataset):
            return dataset
        warped_view = getattr(dataset, "warped_view", None)
        if warped_view is None:  # pragma: no cover - older pyramids
            return reproject(dataset, self.crs)
        return warped_view(self.crs)
