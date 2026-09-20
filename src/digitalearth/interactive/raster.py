"""RasterMixin — raster builders for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``image`` / ``rgb`` / ``quadmesh`` / ``contours`` / ``filled_contours`` / ``spaghetti`` (DI.1a);
``large_image`` viewport loading lands later (DI.14).

Every builder records what it draws; its drawer funnels through ``_to_display_source`` (reproject in
**pyramids**, option A), then emits a **plain HoloViews** element (``hv.Image``/``hv.RGB``/``hv.QuadMesh``)
whose coordinates are already in the display CRS — deliberately *not* ``gv.Image``, whose default
PlateCarree ``crs`` would re-project already-projected coordinates at render time. NoData arrives as a
masked array from pyramids and renders transparent (``NaN``).

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
    LayerSpec,
    RenderTarget,
    Selection,
    Symbology,
    Viewport,
)
from digitalearth.base.spec._serial import thawed_value
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


def draw_image(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the colour-mapped raster a described image layer asks for.

    Args:
        interactive_map: The map being drawn.
        data: The raster the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    props = dict(layer.symbology.props)
    src = interactive_map._to_display_source(data, band=props["band"])
    common = {
        "cmap": interactive_map._auto_cmap(src, props.get("cmap")),
        "clim": props.get("clim"),
        "alpha": props.get("alpha"),
        "colorbar": props.get("colorbar"),
        "clabel": interactive_map._auto_clabel(src, props.get("clabel")),
        **dict(props.get("opts") or {}),
    }
    element = interactive_map._styled(
        interactive_map._image_from_source(src),
        common=common,
        bokeh={"tools": ["hover"]},
    )
    return DrawnLayer(element=element, style=common)


def draw_rgb(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Compose the three-band true-colour image a described RGB layer asks for.

    Args:
        interactive_map: The map being drawn.
        data: The multiband raster the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.

    Raises:
        ValueError: when the recorded ``limits`` do not hold one ``(lo, hi)`` pair per channel. The band
            count is refused by the builder, which names the caller's own argument; the limits are refused
            here, where the stretch that reads them runs.
    """
    from digitalearth.base.sources import get_stack
    from digitalearth.interactive.renderer import DrawnLayer

    _, hv = _require_holoviz()
    props = dict(layer.symbology.props)
    bands = list(props["bands"])
    # Reproject once into a local handle, then feed both the coordinate extraction (get_source, via
    # _to_display_source) and the band stack (get_stack) from it — get_stack needs the same
    # already-reprojected dataset, so a single warp here keeps them consistent (H1).
    if hasattr(data, "to_crs") and interactive_map._needs_reproject(data):
        data = reproject(data, interactive_map.crs)
    src = interactive_map._to_display_source(data, band=bands[0])
    # One shared stretch for every backend (base/stretch.py). A recorded `limits` holds it fixed across a
    # sequence of frames, the same way the matplotlib tier freezes an animation.
    limits = props.get("limits")
    stretched = stretch_to_unit(get_stack(data, bands), limits)
    channels = [stretched[:, :, index] for index in range(3)]
    element = hv.RGB(
        (src.x.values, src.y.values, *channels),
        kdims=["x", "y"],
        vdims=["R", "G", "B"],
    )
    common = dict(props.get("opts") or {})
    return DrawnLayer(
        element=interactive_map._styled(element, common=common or None), style=common
    )


def draw_quadmesh(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the quadrilateral mesh a described mesh layer asks for.

    Args:
        interactive_map: The map being drawn.
        data: The raster the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    _, hv = _require_holoviz()
    props = dict(layer.symbology.props)
    src = interactive_map._to_display_source(data, band=props["band"])
    arr = _masked_to_nan(src.z.values)
    element = hv.QuadMesh(
        (src.x.values, src.y.values, arr),
        kdims=["x", "y"],
        vdims=[interactive_map._vdim_name(src)],
    )
    common = {
        "cmap": interactive_map._auto_cmap(src, props.get("cmap")),
        "clabel": interactive_map._auto_clabel(src, props.get("clabel")),
        **dict(props.get("opts") or {}),
    }
    element = interactive_map._styled(
        element, common=common, bokeh={"tools": ["hover"]}
    )
    return DrawnLayer(element=element, style=common)


def draw_contours(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Trace the contours a described contour layer asks for, filled or as lines.

    Args:
        interactive_map: The map being drawn.
        data: The raster the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    # Guarded before holoviews is reached, so a missing stack is refused by the message that names the
    # extra to install rather than by holoviews' own ImportError.
    _require_holoviz()
    from holoviews.operation import contours as contour_op

    props = dict(layer.symbology.props)
    src = interactive_map._to_display_source(data, band=props["band"])
    # A caller's `levels` always wins; `None` consults autostyle for the variable's canonical contour
    # levels (#230) before falling back to the tier's 10.
    resolved = interactive_map._auto_levels(src, props.get("levels"))
    element = contour_op(
        interactive_map._image_from_source(src),
        levels=10 if resolved is None else thawed_value(resolved),
        filled=props.get("filled", False),
    )
    common = dict(props.get("opts") or {})
    element = interactive_map._styled(
        element, common=common or None, bokeh={"tools": ["hover"]}
    )
    return DrawnLayer(element=element, style=common)


def draw_large_image(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the windowed-read layer a described large-raster layer asks for.

    Args:
        interactive_map: The map being drawn.
        data: The raster the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    props = dict(layer.symbology.props)
    element = interactive_map._draw_large_image(data, props)
    return DrawnLayer(element=element, style=dict(props.get("opts") or {}))


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
        # Called for its actionable ImportError; the element itself comes from `_raster_element`, which is
        # the one place either engine module is named.
        _require_holoviz()
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
        return self.add_element(
            None,
            kind="raster",
            source=data,
            symbology=Symbology(
                props={
                    "via": "image",
                    "band": band,
                    "cmap": cmap,
                    "clim": clim,
                    "alpha": alpha,
                    "colorbar": colorbar,
                    "clabel": clabel,
                    "opts": dict(opts),
                }
            ),
        )

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
        _require_holoviz()
        # Refused here rather than in the drawer, because the message names the argument the caller wrote.
        require_three_bands("rgb", bands)
        return self.add_element(
            None,
            kind="rgb",
            source=data,
            symbology=Symbology(
                props={
                    "via": "rgb",
                    "bands": tuple(bands),
                    "limits": limits,
                    "opts": dict(opts),
                }
            ),
        )

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
        _require_holoviz()
        return self.add_element(
            None,
            kind="mesh",
            source=data,
            symbology=Symbology(
                props={
                    "via": "quadmesh",
                    "band": band,
                    "cmap": cmap,
                    "clabel": clabel,
                    "opts": dict(opts),
                }
            ),
        )

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
        """Record the shared contour recipe: I1 image → ``holoviews.operation.contours`` → styled layer.

        The one place :meth:`contours` and :meth:`filled_contours` describe their layer, so the two cannot
        drift in what they record; :func:`draw_contours` traces it. A caller's ``levels`` always wins there;
        ``None`` consults ``autostyle.auto_style`` for the variable's canonical contour levels (#230) before
        falling back to the tier's 10.

        Args:
            data: A pyramids ``Dataset`` / ``NetCDF`` / ``Source``; reprojected through pyramids.
            band: 1-based band to contour.
            levels: Contour levels — an int (count) or explicit sequence; ``None`` auto-resolves.
            filled: Whether the bands between the levels are filled, which is also what the layer's kind
                records: ``"filled_contours"`` against ``"contours"``.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            The same map instance, so builder calls chain.
        """
        _require_holoviz()
        return self.add_element(
            None,
            kind="filled_contours" if filled else "contours",
            source=data,
            symbology=Symbology(
                props={
                    "via": "contours",
                    "band": band,
                    "levels": levels,
                    "filled": filled,
                    "opts": dict(opts),
                }
            ),
        )

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
        _require_holoviz()
        # Both refusals stay with the builder, because each names an argument the caller wrote.
        if band < 1:
            raise ValueError(f"band is 1-based; got {band!r}")
        if not hasattr(dataset, "read_part") or not hasattr(dataset, "preview"):
            raise AttributeError(
                "large_image needs pyramids' COG/overview read surface (Dataset.read_part / "
                ".preview); upgrade pyramids or use image() for a small raster"
            )
        return self.add_element(
            None,
            kind="raster",
            source=dataset,
            symbology=Symbology(
                props={
                    "via": "large_image",
                    "band": band,
                    "max_pixels": max_pixels,
                    "dynamic": dynamic,
                    "cmap": cmap,
                    "opts": dict(opts),
                }
            ),
        )

    def _draw_large_image(self, dataset: Any, props: dict) -> Any:
        """Build the windowed-read layer a `large_image` description asks for.

        Args:
            dataset: The raster the layer draws.
            props: The layer's recorded properties.

        Returns:
            The element — a single frame, or the `DynamicMap` that re-reads the window on pan and zoom.
        """
        _, hv = _require_holoviz()
        band = props["band"]
        opts = dict(props.get("opts") or {})
        dynamic = props.get("dynamic", True)
        ds = self._windowable(dataset)
        # Resolved once, from the band's name only: reading the array to build a full Source would
        # defeat the whole point of a windowed reader.
        cmap = self._auto_cmap_for_band(ds, band, props.get("cmap"))
        target = RenderTarget(
            "window", width=self.width, height=self.height, budget=props["max_pixels"]
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
            return _frame()
        from holoviews.streams import RangeXY

        dmap = hv.DynamicMap(_frame, streams=[RangeXY()])
        owner["layer"] = dmap
        # Drawn once here as well as on every range event: a `DynamicMap` is lazy, so without this the
        # layer's first frame — and the style filed against it — would not exist until something rendered.
        _frame()
        return dmap

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
