"""RasterMixin — raster builders for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``image`` / ``rgb`` / ``quadmesh`` / ``contours`` / ``filled_contours`` / ``spaghetti`` (DI.1a);
``large_image`` viewport loading lands later (DI.14).

Every builder funnels through ``self._to_display_source`` (reproject in **pyramids**, option A), then
emits a **plain HoloViews** element (``hv.Image``/``hv.RGB``/``hv.QuadMesh``) whose coordinates are
already in the display CRS — deliberately *not* ``gv.Image``, whose default PlateCarree ``crs`` would
re-project already-projected coordinates at render time. NoData arrives as a masked array from pyramids
and renders transparent (``NaN``).
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
        name = vname or src.metadata("variable", None) or src.z.name or "value"
        return hv.Image(
            (src.x.values, src.y.values, arr), kdims=["x", "y"], vdims=[name]
        )

    def image(
        self,
        data: Any,
        *,
        band: int = 1,
        cmap: str = "viridis",
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
            cmap: Colormap name.
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
        element = self._image_element(data, band=band)
        element = self._styled(
            element,
            common={
                "cmap": cmap,
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
        if (
            hasattr(data, "epsg")
            and hasattr(data, "to_crs")
            and self._needs_reproject(data)
        ):
            data = data.to_crs(self.crs)
        src = self._to_display_source(data, band=bands[0])
        stack = get_stack(data, bands)
        channels = []
        for index in range(3):
            channel = stack[:, :, index]
            low, high = np.nanpercentile(channel, (2.0, 98.0))
            scale = (high - low) or 1.0
            channels.append(np.clip((channel - low) / scale, 0.0, 1.0))
        element = hv.RGB(
            (src.x.values, src.y.values, *channels),
            kdims=["x", "y"],
            vdims=["R", "G", "B"],
        )
        return self.add_element(self._styled(element, common=opts or None))

    def quadmesh(
        self, data: Any, *, band: int = 1, cmap: str = "viridis", **opts: Any
    ) -> "RasterMixin":
        """Add a quadrilateral-mesh raster layer (handles non-uniform / curvilinear coordinates).

        Unlike :meth:`image` (regular grid), a ``QuadMesh`` draws each cell from its coordinate
        arrays, so irregularly spaced or 2-D (curvilinear) coordinates render without resampling.

        Args:
            data: A pyramids ``Dataset`` / ``NetCDF`` / ``Source``; reprojected through pyramids.
            band: 1-based band to render.
            cmap: Colormap name.
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
        name = src.metadata("variable", None) or src.z.name or "value"
        element = hv.QuadMesh(
            (src.x.values, src.y.values, arr), kdims=["x", "y"], vdims=[name]
        )
        element = self._styled(
            element, common={"cmap": cmap, **opts}, bokeh={"tools": ["hover"]}
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

    def spaghetti(
        self, collection: Any, *, band: int = 1, **opts: Any
    ) -> "RasterMixin":
        """Overlay each member of a ``DatasetCollection`` as line contours (ensemble spaghetti).

        Args:
            collection: A pyramids ``DatasetCollection`` whose members share a grid.
            band: 1-based band contoured in every member.
            **opts: Extra HoloViews style options applied to each member's contour element.

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
        for member in collection.datasets:
            self.contours(member, band=band, **opts)
        return self
