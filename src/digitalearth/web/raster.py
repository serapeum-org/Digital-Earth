"""RasterMixin — web-tier raster builder (DW.1b, recipe W1).

``add_raster`` puts a pyramids raster on the web map as a MapLibre **image source**: the band is reprojected
to lon/lat through pyramids, colour-mapped to an RGBA PNG (NoData → transparent), embedded as a ``data:`` URI,
and placed by its lon/lat corner coordinates. This is the offline, size-limited path; the large-raster
COG/XYZ-tile path (pyramids ``to_cog``/``to_xyz``) is a follow-up — it needs a tile server or PMTiles and is
tracked in the plan (DW.1b risks).

matplotlib (the colormap → RGBA → PNG encoding) and numpy are imported lazily inside the methods, so importing
the tier needs neither the ``web`` extra nor matplotlib at module load.
"""

from typing import TYPE_CHECKING, Any, List, Optional, Self

from loguru import logger

from digitalearth.web.base import _require_layer_api

#: Pixel count above which the inline image-source path is warned against (use COG/XYZ tiles for big rasters).
_LARGE_RASTER_PIXELS = 4_000_000


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.web.base import WebMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


class RasterMixin(_MixinBase):
    """Raster builder for :class:`~digitalearth.web.map.WebMap` (image-source path)."""

    def add_raster(
        self,
        data: Any,
        *,
        band: int = 1,
        cmap: Optional[str] = None,
        opacity: float = 1.0,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        visible: bool = True,
    ) -> Self:
        """Overlay a pyramids raster band as a colour-mapped MapLibre image source (recipe W1).

        The band is reprojected to the display CRS (lon/lat) through pyramids, normalised over its finite
        range (or the explicit ``vmin``/``vmax``), colour-mapped with ``cmap`` (autostyle default when
        ``None``), and embedded as an RGBA PNG data-URI placed by its lon/lat corners. Masked / non-finite
        cells become fully transparent.

        Args:
            data: A pyramids ``Dataset`` (or anything ``get_source`` accepts).
            band: 1-based band to draw.
            cmap: matplotlib colormap name; ``None`` resolves the autostyle default for the variable.
            opacity: Raster layer opacity in ``[0, 1]``.
            vmin: Lower colour limit; ``None`` uses the band's finite minimum.
            vmax: Upper colour limit; ``None`` uses the band's finite maximum.
            visible: Whether the layer starts visible. ``False`` builds it hidden, which is how
                :meth:`~digitalearth.web.temporal.TemporalMixin.timeslider` stacks time steps without
                every frame showing at once — including in a saved page, which carries no slider.

        Returns:
            This map (chainable).
        """
        import numpy as np

        Layer, LayerType = _require_layer_api()
        source = self._to_display_source(data, band=band)
        cmap_name = self._auto_cmap(source, cmap)

        values = source.z.values
        if getattr(values, "size", 0) > _LARGE_RASTER_PIXELS:
            logger.warning(
                "add_raster: inlining a {}-pixel band as a data-URI image source bloats the page; for large "
                "rasters serve COG/XYZ tiles from pyramids instead",
                getattr(values, "size", 0),
            )
        y = np.asarray(source.y.values, dtype=float)
        if (
            y.size > 1 and y[0] < y[-1]
        ):  # ascending y → flip so PNG row 0 is the northern edge
            values = values[::-1]
        url = self._rgba_png_datauri(values, cmap_name, vmin=vmin, vmax=vmax)
        coordinates = self._image_coordinates(source.x.values, source.y.values)
        # [TL, TR, BR, BL] -> (west, south, east, north), so a raster frames the map like a vector layer.
        self._note_bounds(
            (coordinates[0][0], coordinates[2][1], coordinates[1][0], coordinates[0][1])
        )

        src_id, layer_id = self._uid("raster-src"), self._uid("raster")
        spec = {"type": "image", "url": url, "coordinates": coordinates}
        layer = Layer(
            id=layer_id,
            type=LayerType.RASTER,
            source=src_id,
            paint={"raster-opacity": float(opacity)},
            layout={"visibility": "visible" if visible else "none"},
        )

        def apply(widget: Any) -> None:
            widget.add_source(src_id, spec)
            widget.add_layer(layer)

        self._last_layer_id = layer_id
        return self.add_layer(apply)

    def rgb_composite(
        self,
        dataset: Any,
        bands: Any = (1, 2, 3),
        *,
        mask_nodata: bool = True,
        limits: Optional[Any] = None,
        opacity: float = 1.0,
        visible: bool = True,
        name: Optional[str] = None,
    ) -> Self:
        """Overlay three bands as a true- or false-colour image (recipe W1).

        Satellite imagery is a headline use of a web map, and the tier could only draw one band through a
        colormap. The stretch comes from :mod:`digitalearth.base.stretch` — the same engine-neutral code
        the static tier's ``rgb_composite`` uses — so the same three bands look the same on both tiers.

        Args:
            dataset: A pyramids ``Dataset`` (or anything ``get_stack`` accepts).
            bands: The three 1-based band numbers, in red-green-blue order.
            mask_nodata: Whether NoData becomes NaN (and so transparent) rather than a real value.
            limits: Per-channel ``(lo, hi)`` stretch limits in band order. ``None`` derives them from this
                image; pass a fixed set to keep a series comparable across frames.
            opacity: Raster layer opacity in ``[0, 1]``.
            visible: Whether the layer starts visible, which is what a layer switcher toggles.
            name: What a layer switcher calls this layer; ``None`` uses its generated id.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``bands`` is not exactly three, when ``limits`` does not match them, or when
                the composite has no finite pixels to draw.

        Examples:
            - A true-colour composite from a Landsat-ordered dataset:
                ```python
                >>> from digitalearth.web import WebMap                       # doctest: +SKIP
                >>> WebMap().basemap().rgb_composite(ds, bands=(4, 3, 2))     # doctest: +SKIP

                ```

        See Also:
            digitalearth.base.stretch.channel_limits: derives the ``limits`` this accepts.
            digitalearth.web.raster.RasterMixin.add_raster: the single-band, colormapped path.
        """
        import numpy as np

        from digitalearth.base.sources import get_stack
        from digitalearth.base.stretch import require_three_bands, stretch_to_unit

        Layer, LayerType = _require_layer_api()
        require_three_bands("rgb_composite", bands)
        data = self._to_display_raster(dataset)
        stack = get_stack(data, bands, mask=mask_nodata)
        if stack.size > _LARGE_RASTER_PIXELS * 3:
            logger.warning(
                "rgb_composite: inlining a {}-pixel composite as a data-URI image source bloats the page; "
                "for large rasters serve COG/XYZ tiles from pyramids instead",
                stack.size,
            )
        source = self._to_display_source(dataset, band=int(bands[0]))
        y = np.asarray(source.y.values, dtype=float)
        if (
            y.size > 1 and y[0] < y[-1]
        ):  # ascending y → flip so PNG row 0 is the north edge
            stack = stack[::-1]
        url = self._composite_png_datauri(stretch_to_unit(stack, limits))
        coordinates = self._image_coordinates(source.x.values, source.y.values)
        self._note_bounds(
            (coordinates[0][0], coordinates[2][1], coordinates[1][0], coordinates[0][1])
        )
        src_id, layer_id = self._uid("rgb-src"), self._uid("rgb")
        spec = {"type": "image", "url": url, "coordinates": coordinates}
        layer = Layer(
            id=layer_id,
            type=LayerType.RASTER,
            source=src_id,
            paint={"raster-opacity": float(opacity)},
            layout={"visibility": "visible" if visible else "none"},
        )

        def apply(widget: Any) -> None:
            widget.add_source(src_id, spec)
            widget.add_layer(layer)

        apply._digitalearth_layer_id = layer_id  # type: ignore[attr-defined]
        self._last_layer_id = layer_id
        self._index_layer(layer_id, name)
        return self.add_layer(apply)

    def _to_display_raster(self, dataset: Any) -> Any:
        """Return ``dataset`` in the display CRS, reprojected through pyramids when it is not already.

        ``_to_display_source`` gives one band; a composite needs the dataset itself so ``get_stack`` can
        read three from it.

        Args:
            dataset: A pyramids ``Dataset``.

        Returns:
            The dataset in the display CRS.
        """
        if hasattr(dataset, "to_crs") and self._needs_reproject(dataset):
            return dataset.to_crs(self.crs)
        return dataset

    @staticmethod
    def _composite_png_datauri(unit_stack: Any) -> str:
        """Encode a stretched ``(rows, cols, 3)`` stack as a ``data:image/png;base64,`` URI.

        Args:
            unit_stack: Channel values already stretched to ``[0, 1]``; NaN marks NoData.

        Returns:
            The PNG data-URI string.

        Raises:
            ValueError: when no pixel is finite in all three channels — there is nothing to draw, and an
                empty image would look like a rendering failure instead of an empty input.
        """
        import base64
        import io

        import numpy as np
        from matplotlib import image as mpimage

        stack = np.asarray(unit_stack, dtype=float)
        valid = np.isfinite(stack).all(axis=-1)
        if not valid.any():
            raise ValueError(
                "rgb_composite got a stack with no pixel finite in all three bands"
            )
        rgba = np.zeros(stack.shape[:2] + (4,), dtype=float)
        rgba[..., :3] = np.clip(np.where(np.isfinite(stack), stack, 0.0), 0.0, 1.0)
        rgba[..., 3] = valid.astype(float)  # NoData in any channel → transparent
        buffer = io.BytesIO()
        mpimage.imsave(buffer, (rgba * 255).astype("uint8"), format="png")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    @staticmethod
    def _image_coordinates(x: Any, y: Any) -> List[List[float]]:
        """Return the image-source corner coordinates ``[TL, TR, BR, BL]`` in ``[lng, lat]``.

        Args:
            x: 1-D x / longitude cell-centre coordinates (display CRS, lon/lat).
            y: 1-D y / latitude cell-centre coordinates.

        Returns:
            The four corners top-left, top-right, bottom-right, bottom-left as ``[lng, lat]`` pairs — the
            order MapLibre's image source expects.
        """
        import numpy as np

        xs = np.asarray(x, dtype=float)
        ys = np.asarray(y, dtype=float)
        west, east = float(xs.min()), float(xs.max())
        south, north = float(ys.min()), float(ys.max())
        return [[west, north], [east, north], [east, south], [west, south]]

    @staticmethod
    def _rgba_png_datauri(
        values: Any,
        cmap: str,
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> str:
        """Colour-map a 2-D array to an RGBA PNG and return it as a ``data:image/png;base64,`` URI.

        Non-finite / masked cells are rendered fully transparent (the tier's NoData contract). matplotlib
        is imported lazily here so the tier imports without it.

        Args:
            values: A 2-D (possibly masked) array of band values, already oriented north-up.
            cmap: matplotlib colormap name.
            vmin: Lower colour limit; ``None`` uses the finite minimum.
            vmax: Upper colour limit; ``None`` uses the finite maximum.

        Returns:
            The PNG data-URI string.

        Raises:
            ValueError: when the array has no finite values to colour.
        """
        import base64
        import io

        import numpy as np
        from matplotlib import colormaps
        from matplotlib import image as mpimage
        from matplotlib.colors import Normalize

        array = np.ma.asarray(values).astype(float)
        data = (
            array.filled(np.nan)
            if np.ma.isMaskedArray(array)
            else np.asarray(array, dtype=float)
        )
        valid = np.isfinite(data)
        if not valid.any():
            raise ValueError("add_raster got a band with no finite values to colour")
        lo = float(np.nanmin(data)) if vmin is None else float(vmin)
        hi = float(np.nanmax(data)) if vmax is None else float(vmax)
        if hi <= lo:  # constant band — widen so Normalize stays valid
            hi = lo + 1.0
        norm = Normalize(vmin=lo, vmax=hi)
        rgba = colormaps[cmap](norm(np.where(valid, data, lo)))
        rgba[~valid, 3] = 0.0  # NoData → transparent
        rgba8 = (rgba * 255).astype("uint8")

        buffer = io.BytesIO()
        mpimage.imsave(buffer, rgba8, format="png")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
