"""RasterMixin — web-tier raster builder (DW.1b, recipe W1).

``add_raster`` puts a pyramids raster on the web map as a MapLibre **image source**: the band is reprojected
to lon/lat through pyramids, colour-mapped to an RGBA PNG (NoData → transparent), embedded as a ``data:`` URI,
and placed by its lon/lat corner coordinates. This is the offline, size-limited path; the large-raster
COG/XYZ-tile path (pyramids ``to_cog``/``to_xyz``) is a follow-up — it needs a tile server or PMTiles and is
tracked in the plan (DW.1b risks).

matplotlib (the colormap → RGBA → PNG encoding) and numpy are imported lazily inside the methods, so importing
the tier needs neither the ``web`` extra nor matplotlib at module load.
"""

from typing import Any, List, Optional

from loguru import logger

from digitalearth.web.base import _require_layer_api

#: Pixel count above which the inline image-source path is warned against (use COG/XYZ tiles for big rasters).
_LARGE_RASTER_PIXELS = 4_000_000


class RasterMixin:
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
    ) -> "RasterMixin":
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
        if y.size > 1 and y[0] < y[-1]:  # ascending y → flip so PNG row 0 is the northern edge
            values = values[::-1]
        url = self._rgba_png_datauri(values, cmap_name, vmin=vmin, vmax=vmax)
        coordinates = self._image_coordinates(source.x.values, source.y.values)

        src_id, layer_id = self._uid("raster-src"), self._uid("raster")
        spec = {"type": "image", "url": url, "coordinates": coordinates}
        layer = Layer(
            id=layer_id,
            type=LayerType.RASTER,
            source=src_id,
            paint={"raster-opacity": float(opacity)},
        )

        def apply(widget: Any) -> None:
            widget.add_source(src_id, spec)
            widget.add_layer(layer)

        self._last_layer_id = layer_id
        return self.add_layer(apply)

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
        data = array.filled(np.nan) if np.ma.isMaskedArray(array) else np.asarray(array, dtype=float)
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
