"""RasterMixin — raster field rendering: imshow/contourf/contour/pcolormesh/block, composites, spaghetti.

Wires a pyramids ``Dataset`` (reprojected to the display CRS by the base) into cleopatra ``ArrayGlyph`` field
renders, plus the RGB/HSV composites and the ensemble spaghetti overlay.
"""
from typing import Any, List, Optional, Sequence

import numpy as np
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph, RgbBands

from digitalearth._render_compat import relocate_flat_style
from digitalearth.autostyle import auto_style
from digitalearth.preprocess import add_cyclic_column
from digitalearth.sources import get_stack


def _stretch_to_unit(stack: np.ndarray) -> np.ndarray:
    """Per-channel 2-98 percentile contrast stretch of an ``(rows, cols, n)`` stack into ``[0, 1]``."""
    out = np.empty(stack.shape, dtype="float64")
    for i in range(stack.shape[2]):
        band = stack[..., i].astype("float64")
        lo, hi = np.nanpercentile(band, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        out[..., i] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out



class RasterMixin:
    """Raster field renders and composites for :class:`~digitalearth.scene.map.Map`."""

    def _field(
        self,
        dataset: Any,
        *,
        kind: str,
        band: int = 1,
        cmap: Optional[str] = None,
        levels: Any = None,
        add_colorbar: bool = False,
        **opts,
    ) -> Any:
        """Render a raster ``dataset`` on the shared axes via ``cleopatra.ArrayGlyph`` (the canonical recipe).

        Args:
            dataset: A pyramids ``Dataset`` (reprojected to :attr:`crs` first).
            kind: ArrayGlyph render kind (``auto``/``imshow``/``pcolormesh``/``contour``/``contourf``).
            band: 1-based band index.
            cmap: Optional colormap name.
            levels: Optional discrete levels (int or sequence of edges).
            add_colorbar: When ``False`` (default) the Scene owns the colorbar, not the glyph.
            **opts: Extra styling kwargs; filtered to ``ArrayGlyph``'s accepted options.

        Returns:
            The glyph's mappable (also registered as a Scene layer).
        """
        src = self._prepare(dataset, band)
        z_values, x_values, y_values = src.z.values, src.x.values, src.y.values
        if opts.pop("cyclic", False):  # close the antimeridian seam for global fields (T5.2)
            z_values, x_values = add_cyclic_column(z_values, x_values)
        if cmap is None:
            cmap = auto_style(src).get("cmap")  # per-variable default (T6.2)
        if cmap is not None:
            opts["cmap"] = cmap
        if levels is not None:
            opts["levels"] = levels
        # Geo-reference the data. cleopatra honours `extent` only for imshow (bbox order
        # [xmin, ymin, xmax, ymax]); contour/contourf/pcolormesh plot in array-index space unless
        # given `coords=(x, y)`. The two are mutually exclusive, so pick by kind — otherwise the
        # field lands at the wrong location/zoom.
        if kind == "imshow":
            placement = {"extent": self._extent_of(x_values, y_values)}
        else:
            placement = {"coords": (x_values, y_values)}
        # cleopatra's glyph constructors reject the regrouped styling keys (levels/style/color_scale/…);
        # relocate them onto the plot() call, where `_render_glyph` folds them into their group objects.
        plot_style = relocate_flat_style(opts)
        glyph = ArrayGlyph(
            z_values,
            exclude_value=[float("nan")],
            ax=self.ax,
            fig=self.fig,
            **placement,
            **opts,
        )
        return self._render_glyph(glyph, kind=kind, add_colorbar=add_colorbar, **plot_style)

    def imshow(self, dataset: Any, **kwargs) -> Any:
        """Render a raster as a pixel grid (``ArrayGlyph`` ``kind="imshow"``).

        Returns:
            The image mappable (registered as a Scene layer).
        """
        return self._field(dataset, kind="imshow", **kwargs)

    def contourf(self, dataset: Any, **kwargs) -> Any:
        """Render a raster as filled contours (``ArrayGlyph`` ``kind="contourf"``).

        Returns:
            The filled-contour mappable (registered as a Scene layer).
        """
        return self._field(dataset, kind="contourf", **kwargs)

    def contour(self, dataset: Any, **kwargs) -> Any:
        """Render a raster as line contours (``ArrayGlyph`` ``kind="contour"``).

        Returns:
            The line-contour mappable (registered as a Scene layer).
        """
        return self._field(dataset, kind="contour", **kwargs)

    def pcolormesh(self, dataset: Any, **kwargs) -> Any:
        """Render a raster as a quadrilateral mesh (``ArrayGlyph`` ``kind="pcolormesh"``).

        Returns:
            The ``QuadMesh`` mappable (registered as a Scene layer).
        """
        return self._field(dataset, kind="pcolormesh", **kwargs)

    def block(self, dataset: Any, **kwargs) -> Any:
        """Render a raster as a filled cell mesh — currently an alias of :meth:`pcolormesh`.

        ``block`` is meant for *discrete* per-cell rectangles aligned to cell **edges**. cleopatra's
        ``pcolormesh`` path accepts cell-**centre** coordinates only (it rejects edge arrays of length
        ``n + 1``), so true edge-aligned blocks need an upstream cleopatra option; until that lands ``block``
        draws the same filled mesh as :meth:`pcolormesh`. It is kept as a distinct, stable entry point so
        callers and examples can switch to true blocks transparently once cleopatra supports them.

        Returns:
            The ``QuadMesh`` mappable (registered as a Scene layer).
        """
        return self._field(dataset, kind="pcolormesh", **kwargs)

    @staticmethod
    def _extent_of(x: Any, y: Any) -> List[float]:
        """Return bbox-order ``[xmin, ymin, xmax, ymax]`` from 1-D x/y coordinate arrays (cleopatra order)."""
        return [float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))]

    def _extent(self, ds: Any) -> List[float]:
        """Return bbox-order ``[xmin, ymin, xmax, ymax]`` of a dataset's cell-centre coords (cleopatra order)."""
        return self._extent_of(ds.x, ds.y)

    def rgb_composite(self, dataset: Any, bands: Sequence[int] = (1, 2, 3), *, mask_nodata: bool = True,
                      **opts) -> Any:
        """Render three raster bands as a true/false-colour RGB image (``ArrayGlyph`` RGB path).

        Args:
            dataset: A multiband pyramids ``Dataset`` (reprojected to the display CRS first).
            bands: Three 1-based band indices mapped to R, G, B. Defaults to ``(1, 2, 3)``.
            mask_nodata: When ``True`` (default) each band's nodata cells are excluded from the 2-98
                percentile stretch (and render transparent). Pass ``False`` for the raw values (the
                pre-mask behaviour, where the nodata sentinel participates in the stretch).
            **opts: Styling kwargs, filtered to ``ArrayGlyph``'s accepted options.

        Returns:
            The image mappable (registered as a Scene layer).

        Examples:
            - Composite three bands of a multiband raster into one RGB image:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.scene import Map
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> base = np.nan_to_num(ds.read_array(band=0).astype("float32"))
                >>> rgb = Dataset.from_array(np.stack([base, base, base]),
                ...                          geo_ref=GeoReference(geo=ds.geotransform, epsg=ds.epsg))
                >>> m = Map(crs=rgb.epsg)
                >>> _ = m.rgb_composite(rgb)
                >>> len(m.ax.images)
                1

                ```
        """
        ds = self._reproject(dataset)
        stack = get_stack(ds, bands, mask=mask_nodata)  # (rows, cols, n); nodata -> NaN unless mask_nodata=False
        # cleopatra's RgbBands path is band-FIRST: it does array[indices].transpose(1, 2, 0), so feed
        # (n, rows, cols) and let it transpose back to (rows, cols, n) for imshow.
        band_first = np.moveaxis(_stretch_to_unit(stack), -1, 0)
        plot_style = relocate_flat_style(opts)
        glyph = ArrayGlyph(
            band_first, rgb_bands=RgbBands(list(range(len(bands)))), extent=self._extent(ds),
            ax=self.ax, fig=self.fig, **opts,
        )
        return self._render_glyph(glyph, **plot_style)

    def hsv_composite(self, dataset: Any, bands: Sequence[int] = (1, 2, 3), *, mask_nodata: bool = True,
                      **opts) -> Any:
        """Render three raster bands as an HSV composite (hue/sat/value → RGB → image).

        Args:
            dataset: A multiband pyramids ``Dataset`` (reprojected to the display CRS first).
            bands: Three 1-based band indices mapped to H, S, V. Defaults to ``(1, 2, 3)``.
            mask_nodata: When ``True`` (default) each band's nodata cells are excluded from the 2-98
                percentile stretch (and render transparent). Pass ``False`` for the raw pre-mask behaviour.
            **opts: Styling kwargs, filtered to ``ArrayGlyph``'s accepted options.

        Returns:
            The image mappable (registered as a Scene layer).
        """
        from matplotlib.colors import hsv_to_rgb

        ds = self._reproject(dataset)
        stack = get_stack(ds, bands, mask=mask_nodata)  # (rows, cols, n); nodata -> NaN unless mask_nodata=False
        rgb = hsv_to_rgb(_stretch_to_unit(stack))                      # (rows, cols, 3) RGB
        # band-FIRST for cleopatra's RgbBands path (see rgb_composite); it transposes back to band-last.
        band_first = np.moveaxis(rgb, -1, 0)
        plot_style = relocate_flat_style(opts)
        glyph = ArrayGlyph(
            band_first, rgb_bands=RgbBands([0, 1, 2]), extent=self._extent(ds), ax=self.ax, fig=self.fig,
            **opts,
        )
        return self._render_glyph(glyph, **plot_style)

    def spaghetti(self, collection: Any, band: int = 1, **opts) -> List[Any]:
        """Overlay each member of a ``DatasetCollection`` as line contours on one axes (ensemble spaghetti).

        Args:
            collection: A pyramids ``DatasetCollection`` whose members share a grid.
            band: 1-based band read from each member.
            **opts: Styling kwargs forwarded to the per-member contour call.

        Returns:
            The list of per-member contour mappables (each also registered as a Scene layer).
        """
        return [
            self._field(member, kind="contour", band=band, add_colorbar=False, **opts)
            for member in collection.datasets
        ]

