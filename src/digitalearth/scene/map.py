"""Map — a Scene with a display CRS, pyramids reprojection, and basemap/coastline decoration (no Cartopy).

``Map`` reprojects each input to a chosen display CRS via **pyramids** (``Dataset.to_crs``), renders on a plain
matplotlib axes in that projected space, and decorates it with an XYZ-tile basemap (``cleopatra.tiles``) and
Natural-Earth vector features (``pyramids.basemap.natural_earth``). There is deliberately **no Cartopy**: the
projection is applied to the *data* upstream, not to the axes (see plan §2.4).

The field methods here (``imshow`` and the private ``_field`` recipe) are the foundation T1.1 extends with
``contourf``/``contour``/``pcolormesh``/``block``.
"""
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from cleopatra.array_glyph import ArrayGlyph
from cleopatra.polygon_glyph import PolygonGlyph
from cleopatra.scatter_glyph import ScatterGlyph
from cleopatra.vector_glyph import VectorGlyph
from cleopatra.tiles import add_tiles
from pyramids.basemap import natural_earth

from digitalearth.scene.scene import Scene
from digitalearth.sources import get_source
from digitalearth.sources.source import Source


class Map(Scene):
    """A geospatial :class:`~digitalearth.scene.scene.Scene` that reprojects to a display CRS.

    Args:
        crs: Display CRS as an EPSG int / string / CRS (anything ``Dataset.to_crs`` accepts). Default 3857.
        domain: Optional named region / bbox used to set the extent (resolved in T5.1); ``None`` uses data bounds.
        ax: Existing axes to draw on (a new figure/axes is created when ``None``).
        fig: Figure owning ``ax``.
        figsize: New-figure size when one is created.

    Attributes:
        crs: The display CRS every layer is reprojected to.
        domain: The configured domain (or ``None``).

    Examples:
        - Create a map in Web Mercator and read its display CRS:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.scene import Map
            >>> m = Map(crs=3857)
            >>> m.crs
            3857
            >>> m.layers
            []

            ```
        - Render a reprojected raster, then iterate the registered layers:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from pyramids.dataset import Dataset
            >>> from digitalearth.scene import Map
            >>> ds = Dataset.read_file("examples/data/acc4000.tif")
            >>> m = Map(crs=ds.epsg)          # same CRS -> no reprojection
            >>> _ = m.imshow(ds)
            >>> len(m.layers)
            1

            ```
    """

    def __init__(
        self,
        crs: Any = 3857,
        domain: Any = None,
        ax: Any = None,
        fig: Any = None,
        figsize: Tuple[float, float] = (8, 8),
    ):
        super().__init__(ax=ax, fig=fig, figsize=figsize)
        self.crs = crs
        self.domain = domain

    def _prepare(self, dataset: Any, band: int = 1) -> Source:
        """Reproject ``dataset`` to the display CRS (if needed) and wrap it as a :class:`Source`."""
        ds = dataset if dataset.epsg == self.crs else dataset.to_crs(self.crs)
        return get_source(ds, band=band)

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
        extent = [
            float(src.x.values.min()),
            float(src.x.values.max()),
            float(src.y.values.min()),
            float(src.y.values.max()),
        ]
        if cmap is not None:
            opts["cmap"] = cmap
        if levels is not None:
            opts["levels"] = levels
        glyph = ArrayGlyph(
            src.z.values,
            exclude_value=[float("nan")],
            extent=extent,
            ax=self.ax,
            fig=self.fig,
            **ArrayGlyph.filter_kwargs(opts),
        )
        glyph.plot(kind=kind, add_colorbar=add_colorbar)
        return self._add_layer(glyph, glyph.im)

    def imshow(self, dataset: Any, **kwargs) -> Any:
        """Render a raster as a pixel grid (``ArrayGlyph`` ``kind="imshow"``)."""
        return self._field(dataset, kind="imshow", **kwargs)

    def contourf(self, dataset: Any, **kwargs) -> Any:
        """Render a raster as filled contours (``ArrayGlyph`` ``kind="contourf"``)."""
        return self._field(dataset, kind="contourf", **kwargs)

    def contour(self, dataset: Any, **kwargs) -> Any:
        """Render a raster as line contours (``ArrayGlyph`` ``kind="contour"``)."""
        return self._field(dataset, kind="contour", **kwargs)

    def pcolormesh(self, dataset: Any, **kwargs) -> Any:
        """Render a raster as a quadrilateral mesh (``ArrayGlyph`` ``kind="pcolormesh"``)."""
        return self._field(dataset, kind="pcolormesh", **kwargs)

    def block(self, dataset: Any, **kwargs) -> Any:
        """Render a raster as discrete cell blocks (pcolormesh on cell edges)."""
        return self._field(dataset, kind="pcolormesh", **kwargs)

    def _reproject(self, dataset: Any) -> Any:
        """Reproject a pyramids ``Dataset`` to the display CRS (returns it unchanged when already there)."""
        return dataset if dataset.epsg == self.crs else dataset.to_crs(self.crs)

    def scatter(self, features: Any, **opts) -> Any:
        """Plot a pyramids ``FeatureCollection`` of points, coloured by its value column (``ScatterGlyph``).

        Args:
            features: A pyramids ``FeatureCollection`` (point geometries); reprojected to the display CRS.
            **opts: Styling kwargs, filtered to ``ScatterGlyph``'s accepted options.

        Returns:
            The scatter ``PathCollection`` (registered as a Scene layer).
        """
        src = get_source(features.to_crs(self.crs))
        values = src.z.values if src.z is not None else None
        glyph = ScatterGlyph(
            src.x.values, src.y.values, values=values, ax=self.ax, fig=self.fig,
            **ScatterGlyph.filter_kwargs(opts),
        )
        _, _, pc = glyph.plot()
        return self._add_layer(glyph, pc)

    def grid_points(self, dataset: Any, **opts) -> Any:
        """Plot raster cell centres as points coloured by value (pyramids ``to_xyz`` → ``ScatterGlyph``).

        Args:
            dataset: A pyramids ``Dataset`` (reprojected to the display CRS first).
            **opts: Styling kwargs, filtered to ``ScatterGlyph``'s accepted options.

        Returns:
            The scatter ``PathCollection`` (registered as a Scene layer).
        """
        xyz = self._reproject(dataset).to_xyz()
        x = xyz.iloc[:, 0].to_numpy()
        y = xyz.iloc[:, 1].to_numpy()
        z = xyz.iloc[:, 2].to_numpy()
        glyph = ScatterGlyph(
            x, y, values=z, ax=self.ax, fig=self.fig, **ScatterGlyph.filter_kwargs(opts)
        )
        _, _, pc = glyph.plot()
        return self._add_layer(glyph, pc)

    def point_cloud(self, dataset: Any, **opts) -> Any:
        """Alias of :meth:`grid_points` — scatter raster cell centres coloured by value."""
        return self.grid_points(dataset, **opts)

    def grid_cells(self, dataset: Any, band: int = 1, **opts) -> Any:
        """Draw raster cells as value-coloured polygons (pyramids ``get_cell_polygons`` → ``PolygonGlyph``).

        Args:
            dataset: A pyramids ``Dataset`` (reprojected to the display CRS first).
            band: 1-based band whose values colour the cells.
            **opts: Styling kwargs, filtered to ``PolygonGlyph``'s accepted options.

        Returns:
            The ``PolyCollection`` (registered as a Scene layer).
        """
        ds = self._reproject(dataset)
        polygons = [np.asarray(g.exterior.coords) for g in ds.get_cell_polygons().geometry]
        values = ds.read_array(band=band - 1).astype("float64").ravel()
        nodata = ds.no_data_value[band - 1]
        if nodata is not None:
            values = np.where(np.isclose(values, nodata, rtol=1e-3), np.nan, values)
        glyph = PolygonGlyph(
            polygons, values=values, ax=self.ax, fig=self.fig,
            **PolygonGlyph.filter_kwargs(opts),
        )
        _, _, pc = glyph.plot()
        return self._add_layer(glyph, pc)

    def _vector(self, u_dataset: Any, v_dataset: Any, *, kind: str, band: int = 1, **opts) -> Any:
        """Render a vector field from two rasters (u, v) on a shared grid via ``cleopatra.VectorGlyph``.

        Args:
            u_dataset: pyramids ``Dataset`` of the u (eastward) component.
            v_dataset: pyramids ``Dataset`` of the v (northward) component.
            kind: ``"quiver"``, ``"barbs"`` or ``"streamplot"``.
            band: 1-based band index read from each dataset.
            **opts: Styling kwargs, filtered to ``VectorGlyph``'s accepted options.

        Returns:
            The vector mappable (registered as a Scene layer).
        """
        su = self._prepare(u_dataset, band)
        sv = self._prepare(v_dataset, band)
        xs, ys = su.x.values, su.y.values
        u, v = su.z.values, sv.z.values
        # streamplot (and a tidy grid generally) needs strictly increasing axes; raster y runs
        # north->south, so flip any descending axis and its data columns/rows to match.
        if xs[0] > xs[-1]:
            xs, u, v = xs[::-1], u[:, ::-1], v[:, ::-1]
        if ys[0] > ys[-1]:
            ys, u, v = ys[::-1], u[::-1, :], v[::-1, :]
        x_grid, y_grid = np.meshgrid(xs, ys)
        glyph = VectorGlyph(
            x_grid, y_grid, u, v, ax=self.ax, fig=self.fig,
            **VectorGlyph.filter_kwargs(opts),
        )
        _, _, im = glyph.plot(kind=kind)
        return self._add_layer(glyph, im)

    def quiver(self, u_dataset: Any, v_dataset: Any, **kwargs) -> Any:
        """Draw a vector field as arrows (``VectorGlyph`` ``kind="quiver"``)."""
        return self._vector(u_dataset, v_dataset, kind="quiver", **kwargs)

    def barbs(self, u_dataset: Any, v_dataset: Any, **kwargs) -> Any:
        """Draw a vector field as wind barbs (``VectorGlyph`` ``kind="barbs"``)."""
        return self._vector(u_dataset, v_dataset, kind="barbs", **kwargs)

    def streamplot(self, u_dataset: Any, v_dataset: Any, **kwargs) -> Any:
        """Draw a vector field as streamlines (``VectorGlyph`` ``kind="streamplot"``)."""
        return self._vector(u_dataset, v_dataset, kind="streamplot", **kwargs)

    def _natural_earth(self, layer: str, resolution: str, defaults: dict, **kwargs) -> Any:
        """Draw a Natural-Earth vector layer reprojected to the display CRS."""
        fc = natural_earth(layer, resolution)
        return fc.to_crs(self.crs).plot(ax=self.ax, **{**defaults, **kwargs})

    def coastlines(self, resolution: str = "110m", **kwargs) -> Any:
        """Overlay Natural-Earth coastlines (``pyramids.basemap.natural_earth("coastline")``)."""
        return self._natural_earth(
            "coastline", resolution, {"color": "black", "linewidth": 0.5}, **kwargs
        )

    def borders(self, resolution: str = "110m", **kwargs) -> Any:
        """Overlay Natural-Earth country borders."""
        return self._natural_earth(
            "borders", resolution, {"color": "gray", "linewidth": 0.4}, **kwargs
        )

    def land(self, resolution: str = "110m", **kwargs) -> Any:
        """Fill Natural-Earth land polygons."""
        return self._natural_earth(
            "land", resolution, {"color": "#efefdb", "edgecolor": "none"}, **kwargs
        )

    def ocean(self, resolution: str = "110m", **kwargs) -> Any:
        """Fill Natural-Earth ocean polygons."""
        return self._natural_earth(
            "ocean", resolution, {"color": "#cfe6f5", "edgecolor": "none"}, **kwargs
        )

    def basemap(self, source: Any = None, **kwargs) -> Any:
        """Add an XYZ-tile basemap to the axes via ``cleopatra.tiles.add_tiles`` in the display CRS."""
        return add_tiles(self.ax, source=source, crs=self.crs, **kwargs)

    def set_extent(self, bbox: Sequence[float]) -> None:
        """Set the axes extent.

        Args:
            bbox: ``[xmin, xmax, ymin, ymax]`` in the display CRS.
        """
        self.ax.set_xlim(bbox[0], bbox[1])
        self.ax.set_ylim(bbox[2], bbox[3])
