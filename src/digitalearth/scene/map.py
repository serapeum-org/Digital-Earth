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
from cleopatra.mesh_glyph import MeshGlyph
from cleopatra.polygon_glyph import PolygonGlyph
from cleopatra.scatter_glyph import ScatterGlyph
from cleopatra.vector_glyph import VectorGlyph
from cleopatra.tiles import add_tiles
from pyramids.base.crs import reproject_coordinates
from pyramids.basemap import natural_earth
from pyramids.dataset import Dataset

from digitalearth.autostyle import auto_style
from digitalearth.preprocess import add_cyclic_column
from digitalearth.scene.domains import DomainLike, resolve_domain
from digitalearth.scene.scene import Scene
from digitalearth.sources import get_source
from digitalearth.sources.source import Source


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
        z_values, x_values = src.z.values, src.x.values
        if opts.pop("cyclic", False):  # close the antimeridian seam for global fields (T5.2)
            z_values, x_values = add_cyclic_column(z_values, x_values)
        extent = [
            float(x_values.min()),
            float(x_values.max()),
            float(src.y.values.min()),
            float(src.y.values.max()),
        ]
        if cmap is None:
            cmap = auto_style(src).get("cmap")  # per-variable default (T6.2)
        if cmap is not None:
            opts["cmap"] = cmap
        if levels is not None:
            opts["levels"] = levels
        glyph = ArrayGlyph(
            z_values,
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

        Examples:
            - Scatter the raster's valid cell centres and count the resulting layer:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.scene import Map
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> m = Map(crs=ds.epsg)
                >>> _ = m.grid_points(ds)
                >>> len(m.layers)
                1

                ```
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

        Examples:
            - Draw one polygon per raster cell and confirm the count equals rows*columns:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.scene import Map
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> m = Map(crs=ds.epsg)
                >>> pc = m.grid_cells(ds)
                >>> len(pc.get_paths()) == ds.rows * ds.columns
                True

                ```
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

    def _extent(self, ds: Any) -> List[float]:
        """Return ``[xmin, xmax, ymin, ymax]`` of a (reprojected) dataset's cell-centre coordinates."""
        x, y = ds.x, ds.y
        return [float(x.min()), float(x.max()), float(y.min()), float(y.max())]

    def rgb_composite(self, dataset: Any, bands: Sequence[int] = (1, 2, 3), **opts) -> Any:
        """Render three raster bands as a true/false-colour RGB image (``ArrayGlyph`` RGB path).

        Args:
            dataset: A multiband pyramids ``Dataset`` (reprojected to the display CRS first).
            bands: Three 1-based band indices mapped to R, G, B. Defaults to ``(1, 2, 3)``.
            **opts: Styling kwargs, filtered to ``ArrayGlyph``'s accepted options.

        Returns:
            The image mappable (registered as a Scene layer).

        Examples:
            - Composite three bands of a multiband raster into one RGB image:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.scene import Map
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> base = np.nan_to_num(ds.read_array(band=0).astype("float32"))
                >>> rgb = Dataset.create_from_array(arr=np.stack([base, base, base]),
                ...                                 geo=ds.geotransform, epsg=ds.epsg)
                >>> m = Map(crs=rgb.epsg)
                >>> _ = m.rgb_composite(rgb)
                >>> len(m.ax.images)
                1

                ```
        """
        ds = self._reproject(dataset)
        stack = np.dstack([ds.read_array(band=b - 1) for b in bands])
        glyph = ArrayGlyph(
            _stretch_to_unit(stack), rgb=list(range(len(bands))), extent=self._extent(ds),
            ax=self.ax, fig=self.fig, **ArrayGlyph.filter_kwargs(opts),
        )
        glyph.plot()
        return self._add_layer(glyph, glyph.im)

    def hsv_composite(self, dataset: Any, bands: Sequence[int] = (1, 2, 3), **opts) -> Any:
        """Render three raster bands as an HSV composite (hue/sat/value → RGB → image).

        Args:
            dataset: A multiband pyramids ``Dataset`` (reprojected to the display CRS first).
            bands: Three 1-based band indices mapped to H, S, V. Defaults to ``(1, 2, 3)``.
            **opts: Styling kwargs, filtered to ``ArrayGlyph``'s accepted options.

        Returns:
            The image mappable (registered as a Scene layer).
        """
        from matplotlib.colors import hsv_to_rgb

        ds = self._reproject(dataset)
        stack = np.dstack([ds.read_array(band=b - 1) for b in bands])
        rgb = hsv_to_rgb(_stretch_to_unit(stack))
        glyph = ArrayGlyph(
            rgb, rgb=[0, 1, 2], extent=self._extent(ds), ax=self.ax, fig=self.fig,
            **ArrayGlyph.filter_kwargs(opts),
        )
        glyph.plot()
        return self._add_layer(glyph, glyph.im)

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

    def _scattered(self, data: Any) -> tuple:
        """Return ``(x, y, z)`` 1-D arrays for unstructured/point input (Dataset cells or a FeatureCollection)."""
        if isinstance(data, Dataset):
            xyz = self._reproject(data).to_xyz()
            return (
                xyz.iloc[:, 0].to_numpy(),
                xyz.iloc[:, 1].to_numpy(),
                xyz.iloc[:, 2].to_numpy(),
            )
        src = get_source(data.to_crs(self.crs))
        if src.z is None:
            raise ValueError("FeatureCollection has no numeric value column to contour")
        return src.x.values, src.y.values, src.z.values

    def _tri(self, data: Any, *, kind: str, **opts) -> Any:
        """Triangulate scattered points and render via ``cleopatra.MeshGlyph``."""
        from matplotlib.tri import Triangulation

        x, y, z = self._scattered(data)
        tri = Triangulation(x, y)
        glyph = MeshGlyph(x, y, tri.triangles, ax=self.ax, fig=self.fig)
        if kind == "tripcolor":
            face_values = z[tri.triangles].mean(axis=1)
            _, ax = glyph.plot(face_values, location="face", colorbar=False, **opts)
        else:
            _, ax = glyph.plot(z, location="node", filled=(kind == "tricontourf"), colorbar=False, **opts)
        mappable = ax.collections[-1] if ax.collections else None
        return self._add_layer(glyph, mappable)

    def tricontourf(self, data: Any, **kwargs) -> Any:
        """Filled contours of unstructured/point data (``MeshGlyph`` node data, ``filled=True``)."""
        return self._tri(data, kind="tricontourf", **kwargs)

    def tricontour(self, data: Any, **kwargs) -> Any:
        """Line contours of unstructured/point data (``MeshGlyph`` node data, ``filled=False``)."""
        return self._tri(data, kind="tricontour", **kwargs)

    def tripcolor(self, data: Any, **kwargs) -> Any:
        """Flat-shaded triangles of unstructured/point data (``MeshGlyph`` face data)."""
        return self._tri(data, kind="tripcolor", **kwargs)

    @staticmethod
    def _polygon_vertices(geometry: Any) -> tuple:
        """Return (vertex-arrays, repeat-counts) for a geopandas geometry series.

        Polygons contribute their exterior ring; MultiPolygons contribute one ring per part (so a single
        feature can map to several drawn polygons). The repeat count per feature lets callers expand a
        per-feature value array to per-polygon.
        """
        polygons: List[np.ndarray] = []
        repeats: List[int] = []
        for geom in geometry:
            parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            polygons.extend(np.asarray(p.exterior.coords) for p in parts)
            repeats.append(len(parts))
        return polygons, repeats

    def choropleth(self, features: Any, column: str, **opts) -> Any:
        """Fill polygons coloured by a feature attribute (pyramids ``FeatureCollection`` → ``PolygonGlyph``).

        Args:
            features: A pyramids ``FeatureCollection`` of polygons (reprojected to the display CRS).
            column: Name of the numeric column whose values colour the polygons.
            **opts: Styling kwargs, filtered to ``PolygonGlyph``'s accepted options.

        Returns:
            The ``PolyCollection`` (registered as a Scene layer).

        Examples:
            - Colour buffered point features by their ``fid`` column and count the drawn polygons:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth.scene import Map
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
                >>> fc["geometry"] = fc.geometry.buffer(500.0)
                >>> m = Map(crs=fc.epsg)
                >>> pc = m.choropleth(fc, column="fid")
                >>> len(pc.get_paths()) >= len(fc)
                True

                ```
        """
        gdf = features.to_crs(self.crs)
        polygons, repeats = self._polygon_vertices(gdf.geometry)
        values = np.repeat(gdf[column].to_numpy(), repeats)
        glyph = PolygonGlyph(
            polygons, values=values, ax=self.ax, fig=self.fig,
            **PolygonGlyph.filter_kwargs(opts),
        )
        _, _, pc = glyph.plot()
        return self._add_layer(glyph, pc)

    def shapes(self, features: Any, **opts) -> Any:
        """Draw polygon outlines without fill (pyramids ``FeatureCollection`` → ``PolygonGlyph`` outline mode).

        Args:
            features: A pyramids ``FeatureCollection`` of polygons (reprojected to the display CRS).
            **opts: Styling kwargs, filtered to ``PolygonGlyph``'s accepted options.

        Returns:
            The ``PolyCollection`` (registered as a Scene layer).
        """
        gdf = features.to_crs(self.crs)
        polygons, _ = self._polygon_vertices(gdf.geometry)
        glyph = PolygonGlyph(
            polygons, ax=self.ax, fig=self.fig, **PolygonGlyph.filter_kwargs(opts)
        )
        _, _, pc = glyph.plot(outline_only=True)
        return self._add_layer(glyph, pc)

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

    def set_domain(self, domain: Optional[DomainLike] = None) -> None:
        """Set the axes extent from a named region or bbox, reprojected to the display CRS via pyramids.

        Args:
            domain: A registered region name (e.g. ``"Europe"``), an explicit ``(west, south, east, north)``
                bbox in EPSG:4326, or ``None`` to fall back to the domain passed at construction. A no-op
                when neither resolves to a domain.

        Examples:
            - In a geographic CRS the axes limits equal the named region's bounds:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.scene import Map
                >>> m = Map(crs=4326)
                >>> m.set_domain("europe")
                >>> [float(v) for v in m.ax.get_xlim()]
                [-25.0, 45.0]
                >>> [float(v) for v in m.ax.get_ylim()]
                [34.0, 72.0]

                ```
        """
        bbox = resolve_domain(domain if domain is not None else self.domain)
        if bbox is None:
            return
        west, south, east, north = bbox
        xs, ys = reproject_coordinates(
            [west, east, west, east], [south, south, north, north], from_crs=4326, to_crs=self.crs
        )
        self.set_extent([min(xs), max(xs), min(ys), max(ys)])
