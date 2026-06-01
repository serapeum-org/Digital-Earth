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
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from cleopatra.array_glyph import ArrayGlyph
from cleopatra.mesh_glyph import MeshGlyph
from cleopatra.polygon_glyph import PolygonGlyph
from cleopatra.scatter_glyph import ScatterGlyph
from cleopatra.vector_glyph import VectorGlyph
from cleopatra.projection import apply_projection_frame
from cleopatra.tiles import add_tiles
from pyramids.base.crs import reproject_coordinates
from pyramids.basemap import natural_earth
from pyramids.dataset import Dataset

from digitalearth.autostyle import auto_style
from digitalearth.preprocess import add_cyclic_column
from digitalearth.scene import projections
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
        globe: bool = False,
    ):
        super().__init__(ax=ax, fig=fig, figsize=figsize)
        self.crs = crs
        self.domain = domain
        self.globe = globe
        self._graticule_lines: Optional[List[np.ndarray]] = None  # set by graticule()
        self._last_vector: Optional[tuple] = None  # (glyph, artist, kind) of the most recent vector layer
        self._framed = False
        self._frame_cache: Optional[tuple] = None  # (crs, (boundary, xlim, ylim)) memo

    def _needs_reproject(self, dataset: Any) -> bool:
        """Whether ``dataset`` must be reprojected to the display CRS.

        Only an EPSG-int display CRS can be compared cheaply against ``dataset.epsg``. For a proj4/string
        display CRS (e.g. an orthographic globe) we always reproject — and ``dataset.epsg`` is unreliable for
        non-EPSG results anyway (pyramids returns 4326 for a no-code projection), so we never compare against
        a proj4 CRS structurally here.

        Args:
            dataset: A pyramids ``Dataset`` whose ``.epsg`` is compared against the display CRS.

        Returns:
            ``False`` only when the display CRS is an ``int`` equal to ``dataset.epsg`` (data already in the
            display CRS); ``True`` otherwise — i.e. for a differing EPSG code or any proj4/string CRS.
        """
        return not (isinstance(self.crs, int) and dataset.epsg == self.crs)

    def _prepare(self, dataset: Any, band: int = 1) -> Source:
        """Reproject ``dataset`` to the display CRS (if needed) and wrap it as a :class:`Source`."""
        ds = dataset.to_crs(self.crs) if self._needs_reproject(dataset) else dataset
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
            placement = {"extent": [
                float(x_values.min()), float(y_values.min()),
                float(x_values.max()), float(y_values.max()),
            ]}
        else:
            placement = {"coords": (x_values, y_values)}
        glyph = ArrayGlyph(
            z_values,
            exclude_value=[float("nan")],
            ax=self.ax,
            fig=self.fig,
            **placement,
            **opts,
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
        return dataset.to_crs(self.crs) if self._needs_reproject(dataset) else dataset

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
        opts.setdefault("add_colorbar", False)  # the Scene owns the aggregated colorbar
        glyph = ScatterGlyph(
            src.x.values, src.y.values, values=values, ax=self.ax, fig=self.fig, **opts,
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
        opts.setdefault("add_colorbar", False)  # the Scene owns the aggregated colorbar
        glyph = ScatterGlyph(x, y, values=z, ax=self.ax, fig=self.fig, **opts)
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
        polygons, values = self._finite_polygons(polygons, values)  # drop far-side cells on a globe
        opts.setdefault("add_colorbar", False)  # the Scene owns the aggregated colorbar
        glyph = PolygonGlyph(
            polygons, values=values, ax=self.ax, fig=self.fig, **opts,
        )
        _, _, pc = glyph.plot()
        return self._add_layer(glyph, pc)

    def _extent(self, ds: Any) -> List[float]:
        """Return bbox-order ``[xmin, ymin, xmax, ymax]`` of a dataset's cell-centre coords (cleopatra order)."""
        x, y = ds.x, ds.y
        return [float(x.min()), float(y.min()), float(x.max()), float(y.max())]

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
        stack = np.dstack([ds.read_array(band=b - 1) for b in bands])  # (rows, cols, n)
        # cleopatra's rgb= path is band-FIRST: it does array[rgb].transpose(1, 2, 0), so feed
        # (n, rows, cols) and let it transpose back to (rows, cols, n) for imshow.
        band_first = np.moveaxis(_stretch_to_unit(stack), -1, 0)
        glyph = ArrayGlyph(
            band_first, rgb=list(range(len(bands))), extent=self._extent(ds),
            ax=self.ax, fig=self.fig, **opts,
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
        stack = np.dstack([ds.read_array(band=b - 1) for b in bands])  # (rows, cols, n)
        rgb = hsv_to_rgb(_stretch_to_unit(stack))                      # (rows, cols, 3) RGB
        # band-FIRST for cleopatra's rgb= path (see rgb_composite); it transposes back to band-last.
        band_first = np.moveaxis(rgb, -1, 0)
        glyph = ArrayGlyph(
            band_first, rgb=[0, 1, 2], extent=self._extent(ds), ax=self.ax, fig=self.fig,
            **opts,
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
        opts.setdefault("add_colorbar", False)  # the Scene owns the aggregated colorbar
        glyph = VectorGlyph(
            x_grid, y_grid, u, v, ax=self.ax, fig=self.fig, **opts,
        )
        _, _, im = glyph.plot(kind=kind)
        self._last_vector = (glyph, im, kind)  # remembered for quiverkey()
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

    def quiverkey(self, value: float, text: str, *, x: float = 0.9, y: float = 0.95, labelpos: str = "E",
                  **kwargs) -> Any:
        """Draw the labelled reference arrow for the most recent :meth:`quiver` layer.

        Places a sample arrow of known magnitude with a label (via ``Axes.quiverkey`` on the stored quiver
        artist), so readers can scale the field. Only ``quiver`` arrows carry a key; ``barbs``/``streamplot``
        do not.

        Args:
            value: The reference magnitude the sample arrow represents (data units, e.g. ``10`` for 10 m/s).
            text: The label drawn next to the arrow (e.g. ``"10 m/s"``).
            x: Arrow x position in axes fraction (0–1). Default ``0.9``.
            y: Arrow y position in axes fraction (0–1). Default ``0.95``.
            labelpos: Side of the arrow for the label (``"N"``/``"S"``/``"E"``/``"W"``). Default ``"E"``.
            **kwargs: Forwarded to ``Axes.quiverkey`` (e.g. ``coordinates``, ``color``, ``fontproperties``).

        Returns:
            The :class:`matplotlib.quiver.QuiverKey` added to the axes.

        Raises:
            ValueError: if no :meth:`quiver` layer has been drawn yet (``barbs``/``streamplot`` have no key).
        """
        if self._last_vector is None or self._last_vector[2] != "quiver":
            raise ValueError("quiverkey() needs a prior quiver(...) layer (barbs/streamplot have no key)")
        _, artist, _ = self._last_vector
        return self.ax.quiverkey(artist, x, y, value, text, labelpos=labelpos, **kwargs)

    def text(self, lon: float, lat: float, s: str, *, crs: Any = 4326, **kwargs) -> Any:
        """Place a text label at a ``lon``/``lat`` location (reprojected to the display CRS).

        The point is reprojected from ``crs`` (lon/lat by default) into the display CRS via pyramids, then
        drawn with ``Axes.text``. On a globe, a point on the **far side** reprojects to non-finite coordinates
        and is skipped (no artist, returns ``None``).

        Args:
            lon: Longitude (x) of the label, in ``crs``.
            lat: Latitude (y) of the label, in ``crs``.
            s: The text to draw.
            crs: CRS of ``lon``/``lat`` (default ``4326`` = WGS84 lon/lat).
            **kwargs: Forwarded to ``Axes.text`` (e.g. ``ha``, ``va``, ``fontsize``, ``color``).

        Returns:
            The :class:`matplotlib.text.Text`, or ``None`` if the point is off the visible globe.
        """
        x, y = reproject_coordinates([lon], [lat], from_crs=crs, to_crs=self.crs)
        if not (np.isfinite(x[0]) and np.isfinite(y[0])):
            return None
        return self.ax.text(x[0], y[0], s, **kwargs)

    def annotate(self, lon: float, lat: float, s: str, *, xytext: Any = None, crs: Any = 4326,
                 **kwargs) -> Any:
        """Annotate a ``lon``/``lat`` location (reprojected), optionally with an arrow.

        Like :meth:`text` but via ``Axes.annotate``: the annotated point ``xy`` is the reprojected
        ``lon``/``lat``; pass ``xytext`` (with ``arrowprops``) to draw an arrow from the label to the point.
        A far-side point on a globe is skipped (returns ``None``).

        Args:
            lon: Longitude (x) of the annotated point, in ``crs``.
            lat: Latitude (y) of the annotated point, in ``crs``.
            s: The annotation text.
            xytext: Optional text position (in the coordinate system given by ``textcoords``/``kwargs``); with
                ``arrowprops`` an arrow is drawn from there to the point.
            crs: CRS of ``lon``/``lat`` (default ``4326``).
            **kwargs: Forwarded to ``Axes.annotate`` (e.g. ``arrowprops``, ``textcoords``, ``fontsize``).

        Returns:
            The :class:`matplotlib.text.Annotation`, or ``None`` if the point is off the visible globe.
        """
        x, y = reproject_coordinates([lon], [lat], from_crs=crs, to_crs=self.crs)
        if not (np.isfinite(x[0]) and np.isfinite(y[0])):
            return None
        return self.ax.annotate(s, xy=(x[0], y[0]), xytext=xytext, **kwargs)

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
        finite = np.isfinite(x) & np.isfinite(y)  # drop far-side points on a globe (Triangulation needs finite)
        x, y, z = np.asarray(x)[finite], np.asarray(y)[finite], np.asarray(z)[finite]
        tri = Triangulation(x, y)
        glyph = MeshGlyph(x, y, tri.triangles, ax=self.ax, fig=self.fig)
        if kind == "tripcolor":
            face_values = z[tri.triangles].mean(axis=1)
            glyph.plot(face_values, location="face", colorbar=False, **opts)
        else:
            glyph.plot(z, location="node", filled=(kind == "tricontourf"), colorbar=False, **opts)
        # cleopatra 0.11.0 exposes the tripcolor/tricontour(f) artist on glyph.im (issue #2),
        # so we no longer scrape ax.collections[-1].
        return self._add_layer(glyph, glyph.im)

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

    @staticmethod
    def _finite_polygons(polygons: List[np.ndarray], values: Optional[np.ndarray] = None) -> tuple:
        """Drop polygons with any non-finite vertex (and the matching values).

        On a projected/globe map the far hemisphere reprojects to non-finite coordinates; matplotlib's
        ``PolyCollection`` would otherwise receive ``inf``/``nan`` vertices and fail or draw garbage. Keeps
        only fully-finite rings, preserving order and the positional alignment between rings and ``values``.

        Args:
            polygons: Polygon rings as ``(N, 2)`` vertex arrays; any ring with an ``inf``/``nan`` vertex is
                dropped.
            values: Optional per-polygon scalar values, positionally aligned with ``polygons``. When given,
                the same rings are dropped from it so the two stay aligned.

        Returns:
            ``(kept_polygons, kept_values)`` — the surviving rings, and either the filtered ``values`` array
            or ``None`` when ``values`` was ``None``.

        Examples:
            - A ring with an ``inf`` vertex is dropped along with its value:
                ```python
                >>> import numpy as np
                >>> from digitalearth.scene import Map
                >>> good = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
                >>> bad = np.array([[0.0, 0.0], [np.inf, 0.0], [1.0, 1.0]])
                >>> kept, vals = Map._finite_polygons([good, bad, good], np.array([10.0, 20.0, 30.0]))
                >>> len(kept)
                2
                >>> vals.tolist()
                [10.0, 30.0]

                ```
            - Without values, only the finite rings come back and the second slot is ``None``:
                ```python
                >>> import numpy as np
                >>> from digitalearth.scene import Map
                >>> good = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
                >>> bad = np.array([[np.nan, 0.0], [1.0, 0.0], [1.0, 1.0]])
                >>> kept, vals = Map._finite_polygons([good, bad])
                >>> len(kept), vals
                (1, None)

                ```
        """
        keep = [i for i, p in enumerate(polygons) if np.isfinite(p).all()]
        kept = [polygons[i] for i in keep]
        if values is None:
            return kept, None
        return kept, np.asarray(values)[keep]

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
        polygons, values = self._finite_polygons(polygons, values)  # drop far-side polygons on a globe
        opts.setdefault("add_colorbar", False)  # the Scene owns the aggregated colorbar
        glyph = PolygonGlyph(
            polygons, values=values, ax=self.ax, fig=self.fig, **opts,
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
        polygons, _ = self._finite_polygons(polygons)  # drop far-side polygons on a globe
        glyph = PolygonGlyph(polygons, ax=self.ax, fig=self.fig, **opts)
        _, _, pc = glyph.plot(outline_only=True)
        return self._add_layer(glyph, pc)

    def _project_line_features(self, fc: Any) -> List[np.ndarray]:
        """Project a line FeatureCollection (lon/lat) to the display CRS, split at the projection limb.

        Reprojecting a global line to a clipped projection (e.g. orthographic) sends the far side to
        non-finite coordinates; per-line splitting at those gaps keeps the visible arcs and avoids the
        ``NaN/Inf`` errors geopandas' ``clip``/``plot`` raise on such geometry.
        """
        from_crs = fc.epsg if fc.epsg is not None else 4326
        segments: List[np.ndarray] = []
        for geom in fc.geometry:
            if geom is None:
                continue
            parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
            for part in parts:
                xy = np.asarray(part.coords, dtype=float)
                if xy.size == 0:
                    continue
                x, y = reproject_coordinates(xy[:, 0].tolist(), xy[:, 1].tolist(),
                                             from_crs=from_crs, to_crs=self.crs)
                segments += projections._split_finite(np.asarray(x, float), np.asarray(y, float))
        return segments

    def _project_polygon_features(self, fc: Any) -> List[np.ndarray]:
        """Project a polygon FeatureCollection (lon/lat) to the display CRS as finite, limb-clipped fill rings.

        The fill analogue of :meth:`_project_line_features`: each exterior ring is densified, reprojected, and
        re-closed against the projection boundary (``self._frame()[0]``) so a polygon straddling the limb
        becomes one or more closed, fully-finite rings instead of injecting ``inf``/``nan`` into the fill.
        Interior rings (holes) are dropped in v1 — see Digital-Earth#43.

        Args:
            fc: A pyramids ``FeatureCollection`` of polygons in lon/lat (or any CRS via ``fc.epsg``).

        Returns:
            A list of closed ``(N, 2)`` projected fill rings (empty when nothing is on the near side).
        """
        from_crs = fc.epsg if fc.epsg is not None else 4326
        boundary = self._frame()[0]
        rings: List[np.ndarray] = []
        for geom in fc.geometry:
            if geom is None:
                continue
            parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
            for part in parts:
                xy = np.asarray(part.exterior.coords, dtype=float)
                if xy.size == 0:
                    continue
                xy = projections.densify_lonlat(xy, step_deg=1.0)
                x, y = reproject_coordinates(xy[:, 0].tolist(), xy[:, 1].tolist(),
                                             from_crs=from_crs, to_crs=self.crs)
                rings += projections.close_visible_runs(np.asarray(x, float), np.asarray(y, float), boundary)
        return rings

    def _fill_globe_polygons(self, rings: List[np.ndarray], *, facecolor: Any, zorder: float) -> Any:
        """Fill projected rings with a solid colour on a globe (map-specific overlay; clipped at frame time).

        cleopatra ``PolygonGlyph`` only fills when given per-polygon *values*, so a uniform land/ocean fill is
        drawn as a plain ``PolyCollection`` directly on the axes — exactly like the globe coastline ``ax.plot``
        overlay. ``apply_projection_frame(clip_artists=True)`` clips it to the boundary at render time. The
        view limits are preserved so a global fill never autoscales the axes out.

        Args:
            rings: Closed, finite projected fill rings (from :meth:`_project_polygon_features`).
            facecolor: Solid fill colour.
            zorder: Draw order (ocean below land below data below coastlines).

        Returns:
            The ``PolyCollection`` (registered as a Scene layer), or ``None`` when ``rings`` is empty.
        """
        if not rings:
            return None
        has_data = bool(self.layers) or bool(self.ax.images) or bool(self.ax.collections)
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        pc = PolyCollection(rings, facecolors=facecolor, edgecolors="none", zorder=zorder)
        self.ax.add_collection(pc)
        if has_data:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        return self._add_layer(None, pc)

    def _natural_earth(self, layer: str, resolution: str, defaults: dict, *, polygon: bool = False,
                       zorder: float = 0.5, **kwargs) -> Any:
        """Draw a Natural-Earth vector layer reprojected to the display CRS, clipped to the current view.

        On a **globe** map, line layers (coastline/borders/rivers) are projected per-line and split at the
        projection limb (the far side reprojects to non-finite coords) and drawn as plain polylines; polygon
        layers (``polygon=True``: land/lakes) are projected and re-closed at the limb into finite rings and
        filled via :meth:`_fill_globe_polygons`. Both are clipped to the boundary when the frame is applied.
        On a **flat** map, the layer is reprojected with ``GeoDataFrame.plot``; since Natural Earth is global
        and that autoscales the axes, the data's limits are preserved when a data layer is already present.

        Args:
            layer: Natural-Earth layer name (e.g. ``"coastline"``, ``"land"``).
            resolution: Natural-Earth resolution (``"110m"``/``"50m"``/``"10m"``).
            defaults: Base style; ``color``/``facecolor`` is the globe fill colour for polygon layers.
            polygon: When True, treat the layer as filled polygons on a globe (else as lines).
            zorder: Globe draw order for polygon fills.
            **kwargs: Style overrides merged over ``defaults``.
        """
        fc = natural_earth(layer, resolution)
        if self.globe:
            style = {**defaults, **kwargs}
            if polygon:
                facecolor = style.get("facecolor", style.get("color", "#efefdb"))
                return self._fill_globe_polygons(self._project_polygon_features(fc),
                                                 facecolor=facecolor, zorder=zorder)
            style.pop("edgecolor", None); style.pop("facecolor", None)
            artists = [self.ax.plot(seg[:, 0], seg[:, 1], **style)[0] for seg in self._project_line_features(fc)]
            return artists
        has_data = bool(self.layers) or bool(self.ax.images) or bool(self.ax.collections)
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        artist = fc.to_crs(self.crs).plot(ax=self.ax, **{**defaults, **kwargs})
        if has_data:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        return artist

    def coastlines(self, resolution: str = "110m", **kwargs) -> Any:
        """Overlay Natural-Earth coastlines (``pyramids.basemap.natural_earth("coastline")``)."""
        return self._natural_earth(
            "coastline", resolution, {"color": "black", "linewidth": 0.5}, zorder=2.5, **kwargs
        )

    def borders(self, resolution: str = "110m", **kwargs) -> Any:
        """Overlay Natural-Earth country borders."""
        return self._natural_earth(
            "borders", resolution, {"color": "gray", "linewidth": 0.4}, zorder=2.5, **kwargs
        )

    def land(self, resolution: str = "110m", **kwargs) -> Any:
        """Fill Natural-Earth land polygons.

        On a **flat** map the polygons are reprojected and filled directly. On a **globe** map they are
        re-closed at the projection limb into finite rings and filled as a map-specific overlay (drawn below
        data and coastlines, clipped to the boundary). Interior rings (holes) are dropped in v1 — see #43.
        """
        return self._natural_earth(
            "land", resolution, {"color": "#efefdb", "edgecolor": "none"}, polygon=True, zorder=-1.5, **kwargs
        )

    def ocean(self, resolution: str = "110m", **kwargs) -> Any:
        """Fill Natural-Earth ocean polygons.

        On a **globe** map, ``ocean`` fills the whole projection disc (the boundary ring) with the ocean
        colour and lets land overlay it — exact and far cheaper than clipping the global ocean polygon. On a
        **flat** map, the Natural-Earth ocean polygons are reprojected and filled directly.
        """
        color = kwargs.pop("color", "#cfe6f5")
        if self.globe:
            boundary = self._frame()[0]
            return self._fill_globe_polygons([np.asarray(boundary)], facecolor=color, zorder=-2.0)
        return self._natural_earth("ocean", resolution, {"color": color, "edgecolor": "none"}, **kwargs)

    def lakes(self, resolution: str = "110m", **kwargs) -> Any:
        """Fill Natural-Earth lake polygons.

        Like :meth:`land`, but with a water colour and drawn just above land (so lakes sit on the land) and
        still below data and coastlines. On a globe the polygons are re-closed at the projection limb.
        """
        return self._natural_earth(
            "lakes", resolution, {"color": "#cfe6f5", "edgecolor": "none"}, polygon=True, zorder=-1.4, **kwargs
        )

    def rivers(self, resolution: str = "110m", **kwargs) -> Any:
        """Overlay Natural-Earth rivers (line centerlines), split at the projection limb on a globe."""
        return self._natural_earth(
            "rivers", resolution, {"color": "#5a8fcf", "linewidth": 0.4}, zorder=2.4, **kwargs
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

    # ------------------------------------------------------------------ globe / projection frame

    def graticule(self, lon_step: float = 30.0, lat_step: float = 30.0) -> None:
        """Add a lon/lat graticule to a projected map (drawn when the frame is applied).

        Args:
            lon_step: Meridian spacing in degrees.
            lat_step: Parallel spacing in degrees.
        """
        self._graticule_lines = projections.graticule(self.crs, lon_step=lon_step, lat_step=lat_step)

    def _frame(self) -> tuple:
        """Return the cached ``(boundary, xlim, ylim)`` for the display CRS (computed once per CRS).

        ``projection_frame`` reprojects a dense lon/lat sample of the whole sphere, so it is memoised here to
        avoid recomputing it for both ``set_global`` and ``_apply_frame``. The cache is keyed on the display
        CRS and recomputed only when the CRS changes.

        Returns:
            The ``(boundary_xy, (xmin, xmax), (ymin, ymax))`` tuple from
            :func:`digitalearth.scene.projections.projection_frame` for the current display CRS — a closed
            ``(N, 2)`` boundary ring plus the projected x/y limits.
        """
        if self._frame_cache is None or self._frame_cache[0] != self.crs:
            self._frame_cache = (self.crs, projections.projection_frame(self.crs))
        return self._frame_cache[1]

    def set_global(self) -> None:
        """Set the axes extent to the full projection domain (the whole globe/world)."""
        _, xlim, ylim = self._frame()
        self.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]])

    def _apply_frame(self) -> Any:
        """Draw the projection boundary + graticule and clip the layers to it (once, at render time)."""
        if not self.globe or self._framed:
            return None
        boundary, xlim, ylim = self._frame()
        patch = apply_projection_frame(
            self.ax, boundary_xy=boundary, xlim=xlim, ylim=ylim,
            graticule_lines=self._graticule_lines,
        )
        self._framed = True
        return patch

    def render(self) -> None:
        """Apply the projection frame if this is a globe map (idempotent). Call before showing/saving."""
        self._apply_frame()

    def save(self, path: str, **kwargs) -> None:
        """Apply the projection frame (for a globe map) then save the figure."""
        self._apply_frame()
        super().save(path, **kwargs)

    def show(self) -> None:
        """Apply the projection frame (for a globe map) then show the figure."""
        self._apply_frame()
        super().show()

    def _animate_frames(self, draw_one: Any, n_frames: int, fps: float) -> FuncAnimation:
        """Drive ``n_frames`` of ``draw_one(i)`` on this Map's axes as a :class:`FuncAnimation`.

        Each frame clears the axes and resets the per-frame layer/frame state, calls ``draw_one(i)`` to draw
        frame ``i``, then (on a globe) sets the full-domain extent and applies the projection frame. No
        colorbar is added per frame — pass a fixed ``vmin``/``vmax`` to keep colours stable instead.
        """
        def _f(i: int) -> None:
            self.ax.clear()
            self.layers = []
            self._framed = False
            draw_one(i)
            if self.globe:
                self.set_global()
                self._apply_frame()

        return FuncAnimation(self.fig, _f, frames=n_frames, interval=1000.0 / fps, blit=False)

    @staticmethod
    def _stack_clim(datasets: Sequence[Any]) -> Tuple[float, float]:
        """Return the ``(min, max)`` of the first band across ``datasets``, ignoring nodata/non-finite."""
        lows: List[float] = []
        highs: List[float] = []
        for ds in datasets:
            arr = np.asarray(ds.read_array(band=0), dtype="float64")
            nodata = ds.no_data_value[0]
            if nodata is not None:
                arr = arr[arr != nodata]
            arr = arr[np.isfinite(arr)]
            if arr.size:
                lows.append(float(arr.min()))
                highs.append(float(arr.max()))
        return (min(lows), max(highs)) if lows else (0.0, 1.0)

    def _resolve_animation_clim(self, datasets: Sequence[Any], opts: dict) -> None:
        """Ensure ``opts`` carries a shared ``vmin``/``vmax`` so every animation frame uses one colour scale.

        Without this, each frame's renderer auto-scales to its own data range, so the colours (and any
        colorbar) flicker between frames. Any ``vmin``/``vmax`` already in ``opts`` is kept; a missing bound
        is filled once from the whole stack (ignoring nodata/non-finite) and written back, so all frames —
        and the colorbar — share it.
        """
        vmin, vmax = opts.get("vmin"), opts.get("vmax")
        if vmin is None or vmax is None:
            lo, hi = self._stack_clim(datasets)
            opts["vmin"] = lo if vmin is None else vmin
            opts["vmax"] = hi if vmax is None else vmax

    def _animation_colorbar(self, opts: dict, label: Optional[str]) -> Any:
        """Add one static colorbar for an animation from the already-resolved ``cmap``/``vmin``/``vmax``.

        The colorbar lives on its own figure axes (not the data axes that each frame clears), so it persists
        across frames. Call :meth:`_resolve_animation_clim` first so ``opts`` has the shared clim.
        """
        cmap = opts.setdefault("cmap", "viridis")
        mappable = ScalarMappable(norm=Normalize(vmin=opts.get("vmin"), vmax=opts.get("vmax")), cmap=cmap)
        mappable.set_array([])
        cbar = self.fig.colorbar(mappable, ax=self.ax)
        if label is not None:
            cbar.set_label(label)
        return cbar

    def animate(self, stack: Any, *, kind: str = "imshow", fps: float = 3.0,
                titles: Optional[Sequence[str]] = None, ocean: bool = False, coastlines: bool = False,
                colorbar: bool = False, cbar_label: Optional[str] = None, **kwargs) -> FuncAnimation:
        """Animate a stack of rasters over this map, returning a matplotlib :class:`FuncAnimation`.

        Each frame reprojects ``stack[i]`` to the display CRS (pyramids), renders it with the ``kind`` method
        (cleopatra), optionally draws the ocean disc / coastlines, and — on a globe — applies the projection
        frame, so every frame gets the boundary, graticule, and limb-clipping for free. The returned
        animation is lazy: call ``anim.save("out.gif", writer=PillowWriter(fps=...))`` or display it.

        All frames share **one colour scale**: ``vmin``/``vmax`` from ``kwargs`` if given, else computed once
        from the whole stack — so the colours (and the colorbar) do not flicker between frames.

        Args:
            stack: An ordered, indexable collection of pyramids ``Dataset`` frames (e.g. a list, or a
                ``DatasetCollection`` datacube) — one raster per animation frame.
            kind: The field method used to draw each frame (``"imshow"`` / ``"contourf"`` / ``"pcolormesh"``).
            fps: Frames per second (sets the inter-frame interval).
            titles: Optional per-frame titles; must match the stack length when given.
            ocean: When True, fill the ocean disc behind each frame (globe maps only).
            coastlines: When True, overlay coastlines each frame (best-effort; ignored if unreachable).
            colorbar: When True, add one static colorbar (drawn once, not per frame) using the shared
                colour scale.
            cbar_label: Optional label for the colorbar.
            **kwargs: Forwarded to the ``kind`` method (e.g. ``cmap``, ``vmin``, ``vmax``).

        Returns:
            A :class:`matplotlib.animation.FuncAnimation` over ``len(stack)`` frames.

        Raises:
            ValueError: if ``stack`` is empty, or ``titles`` is given with a mismatched length.
        """
        frames = list(stack)
        if not frames:
            raise ValueError("animate got an empty stack (nothing to animate)")
        if titles is not None and len(titles) != len(frames):
            raise ValueError(f"titles length ({len(titles)}) must match the stack length ({len(frames)})")
        self._resolve_animation_clim(frames, kwargs)  # one colour scale for every frame
        if colorbar:
            self._animation_colorbar(kwargs, cbar_label)

        def draw_one(i: int) -> None:
            if ocean and self.globe:
                self.ocean()
            getattr(self, kind)(frames[i], **kwargs)
            if coastlines:
                try:
                    self.coastlines()
                except Exception:  # network/data unavailable — decoration is best-effort
                    pass
            if titles is not None:
                self.set_title(titles[i])

        return self._animate_frames(draw_one, len(frames), fps)

    def rotate(self, dataset: Any, *, lat: float = 15.0, n_frames: int = 24, fps: float = 8.0,
               lon0: float = -180.0, kind: str = "imshow", ocean: bool = False, coastlines: bool = False,
               colorbar: bool = False, cbar_label: Optional[str] = None, **kwargs) -> FuncAnimation:
        """Spin an orthographic globe over a single field by sweeping the centre longitude.

        Forces a globe map and redraws ``dataset`` on ``n_frames`` orthographic projections whose centre
        longitude steps a full 360 degrees from ``lon0``. The display CRS (:attr:`crs`) is swept as the
        animation renders, so after rendering it holds the last frame's projection.

        Args:
            dataset: The pyramids ``Dataset`` to spin (reprojected per frame).
            lat: Centre latitude of every orthographic view.
            n_frames: Number of frames spanning the full 360-degree turn.
            fps: Frames per second.
            lon0: Starting centre longitude.
            kind: The field method used to draw the data (``"imshow"`` / ``"contourf"`` / ``"pcolormesh"``).
            ocean: When True, fill the ocean disc behind the data each frame.
            coastlines: When True, overlay coastlines each frame (best-effort).
            colorbar: When True, add one static colorbar (drawn once) using the shared colour scale.
            cbar_label: Optional label for the colorbar.
            **kwargs: Forwarded to the ``kind`` method (e.g. ``cmap``, ``vmin``, ``vmax``).

        Returns:
            A :class:`matplotlib.animation.FuncAnimation` over ``n_frames`` frames.

        Raises:
            ValueError: if ``n_frames`` is less than 1.
        """
        if n_frames < 1:
            raise ValueError("rotate needs n_frames >= 1")
        self.globe = True
        self._resolve_animation_clim([dataset], kwargs)  # one colour scale for every frame
        if colorbar:
            self._animation_colorbar(kwargs, cbar_label)
        lons = [lon0 + k * (360.0 / n_frames) for k in range(n_frames)]

        def draw_one(i: int) -> None:
            self.crs = projections.orthographic(lon=lons[i], lat=lat)
            if ocean:
                self.ocean()
            getattr(self, kind)(dataset, **kwargs)
            if coastlines:
                try:
                    self.coastlines()
                except Exception:  # network/data unavailable — decoration is best-effort
                    pass

        return self._animate_frames(draw_one, n_frames, fps)
