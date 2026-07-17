"""VectorMixin — vector data and vector-field renders.

Points (scatter/grid_points/grid_cells), polygon products (choropleth/shapes/voronoi/cartogram/quadtree),
unstructured triangulations (tricontour/tricontourf/tripcolor), kernel density, flow/Sankey, and the u/v
vector field (quiver/barbs/streamplot/quiverkey) — all wired onto the matching cleopatra glyphs.
"""
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from shapely import MultiPoint, box, voronoi_polygons
from shapely.affinity import scale as affine_scale
from matplotlib.path import Path as MplPath
from cleopatra.flow_glyph import FlowGlyph
from cleopatra.kde_glyph import KDEGlyph
from cleopatra.mesh_glyph import MeshGlyph
from cleopatra.polygon_glyph import PolygonGlyph
from cleopatra.scatter_glyph import ScatterGlyph
from cleopatra.vector_glyph import VectorGlyph
from pyramids.dataset import Dataset

from digitalearth._arrays import NAN_REDUCERS, read_masked_band
from digitalearth._symbology import MISSING_COLOR, nulls_to_none, resolve_categorical_cmap
from digitalearth.sources import get_source

#: Per-cell reducers accepted by ``Map.quadtree``'s ``agg`` — the shared NaN-aware registry plus a special
#: ``"count"`` (``len`` over the per-cell index array, ignoring the column).
_QUADTREE_AGG = {**NAN_REDUCERS, "count": len}


def _draw_missing_neutral(artist: Any) -> None:
    """Colour a categorical layer's missing values neutral instead of invisible.

    cleopatra maps a missing category to ``NaN`` in the class codes, and a ``ListedColormap``'s default "bad"
    colour is fully transparent — so a feature whose attribute is missing is drawn as *nothing*, making it
    indistinguishable from a feature that was never in the collection. The web and interactive tiers both draw
    :data:`~digitalearth._symbology.MISSING_COLOR` there; this matches them, so missing data reads as missing on
    all three tiers.

    ``with_extremes`` returns a *new* colormap rather than mutating in place, which matters: the glyph's may be
    a colormap registered globally under a name, and setting the bad colour on that instance would leak this
    policy into every other plot in the process.

    Args:
        artist: The rendered mappable (a ``PolyCollection``) whose colormap gets the neutral "bad" colour.
    """
    artist.set_cmap(artist.get_cmap().with_extremes(bad=MISSING_COLOR))


class VectorMixin:
    """Vector-data and vector-field renders for :class:`~digitalearth.scene.map.Map`."""

    def _vector_input(self, features: Any, *, geom_types: Optional[Sequence[str]] = None,
                      name: str = "layer", geom_label: Optional[str] = None) -> Any:
        """Reproject a ``FeatureCollection`` to the display CRS, reject empty, and validate its geometry type.

        Consolidates the preamble shared by the validating vector methods. Reprojects ``features`` to
        :attr:`crs`, raises on an empty collection, and — when ``geom_types`` is given — requires every
        geometry to be one of those types.

        Args:
            features: A pyramids ``FeatureCollection`` (reprojected to the display CRS).
            geom_types: Allowed shapely ``geom_type`` names (e.g. ``("Point",)``); ``None`` skips the check.
            name: The calling method's name, used in error messages.
            geom_label: Human label for the allowed geometry in the error (defaults to the joined types).

        Returns:
            The reprojected GeoDataFrame.

        Raises:
            ValueError: if the collection is empty, or a geometry is not one of ``geom_types``.
        """
        gdf = features.to_crs(self.crs)
        if len(gdf) == 0:
            raise ValueError(f"{name} got an empty FeatureCollection (nothing to draw)")
        if geom_types is not None and not gdf.geometry.geom_type.isin(list(geom_types)).all():
            label = geom_label or " / ".join(geom_types)
            raise ValueError(f"{name} requires a FeatureCollection of {label} geometries")
        return gdf

    def _polygon_layer(self, polygons: List[np.ndarray], values: Optional[np.ndarray] = None, **opts) -> Any:
        """Draw polygons as a value-filled (``values`` given) or outline-only ``PolygonGlyph`` layer.

        Consolidates the fill-vs-outline branch shared by :meth:`grid_cells`, :meth:`choropleth`,
        :meth:`shapes`, :meth:`voronoi`, :meth:`cartogram` and :meth:`quadtree`. The Scene owns the
        aggregated colorbar, so the glyph's own colorbar is suppressed by default — except under
        ``scheme="categorical"``, where the value key is a per-class swatch legend the glyph builds from its
        own mapping (``PolygonGlyph.category_legend``). The Scene's colorbar cannot stand in for it: a
        categorical fill feeds the mappable opaque integer class codes, so a colorbar over them would read
        ``0, 1, 2 …`` instead of the category labels. Passing ``add_colorbar=False`` suppresses that swatch
        legend — for a caller keying the map some other way, e.g. drawing one shared legend across several
        layers via :meth:`~digitalearth.scene.scene.Scene.legend` (which takes explicit ``colors``/``labels``;
        read the drawn legend's swatches/texts off ``layer.category_legend`` to feed it).

        Args:
            polygons: Polygon rings as ``(N, 2)`` vertex arrays.
            values: Optional per-polygon scalar values; when ``None`` only the outlines are drawn.
            **opts: Styling kwargs forwarded to ``PolygonGlyph``.

        Returns:
            The ``PolyCollection`` (registered as a Scene layer).
        """
        scheme = opts.get("scheme")
        # `isinstance` states the intent: cleopatra's `classify` also accepts a list/ndarray of explicit bin
        # edges as `scheme`, which must never be stringified into this comparison.
        categorical = isinstance(scheme, str) and scheme.lower() == "categorical"
        opts.setdefault("add_colorbar", categorical)  # the Scene owns the colorbar; the glyph owns the legend
        if categorical:
            # Normalize the spelling: cleopatra dispatches on an exact, case-sensitive `== "categorical"`, so a
            # case variant would set up a categorical render here and then fall through to the continuous path
            # there — dying inside `np.isfinite` on a string column. The web tier accepts any case, so
            # normalizing (rather than matching cleopatra's exactness) keeps a `scheme` portable across tiers.
            opts["scheme"] = "categorical"
            # Resolve the colormap here so all three tiers key off ONE sentinel. cleopatra applies the same
            # "swap the continuous default for a qualitative one" rule against a *different* sentinel (its own
            # default, "coolwarm_r"), so left to itself it would honour an explicit cmap="viridis" that the
            # web/interactive tiers swap for tab10 — one cmap, two maps.
            opts["cmap"] = resolve_categorical_cmap(opts.get("cmap"))
            if values is not None:
                # cleopatra's null test is `is None` plus a float-NaN check, so a `pd.NA` from a pandas
                # nullable dtype (`string`, `Int64`, …) would survive it and become a coloured `<NA>` class,
                # shifting every other category's colour — i.e. the same data would render differently
                # depending only on the column's dtype.
                values = nulls_to_none(values)
        if values is not None:
            glyph = PolygonGlyph(polygons, values=values, ax=self.ax, fig=self.fig, **opts)
            artist = self._render_glyph(glyph, artist="plot")
            if categorical:
                _draw_missing_neutral(artist)
            return artist
        glyph = PolygonGlyph(polygons, ax=self.ax, fig=self.fig, **opts)
        return self._render_glyph(glyph, artist="plot", outline_only=True)

    def scatter(self, features: Any, *, scale: Optional[str] = None, **opts) -> Any:
        """Plot a pyramids ``FeatureCollection`` of points, coloured by its value column (``ScatterGlyph``).

        Args:
            features: A pyramids ``FeatureCollection`` (point geometries); reprojected to the display CRS.
            scale: Optional column name whose values set the per-point marker size. Pair
                it with ``size_legend=True`` (and optionally ``size_limits`` / ``size_scale``) to draw a size
                legend. ``None`` (default) uses a single uniform marker size.
            **opts: Styling kwargs forwarded to ``ScatterGlyph`` (``cmap``, ``scheme``, ``size_limits``,
                ``size_scale``, ``size_legend``, ``size_legend_values``, …).

        Returns:
            The scatter ``PathCollection`` (registered as a Scene layer).
        """
        fc = self._vector_input(features, name="scatter")  # empty-guard; any geometry (centroid fallback) OK
        src = get_source(fc)
        values = src.z.values if src.z is not None else None
        sizes = np.asarray(fc[scale], dtype=float) if scale is not None else None
        opts.setdefault("add_colorbar", False)  # the Scene owns the aggregated colorbar
        glyph = ScatterGlyph(
            src.x.values, src.y.values, values=values, sizes=sizes, ax=self.ax, fig=self.fig, **opts,
        )
        return self._render_glyph(glyph, artist="plot")

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
        return self._render_glyph(glyph, artist="plot")

    def point_cloud(self, dataset: Any, **opts) -> Any:
        """Alias of :meth:`grid_points` — scatter raster cell centres coloured by value.

        Returns:
            The scatter ``PathCollection`` (registered as a Scene layer).
        """
        return self.grid_points(dataset, **opts)

    def grid_cells(self, dataset: Any, band: int = 1, **opts) -> Any:
        """Draw raster cells as value-coloured polygons (pyramids ``get_cell_polygons`` → ``PolygonGlyph``).

        Args:
            dataset: A pyramids ``Dataset`` (reprojected to the display CRS first).
            band: 1-based band whose values colour the cells.
            **opts: Styling kwargs, filtered to ``PolygonGlyph``'s accepted options. A ``scheme`` (including
                ``scheme="categorical"``, keyed by a swatch legend) is honoured the same way :meth:`choropleth`
                describes — see ``_polygon_layer``.

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
        values = read_masked_band(ds, band).ravel()  # 1-based band, nodata -> NaN (shared helper)
        polygons, values = self._finite_polygons(polygons, values)  # drop far-side cells on a globe
        return self._polygon_layer(polygons, values, **opts)


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
        im = self._render_glyph(glyph, artist="plot", kind=kind)
        self._last_vector = (glyph, im, kind)  # remembered for quiverkey()
        return im

    def quiver(self, u_dataset: Any, v_dataset: Any, **kwargs) -> Any:
        """Draw a vector field as arrows (``VectorGlyph`` ``kind="quiver"``).

        Returns:
            The ``Quiver`` mappable (registered as a Scene layer; carries the key for :meth:`quiverkey`).
        """
        return self._vector(u_dataset, v_dataset, kind="quiver", **kwargs)

    def barbs(self, u_dataset: Any, v_dataset: Any, **kwargs) -> Any:
        """Draw a vector field as wind barbs (``VectorGlyph`` ``kind="barbs"``).

        Returns:
            The ``Barbs`` mappable (registered as a Scene layer).
        """
        return self._vector(u_dataset, v_dataset, kind="barbs", **kwargs)

    def streamplot(self, u_dataset: Any, v_dataset: Any, **kwargs) -> Any:
        """Draw a vector field as streamlines (``VectorGlyph`` ``kind="streamplot"``).

        Returns:
            The streamplot mappable (registered as a Scene layer).
        """
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
        # cleopatra 0.11.0 exposes the tripcolor/tricontour(f) artist on glyph.im (issue #2).
        if kind == "tripcolor":
            face_values = z[tri.triangles].mean(axis=1)
            return self._render_glyph(glyph, face_values, location="face", colorbar=False, **opts)
        return self._render_glyph(
            glyph, z, location="node", filled=(kind == "tricontourf"), colorbar=False, **opts
        )

    def tricontourf(self, data: Any, **kwargs) -> Any:
        """Filled contours of unstructured/point data (``MeshGlyph`` node data, ``filled=True``).

        Returns:
            The tricontourf mappable (registered as a Scene layer).
        """
        return self._tri(data, kind="tricontourf", **kwargs)

    def tricontour(self, data: Any, **kwargs) -> Any:
        """Line contours of unstructured/point data (``MeshGlyph`` node data, ``filled=False``).

        Returns:
            The tricontour mappable (registered as a Scene layer).
        """
        return self._tri(data, kind="tricontour", **kwargs)

    def tripcolor(self, data: Any, **kwargs) -> Any:
        """Flat-shaded triangles of unstructured/point data (``MeshGlyph`` face data).

        Returns:
            The tripcolor mappable (registered as a Scene layer).
        """
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
            column: Name of the column whose values colour the polygons — numeric for a continuous or
                graduated scale, or any nominal labels (strings, region codes, …) under
                ``scheme="categorical"``.
            **opts: Styling kwargs forwarded to ``PolygonGlyph``. Pass ``scheme`` (e.g. ``"quantiles"`` /
                ``"fisher_jenks"``) + ``k`` to colour by discrete classes instead of a continuous scale, or
                ``scheme="categorical"`` to give every distinct value its own colour (an unordered attribute
                such as a land-use class or region name — ``k`` does not apply, and ``vmin``/``vmax``/
                ``levels``/``color_scale`` are ignored). A categorical fill is keyed by a swatch legend rather
                than a colorbar. For a categorical scheme, ``cmap`` should be a **qualitative**
                (``ListedColormap``) map — ``"tab10"`` (the default), ``"Set2"``, ``"Paired"``, … A continuous
                map (``"viridis"``, ``"plasma"``) is accepted but reads poorly: category colours are drawn from
                its first *n* LUT entries, which on a 256-entry continuous map are near-identical shades (a known
                cleopatra limitation — see `planning/geolibre-parity/upstream-cleopatra-categorical.md`); to
                build categories from a continuous map, sample it into a ``ListedColormap`` yourself. Missing
                values (``NaN``/``None``/``pd.NA``) are drawn a neutral grey, not dropped.
                Note the default ``scheme`` differs by tier: this static tier (like the interactive
                ``choropleth``) defaults to a **continuous** scale, whereas the **web** ``choropleth`` is
                graduated-by-default (``"quantiles"``). Pass ``scheme`` explicitly for identical classification
                across tiers.

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
            - Colour by an unordered attribute — one colour per distinct class, keyed by a swatch legend:
                ```python
                >>> fc["zone"] = ["urban", "rural"] * (len(fc) // 2) + ["urban"] * (len(fc) % 2)
                >>> m = Map(crs=fc.epsg)
                >>> pc = m.choropleth(fc, column="zone", scheme="categorical")
                >>> from matplotlib.colors import BoundaryNorm
                >>> isinstance(pc.norm, BoundaryNorm)  # discrete class codes, not a continuous scale
                True
                >>> [t.get_text() for t in m.layers[-1][0].category_legend.get_texts()]
                ['rural', 'urban']

                ```
        """
        gdf = self._vector_input(features, geom_types=("Polygon", "MultiPolygon"), name="choropleth",
                                 geom_label="polygon")
        polygons, repeats = self._polygon_vertices(gdf.geometry)
        values = np.repeat(gdf[column].to_numpy(), repeats)
        polygons, values = self._finite_polygons(polygons, values)  # drop far-side polygons on a globe
        return self._polygon_layer(polygons, values, **opts)

    def shapes(self, features: Any, **opts) -> Any:
        """Draw polygon outlines without fill (pyramids ``FeatureCollection`` → ``PolygonGlyph`` outline mode).

        Args:
            features: A pyramids ``FeatureCollection`` of polygons (reprojected to the display CRS).
            **opts: Styling kwargs, filtered to ``PolygonGlyph``'s accepted options.

        Returns:
            The ``PolyCollection`` (registered as a Scene layer).
        """
        gdf = self._vector_input(features, geom_types=("Polygon", "MultiPolygon"), name="shapes",
                                 geom_label="polygon")
        polygons, _ = self._polygon_vertices(gdf.geometry)
        polygons, _ = self._finite_polygons(polygons)  # drop far-side polygons on a globe
        return self._polygon_layer(polygons, **opts)

    def _clip_geometry(self, clip: Any) -> Any:
        """Resolve a clip boundary to a single geometry in the display CRS, or ``None``.

        Accepts a pyramids ``FeatureCollection`` / geopandas ``GeoDataFrame``/``GeoSeries`` (reprojected to the
        display CRS and unioned) or a shapely geometry (assumed already in the display CRS). ``None`` → no clip.

        Args:
            clip: A ``FeatureCollection``/``GeoDataFrame``/``GeoSeries`` (reprojected + unioned), a shapely
                geometry already in the display CRS, or ``None``.

        Returns:
            A single shapely geometry in the display CRS, or ``None`` when ``clip`` is ``None``.
        """
        if clip is None:
            return None
        if hasattr(clip, "to_crs"):  # FeatureCollection / GeoDataFrame / GeoSeries
            geoms = clip.to_crs(self.crs)
            geoms = geoms.geometry if hasattr(geoms, "geometry") else geoms
            return geoms.union_all()
        return clip  # shapely geometry, assumed already in the display CRS

    @staticmethod
    def _finite_point_xy(geom: Any, values: Optional[np.ndarray] = None):
        """Return ``(xs, ys, values)`` for points with finite coordinates.

        Points that reproject to non-finite coordinates (the far side of a clipped/globe display CRS) are
        dropped — together with their matching ``values`` — so downstream tessellation / binning / KDE never
        receives ``inf`` / ``nan``.

        Args:
            geom: A geopandas point ``GeoSeries`` (already in the display CRS).
            values: Optional per-point array aligned with ``geom``; filtered by the same finite mask.

        Returns:
            tuple: ``(xs, ys, values)`` numpy arrays of finite points; ``values`` is ``None`` when not given.
        """
        xs = np.asarray(geom.x, dtype=float)
        ys = np.asarray(geom.y, dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if values is not None:
            values = np.asarray(values)[mask]
        return xs[mask], ys[mask], values

    def voronoi(
        self, features: Any, column: Optional[str] = None, *, clip: Any = None, **opts
    ) -> Any:
        """Voronoi diagram of a point ``FeatureCollection`` (pyramids points → cells → ``PolygonGlyph``).

        Tessellates the points into Voronoi cells (``shapely.voronoi_polygons`` with ``ordered=True``, so cell
        *i* belongs to point *i*) and renders them. With ``column`` the cells are filled and coloured by that
        point's value (like :meth:`choropleth`); without it only the cell outlines are drawn (like
        :meth:`shapes`). Points that reproject to non-finite coordinates (the far side of a clipped/globe
        display CRS), and duplicate points, produce no cell and are silently skipped.

        Args:
            features: A pyramids ``FeatureCollection`` of point geometries (reprojected to the display CRS).
            column: Name of the numeric column whose value colours each cell, or ``None`` for outlines only.
            clip: Optional boundary the cells are clipped to — a ``FeatureCollection``/``GeoDataFrame`` (reprojected
                to the display CRS) or a shapely geometry already in the display CRS. ``None`` leaves shapely's
                default bounded cells.
            **opts: Styling kwargs forwarded to ``PolygonGlyph``. Pass ``scheme`` (e.g. ``"quantiles"`` /
                ``"fisher_jenks"``) + ``k`` to colour cells by discrete classes instead of a continuous scale,
                or ``scheme="categorical"`` for one colour per distinct value (keyed by a swatch legend the same
                way :meth:`choropleth` describes — see ``_polygon_layer``).

        Returns:
            The ``PolyCollection`` (registered as a Scene layer).

        Raises:
            ValueError: if ``features`` is not all single ``Point`` geometries.

        Examples:
            - Tessellate the point fixture and colour the cells by ``fid``:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth.scene import Map
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
                >>> m = Map(crs=fc.epsg)
                >>> pc = m.voronoi(fc, column="fid")
                >>> len(m.layers)
                1

                ```
        """
        gdf = self._vector_input(features, geom_types=("Point",), name="voronoi", geom_label="point")
        geom = gdf.geometry
        col_vals = gdf[column].to_numpy() if column is not None else None
        # Drop points with non-finite reprojected coords (far side of a clipped/globe CRS); ordered=True
        # then keeps cell i aligned with input point i, so values map by position.
        xs, ys, col_vals = self._finite_point_xy(geom, col_vals)
        if xs.size == 0:
            raise ValueError("voronoi: no finite points in the display CRS")
        cells = voronoi_polygons(MultiPoint(list(zip(xs, ys))), ordered=True)
        boundary = self._clip_geometry(clip)

        polygons: List[np.ndarray] = []
        values: Optional[list] = [] if column is not None else None
        for i, cell in enumerate(cells.geoms):
            if boundary is not None:
                cell = cell.intersection(boundary)
            if cell.is_empty:
                continue
            parts = cell.geoms if cell.geom_type.startswith("Multi") else [cell]
            for part in parts:
                if part.geom_type != "Polygon" or part.is_empty:
                    continue
                polygons.append(np.asarray(part.exterior.coords))
                if values is not None:
                    values.append(col_vals[i])
        values_arr = np.asarray(values) if values is not None else None
        polygons, values_arr = self._finite_polygons(polygons, values_arr)  # drop far-side cells on a globe
        return self._polygon_layer(polygons, values_arr, **opts)

    @staticmethod
    def _scale_factors(values: np.ndarray, limits: Tuple[float, float]) -> np.ndarray:
        """Linearly map ``values`` onto ``limits`` (a constant input maps to the midpoint factor).

        Args:
            values: Per-feature magnitudes to normalise.
            limits: ``(min, max)`` output range the smallest / largest value map to.

        Returns:
            np.ndarray: Per-feature scale factors spanning ``limits``, the same shape as ``values``.
        """
        lo, hi = limits
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        if vmax > vmin:
            return lo + (values - vmin) * (hi - lo) / (vmax - vmin)
        return np.full(values.shape, (lo + hi) / 2.0)

    def cartogram(
        self, features: Any, scale: str, column: Optional[str] = None, *,
        limits: Tuple[float, float] = (0.2, 1.0), **opts,
    ) -> Any:
        """Cartogram: scale each polygon about its centroid by a value column (pyramids → ``PolygonGlyph``).

        Each feature's geometry is affine-scaled about its own centroid by a factor derived from ``scale``
        (linearly normalised across the layer to ``limits``), distorting area to encode magnitude. With
        ``column`` the scaled polygons are filled and coloured by that column (like :meth:`choropleth`);
        without it only the outlines are drawn (like :meth:`shapes`).

        Args:
            features: A pyramids ``FeatureCollection`` of polygon geometries (reprojected to the display CRS).
            scale: Name of the numeric column whose value sets each feature's size (normalised to ``limits``).
            column: Optional column whose value colours each scaled polygon, or ``None`` for outlines only.
            limits: ``(min, max)`` scale factors mapped to the smallest/largest ``scale`` value.
            **opts: Styling kwargs forwarded to ``PolygonGlyph``. Pass ``scheme`` (e.g. ``"quantiles"`` /
                ``"fisher_jenks"``) + ``k`` to colour by discrete classes instead of a continuous scale, or
                ``scheme="categorical"`` for one colour per distinct value (keyed by a swatch legend the same
                way :meth:`choropleth` describes — see ``_polygon_layer``).

        Returns:
            The ``PolyCollection`` (registered as a Scene layer).

        Raises:
            ValueError: if ``features`` contains non-polygon geometry.

        Examples:
            - Scale buffered points by ``fid`` and colour them by the same column:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth.scene import Map
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
                >>> fc["geometry"] = fc.geometry.buffer(500.0)
                >>> m = Map(crs=fc.epsg)
                >>> pc = m.cartogram(fc, scale="fid", column="fid")
                >>> len(m.layers)
                1

                ```
        """
        gdf = self._vector_input(features, geom_types=("Polygon", "MultiPolygon"), name="cartogram",
                                 geom_label="polygon")
        geom = gdf.geometry
        factors = self._scale_factors(gdf[scale].to_numpy(dtype=float), limits)
        scaled = [
            affine_scale(g, xfact=f, yfact=f, origin="centroid")
            for g, f in zip(geom, factors)
        ]
        polygons, repeats = self._polygon_vertices(scaled)
        if column is not None:
            values = np.repeat(gdf[column].to_numpy(), repeats)
            polygons, values = self._finite_polygons(polygons, values)
            return self._polygon_layer(polygons, values, **opts)
        polygons, _ = self._finite_polygons(polygons)
        return self._polygon_layer(polygons, **opts)

    @staticmethod
    def _quadtree_cells(
        xs: np.ndarray, ys: np.ndarray, agg_fn: Any, nmax: int, nmin: int, max_depth: int = 20,
    ) -> List[Tuple[float, float, float, float, float]]:
        """Recursively split the points' bbox into quadrants until each cell holds ``<= nmax`` points.

        Returns ``(xmin, ymin, xmax, ymax, value)`` per kept cell, where ``value = agg_fn(point_indices)``.
        A cell with fewer than ``nmin`` points is dropped; splitting stops at ``max_depth`` and when a split
        makes no progress (all points fall in one child), so coincident points cannot recurse forever.

        Args:
            xs: Finite point x-coordinates.
            ys: Finite point y-coordinates, aligned with ``xs``.
            agg_fn: Callable mapping an index array of the points in a cell to that cell's scalar value.
            nmax: Maximum points in a cell before it is split.
            nmin: Cells with fewer than this many points are dropped.
            max_depth: Hard recursion-depth cap guarding against coincident points. Default 20.

        Returns:
            list[tuple]: ``(xmin, ymin, xmax, ymax, value)`` for each kept cell.
        """
        x0, x1 = float(np.min(xs)), float(np.max(xs))
        y0, y1 = float(np.min(ys)), float(np.max(ys))
        if x1 <= x0:
            x1 = x0 + 1.0
        if y1 <= y0:
            y1 = y0 + 1.0
        out: List[Tuple[float, float, float, float, float]] = []
        stack = [(x0, y0, x1, y1, np.arange(len(xs)), 0)]
        while stack:
            xmin, ymin, xmax, ymax, idx, depth = stack.pop()
            n = len(idx)
            if n == 0:
                continue
            if n <= nmax or depth >= max_depth:
                if n >= nmin:
                    out.append((xmin, ymin, xmax, ymax, float(agg_fn(idx))))
                continue
            xmid, ymid = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
            cx, cy = xs[idx], ys[idx]
            quads = [
                (xmin, ymin, xmid, ymid, idx[(cx <= xmid) & (cy <= ymid)]),
                (xmid, ymin, xmax, ymid, idx[(cx > xmid) & (cy <= ymid)]),
                (xmin, ymid, xmid, ymax, idx[(cx <= xmid) & (cy > ymid)]),
                (xmid, ymid, xmax, ymax, idx[(cx > xmid) & (cy > ymid)]),
            ]
            nonempty = [q for q in quads if len(q[4]) > 0]
            if len(nonempty) == 1 and len(nonempty[0][4]) == n:  # no progress (coincident points)
                if n >= nmin:
                    out.append((xmin, ymin, xmax, ymax, float(agg_fn(idx))))
                continue
            for qx0, qy0, qx1, qy1, qidx in quads:
                stack.append((qx0, qy0, qx1, qy1, qidx, depth + 1))
        return out

    def quadtree(
        self, features: Any, column: Optional[str] = None, *, agg: Any = "mean",
        nmax: int = 100, nmin: int = 0, clip: Any = None, **opts,
    ) -> Any:
        """Quadtree choropleth: aggregate points into adaptive cells (pyramids points → ``PolygonGlyph``).

        Recursively splits the points' bounding box into quadrants until each cell holds ``<= nmax`` points,
        then colours each cell by an aggregate of ``column`` (or the point **count** when ``column`` is
        ``None``). The cells are always filled (a quadtree is a choropleth).

        Args:
            features: A pyramids ``FeatureCollection`` of point geometries (reprojected to the display CRS).
            column: Numeric column aggregated per cell, or ``None`` to colour by point count (density).
            agg: Per-cell reducer — one of ``"mean"``/``"sum"``/``"median"``/``"min"``/``"max"``/``"std"``/
                ``"count"`` or a callable taking a 1-D array. Ignored when ``column`` is ``None`` (count).
            nmax: Maximum points in a cell before it is split (smaller → finer grid).
            nmin: Cells with fewer than this many points are dropped.
            clip: Optional boundary the cells are clipped to (``FeatureCollection``/``GeoDataFrame`` reprojected,
                or a shapely geometry in the display CRS). ``None`` keeps the full rectangular cells.
            **opts: Styling kwargs forwarded to ``PolygonGlyph``. Pass ``scheme`` (e.g. ``"quantiles"`` /
                ``"fisher_jenks"``) + ``k`` to colour cells by discrete classes instead of a continuous scale,
                or ``scheme="categorical"`` for one colour per distinct value (keyed by a swatch legend the same
                way :meth:`choropleth` describes — see ``_polygon_layer``).

        Returns:
            The ``PolyCollection`` (registered as a Scene layer).

        Raises:
            ValueError: if ``features`` is not all single ``Point`` geometries, or ``agg`` is an unknown name.

        Examples:
            - Aggregate the point fixture into a fine density grid:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth.scene import Map
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
                >>> m = Map(crs=fc.epsg)
                >>> pc = m.quadtree(fc, nmax=1)
                >>> len(m.layers)
                1

                ```
        """
        gdf = self._vector_input(features, geom_types=("Point",), name="quadtree", geom_label="point")
        geom = gdf.geometry
        # Drop points with non-finite reprojected coords (far side of a clipped/globe CRS) before binning.
        col_vals_full = gdf[column].to_numpy(dtype=float) if column is not None else None
        xs, ys, col_vals = self._finite_point_xy(geom, col_vals_full)
        if xs.size == 0:
            raise ValueError("quadtree: no finite points in the display CRS")
        if column is None:
            def agg_fn(idx):
                return float(len(idx))
        else:
            if callable(agg):
                reducer = agg
            elif agg in _QUADTREE_AGG:
                reducer = _QUADTREE_AGG[agg]
            else:
                raise ValueError(
                    f"unknown agg {agg!r}; choose one of {sorted(_QUADTREE_AGG)} or a callable"
                )

            def agg_fn(idx):
                return float(reducer(col_vals[idx]))

        cells = self._quadtree_cells(xs, ys, agg_fn, nmax, nmin)
        boundary = self._clip_geometry(clip)
        polygons: List[np.ndarray] = []
        values: List[float] = []
        for xmin, ymin, xmax, ymax, val in cells:
            if boundary is None:
                polygons.append(
                    np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
                )
                values.append(val)
                continue
            inter = box(xmin, ymin, xmax, ymax).intersection(boundary)
            if inter.is_empty:
                continue
            parts = inter.geoms if inter.geom_type.startswith("Multi") else [inter]
            for part in parts:
                if part.geom_type != "Polygon" or part.is_empty:
                    continue
                polygons.append(np.asarray(part.exterior.coords))
                values.append(val)
        values_arr = np.asarray(values, dtype=float)
        polygons, values_arr = self._finite_polygons(polygons, values_arr)
        return self._polygon_layer(polygons, values_arr, **opts)

    @staticmethod
    def _polygons_of(geom: Any) -> list:
        """Return the ``Polygon`` parts of a shapely geometry (``[]`` for non-polygonal input).

        Args:
            geom: Any shapely geometry, or ``None``.

        Returns:
            list: ``[geom]`` for a Polygon, each Polygon part of a MultiPolygon / GeometryCollection, or
            ``[]`` for ``None`` / non-polygonal geometry.
        """
        if geom is None:
            return []
        kind = geom.geom_type
        if kind == "Polygon":
            return [geom]
        if kind in ("MultiPolygon", "GeometryCollection"):
            return [g for g in geom.geoms if g.geom_type == "Polygon"]
        return []

    def _clip_path(self, clip: Any) -> Optional[MplPath]:
        """Resolve a clip boundary to a matplotlib ``Path`` (data coords) for contour clipping, or ``None``.

        Reuses :meth:`_clip_geometry` to reproject/union the boundary, then turns each polygon exterior ring
        into a sub-path so a ``MultiPolygon`` clips correctly.

        Args:
            clip: A clip boundary accepted by :meth:`_clip_geometry`, or ``None``.

        Returns:
            matplotlib.path.Path | None: A path (in data coords) covering the boundary polygons, or ``None``
            when there is no usable boundary.
        """
        polys = self._polygons_of(self._clip_geometry(clip))
        verts: List[list] = []
        codes: List[int] = []
        for poly in polys:
            ring = np.asarray(poly.exterior.coords)
            if len(ring) < 3:
                continue
            verts.extend(ring.tolist())
            codes.extend(
                [MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 2) + [MplPath.CLOSEPOLY]
            )
        if not verts:
            return None
        return MplPath(np.asarray(verts), codes)

    def kde(self, features: Any, *, clip: Any = None, **opts) -> Any:
        """2-D kernel-density (isochrone) plot of a point ``FeatureCollection`` (pyramids points → ``KDEGlyph``).

        Estimates the point density on a grid and draws it as filled (``shade=True``) or line contours, coloured
        through the shared scalar-mapping pipeline. The KDE is numpy-only (cleopatra ``KDEGlyph``).

        Args:
            features: A pyramids ``FeatureCollection`` of point geometries (reprojected to the display CRS).
            clip: Optional boundary the density is clipped to (``FeatureCollection``/``GeoDataFrame`` reprojected,
                or a shapely geometry in the display CRS). ``None`` draws the full grid.
            **opts: Styling kwargs forwarded to ``KDEGlyph`` (``levels``, ``shade``, ``gridsize``,
                ``bw_method``, ``cmap``, …).

        Returns:
            The contour set (``QuadContourSet``) registered as a Scene layer.

        Raises:
            ValueError: if ``features`` is not all single ``Point`` geometries.

        Examples:
            - Draw a density surface for the point fixture:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth.scene import Map
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
                >>> m = Map(crs=fc.epsg)
                >>> cs = m.kde(fc)
                >>> len(m.layers)
                1

                ```
        """
        gdf = self._vector_input(features, geom_types=("Point",), name="kde", geom_label="point")
        geom = gdf.geometry
        # Drop points with non-finite reprojected coords (far side of a clipped/globe CRS) before the KDE.
        xs, ys, _ = self._finite_point_xy(geom)
        if xs.size == 0:
            raise ValueError("kde: no finite points in the display CRS")
        opts.setdefault("add_colorbar", False)  # the Scene owns the aggregated colorbar
        glyph = KDEGlyph(
            xs, ys, clip_path=self._clip_path(clip), ax=self.ax, fig=self.fig, **opts,
        )
        return self._render_glyph(glyph, artist="plot")

    def sankey(
        self, features: Any, column: Optional[str] = None, scale: Optional[str] = None, **opts,
    ) -> Any:
        """Spatial flow / Sankey map of a line ``FeatureCollection`` (pyramids lines → ``FlowGlyph``).

        Draws each line as a path whose **colour** encodes ``column`` and whose **width** encodes ``scale``
        (each optional). MultiLineStrings contribute one path per part.

        Args:
            features: A pyramids ``FeatureCollection`` of ``LineString``/``MultiLineString`` geometries
                (reprojected to the display CRS).
            column: Numeric column whose value colours each path, or ``None`` for a single colour.
            scale: Numeric column whose value sets each path's line width, or ``None`` for a uniform width.
            **opts: Styling kwargs forwarded to ``FlowGlyph`` (``width_limits``, ``width_scale``, ``cmap``,
                ``size_legend``, …).

        Returns:
            The ``LineCollection`` (registered as a Scene layer).

        Raises:
            ValueError: if ``features`` contains non-line geometry.

        Examples:
            - Draw flow lines coloured and width-scaled by columns:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import geopandas as gpd
                >>> from shapely.geometry import LineString
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth.scene import Map
                >>> gdf = gpd.GeoDataFrame(
                ...     {"flow": [1.0, 2.0]},
                ...     geometry=[LineString([(0, 0), (1, 1)]), LineString([(0, 1), (1, 2)])],
                ...     crs="EPSG:4326",
                ... )
                >>> m = Map(crs=4326)
                >>> lc = m.sankey(FeatureCollection(gdf), column="flow", scale="flow")
                >>> len(m.layers)
                1

                ```
        """
        gdf = self._vector_input(features, geom_types=("LineString", "MultiLineString"), name="sankey",
                                 geom_label="line")
        geom = gdf.geometry
        paths: List[np.ndarray] = []
        repeats: List[int] = []
        for g in geom:
            parts = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
            paths.extend(np.asarray(p.coords) for p in parts)
            repeats.append(len(parts))
        rep = np.asarray(repeats)
        values = np.repeat(gdf[column].to_numpy(), rep) if column is not None else None
        widths = np.repeat(gdf[scale].to_numpy(), rep) if scale is not None else None
        opts.setdefault("add_colorbar", False)  # the Scene owns the aggregated colorbar
        glyph = FlowGlyph(
            paths, values=values, widths=widths, ax=self.ax, fig=self.fig, **opts,
        )
        return self._render_glyph(glyph, artist="plot")

