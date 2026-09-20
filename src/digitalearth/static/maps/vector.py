"""VectorMixin — vector data and vector-field renders.

Points (scatter/grid_points/grid_cells), polygon products (choropleth/shapes/voronoi/cartogram/quadtree),
unstructured triangulations (tricontour/tricontourf/tripcolor), kernel density, flow/Sankey, and the u/v
vector field (quiver/barbs/streamplot/quiverkey) — all wired onto the matching cleopatra glyphs.
"""

from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
from cleopatra.glyphs.gridded.vector_glyph import VectorGlyph
from cleopatra.glyphs.primitives.flow_glyph import FlowGlyph
from cleopatra.glyphs.primitives.polygon_glyph import PolygonGlyph
from cleopatra.glyphs.primitives.scatter_glyph import ScatterGlyph
from cleopatra.glyphs.stats.kde_glyph import KDEGlyph
from matplotlib.path import Path as MplPath
from pyramids.dataset import Dataset
from shapely import MultiPoint, box, voronoi_polygons
from shapely.affinity import scale as affine_scale

from digitalearth.base.arrays import NAN_REDUCERS, read_masked_band
from digitalearth.base.crs import reproject
from digitalearth.base.deprecation import renamed_parameter
from digitalearth.base.points import PointArrays
from digitalearth.base.sources import get_source
from digitalearth.base.symbology import (
    MISSING_COLOR,
    nulls_to_none,
    resolve_categorical_cmap,
)
from digitalearth.static.maps.base import OffLimbError
from digitalearth.static.render_compat import relocate_flat_style

#: Per-cell reducers accepted by ``Map.quadtree``'s ``agg`` — the shared NaN-aware registry plus a special
#: ``"count"`` (``len`` over the per-cell index array, ignoring the column).
_QUADTREE_AGG = {**NAN_REDUCERS, "count": len}


def _draw_missing_neutral(artist: Any) -> None:
    """Colour a classified layer's missing values neutral instead of invisible.

    cleopatra maps a missing category to ``NaN`` in the class codes, and a colormap's default "bad" colour is
    fully transparent — so a feature whose attribute is missing is drawn as *nothing*, making it
    indistinguishable from a feature that was never in the collection. The same is true of a **graduated**
    scheme, whose ``NaN`` never falls in a class either: the 3-D tier lifts it onto
    :data:`~digitalearth.base.symbology.MISSING_COLOR` (``classified_scalars``), and this does the same here,
    so one classified column reads the same way whichever tier draws it. A *continuous* ramp is left alone —
    it has no classes, and nothing on any tier repaints it.

    ``with_extremes`` returns a *new* colormap rather than mutating in place, which matters: the glyph's may be
    a colormap registered globally under a name, and setting the bad colour on that instance would leak this
    policy into every other plot in the process.

    Args:
        artist: The rendered mappable (a ``PolyCollection``) whose colormap gets the neutral "bad" colour.
    """
    artist.set_cmap(artist.get_cmap().with_extremes(bad=MISSING_COLOR))


def _skips_off_limb(builder: Callable) -> Callable:
    """Wrap a vector builder so data the display CRS cannot place skips the layer instead of raising (#257).

    The raster builders answer an off-limb warp inline (``except OffLimbError: self._skipped_off_limb(kind)``)
    because each of them reprojects at its own point in its own body. The validating vector builders all
    reproject in one place — :meth:`VectorMixin._vector_input` — so the same answer is written once, here, and
    covers the whole body: the warp that places no geometry, and the value/geometry work that follows it.

    Args:
        builder: A vector layer method whose body reprojects through :func:`~digitalearth.base.crs.reproject`.

    Returns:
        The same method, with an off-limb reprojection turned into a skipped layer (``None``) — or, under
        ``strict=True``, into an :class:`~digitalearth.base.crs.OffLimbError` naming the layer.
    """

    @wraps(builder)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Run the builder, answering an off-limb reprojection with a skipped layer.

        Args:
            self: The composed ``Map`` the builder is bound to.
            *args: The builder's positional arguments.
            **kwargs: The builder's keyword arguments.

        Returns:
            Whatever the builder returns, or ``None`` when the layer was skipped.

        Raises:
            OffLimbError: when the map was built with ``strict=True``.
        """
        try:
            return builder(self, *args, **kwargs)
        except OffLimbError:
            self._skipped_off_limb(builder.__name__)
            return None

    return guarded


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.static.maps.base import GeoLayerBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


def _quadtree_reducer(agg: Any, column: Optional[str], col_vals: Any) -> Any:
    """Return the function that gives one quadtree cell its value.

    Args:
        agg: A named reducer, a callable, or ignored when `column` is `None`.
        column: The column being aggregated, or `None` to count points instead.
        col_vals: The per-point values, positionally aligned with the points.

    Returns:
        A callable taking a cell's point indices and returning its scalar value.

    Raises:
        ValueError: for a named reducer nobody registered, listing the ones that are.
    """
    if column is None:
        return lambda idx: float(len(idx))
    if callable(agg):
        reducer = agg
    elif agg in _QUADTREE_AGG:
        reducer = _QUADTREE_AGG[agg]
    else:
        raise ValueError(
            f"unknown agg {agg!r}; choose one of {sorted(_QUADTREE_AGG)} or a callable"
        )
    return lambda idx: float(reducer(col_vals[idx]))


def _clipped_cell_boxes(cells: Any, boundary: Any) -> tuple:
    """Return each quadtree cell as a ring, clipped to `boundary`, with the value it carries.

    Unclipped, a cell is its own four corners. Clipped, it can vanish, or break into several parts that
    are not all polygons — an intersection against a globe's limb produces both — so a cell contributes
    zero, one or several rings, each carrying the cell's value.

    Args:
        cells: `(xmin, ymin, xmax, ymax, value)` per cell.
        boundary: The clip geometry, or `None` to keep every cell whole.

    Returns:
        `(rings, values)`, positionally aligned.
    """
    rings: List[np.ndarray] = []
    values: List[float] = []
    for xmin, ymin, xmax, ymax, value in cells:
        if boundary is None:
            rings.append(
                np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
            )
            values.append(value)
            continue
        clipped = box(xmin, ymin, xmax, ymax).intersection(boundary)
        for ring in _polygon_rings(clipped):
            rings.append(ring)
            values.append(value)
    return rings, values


def _quadrants(xs: np.ndarray, ys: np.ndarray, idx: np.ndarray, box: tuple) -> list:
    """Split one quadtree cell into its four children, partitioning the points that fall in it.

    Args:
        xs: All point x-coordinates.
        ys: All point y-coordinates, aligned with `xs`.
        idx: The indices of the points inside this cell.
        box: The cell as `(xmin, ymin, xmax, ymax)`.

    Returns:
        Four `(xmin, ymin, xmax, ymax, indices)` tuples, south-west first then clockwise. A child may hold
        no points; that is the caller's cue, not an error.
    """
    xmin, ymin, xmax, ymax = box
    xmid, ymid = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    cx, cy = xs[idx], ys[idx]
    return [
        (xmin, ymin, xmid, ymid, idx[(cx <= xmid) & (cy <= ymid)]),
        (xmid, ymin, xmax, ymid, idx[(cx > xmid) & (cy <= ymid)]),
        (xmin, ymid, xmid, ymax, idx[(cx <= xmid) & (cy > ymid)]),
        (xmid, ymid, xmax, ymax, idx[(cx > xmid) & (cy > ymid)]),
    ]


def _split_made_no_progress(quads: list, held: int) -> bool:
    """Whether splitting a cell moved nothing — every point landed in one child.

    Args:
        quads: The four children from :func:`_quadrants`.
        held: How many points the parent held.

    Returns:
        `True` when one child holds them all, which happens for coincident points and would otherwise
        recurse until the depth cap.
    """
    nonempty = [quad for quad in quads if len(quad[4]) > 0]
    return len(nonempty) == 1 and len(nonempty[0][4]) == held


def _polygon_rings(geometry: Any) -> list:
    """Return the exterior ring of every polygon part of a geometry.

    A clip can hand back nothing, one polygon, or a multipart geometry whose parts are not all polygons —
    an intersection against a globe's limb produces both. A part with no area has no exterior ring to read,
    so it contributes nothing rather than raising from `part.exterior`.

    Note:
        Only a `Multi*` geometry is unpacked, which is what both callers did inline before sharing this.
        A `GeometryCollection` is therefore read as a single non-polygon and contributes nothing; that is
        existing behaviour, preserved deliberately rather than quietly widened here.

    Args:
        geometry: Any shapely geometry, typically the result of an intersection.

    Returns:
        One coordinate array per polygon part, in the order the parts are held.
    """
    if geometry.is_empty:
        return []
    parts = geometry.geoms if geometry.geom_type.startswith("Multi") else [geometry]
    return [
        np.asarray(part.exterior.coords)
        for part in parts
        if part.geom_type == "Polygon" and not part.is_empty
    ]


def _clipped_cell_rings(
    cells: Any, boundary: Any, col_vals: Any, column: Optional[str]
) -> tuple:
    """Return each Voronoi cell's exterior ring, clipped to `boundary`, with the value it carries.

    A clipped cell can come back empty, or as a multipart geometry whose parts are not all polygons — an
    intersection against a globe's limb produces both — so a cell contributes zero, one or several rings.
    `ordered=True` keeps cell *i* aligned with input point *i*, which is what lets the value follow the
    ring however many parts the cell broke into.

    Args:
        cells: The `voronoi_polygons` result.
        boundary: The clip geometry, or `None` to keep every cell whole.
        col_vals: The per-point values, positionally aligned with `cells`, or `None`.
        column: The column the values came from, or `None` when the draw is outline-only.

    Returns:
        `(rings, values)` — the exterior coordinate arrays, and the values beside them or `None`.
    """
    rings: List[np.ndarray] = []
    values: Optional[list] = [] if column is not None else None
    for index, cell in enumerate(cells.geoms):
        clipped = cell if boundary is None else cell.intersection(boundary)
        for ring in _polygon_rings(clipped):
            rings.append(ring)
            if values is not None:
                values.append(col_vals[index])
    return rings, values


class VectorMixin(_MixinBase):
    """Vector-data and vector-field renders for :class:`~digitalearth.static.map.Map`.

    A capability mixin of :class:`~digitalearth.static.map.Map`: it is only ever composed into that map class, never
    instantiated or subclassed on its own. Its methods reach the shared figure/axes, the layer registry and the
    display CRS — and the sibling mixins' methods — through ``self``, and only the composition supplies those.

    The ``if TYPE_CHECKING`` base declared above the class is what records that contract for a type checker: it
    resolves each ``self.<attr>`` against :class:`~digitalearth.static.maps.base.GeoLayerBase`, the state ``Map``
    inherits. At runtime that base is plain ``object``, so composing this mixin leaves the ``Map`` MRO exactly what
    it was before the annotation.

    See Also:
        digitalearth.static.map.Map: the composition that supplies the state these methods use.
        digitalearth.static.maps.base.GeoLayerBase: the typing-only base declared above the class.
    """

    def _vector_input(
        self,
        features: Any,
        *,
        geom_types: Optional[Sequence[str]] = None,
        name: str = "layer",
        geom_label: Optional[str] = None,
    ) -> Any:
        """Reproject a ``FeatureCollection`` to the display CRS, reject empty, and validate its geometry type.

        Consolidates the preamble shared by the validating vector methods. Reprojects ``features`` to
        :attr:`crs`, raises on an empty collection, and — when ``geom_types`` is given — requires every
        geometry to be one of those types.

        Reprojection goes through the shared :func:`~digitalearth.base.crs.reproject` helper rather than
        calling ``to_crs`` directly: a vector warp does not raise when it places nothing, it hands back
        ``inf`` coordinates, and only that helper reads the result and reports it. Calling ``to_crs`` here
        is what kept the off-limb contract off this tier — the one tier with a clipped globe to be behind,
        and so the one where a layer really can land nowhere.

        Args:
            features: A pyramids ``FeatureCollection`` (reprojected to the display CRS).
            geom_types: Allowed shapely ``geom_type`` names (e.g. ``("Point",)``); ``None`` skips the check.
            name: The calling method's name, used in error messages.
            geom_label: Human label for the allowed geometry in the error (defaults to the joined types).

        Returns:
            The reprojected GeoDataFrame.

        Raises:
            OffLimbError: when the warp places none of the geometry in the display CRS. The public builders
                wrap it with :func:`_skips_off_limb`, so the layer is skipped (or raised under ``strict``).
            ValueError: if the collection is empty, or a geometry is not one of ``geom_types``.
        """
        gdf = reproject(features, self.crs)
        if len(gdf) == 0:
            raise ValueError(f"{name} got an empty FeatureCollection (nothing to draw)")
        if (
            geom_types is not None
            and not gdf.geometry.geom_type.isin(list(geom_types)).all()
        ):
            label = geom_label or " / ".join(geom_types)
            raise ValueError(
                f"{name} requires a FeatureCollection of {label} geometries"
            )
        return gdf

    def _polygon_layer(
        self, polygons: List[np.ndarray], values: Optional[np.ndarray] = None, **opts
    ) -> Any:
        """Draw polygons as a value-filled (``values`` given) or outline-only ``PolygonGlyph`` layer.

        Consolidates the fill-vs-outline branch shared by :meth:`grid_cells`, :meth:`choropleth`,
        :meth:`shapes`, :meth:`voronoi`, :meth:`cartogram` and :meth:`quadtree`. The Scene owns the
        aggregated colorbar, so the glyph's own colorbar is suppressed by default — except under
        ``scheme="categorical"``, where the value key is a per-class swatch legend the glyph builds from its
        own mapping (``PolygonGlyph.category_legend``). The Scene's colorbar cannot stand in for it: a
        categorical fill feeds the mappable opaque integer class codes, so a colorbar over them would read
        ``0, 1, 2 …`` instead of the category labels. Passing ``add_colorbar=False`` suppresses that swatch
        legend — for a caller keying the map some other way, e.g. drawing one shared legend across several
        layers via :meth:`~digitalearth.static.scene.Scene.legend` (which takes explicit ``colors``/``labels``;
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
        # Any scheme — categorical or graduated — puts a missing value outside every class, so it is the
        # classified layers (not the continuous ramp) that need the neutral "bad" colour.
        classified = scheme is not None
        opts.setdefault(
            "add_colorbar", categorical
        )  # the Scene owns the colorbar; the glyph owns the legend
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
        # scheme/k moved onto the plot() `classify` group; pull them off the constructor kwargs.
        plot_style = relocate_flat_style(opts)
        if values is not None:
            glyph = PolygonGlyph(
                polygons, values=values, ax=self.ax, fig=self.fig, **opts
            )
            artist = self._render_glyph(glyph, artist="plot", **plot_style)
            if classified:
                _draw_missing_neutral(artist)
            return artist
        glyph = PolygonGlyph(polygons, ax=self.ax, fig=self.fig, **opts)
        return self._render_glyph(glyph, artist="plot", outline_only=True, **plot_style)

    @_skips_off_limb
    def scatter(
        self,
        features: Any,
        *,
        size_column: Optional[str] = None,
        scale: Optional[str] = None,
        **opts,
    ) -> Any:
        """Plot a pyramids ``FeatureCollection`` of points, sized by a column (``ScatterGlyph``).

        Args:
            features: A pyramids ``FeatureCollection`` (point geometries); reprojected to the display CRS.
            size_column: Optional column name whose values set the per-point marker size. Pair it with
                ``size_legend=True`` (and optionally ``size_limits`` / ``size_scale``) to draw a size
                legend. ``None`` (default) uses a single uniform marker size — set that size with
                ``size`` (which every backend spells the same way).
            scale: Deprecated spelling of ``size_column``; it names a column, not a magnification, and ``size``
                is what sets a marker's visual size on every backend. Still accepted (with a
                ``DeprecationWarning``) for one release.
            **opts: Styling kwargs forwarded to ``ScatterGlyph`` (``cmap``, ``scheme``, ``k``, ``size``,
                ``size_limits``, ``size_scale``, ``size_legend``, ``size_legend_values``, …).
                ``size`` is the marker's visual size, spelled the same way on every backend;
                cleopatra's own ``point_size`` is the deprecated spelling of it, still
                accepted (with a ``DeprecationWarning``) for one release, and passing both
                is a ``TypeError``.

        Returns:
            The scatter ``PathCollection`` (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            TypeError: if both ``size_column`` and the deprecated ``scale`` are passed, or both
                ``size`` and the deprecated ``point_size`` — each pair names one parameter,
                so preferring one silently would drop the other.

        Warns:
            DeprecationWarning: when ``scale=`` is used instead of ``size_column=``, or when cleopatra's
                ``point_size=`` is used instead of ``size=``. Both old spellings keep working
                for one release.
        """
        size_column = renamed_parameter(
            new="size_column",
            value=size_column,
            old="scale",
            alias=scale,
            caller="Map.scatter()",
        )
        fc = self._vector_input(
            features, name="scatter"
        )  # empty-guard; any geometry (centroid fallback) OK
        src = get_source(fc)
        values = src.z.values if src.z is not None else None
        sizes = (
            np.asarray(fc[size_column], dtype=float)
            if size_column is not None
            else None
        )
        opts.setdefault("add_colorbar", False)  # the Scene owns the aggregated colorbar
        # scheme/k -> plot() classify group; the `size` channel -> the glyph's own `point_size`.
        plot_style = relocate_flat_style(opts, marker_size_for="Map.scatter()")
        glyph = ScatterGlyph(
            src.x.values,
            src.y.values,
            values=values,
            sizes=sizes,
            ax=self.ax,
            fig=self.fig,
            **opts,
        )
        return self._render_glyph(glyph, artist="plot", **plot_style)

    def grid_points(
        self,
        dataset: Any,
        *,
        _alias_caller: str = "Map.grid_points()",
        _alias_depth: int = 5,
        **opts,
    ) -> Any:
        """Plot raster cell centres as points coloured by value (pyramids ``to_xyz`` → ``ScatterGlyph``).

        Args:
            dataset: A pyramids ``Dataset`` (reprojected to the display CRS first).
            _alias_caller: Which method a ``DeprecationWarning`` raised on the way through names.
                Defaults to ``"Map.grid_points()"``; :meth:`point_cloud` passes its own name, so the
                warning blames the method the caller actually wrote.
            _alias_depth: How many stack frames sit between that warning and the caller's line.
                Defaults to ``5`` for a direct call; :meth:`point_cloud` passes ``6``, the one extra
                frame its delegation adds.
            **opts: Styling kwargs, filtered to ``ScatterGlyph``'s accepted options. ``size``
                sets the marker size (the cross-backend spelling); cleopatra's ``point_size``
                is its deprecated alias, accepted with a ``DeprecationWarning`` for one
                release, and passing both raises.

        Returns:
            The scatter ``PathCollection`` (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Examples:
            - Scatter the raster's valid cell centres and count the resulting layer:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.static import Map
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> m = Map(crs=ds.epsg)
                >>> _ = m.grid_points(ds)
                >>> len(m.layers)
                1

                ```
        """
        try:
            xyz = self._reproject(dataset).to_xyz()
        except OffLimbError:
            self._skipped_off_limb("grid_points")
            return None
        x = xyz.iloc[:, 0].to_numpy()
        y = xyz.iloc[:, 1].to_numpy()
        z = xyz.iloc[:, 2].to_numpy()
        opts.setdefault("add_colorbar", False)  # the Scene owns the aggregated colorbar
        # The caller and frame depth are parameters because :meth:`point_cloud` delegates here: a
        # deprecation warning must name the method the user actually called, and point at their line.
        plot_style = relocate_flat_style(
            opts, marker_size_for=_alias_caller, depth=_alias_depth
        )  # scheme/k -> plot() classify group; the `size` channel -> the glyph's own `point_size`
        glyph = ScatterGlyph(x, y, values=z, ax=self.ax, fig=self.fig, **opts)
        return self._render_glyph(glyph, artist="plot", **plot_style)

    def point_cloud(self, dataset: Any, **opts) -> Any:
        """Alias of :meth:`grid_points` — scatter raster cell centres coloured by value.

        A ``DeprecationWarning`` raised on the way through (cleopatra's ``point_size=`` instead of
        ``size=``) names ``Map.point_cloud()`` and points at the caller's own line, rather than at the
        method this delegates to.

        Args:
            dataset: A pyramids ``Dataset`` (reprojected to the display CRS first).
            **opts: Styling kwargs, forwarded to :meth:`grid_points` unchanged.

        Returns:
            The scatter ``PathCollection`` (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.
        """
        return self.grid_points(
            dataset,
            _alias_caller="Map.point_cloud()",
            _alias_depth=6,  # one frame further out than grid_points: this alias delegates to it
            **opts,
        )

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
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Examples:
            - Draw one polygon per raster cell and confirm the count equals rows*columns:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.static import Map
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> m = Map(crs=ds.epsg)
                >>> pc = m.grid_cells(ds)
                >>> len(pc.get_paths()) == ds.rows * ds.columns
                True

                ```
        """
        try:
            ds = self._reproject(dataset)
        except OffLimbError:
            self._skipped_off_limb("grid_cells")
            return None
        if ds.epsg is None:
            # Work around pyramids#979: get_cell_polygons labels the returned frame with `ds.epsg` and raises
            # on `None` (pyramids >=0.47 no longer fabricates EPSG:4326 for a CRS with no authority — e.g. an
            # orthographic globe). We read only the cell geometry, in display-CRS coordinates computed from
            # the geotransform, so the authority code is cosmetic; give the transient reprojected copy a
            # placeholder EPSG (coordinates are untouched) so the call runs. Drop this once pyramids#979 ships.
            ds.epsg = 4326
        polygons = [
            np.asarray(g.exterior.coords) for g in ds.get_cell_polygons().geometry
        ]
        values = read_masked_band(
            ds, band
        ).ravel()  # 1-based band, nodata -> NaN (shared helper)
        polygons, values = self._finite_polygons(
            polygons, values
        )  # drop far-side cells on a globe
        return self._polygon_layer(polygons, values, **opts)

    def _vector(
        self, u_dataset: Any, v_dataset: Any, *, kind: str, band: int = 1, **opts
    ) -> Any:
        """Render a vector field from two rasters (u, v) on a shared grid via ``cleopatra.VectorGlyph``.

        Args:
            u_dataset: pyramids ``Dataset`` of the u (eastward) component.
            v_dataset: pyramids ``Dataset`` of the v (northward) component.
            kind: ``"quiver"``, ``"barbs"`` or ``"streamplot"``.
            band: 1-based band index read from each dataset.
            **opts: Styling kwargs, filtered to ``VectorGlyph``'s accepted options.

        Returns:
            The vector mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.
        """
        try:
            su = self._prepare(u_dataset, band)
            sv = self._prepare(v_dataset, band)
        except OffLimbError:
            self._skipped_off_limb(kind)
            return None
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
        plot_style = relocate_flat_style(opts)
        glyph = VectorGlyph(
            x_grid,
            y_grid,
            u,
            v,
            ax=self.ax,
            fig=self.fig,
            **opts,
        )
        im = self._render_glyph(glyph, artist="plot", kind=kind, **plot_style)
        self._last_vector = (glyph, im, kind)  # remembered for quiverkey()
        return im

    def quiver(self, u_dataset: Any, v_dataset: Any, **kwargs) -> Any:
        """Draw a vector field as arrows (``VectorGlyph`` ``kind="quiver"``).

        Returns:
            The ``Quiver`` mappable (registered as a Scene layer; carries the key for :meth:`quiverkey`).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.
        """
        return self._vector(u_dataset, v_dataset, kind="quiver", **kwargs)

    def barbs(self, u_dataset: Any, v_dataset: Any, **kwargs) -> Any:
        """Draw a vector field as wind barbs (``VectorGlyph`` ``kind="barbs"``).

        Returns:
            The ``Barbs`` mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.
        """
        return self._vector(u_dataset, v_dataset, kind="barbs", **kwargs)

    def streamplot(self, u_dataset: Any, v_dataset: Any, **kwargs) -> Any:
        """Draw a vector field as streamlines (``VectorGlyph`` ``kind="streamplot"``).

        Returns:
            The streamplot mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.
        """
        return self._vector(u_dataset, v_dataset, kind="streamplot", **kwargs)

    def quiverkey(
        self,
        value: float,
        text: str,
        *,
        x: float = 0.9,
        y: float = 0.95,
        labelpos: str = "E",
        **kwargs,
    ) -> Any:
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
            raise ValueError(
                "quiverkey() needs a prior quiver(...) layer (barbs/streamplot have no key)"
            )
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
        src = get_source(reproject(data, self.crs))
        if src.z is None:
            raise ValueError("FeatureCollection has no numeric value column to contour")
        return src.x.values, src.y.values, src.z.values

    def _tri(self, data: Any, *, kind: str, **opts) -> Any:
        """Triangulate scattered points and render via ``cleopatra.MeshGlyph``.

        Args:
            data: A pyramids ``Dataset`` (its cells become points) or a ``FeatureCollection``.
            kind: The triangulated render to draw (``tricontourf`` / ``tricontour`` /
                ``tripcolor``).
            **opts: Styling kwargs forwarded to the glyph.

        Returns:
            The mappable (registered as a Scene layer), or ``None`` when three points were supplied
            but fewer than three survive the reprojection — a vector warp sends off-view points to
            infinity rather than raising, and losing them all means the same as an off-limb raster:
            nothing on the view to draw.

        Raises:
            ValueError: when fewer than three points were supplied in the first place, which no
                projection can fix.
        """
        from matplotlib.tri import Triangulation

        try:
            x, y, z = self._scattered(data)
        except OffLimbError:
            self._skipped_off_limb(kind)
            return None
        finite = np.isfinite(x) & np.isfinite(
            y
        )  # drop far-side points on a globe (Triangulation needs finite)
        supplied = np.asarray(x).size
        x, y, z = np.asarray(x)[finite], np.asarray(y)[finite], np.asarray(z)[finite]
        if x.size < 3:
            if supplied < 3:
                # Never enough points to triangulate, whatever the projection — a caller error, and the
                # accurate complaint is the one matplotlib would give.
                raise ValueError(
                    f"{kind}() needs at least three points to triangulate, got {supplied}"
                )
            # There were enough, and the reprojection took them: a vector warp does not raise when the
            # data is off the view, it sends the points to infinity for the filter above to drop. That
            # means the same as an OffLimbError does for a raster — nothing on the view to draw.
            self._skipped_off_limb(kind)
            return None
        tri = Triangulation(x, y)
        glyph = MeshGlyph(x, y, tri.triangles, ax=self.ax, fig=self.fig)
        # cleopatra 0.11.0 exposes the tripcolor/tricontour(f) artist on glyph.im (issue #2).
        if kind == "tripcolor":
            face_values = z[tri.triangles].mean(axis=1)
            return self._render_glyph(
                glyph, face_values, location="face", colorbar=False, **opts
            )
        return self._render_glyph(
            glyph,
            z,
            location="node",
            filled=(kind == "tricontourf"),
            colorbar=False,
            **opts,
        )

    def tricontourf(self, data: Any, **kwargs) -> Any:
        """Filled contours of unstructured/point data (``MeshGlyph`` node data, ``filled=True``).

        Returns:
            The tricontourf mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.
        """
        return self._tri(data, kind="tricontourf", **kwargs)

    def tricontour(self, data: Any, **kwargs) -> Any:
        """Line contours of unstructured/point data (``MeshGlyph`` node data, ``filled=False``).

        Returns:
            The tricontour mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.
        """
        return self._tri(data, kind="tricontour", **kwargs)

    def tripcolor(self, data: Any, **kwargs) -> Any:
        """Flat-shaded triangles of unstructured/point data (``MeshGlyph`` face data).

        Returns:
            The tripcolor mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.
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
    def _finite_polygons(
        polygons: List[np.ndarray], values: Optional[np.ndarray] = None
    ) -> tuple:
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
                >>> from digitalearth.static import Map
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
                >>> from digitalearth.static import Map
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

    @_skips_off_limb
    def choropleth(
        self,
        features: Any,
        column: str,
        *,
        scheme: Optional[Any] = None,
        k: int = 5,
        **opts,
    ) -> Any:
        """Fill polygons coloured by a feature attribute (pyramids ``FeatureCollection`` → ``PolygonGlyph``).

        Args:
            features: A pyramids ``FeatureCollection`` of polygons (reprojected to the display CRS).
            column: Name of the column whose values colour the polygons — numeric for a continuous or
                graduated scale, or any nominal labels (strings, region codes, …) under
                ``scheme="categorical"``.
            scheme: How the values are classified. ``None`` (default) is a **continuous** ramp; a named
                scheme (``"quantiles"``, ``"fisher_jenks"``, ``"equal_interval"``, …) or an explicit
                sequence of bin edges is graduated into ``k`` classes; ``"categorical"`` gives every
                distinct value its own colour (an unordered attribute such as a land-use class or region
                name — ``k`` does not apply, and ``vmin``/``vmax``/``levels``/``color_scale`` are ignored),
                keyed by a swatch legend rather than a colorbar. Spelled the same way, with the same
                default, on every backend: ``scheme=None`` is a continuous ramp everywhere, so the same
                call classifies identically on all four tiers. Either kind of classification leaves a
                missing value (``NaN``/``None``/``pd.NA``) outside every class, so those features are
                drawn a neutral grey — not dropped, and not painted as the lowest class. A continuous
                ramp has no classes and is left to matplotlib.
            k: Number of classes a named ``scheme`` is cut into (ignored when ``scheme`` is ``None`` or
                ``"categorical"``).
            **opts: Styling kwargs forwarded to ``PolygonGlyph``. For a categorical scheme, ``cmap`` should
                be a **qualitative** (``ListedColormap``) map — ``"tab10"`` (the default), ``"Set2"``,
                ``"Paired"``, … A continuous ``LinearSegmentedColormap`` (``"coolwarm"``, ``"RdBu"``) is
                sampled at evenly-spaced points so the categories stay distinct; a perceptual
                ``ListedColormap`` (``"viridis"``, ``"plasma"``) is accepted but reads poorly (its first
                *n* of 256 entries are near-identical shades). The colours are identical to the
                web/interactive tiers either way.

        Returns:
            The ``PolyCollection`` (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Examples:
            - Colour buffered point features by their ``fid`` column and count the drawn polygons:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth.static import Map
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
        gdf = self._vector_input(
            features,
            geom_types=("Polygon", "MultiPolygon"),
            name="choropleth",
            geom_label="polygon",
        )
        polygons, repeats = self._polygon_vertices(gdf.geometry)
        values = np.repeat(gdf[column].to_numpy(), repeats)
        polygons, values = self._finite_polygons(
            polygons, values
        )  # drop far-side polygons on a globe
        if scheme is not None:  # None means a continuous ramp: classify nothing
            opts["scheme"], opts["k"] = scheme, k
        return self._polygon_layer(polygons, values, **opts)

    @_skips_off_limb
    def shapes(self, features: Any, **opts) -> Any:
        """Draw polygon outlines without fill (pyramids ``FeatureCollection`` → ``PolygonGlyph`` outline mode).

        Args:
            features: A pyramids ``FeatureCollection`` of polygons (reprojected to the display CRS).
            **opts: Styling kwargs, filtered to ``PolygonGlyph``'s accepted options.

        Returns:
            The ``PolyCollection`` (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.
        """
        gdf = self._vector_input(
            features,
            geom_types=("Polygon", "MultiPolygon"),
            name="shapes",
            geom_label="polygon",
        )
        polygons, _ = self._polygon_vertices(gdf.geometry)
        polygons, _ = self._finite_polygons(
            polygons
        )  # drop far-side polygons on a globe
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

        This is now a thin adapter over :class:`~digitalearth.base.points.PointArrays`, which is this helper
        lifted into `base/` so the other three tiers stop hand-rolling it. Kept as a method because the three
        call sites here want the flat ``(xs, ys, values)`` triple rather than the value object.

        Args:
            geom: A geopandas point ``GeoSeries`` (already in the display CRS).
            values: Optional per-point array aligned with ``geom``; filtered by the same finite mask.

        Returns:
            tuple: ``(xs, ys, values)`` numpy arrays of finite points; ``values`` is ``None`` when not given.
        """
        points, (filtered,) = PointArrays.of(geom.x, geom.y).finite(values)
        return points.x, points.y, filtered

    @_skips_off_limb
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
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: if ``features`` is not all single ``Point`` geometries.

        Examples:
            - Tessellate the point fixture and colour the cells by ``fid``:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth.static import Map
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
                >>> m = Map(crs=fc.epsg)
                >>> pc = m.voronoi(fc, column="fid")
                >>> len(m.layers)
                1

                ```
        """
        gdf = self._vector_input(
            features, geom_types=("Point",), name="voronoi", geom_label="point"
        )
        geom = gdf.geometry
        col_vals = gdf[column].to_numpy() if column is not None else None
        # Drop points with non-finite reprojected coords (far side of a clipped/globe CRS); ordered=True
        # then keeps cell i aligned with input point i, so values map by position.
        xs, ys, col_vals = self._finite_point_xy(geom, col_vals)
        if xs.size == 0:
            raise ValueError("voronoi: no finite points in the display CRS")
        cells = voronoi_polygons(MultiPoint(list(zip(xs, ys))), ordered=True)
        boundary = self._clip_geometry(clip)

        polygons, values = _clipped_cell_rings(cells, boundary, col_vals, column)
        values_arr = np.asarray(values) if values is not None else None
        polygons, values_arr = self._finite_polygons(
            polygons, values_arr
        )  # drop far-side cells on a globe
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

    @_skips_off_limb
    def cartogram(
        self,
        features: Any,
        scale: str,
        column: Optional[str] = None,
        *,
        limits: Tuple[float, float] = (0.2, 1.0),
        **opts,
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
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: if ``features`` contains non-polygon geometry.

        Examples:
            - Scale buffered points by ``fid`` and colour them by the same column:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth.static import Map
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
                >>> fc["geometry"] = fc.geometry.buffer(500.0)
                >>> m = Map(crs=fc.epsg)
                >>> pc = m.cartogram(fc, scale="fid", column="fid")
                >>> len(m.layers)
                1

                ```
        """
        gdf = self._vector_input(
            features,
            geom_types=("Polygon", "MultiPolygon"),
            name="cartogram",
            geom_label="polygon",
        )
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
        xs: np.ndarray,
        ys: np.ndarray,
        agg_fn: Any,
        nmax: int,
        nmin: int,
        max_depth: int = 20,
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
            quads = _quadrants(xs, ys, idx, (xmin, ymin, xmax, ymax))
            if _split_made_no_progress(quads, n):
                # Coincident points: every child holds the whole cell, so splitting again would recurse
                # forever. The cell is kept whole instead.
                if n >= nmin:
                    out.append((xmin, ymin, xmax, ymax, float(agg_fn(idx))))
                continue
            for qx0, qy0, qx1, qy1, qidx in quads:
                stack.append((qx0, qy0, qx1, qy1, qidx, depth + 1))
        return out

    @_skips_off_limb
    def quadtree(
        self,
        features: Any,
        column: Optional[str] = None,
        *,
        agg: Any = "mean",
        nmax: int = 100,
        nmin: int = 0,
        clip: Any = None,
        **opts,
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
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: if ``features`` is not all single ``Point`` geometries, or ``agg`` is an unknown name.

        Examples:
            - Aggregate the point fixture into a fine density grid:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth.static import Map
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
                >>> m = Map(crs=fc.epsg)
                >>> pc = m.quadtree(fc, nmax=1)
                >>> len(m.layers)
                1

                ```
        """
        gdf = self._vector_input(
            features, geom_types=("Point",), name="quadtree", geom_label="point"
        )
        geom = gdf.geometry
        # Drop points with non-finite reprojected coords (far side of a clipped/globe CRS) before binning.
        col_vals_full = (
            gdf[column].to_numpy(dtype=float) if column is not None else None
        )
        xs, ys, col_vals = self._finite_point_xy(geom, col_vals_full)
        if xs.size == 0:
            raise ValueError("quadtree: no finite points in the display CRS")
        agg_fn = _quadtree_reducer(agg, column, col_vals)
        cells = self._quadtree_cells(xs, ys, agg_fn, nmax, nmin)
        boundary = self._clip_geometry(clip)
        polygons, values = _clipped_cell_boxes(cells, boundary)
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
                [MplPath.MOVETO]
                + [MplPath.LINETO] * (len(ring) - 2)
                + [MplPath.CLOSEPOLY]
            )
        if not verts:
            return None
        return MplPath(np.asarray(verts), codes)

    @_skips_off_limb
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
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: if ``features`` is not all single ``Point`` geometries.

        Examples:
            - Draw a density surface for the point fixture:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth.static import Map
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
                >>> m = Map(crs=fc.epsg)
                >>> cs = m.kde(fc)
                >>> len(m.layers)
                1

                ```
        """
        gdf = self._vector_input(
            features, geom_types=("Point",), name="kde", geom_label="point"
        )
        geom = gdf.geometry
        # Drop points with non-finite reprojected coords (far side of a clipped/globe CRS) before the KDE.
        xs, ys, _ = self._finite_point_xy(geom)
        if xs.size == 0:
            raise ValueError("kde: no finite points in the display CRS")
        opts.setdefault("add_colorbar", False)  # the Scene owns the aggregated colorbar
        plot_style = relocate_flat_style(
            opts
        )  # levels/… -> plot() contour/data_style groups
        glyph = KDEGlyph(
            xs,
            ys,
            clip_path=self._clip_path(clip),
            ax=self.ax,
            fig=self.fig,
            **opts,
        )
        return self._render_glyph(glyph, artist="plot", **plot_style)

    @_skips_off_limb
    def sankey(
        self,
        features: Any,
        column: Optional[str] = None,
        scale: Optional[str] = None,
        **opts,
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
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

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
                >>> from digitalearth.static import Map
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
        gdf = self._vector_input(
            features,
            geom_types=("LineString", "MultiLineString"),
            name="sankey",
            geom_label="line",
        )
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
        plot_style = relocate_flat_style(opts)  # scheme/k -> plot() classify group
        glyph = FlowGlyph(
            paths,
            values=values,
            widths=widths,
            ax=self.ax,
            fig=self.fig,
            **opts,
        )
        return self._render_glyph(glyph, artist="plot", **plot_style)
