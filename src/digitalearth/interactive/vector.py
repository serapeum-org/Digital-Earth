"""VectorMixin — vector builders for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``points`` / ``path`` / ``polygons`` / ``choropleth`` (DI.1b) and the u/v vector fields
``vectorfield`` / ``streamlines`` / ``barbs`` (DI.5, recipe I6); meshes/density (DI.6) and graphs
(DI.15) land later.

Vector layers come straight from the GeoDataFrame pyramids hands over (``FeatureCollection`` *is a*
GeoDataFrame) — GeoViews reads the geometry column natively, so **no shapely/geopandas import** is ever
needed here. The data CRS is declared via ``gv.util.process_crs(self.crs)``: GeoViews builds the cartopy
CRS object *internally* (the hvPlot pattern), keeping cartopy out of this package (DX.3). Reprojection to
the display CRS happens upstream in pyramids (``FeatureCollection.to_crs``) before the element is built.

**Barbs are matplotlib-only** — Bokeh has no wind-barb glyph, so ``barbs`` renders through HoloViews'
matplotlib backend (a static PNG via ``save``); it logs that it is not interactive rather than silently
producing an empty Bokeh layer.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Self, Tuple

from digitalearth.base.crs import reproject
from digitalearth.base.points import PointArrays
from digitalearth.base.spec import DataRef, LayerSpec, Scale, Symbology
from digitalearth.base.spec._serial import thawed_value
from digitalearth.base.symbology import sample_cmap
from digitalearth.interactive.base import (
    _masked_to_nan,
    _require_holoviz,
    _skips_off_limb,
    cmap_name,
    describe,
    describe_style,
    held_props,
    style_value,
)

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.interactive.base import InteractiveMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


#: The style options HoloViews takes as a `list` and refuses as a tuple. These are the builder's **own**
#: resolved values — a sampled colour ramp, a classifier's edges — which the description carries and so
#: stores as tuples: a ramp read back from one is spelled `("#440154", ...)`, which `color_levels` rejects
#: outright (`ClassSelector` of `(int, list, range)`) and which makes a palette harder to read wherever a
#: style is printed. Everything else keeps the spelling it was stored in, `clim` included: that one *is* a
#: pair, and HoloViews reads it as one. The caller's own keywords never go through the freeze at all now —
#: they are held beside the layer, exactly as passed (C1/H3/M9) — so nothing in `opts` needs thawing.
_AS_LISTS: Tuple[str, ...] = ("cmap", "color_levels")


def _classifiable(features: Any) -> Any:
    """Return the frame a column is classified from, opening a path the way the drawer opens one.

    Every builder on this tier takes a path as well as an opened object, and a path is the only input a
    figure can be *written* with: `DataRef.of` records a path as a path and anything else as an `object:`
    reference, which `FigureSpec.to_dict` refuses. The unclassified branches never had to read the data
    themselves — they hand the caller's argument to `add_element` and the renderer opens it at draw time —
    but classifying is different: the class edges or the distinct values have to be known before the layer
    is described, because they are part of the description. Indexing the argument itself answered
    `TypeError: string indices must be integers` for the one input that yields a storable figure
    (review M8).

    Opened through the same reference the layer is registered under, so a path is read by exactly the
    resolver that will read it again at draw time. Only the classification reads this frame; the caller's
    own argument stays the layer's source, so the figure still records the path rather than an object.

    Args:
        features: The caller's argument — a `FeatureCollection`, a GeoDataFrame, or a path or URL to one.

    Returns:
        A frame whose columns can be read. An object is handed back untouched, so the common case opens
        nothing.
    """
    if isinstance(features, (str, os.PathLike)):
        return DataRef.of(features).open()
    return features


def _as_labels(gdf: Any, column: str, missing: str) -> Any:
    """Return a copy of `gdf` with one column read as discrete labels.

    Bokeh colours a numeric column by interpolating the palette; a categorical layer wants one colour per
    value, which it gets by the column arriving as strings. Missing rows take a label of their own so the
    palette can give them the neutral colour rather than a category's.

    Args:
        gdf: The frame to relabel.
        column: The column drawn as categories.
        missing: The label missing rows take.

    Returns:
        A copy of the frame with `column` as strings.
    """
    gdf = gdf.copy()
    absent = gdf[column].isna()
    gdf[column] = gdf[column].astype(str).mask(absent, missing)
    return gdf


def _vector_symbology(
    hv_type: str, vdims: Any, common: dict, labels: Optional[dict] = None
) -> Symbology:
    """Return the description a vector layer is drawn from.

    Args:
        hv_type: The HoloViews element type the layer is built as — `"Points"`, `"Path"`, `"Polygons"`.
            Recorded under `via` as `"geometry"`: these five draw a frame of geometry, which is what tells
            them apart from the aggregating builders that draw the same kinds.
        vdims: The value dimensions the element carries, or `None`.
        common: The resolved style options, as values — the builder's own half. The caller's raw
            keywords are held beside the layer and merged over this by the drawer.
        labels: For a layer coloured by category, the column drawn as labels and the label missing rows
            take — what :func:`_as_labels` needs. `None` for every other layer, which draws its column as
            it is.

    Returns:
        The symbology. Written by a helper rather than inline at five call sites, so the five kinds cannot
        drift in what they record.
    """
    return Symbology(
        props={
            "via": "geometry",
            "hv_type": hv_type,
            "vdims": list(vdims) if vdims else None,
            "common": dict(common),
            "labels": dict(labels) if labels else None,
        }
    )


def draw_vector(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the HoloViews element for any of the kinds that share the vector shape.

    Points, lines, polygons and both choropleth paths are one GeoDataFrame turned into one typed element
    and styled, differing only in the type, the value dimensions and the resolved style — which is why they
    share a drawer as well as a registration funnel.

    Args:
        interactive_map: The map being drawn.
        data: The layer's source — the feature collection it draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    props = held_props(interactive_map, layer)
    gdf = interactive_map._display_gdf(data)
    labels = props.get("labels")
    if labels:
        gdf = _as_labels(gdf, labels["column"], labels["missing"])
    element = interactive_map._vector_element(
        props["hv_type"],
        gdf,
        # Thawed, because the description stores every sequence as a tuple and HoloViews reads a tuple of
        # dimensions as a `(name, label)` pair — the stored `("fid",)` is refused where `["fid"]` is not.
        # Only this property: a `clim` is a pair, and reaches Bokeh as the pair it was written as.
        vdims=thawed_value(props.get("vdims")),
    )
    # The builder's own resolved style, then the caller's raw keywords over it — the precedence the
    # builder applied before the two were split (the description carries the first, the map the second).
    common = {
        key: thawed_value(value) if key in _AS_LISTS else value
        for key, value in dict(props.get("common") or {}).items()
    }
    common.update(dict(props.get("opts") or {}))
    element = interactive_map._styled(
        element, common=common or None, bokeh={"tools": ["hover"]}
    )
    return DrawnLayer(element=element, style=common)


def draw_hexbin(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Bin a layer's points into hexagons and colour them by the described reduction.

    Args:
        interactive_map: The map being drawn.
        data: The layer's source — the points to bin.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.

    Raises:
        KeyError: when the recorded ``column`` is absent from the reprojected frame — the value column a
            hex is coloured by is only read here, so a name that matches nothing is refused here.
    """
    import numpy as np

    from digitalearth.interactive.renderer import DrawnLayer

    gv, _ = _require_holoviz()
    props = held_props(interactive_map, layer)
    gdf = interactive_map._display_gdf(data)
    # Build from explicit display-CRS x/y(/value) arrays, not the geometry GeoDataFrame: GeoViews
    # mis-projects a GeoDataFrame's point geometry at bokeh render time. HoloViews' hex aggregation
    # also needs the reducer as a numpy callable (np.size = count), not a string.
    x = gdf.geometry.x.to_numpy()
    y = gdf.geometry.y.to_numpy()
    reducers = {
        "count": np.size,
        "mean": np.mean,
        "sum": np.sum,
        "min": np.min,
        "max": np.max,
        "std": np.std,
    }
    crs = gv.util.process_crs(interactive_map.crs)
    column = props.get("column")
    if column:
        reducer = reducers.get(props.get("aggregator", "mean"), np.mean)
        element = gv.HexTiles(
            (x, y, gdf[column].to_numpy()),
            kdims=["x", "y"],
            vdims=[column],
            crs=crs,
        )
    else:  # no value column -> count points per hex (np.size), no value dimension
        reducer = np.size
        element = gv.HexTiles((x, y), kdims=["x", "y"], crs=crs)
    common = {
        "cmap": props.get("cmap"),
        "colorbar": True,
        **dict(props.get("opts") or {}),
    }
    element = interactive_map._styled(
        element,
        common=common,
        bokeh={
            "gridsize": props.get("gridsize"),
            "aggregator": reducer,
            "tools": ["hover"],
        },
    )
    return DrawnLayer(element=element, style=common)


def draw_kde(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Estimate the density of a layer's points as a bivariate surface.

    Args:
        interactive_map: The map being drawn.
        data: The layer's source — the points whose density is drawn.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    _, hv = _require_holoviz()
    props = held_props(interactive_map, layer)
    gdf = interactive_map._display_gdf(data)
    element = hv.Bivariate((gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()))
    common = {"cmap": props.get("cmap"), **dict(props.get("opts") or {})}
    element = interactive_map._styled(
        element,
        common=common,
        bokeh={"filled": props.get("filled"), "colorbar": True},
    )
    return DrawnLayer(element=element, style=common)


def draw_uv_field(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the arrow, barb or streamline field a described u/v layer asks for.

    The three share this drawer because they share their data: one decimated u/v grid, differing only in
    the glyph GeoViews draws it with and in which backend renders it.

    Args:
        interactive_map: The map being drawn.
        data: The layer's source — the ``(u, v)`` pair it draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.

    Raises:
        ValueError: when the recorded ``density`` is not in ``(0, 1]`` — the field is decimated by
            ``round(1 / density)``, which nothing outside that range names.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    gv, _ = _require_holoviz()
    props = held_props(interactive_map, layer)
    via = props.get("via")
    u, v = data
    arrays = interactive_map._uv_arrays(
        u, v, band=props.get("band"), density=props.get("density")
    )
    crs = gv.util.process_crs(interactive_map.crs)
    opts = dict(props.get("opts") or {})
    if via == "barbs":
        element = gv.WindBarbs.from_uv(arrays, crs=crs)
    else:
        # Streamlines carry the same data as an arrow field; the matplotlib backend is what draws them
        # as streamlines rather than as arrows.
        element = gv.VectorField.from_uv(arrays, crs=crs)
    if via == "vectorfield":
        common: dict = dict(opts)
        if props.get("color_by") == "magnitude":
            common.update(
                {"color": "Magnitude", "cmap": props.get("cmap"), "colorbar": True}
            )
        element = interactive_map._styled(
            element, common=common, bokeh={"tools": ["hover"]}
        )
        return DrawnLayer(element=element, style=common)
    element = element.opts(backend="matplotlib", **opts) if opts else element
    return DrawnLayer(element=element, style=opts)


def _graph_element(interactive_map: Any, nodes: Any, edges: Any, props: dict) -> Any:
    """Build the GeoViews graph a described graph layer asks for.

    Args:
        interactive_map: The map being drawn.
        nodes: The node table — point geometries carrying the ids the edges name.
        edges: The edge table, or an iterable of ``(source, target[, weight])`` rows.
        props: The layer's recorded properties.

    Returns:
        The ``gv.Graph``.
    """
    import numpy as np
    import pandas as pd

    gv, _ = _require_holoviz()
    weight = props.get("weight")
    node_id = props.get("node_id")
    gdf = interactive_map._display_gdf(nodes)
    crs = gv.util.process_crs(interactive_map.crs)
    ids = (
        gdf[node_id].to_numpy()
        if node_id in getattr(gdf, "columns", [])
        else np.arange(len(gdf))
    )
    gv_nodes = gv.Nodes(
        (gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy(), ids), crs=crs
    )
    # Normalise edges to a DataFrame with named columns - a 3-tuple (src, dst, weight) list with
    # vdims trips gv.Graph's source/target merge, so build the frame explicitly.
    edge_df = edges if isinstance(edges, pd.DataFrame) else pd.DataFrame(list(edges))
    ncols = edge_df.shape[1]
    names = ["source", "target"] + ([weight] if weight and ncols > 2 else [])
    edge_df = edge_df.iloc[:, : len(names)]
    edge_df.columns = names
    vdims = [weight] if (weight and weight in edge_df.columns) else []
    if (
        weight and not vdims
    ):  # weight requested but the edges carry no weight column - say so
        from loguru import logger

        logger.info(
            f"graph: weight={weight!r} requested but the edges have no weight column "
            "(2-tuple edges) - drawing unweighted; pass (src, dst, weight) tuples to weight them"
        )
    return gv.Graph((edge_df, gv_nodes), vdims=vdims, crs=crs)


def draw_graph(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the graph layer a description asks for, bundled or as drawn edges.

    Args:
        interactive_map: The map being drawn.
        data: The layer's source — the ``(nodes, edges)`` pair it draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    props = held_props(interactive_map, layer)
    nodes, edges = data
    graph = _graph_element(interactive_map, nodes, edges, props)
    opts = dict(props.get("opts") or {})
    if props.get("bundle"):
        # Datashade the straight edge paths into a density image (the cheap path; the heavy
        # hammer_bundle is intentionally avoided - see the plan's DI.15 note).
        from holoviews.operation.datashader import datashade

        return DrawnLayer(
            element=interactive_map._styled(
                datashade(graph.edgepaths), common=opts or None
            ),
            style=opts,
        )
    weight = props.get("weight")
    common: dict = dict(opts)
    if weight:
        common.update(
            {"edge_color": weight, "edge_cmap": props.get("cmap"), "colorbar": True}
        )
    element = interactive_map._styled(
        graph, common=common or None, bokeh={"tools": ["hover"]}
    )
    return DrawnLayer(element=element, style=common)


def draw_trimesh(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the triangular mesh a described mesh layer asks for.

    Args:
        interactive_map: The map being drawn.
        data: The layer's source — a UGRID mesh, or the points to triangulate.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    gv, _ = _require_holoviz()
    props = held_props(interactive_map, layer)
    value_column = props.get("value_column")
    # The builder had to build this mesh to count its faces, and hands it over while its own call is still
    # running. Taken only when it was built from this very object and column, so a figure drawn from its
    # description alone — by the renderer, or on another map — builds its own.
    handed = getattr(interactive_map, "_built_mesh", None)
    if handed is not None and handed[0] is data and handed[1] == value_column:
        nodes, simplices, vdims = handed[2]
    else:
        nodes, simplices, vdims = interactive_map._mesh_inputs(data, value_column)
    trimesh = gv.TriMesh(
        (simplices, nodes), crs=gv.util.process_crs(interactive_map.crs)
    )
    opts = dict(props.get("opts") or {})
    # The colormap is only meaningful when the mesh carries a value to colour by; an unvalued mesh draws
    # its wireframe, and a `cmap` on it would be a style nothing reads.
    common = {"cmap": props.get("cmap"), **opts} if vdims else opts
    element = interactive_map._styled(
        trimesh, common=common or None, bokeh={"tools": ["hover"]}
    )
    return DrawnLayer(element=element, style=common)


class VectorMixin(_MixinBase):
    """Vector builders (DI.1b): point, line and polygon layers with hover tooltips.

    A capability mixin of :class:`~digitalearth.interactive.map.InteractiveMap`: it is only ever composed into that
    map class, never instantiated or subclassed on its own. Its methods reach the element registry, the display CRS
    and the render/save lifecycle — and the sibling mixins' methods — through ``self``, and only the composition
    supplies those.

    The ``if TYPE_CHECKING`` base declared above the class is what records that contract for a type checker: it
    resolves each ``self.<attr>`` against :class:`~digitalearth.interactive.base.InteractiveMapBase`, the state
    ``InteractiveMap`` inherits. At runtime that base is plain ``object``, so composing this mixin leaves the
    ``InteractiveMap`` MRO exactly what it was before the annotation.

    See Also:
        digitalearth.interactive.map.InteractiveMap: the composition that supplies the state these methods use.
        digitalearth.interactive.base.InteractiveMapBase: the typing-only base declared above the class.
    """

    def _display_gdf(self, features: Any) -> Any:
        """Reproject ``features`` to the display CRS through pyramids and return the GeoDataFrame.

        Args:
            features: A pyramids ``FeatureCollection`` (or any GeoDataFrame-like with
                ``epsg``/``to_crs``); plain GeoDataFrames already in the display CRS pass through.

        Reprojection goes through the shared :func:`~digitalearth.base.crs.reproject` helper rather than
        calling ``to_crs`` directly, so a warp that places none of the geometry is reported as an
        ``OffLimbError`` here exactly as it is for a raster — which is what lets one skip-and-warn policy
        (#257) cover both halves of the tier.

        Returns:
            The display-CRS GeoDataFrame (read-only from here on — geometry is never rebuilt).
        """
        if (
            hasattr(features, "epsg")
            and hasattr(features, "to_crs")
            and self._needs_reproject(features)
        ):
            features = reproject(features, self.crs)
        return features

    def _vector_element(self, kind: str, gdf: Any, vdims: Optional[list] = None) -> Any:
        """Build a GeoViews element of ``kind`` from a display-CRS GeoDataFrame.

        The element declares its (already display) CRS via ``gv.util.process_crs`` — GeoViews builds
        the cartopy CRS internally, so this package never imports cartopy (DX.3). The ``datatype`` is
        pinned to the geodataframe interface: pyramids' ``FeatureCollection`` is a GeoDataFrame
        *subclass*, which HoloViews' narwhals interface otherwise claims first and mis-reads the
        attribute columns as key dimensions.

        Args:
            kind: One of ``"Points"`` / ``"Path"`` / ``"Polygons"``.
            gdf: The display-CRS GeoDataFrame.
            vdims: Optional value dimensions (attribute columns) carried for colour/hover; ``None``
                lets GeoViews infer them from the non-geometry columns.

        Returns:
            The GeoViews element.
        """
        gv, _ = _require_holoviz()
        crs = gv.util.process_crs(self.crs)
        factory = getattr(gv, kind)
        # geodataframe first so the GeoPandas interface wins at construction; the rest keep
        # render-time clones (projection produces dict/array data) dispatchable.
        datatype = ["geodataframe", "multitabular", "dictionary", "dataframe", "array"]
        kwargs: dict = {"crs": crs, "datatype": datatype}
        if vdims:
            kwargs["vdims"] = vdims
        return factory(gdf, **kwargs)

    @_skips_off_limb
    def points(
        self,
        features: Any,
        *,
        value_column: Optional[str] = None,
        scheme: Optional[Any] = None,
        k: int = 5,
        size: float = 6.0,
        cmap: str = "viridis",
        rasterize: Any = "auto",
        big_data_threshold: Optional[int] = None,
        rasterize_threshold: Optional[int] = None,
        **opts: Any,
    ) -> Self:
        """Add a point layer, optionally coloured by an attribute column.

        Args:
            features: A pyramids ``FeatureCollection`` of point geometries; reprojected to the
                display CRS through pyramids when needed.
            value_column: Optional numeric column colouring the points (also shown on hover).
            scheme: Optional classification scheme for ``value_column`` — ``None`` (the default) is a
                continuous ramp, a named ``cleopatra.styling.styles.classify`` scheme
                (``"quantiles"``, ``"equal_interval"``, ``"fisher_jenks"``, …) cuts it into ``k``
                classes, and ``"categorical"`` gives every distinct value its own colour (the same
                unordered-attribute colouring :meth:`choropleth` does, so a point layer and a polygon
                layer key one column the same way). Spelled and computed the same way on every tier that
                classifies. It styles ``value_column``, so naming a scheme without one is refused rather
                than ignored.
            k: Number of classes a named ``scheme`` is cut into; ignored when ``scheme`` is ``None`` or
                ``"categorical"``.
            size: Marker size in screen pixels — the one thing ``size`` ever means on any tier (#251).
            cmap: Colormap used when ``value_column`` is given.
            rasterize: ``"auto"`` (default) routes through Datashader above
                ``big_data_threshold`` rows — logged, never silent; ``True``/``False`` force it.
            big_data_threshold: Row count above which ``"auto"`` switches to Datashader; ``None``
                (default) uses the map's ``big_data_threshold`` attribute (#250).
            rasterize_threshold: **Deprecated** spelling of ``big_data_threshold`` — the same
                number under the tier's old name. Still accepted (with a ``DeprecationWarning``)
                for one release; passing it together with ``big_data_threshold`` is a
                ``TypeError``, since they name one cutoff (#250).
            **opts: Extra HoloViews style options applied to the element. Anything written here **wins**
                over the option a ``scheme`` derived — the same precedence :meth:`choropleth` applies — so
                ``colorbar=False`` drops the colorbar on a classified point layer just as it does on a
                classified polygon one.

        Examples:
            - Colour gauging stations by an attribute column:
                ```python
                >>> from pyramids.feature import FeatureCollection              # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")  # doctest: +SKIP
                >>> m = InteractiveMap().points(fc, value_column="fid")         # doctest: +SKIP
                >>> [d.name for d in m.layers[0].vdims]                         # doctest: +SKIP
                ['fid']

                ```

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            TypeError: if both ``big_data_threshold`` and the deprecated ``rasterize_threshold``
                are passed — two values for one cutoff, so neither can be silently preferred.
            ValueError: if ``scheme`` is given without a ``value_column`` to classify, or when the
                column cannot be classified (unknown scheme, no spread, ``k < 1``, or an explicit
                colour list shorter than the class count).

        Warns:
            DeprecationWarning: when the deprecated ``rasterize_threshold=`` is used instead of
                ``big_data_threshold=``.
        """
        from digitalearth.interactive.bigdata import _route_through_rasterize

        if scheme is not None and not value_column:
            # A scheme with nothing to classify used to draw plain points and say nothing — a dropped
            # styling request, which is exactly what this tier stopped doing elsewhere (review M2).
            raise ValueError(
                "InteractiveMap.points(): scheme= classifies value_column=, which was not given; "
                "name the column to classify, or drop scheme="
            )
        threshold = self._resolve_big_data_threshold(
            big_data_threshold, rasterize_threshold, caller="InteractiveMap.points()"
        )
        # Nothing here is reprojected: the drawer warps what it draws, and what the description needs — the
        # row count and a column's values — is the same before a warp as after it. Warping here as well did
        # the work twice and threw one result away (review M8). A frame the display CRS cannot place is
        # still refused, by the drawer, and `_skips_off_limb` answers it as before.
        if rasterize is True or (
            rasterize == "auto"
            and _route_through_rasterize("points", len(features), threshold)
        ):
            aggregator = "mean" if value_column else "count"
            return self.rasterize(
                features, aggregator=aggregator, column=value_column, cmap=cmap, **opts
            )
        styling: dict = {}
        labels: Optional[dict] = None
        if value_column and isinstance(scheme, str) and scheme.lower() == "categorical":
            # Categorical colouring works off the string form of the value, so it has to relabel the frame
            # before the element is built — and it is the same relabelling `_categorical_polygons` does, so
            # a point layer and a polygon layer key an unordered column identically (review M3).
            styling, categories, labels = self._categorical_style(
                features, value_column, cmap=cmap
            )
            self.last_breaks = categories
        elif value_column and scheme is not None:
            styling = self._graduated_style(
                features, value_column, scheme=scheme, k=k, cmap=cmap
            )
            self.last_breaks = list(styling["color_levels"])
        elif value_column:
            styling = {"color": value_column, "cmap": cmap, "colorbar": True}
        # The caller's `**opts` are held beside the layer rather than recorded (C1/H3/M9) and merged over
        # this by the drawer, which keeps the precedence: an explicit style the caller wrote outranks the
        # one classification derived, so one scheme cannot mean two things (review M4).
        held: Dict[str, Any] = {"opts": dict(opts)}
        common: dict = {"size": size, **styling}
        return self.add_element(
            None,
            kind="points",
            source=features,
            held=held,
            symbology=_vector_symbology(
                "Points",
                [value_column] if value_column else None,
                describe_style(held, common),
                labels,
            ),
        )

    @_skips_off_limb
    def path(self, features: Any, **opts: Any) -> Self:
        """Add a line layer (LineString / MultiLineString features).

        Args:
            features: A pyramids ``FeatureCollection`` of line geometries; reprojected through
                pyramids when needed.
            **opts: Extra HoloViews style options applied to the element.

        Examples:
            - Draw river reaches as an interactive line layer:
                ```python
                >>> from pyramids.feature import FeatureCollection              # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> reaches = FeatureCollection.read_file("reaches.geojson")    # doctest: +SKIP
                >>> InteractiveMap().path(reaches).save("reaches.html").name    # doctest: +SKIP
                'reaches.html'

                ```

        Returns:
            The same map instance, so builder calls chain.
        """
        return self.add_element(
            None,
            kind="lines",
            source=features,
            held={"opts": dict(opts or {})},
            symbology=_vector_symbology("Path", None, {}),
        )

    @_skips_off_limb
    def polygons(
        self,
        features: Any,
        *,
        column: Optional[str] = None,
        cmap: str = "viridis",
        rasterize: Any = "auto",
        big_data_threshold: Optional[int] = None,
        rasterize_threshold: Optional[int] = None,
        **opts: Any,
    ) -> Self:
        """Add a polygon layer — outlines only, or filled by an attribute column.

        Args:
            features: A pyramids ``FeatureCollection`` of polygon geometries; reprojected through
                pyramids when needed.
            column: Optional numeric column filling the polygons (also shown on hover); ``None``
                draws unfilled outlines.
            cmap: Colormap used when ``column`` is given.
            rasterize: ``"auto"`` (default) routes through Datashader above
                ``big_data_threshold`` rows — logged, never silent; ``True``/``False`` force it.
                Polygon datashading needs the optional ``spatialpandas`` package.
            big_data_threshold: Row count above which ``"auto"`` switches to Datashader; ``None``
                (default) uses the map's ``big_data_threshold`` attribute (#250).
            rasterize_threshold: **Deprecated** spelling of ``big_data_threshold`` — the same
                number under the tier's old name. Still accepted (with a ``DeprecationWarning``)
                for one release; passing it together with ``big_data_threshold`` is a
                ``TypeError``, since they name one cutoff (#250).
            **opts: Extra HoloViews style options applied to the element.

        Examples:
            - Outline catchment polygons over a basemap:
                ```python
                >>> from pyramids.feature import FeatureCollection                # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap           # doctest: +SKIP
                >>> basins = FeatureCollection.read_file("basins.geojson")        # doctest: +SKIP
                >>> InteractiveMap().polygons(basins).tiles().save("m.html").name  # doctest: +SKIP
                'm.html'

                ```

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ImportError: when the layer routes through Datashader (``rasterize=True``, or
                ``"auto"`` above the threshold) and ``spatialpandas`` is not installed. It is
                datashader's own polygon backend, not a Digital-Earth dependency, so the tier says
                which package to install — or to pass ``rasterize=False`` and draw raw glyphs —
                rather than failing deeper inside datashader.
            TypeError: if both ``big_data_threshold`` and the deprecated ``rasterize_threshold``
                are passed — two values for one cutoff, so neither can be silently preferred.

        Warns:
            DeprecationWarning: when the deprecated ``rasterize_threshold=`` is used instead of
                ``big_data_threshold=``.
        """
        threshold = self._resolve_big_data_threshold(
            big_data_threshold, rasterize_threshold, caller="InteractiveMap.polygons()"
        )
        return self._polygon_layer(
            features,
            "polygons",
            column=column,
            cmap=cmap,
            rasterize=rasterize,
            threshold=threshold,
            **opts,
        )

    def _polygon_layer(
        self,
        features: Any,
        kind: str,
        *,
        column: Optional[str],
        cmap: str,
        rasterize: Any,
        threshold: int,
        **opts: Any,
    ) -> Self:
        """Draw polygons outlined or filled by a column, described under the kind the caller asked for.

        The body `polygons()` and the continuous `choropleth()` share. It takes the kind as an argument
        because the two draw the same element yet are different layers to a reader of the figure: the
        continuous ramp used to reach here through `polygons()` and so was described as `polygons`, while
        the categorical and graduated schemes said `choropleth` (review L6). The big-data cutoff arrives
        resolved, since the deprecation warning for its old spelling has to be raised by the public builder
        the caller wrote, a fixed number of frames above them.

        Args:
            features: A pyramids ``FeatureCollection`` of polygon geometries.
            kind: The registered kind to describe the layer as — ``"polygons"`` or ``"choropleth"``.
            column: Optional numeric column filling the polygons; ``None`` draws outlines.
            cmap: Colormap used when ``column`` is given.
            rasterize: ``"auto"``, ``True`` or ``False``, as the public builders take it.
            threshold: The resolved row count above which ``"auto"`` routes through Datashader.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ImportError: when the layer routes through Datashader and ``spatialpandas`` is not installed.
        """
        from digitalearth.interactive.bigdata import _route_through_rasterize

        # Counted on the caller's frame, not a warped copy: the drawer warps what it draws, and a warp keeps
        # every row (review M8). Only the datashaded path below builds its element here, so only it warps.
        if rasterize is True or (
            rasterize == "auto"
            and _route_through_rasterize(kind, len(features), threshold)
        ):
            from importlib.util import find_spec

            if (
                find_spec("spatialpandas") is None
            ):  # datashader's polygon backend (its dep, not ours)
                raise ImportError(
                    "polygon datashading needs the optional spatialpandas package "
                    "(pip install spatialpandas) — or pass rasterize=False to draw raw glyphs"
                )
            element = (
                self._vector_element(  # pragma: no cover - needs optional spatialpandas
                    "Polygons",
                    self._display_gdf(features),
                    vdims=[column] if column else None,
                )
            )
            return self.rasterize(  # pragma: no cover - needs optional spatialpandas
                element,
                aggregator="mean" if column else "count",
                column=column,
                cmap=cmap,
                **opts,
            )
        held: Dict[str, Any] = {"opts": dict(opts)}
        common: dict = {}
        if column:
            common.update(
                {
                    "color": column,
                    "cmap": style_value(held, "cmap", cmap, cmap_name(cmap)),
                    "colorbar": True,
                }
            )
        elif "fill_alpha" not in opts:
            common["fill_alpha"] = 0.0
        return self.add_element(
            None,
            kind=kind,
            source=features,
            held=held,
            symbology=_vector_symbology(
                "Polygons", [column] if column else None, common
            ),
        )

    def _categorical_polygons(
        self, features: Any, column: str, *, cmap: str = "viridis", **opts: Any
    ) -> Self:
        """Fill polygons by a distinct-value attribute, one colour per category (DC.8).

        The categorical counterpart of the continuous :meth:`polygons` path: each distinct value of ``column``
        gets a colour from :func:`digitalearth.base.symbology.categorical_colors`. The colour column is cast to
        **string** and the colours are handed to GeoViews as a ``{label: colour}`` dict ``cmap`` — a numeric
        column would otherwise be treated as a continuous dimension and the palette interpolated, so this is
        what guarantees one discrete colour per distinct value (no continuous colorbar). Missing values
        (``NaN``/``None``) are drawn with a neutral ``"#cccccc"`` fallback, matching the web tier's default.
        The categories (original values, in classifier order) are recorded on ``last_breaks`` for legend parity
        with the web tier.

        Because the fill is keyed on the **string** form of each value, this assumes a single-dtype attribute
        column: two categories that stringify identically (e.g. the integer ``1`` and the string ``"1"``, or
        ``1`` and ``1.0``) collapse to one colour. Realistic categorical columns are single-dtype; the web tier
        keeps such mixed values distinct via its native ``match``.

        Args:
            features: A pyramids ``FeatureCollection`` of polygons (reprojected through pyramids).
            column: The attribute to colour by (any hashable value; assumed single-dtype — see above).
            cmap: A qualitative colormap name (defaults to ``"tab10"`` when left at the continuous default).
            **opts: Extra HoloViews style options.

        Returns:
            The same map instance, so builder calls chain.
        """
        styling, categories, labels = self._categorical_style(
            features, column, cmap=cmap
        )
        self.last_breaks = categories
        held: Dict[str, Any] = {"opts": dict(opts)}
        return self.add_element(
            None,
            kind="choropleth",
            source=features,
            held=held,
            symbology=_vector_symbology(
                "Polygons",
                [column],
                describe_style(held, styling),
                labels,
            ),
        )

    def _categorical_style(self, gdf: Any, column: str, *, cmap: str) -> tuple:
        """Relabel ``column`` as discrete strings and return the options that colour one per value.

        The one distinct-value colouring path in this tier, shared by :meth:`points` and
        :meth:`_categorical_polygons`, so an unordered column keys a point layer and a polygon layer
        identically. Splitting it out is what let ``points(scheme="categorical")`` work at all: it used to
        fall through to :meth:`_graduated_style`, whose classifier coerces the column to ``float`` and so
        died with "could not convert string to float" on the very columns categorical colouring is for
        (review M3).

        Args:
            gdf: The frame carrying ``column`` — the caller's own, unwarped: a column's values are the same
                in any CRS, so the drawer's warp is the only one the layer needs (never mutated — a copy is
                relabelled and returned). A path or URL is opened first, by :func:`_classifiable`, because
                a path is the only input a figure can be written with (review M8).
            column: The attribute to colour by (any hashable value; assumed single-dtype — see
                :meth:`_categorical_polygons` on the string-keyed collapse).
            cmap: A qualitative colormap name, already resolved by the caller's default.

        Returns:
            tuple: ``(options, categories, labels)`` — the ``color``/``cmap``/``colorbar`` options for the
            element, the original values in classifier order (what ``last_breaks`` records), and the
            relabelling the drawer applies: the column to read as discrete strings and the sentinel label
            missing rows take, as :func:`_as_labels` reads them.
        """
        from digitalearth.base.symbology import (
            MISSING_COLOR,
            categorical_colors,
            resolve_categorical_cmap,
        )

        gdf = _classifiable(gdf)
        categories, colors = categorical_colors(
            gdf[column], resolve_categorical_cmap(cmap)
        )
        cmap_by_label = {
            str(category): color for category, color in zip(categories, colors)
        }
        # The column is drawn as discrete labels, each mapped to its own colour, so Bokeh colours it
        # categorically (a numeric column would map continuously and interpolate the palette). The
        # relabelling itself belongs to the drawer — :func:`_as_labels` — because a frame built here would
        # not survive into the description the layer is rebuilt from.
        missing = gdf[column].isna()
        # A sentinel label for missing rows that is guaranteed not to collide with a real category (which
        # could itself stringify to "n/a"/"nan"), so a genuine category is never overwritten by the fallback.
        sentinel = "n/a"
        while sentinel in cmap_by_label:
            sentinel += "_"
        if missing.any():
            # Missing values get an explicit neutral fallback, shared with the web and static tiers.
            cmap_by_label[sentinel] = MISSING_COLOR
        options = {"color": column, "cmap": cmap_by_label, "colorbar": False}
        return options, list(categories), {"column": column, "missing": sentinel}

    def _graduated_style(
        self, gdf: Any, column: str, *, scheme: Any, k: int, cmap: str
    ) -> dict:
        """Classify ``column`` and return the HoloViews options that draw it as flat classes.

        The one classification path in this tier, shared by :meth:`points` and :meth:`choropleth`, so a
        classified point layer and a classified polygon layer cannot land on different class edges. It is a
        thin adapter over :meth:`~digitalearth.base.spec.scale.Scale.breaks_of` — the **same** class edges
        the web and static tiers cut, through the one classifier seam — whose edges become HoloViews
        ``color_levels`` alongside a per-class ``cmap`` list, which is what makes Bokeh draw flat classes
        instead of interpolating the ramp.

        Args:
            gdf: The frame carrying ``column`` — the caller's own, unwarped: a column's values are the same
                in any CRS, so the drawer's warp is the only one the layer needs. A path or URL is opened
                first, by :func:`_classifiable`, because a path is the only input a figure can be written
                with (review M8).
            column: The numeric attribute to classify and colour by.
            scheme: A named scheme or an explicit sequence of class edges.
            k: Number of classes for a named scheme.
            cmap: Colormap sampled once per class, or an explicit sequence of colours — which is taken as
                given, so it must carry exactly one colour per class. A shorter list is **refused**, not
                recycled: the renderer clamps the extra classes onto the last colour it was given, so they
                would render identically and the classification would silently lose classes (review M5).

        Returns:
            dict: the ``color`` / ``cmap`` / ``color_levels`` / ``colorbar`` options for the element.

        Raises:
            ValueError: when the column cannot be classified (unknown scheme, no spread, ``k < 1``, …),
                wrapped with the column/scheme/``k`` context exactly as the web tier's ``_color_expr`` does;
                and when an explicit ``cmap`` sequence carries a different number of colours than the
                scheme produced classes — see the note above.
        """
        gdf = _classifiable(gdf)
        try:
            edges = Scale.breaks_of(gdf[column].to_numpy(), scheme, k)
        except ValueError as err:  # constant column, unknown scheme, k < 1, …
            # Scale's own message already names the scheme and `k`; this adds the one fact it cannot
            # know, the column. Repeating scheme/k here printed both twice in a row.
            raise ValueError(f"cannot classify column {column!r}: {err}") from err
        n_classes = len(edges) - 1
        colours = sample_cmap(cmap, n_classes)
        if len(colours) != n_classes:
            # A colormap *name* is sampled to fit; an explicit sequence is taken as given, so a short one
            # leaves classes sharing the last colour and a long one leaves colours unused — either way the
            # picture no longer shows the classification it claims (review M5).
            raise ValueError(
                f"cannot classify column {column!r}: cmap has "
                f"{len(colours)} colours for {n_classes} classes; pass one colour per class, or a "
                "colormap name to sample"
            )
        return {
            "color": column,
            "cmap": colours,
            "color_levels": [float(edge) for edge in edges],
            "colorbar": True,
        }

    def _graduated_polygons(
        self,
        features: Any,
        column: str,
        *,
        scheme: Any,
        k: int,
        cmap: str,
        **opts: Any,
    ) -> Self:
        """Fill polygons by classified value — one flat colour per class (the graduated scheme).

        A thin adapter over :class:`~digitalearth.base.spec.scale.Scale`, which every tier now cuts its
        class edges through, so ``scheme``/``k`` mean one thing across the four. The edges it returns are
        handed to HoloViews as ``color_levels`` alongside a per-class ``cmap`` list, which is what makes
        Bokeh draw discrete classes instead of interpolating the ramp. The classification itself is never
        reimplemented here.

        Args:
            features: A pyramids ``FeatureCollection`` of polygons (reprojected through pyramids).
            column: The numeric attribute to classify and colour by.
            scheme: A cleopatra classification scheme (``"quantiles"``/``"equal_interval"``/
                ``"fisher_jenks"``/… or an explicit edge sequence).
            k: Number of classes.
            cmap: Colormap name sampled once per class.
            **opts: Extra HoloViews style options.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when the column cannot be classified (unknown scheme, no spread, ``k < 1``, …),
                wrapped with the column/scheme/``k`` context exactly as the web tier's ``_color_expr`` does.
        """
        classified = self._graduated_style(
            features, column, scheme=scheme, k=k, cmap=cmap
        )
        self.last_breaks = list(classified["color_levels"])
        held: Dict[str, Any] = {"opts": dict(opts)}
        return self.add_element(
            None,
            kind="choropleth",
            source=features,
            held=held,
            symbology=_vector_symbology(
                "Polygons",
                [column],
                describe_style(held, classified),
            ),
        )

    @_skips_off_limb
    def choropleth(
        self,
        features: Any,
        column: str,
        *,
        scheme: Optional[str] = None,
        k: int = 5,
        cmap: str = "viridis",
        clim: Optional[Tuple[float, float]] = None,
        **opts: Any,
    ) -> Self:
        """Add a choropleth — polygons filled and coloured by ``column`` (hover shows the value).

        A thin colour-by-attribute :meth:`polygons`, mirroring the static ``Map.choropleth``. Pass
        ``scheme="categorical"`` to colour an unordered attribute by distinct value instead of a continuous
        ramp (DC.8), or a **graduated** scheme (``"quantiles"``/``"equal_interval"``/``"fisher_jenks"``/…)
        to classify a numeric column into ``k`` flat classes. Either way the categories or class edges are
        recorded on ``last_breaks`` for legend parity with the web tier.

        Graduated classification goes through :class:`~digitalearth.base.spec.scale.Scale` — the one place
        every tier reaches the classifier — so the same ``column``/``scheme``/``k`` yields the same breaks on
        every tier. Note the *default* ``scheme`` still differs: this interactive tier (like the static
        ``Map.choropleth``) defaults to a **continuous** ramp, whereas the **web** ``choropleth`` is
        graduated-by-default (``"quantiles"``). Pass ``scheme`` explicitly for identical classification.

        Args:
            features: A pyramids ``FeatureCollection`` of polygon geometries; reprojected through
                pyramids when needed.
            column: The column driving the fill colour (required); numeric for the continuous ramp, or any
                hashable value for ``scheme="categorical"``.
            scheme: ``"categorical"`` for distinct-value colouring; a graduated scheme name (or an explicit
                sequence of class edges) for classified colouring; ``None`` (default) for a continuous ramp.
            k: Number of classes for a graduated ``scheme`` (ignored by ``"categorical"``/``None``);
                ``5`` is the shared cross-tier default (#246).
            cmap: Colormap name (a qualitative map such as ``"tab10"`` is used for the categorical scheme when
                left at the default); sampled once per class for a graduated scheme.
            clim: Optional ``(vmin, vmax)`` colour limits for the continuous ramp; ``None`` auto-scales.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            The same map instance, so builder calls chain.

        Examples:
            - Fill polygons by a population column with fixed colour limits:
                ```python
                >>> from pyramids.feature import FeatureCollection              # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> admin = FeatureCollection.read_file("admin.geojson")        # doctest: +SKIP
                >>> m = InteractiveMap().choropleth(admin, "pop", clim=(0, 1e6))  # doctest: +SKIP
                >>> [d.name for d in m.layers[0].vdims]                         # doctest: +SKIP
                ['pop']

                ```

        Raises:
            KeyError: when ``column`` is not a column of ``features``.
            ValueError: when a graduated ``scheme`` cannot classify ``column`` (unknown scheme name, a
                constant column, ``k < 1``, …).
        """
        if column not in getattr(features, "columns", [column]):
            raise KeyError(
                f"choropleth column {column!r} not found in the feature attributes"
            )
        if isinstance(scheme, str) and scheme.lower() == "categorical":
            return self._categorical_polygons(features, column, cmap=cmap, **opts)
        if scheme is not None:
            if clim is not None:
                opts = {"clim": clim, **opts}
            return self._graduated_polygons(
                features, column, scheme=scheme, k=k, cmap=cmap, **opts
            )
        # Continuous ramp: no discrete breaks — clear any recorded from a prior categorical call.
        self.last_breaks = None
        if clim is not None:
            opts = {"clim": clim, **opts}
        # Resolved here rather than by `polygons()`: this is the frame the deprecation warning for the old
        # spelling is counted from, so a `rasterize_threshold=` written on `choropleth()` is attributed to the
        # caller's line.
        threshold = self._resolve_big_data_threshold(
            opts.pop("big_data_threshold", None),
            opts.pop("rasterize_threshold", None),
            caller="InteractiveMap.choropleth()",
        )
        return self._polygon_layer(
            features,
            "choropleth",
            column=column,
            cmap=cmap,
            rasterize=opts.pop("rasterize", "auto"),
            threshold=threshold,
            **opts,
        )

    def _uv_arrays(self, u: Any, v: Any, *, band: int, density: float) -> tuple:
        """Extract subsampled ``(x, y, u, v)`` display-CRS arrays from two pyramids bands.

        Args:
            u: The eastward-component ``Dataset`` / ``Source`` (reprojected through pyramids).
            v: The northward-component ``Dataset`` / ``Source`` (same grid as ``u``).
            band: 1-based band read from each.
            density: Keep-fraction in ``(0, 1]`` — the grid is strided by ``round(1/density)`` so the
                field stays legible at web resolution (``1.0`` keeps every cell).

        Returns:
            ``(x, y, u, v)`` — 1-D ``x``/``y`` cell-centre coords and 2-D ``u``/``v`` arrays, strided.

        Raises:
            ValueError: when ``density`` is not in ``(0, 1]``.
        """
        if not 0.0 < density <= 1.0:
            raise ValueError(f"density must be in (0, 1], got {density!r}")
        su = self._to_display_source(u, band=band)
        sv = self._to_display_source(v, band=band)
        step = max(1, round(1.0 / density))
        x = su.x.values[::step]
        y = su.y.values[::step]
        u_arr = _masked_to_nan(su.z.values)[::step, ::step]
        v_arr = _masked_to_nan(sv.z.values)[::step, ::step]
        return x, y, u_arr, v_arr

    @_skips_off_limb
    def vectorfield(
        self,
        u: Any,
        v: Any,
        *,
        band: int = 1,
        density: float = 1.0,
        color_by: Optional[str] = "magnitude",
        cmap: str = "viridis",
        **opts: Any,
    ) -> Self:
        """Add a u/v vector field as interactive arrows (parity with ``Map.quiver``, recipe I6).

        Args:
            u: Eastward-component ``Dataset`` / ``Source``; reprojected through pyramids.
            v: Northward-component ``Dataset`` / ``Source`` on the same grid.
            band: 1-based band read from each component.
            density: Keep-fraction in ``(0, 1]`` controlling arrow density (``1.0`` = every cell).
            color_by: ``"magnitude"`` colours arrows by speed; ``None`` draws uniform arrows.
            cmap: Colormap used when ``color_by="magnitude"``.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``density`` is not in ``(0, 1]``.
        """
        _require_holoviz()
        held: Dict[str, Any] = {"opts": dict(opts)}
        return self.add_element(
            None,
            kind="vectors",
            # The pair is the source: a field is not drawable from either component alone.
            source=(u, v),
            held=held,
            symbology=Symbology(
                props={
                    "via": "vectorfield",
                    "band": band,
                    "density": density,
                    "color_by": color_by,
                    "cmap": describe(held, "cmap", cmap, cmap_name(cmap)),
                }
            ),
        )

    @_skips_off_limb
    def streamlines(
        self, u: Any, v: Any, *, band: int = 1, density: float = 1.0, **opts: Any
    ) -> Self:
        """Add streamlines of a u/v field via the matplotlib backend (parity with ``Map.streamplot``).

        Bokeh has no streamline integrator, so streamlines render through HoloViews' matplotlib
        backend (a static layer in the saved PNG). The element is built so ``save("x.png")`` works;
        it is logged as non-interactive rather than emitting an empty Bokeh layer.

        Args:
            u: Eastward-component ``Dataset`` / ``Source``; reprojected through pyramids.
            v: Northward-component ``Dataset`` / ``Source`` on the same grid.
            band: 1-based band read from each component.
            density: Keep-fraction in ``(0, 1]`` subsampling the field before streamline
                integration (same meaning as :meth:`vectorfield`'s ``density``).
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``density`` is not in ``(0, 1]``.
        """
        from loguru import logger

        _require_holoviz()
        logger.info(
            "streamlines render through the matplotlib backend (Bokeh has no streamline glyph); "
            "save to a .png/.svg, not interactive .html"
        )
        return self.add_element(
            None,
            kind="streamlines",
            source=(u, v),
            held={"opts": dict(opts)},
            symbology=Symbology(
                props={
                    "via": "streamlines",
                    "band": band,
                    "density": density,
                }
            ),
        )

    def barbs(
        self, u: Any, v: Any, *, band: int = 1, density: float = 1.0, **opts: Any
    ) -> Self:
        """Add wind barbs of a u/v field — **matplotlib backend only** (parity with ``Map.barbs``).

        ``gv.WindBarbs`` has no Bokeh renderer, so barbs are a static matplotlib layer; this logs
        that they are non-interactive rather than silently producing an empty Bokeh layer.

        Args:
            u: Eastward-component ``Dataset`` / ``Source``; reprojected through pyramids.
            v: Northward-component ``Dataset`` / ``Source`` on the same grid.
            band: 1-based band read from each component.
            density: Keep-fraction in ``(0, 1]`` subsampling the field before drawing barbs (same
                meaning as :meth:`vectorfield`'s ``density``; ``1.0`` keeps every cell).
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ImportError: when the installed GeoViews has no ``WindBarbs`` element.
            ValueError: when ``density`` is not in ``(0, 1]``.
        """
        from loguru import logger

        gv, _ = _require_holoviz()
        # Refused here, because the message answers the builder the caller called.
        if not hasattr(gv, "WindBarbs"):
            raise ImportError(
                "barbs need gv.WindBarbs (GeoViews ≥1.11 with the matplotlib backend); it is "
                "absent in this GeoViews build"
            )
        logger.info(
            "barbs render through the matplotlib backend only (Bokeh has no wind-barb glyph); "
            "save to a .png/.svg, not interactive .html"
        )
        return self.add_element(
            None,
            kind="vectors",
            source=(u, v),
            held={"opts": dict(opts)},
            symbology=Symbology(
                props={
                    "via": "barbs",
                    "band": band,
                    "density": density,
                }
            ),
        )

    @_skips_off_limb
    def trimesh(
        self,
        data: Any,
        *,
        value_column: Optional[str] = None,
        rasterize: Any = "auto",
        big_data_threshold: Optional[int] = None,
        rasterize_threshold: Optional[int] = None,
        cmap: str = "viridis",
        **opts: Any,
    ) -> Self:
        """Add an unstructured triangular mesh (parity with ``Map.tricontour``/``tripcolor``, recipe I7).

        Connectivity comes from one of two sources, both pyramids-fed:

        - a **true UGRID mesh** (``pyramids.netcdf.ugrid.Mesh2d`` — anything exposing ``node_x`` /
          ``node_y`` / ``fan_triangles``): nodes + triangles are taken straight off it, no
          triangulation needed;
        - a **point** ``FeatureCollection``: Delaunay-triangulated locally with ``matplotlib.tri``
          (matplotlib is not a forbidden GIS engine), mirroring the static ``tri*`` path.

        Above ``big_data_threshold`` faces the mesh auto-``rasterize``s to a density image (logged).

        Args:
            data: A UGRID-mesh object or a point ``FeatureCollection`` (reprojected via pyramids).
            value_column: Node-value column (for the FeatureCollection path) driving the colour.
            rasterize: ``"auto"`` (default) rasterizes above the threshold; ``True``/``False`` force it.
            big_data_threshold: Face count above which ``"auto"`` rasterizes; ``None`` (default) uses
                the map's ``big_data_threshold`` attribute (#250).
            rasterize_threshold: **Deprecated** spelling of ``big_data_threshold`` — the same
                number under the tier's old name. Still accepted (with a ``DeprecationWarning``)
                for one release; passing it together with ``big_data_threshold`` is a
                ``TypeError``, since they name one cutoff (#250).
            cmap: Colormap name.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            TypeError: if both ``big_data_threshold`` and the deprecated ``rasterize_threshold``
                are passed — two values for one cutoff, so neither can be silently preferred.

        Warns:
            DeprecationWarning: when the deprecated ``rasterize_threshold=`` is used instead of
                ``big_data_threshold=``.

        Examples:
            - Delaunay-triangulate scattered points; the nodes carry the value column:
                ```python
                >>> from pyramids.feature import FeatureCollection                # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap           # doctest: +SKIP
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")  # doctest: +SKIP
                >>> m = InteractiveMap().trimesh(fc, value_column="fid")          # doctest: +SKIP
                >>> len(m.layers[0].nodes) == len(fc)                             # doctest: +SKIP
                True
                >>> [d.name for d in m.layers[0].nodes.vdims]                     # doctest: +SKIP
                ['fid']

                ```
            - A UGRID mesh brings its own connectivity, so nothing is triangulated locally — its
              four nodes and two faces come through as they are:
                ```python
                >>> import numpy as np                                            # doctest: +SKIP
                >>> from pyramids.netcdf.ugrid import Connectivity, Mesh2d        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap           # doctest: +SKIP
                >>> mesh = Mesh2d(                                                # doctest: +SKIP
                ...     node_x=np.array([0.0, 1.0, 0.5, 1.5]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=Connectivity(
                ...         data=np.array([[0, 1, 2], [1, 3, 2]]),
                ...         fill_value=-1,
                ...         cf_role="face_node_connectivity",
                ...         original_start_index=0,
                ...     ),
                ... )
                >>> len(InteractiveMap().trimesh(mesh).layers[0].nodes)           # doctest: +SKIP
                4

                ```
            - Lowering the cutoff routes the same mesh through Datashader instead, and the
              density image is styled (and recorded) with the ``cmap`` that was asked for:
                ```python
                >>> from digitalearth.interactive import InteractiveMap           # doctest: +SKIP
                >>> m = InteractiveMap()                                          # doctest: +SKIP
                >>> m = m.trimesh(fc, big_data_threshold=1, cmap="magma")         # doctest: +SKIP
                >>> m.layer_styles[0]["common"]["cmap"]                           # doctest: +SKIP
                'magma'
                >>> m.layer_styles[0]["common"]["colorbar"]                       # doctest: +SKIP
                True

                ```
        """
        gv, _ = _require_holoviz()
        threshold = self._resolve_big_data_threshold(
            big_data_threshold, rasterize_threshold, caller="InteractiveMap.trimesh()"
        )
        # The mesh is built here because the routing decision below is made on the face count, which only the
        # built connectivity knows. The drawer would build the same one again from the same data, so the one
        # built here is handed to it for the length of this call (review M8).
        built = self._mesh_inputs(data, value_column)
        nodes, simplices, _ = built
        n_faces = len(simplices)
        if rasterize is True or (rasterize == "auto" and n_faces > threshold):
            from loguru import logger

            if rasterize == "auto":
                logger.info(
                    f"trimesh: {n_faces:,} faces exceed big_data_threshold={threshold:,}"
                    " — rasterizing the mesh to a density image"
                )
            trimesh = gv.TriMesh((simplices, nodes), crs=gv.util.process_crs(self.crs))
            return self.rasterize(trimesh, dynamic=True, cmap=cmap, **opts)
        self._built_mesh = (data, value_column, built)
        try:
            held: Dict[str, Any] = {"opts": dict(opts)}
            return self.add_element(
                None,
                kind="unstructured",
                source=data,
                held=held,
                symbology=Symbology(
                    props={
                        "via": "trimesh",
                        "value_column": value_column,
                        "cmap": describe(held, "cmap", cmap, cmap_name(cmap)),
                    }
                ),
            )
        finally:
            # Released as soon as the layer is drawn: holding it would keep the mesh alive for as long as the
            # map, and a later draw from the description must build its own.
            self._built_mesh = None

    def _mesh_inputs(self, data: Any, value_column: Optional[str]) -> tuple:
        """Return ``(nodes_points, simplices, vdims)`` for :meth:`trimesh`.

        Args:
            data: A UGRID-mesh object (``node_x``/``node_y``/``fan_triangles``) or a point
                ``FeatureCollection``.
            value_column: Node-value column for the FeatureCollection path.

        Returns:
            ``(gv.Points nodes, (n_faces, 3) simplices, vdims list)``.
        """
        import numpy as np

        gv, _ = _require_holoviz()
        crs = gv.util.process_crs(self.crs)
        if all(hasattr(data, attr) for attr in ("node_x", "node_y", "fan_triangles")):
            x = np.asarray(data.node_x)
            y = np.asarray(data.node_y)
            simplices = np.asarray(data.fan_triangles)
            nodes = gv.Points((x, y), crs=crs)
            return nodes, simplices, []
        from matplotlib.tri import (
            Triangulation,
        )  # matplotlib, not a forbidden GIS engine

        gdf = self._display_gdf(data)
        # The extraction and the non-finite drop are PointArrays' — and it keeps the value column in step
        # with the points it removes, which this site was doing with a second application of the mask.
        column = gdf[value_column].to_numpy() if value_column else None
        # centroids=False preserves the error this site raised before: a Delaunay mesh over polygon
        # centroids is exactly the caller whose maths is only meaningful on real points.
        points, (column,) = PointArrays.from_features(gdf, centroids=False).finite(
            column
        )
        x, y = points.x, points.y
        simplices = Triangulation(x, y).triangles
        if value_column:
            z = column
            nodes = gv.Points((x, y, z), vdims=[value_column], crs=crs)
            return nodes, simplices, [value_column]
        return gv.Points((x, y), crs=crs), simplices, []

    @_skips_off_limb
    def hexbin(
        self,
        features: Any,
        *,
        gridsize: int = 30,
        aggregator: str = "mean",
        column: Optional[str] = None,
        cmap: str = "viridis",
        **opts: Any,
    ) -> Self:
        """Add an equal-area hex-bin density layer (honest no-overplot density, recipe I7).

        Args:
            features: A point ``FeatureCollection``; reprojected through pyramids.
            gridsize: Number of hexagons across — higher is finer.
            aggregator: Per-bin reducer (``"count"``/``"mean"``/``"sum"``/…); ``"count"`` ignores
                ``column``.
            column: Value column aggregated per bin (required for non-``count`` aggregators).
            cmap: Colormap name.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            The same map instance, so builder calls chain.
        """
        _require_holoviz()
        held: Dict[str, Any] = {"opts": dict(opts)}
        return self.add_element(
            None,
            kind="heatmap",
            source=features,
            held=held,
            symbology=Symbology(
                props={
                    "via": "hexbin",
                    "gridsize": gridsize,
                    "aggregator": aggregator,
                    "column": column,
                    "cmap": describe(held, "cmap", cmap, cmap_name(cmap)),
                }
            ),
        )

    @_skips_off_limb
    def kde(
        self,
        features: Any,
        *,
        filled: bool = True,
        cmap: str = "viridis",
        **opts: Any,
    ) -> Self:
        """Add a 2-D kernel-density layer of point positions (parity with ``Map.kde``, recipe I7).

        Args:
            features: A point ``FeatureCollection``; reprojected through pyramids.
            filled: Fill the density bands (``True``) or draw contour lines (``False``).
            cmap: Colormap name.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            The same map instance, so builder calls chain.
        """
        _require_holoviz()
        held: Dict[str, Any] = {"opts": dict(opts)}
        return self.add_element(
            None,
            kind="heatmap",
            source=features,
            held=held,
            symbology=Symbology(
                props={
                    "via": "kde",
                    "filled": filled,
                    "cmap": describe(held, "cmap", cmap, cmap_name(cmap)),
                }
            ),
        )

    @_skips_off_limb
    def graph(
        self,
        nodes: Any,
        edges: Any,
        *,
        weight: Optional[str] = None,
        bundle: bool = False,
        node_id: str = "id",
        cmap: str = "viridis",
        **opts: Any,
    ) -> Self:
        """Add a network / origin-destination flow map (parity-plus for ``Map.sankey``, recipe I9).

        Args:
            nodes: A point ``FeatureCollection``/GeoDataFrame of network nodes; reprojected through
                pyramids. Each node's ``node_id`` column is the index the edges reference.
            edges: An iterable of ``(src_id, dst_id[, weight])`` tuples, or a DataFrame with those
                columns.
            weight: Optional edge-weight column driving line width/colour.
            bundle: Bundle edges (``holoviews.operation.connect_edges``) then datashade them — for
                dense networks. Note ``hammer_bundle`` (the heavy variant) is not used; this is the
                cheap ``connect_edges``.
            node_id: The node-id column name on ``nodes``.
            cmap: Colormap used when ``weight`` is given.
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            The same map instance, so builder calls chain.
        """
        _require_holoviz()
        held: Dict[str, Any] = {"opts": dict(opts)}
        return self.add_element(
            None,
            kind="flow",
            # One source, because the layer draws the join of the two: neither the node table nor the
            # edge table describes it on its own.
            source=(nodes, edges),
            held=held,
            symbology=Symbology(
                props={
                    "via": "graph",
                    "weight": weight,
                    "bundle": bundle,
                    "node_id": node_id,
                    "cmap": describe(held, "cmap", cmap, cmap_name(cmap)),
                }
            ),
        )

    def flow(
        self, nodes: Any, edges: Any, *, weight: Optional[str] = None, **opts: Any
    ) -> Self:
        """Spatial-flow alias of :meth:`graph` mirroring ``Map.sankey``'s framing (DI.15).

        Args:
            nodes: A point ``FeatureCollection`` of flow endpoints.
            edges: ``(src_id, dst_id[, weight])`` tuples.
            weight: Optional flow-magnitude column.
            **opts: Forwarded to :meth:`graph`.

        Returns:
            The same map instance, so builder calls chain.
        """
        return self.graph(nodes, edges, weight=weight, **opts)
