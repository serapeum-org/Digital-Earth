"""BigDataMixin — Datashader wrappers for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``rasterize`` / ``datashade`` (DI.2), categorical aggregation (DI.2a) and trajectory aggregation
(DI.2b). The point of this mixin: million-plus-row layers render as **server-rasterized density images**
that re-aggregate on every pan/zoom (``dynamic=True`` attaches a viewport stream), instead of emitting one
browser glyph per feature — the reason this tier scales where the static one cannot.

Datashader output depends on the canvas size, so callers (and tests) pin ``width``/``height`` for
deterministic arrays. Reprojection still happens upstream in pyramids; Datashader aggregates already-
projected planar coordinates.
"""

from typing import TYPE_CHECKING, Any, Mapping, Optional, Self

from loguru import logger

from digitalearth.base.spec import LayerSpec, Symbology
from digitalearth.interactive.base import (
    _require_holoviz,
    cmap_name,
    describe,
    describe_opts,
    held_props,
    style_value,
)

#: Datashader reduction names accepted as ``aggregator=`` strings. ``count`` needs no column; the rest
#: aggregate the ``column=`` argument; ``count_cat`` blends per-category counts (DI.2a).
_AGGREGATORS = ("count", "any", "sum", "mean", "min", "max", "std", "var", "count_cat")


def _resolve_aggregator(aggregator: Any, column: Optional[str]) -> Any:
    """Turn an ``aggregator`` name (+ optional ``column``) into a Datashader reduction.

    Args:
        aggregator: A name from :data:`_AGGREGATORS`, or an already-built Datashader reduction
            (passed through untouched).
        column: The value column the reduction aggregates; required for everything but ``count``.

    Returns:
        A Datashader reduction object.

    Raises:
        ValueError: for an unknown aggregator name, or a column-requiring name without ``column``.
    """
    import datashader as ds

    if aggregator is None:
        # A reduction object has no JSON form, so a figure read back without the held object describes
        # none. Counting is what this tier's builders default to, and what such a figure draws.
        return ds.count()
    if not isinstance(aggregator, str):
        return aggregator
    if aggregator not in _AGGREGATORS:
        raise ValueError(
            f"unknown aggregator {aggregator!r}; choose from {_AGGREGATORS}"
        )
    if aggregator == "count":
        return ds.count()
    if column is None:
        raise ValueError(f"aggregator {aggregator!r} needs a column= to aggregate")
    return getattr(ds, aggregator)(column)


def _color_key(recorded: Any) -> Any:
    """Return a recorded `color_key` in the shape Datashader reads it.

    HoloViews and Datashader take either a ``{category: colour}`` mapping or a colour list in category order.
    A description stores a list as a tuple, so the drawer gets one back as a tuple — and passing that through
    ``dict()``, as both drawers did, refused every colour list (review L6).

    Args:
        recorded: The `color_key` the layer recorded: a mapping, or a sequence of colours.

    Returns:
        A plain ``dict`` for a mapping, a ``list`` for a sequence.
    """
    if isinstance(recorded, Mapping):
        return dict(recorded)
    return list(recorded)


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.interactive.base import InteractiveMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


def _track_path(
    hv: Any, gdf: Any, track_column: Optional[str], by: Optional[str]
) -> Any:
    """Connect ordered point rows into one NaN-separated path per track.

    Datashader aggregates a path with `Canvas.line`, which needs every track in one table separated by a
    row of NaNs — otherwise the last point of one track is joined to the first of the next.

    Args:
        hv: The HoloViews module.
        gdf: The reprojected point frame, in track order.
        track_column: The column identifying each track, or `None` for one track.
        by: A categorical column colouring the tracks, carried as the path's value dimension.

    Returns:
        A ``hv.Path`` over the concatenated tracks.
    """
    import numpy as np
    import pandas as pd

    groups = gdf.groupby(track_column, sort=False) if track_column else [(None, gdf)]
    frames = []
    for _, track in groups:
        frame = pd.DataFrame(
            {"x": track.geometry.x.to_numpy(), "y": track.geometry.y.to_numpy()}
        )
        if by is not None:
            frame[by] = track[by].to_numpy()
        frames.append(frame)
        frames.append(
            frame.iloc[:1].assign(x=np.nan, y=np.nan)
        )  # NaN row separates tracks
    table = pd.concat(frames[:-1], ignore_index=True)
    if by is not None:
        table[by] = table[by].astype("category")
    return hv.Path(table, kdims=["x", "y"], vdims=[by] if by else [])


def draw_rasterize(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Re-aggregate a layer's rows into a numeric image at the described reduction.

    Args:
        interactive_map: The map being drawn.
        data: The layer's source — an element, or a frame that becomes a point layer.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.

    Raises:
        ValueError: when the recorded reduction is not one :func:`_resolve_aggregator` knows, or is one
            that needs a column and none was recorded.
    """
    from holoviews.operation.datashader import rasterize as _rasterize

    from digitalearth.interactive.renderer import DrawnLayer

    _require_holoviz()
    props = held_props(interactive_map, layer)
    column = props.get("column")
    element = interactive_map._as_element(data, vdims=[column] if column else None)
    rasterized = _rasterize(
        element,
        aggregator=_resolve_aggregator(props.get("aggregator"), column),
        dynamic=props.get("dynamic", True),
        **dict(props.get("canvas") or {}),
    )
    # The builder's own style, then the caller's raw keywords over it: the two halves the description
    # and the map hold between them (C1/H3/M9).
    common = {**dict(props.get("common") or {}), **dict(props.get("opts") or {})}
    rasterized = interactive_map._styled(
        rasterized, common=common, bokeh={"tools": ["hover"]}
    )
    return DrawnLayer(element=rasterized, style=common)


def draw_datashade(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Shade a layer's rows into an RGB image at the described reduction.

    Args:
        interactive_map: The map being drawn.
        data: The layer's source — an element, or a frame that becomes a point layer.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.

    Raises:
        ValueError: when the recorded reduction is not one :func:`_resolve_aggregator` knows, or is one
            that needs a column and none was recorded.
    """
    from holoviews.operation.datashader import datashade as _datashade

    from digitalearth.interactive.renderer import DrawnLayer

    _require_holoviz()
    props = held_props(interactive_map, layer)
    column = props.get("column")
    element = interactive_map._as_element(data, vdims=[column] if column else None)
    op_kwargs: dict = dict(props.get("canvas") or {})
    color_key = props.get("color_key")
    if color_key is not None:
        op_kwargs["color_key"] = _color_key(color_key)
    else:
        op_kwargs["cmap"] = props.get("cmap")
    shaded = _datashade(
        element,
        aggregator=_resolve_aggregator(props.get("aggregator"), column),
        dynamic=props.get("dynamic", True),
        **op_kwargs,
    )
    common = {**dict(props.get("common") or {}), **dict(props.get("opts") or {})}
    return DrawnLayer(
        element=interactive_map._styled(shaded, common=common or None), style=common
    )


def draw_trajectory(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Shade a layer's ordered points as track density.

    Args:
        interactive_map: The map being drawn.
        data: The layer's source — the point frame whose row order walks each track.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.
    """
    from holoviews.operation.datashader import datashade as _datashade
    from holoviews.operation.datashader import dynspread as _dynspread

    from digitalearth.interactive.renderer import DrawnLayer

    _, hv = _require_holoviz()
    props = held_props(interactive_map, layer)
    by = props.get("by")
    path = _track_path(
        hv, interactive_map._display_gdf(data), props.get("track_column"), by
    )
    op_kwargs: dict = dict(props.get("canvas") or {})
    if by is not None:
        import datashader as ds

        op_kwargs["aggregator"] = ds.count_cat(by)
        color_key = props.get("color_key")
        if color_key is not None:
            op_kwargs["color_key"] = _color_key(color_key)
    else:
        op_kwargs["cmap"] = props.get("cmap")
    shaded = _datashade(path, dynamic=props.get("dynamic", True), **op_kwargs)
    if props.get("dynspread"):
        shaded = _dynspread(shaded)
    common = {**dict(props.get("common") or {}), **dict(props.get("opts") or {})}
    return DrawnLayer(
        element=interactive_map._styled(shaded, common=common or None), style=common
    )


class BigDataMixin(_MixinBase):
    """Datashader builders (DI.2): viewport-rasterized density layers for huge vector data.

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

    def _as_element(self, layer: Any, *, vdims: Optional[list] = None) -> Any:
        """Return ``layer`` as a HoloViews element (GeoDataFrames become point layers).

        Args:
            layer: A HoloViews/GeoViews element, or a (Feature)GeoDataFrame routed through the
                vector plumbing (reproject in pyramids → ``gv.Points``).
            vdims: Value dimensions to carry when building from a GeoDataFrame.

        Returns:
            The HoloViews element to rasterize.
        """
        _, hv = _require_holoviz()
        if isinstance(layer, hv.core.Dimensioned):
            return layer
        gdf = self._display_gdf(layer)
        return self._vector_element("Points", gdf, vdims=vdims)

    def rasterize(
        self,
        layer: Any,
        *,
        aggregator: Any = "count",
        column: Optional[str] = None,
        dynamic: bool = True,
        cmap: str = "viridis",
        **opts: Any,
    ) -> Self:
        """Add a server-rasterized density layer that re-aggregates on zoom.

        Produces a numeric image (Bokeh keeps colorbar + hover + live recolor), aggregating all
        rows per screen pixel — a million points cost the same as a thousand.

        Args:
            layer: A HoloViews element or a (Feature)GeoDataFrame (becomes a point layer).
            aggregator: Reduction name (``"count"``/``"mean"``/``"sum"``/…/``"count_cat"``) or a
                Datashader reduction object.
            column: The value column aggregated by column-requiring reductions.
            dynamic: Re-rasterize on every viewport change (needs a live kernel/server); ``False``
                bakes a single static image (deterministic — what the tests assert on).
            cmap: Colormap for the rasterized image.
            **opts: ``width``/``height`` pin the canvas; everything else styles the result.

        Returns:
            The same map instance, so builder calls chain.
        """
        _require_holoviz()
        canvas = {key: opts.pop(key) for key in ("width", "height") if key in opts}
        held: dict = {}
        described_opts = describe_opts(held, opts)
        return self.add_element(
            None,
            kind="raster",
            source=layer,
            held=held,
            symbology=Symbology(
                props={
                    "via": "rasterize",
                    # A Datashader reduction is a live object; held beside the layer, and described as
                    # nothing, which is the `"count"` every reader without it aggregates by.
                    "aggregator": describe(held, "aggregator", aggregator),
                    "column": column,
                    "dynamic": dynamic,
                    "canvas": canvas,
                    "common": {
                        "cmap": style_value(held, "cmap", cmap, cmap_name(cmap)),
                        "colorbar": True,
                    },
                    "opts": described_opts,
                }
            ),
        )

    def datashade(
        self,
        layer: Any,
        *,
        cmap: str = "viridis",
        color_key: Optional[Any] = None,
        aggregator: Any = "count",
        column: Optional[str] = None,
        dynamic: bool = True,
        **opts: Any,
    ) -> Self:
        """Add a fully shaded (RGB) density layer — categorical blends via ``color_key`` (DI.2a).

        Unlike :meth:`rasterize` the colour-mapping happens server-side too (no Bokeh colorbar /
        live recolor) — use it when categories must blend per pixel, e.g. landcover classes.

        Args:
            layer: A HoloViews element or a (Feature)GeoDataFrame (becomes a point layer).
            cmap: Colormap for continuous shading (ignored when ``color_key`` is given).
            color_key: ``{category: colour}`` mapping for categorical shading, or a list of colours in
                category order — the two forms HoloViews takes; requires a
                ``count_cat``/``by`` aggregation over ``column``. Non-categorical columns are
                converted (and the conversion logged — no silent dtype switch).
            aggregator: Reduction name or Datashader reduction; ``color_key`` implies
                ``count_cat`` when the aggregator is left at ``"count"``.
            column: The value/category column.
            dynamic: Re-shade on every viewport change; ``False`` bakes a static RGB.
            **opts: ``width``/``height`` pin the canvas; everything else styles the result.

        Returns:
            The same map instance, so builder calls chain.
        """
        _require_holoviz()
        if color_key is not None and aggregator == "count":
            aggregator = "count_cat"
        if aggregator == "count_cat" and column is not None:
            # Cast before the source is recorded, so the frame the layer draws is the categorical one the
            # reduction needs — the drawer opens what was recorded, not what the caller passed.
            layer = self._ensure_categorical(layer, column)
        canvas: dict = {
            key: opts.pop(key) for key in ("width", "height") if key in opts
        }
        held: dict = {}
        described_opts = describe_opts(held, opts)
        return self.add_element(
            None,
            kind="points",
            source=layer,
            held=held,
            symbology=Symbology(
                props={
                    "via": "datashade",
                    "aggregator": describe(held, "aggregator", aggregator),
                    "column": column,
                    "dynamic": dynamic,
                    "canvas": canvas,
                    "color_key": describe(held, "color_key", color_key),
                    "cmap": describe(held, "cmap", cmap, cmap_name(cmap)),
                    "common": {},
                    "opts": described_opts,
                }
            ),
        )

    def trajectory(
        self,
        features: Any,
        *,
        track_column: Optional[str] = None,
        by: Optional[str] = None,
        dynspread: bool = True,
        cmap: str = "viridis",
        color_key: Optional[Any] = None,
        dynamic: bool = True,
        **opts: Any,
    ) -> Self:
        """Datashade millions of ordered track points as line density (GPS/AIS, DI.2b).

        Point rows are connected into per-track paths (NaN-separated, the ``Canvas.line`` recipe)
        and aggregated server-side; with ``by`` the tracks blend per class via ``count_cat``.

        Args:
            features: A point (Feature)GeoDataFrame whose row order walks each track; reprojected
                through pyramids first.
            track_column: Column identifying the track each point belongs to; ``None`` treats the
                whole table as one track.
            by: Optional categorical column colouring tracks per class (``count_cat`` blend).
            dynspread: Grow isolated pixels so sparse tracks stay visible.
            cmap: Colormap for continuous shading (ignored when ``color_key`` is given).
            color_key: ``{category: colour}`` mapping used with ``by``, or a list of colours in category
                order — the two forms HoloViews takes.
            dynamic: Re-shade on every viewport change; ``False`` bakes a static RGB.
            **opts: ``width``/``height`` pin the canvas; everything else styles the result.

        Returns:
            The same map instance, so builder calls chain.
        """
        _require_holoviz()
        canvas: dict = {
            key: opts.pop(key) for key in ("width", "height") if key in opts
        }
        held: dict = {}
        described_opts = describe_opts(held, opts)
        return self.add_element(
            None,
            kind="lines",
            source=features,
            held=held,
            symbology=Symbology(
                props={
                    "via": "trajectory",
                    "track_column": track_column,
                    "by": by,
                    "dynspread": dynspread,
                    "cmap": describe(held, "cmap", cmap, cmap_name(cmap)),
                    "color_key": describe(held, "color_key", color_key),
                    "dynamic": dynamic,
                    "canvas": canvas,
                    "common": {},
                    "opts": described_opts,
                }
            ),
        )

    def _ensure_categorical(self, features: Any, column: str) -> Any:
        """Return ``features`` with ``column`` as a pandas category dtype (logged, never silent).

        Datashader's ``count_cat``/``by`` reductions require a categorical column; converting a
        plain object/int column is a cheap pandas cast, not a GIS operation.

        Args:
            features: The (Feature)GeoDataFrame (or HoloViews element, returned untouched).
            column: The category column name.

        Returns:
            The input with ``column`` cast to ``category`` when it was not already.
        """
        dtype = getattr(getattr(features, column, None), "dtype", None)
        if dtype is not None and str(dtype) != "category":
            logger.info(
                f"datashade: casting column {column!r} (dtype {dtype}) to 'category' for count_cat"
            )
            features = features.copy()
            features[column] = features[column].astype("category")
        return features


def _route_through_rasterize(kind: str, n_features: int, threshold: int) -> bool:
    """Decide (and log) whether a vector builder auto-routes through Datashader.

    The no-silent-caps rule: when the row count crosses the threshold the switch is logged, so a
    user always knows their glyph layer became a density image.

    Args:
        kind: The calling builder name (for the log line).
        n_features: Row count of the layer.
        threshold: The auto-routing threshold.

    Returns:
        ``True`` when the layer should be rasterized instead of drawn as glyphs.
    """
    if n_features <= threshold:
        return False
    logger.info(
        f"{kind}: {n_features:,} features exceed big_data_threshold={threshold:,} — "
        "auto-routing through Datashader (pass rasterize=False to force raw glyphs)"
    )
    return True
