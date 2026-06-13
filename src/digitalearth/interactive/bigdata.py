"""BigDataMixin — Datashader wrappers for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``rasterize`` / ``datashade`` (DI.2), categorical aggregation (DI.2a) and trajectory aggregation
(DI.2b). The point of this mixin: million-plus-row layers render as **server-rasterized density images**
that re-aggregate on every pan/zoom (``dynamic=True`` attaches a viewport stream), instead of emitting one
browser glyph per feature — the reason this tier scales where the static one cannot.

Datashader output depends on the canvas size, so callers (and tests) pin ``width``/``height`` for
deterministic arrays. Reprojection still happens upstream in pyramids; Datashader aggregates already-
projected planar coordinates.
"""

from typing import Any, Optional

from loguru import logger

from digitalearth.interactive.base import _require_holoviz

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


class BigDataMixin:
    """Datashader builders (DI.2): viewport-rasterized density layers for huge vector data."""

    def _as_element(self, layer: Any, *, vdims: Optional[list] = None) -> Any:
        """Return ``layer`` as a HoloViews element (GeoDataFrames become point layers).

        Args:
            layer: A HoloViews/GeoViews element, or a (Feature)GeoDataFrame routed through the
                vector plumbing (reproject in pyramids → ``gv.Points``).
            vdims: Value dimensions to carry when building from a GeoDataFrame.

        Returns:
            The HoloViews element to rasterize.
        """
        gv, hv = _require_holoviz()
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
    ) -> "BigDataMixin":
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
            This map (chainable).
        """
        gv, hv = _require_holoviz()
        from holoviews.operation.datashader import rasterize as _rasterize

        element = self._as_element(layer, vdims=[column] if column else None)
        op_kwargs = {key: opts.pop(key) for key in ("width", "height") if key in opts}
        rasterized = _rasterize(
            element,
            aggregator=_resolve_aggregator(aggregator, column),
            dynamic=dynamic,
            **op_kwargs,
        )
        rasterized = self._styled(
            rasterized,
            common={"cmap": cmap, "colorbar": True, **opts},
            bokeh={"tools": ["hover"]},
        )
        return self.add_element(rasterized)

    def datashade(
        self,
        layer: Any,
        *,
        cmap: str = "viridis",
        color_key: Optional[dict] = None,
        aggregator: Any = "count",
        column: Optional[str] = None,
        dynamic: bool = True,
        **opts: Any,
    ) -> "BigDataMixin":
        """Add a fully shaded (RGB) density layer — categorical blends via ``color_key`` (DI.2a).

        Unlike :meth:`rasterize` the colour-mapping happens server-side too (no Bokeh colorbar /
        live recolor) — use it when categories must blend per pixel, e.g. landcover classes.

        Args:
            layer: A HoloViews element or a (Feature)GeoDataFrame (becomes a point layer).
            cmap: Colormap for continuous shading (ignored when ``color_key`` is given).
            color_key: ``{category: colour}`` mapping for categorical shading; requires a
                ``count_cat``/``by`` aggregation over ``column``. Non-categorical columns are
                converted (and the conversion logged — no silent dtype switch).
            aggregator: Reduction name or Datashader reduction; ``color_key`` implies
                ``count_cat`` when the aggregator is left at ``"count"``.
            column: The value/category column.
            dynamic: Re-shade on every viewport change; ``False`` bakes a static RGB.
            **opts: ``width``/``height`` pin the canvas; everything else styles the result.

        Returns:
            This map (chainable).
        """
        gv, hv = _require_holoviz()
        from holoviews.operation.datashader import datashade as _datashade

        if color_key is not None and aggregator == "count":
            aggregator = "count_cat"
        if aggregator == "count_cat" and column is not None:
            layer = self._ensure_categorical(layer, column)
        element = self._as_element(layer, vdims=[column] if column else None)
        op_kwargs: dict = {
            key: opts.pop(key) for key in ("width", "height") if key in opts
        }
        if color_key is not None:
            op_kwargs["color_key"] = color_key
        else:
            op_kwargs["cmap"] = cmap
        shaded = _datashade(
            element,
            aggregator=_resolve_aggregator(aggregator, column),
            dynamic=dynamic,
            **op_kwargs,
        )
        return self.add_element(self._styled(shaded, common=opts or None))

    def trajectory(
        self,
        features: Any,
        *,
        track_column: Optional[str] = None,
        by: Optional[str] = None,
        dynspread: bool = True,
        cmap: str = "viridis",
        color_key: Optional[dict] = None,
        dynamic: bool = True,
        **opts: Any,
    ) -> "BigDataMixin":
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
            color_key: ``{category: colour}`` mapping used with ``by``.
            dynamic: Re-shade on every viewport change; ``False`` bakes a static RGB.
            **opts: ``width``/``height`` pin the canvas; everything else styles the result.

        Returns:
            This map (chainable).
        """
        import numpy as np
        import pandas as pd

        gv, hv = _require_holoviz()
        from holoviews.operation.datashader import datashade as _datashade
        from holoviews.operation.datashader import dynspread as _dynspread

        gdf = self._display_gdf(features)
        groups = (
            gdf.groupby(track_column, sort=False) if track_column else [(None, gdf)]
        )
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
        path = hv.Path(table, kdims=["x", "y"], vdims=[by] if by else [])

        op_kwargs: dict = {
            key: opts.pop(key) for key in ("width", "height") if key in opts
        }
        if by is not None:
            import datashader as ds

            op_kwargs["aggregator"] = ds.count_cat(by)
            if color_key is not None:
                op_kwargs["color_key"] = color_key
        else:
            op_kwargs["cmap"] = cmap
        shaded = _datashade(path, dynamic=dynamic, **op_kwargs)
        if dynspread:
            shaded = _dynspread(shaded)
        return self.add_element(self._styled(shaded, common=opts or None))

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
        f"{kind}: {n_features:,} features exceed rasterize_threshold={threshold:,} — "
        "auto-routing through Datashader (pass rasterize=False to force raw glyphs)"
    )
    return True
