"""TemporalMixin — web-tier time-slider (DW.5, recipe W6).

``timeslider`` draws a vector layer whose features carry a time field and adds an ``ipywidgets`` slider over
the distinct times; moving the slider sets a MapLibre filter so only that time step is shown. The colour
classification is computed once over the **whole** series, so the colour scale is stable across frames.

The slider is wired at :meth:`render` time via :meth:`_wrap_temporal` (returning a slider + map composite);
:meth:`~digitalearth.web.base.WebMapBase.save` still serialises just the map. ipywidgets/maplibre are
imported lazily.
"""

from typing import Any, List, Optional

from digitalearth.web.base import _require_layer_api


class TemporalMixin:
    """Time-slider builder for :class:`~digitalearth.web.map.WebMap`."""

    def timeslider(
        self,
        features: Any,
        *,
        kdim: str = "time",
        column: Optional[str] = None,
        scheme: Optional[Any] = "quantiles",
        k: int = 5,
        cmap: str = "viridis",
        opacity: float = 0.85,
    ) -> "TemporalMixin":
        """Render a time-stepped vector layer with a slider over the ``kdim`` field (recipe W6).

        All features are drawn once (so the classification spans the whole series and the colour scale is
        stable across frames); the slider sets a MapLibre filter to reveal one time step at a time. Polygon
        inputs render as fills (a choropleth when ``column`` is given), point inputs as circles.

        Args:
            features: A pyramids ``FeatureCollection`` / GeoDataFrame whose features carry the ``kdim`` field.
            kdim: The time attribute name to scrub.
            column: Optional value column to colour by (graduated choropleth/circles when set).
            scheme: A cleopatra classification scheme for graduated colouring (with ``column``).
            k: Number of classes for the graduated schemes.
            cmap: matplotlib colormap for the value colouring.
            opacity: Layer opacity in ``[0, 1]``.

        Returns:
            This map (chainable). The slider appears when the map is rendered/shown in a notebook.

        Raises:
            KeyError: when ``kdim`` is not a feature attribute.
        """
        _require_layer_api()
        gdf = self._display_gdf(features)
        if kdim not in getattr(gdf, "columns", []):
            raise KeyError(f"time field {kdim!r} not found in the feature attributes")

        geom_types = set(gdf.geometry.geom_type.unique())
        is_polygon = geom_types <= {"Polygon", "MultiPolygon"}
        if is_polygon and column is not None:
            self.choropleth(gdf, column=column, scheme=scheme, k=k, cmap=cmap, opacity=opacity)
        elif is_polygon:
            self.polygons(gdf, opacity=opacity)
        else:
            self.points(gdf, column=column, scheme=scheme, k=k, cmap=cmap, big=False)

        times = sorted(gdf[kdim].unique().tolist())
        self._temporal = {"layer_id": self._last_layer_id, "kdim": kdim, "times": times}
        return self

    def _wrap_temporal(self, widget: Any) -> Any:
        """Wrap the map ``widget`` in a slider composite that filters the layer by the active time step.

        Args:
            widget: The built MapLibre ``MapWidget``.

        Returns:
            An ``ipywidgets.VBox`` of ``[slider, widget]``; moving the slider calls ``set_filter`` on the
            temporal layer. The first time step is shown initially.
        """
        import ipywidgets

        config = self._temporal or {}
        layer_id, kdim, times = config["layer_id"], config["kdim"], config["times"]

        def _filter_for(value: Any) -> list:
            return ["==", ["get", kdim], value]

        slider = ipywidgets.SelectionSlider(
            options=[(str(t), t) for t in times],
            description=kdim,
            continuous_update=False,
        )

        def _on_change(change: Any) -> None:
            widget.set_filter(layer_id, _filter_for(change["new"]))

        slider.observe(_on_change, names="value")
        if times:
            widget.set_filter(layer_id, _filter_for(times[0]))
        return ipywidgets.VBox([slider, widget])

    def _temporal_times(self) -> List[Any]:
        """Return the distinct time steps of the active time-slider (empty when none is set).

        Returns:
            The sorted distinct ``kdim`` values, or ``[]`` when :meth:`timeslider` has not been called.
        """
        return list(self._temporal["times"]) if self._temporal else []
