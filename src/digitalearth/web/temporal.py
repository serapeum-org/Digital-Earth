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

    @staticmethod
    def _require_features(features: Any, method: str) -> None:
        """Raise ``TypeError`` unless ``features`` is a vector layer (the only thing the slider maps).

        The raster/vector split across the temporal APIs is the thing a caller is most likely to get wrong,
        and a raster that reaches the attribute-membership test below dies inside it on an unrelated error:
        a pyramids ``Dataset``/``DatasetCollection`` exposes ``columns`` as an ``int`` (the grid width in
        cells), so ``kdim not in ...`` raises ``TypeError: argument of type 'int' is not iterable`` before
        the intended ``KeyError`` can. Called *before* :meth:`_display_gdf` so a raster in another CRS is
        not warped on its way to being rejected.

        Args:
            features: The caller's input, expected to be a pyramids ``FeatureCollection`` / GeoDataFrame.
            method: The calling builder name (quoted in the error).

        Raises:
            TypeError: when ``features`` exposes no ``geometry`` — a raster, a bare array, anything
                non-vector — with the raster-capable tiers named in the message.

        Examples:
            - A vector-like input passes the guard silently (the check returns nothing):
                ```python
                >>> from digitalearth.web import WebMap
                >>> layer = type("Layer", (), {"geometry": ()})()
                >>> print(WebMap._require_features(layer, "timeslider"))
                None

                ```
            - A raster is turned away, and the message names the type that arrived:
                ```python
                >>> from digitalearth.web import WebMap
                >>> try:
                ...     WebMap._require_features(42, "timeslider")
                ... except TypeError as err:
                ...     print(str(err).split(". ")[0])
                timeslider() renders a time-stepped vector layer; got a int

                ```
            - The message routes the caller to the tiers that do take a raster stack:
                ```python
                >>> from digitalearth.web import WebMap
                >>> try:
                ...     WebMap._require_features(42, "timeslider")
                ... except TypeError as err:
                ...     print("Map.animate" in str(err), "InteractiveMap.timecube" in str(err))
                True True

                ```

        See Also:
            digitalearth.web.bigdata.BigDataMixin._require_points: the sibling geometry-kind guard this
                mirrors, which rejects non-point input for the heatmap/cluster builders.
        """
        if not hasattr(features, "geometry"):
            raise TypeError(
                f"{method}() renders a time-stepped vector layer; got a "
                f"{type(features).__name__}. For a raster time stack use the matplotlib tier "
                f"(Map.animate) or the interactive tier (InteractiveMap.timecube)."
            )

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
            TypeError: when ``features`` is not a vector layer (e.g. a pyramids ``Dataset`` /
                ``DatasetCollection`` — use ``Map.animate`` or ``InteractiveMap.timecube`` for those).
            KeyError: when ``kdim`` is not a feature attribute.

        Examples:
            - Scrub a point series by its ``time`` attribute, colouring by ``pop`` (needs the ``web``
              extra, so this example is not executed here):
                ```python
                >>> import geopandas as gpd                          # doctest: +SKIP
                >>> from shapely.geometry import Point               # doctest: +SKIP
                >>> from digitalearth.web import WebMap              # doctest: +SKIP
                >>> gdf = gpd.GeoDataFrame(                          # doctest: +SKIP
                ...     {"time": [2000, 2010], "pop": [1.0, 2.0]},
                ...     geometry=[Point(0, 0), Point(1, 1)],
                ...     crs=4326,
                ... )
                >>> WebMap().timeslider(gdf, kdim="time", column="pop")._temporal_times()  # doctest: +SKIP
                [2000, 2010]

                ```
            - A raster is rejected up front, before any reprojection happens:
                ```python
                >>> from digitalearth.web import WebMap              # doctest: +SKIP
                >>> from pyramids.dataset import Dataset             # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")  # doctest: +SKIP
                >>> WebMap().timeslider(dem)                         # doctest: +SKIP
                Traceback (most recent call last):
                    ...
                TypeError: timeslider() renders a time-stepped vector layer; got a Dataset. ...

                ```

        See Also:
            digitalearth.static.maps.animation.AnimationMixin.animate: the matplotlib tier's raster
                time-stack animation, which takes a ``DatasetCollection``.
            digitalearth.interactive.temporal.TemporalMixin.timecube: the interactive tier's raster
                time cube, the HoloViz counterpart for stacks.
        """
        _require_layer_api()
        self._require_features(features, "timeslider")
        gdf = self._display_gdf(features)
        if kdim not in getattr(gdf, "columns", []):
            raise KeyError(f"time field {kdim!r} not found in the feature attributes")

        geom_types = set(gdf.geometry.geom_type.unique())
        is_polygon = geom_types <= {"Polygon", "MultiPolygon"}
        # big=False on every path: the slider filters a per-feature MapLibre layer, so the data must not
        # auto-route to a deck.gl layer (which has no per-feature layer id to filter) (M3).
        if is_polygon and column is not None:
            self.choropleth(gdf, column=column, scheme=scheme, k=k, cmap=cmap, opacity=opacity)
        elif is_polygon:
            self.polygons(gdf, opacity=opacity, big=False)
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

        Examples:
            - A fresh map has no slider, so there are no time steps:
                ```python
                >>> from digitalearth.web import WebMap
                >>> WebMap()._temporal_times()
                []

                ```
            - Once a slider is configured, the distinct steps come back in order:
                ```python
                >>> from digitalearth.web import WebMap
                >>> m = WebMap()
                >>> m._temporal = {"layer_id": "pop", "kdim": "year", "times": [2000, 2010, 2020]}
                >>> m._temporal_times()
                [2000, 2010, 2020]

                ```
            - The returned list is a snapshot, so editing it leaves the slider config intact:
                ```python
                >>> from digitalearth.web import WebMap
                >>> m = WebMap()
                >>> m._temporal = {"layer_id": "pop", "kdim": "year", "times": [2000, 2010]}
                >>> steps = m._temporal_times()
                >>> steps.append(2030)
                >>> m._temporal["times"]
                [2000, 2010]

                ```
        """
        return list(self._temporal["times"]) if self._temporal else []
