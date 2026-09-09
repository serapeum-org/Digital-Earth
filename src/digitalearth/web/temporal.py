"""TemporalMixin — web-tier time-slider (DW.5, recipe W6).

``timeslider`` scrubs a time dimension with an ``ipywidgets`` slider, in either of the two forms recipe W6
specifies:

- **vector** — a layer whose features carry a time field. Every feature is drawn once (so the colour
  classification spans the whole series and the scale is stable across frames) and the slider sets a MapLibre
  filter so only the active time step is shown.
- **raster** — a pyramids ``DatasetCollection`` whose members are ordered time steps. Each member is added as
  its own image layer under one frozen colour range, and the slider swaps which layer is visible.

The slider is wired at :meth:`render` time via :meth:`_wrap_temporal` (returning a slider + map composite);
:meth:`~digitalearth.web.base.WebMapBase.save` still serialises just the map. ipywidgets/maplibre are
imported lazily.
"""

from typing import Any, List, Optional, Sequence, Tuple

from digitalearth.web.base import _require_layer_api


class TemporalMixin:
    """Time-slider builder for :class:`~digitalearth.web.map.WebMap`."""

    @staticmethod
    def _require_vector(features: Any, method: str) -> None:
        """Raise ``TypeError`` unless ``features`` is a vector layer.

        Reached only after the ``DatasetCollection`` branch, so by here the input is neither a vector layer
        nor a raster stack. Without this check a pyramids raster would die in the attribute-membership test
        below on an unrelated error: a ``Dataset`` exposes ``columns`` as an ``int`` (the grid width in
        cells), so ``kdim not in ...`` raises ``TypeError: argument of type 'int' is not iterable``. Called
        *before* :meth:`_display_gdf` so a raster in another CRS is not warped on its way to being rejected.

        Args:
            features: The caller's input, expected to be a pyramids ``FeatureCollection`` / GeoDataFrame.
            method: The calling builder name (quoted in the error).

        Raises:
            TypeError: when ``features`` exposes no ``geometry`` — a single raster, a bare array, anything
                non-vector — naming the accepted forms and the single-raster alternative. A table that has
                columns but no *active* geometry (``set_geometry`` never called) gets its own message
                naming ``set_geometry``, rather than being misreported as a raster.

        Examples:
            - A vector-like input passes the guard silently (the check returns nothing):
                ```python
                >>> from digitalearth.web import WebMap
                >>> layer = type("Layer", (), {"geometry": ()})()
                >>> print(WebMap._require_vector(layer, "timeslider"))
                None

                ```
            - Anything else is turned away, and the message names the type that arrived:
                ```python
                >>> from digitalearth.web import WebMap
                >>> try:
                ...     WebMap._require_vector(42, "timeslider")
                ... except TypeError as err:
                ...     print(str(err).split(",")[0])
                timeslider() needs a vector layer with a time attribute

                ```
            - The message points at the single-raster builder in this same tier:
                ```python
                >>> from digitalearth.web import WebMap
                >>> try:
                ...     WebMap._require_vector(42, "timeslider")
                ... except TypeError as err:
                ...     print("add_raster" in str(err))
                True

                ```

        See Also:
            digitalearth.web.bigdata.BigDataMixin._require_points: the sibling geometry-kind guard this
                mirrors, which rejects non-point input for the heatmap/cluster builders.
        """
        if hasattr(features, "geometry"):
            return
        # A raster reports `columns` as an int (the grid width); a table reports an Index of names. Telling
        # the two apart keeps a GeoDataFrame whose geometry was never activated from being called a raster.
        columns = getattr(features, "columns", None)
        if columns is not None and not isinstance(columns, int):
            raise TypeError(
                f"{method}() needs a layer with an active geometry column; got a "
                f"{type(features).__name__} whose columns are {list(columns)}. Call set_geometry(...) on "
                f"it first."
            )
        raise TypeError(
            f"{method}() needs a vector layer with a time attribute, or a DatasetCollection for a "
            f"raster time stack; got {type(features).__name__}. For a single raster (no time "
            f"dimension) use add_raster()."
        )

    def _global_clim(self, collection: Any, band: int) -> Tuple[float, float]:
        """Compute one ``(vmin, vmax)`` over every member so the colour range never jumps between frames.

        Note: this pass is **eager** — it reprojects and reads every member once at ``timeslider``
        construction time (each member is then warped again when its image layer is built). For a very
        large stack, pass an explicit ``clim`` to skip this whole-stack scan.

        Args:
            collection: A pyramids ``DatasetCollection``.
            band: 1-based band read from each member.

        Returns:
            ``(vmin, vmax)`` finite colour limits across the whole stack, or ``(0.0, 1.0)`` when no member
            holds a finite value.
        """
        from digitalearth.base.arrays import finite

        lows: List[float] = []
        highs: List[float] = []
        for member in collection.datasets:
            values = finite(self._to_display_source(member, band=band).z.values)
            if values.size:
                lows.append(float(values.min()))
                highs.append(float(values.max()))
        return (min(lows), max(highs)) if lows else (0.0, 1.0)

    def timeslider(
        self,
        features: Any,
        *,
        kdim: str = "time",
        labels: Optional[Sequence] = None,
        band: int = 1,
        column: Optional[str] = None,
        scheme: Optional[Any] = "quantiles",
        k: int = 5,
        cmap: str = "viridis",
        opacity: float = 0.85,
        clim: Optional[Tuple[float, float]] = None,
    ) -> "TemporalMixin":
        """Render a time-stepped layer with a slider over its time steps (recipe W6).

        Accepts either form the recipe specifies. A **vector** layer (a ``FeatureCollection`` /
        GeoDataFrame carrying ``kdim``) is drawn once and filtered per step, so the classification spans the
        whole series; polygons render as fills (a choropleth when ``column`` is given), points as circles. A
        **raster** stack (a pyramids ``DatasetCollection``) becomes one image layer per member under a single
        frozen colour range, and the slider swaps which member is visible.

        Args:
            features: A ``FeatureCollection`` / GeoDataFrame whose features carry ``kdim``, or a pyramids
                ``DatasetCollection`` whose members are ordered time steps.
            kdim: The time attribute to scrub (vector), or the slider's label (raster).
            labels: Raster only — per-member slider labels (e.g. datetimes) shown instead of the integer
                index; must match the member count and be unique.
            band: Raster only — the 1-based band drawn for every member.
            column: Vector only — value column to colour by (graduated choropleth/circles when set).
            scheme: Vector only — a cleopatra classification scheme for graduated colouring.
            k: Vector only — number of classes for the graduated schemes.
            cmap: matplotlib colormap for the value colouring.
            opacity: Layer opacity in ``[0, 1]``.
            clim: Raster only — frozen ``(vmin, vmax)``; ``None`` computes one range over the whole stack.

        Returns:
            This map (chainable). The slider appears when the map is rendered/shown in a notebook.

        Raises:
            TypeError: when ``features`` is neither a vector layer nor a ``DatasetCollection``.
            KeyError: when ``kdim`` is not a feature attribute (vector).
            ValueError: when the series has no time steps, or when ``labels`` does not match the member
                count or repeats a label (raster).

        Examples:
            - Scrub a point series by its ``time`` attribute (needs the ``web`` extra, so this example is
              not executed here):
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
            - Scrub a raster stack, labelling the slider with real dates:
                ```python
                >>> from pyramids.dataset.collection import DatasetCollection  # doctest: +SKIP
                >>> from digitalearth.web import WebMap                        # doctest: +SKIP
                >>> dc = DatasetCollection.from_files(["jan.tif", "feb.tif"])  # doctest: +SKIP
                >>> m = WebMap().timeslider(dc, labels=["Jan", "Feb"])         # doctest: +SKIP
                >>> m._temporal_times()                                        # doctest: +SKIP
                ['Jan', 'Feb']

                ```

        See Also:
            digitalearth.web.raster.RasterMixin.add_raster: draw a single raster with no time dimension.
            digitalearth.static.maps.animation.AnimationMixin.animate: the matplotlib tier's raster
                time-stack animation.
            digitalearth.interactive.temporal.TemporalMixin.timecube: the interactive tier's raster
                time cube.
        """
        _require_layer_api()
        if hasattr(features, "datasets"):  # a pyramids DatasetCollection — the raster stack path
            return self._timeslider_stack(
                features, kdim=kdim, labels=labels, band=band, cmap=cmap, opacity=opacity, clim=clim
            )

        self._require_vector(features, "timeslider")
        gdf = self._display_gdf(features)
        if kdim not in getattr(gdf, "columns", []):
            raise KeyError(f"time field {kdim!r} not found in the feature attributes")

        times = sorted(gdf[kdim].unique().tolist())
        if not times:
            raise ValueError(
                f"timeslider() needs at least one time step, but no feature carries a {kdim!r} value"
            )

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

        self._temporal = {
            "mode": "vector",
            "layer_id": self._last_layer_id,
            "kdim": kdim,
            "times": times,
        }
        return self

    def _timeslider_stack(
        self,
        collection: Any,
        *,
        kdim: str,
        labels: Optional[Sequence],
        band: int,
        cmap: str,
        opacity: float,
        clim: Optional[Tuple[float, float]],
    ) -> "TemporalMixin":
        """Build the raster half of :meth:`timeslider`: one image layer per member, swapped by the slider.

        Every member is drawn with the same ``(vmin, vmax)`` so the colour scale is identical on the first
        and last frame — the acceptance criterion for a stack. The layers are all registered up front;
        :meth:`_wrap_temporal` hides all but the active one at render time.

        Args:
            collection: A pyramids ``DatasetCollection`` whose members are ordered time steps.
            kdim: The slider's label.
            labels: Optional per-member labels shown instead of the integer index.
            band: 1-based band drawn for every member.
            cmap: matplotlib colormap applied to every member.
            opacity: Image-layer opacity in ``[0, 1]``.
            clim: Frozen ``(vmin, vmax)``; ``None`` computes one range over the whole stack.

        Returns:
            This map (chainable).

        Raises:
            ValueError: when the collection is empty, or ``labels`` does not match the member count or
                repeats a label (a duplicate collapses the slider and makes a frame unreachable).
        """
        members = collection.datasets
        count = len(members)
        if count == 0:
            raise ValueError("timeslider() needs at least one time step, but the collection is empty")
        if labels is not None:
            if len(labels) != count:
                raise ValueError(
                    f"labels has {len(labels)} entries but the collection has {count} members"
                )
            if len(set(labels)) != count:
                raise ValueError(
                    "timeslider labels must be unique — duplicate labels collapse the slider and make "
                    "the matching frames unreachable"
                )

        vmin, vmax = clim if clim is not None else self._global_clim(collection, band)
        layer_ids: List[str] = []
        for member in members:
            self.add_raster(member, band=band, cmap=cmap, opacity=opacity, vmin=vmin, vmax=vmax)
            layer_ids.append(self._last_layer_id)

        self._temporal = {
            "mode": "raster",
            "layer_id": layer_ids[0],
            "layer_ids": layer_ids,
            "kdim": kdim,
            "times": list(labels) if labels is not None else list(range(count)),
        }
        return self

    def _wrap_temporal(self, widget: Any) -> Any:
        """Wrap the map ``widget`` in a slider composite that reveals one time step at a time.

        The slider drives whichever mechanism the active mode needs: a MapLibre filter on the single vector
        layer, or the visibility of one image layer out of the stack.

        Args:
            widget: The built MapLibre ``MapWidget``.

        Returns:
            An ``ipywidgets.VBox`` of ``[slider, widget]``. The first time step is shown initially.
        """
        import ipywidgets

        config = self._temporal or {}
        kdim, times = config["kdim"], config["times"]
        show = self._step_shower(widget, config)

        slider = ipywidgets.SelectionSlider(
            options=[(str(t), t) for t in times],
            description=kdim,
            continuous_update=False,
        )

        def _on_change(change: Any) -> None:
            show(change["new"])

        slider.observe(_on_change, names="value")
        if times:
            show(times[0])
        return ipywidgets.VBox([slider, widget])

    @staticmethod
    def _step_shower(widget: Any, config: dict) -> Any:
        """Return the callable that reveals one time step on ``widget`` for the configured mode.

        Args:
            widget: The built MapLibre ``MapWidget``.
            config: The ``_temporal`` config recorded by :meth:`timeslider`.

        Returns:
            A one-argument callable taking the slider's value and updating the map: a MapLibre filter for
            the vector mode, per-layer visibility for the raster mode.
        """
        if config.get("mode") == "raster":
            layer_ids = config["layer_ids"]
            index_of = {step: index for index, step in enumerate(config["times"])}

            def show_raster(value: Any) -> None:
                active = index_of[value]
                for index, layer_id in enumerate(layer_ids):
                    widget.set_visibility(layer_id, index == active)

            return show_raster

        layer_id, kdim = config["layer_id"], config["kdim"]

        def show_vector(value: Any) -> None:
            widget.set_filter(layer_id, ["==", ["get", kdim], value])

        return show_vector

    def _temporal_times(self) -> List[Any]:
        """Return the distinct time steps of the active time-slider (empty when none is set).

        Returns:
            The sorted distinct ``kdim`` values (vector) or the member labels/indices (raster), or ``[]``
            when :meth:`timeslider` has not been called.

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
