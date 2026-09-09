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

from loguru import logger

from digitalearth.web.base import _require_layer_api

#: Members scanned when computing a stack's shared colour range. Mirrors the static tier's cap: the scan
#: reprojects and reads each member, so an unbounded one makes `timeslider` O(stack) before it draws
#: anything. Pass an explicit `clim` to skip the scan entirely.
_CLIM_SCAN_CAP = 50

#: Total pixels above which an inlined stack is warned against. `add_raster` warns per member, which never
#: fires for a stack of individually-modest members that is collectively enormous.
_LARGE_STACK_PIXELS = 8_000_000


class TemporalMixin:
    """Time-slider builder for :class:`~digitalearth.web.map.WebMap`."""

    def _global_clim(self, collection: Any, band: int) -> Tuple[float, float]:
        """Compute one ``(vmin, vmax)`` over every member so the colour range never jumps between frames.

        Note: this pass is **eager** — it reprojects and reads each scanned member once at ``timeslider``
        construction time (every member is then warped again when its image layer is built). At most
        :data:`_CLIM_SCAN_CAP` members are scanned, mirroring the static tier's cap; pass an explicit
        ``clim`` to skip the scan entirely.

        Args:
            collection: A pyramids ``DatasetCollection``.
            band: 1-based band read from each member.

        Returns:
            ``(vmin, vmax)`` finite colour limits across the whole stack, or ``(0.0, 1.0)`` when no member
            holds a finite value.
        """
        import numpy as np

        from digitalearth.base.arrays import finite

        lows: List[float] = []
        highs: List[float] = []
        for member in collection.datasets[:_CLIM_SCAN_CAP]:
            values = self._to_display_source(member, band=band).z.values
            # `finite` goes through np.asarray, which drops a mask and would let the fill value (-9999)
            # through as a real number. pyramids' extractor already NaN-fills nodata, so this only bites
            # a caller handing us a masked array directly — fill it first and the two agree.
            if np.ma.isMaskedArray(values):
                values = values.filled(np.nan)
            values = finite(values)
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

        gdf = self._display_gdf(features, method="timeslider")
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
        and last frame — the acceptance criterion for a stack. All members are registered up front, but only
        the first is built **visible**: the slider toggles from that state, and :meth:`save`, which
        serialises the map without a slider, then writes a page showing one frame instead of the whole
        stack piled up with only the last member on top.

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
            try:
                unique = len(set(labels))
            except TypeError as err:  # an unhashable label (a list, a dict, …)
                raise ValueError(
                    "timeslider labels must be hashable so the slider can map a value back to its frame; "
                    f"got {[type(label).__name__ for label in labels]}"
                ) from err
            if unique != count:
                raise ValueError(
                    "timeslider labels must be unique — duplicate labels collapse the slider and make "
                    "the matching frames unreachable"
                )
        self._check_stack_is_drawable(members, band)

        vmin, vmax = clim if clim is not None else self._global_clim(collection, band)
        layer_ids: List[str] = []
        for index, member in enumerate(members):
            # Only the first frame is built visible. The slider toggles from there, and a page saved
            # without a slider then shows one frame rather than the whole stack piled up.
            self.add_raster(
                member,
                band=band,
                cmap=cmap,
                opacity=opacity,
                vmin=vmin,
                vmax=vmax,
                visible=index == 0,
            )
            layer_ids.append(self._last_layer_id)

        self._temporal = {
            "mode": "raster",
            "layer_ids": layer_ids,
            "kdim": kdim,
            "times": list(labels) if labels is not None else list(range(count)),
        }
        return self

    def _check_stack_is_drawable(self, members: Sequence, band: int) -> None:
        """Fail before any layer is registered if a member cannot be drawn, and warn on a huge page.

        ``add_raster`` raises for a band with no finite values — routine in EO, where a whole time step can
        be cloud-masked away. Discovering that midway through the loop left the map holding layers for the
        members already added, so a caught error left a half-built stack behind. Checking up front keeps
        ``timeslider`` all-or-nothing.

        The size warning is the stack-level counterpart of ``add_raster``'s per-member one: every member is
        inlined as a base64 PNG, so a stack of individually-modest members can still produce an enormous
        page without any single member tripping the per-member threshold.

        Args:
            members: The collection's members, in slider order.
            band: 1-based band each member will be drawn from.

        Raises:
            ValueError: when a member holds no finite values in ``band``.
        """
        import numpy as np

        from digitalearth.base.arrays import finite

        total_pixels = 0
        for index, member in enumerate(members):
            values = self._to_display_source(member, band=band).z.values
            if np.ma.isMaskedArray(values):
                values = values.filled(np.nan)
            total_pixels += int(getattr(values, "size", 0))
            if not finite(values).size:
                raise ValueError(
                    f"timeslider() cannot draw member {index} of {len(members)}: band {band} holds no "
                    f"finite values, so it has no colour range. Drop the empty step from the collection "
                    f"or pass an explicit clim."
                )
        if total_pixels > _LARGE_STACK_PIXELS:
            logger.warning(
                "timeslider: inlining a {}-member stack of {} pixels as data-URI images bloats the page; "
                "for a large stack serve COG/XYZ tiles from pyramids instead",
                len(members),
                total_pixels,
            )

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
            showing = [0]  # the frame currently visible; layers are built with only the first shown

            def show_raster(value: Any) -> None:
                active = index_of[value]
                if active == showing[0]:
                    return
                # Only two layers change per step, so touching all N would send O(stack) widget messages
                # for every slider move.
                widget.set_visibility(layer_ids[showing[0]], False)
                widget.set_visibility(layer_ids[active], True)
                showing[0] = active

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
