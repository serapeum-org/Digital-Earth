"""TemporalMixin — web-tier time-slider (DW.5, recipe W6).

``timeslider`` scrubs a time dimension with an ``ipywidgets`` slider, in either of the two forms recipe W6
specifies:

- **vector** — a layer whose features carry a time field. One layer is built per time step (each carrying
  that step's features), all sharing a single colour classification resolved over the whole series so the
  scale is stable across steps, and the slider swaps which step's layer is visible.
- **raster** — a pyramids ``DatasetCollection`` whose members are ordered time steps. Each member is added as
  its own image layer under one frozen colour range, and the slider swaps which layer is visible.

Both forms build one layer per step, so a page saved without the live slider carries a
``LayerSwitcherControl`` step picker over those layers (WB-4) rather than piling every step up at once.

The slider is wired at :meth:`render` time via :meth:`_wrap_temporal` (returning a slider + map composite);
:meth:`~digitalearth.web.base.WebMapBase.save` still serialises just the map. ipywidgets/maplibre are
imported lazily.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self

from loguru import logger

from digitalearth.base.clim import (
    DEFAULT_CLIM_SCAN_CAP,
    sample_evenly,
    stack_scale,
)
from digitalearth.base.crs import OffLimbError
from digitalearth.base.spec import DEFAULT_BAND
from digitalearth.web.base import _require_layer_api

#: Members scanned when computing a stack's shared colour range — `digitalearth.base.clim`'s cap, the one
#: number the static and interactive tiers read too, picked at the same evenly-spread positions. The scan
#: reprojects and reads each member, so an unbounded one makes `timeslider` O(stack) before it draws
#: anything. Pass an explicit `clim` to skip the scan entirely.
_CLIM_SCAN_CAP = DEFAULT_CLIM_SCAN_CAP

#: Total pixels above which an inlined stack is warned against. `field` warns per member, which never
#: fires for a stack of individually-modest members that is collectively enormous.
_LARGE_STACK_PIXELS = 8_000_000


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.web.base import WebMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


def _check_slider_labels(labels: Sequence | None, count: int) -> None:
    """Refuse a label sequence the slider could not map back to its frames.

    Args:
        labels: The caller's per-member labels, or `None` to label by integer index.
        count: How many members the collection holds.

    Raises:
        ValueError: when the sequence is the wrong length, holds an unhashable label, or repeats one.
    """
    if labels is None:
        return
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


class TemporalMixin(_MixinBase):
    """Time-slider builder for :class:`~digitalearth.web.map.WebMap`."""

    def _global_clim(self, collection: Any, band: int) -> tuple[float, float]:
        """Compute one ``(vmin, vmax)`` for the whole series so the colour range never jumps between frames.

        Note: this pass is **eager** — it reprojects and reads each scanned member once at ``timeslider``
        construction time (every member is then warped again when its image layer is built). At most
        :data:`_CLIM_SCAN_CAP` members are scanned, spread evenly across the whole series rather than taken
        from its head — the rule the static and interactive tiers follow too, so the same members are read,
        the same way, whichever tier draws it. The resulting numbers can still differ slightly: each tier
        warps to its own display CRS before measuring. Pass an explicit ``clim`` to skip the scan entirely.

        Args:
            collection: A pyramids ``DatasetCollection``.
            band: 1-based band read from each member.

        Returns:
            ``(vmin, vmax)`` finite colour limits taken from the sampled members, or ``(0.0, 1.0)`` when none
            holds a finite value. The pair comes from the frozen
            :class:`~digitalearth.base.spec.scale.Scale` the measurement builds, which is the same widening
            the per-member :meth:`~digitalearth.web.raster.RasterMixin.field` call applies to it downstream —
            so a stack of identical members now reports the range it is actually drawn with.
        """
        return stack_scale(
            self._to_display_source(member, band=band).z.values
            for member in sample_evenly(collection.datasets, cap=_CLIM_SCAN_CAP)
        ).as_limits()

    def timeslider(
        self,
        features: Any,
        *,
        kdim: str = "time",
        labels: Sequence | None = None,
        band: int = DEFAULT_BAND,
        column: str | None = None,
        scheme: Any | None = None,
        k: int = 5,
        cmap: str = "viridis",
        opacity: float = 0.85,
        clim: tuple[float, float] | None = None,
    ) -> Self:
        """Render a time-stepped layer with a slider over its time steps (recipe W6).

        Accepts either form the recipe specifies. A **vector** layer (a ``FeatureCollection`` /
        GeoDataFrame carrying ``kdim``) becomes one layer per time step, all sharing a single colour
        classification resolved over the whole series so the scale is stable across steps; polygons render as
        fills (a choropleth when ``column`` is given), points as circles. A **raster** stack (a pyramids
        ``DatasetCollection``) becomes one image layer per member under a single frozen colour range. Either
        way the slider swaps which step's layer is visible, and a page saved without the slider carries a
        step-picker control over those layers (WB-4).

        Args:
            features: A ``FeatureCollection`` / GeoDataFrame whose features carry ``kdim``, or a pyramids
                ``DatasetCollection`` whose members are ordered time steps.
            kdim: The time attribute to scrub (vector), or the slider's label (raster). A feature whose
                ``kdim`` is missing (null) carries no step value, so it sits in no step's layer, is not
                drawn — live or saved — and does not enter the colour classification. Its geometry can still
                widen the map's initial extent, since the frame is placed before the null rows are dropped;
                drop such rows before calling if that matters.
            labels: Raster only — per-member slider labels (e.g. datetimes) shown instead of the integer
                index; must match the member count and be unique.
            band: Raster only — the 1-based band drawn for every member.
            column: Vector only — value column to colour by (graduated choropleth/circles when set).
            scheme: Vector only — a cleopatra classification scheme for graduated colouring;
                ``None`` (the default) is a continuous ramp, matching every other builder.
            k: Vector only — number of classes for the graduated schemes.
            cmap: matplotlib colormap for the value colouring.
            opacity: Layer opacity in ``[0, 1]``.
            clim: Raster only — frozen ``(vmin, vmax)``; ``None`` computes one range for the whole series
                from at most :data:`_CLIM_SCAN_CAP` members sampled evenly across it.

        Returns:
            This map (chainable). The slider appears when the map is rendered/shown in a notebook.

        Raises:
            TypeError: when ``features`` is neither a vector layer nor a ``DatasetCollection``.
            KeyError: when ``kdim`` is not a feature attribute (vector).
            ValueError: when the series has no time step left once the missing values are dropped, or
                when ``labels`` does not match the member count or repeats a label (raster).

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
            digitalearth.web.raster.RasterMixin.field: draw a single raster with no time dimension.
            digitalearth.static.maps.animation.AnimationMixin.animate: the matplotlib tier's raster
                time-stack animation.
            digitalearth.interactive.temporal.TemporalMixin.timecube: the interactive tier's raster
                time cube.
        """
        import pandas as pd

        _require_layer_api()
        if hasattr(
            features, "datasets"
        ):  # a pyramids DatasetCollection — the raster stack path
            return self._timeslider_stack(
                features,
                kdim=kdim,
                labels=labels,
                band=band,
                cmap=cmap,
                opacity=opacity,
                clim=clim,
            )

        gdf = self._display_gdf(features, method="timeslider")
        if kdim not in getattr(gdf, "columns", []):
            raise KeyError(f"time field {kdim!r} not found in the feature attributes")

        # Features with no time value cannot sit at any step, and a None among the values would make the
        # sort raise once `_display_gdf` has encoded a NaT to None ("'<' not supported between instances
        # of 'NoneType' and 'str'"). Drop them; the empty check below still catches a series with none.
        times = sorted(
            value for value in gdf[kdim].unique().tolist() if not pd.isna(value)
        )
        if not times:
            raise ValueError(
                f"timeslider() needs at least one time step, but no feature carries a {kdim!r} value"
            )

        return self._timeslider_vector(
            gdf,
            kdim=kdim,
            times=times,
            column=column,
            scheme=scheme,
            k=k,
            cmap=cmap,
            opacity=opacity,
        )

    def _timeslider_stack(
        self,
        collection: Any,
        *,
        kdim: str,
        labels: Sequence | None,
        band: int,
        cmap: str,
        opacity: float,
        clim: tuple[float, float] | None,
    ) -> Self:
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
            clim: Frozen ``(vmin, vmax)``; ``None`` computes one range for the whole series from at most
                :data:`_CLIM_SCAN_CAP` members sampled evenly across it.

        Returns:
            This map (chainable).

        Raises:
            ValueError: when the collection is empty, or ``labels`` does not match the member count or
                repeats a label (a duplicate collapses the slider and makes a frame unreachable).
        """
        members = collection.datasets
        count = len(members)
        if count == 0:
            raise ValueError(
                "timeslider() needs at least one time step, but the collection is empty"
            )
        _check_slider_labels(labels, count)
        try:
            self._check_stack_is_drawable(members, band)
            vmin, vmax = (
                clim if clim is not None else self._global_clim(collection, band)
            )
        except OffLimbError as error:
            # A series missing frames is not a series, so the whole slider goes rather than part of it.
            # Both reads happen before any layer is added, so skipping here leaves the map as it was.
            self._skipped("timeslider", str(error), error)
            return self
        step_names = [
            str(value) for value in (labels if labels is not None else range(count))
        ]
        # Snapshotted before the first frame is built, so the unwind below can put back the derived state
        # the discarded frames moved. `remove_layer` takes the layers off the map but leaves the running
        # extent widened — it only clears it when the *last* layer goes, so a stack added on top of an
        # existing layer kept the abandoned frames' corners — and leaves `last_units` describing a frame
        # that is no longer drawn. A later `set_bounds()` then framed on data the map does not show (L12).
        bounds_before = (
            list(self._data_bounds) if self._data_bounds is not None else None
        )
        units_before = self.last_units
        layer_ids: list[str] = []
        for index, member in enumerate(members):
            # Only the first frame is built visible. The slider toggles from there, and a page saved
            # without a slider then shows one frame rather than the whole stack piled up.
            previous = self._last_layer_id
            self.field(
                member,
                band=band,
                cmap=cmap,
                opacity=opacity,
                vmin=vmin,
                vmax=vmax,
                visible=index == 0,
                # The switcher captions each row with the layer id, so the step's label has to *be* it.
                name=step_names[index],
            )
            if self._last_layer_id == previous:
                # `field` skipped this member (it could not be placed). Keeping the frames either
                # side would leave a slider with a hole in it, so the steps already built are unwound.
                for built in layer_ids:
                    self.remove_layer(built)
                self._data_bounds = bounds_before
                self.last_units = units_before
                self._skipped(
                    "timeslider",
                    f"step {step_names[index]!r} could not be placed, so the series was abandoned",
                )
                return self
            layer_ids.append(self._last_layer_id)

        steps = list(labels) if labels is not None else list(range(count))
        self._temporal = {
            "mode": "raster",
            "layer_ids": layer_ids,
            "kdim": kdim,
            "times": steps,
        }
        self._record_furniture(
            "time_slider",
            mode="raster",
            layers=tuple(layer_ids),
            kdim=kdim,
            frames=tuple(str(step) for step in steps),
        )
        return self

    def _timeslider_vector(
        self,
        gdf: Any,
        *,
        kdim: str,
        times: list,
        column: str | None,
        scheme: Any | None,
        k: int,
        cmap: str,
        opacity: float,
    ) -> Self:
        """Build the vector half of :meth:`timeslider`: one layer per time step, swapped by the slider.

        Mirrors :meth:`_timeslider_stack`. The colour classification is resolved **once over the whole
        series**, so the scale is identical on every step — the acceptance criterion a time series must
        meet — and that one frozen paint is applied to each step's layer, which carries only that step's
        features. A MapLibre colour expression is a value-to-colour function, so a step holding only part of
        the range still paints in the whole series' classes. Only the first step is built visible: the slider
        toggles from there, and a page saved without a slider then shows one step rather than the whole
        series piled up.

        Because every step is its own registered layer, a saved page carries a ``LayerSwitcherControl`` step
        picker (WB-4) exactly as the raster stack does — where the old single-filtered-layer form left a
        shared page showing every feature at once, the slider having gone with the live kernel.

        Args:
            gdf: The display-CRS GeoDataFrame, already placed, whose ``kdim`` column carries the time steps.
            kdim: The time attribute scrubbed; each step draws the rows whose ``kdim`` equals its value.
            times: The distinct, sorted, non-null step values — the slider's stops, in order.
            column: Value column to colour by (a graduated/continuous choropleth or circles), or ``None`` for
                a flat colour.
            scheme: A cleopatra classification scheme for graduated colouring, or ``None`` for a continuous
                ramp. Classified once over the whole series.
            k: Number of classes for the graduated schemes.
            cmap: matplotlib colormap for the value colouring.
            opacity: Layer opacity in ``[0, 1]``.

        Returns:
            This map (chainable).
        """
        from digitalearth.web.vector import POINT_SIZE, VECTOR_COLOR

        _, layer_types = _require_layer_api()
        geom_types = set(gdf.geometry.geom_type.unique())
        is_polygon = geom_types <= {"Polygon", "MultiPolygon"}
        # Classify over the DRAWN subset, not the whole frame: a row with a null `kdim` lands in no step
        # (`gdf[gdf[kdim] == step]` never matches it), so it is never drawn — and it must not shift the
        # breaks of the rows that are. `drawn` is the union of the per-step subsets; the paint is resolved
        # once over it so every step colours by the same classification. `_color_expr` records
        # `last_breaks`/`last_legend` here too, describing the drawn series the colour key belongs to.
        drawn = gdf[gdf[kdim].notna()]
        color_encoding: Any = None
        if is_polygon and column is not None:
            fill, color_encoding = self._color_expr(
                self._require_column(drawn, column), column, scheme, k, cmap
            )
            prefix, layer_type, kind = "fill", layer_types.FILL, "choropleth"
            paint = {
                "fill-color": fill,
                "fill-opacity": float(opacity),
                "fill-outline-color": "#ffffff",
            }
        elif is_polygon:
            prefix, layer_type, kind = "fill", layer_types.FILL, "polygons"
            paint = self._fill_paint(opacity, "#ffffff", VECTOR_COLOR)
        else:
            prefix, layer_type, kind = "circle", layer_types.CIRCLE, "points"
            if column is not None:
                circle_color, color_encoding = self._color_expr(
                    self._require_column(drawn, column), column, scheme, k, cmap
                )
            else:
                circle_color = VECTOR_COLOR
            paint = {
                "circle-radius": POINT_SIZE,
                "circle-opacity": float(opacity),
                "circle-color": circle_color,
            }

        layer_ids: list[str] = []
        for index, step in enumerate(times):
            step_gdf = gdf[gdf[kdim] == step]
            # dict(paint): each layer takes its own copy of the one frozen paint, so a later per-layer
            # restyle touches one step rather than aliasing them all. The colour encoding is filed on every
            # step, so the key is present whichever step is the visible one.
            self._vector_layer(
                step_gdf,
                prefix,
                layer_type,
                dict(paint),
                kind=kind,
                name=str(step),
                visible=index == 0,
                source=step_gdf,
                color_encoding=color_encoding,
            )
            layer_ids.append(self._last_layer_id)

        self._temporal = {
            "mode": "vector",
            "layer_ids": layer_ids,
            "kdim": kdim,
            "times": list(times),
        }
        # Furniture mirrors the raster half (`layers=`, not a single `layer=`): the slider steps through the
        # per-step layers, which is what a renderer draws the control from (#292).
        self._record_furniture(
            "time_slider",
            mode="vector",
            layers=tuple(layer_ids),
            kdim=kdim,
            frames=tuple(str(step) for step in times),
        )
        return self

    def _check_stack_is_drawable(self, members: Sequence, band: int) -> None:
        """Fail before any layer is registered if a member cannot be drawn, and warn on a huge page.

        ``field`` raises for a band with no finite values — routine in EO, where a whole time step can
        be cloud-masked away. Discovering that midway through the loop left the map holding layers for the
        members already added, so a caught error left a half-built stack behind. Checking up front keeps
        ``timeslider`` all-or-nothing.

        The size warning is the stack-level counterpart of ``field``'s per-member one: every member is
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
                # Cast before filling: a nodata sentinel is usually an integer one, and `filled(nan)` on an
                # int band raises rather than widening it. Same defect `measure_clim` carried.
                values = np.ma.filled(values.astype("float64"), np.nan)
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

    def _temporal_switcher(self) -> dict | None:
        """Return the step picker a saved page needs, or ``None`` when this map is not a stepped series.

        ``render`` wraps the map in an ``ipywidgets`` slider, which exists only in a live kernel:
        ``to_html`` serialises the map alone, so a shared page would show one frozen frame and no way to
        move. Both series forms — the raster stack and the vector series (WB-4) — now build one layer per
        step, so a switcher over those layers makes every step reachable in the saved page: a step picker
        rather than a scrubber, but the difference between a usable artifact and a screenshot. The steps
        carry their labels as their layer ids, because the switcher captions each row with the id.

        Returns:
            The switcher request, or ``None`` when fewer than two step layers were built.
        """
        config = self._temporal
        layer_ids = list((config or {}).get("layer_ids") or [])
        # One step is not a series; a picker over it would be noise.
        if len(layer_ids) < 2:
            return None
        return {
            "layer_ids": layer_ids,
            "theme": "default",
            "position": "top-right",
            "minimum": 2,
        }

    def _wrap_temporal(self, widget: Any) -> Any:
        """Wrap the map ``widget`` in a slider composite that reveals one time step at a time.

        The slider toggles which per-step layer is visible — the same mechanism for the vector series and
        the raster stack, since both build one layer per step.

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
        """Return the callable that reveals one time step on ``widget`` by toggling layer visibility.

        Both series forms build one layer per step, so stepping is the same for each: hide the step that
        was showing and show the selected one. Only two layers change per move, so a series of N steps does
        not send O(N) widget messages on every slider move.

        Args:
            widget: The built MapLibre ``MapWidget``.
            config: The ``_temporal`` config recorded by :meth:`timeslider` / :meth:`_timeslider_vector` /
                :meth:`_timeslider_stack`.

        Returns:
            A one-argument callable taking the slider's value and showing that step's layer while hiding the
            one that was visible.
        """
        layer_ids = config["layer_ids"]
        index_of = {step: index for index, step in enumerate(config["times"])}
        # the step currently visible; layers are built with only the first shown
        showing = [0]

        def show(value: Any) -> None:
            active = index_of[value]
            if active == showing[0]:
                return
            widget.set_visibility(layer_ids[showing[0]], False)
            widget.set_visibility(layer_ids[active], True)
            showing[0] = active

        return show

    def _temporal_times(self) -> list[Any]:
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
                >>> m._temporal = {"layer_ids": ["pop-0"], "kdim": "year", "times": [2000, 2010, 2020]}
                >>> m._temporal_times()
                [2000, 2010, 2020]

                ```
            - The returned list is a snapshot, so editing it leaves the slider config intact:
                ```python
                >>> from digitalearth.web import WebMap
                >>> m = WebMap()
                >>> m._temporal = {"layer_ids": ["pop-0"], "kdim": "year", "times": [2000, 2010]}
                >>> steps = m._temporal_times()
                >>> steps.append(2030)
                >>> m._temporal["times"]
                [2000, 2010]

                ```
        """
        return list(self._temporal["times"]) if self._temporal else []
