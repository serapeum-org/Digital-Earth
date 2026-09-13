"""DashboardMixin — Panel dashboards for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``dashboard`` / ``serve`` / ``save_app`` (DI.4) and a ``cross_tier_pane`` helper that embeds an M1
``Scene3D`` beside the map (DI.4c); the layer-control / attribute-table / URL-share growth is DI.13.

Panel wraps the rendered HoloViews map plus reactive widgets into an app that serves live
(``panel serve``) and exports to a single self-contained HTML file. The reactive wiring uses ``pn.bind``:
each widget feeds a function that re-renders the map with updated style opts, so moving the cmap selector
or alpha slider updates the displayed map. ``save_app`` bakes the widget states into static HTML via
Panel's ``embed`` (subject to its combinatorial limits — documented on the method).

WASM/Pyodide export is **out** for a pyramids-backed app: GDAL/pyramids cannot run in the browser, so a
live app must be *served*, not converted. ``save_app`` is the offline path (pre-rendered states only).
"""

from functools import reduce
from importlib.util import find_spec
from operator import mul as _mul
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from digitalearth.interactive.base import _require_holoviz
from digitalearth.interactive.decoration import DEFAULT_BASEMAP_PROVIDER

#: Basemap providers offered by the basemap switchers — the **one** list behind both the dashboard's
#: ``"basemap"`` widget and the layer-control switch, so the two cannot offer different providers.
_BASEMAP_CHOICES = list(
    dict.fromkeys(
        [DEFAULT_BASEMAP_PROVIDER, "CartoLight", "CartoDark", "OSM", "EsriImagery"]
    )
)

#: Element type names that are a tile basemap — replaced (not stacked under) when a basemap widget picks
#: a provider, so a map that already carries tiles still responds to the switcher.
_TILE_TYPES = ("WMTS", "Tiles")

#: Element type names the dashboard's ``cmap``/``alpha`` overrides apply to (the colour-mapped raster
#: elements). Their recorded style is what an override is merged over.
_COLOR_MAPPED_TYPES = ("Image", "QuadMesh")

#: Style keys a widget may override on a colour-mapped layer. Anything the builders recorded under these
#: keys is read back and merged *under* the widget value, so an override never silently drops the rest of
#: the styling a builder applied.
_OVERRIDABLE_STYLE = ("cmap", "clim", "alpha")

#: Built-in colormaps offered by the dashboard cmap selector.
_CMAP_CHOICES = [
    "viridis",
    "magma",
    "inferno",
    "plasma",
    "cividis",
    "terrain",
    "Blues",
    "RdBu_r",
]


def _require_panel() -> Any:
    """Import and return the ``panel`` module, raising an actionable error when absent.

    Returns:
        The imported ``panel`` module.

    Raises:
        ImportError: when the ``interactive`` extra (which provides panel) is not installed.
    """
    _require_holoviz()  # panel ships with the same extra; reuse its actionable message first
    import panel as pn

    return pn


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.interactive.base import InteractiveMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


class DashboardMixin(_MixinBase):
    """Panel dashboard builders (DI.4): wrap the map + reactive widgets into a servable/exportable app.

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

    def dashboard(
        self,
        *,
        widgets: Sequence[str] = ("cmap", "alpha"),
        sidebar: bool = True,
        title: str = "",
    ) -> Any:
        """Wrap the map and reactive widgets into a Panel layout.

        The returned object is a ``panel.viewable.Viewable``: a sidebar (or inline row) of widgets
        bound to the map's style, beside the live map. Supported widgets: ``"cmap"`` (colormap
        selector), ``"alpha"`` (opacity slider), ``"basemap"`` (tile-provider selector, which swaps the
        tile layer under the map and is available on a Web-Mercator map only).

        Args:
            widgets: Which widgets to expose, in order.
            sidebar: Lay widgets out in a left sidebar (``True``) or a top row (``False``).
            title: Dashboard title (falls back to the map's ``title``).

        Returns:
            A ``panel.viewable.Viewable`` hosting the map + widgets.

        Raises:
            ValueError: for an unknown widget name, or when ``"basemap"`` is requested on a map whose
                display CRS is not EPSG:3857 (Bokeh renders tiles in Web Mercator only).

        Examples:
            - Build a dashboard with colormap + opacity controls:
                ```python
                >>> from pyramids.dataset import Dataset                       # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")       # doctest: +SKIP
                >>> app = InteractiveMap().image(dem).dashboard(widgets=("cmap", "alpha"))  # doctest: +SKIP
                >>> sorted({w.name for w in app.select()} & {"Colormap", "Opacity"})  # doctest: +SKIP
                ['Colormap', 'Opacity']

                ```
        """
        pn = _require_panel()
        controls, bindings = self._build_widgets(pn, widgets)

        def _view(**values: Any) -> Any:
            return self._render_with_overrides(values)

        view = pn.bind(_view, **bindings)
        panel_map = pn.panel(view)
        heading = (
            pn.pane.Markdown(f"## {title or self.title}")
            if (title or self.title)
            else None
        )
        controls_box = pn.Column(*controls) if controls else None
        if sidebar:
            body = (
                pn.Row(controls_box, panel_map) if controls_box else pn.Row(panel_map)
            )
        else:
            body = (
                pn.Column(controls_box, panel_map)
                if controls_box
                else pn.Column(panel_map)
            )
        return pn.Column(heading, body) if heading else body

    def _build_widgets(self, pn: Any, widgets: Sequence[str]) -> tuple:
        """Construct the requested widgets and the ``pn.bind`` keyword map.

        Args:
            pn: The imported panel module.
            widgets: Widget names to build.

        Returns:
            ``(controls, bindings)`` — the widget objects (for layout) and the ``name -> widget``
            map passed to ``pn.bind``.

        Raises:
            ValueError: for an unknown widget name, or when ``"basemap"`` is requested on a
                non-Web-Mercator map (Bokeh renders tiles in EPSG:3857 only).
        """
        controls, bindings = [], {}
        for name in widgets:
            if name == "cmap":
                widget = pn.widgets.Select(label="Colormap", options=_CMAP_CHOICES)
            elif name == "alpha":
                widget = pn.widgets.FloatSlider(
                    label="Opacity", start=0.0, end=1.0, value=1.0
                )
            elif name == "basemap":
                # Refuse rather than draw a misaligned basemap: the widget swaps in a Bokeh tile layer,
                # which only registers with the data on a Web-Mercator map.
                self._require_web_mercator("dashboard basemap")
                widget = pn.widgets.Select(label="Basemap", options=_BASEMAP_CHOICES)
            else:
                raise ValueError(
                    f"unknown dashboard widget {name!r}; choose from 'cmap'/'alpha'/'basemap'"
                )
            controls.append(widget)
            bindings[name] = widget
        return controls, bindings

    def _recorded_overridable_style(self) -> dict:
        """Merge the overridable style the builders recorded for the colour-mapped layers.

        Read back through :meth:`~digitalearth.interactive.base.InteractiveMapBase.style_of`, so a widget
        override lands *over* the styling the builders applied instead of on top of nothing — a recorded
        ``clim`` survives a ``cmap`` change, and the widget value is what differs.

        Returns:
            dict: the ``cmap``/``clim``/``alpha`` entries recorded for the ``Image``/``QuadMesh`` layers.
        """
        merged: dict = {}
        for layer in self.layers:
            if type(layer).__name__ not in _COLOR_MAPPED_TYPES:
                continue
            recorded = self.style_of(layer)["common"]
            merged.update(
                {
                    key: value
                    for key, value in recorded.items()
                    if key in _OVERRIDABLE_STYLE
                }
            )
        return merged

    def _with_basemap(self, obj: Any, provider: str) -> Any:
        """Compose ``obj`` over the ``provider`` tile layer, replacing any basemap already in it.

        The provider name is resolved through the same
        :meth:`~digitalearth.interactive.decoration.DecorationMixin._build_tiles` path ``tiles()`` uses, so
        the widget and the builder agree on provider names. Any tile element already in ``obj`` is dropped
        first — stacking a second basemap *underneath* the old one is what would make the switcher look
        inert on a map that already called ``tiles()``.

        Args:
            obj: The composed HoloViews object to draw over the basemap.
            provider: A provider name from :data:`_BASEMAP_CHOICES` (or anything ``tiles()`` accepts).

        Returns:
            The overlay with the chosen basemap underneath.

        Raises:
            ValueError: when the display CRS is not Web Mercator, or the provider name is unknown.
        """
        gv, hv = _require_holoviz()
        self._require_web_mercator("dashboard basemap")
        # A sibling-mixin call: _build_tiles lives on DecorationMixin, which the composed InteractiveMap
        # supplies. The TYPE_CHECKING base above carries the shared state only, not the sibling mixins.
        tile = self._build_tiles(provider, None).opts(level="underlay")  # type: ignore[attr-defined]
        elements = list(obj) if isinstance(obj, hv.Overlay) else [obj]
        data = [
            element for element in elements if type(element).__name__ not in _TILE_TYPES
        ]
        return reduce(_mul, [tile, *data])

    def _render_with_overrides(self, values: dict) -> Any:
        """Re-render the composed map, applying widget values as style overrides.

        Args:
            values: Widget values keyed by widget name (``cmap``/``alpha``/``basemap``).

        Returns:
            The composed HoloViews object with the overrides applied — the colour-mapped layers restyled
            with the widget values merged over their recorded style, over the chosen tile basemap.
        """
        gv, hv = _require_holoviz()
        obj = self.render()
        overrides: dict = {}
        if values.get("cmap"):
            overrides["cmap"] = values["cmap"]
        if values.get("alpha") is not None:
            overrides["alpha"] = values["alpha"]
        if overrides:
            merged = {**self._recorded_overridable_style(), **overrides}
            # cmap/alpha apply to the colour-mapped element types; HoloViews applies each spec only to
            # the matching elements and tolerates a map without them (a vector-only map is returned
            # unchanged), so no error-swallowing wrapper is needed here.
            obj = obj.opts(hv.opts.Image(**merged), hv.opts.QuadMesh(**merged))
        if values.get("basemap"):
            obj = self._with_basemap(obj, values["basemap"])
        return obj

    def serve(self, **kwargs: Any) -> Any:
        """Mark the dashboard servable for ``panel serve`` and return it.

        Args:
            **kwargs: Forwarded to :meth:`dashboard`.

        Returns:
            The servable ``panel.viewable.Viewable``.
        """
        app = self.dashboard(**kwargs)
        app.servable()
        return app

    def save_app(self, path: Any, *, embed: bool = True, **kwargs: Any) -> Path:
        """Export the dashboard to a standalone HTML file (no server).

        Panel's ``embed`` bakes the discrete widget states into the page, so the exported file is
        interactive offline within Panel's combinatorial limits (a few discrete widgets; continuous
        sliders are sampled). A live, unrestricted app must be served, not exported — and a
        pyramids-backed app cannot run in WASM/Pyodide (no GDAL in the browser).

        Args:
            path: Destination ``.html`` file (``str`` or ``pathlib.Path``).
            embed: Bake widget states for offline interactivity (``True``) or export a static
                snapshot (``False``).
            **kwargs: Forwarded to :meth:`dashboard`.

        Returns:
            pathlib.Path: the file written, matching the tier's ``save``/``save_animation`` (#248).

        Examples:
            - Export an offline, self-contained dashboard page:
                ```python
                >>> from pyramids.dataset import Dataset                       # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")       # doctest: +SKIP
                >>> InteractiveMap().image(dem).save_app("app.html").name     # doctest: +SKIP
                'app.html'

                ```
        """
        app = self.dashboard(**kwargs)
        app.save(path, embed=embed)
        return Path(path)

    def cross_tier_pane(self, scene3d: Any, **kwargs: Any) -> Any:
        """Embed an M1 ``Scene3D`` (PyVista) as a Panel pane beside this map (DI.4c).

        Imports the 3-D tier **lazily** and degrades with a clear error if the ``[3d]`` extra is
        absent — the tiers compose but neither hard-depends on the other.

        Args:
            scene3d: A ``digitalearth.three_d.Scene3D`` (or its ``plotter``) to embed.
            **kwargs: Forwarded to ``panel.pane.VTK``.

        Returns:
            A ``panel.pane.VTK`` rendering the 3-D scene.

        Raises:
            ImportError: when the ``[3d]`` extra (PyVista/VTK) is not installed.
        """
        pn = _require_panel()
        if find_spec("vtk") is None and find_spec("vtkmodules") is None:
            raise ImportError(
                "cross_tier_pane needs the 3-D tier (pip install 'digitalearth[3d]') for the "
                "PyVista/VTK pane"
            )
        plotter = getattr(scene3d, "plotter", scene3d)
        render_window = getattr(plotter, "ren_win", plotter)
        return pn.pane.VTK(render_window, **kwargs)

    def layer_control(
        self,
        *,
        opacity: bool = True,
        reorder: bool = False,
        basemap_switch: bool = True,
    ) -> Any:
        """Build a layer manager: per-layer visibility toggles, opacity slider, basemap switch (DI.13).

        The toggle order follows add order. Drag-reorder is **not implemented**: reordering needs a stable
        handle per layer (roadmap IN-1; the static twin is #216), which the registry does not yet give, so
        ``reorder=True`` is refused outright rather than accepted and ignored.

        The basemap switch is offered only on a Web-Mercator map — Bokeh renders tiles in EPSG:3857 only,
        so on any other display CRS the switch is dropped (and logged), never drawn misaligned.

        Args:
            opacity: Include an opacity slider driving the visible colour-mapped layers.
            reorder: Must stay ``False``; ``True`` raises ``NotImplementedError``.
            basemap_switch: Include a basemap-provider ``Select``, bound so picking a provider swaps the
                tile layer under the map.

        Returns:
            A ``panel.viewable.Viewable`` whose widgets reactively rebuild the map overlay — toggling
            a layer hides/shows it; the opacity slider sets its alpha; the basemap ``Select`` swaps tiles.

        Raises:
            ValueError: when there are no layers to control.
            NotImplementedError: when ``reorder=True`` is passed — reordering is not implemented,
                and the flag is refused rather than accepted and ignored (roadmap IN-1; the static
                twin is #216).

        Examples:
            - One toggle per registered layer, labelled by index and element type, in add order:
                ```python
                >>> from pyramids.dataset import Dataset                       # doctest: +SKIP
                >>> from pyramids.feature import FeatureCollection             # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")       # doctest: +SKIP
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")  # doctest: +SKIP
                >>> m = InteractiveMap().image(dem).points(fc)                 # doctest: +SKIP
                >>> panel = m.layer_control()                                  # doctest: +SKIP
                >>> panel[0][0].options                                        # doctest: +SKIP
                ['0: Image', '1: Points']

                ```
            - ``reorder=True`` is refused outright: the registry has no stable per-layer handle to
              reorder by, and silently ignoring the flag was how the control looked inert:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> m = InteractiveMap().image(dem)                            # doctest: +SKIP
                >>> try:                                                       # doctest: +SKIP
                ...     m.layer_control(reorder=True)
                ... except NotImplementedError as error:
                ...     print(str(error).split(" — ")[0])
                layer_control(reorder=True) is not implemented

                ```
            - On a Web-Mercator map the controls are toggles + opacity + basemap; on any other
              display CRS the basemap switch is dropped (and logged) rather than drawn misaligned:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> mercator = InteractiveMap().image(dem).layer_control()     # doctest: +SKIP
                >>> len(mercator[0])                                           # doctest: +SKIP
                3
                >>> other = InteractiveMap(crs=4326).image(dem)                 # doctest: +SKIP
                >>> len(other.layer_control()[0])                              # doctest: +SKIP
                2

                ```
        """
        gv, hv = _require_holoviz()
        pn = _require_panel()
        if reorder:
            raise NotImplementedError(
                "layer_control(reorder=True) is not implemented — reordering needs a stable per-layer "
                "handle (roadmap IN-1; static twin #216) before draw order can be manipulated. Pass reorder=False "
                "(the default); the toggles follow add order, which you control by the order you build in."
            )
        if not self.layers:
            raise ValueError(
                "layer_control() needs at least one layer — add a builder call first"
            )
        names = [f"{i}: {type(layer).__name__}" for i, layer in enumerate(self.layers)]
        visible = pn.widgets.CheckBoxGroup(value=names, options=names)
        controls = [visible]
        alpha = (
            pn.widgets.FloatSlider(label="Opacity", start=0.0, end=1.0, value=1.0)
            if opacity
            else None
        )
        if alpha is not None:
            controls.append(alpha)
        basemap = self._basemap_select(pn) if basemap_switch else None
        if basemap is not None:
            controls.append(basemap)

        bind_kwargs = {"shown": visible}
        if alpha is not None:
            bind_kwargs["op"] = alpha
        if basemap is not None:
            bind_kwargs["basemap"] = basemap
        view = pn.bind(self._compose_visible_layers, **bind_kwargs)
        return pn.Row(pn.Column(*controls), pn.panel(view))

    def _basemap_select(self, pn: Any) -> Any:
        """Build the layer-control basemap ``Select``, or ``None`` on a non-Web-Mercator map.

        Args:
            pn: The imported panel module.

        Returns:
            A ``panel.widgets.Select`` over :data:`_BASEMAP_CHOICES`, or ``None`` when the display CRS is
            not EPSG:3857 — in which case the omission is logged rather than left to be guessed at.
        """
        if self.crs != 3857:
            from loguru import logger

            logger.info(
                f"layer_control(): basemap switch omitted — Bokeh renders tiles in EPSG:3857 only and "
                f"this map uses crs={self.crs!r}; build the map with InteractiveMap(crs=3857) for it."
            )
            return None
        return pn.widgets.Select(label="Basemap", options=_BASEMAP_CHOICES)

    def _compose_visible_layers(
        self, shown: list, op: float = 1.0, basemap: Any = None
    ) -> Any:
        """Compose the layers named in ``shown`` into a styled overlay (the layer-control view).

        The opacity is merged over the style the builders recorded for those layers (read back through
        :meth:`~digitalearth.interactive.base.InteractiveMapBase.style_of`), so hiding and re-showing a
        layer cannot quietly drop its ``cmap``/``clim``.

        Args:
            shown: The ``"i: Type"`` labels of the visible layers (from the toggle widget).
            op: Opacity applied to colour-mapped layers.
            basemap: Provider name from the basemap ``Select``, or ``None`` to leave the basemap as built.

        Returns:
            A blank ``hv.Overlay`` when nothing is shown, else the overlay of the chosen layers at
            opacity ``op``, over the chosen basemap when one is selected.
        """
        gv, hv = _require_holoviz()
        chosen = [self.layers[int(label.split(":")[0])] for label in shown]
        if not chosen:
            return hv.Overlay([])
        overlay = chosen[0] if len(chosen) == 1 else reduce(_mul, chosen)
        merged = {**self._recorded_overridable_style(), "alpha": op}
        overlay = overlay.opts(hv.opts.Image(**merged), hv.opts.RGB(alpha=op))
        if basemap:
            overlay = self._with_basemap(overlay, basemap)
        return overlay

    def attribute_table(self, features: Any, *, linked: bool = False) -> Any:
        """Build a read-only ``Tabulator`` attribute table of a vector ``FeatureCollection`` (DI.13).

        The table is a **read-only view** of the attributes, not a selection-linked companion to the map:
        two-way linking needs a ``holoviews.link_selections`` (or a selection stream) shared with the map's
        elements, which is roadmap IN-5 / DE-13. ``linked=True`` is refused rather than accepted and
        ignored.

        Args:
            features: A pyramids ``FeatureCollection`` (or GeoDataFrame) whose non-geometry columns
                populate the table.
            linked: Must stay ``False``; ``True`` raises ``NotImplementedError``. It used to
                default to ``True`` and do nothing, which read as "linking is on" when no
                selection was ever shared with the map.

        Returns:
            A ``panel.widgets.Tabulator`` of the attribute columns — paginated, and ``disabled`` so
            the view cannot be edited into disagreeing with the data on the map.

        Raises:
            NotImplementedError: when ``linked=True`` is passed — two-way selection linking
                is not implemented (roadmap IN-5 / DE-13), and the flag is refused rather
                than accepted and ignored.

        Examples:
            - The geometry column is dropped; what is left is the attributes, unchanged:
                ```python
                >>> from pyramids.feature import FeatureCollection             # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")  # doctest: +SKIP
                >>> table = InteractiveMap().attribute_table(fc)               # doctest: +SKIP
                >>> list(table.value.columns)                                  # doctest: +SKIP
                ['fid']
                >>> len(table.value) == len(fc)                                # doctest: +SKIP
                True

                ```
            - The view is read-only by construction — a viewer cannot edit a cell into disagreeing
              with the map:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> InteractiveMap().attribute_table(fc).disabled              # doctest: +SKIP
                True

                ```
            - ``linked=True`` is refused: nothing shares a selection with the map's elements yet,
              and the table is ``disabled``, so it could not emit one either:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> try:                                                       # doctest: +SKIP
                ...     InteractiveMap().attribute_table(fc, linked=True)
                ... except NotImplementedError as error:
                ...     print(str(error).split(" — ")[0])
                attribute_table(linked=True) is not implemented

                ```
        """
        if linked:
            raise NotImplementedError(
                "attribute_table(linked=True) is not implemented — two-way selection linking needs a "
                "holoviews.link_selections shared with the map's elements (roadmap IN-5/DE-13), and the "
                "table is disabled=True, so it cannot emit a selection. Pass linked=False (the default) "
                "for the read-only attribute view."
            )
        pn = _require_panel()
        pn.extension("tabulator")
        frame = features.drop(columns=[features.geometry.name], errors="ignore")
        return pn.widgets.Tabulator(
            frame, disabled=True, pagination="remote", page_size=20
        )

    def share(
        self, *, params: Sequence[str] = ("cmap", "extent", "time", "basemap")
    ) -> Any:
        """Sync the listed view parameters into the URL for a shareable, reproducible view (DI.13).

        Uses ``panel.state.location`` — available only under a running ``panel serve``. Off-server
        (e.g. in a notebook or test) it returns the params it *would* sync rather than failing, so the
        call is safe everywhere. A pyramids-backed app must be **served**, not exported to WASM/Pyodide
        (no GDAL in the browser).

        Args:
            params: The view parameters to serialise into the URL query string.

        Returns:
            The ``panel.io.location.Location`` it synced to, or the ``params`` tuple when off-server.
        """
        pn = _require_panel()
        location = getattr(pn.state, "location", None)
        if location is None:  # off-server (notebook/test) — nothing to bind to yet
            return tuple(params)
        location.sync(self, {name: name for name in params})
        return location
