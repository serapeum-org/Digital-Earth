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
from typing import Any, Sequence

from digitalearth.interactive.base import _require_holoviz

#: Basemap providers offered by the layer-control basemap switcher.
_BASEMAP_CHOICES = ["CartoLight", "CartoDark", "OSM", "EsriImagery"]

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


class DashboardMixin:
    """Panel dashboard builders (DI.4): wrap the map + reactive widgets into a servable/exportable app."""

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
        selector), ``"alpha"`` (opacity slider), ``"basemap"`` (tile-provider selector).

        Args:
            widgets: Which widgets to expose, in order.
            sidebar: Lay widgets out in a left sidebar (``True``) or a top row (``False``).
            title: Dashboard title (falls back to the map's ``title``).

        Returns:
            A ``panel.viewable.Viewable`` hosting the map + widgets.

        Raises:
            ValueError: for an unknown widget name.

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
            ValueError: for an unknown widget name.
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
                widget = pn.widgets.Select(
                    label="Basemap", options=["CartoLight", "CartoDark", "OSM"]
                )
            else:
                raise ValueError(
                    f"unknown dashboard widget {name!r}; choose from 'cmap'/'alpha'/'basemap'"
                )
            controls.append(widget)
            bindings[name] = widget
        return controls, bindings

    def _render_with_overrides(self, values: dict) -> Any:
        """Re-render the composed map, applying widget values as style overrides.

        Args:
            values: Widget values keyed by widget name (``cmap``/``alpha``/``basemap``).

        Returns:
            The composed HoloViews object with the overrides applied.
        """
        gv, hv = _require_holoviz()
        obj = self.render()
        opts: dict = {}
        if values.get("cmap"):
            opts["cmap"] = values["cmap"]
        if values.get("alpha") is not None:
            opts["alpha"] = values["alpha"]
        if not opts:
            return obj
        # cmap/alpha apply to the colour-mapped element types; HoloViews applies each spec only to
        # the matching elements and tolerates a map without them (a vector-only map is returned
        # unchanged), so no error-swallowing wrapper is needed here.
        return obj.opts(hv.opts.Image(**opts), hv.opts.QuadMesh(**opts))

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

    def save_app(self, path: str, *, embed: bool = True, **kwargs: Any) -> str:
        """Export the dashboard to a standalone HTML file (no server).

        Panel's ``embed`` bakes the discrete widget states into the page, so the exported file is
        interactive offline within Panel's combinatorial limits (a few discrete widgets; continuous
        sliders are sampled). A live, unrestricted app must be served, not exported — and a
        pyramids-backed app cannot run in WASM/Pyodide (no GDAL in the browser).

        Args:
            path: Destination ``.html`` file.
            embed: Bake widget states for offline interactivity (``True``) or export a static
                snapshot (``False``).
            **kwargs: Forwarded to :meth:`dashboard`.

        Returns:
            The ``path`` written.

        Examples:
            - Export an offline, self-contained dashboard page:
                ```python
                >>> from pyramids.dataset import Dataset                       # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")       # doctest: +SKIP
                >>> InteractiveMap().image(dem).save_app("app.html")          # doctest: +SKIP
                'app.html'

                ```
        """
        app = self.dashboard(**kwargs)
        app.save(path, embed=embed)
        return str(path)

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
        self, *, opacity: bool = True, reorder: bool = True, basemap_switch: bool = True
    ) -> Any:
        """Build a layer manager: per-layer visibility toggles, opacity sliders, basemap switch (DI.13).

        Args:
            opacity: Include a per-layer opacity slider.
            reorder: Reserved for drag-reorder (currently the toggle order follows add order).
            basemap_switch: Include a basemap-provider ``Select``.

        Returns:
            A ``panel.viewable.Viewable`` whose widgets reactively rebuild the map overlay — toggling
            a layer hides/shows it; the opacity slider sets its alpha.

        Raises:
            ValueError: when there are no layers to control.
        """
        gv, hv = _require_holoviz()
        pn = _require_panel()
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
        if basemap_switch:
            controls.append(
                pn.widgets.Select(label="Basemap", options=_BASEMAP_CHOICES)
            )

        bind_kwargs = {"shown": visible}
        if alpha is not None:
            bind_kwargs["op"] = alpha
        view = pn.bind(self._compose_visible_layers, **bind_kwargs)
        return pn.Row(pn.Column(*controls), pn.panel(view))

    def _compose_visible_layers(self, shown: list, op: float = 1.0) -> Any:
        """Compose the layers named in ``shown`` into a styled overlay (the layer-control view).

        Args:
            shown: The ``"i: Type"`` labels of the visible layers (from the toggle widget).
            op: Opacity applied to colour-mapped layers.

        Returns:
            A blank ``hv.Overlay`` when nothing is shown, the single element when one is, else the
            overlay of the chosen layers at opacity ``op``.
        """
        gv, hv = _require_holoviz()
        chosen = [self.layers[int(label.split(":")[0])] for label in shown]
        if not chosen:
            return hv.Overlay([])
        overlay = chosen[0] if len(chosen) == 1 else reduce(_mul, chosen)
        return overlay.opts(hv.opts.Image(alpha=op), hv.opts.RGB(alpha=op))

    def attribute_table(self, features: Any, *, linked: bool = True) -> Any:
        """Build a ``Tabulator`` attribute table of a vector ``FeatureCollection`` (DI.13).

        Args:
            features: A pyramids ``FeatureCollection`` (or GeoDataFrame) whose non-geometry columns
                populate the table.
            linked: Reserved for two-way selection linking with the map (needs a live server).

        Returns:
            A ``panel.widgets.Tabulator`` of the attribute columns.
        """
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
