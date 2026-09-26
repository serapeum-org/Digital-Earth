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

import warnings
from functools import reduce
from importlib.util import find_spec
from operator import mul as _mul
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Self, Sequence, Tuple

from digitalearth.base.controls import (
    LAYER_CONTROLS,
    check_control_position,
    resolved_controls,
)
from digitalearth.base.deprecation import REMOVED_IN
from digitalearth.base.spec.bounds import same_crs
from digitalearth.interactive.base import _require_holoviz
from digitalearth.interactive.decoration import DEFAULT_BASEMAP_PROVIDER

#: Basemap providers offered by the basemap switchers — the **one** list behind both the dashboard's
#: ``"basemap"`` widget and the layer-control switch, so the two cannot offer different providers.
_BASEMAP_CHOICES = list(
    dict.fromkeys(
        [DEFAULT_BASEMAP_PROVIDER, "CartoLight", "CartoDark", "OSM", "EsriImagery"]
    )
)


def _controls_from_flags(
    controls: Optional[Sequence[str]],
    *,
    opacity: Optional[bool],
    basemap_switch: Optional[bool],
    caller: str,
) -> Tuple[Tuple[str, ...], bool]:
    """Settle which controls a layer manager exposes, from ``controls=`` or the two deprecated booleans.

    ``opacity=`` and ``basemap_switch=`` were this tier's way of saying "expose these controls", one boolean
    per control, and the web tier had no way of saying it at all. One list says it on both tiers (#264), so
    the booleans are resolved into it here and warn — the same promise
    :func:`~digitalearth.base.deprecation.renamed_parameter` keeps for a renamed parameter, written out
    because this is two old names collapsing into one new one rather than a rename.

    Args:
        controls: The caller's control names, or ``None`` for "not passed".
        opacity: The deprecated flag, or ``None`` for "not passed". Its effective default is ``True``.
        basemap_switch: The deprecated flag, or ``None`` for "not passed" — which is also its documented
            "offer it unprompted" value, so the two coincide and nothing is lost by the sentinel.
        caller: The public method the keywords were written on, for every message to quote.

    Returns:
        ``(controls, requested)`` — the control names to build, and whether the caller asked for them
        **outright**. The second half is what decides how a basemap switch on a non-Web-Mercator map is
        answered: a named control is a request, and is refused with the CRS; the unprompted default is
        furniture, and is dropped with a log line.

    Raises:
        TypeError: when ``controls`` is passed together with either boolean. They are two spellings of one
            request, so honouring either would silently drop the other.
        ValueError: from :func:`~digitalearth.base.controls.resolved_controls`, for a name no tier has or
            one this tier cannot build.

    Warns:
        DeprecationWarning: naming the boolean used and ``controls=`` as its replacement.
    """
    deprecated = {"opacity": opacity, "basemap_switch": basemap_switch}
    passed = sorted(name for name, value in deprecated.items() if value is not None)
    if controls is not None and passed:
        raise TypeError(
            f"{caller} got controls= together with {passed}; those are the deprecated spelling of the same "
            "request, so neither can be silently preferred. Keep controls=."
        )
    if controls is not None:
        return resolved_controls(controls, offered=LAYER_CONTROLS, caller=caller), True
    if not passed:
        return tuple(LAYER_CONTROLS), False
    warnings.warn(
        f"{caller}: {', '.join(f'{name}=' for name in passed)} is deprecated and will be removed in "
        f"{REMOVED_IN}; name the controls to expose with controls= instead, e.g. "
        "controls=('visibility', 'opacity')",
        DeprecationWarning,
        stacklevel=3,
    )
    named = ["visibility"]
    if opacity is not False:
        named.append("opacity")
    if basemap_switch is not False:
        named.append("basemap")
    resolved = resolved_controls(named, offered=LAYER_CONTROLS, caller=caller)
    return resolved, basemap_switch is True


#: Element type names that are a tile basemap — replaced (not stacked under) when a basemap widget picks
#: a provider, so a map that already carries tiles still responds to the switcher.
_TILE_TYPES = ("WMTS", "Tiles")


def _drawn_type(layer: Any) -> str:
    """Return the name of the element type a registered layer actually draws.

    A `DynamicMap` is a wrapper: `large_image` produces `Image` frames through one, and `datashade`,
    `trajectory` and a bundled `network` produce **RGB** frames through one. Choosing the restyle branch by
    the wrapper's own type name sent `cmap` to all of them — and HoloViews refuses `cmap` on an RGB, from
    inside the callback, where `param` logs the error and the map silently stops updating (review H5).

    Args:
        layer: A registered layer.

    Returns:
        The produced element's type name where a wrapper declares one, else the layer's own. A `DynamicMap`
        that has not drawn a frame yet declares nothing, and comes back as `"DynamicMap"` — which is in
        neither style table, so such a layer is left alone until it has drawn.
    """
    produced = getattr(layer, "type", None)
    if produced is not None:
        return str(produced.__name__)
    return type(layer).__name__


#: Element type names the dashboard's ``cmap``/``alpha`` overrides apply to (the colour-mapped raster
#: elements). Their recorded style is what an override is merged over.
#:
#: A dynamic layer is classified by what it *draws* — see :func:`_drawn_type` — so a `large_image` lands
#: here through its `Image` frames while a `datashade` lands in :data:`_ALPHA_ONLY_TYPES` through its `RGB`
#: ones. Naming `DynamicMap` itself here sent `cmap` to both (review H5).
_COLOR_MAPPED_TYPES = ("Image", "QuadMesh")

#: Style keys a widget may override on a colour-mapped layer. Anything the builders recorded under these
#: keys is read back and merged *under* the widget value, so an override never silently drops the rest of
#: the styling a builder applied.
_OVERRIDABLE_STYLE = ("cmap", "clim", "alpha")

#: Element type names an override reaches that carry no scalar to colour-map: an ``RGB`` composite is
#: already three colour channels, so ``cmap``/``clim`` have nothing to act on and only the alpha of the
#: merged style applies to it. Listing it here is what keeps it from being skipped altogether.
_ALPHA_ONLY_TYPES = ("RGB",)

#: The subset of :data:`_OVERRIDABLE_STYLE` an :data:`_ALPHA_ONLY_TYPES` element can take.
_ALPHA_ONLY_STYLE = ("alpha",)

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
                >>> app = InteractiveMap().field(dem).dashboard(widgets=("cmap", "alpha"))  # doctest: +SKIP
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

    def _restyled_layers(
        self, overrides: dict, layers: Optional[Sequence[Any]] = None
    ) -> list:
        """Return each layer restyled with **its own** recorded style, the widget's values over the top.

        The defect this replaces: the widget values were merged with the style of *every* colour-mapped layer
        into one dict — last layer wins — and that dict was applied to all of them through
        ``hv.opts.Image(**merged)``, which HoloViews applies per element *type*. Moving the opacity slider on a
        map with two rasters therefore gave both the second one's colormap and colour limits. A layer's style
        is the layer's, so it is applied to the layer.

        Args:
            overrides: The widget values — `cmap`, `alpha` — to apply over each layer's own style.
            layers: The layers to restyle, for a widget that shows a subset of them — the layer switcher
                hides the ones nobody ticked. `None` (the default) restyles the whole map.

        Returns:
            The layers in draw order, each carrying its own styling with the overrides on top. A layer no
            widget claims is returned unchanged.
        """
        alpha_only = {
            key: value for key, value in overrides.items() if key in _ALPHA_ONLY_STYLE
        }
        styled = []
        for layer in self.layers if layers is None else layers:
            name = _drawn_type(layer)
            if name in _COLOR_MAPPED_TYPES:
                recorded = {
                    key: value
                    for key, value in self.style_of(layer)["common"].items()
                    if key in _OVERRIDABLE_STYLE
                }
                styled.append(layer.opts(**{**recorded, **overrides}))
            elif name in _ALPHA_ONLY_TYPES and alpha_only:
                # An RGB composite is already three colour channels: it has no scalar for a cmap or a clim
                # to map, so only the opacity reaches it.
                styled.append(layer.opts(**alpha_only))
            else:
                styled.append(layer)
        return styled

    def _with_basemap(
        self, obj: Any, provider: str, *, context: str = "dashboard basemap"
    ) -> Any:
        """Compose ``obj`` over the ``provider`` tile layer, replacing any basemap already in it.

        The provider name is resolved through the same
        :meth:`~digitalearth.interactive.decoration.DecorationMixin._build_tiles` path ``tiles()`` uses, so
        the widget and the builder agree on provider names. Any tile element already in ``obj`` is dropped
        first — stacking a second basemap *underneath* the old one is what would make the switcher look
        inert on a map that already called ``tiles()``.

        Args:
            obj: The composed HoloViews object to draw over the basemap.
            provider: A provider name from :data:`_BASEMAP_CHOICES` (or anything ``tiles()`` accepts).
            context: The requesting entry point, quoted in the Web-Mercator refusal so the message
                names the widget the caller actually touched.

        Returns:
            The overlay with the chosen basemap underneath.

        Raises:
            ValueError: when the display CRS is not Web Mercator, or the provider name is unknown.
        """
        _, hv = _require_holoviz()
        self._require_web_mercator(context)
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
            The composed HoloViews object with the overrides applied — each layer redrawn with the widget
            values over **its own** recorded style, over the chosen tile basemap.

            Whether a widget moved decides how the layers are styled, and nothing else: both branches flush
            the deferred basemap and apply the map's projection, which is what `render()` does besides
            composing. Skipping them left a map built with `tiles=` without one, permanently — both default
            widgets carry a value, so the override branch is taken on the very first render.
        """
        overrides: dict = {}
        if values.get("cmap"):
            overrides["cmap"] = values["cmap"]
        if values.get("alpha") is not None:
            overrides["alpha"] = values["alpha"]
        # Before the layers are read: a deferred basemap is one of them.
        self._flush_deferred_tiles()
        if not overrides:
            obj = self._projected(self._compose(self.layers))
        else:
            # Composed from the restyled layers rather than restyled after composing: `.opts()` on an overlay
            # applies per element *type*, so one spec cannot give two rasters different colormaps (#300).
            obj = self._projected(self._compose(self._restyled_layers(overrides)))
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
                >>> InteractiveMap().field(dem).save_app("app.html").name     # doctest: +SKIP
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

    #: The Panel layer manager the last :meth:`layer_control` built, or ``None`` before the first one.
    #: Declared on the class rather than in ``__init__`` — which belongs to the shared base, not to this
    #: capability — so a map that never asks for a layer control never carries one.
    _layer_control_panel: Any = None

    @property
    def layer_control_panel(self) -> Any:
        """The Panel layer manager :meth:`layer_control` built, or ``None`` when none has been built.

        :meth:`layer_control` returns the map, the way every other builder on both 2-D tiers does (#264), so
        the object a caller *displays* is held on the map and read back here — the same place the static tier
        keeps its ``fig``/``ax`` and this tier keeps its ``layers``. Read-only on purpose: a panel this map
        did not build is not this map's control.

        Returns:
            A ``panel.viewable.Viewable`` whose widgets reactively rebuild the map overlay — toggling a
            layer hides/shows it, the opacity slider sets its alpha, and the basemap ``Select`` swaps the
            tiles underneath. ``None`` until :meth:`layer_control` has been called; a second call replaces
            it rather than stacking a second control.

        Examples:
            - The control is reached through the map the builder returned:
                ```python
                >>> from pyramids.dataset import Dataset                       # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")       # doctest: +SKIP
                >>> m = InteractiveMap().field(dem).layer_control()            # doctest: +SKIP
                >>> m.layer_control_panel is None                              # doctest: +SKIP
                False

                ```
        """
        return self._layer_control_panel

    def _control_layer_indices(
        self, layers: Optional[Sequence[str]]
    ) -> Tuple[int, ...]:
        """Return the positions of the layers a control offers, in draw order.

        Positions rather than ids, because the toggle labels are indexed (``"0: Image"``) and
        :meth:`_compose_visible_layers` reads the index back out of the label it was handed.
        ``self.layers`` and :attr:`~digitalearth.interactive.base.InteractiveMapBase.layer_ids` are built in
        step — each element is inserted at the position its id holds in the tree — so one indexes the other.

        Args:
            layers: The ids the caller wants offered, or ``None`` for every layer on the map.

        Returns:
            The positions, ascending, de-duplicated.

        Raises:
            ValueError: when an id is not on this map. A typo would otherwise build a control over fewer
                layers than the caller named, and say nothing.
        """
        available = self.layer_ids
        if layers is None:
            return tuple(range(len(available)))
        wanted = list(dict.fromkeys(layers))
        unknown = [layer_id for layer_id in wanted if layer_id not in available]
        if unknown:
            raise ValueError(
                f"layer_control() was given {unknown}, which are not on this map; its layers are "
                f"{available}"
            )
        return tuple(sorted(available.index(layer_id) for layer_id in wanted))

    def _layer_control_widgets(
        self, pn: Any, wanted: Sequence[str], *, labels: list, requested: bool
    ) -> dict:
        """Build the widgets a layer control exposes, keyed by the control each answers to.

        Args:
            pn: The imported panel module.
            wanted: The controls to build, from :data:`~digitalearth.base.controls.LAYER_CONTROLS`.
            labels: The ``"i: Type"`` label per offered layer, which the toggle group lists.
            requested: Whether the controls were named outright, which decides how a basemap switch on a
                non-Web-Mercator map is answered — see :meth:`_basemap_select`.

        Returns:
            ``{control: widget}`` in build order, so the layout lays them out in it. ``"basemap"`` is absent
            when the switch was offered unprompted and the display CRS cannot take it.
        """
        built = {"visibility": pn.widgets.CheckBoxGroup(value=labels, options=labels)}
        if "opacity" in wanted:
            built["opacity"] = pn.widgets.FloatSlider(
                label="Opacity", start=0.0, end=1.0, value=1.0
            )
        if "basemap" in wanted:
            select = self._basemap_select(pn, requested=requested)
            if select is not None:
                built["basemap"] = select
        return built

    def layer_control(
        self,
        *,
        layers: Optional[Sequence[str]] = None,
        position: str = "top-right",
        controls: Optional[Sequence[str]] = None,
        opacity: Optional[bool] = None,
        reorder: bool = False,
        basemap_switch: Optional[bool] = None,
    ) -> Self:
        """Build a layer manager — per-layer visibility, an opacity slider, a basemap switch (DI.13).

        The three keywords are the ones the Tier-2 contract declares and the web tier answers to as well —
        the layers to include, the position, the controls to expose — and **the map comes back**, so one call
        adds a layer control on either tier (#264). The panel itself is held on the map and read back through
        :attr:`layer_control_panel`; before, this method returned it, and so ended any chain it appeared in.

        The toggle order follows **draw** order — the order `layers` is in, which is the band each layer's
        kind declares and then the order they were built in within that band, so a basemap added last is
        still the first toggle. Drag-reorder is **not implemented**: reordering needs a stable
        handle per layer (roadmap IN-1; the static twin is #216), which the registry does not yet give, so
        ``reorder=True`` is refused outright rather than accepted and ignored. That is also why ``reorder``
        is not one of the ``controls``: it names a manipulation no tier can offer, not a widget.

        Bokeh renders tiles in EPSG:3857 only, so the basemap switch is Web-Mercator-only — and which
        way that lands depends on whether it was *asked for*. Left at the default ``controls`` the switch
        is furniture this method offers unprompted, so a non-Mercator map drops it and logs why; named in
        ``controls`` it is an explicit request, and a non-Mercator map is refused with its CRS named. That
        is the one policy this module applies to an explicit basemap request — the same answer
        ``dashboard(widgets=("basemap",))`` gives.

        Args:
            layers: The layers to offer, by id from
                :attr:`~digitalearth.interactive.base.InteractiveMapBase.layer_ids`; ``None`` (default)
                offers every layer on the map. A layer left out is hidden from the switch, **not** from the
                map — it stays in the composed overlay whatever the toggles say, which is what the web
                tier's ``layers=`` means too.
            position: Which of :data:`~digitalearth.base.controls.CONTROL_POSITIONS` the widget column sits
                in. The corner is honoured as far as a two-cell row can: ``*-left`` puts the widgets before
                the map and ``*-right`` after it, while ``top-*``/``bottom-*`` align the column to the top or
                the bottom of the row. Note ``"top-right"`` is the default both tiers share, so the widgets
                sit to the **right** of the map rather than the left.
            controls: Which controls to expose, from
                :data:`~digitalearth.base.controls.LAYER_CONTROLS`; ``None`` (default) offers all three, and
                ``"visibility"`` cannot be dropped. Naming a control is an explicit request — see the
                basemap policy above.
            opacity: **Deprecated** spelling of ``"opacity"`` in ``controls`` (effective default ``True``);
                honoured for one release, after a ``DeprecationWarning``.
            reorder: Must stay ``False``; ``True`` raises ``NotImplementedError``.
            basemap_switch: **Deprecated** spelling of ``"basemap"`` in ``controls``; honoured for one
                release, after a ``DeprecationWarning``. ``True`` requires the switch, ``False`` drops it,
                and ``None`` (default) offers it unprompted.

        Returns:
            The same map instance, so builder calls chain. The panel is :attr:`layer_control_panel`.

        Raises:
            TypeError: when ``controls`` is passed together with ``opacity`` or ``basemap_switch`` — two
                spellings of one request.
            ValueError: when ``position`` is not one of the four corners; when ``controls`` names something
                outside the shared vocabulary, drops ``"visibility"``, or names a control this tier cannot
                build; when there are no layers to control; when a ``layers`` id is not on this map; or when
                ``"basemap"`` is named on a map whose display CRS is not EPSG:3857 (Bokeh renders tiles in
                Web Mercator only).
            NotImplementedError: when ``reorder=True`` is passed — reordering is not implemented,
                and the flag is refused rather than accepted and ignored (roadmap IN-1; the static
                twin is #216).

        Examples:
            - One toggle per registered layer, labelled by index and element type, in draw order — and the
              map comes back, so the control is part of the chain rather than the end of it:
                ```python
                >>> from pyramids.dataset import Dataset                       # doctest: +SKIP
                >>> from pyramids.feature import FeatureCollection             # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")       # doctest: +SKIP
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")  # doctest: +SKIP
                >>> m = InteractiveMap().field(dem).points(fc).layer_control()  # doctest: +SKIP
                >>> m.layer_control_panel[1][0].options                        # doctest: +SKIP
                ['0: Image', '1: Points']

                ```
            - ``reorder=True`` is refused outright: the registry has no stable per-layer handle to
              reorder by, and silently ignoring the flag was how the control looked inert:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> m = InteractiveMap().field(dem)                            # doctest: +SKIP
                >>> try:                                                       # doctest: +SKIP
                ...     m.layer_control(reorder=True)
                ... except NotImplementedError as error:
                ...     print(str(error).split(" — ")[0])
                layer_control(reorder=True) is not implemented

                ```
            - Unprompted, the switch is furniture: on a Web-Mercator map the controls are toggles +
              opacity + basemap; on any other display CRS it is dropped (and logged) rather than
              drawn misaligned:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> mercator = InteractiveMap().field(dem).layer_control()     # doctest: +SKIP
                >>> len(mercator.layer_control_panel[1])                       # doctest: +SKIP
                3
                >>> other = InteractiveMap(crs=4326).field(dem).layer_control()  # doctest: +SKIP
                >>> len(other.layer_control_panel[1])                          # doctest: +SKIP
                2

                ```
            - Named outright, the same condition is refused instead, with the display CRS named:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> try:                                                       # doctest: +SKIP
                ...     other.layer_control(controls=("visibility", "basemap"))
                ... except ValueError as error:
                ...     print(str(error).split(" — ")[0])
                layer_control basemap() needs the Web-Mercator display CRS (crs=3857)

                ```

        See Also:
            layer_control_panel: the panel this builds, held on the map.
            digitalearth.base.controls.resolved_controls: the shared control vocabulary.
        """
        pn = _require_panel()
        # Before the labels are built. A map built with `tiles=` defers its basemap until something renders,
        # and the flush inserts it at index 0 — so labels frozen beforehand named the layer one place to
        # their left, and the control drew the basemap where the data should be (review H10).
        self._flush_deferred_tiles()
        if reorder:
            raise NotImplementedError(
                "layer_control(reorder=True) is not implemented — reordering needs a stable per-layer "
                "handle (roadmap IN-1; static twin #216) before draw order can be manipulated. Pass reorder=False "
                "(the default); the toggles follow draw order — each layer's band, then the order you "
                "built in within that band."
            )
        check_control_position(position)
        wanted, requested = _controls_from_flags(
            controls,
            opacity=opacity,
            basemap_switch=basemap_switch,
            caller="InteractiveMap.layer_control()",
        )
        if not self.layers:
            raise ValueError(
                "layer_control() needs at least one layer — add a builder call first"
            )
        offered = self._control_layer_indices(layers)
        labels = [f"{index}: {type(self.layers[index]).__name__}" for index in offered]
        widgets = self._layer_control_widgets(
            pn, wanted, labels=labels, requested=requested
        )
        # A layer nobody offered is not a layer the viewer chose to hide, so it is composed in whatever the
        # toggles say — the same thing `layers=` means on the web tier, where the switch lists a subset of a
        # map that still draws all of it.
        bound: dict = {
            "shown": widgets["visibility"],
            "always": tuple(
                index for index in range(len(self.layers)) if index not in offered
            ),
        }
        if "opacity" in widgets:
            bound["op"] = widgets["opacity"]
        if "basemap" in widgets:
            bound["basemap"] = widgets["basemap"]
        view = pn.bind(self._compose_visible_layers, **bound)
        # The corner, as far as a row of two cells can honour one: the side is the order of the cells, and
        # the top/bottom half is the column's own alignment across the row.
        column = pn.Column(
            *widgets.values(), align="start" if position.startswith("top") else "end"
        )
        body = pn.panel(view)
        self._layer_control_panel = (
            pn.Row(column, body) if position.endswith("left") else pn.Row(body, column)
        )
        return self

    def _basemap_select(self, pn: Any, *, requested: bool) -> Any:
        """Build the layer-control basemap ``Select``, or ``None`` on a non-Web-Mercator map.

        Args:
            pn: The imported panel module.
            requested: Whether the caller asked for the switch outright (``basemap_switch=True``) rather
                than leaving it at the ``None`` default. An explicit request on a non-Web-Mercator map is
                refused — the answer ``dashboard(widgets=("basemap",))`` already gives — while an
                unprompted one is dropped and logged, since nothing was asked for.

        Returns:
            A ``panel.widgets.Select`` over :data:`_BASEMAP_CHOICES`, or ``None`` when the display CRS is
            not EPSG:3857 and the switch was not requested — in which case the omission is logged rather
            than left to be guessed at.

        Raises:
            ValueError: when ``requested`` is true and the display CRS is not EPSG:3857; the message
                names the CRS the map actually uses.
        """
        if not same_crs(self.crs, 3857):
            if requested:
                self._require_web_mercator("layer_control basemap")
            from loguru import logger

            logger.info(
                f"layer_control(): basemap switch omitted — Bokeh renders tiles in EPSG:3857 only and "
                f"this map uses crs={self.crs!r}; build the map with InteractiveMap(crs=3857) for it."
            )
            return None
        return pn.widgets.Select(label="Basemap", options=_BASEMAP_CHOICES)

    def _compose_visible_layers(
        self,
        shown: list,
        op: float = 1.0,
        basemap: Any = None,
        always: Sequence[int] = (),
    ) -> Any:
        """Compose the layers named in ``shown`` into a styled overlay (the layer-control view).

        The opacity is applied over the style the builders recorded for **each** of those layers (read back
        through :meth:`~digitalearth.interactive.base.InteractiveMapBase.style_of`), so hiding and re-showing
        a layer cannot quietly drop its ``cmap``/``clim`` — nor repaint it in another layer's colours, which
        is what merging them into one spec did (#300).

        Args:
            shown: The ``"i: Type"`` labels of the visible layers (from the toggle widget).
            op: Opacity applied to colour-mapped layers.
            basemap: Provider name from the basemap ``Select``, or ``None`` to leave the basemap as built.
            always: Positions of the layers the control never offered — ``layer_control(layers=...)`` lists a
                subset, and a layer nobody can toggle is not a layer anybody hid, so it is composed in
                regardless of ``shown``. Empty when the control offered every layer.

        Returns:
            A blank ``hv.Overlay`` when nothing is shown and nothing is always-on, else the overlay of the
            chosen layers in draw order at opacity ``op``, over the chosen basemap when one is selected.
        """
        gv, hv = _require_holoviz()
        # The other widget path does this too: a deferred basemap is one of the layers, and composing
        # without flushing drew a map that never had one (review H9). `layer_control` has already flushed,
        # so this is a no-op there and the indices below still mean what the labels meant.
        self._flush_deferred_tiles()
        # Back into draw order, and de-duplicated: the toggles and the always-on set are two ways of naming
        # positions in `self.layers`, and the overlay has to be built bottom-first either way.
        indices = sorted(
            {int(label.split(":")[0]) for label in shown} | {int(i) for i in always}
        )
        chosen = [self.layers[index] for index in indices]
        if not chosen:
            return hv.Overlay([])
        overlay = self._compose(self._restyled_layers({"alpha": op}, layers=chosen))
        if basemap:
            overlay = self._with_basemap(
                overlay, basemap, context="layer_control basemap"
            )
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
