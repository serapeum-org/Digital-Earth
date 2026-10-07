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

from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from importlib.util import find_spec
from math import prod
from operator import mul as _mul
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from digitalearth.base.controls import (
    LAYER_CONTROLS,
    check_control_position,
    resolved_controls,
)
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


def _controls_named(
    controls: Sequence[str] | None,
    *,
    caller: str,
) -> tuple[tuple[str, ...], bool]:
    """Settle which controls a layer manager exposes, and whether the caller asked for them outright.

    One list says it on both tiers (#264): this tier once spelled the same request as one boolean per
    control, and the web tier had no way of spelling it at all.

    Args:
        controls: The caller's control names, or ``None`` for "not passed".
        caller: The public method the keyword was written on, for every message to quote.

    Returns:
        ``(controls, requested)`` — the control names to build, and whether the caller asked for them
        **outright**. The second half is what decides how a basemap switch on a non-Web-Mercator map is
        answered: a named control is a request, and is refused with the CRS; the unprompted default is
        furniture, and is dropped with a log line.

    Raises:
        ValueError: from :func:`~digitalearth.base.controls.resolved_controls`, for a name no tier has or
            one this tier cannot build.
    """
    if controls is None:
        return tuple(LAYER_CONTROLS), False
    return resolved_controls(controls, offered=LAYER_CONTROLS, caller=caller), True


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

#: The dashboard's declared widget vocabulary (IN-9): the names `dashboard(widgets=...)` accepts, in the
#: order they read. Held as data so a name outside it is refused with a did-you-mean rather than silently
#: dropped, and a new widget is one builder in `_dashboard_widget_builders`, not another `elif`.
_DASHBOARD_WIDGETS: tuple[str, ...] = ("cmap", "alpha", "basemap")

#: The widget set `dashboard`/`export_plan`/`save_app` default to when the caller names none. One constant so
#: the export-size pre-check in `save_app` cannot disagree with what `dashboard` actually builds (F6).
_DEFAULT_DASHBOARD_WIDGETS: tuple[str, ...] = ("cmap", "alpha")

#: The Panel templates `dashboard(template=...)` can host the app in (IN-9): the short name → the class on
#: `panel.template`. Held as data for the same reason the widgets are.
_DASHBOARD_TEMPLATES: dict[str, str] = {
    "fast": "FastListTemplate",
    "bootstrap": "BootstrapTemplate",
    "material": "MaterialTemplate",
    "vanilla": "VanillaTemplate",
}


@dataclass(frozen=True)
class ExportPlan:
    """What a ``save_app(embed=True)`` export will contain, modelled before it is written (IN-10).

    Panel's ``embed`` bakes a pre-rendered page for the cartesian product of the widgets' discrete states —
    a Select contributes one state per option, a continuous slider is *sampled* down to a few. The product
    grows fast, and past Panel's cap the export silently drops states. This states the shape up front: the
    per-widget state counts, their product, the cap, and whether the product is within it — so a caller finds
    out their app survives the trip *before* exporting rather than from a truncated file.

    Attributes:
        widgets: ``(name, state_count)`` per embedded widget, in order.
        total_states: The product of the state counts — the number of pages ``embed`` would bake.
        max_states: The cap the product is judged against (Panel warns and truncates past its own).

    Examples:
        - A colormap Select (8 options) and a sampled opacity slider (3) make 24 pre-rendered states:
            ```python
            >>> from digitalearth.interactive.dashboard import ExportPlan
            >>> plan = ExportPlan(widgets=(("cmap", 8), ("alpha", 3)), total_states=24, max_states=1000)
            >>> plan.total_states, plan.embeddable
            (24, True)

            ```
    """

    widgets: tuple[tuple[str, int], ...]
    total_states: int
    max_states: int

    @property
    def embeddable(self) -> bool:
        """Whether the export's state count is within the cap.

        Returns:
            ``True`` when :attr:`total_states` does not exceed :attr:`max_states`.
        """
        return self.total_states <= self.max_states

    @property
    def warning(self) -> str | None:
        """A one-line caution when the export would exceed the cap, else ``None``.

        Returns:
            The warning naming the counts and the cap, or ``None`` when the export is within it.
        """
        if self.embeddable:
            return None
        breakdown = " × ".join(f"{name}:{count}" for name, count in self.widgets)
        return (
            f"save_app(embed=True) would bake {self.total_states} states ({breakdown}), over the "
            f"{self.max_states} cap — Panel will truncate it. Drop a widget, or serve() the app instead."
        )


def _did_you_mean(kind: str, name: str, known: Sequence[str]) -> str:
    """Return a refusal message for an unknown name, with a difflib suggestion (IN-13).

    The interactive tier's own did-you-mean: hvPlot and HoloViews suggest the nearest option for an unknown
    key, and the tier's declared vocabularies — the dashboard widgets and templates — answer the same way
    rather than listing the choices and leaving the caller to spot their typo.

    Args:
        kind: What the name is (``"dashboard widget"``, ``"dashboard template"``), quoted in the message.
        name: The name the caller passed.
        known: The names that are valid, which the message lists and suggests from.

    Returns:
        A message naming the unknown value, the nearest match where there is one, and the valid set.
    """
    from difflib import get_close_matches

    match = get_close_matches(str(name), [str(k) for k in known], n=1)
    suggestion = f" (did you mean {match[0]!r}?)" if match else ""
    return f"unknown {kind} {name!r}{suggestion}; choose from {list(known)}"


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
        widgets: Sequence[str] = _DEFAULT_DASHBOARD_WIDGETS,
        sidebar: bool = True,
        title: str = "",
        template: str | None = None,
    ) -> Any:
        """Wrap the map and reactive widgets into a Panel layout.

        The returned object is a ``panel.viewable.Viewable``: a sidebar (or inline row) of widgets
        bound to the map's style, beside the live map. The widget set is a **declared vocabulary** (IN-9),
        :data:`_DASHBOARD_WIDGETS` — ``"cmap"`` (colormap selector), ``"alpha"`` (opacity slider),
        ``"basemap"`` (tile-provider selector, Web-Mercator-only) — so a name outside it is refused with a
        did-you-mean rather than silently dropped, and a new widget is one entry, not another ``elif``.

        Args:
            widgets: Which widgets to expose, in order.
            sidebar: Lay widgets out in a left sidebar (``True``) or a top row (``False``). Ignored when
                ``template`` is given — a template has its own sidebar.
            title: Dashboard title (falls back to the map's ``title``).
            template: A Panel template to host the app in (IN-9) — ``"fast"``, ``"bootstrap"``,
                ``"material"`` or ``"vanilla"``. ``None`` (default) returns the plain row/column layout.

        Returns:
            A ``panel.viewable.Viewable`` hosting the map + widgets — a template instance when ``template``
            is given, else the row/column layout.

        Raises:
            ValueError: for an unknown widget name (with a did-you-mean), an unknown ``template`` name, or
                when ``"basemap"`` is requested on a map whose display CRS is not EPSG:3857 (Bokeh renders
                tiles in Web Mercator only).

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
        if template is not None:
            return self._templated(pn, template, controls, panel_map, title)
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

    def _templated(
        self, pn: Any, template: str, controls: list, panel_map: Any, title: str
    ) -> Any:
        """Host the map and widgets in a named Panel template (IN-9).

        Args:
            pn: The imported panel module.
            template: A key of :data:`_DASHBOARD_TEMPLATES`.
            controls: The widget objects for the sidebar.
            panel_map: The reactive map pane for the main area.
            title: The template title (falls back to the map's ``title``).

        Returns:
            The instantiated Panel template.

        Raises:
            ValueError: for a template name outside the vocabulary, with a did-you-mean.
        """
        class_name = _DASHBOARD_TEMPLATES.get(template)
        if class_name is None:
            raise ValueError(
                _did_you_mean(
                    "dashboard template", template, sorted(_DASHBOARD_TEMPLATES)
                )
            )
        template_class = getattr(pn.template, class_name)
        return template_class(
            title=title or self.title or "Digital-Earth",
            sidebar=list(controls),
            main=[panel_map],
        )

    def _build_widgets(self, pn: Any, widgets: Sequence[str]) -> tuple:
        """Construct the requested widgets and the ``pn.bind`` keyword map from the declared vocabulary.

        The vocabulary (:data:`_DASHBOARD_WIDGETS`) is consulted as data rather than branched on (IN-9), so
        a name outside it is refused with a did-you-mean (IN-13) and adding a widget is adding a builder, not
        another ``elif``.

        Args:
            pn: The imported panel module.
            widgets: Widget names to build.

        Returns:
            ``(controls, bindings)`` — the widget objects (for layout) and the ``name -> widget``
            map passed to ``pn.bind``.

        Raises:
            ValueError: for an unknown widget name (with a did-you-mean), or when ``"basemap"`` is requested
                on a non-Web-Mercator map (Bokeh renders tiles in EPSG:3857 only).
        """
        builders = self._dashboard_widget_builders(pn)
        controls, bindings = [], {}
        for name in widgets:
            # The vocabulary consulted is the declared :data:`_DASHBOARD_WIDGETS` tuple, in its own order, so
            # the "choose from" list and the did-you-mean read the data rather than the builder dict's keys
            # (F3). The builders cover exactly that tuple — pinned by a test — so the lookup below is total.
            if name not in _DASHBOARD_WIDGETS:
                raise ValueError(
                    _did_you_mean("dashboard widget", name, _DASHBOARD_WIDGETS)
                )
            widget = builders[name]()
            controls.append(widget)
            bindings[name] = widget
        return controls, bindings

    def _dashboard_widget_builders(self, pn: Any) -> dict:
        """Return the builder for each name in the declared vocabulary :data:`_DASHBOARD_WIDGETS` (IN-9).

        Keyed by exactly the names in :data:`_DASHBOARD_WIDGETS` (pinned by
        ``test_the_widget_vocabulary_is_declared_data``), so the constant is the single source of truth the
        validation and did-you-mean read, and this just says how each declared name is built.

        Args:
            pn: The imported panel module.

        Returns:
            A ``{name: () -> widget}`` mapping over :data:`_DASHBOARD_WIDGETS`. ``"basemap"`` carries the
            Web-Mercator guard in its builder, so it refuses rather than draws a misaligned tile layer.
        """

        def _basemap() -> Any:
            # Refuse rather than draw a misaligned basemap: the widget swaps in a Bokeh tile layer, which
            # only registers with the data on a Web-Mercator map.
            self._require_web_mercator("dashboard basemap")
            return pn.widgets.Select(label="Basemap", options=_BASEMAP_CHOICES)

        return {
            "cmap": lambda: pn.widgets.Select(label="Colormap", options=_CMAP_CHOICES),
            "alpha": lambda: pn.widgets.FloatSlider(
                label="Opacity", start=0.0, end=1.0, value=1.0
            ),
            "basemap": _basemap,
        }

    def _restyled_layers(
        self, overrides: dict, layers: Sequence[Any] | None = None
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

    def _reconcile_view(
        self,
        layers: Sequence[Any],
        overrides: dict,
        basemap: Any = None,
        *,
        basemap_context: str = "dashboard basemap",
    ) -> Any:
        """Recompose ``layers`` with ``overrides`` applied, reusing every untouched element (IN-2).

        **The one composition path every widget event takes.** The dashboard's cmap/alpha/basemap widgets and
        the layer switcher's visibility/opacity/basemap widgets used to each rebuild the overlay their own
        way — ``_render_with_overrides`` and ``_compose_visible_layers`` grew a reduce apiece beside
        ``render``'s, so a style fix had to be made in three places and a divergence between them was a bug
        waiting (detail doc gap 2). They route through here now, so there is one recipe: flush the deferred
        basemap, restyle only the layers a widget actually claims, compose, project, and drop the chosen
        basemap underneath.

        This is a reconcile rather than a rebuild: :meth:`_restyled_layers` re-``.opts()``s only the
        colour-mapped (and alpha-only) layers a widget value reaches and hands **the same element object**
        back for every other layer, so a slider tick on a two-layer map re-styles the one raster and leaves
        the points layer's element — and the hover tools and colorbar filed against it — untouched. When no
        override is in play the layers pass through with their identity intact, so a basemap switch alone
        never re-styles a thing.

        Args:
            layers: The elements to compose, bottom first — every layer for the dashboard, the chosen subset
                for the switcher.
            overrides: The widget values to apply over each layer's own recorded style (``cmap``/``alpha``);
                empty leaves every element exactly as it was.
            basemap: Provider name from a basemap ``Select``, or ``None`` to leave the basemap as built.
            basemap_context: The entry point quoted in the Web-Mercator refusal, so a basemap switch names
                the widget the caller actually touched.

        Returns:
            The composed, projected HoloViews object, over the chosen basemap when one is selected.
        """
        # Before the layers are read: a deferred basemap is one of them, so composing without flushing drew a
        # map that never had one (review H9). The theme is set on the renderer here too, so a dashboard draws
        # under the same theme a bare render would (IN-18).
        self._flush_deferred_tiles()
        self._apply_theme()
        # Restyle only when a widget value is in play; otherwise the elements pass through untouched, which is
        # what makes a basemap-only change a reconcile rather than a rebuild. `.opts()` on an overlay applies
        # per element *type*, so the restyle happens per layer before the compose, not after it (#300).
        restyled = (
            self._restyled_layers(overrides, layers) if overrides else list(layers)
        )
        obj = self._projected(self._compose(restyled))
        if basemap:
            obj = self._with_basemap(obj, basemap, context=basemap_context)
        return obj

    def _render_with_overrides(self, values: dict) -> Any:
        """Re-render the whole map, applying the dashboard widgets' values as style overrides.

        A thin reading of the widget values into the one :meth:`_reconcile_view` path (IN-2).

        Args:
            values: Widget values keyed by widget name (``cmap``/``alpha``/``basemap``).

        Returns:
            The composed HoloViews object with the overrides applied — each colour-mapped layer redrawn with
            the widget values over **its own** recorded style, every other element reused, over the chosen
            tile basemap.
        """
        overrides: dict = {}
        if values.get("cmap"):
            overrides["cmap"] = values["cmap"]
        if values.get("alpha") is not None:
            overrides["alpha"] = values["alpha"]
        return self._reconcile_view(self.layers, overrides, values.get("basemap"))

    def export_plan(
        self,
        *,
        widgets: Sequence[str] = _DEFAULT_DASHBOARD_WIDGETS,
        max_states: int = 1000,
        max_opts: int = 3,
    ) -> ExportPlan:
        """Model what a ``save_app(embed=True)`` of these widgets would bake, before writing it (IN-10).

        Each widget's discrete state count is read the way Panel's ``embed`` reads it — a Select contributes
        one state per option, a continuous slider is sampled to ``max_opts`` — and their product is the
        number of pages the export would pre-render. :class:`ExportPlan` says whether that product is within
        ``max_states``, so a caller can decide between an offline export and a served app *before* the file
        is written rather than after it is truncated.

        Args:
            widgets: The widgets the export would carry (as :meth:`dashboard` takes them).
            max_states: The cap the baked-state count is judged against.
            max_opts: How many samples a continuous slider contributes under ``embed`` (Panel's own default
                is 3).

        Returns:
            The :class:`ExportPlan` for this widget set.

        Raises:
            ValueError: for an unknown widget name (with a did-you-mean), or ``"basemap"`` on a
                non-Web-Mercator map — the same vocabulary :meth:`dashboard` enforces.
        """
        pn = _require_panel()
        controls, _ = self._build_widgets(pn, widgets)
        counts: list[tuple[str, int]] = []
        for name, widget in zip(widgets, controls):
            options = getattr(widget, "options", None)
            # A Select carries its discrete options; a continuous slider has none, so embed samples it.
            counts.append((name, len(options) if options is not None else max_opts))
        total = prod(count for _, count in counts) if counts else 1
        return ExportPlan(
            widgets=tuple(counts), total_states=total, max_states=max_states
        )

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

        Before writing, the export is modelled with :meth:`export_plan` (IN-10): if ``embed`` would bake more
        states than the cap, the oversize is logged with the per-widget breakdown — so a truncated file is
        explained rather than discovered. :meth:`export_plan` lets a caller check this before calling here.

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
        if embed:
            widgets = kwargs.get("widgets", _DEFAULT_DASHBOARD_WIDGETS)
            plan = self.export_plan(widgets=widgets)
            if plan.warning is not None:
                from loguru import logger

                logger.warning(plan.warning)
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

    def _control_layer_indices(self, layers: Sequence[str] | None) -> tuple[int, ...]:
        """Return the positions of the layers a control offers, in draw order.

        Positions, because the human-readable toggle label still shows the build-time index (``"0: Image"``);
        the control's *value*, though, is the layer id the position maps to (IN-1), so
        :meth:`_compose_visible_layers` reads ids, never a parsed position. ``self.layers`` and
        :attr:`~digitalearth.interactive.base.InteractiveMapBase.layer_ids` are built in step — each element
        is inserted at the position its id holds in the tree — so one indexes the other.

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
        self, pn: Any, wanted: Sequence[str], *, label_to_id: dict, requested: bool
    ) -> dict:
        """Build the widgets a layer control exposes, keyed by the control each answers to.

        Args:
            pn: The imported panel module.
            wanted: The controls to build, from :data:`~digitalearth.base.controls.LAYER_CONTROLS`.
            label_to_id: The ``"i: Type"`` label → layer **id** mapping for each offered layer, in draw
                order. The toggle group lists the labels but carries the ids as its values, so toggling,
                reordering and removal stay keyed to the layer rather than to the position it held when the
                control was built (IN-1).
            requested: Whether the controls were named outright, which decides how a basemap switch on a
                non-Web-Mercator map is answered — see :meth:`_basemap_select`.

        Returns:
            ``{control: widget}`` in build order, so the layout lays them out in it. ``"basemap"`` is absent
            when the switch was offered unprompted and the display CRS cannot take it.
        """
        # Options are a `{label: id}` mapping, so the group's *value* is the list of layer ids rather than
        # the list of labels — the toggle carries the id, and `_compose_visible_layers` never has to parse a
        # position back out of a label that a later reorder or removal would have invalidated (IN-1).
        built = {
            "visibility": pn.widgets.CheckBoxGroup(
                value=list(label_to_id.values()), options=dict(label_to_id)
            )
        }
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
        layers: Sequence[str] | None = None,
        position: str = "top-right",
        controls: Sequence[str] | None = None,
        reorder: bool = False,
    ) -> Self:
        """Build a layer manager — per-layer visibility, an opacity slider, a basemap switch (DI.13).

        The three keywords are the ones the Tier-2 contract declares and the web tier answers to as well —
        the layers to include, the position, the controls to expose — and **the map comes back**, so one call
        adds a layer control on either tier (#264). The panel itself is held on the map and read back through
        :attr:`layer_control_panel`; before, this method returned it, and so ended any chain it appeared in.

        The toggle order follows **draw** order — the order `layers` is in, which is the band each layer's
        kind declares and then the order they were built in within that band, so a basemap added last is
        still the first toggle. ``reorder=True`` adds a ``▲``/``▼`` button per offered layer that moves it
        within its band (``move_layer(layer_id, index)``) and re-stacks the overlay live (IN-1); it is **off
        by default** because the buttons are furniture most maps do not need. It is still not one of the
        ``controls``: those are the vocabulary the two tiers share, and only this tier builds the reorder
        buttons, so advertising it there would make the web tier refuse a name this one honours.

        The control is keyed by layer **id**, not by the position a layer held when it was built: toggling,
        reordering and removing all address the layer, so the switch keeps working after the overlay is
        re-stacked or a layer is taken off the map.

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
            reorder: Add a ``▲``/``▼`` reorder button per offered layer (IN-1). ``False`` (default) builds
                visibility/opacity/basemap only.

        Returns:
            The same map instance, so builder calls chain. The panel is :attr:`layer_control_panel`.

        Raises:
            ValueError: when ``position`` is not one of the four corners; when ``controls`` names something
                outside the shared vocabulary, drops ``"visibility"``, or names a control this tier cannot
                build; when there are no layers to control; when a ``layers`` id is not on this map; or when
                ``"basemap"`` is named on a map whose display CRS is not EPSG:3857 (Bokeh renders tiles in
                Web Mercator only).

        Examples:
            - One toggle per registered layer, labelled by index and element type, in draw order, carrying
              each layer's id as its value — and the map comes back, so the control is part of the chain
              rather than the end of it:
                ```python
                >>> from pyramids.dataset import Dataset                       # doctest: +SKIP
                >>> from pyramids.feature import FeatureCollection             # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")       # doctest: +SKIP
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")  # doctest: +SKIP
                >>> m = InteractiveMap().field(dem).points(fc).layer_control()  # doctest: +SKIP
                >>> list(m.layer_control_panel[1][0].options)                  # doctest: +SKIP
                ['0: Image', '1: Points']

                ```
            - ``reorder=True`` adds the move buttons, so the overlay can be re-stacked from the control:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> m = InteractiveMap().field(dem).points(fc).layer_control(reorder=True)  # doctest: +SKIP
                >>> bool(m.layer_control_panel.select(type(m.layer_control_panel[1][-1][0])))  # doctest: +SKIP
                True

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
        check_control_position(position)
        wanted, requested = _controls_named(
            controls, caller="InteractiveMap.layer_control()"
        )
        if not self.layers:
            raise ValueError(
                "layer_control() needs at least one layer — add a builder call first"
            )
        offered = self._control_layer_indices(layers)
        # Label → id, in draw order. The label still shows the build-time position for a human to read, but
        # the control carries the **id**: a toggle, a reorder or a removal addresses the layer, not the
        # position it happened to hold when the control was built (IN-1).
        offered_ids = [self.layer_ids[index] for index in offered]
        label_to_id = {
            f"{index}: {type(self.layers[index]).__name__}": layer_id
            for index, layer_id in zip(offered, offered_ids)
        }
        widgets = self._layer_control_widgets(
            pn, wanted, label_to_id=label_to_id, requested=requested
        )
        # A layer nobody offered is not a layer the viewer chose to hide, so it is composed in whatever the
        # toggles say — the same thing `layers=` means on the web tier, where the switch lists a subset of a
        # map that still draws all of it. Named by id, so a reorder cannot turn "always on" into a different
        # layer than the one the caller left off the switch.
        offered_set = set(offered_ids)
        bound: dict = {
            "shown": widgets["visibility"],
            "always": tuple(
                layer_id for layer_id in self.layer_ids if layer_id not in offered_set
            ),
        }
        if "opacity" in widgets:
            bound["op"] = widgets["opacity"]
        if "basemap" in widgets:
            bound["basemap"] = widgets["basemap"]
        view = pn.bind(self._compose_visible_layers, **bound)
        controls_built = list(widgets.values())
        if reorder:
            # Reordering is `move_layer(layer_id, index)`, which this tier has; what no tier had was a
            # widget to drive it (#242). A pair of up/down buttons per offered layer is that widget: each
            # click moves the layer within its band and re-triggers the composed view, so the overlay
            # re-stacks live. The toggles keep working across the move because they carry ids, not positions.
            controls_built.append(
                self._reorder_controls(pn, offered_ids, widgets["visibility"])
            )
        # The corner, as far as a row of two cells can honour one: the side is the order of the cells, and
        # the top/bottom half is the column's own alignment across the row.
        column = pn.Column(
            *controls_built, align="start" if position.startswith("top") else "end"
        )
        body = pn.panel(view)
        self._layer_control_panel = (
            pn.Row(column, body) if position.endswith("left") else pn.Row(body, column)
        )
        return self

    def _reorder_controls(
        self, pn: Any, offered_ids: Sequence[str], trigger: Any
    ) -> Any:
        """Build the per-layer up/down reorder buttons for ``layer_control(reorder=True)`` (IN-1).

        Each offered layer gets a ``▲``/``▼`` pair whose click moves it one place within its band
        (:meth:`~digitalearth.interactive.base.InteractiveMapBase.move_layer`) and then re-triggers the
        composed view by poking ``trigger`` — the visibility group the bound compose function reads — so the
        overlay re-stacks without rebuilding the control. A move past either end is a no-op: the top end
        overshoots the list and ``move_layer`` raises ``IndexError``, which is caught; the bottom end would
        reach index ``-1``, which ``move_layer`` instead *wraps* to the top, so a sub-zero target is refused
        explicitly before the call (F1) rather than left to a guard that never fires for it.

        Args:
            pn: The imported panel module.
            offered_ids: The ids the control offers, in draw order.
            trigger: The widget whose ``value`` the composed view is bound to; poked after each move so the
                view recomposes in the new order.

        Returns:
            A ``panel.Column`` of one ``▲``/``▼`` button row per offered layer, labelled by id.
        """

        def _mover(layer_id: str, delta: int) -> Any:
            """Return the click handler that nudges ``layer_id`` by ``delta`` places in draw order.

            Args:
                layer_id: The layer the button moves.
                delta: ``-1`` to move down the overlay (drawn earlier), ``+1`` to move up (drawn later).

            Returns:
                A ``callback(event)`` for ``Button.on_click``.
            """

            def _move(_event: Any) -> None:
                """Move the layer and re-trigger the composed view, ignoring a move out of its band.

                Args:
                    _event: The Panel button event, unused — the layer and direction are captured above.
                """
                order = self.layer_ids
                if layer_id not in order:  # removed since the control was built
                    return
                target = order.index(layer_id) + delta
                # Refuse a sub-zero target explicitly: `move_layer` accepts a negative index and wraps it
                # (`index % count`) to the top of the band, so `-1` on the bottom layer would invert the
                # overlay instead of doing nothing (F1). The top end raises IndexError and is caught below.
                if target < 0:
                    return
                try:
                    self.move_layer(layer_id, target)
                except IndexError:  # already at the top of its band — nothing to do
                    return
                trigger.param.trigger("value")

            return _move

        rows = []
        for layer_id in offered_ids:
            up = pn.widgets.Button(label=f"▲ {layer_id}", width=110)
            down = pn.widgets.Button(label=f"▼ {layer_id}", width=110)
            up.on_click(_mover(layer_id, 1))
            down.on_click(_mover(layer_id, -1))
            rows.append(pn.Row(up, down))
        return pn.Column(*rows)

    def _basemap_select(self, pn: Any, *, requested: bool) -> Any:
        """Build the layer-control basemap ``Select``, or ``None`` on a non-Web-Mercator map.

        Args:
            pn: The imported panel module.
            requested: Whether the caller named ``"basemap"`` in ``controls`` rather than leaving the
                default. An explicit request on a non-Web-Mercator map is
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
        shown: Sequence[str],
        op: float = 1.0,
        basemap: Any = None,
        always: Sequence[str] = (),
    ) -> Any:
        """Compose the layers named in ``shown`` into a styled overlay (the layer-control view).

        The opacity is applied over the style the builders recorded for **each** of those layers (read back
        through :meth:`~digitalearth.interactive.base.InteractiveMapBase.style_of`), so hiding and re-showing
        a layer cannot quietly drop its ``cmap``/``clim`` — nor repaint it in another layer's colours, which
        is what merging them into one spec did (#300).

        Args:
            shown: The **ids** of the visible layers (the toggle widget's values — IN-1). An id no longer on
                the map (removed since the control was built) is skipped rather than raising.
            op: Opacity applied to colour-mapped layers.
            basemap: Provider name from the basemap ``Select``, or ``None`` to leave the basemap as built.
            always: Ids of the layers the control never offered — ``layer_control(layers=...)`` lists a
                subset, and a layer nobody can toggle is not a layer anybody hid, so it is composed in
                regardless of ``shown``. Empty when the control offered every layer.

        Returns:
            A blank ``hv.Overlay`` when nothing is shown and nothing is always-on, else the overlay of the
            chosen layers in draw order at opacity ``op``, over the chosen basemap when one is selected.
        """
        gv, hv = _require_holoviz()
        # Ids → current positions, read fresh each call so a reorder since the control was built is honoured.
        # De-duplicated and sorted into draw order: the toggles and the always-on set are two ways of naming
        # layers, and the overlay has to be built bottom-first either way. The flush a stale basemap needs is
        # inside `_reconcile_view`, the one composition path this now shares with the dashboard widgets (IN-2).
        order = self.layer_ids
        wanted_ids = {layer_id for layer_id in shown} | {
            layer_id for layer_id in always
        }
        indices = sorted(
            order.index(layer_id) for layer_id in wanted_ids if layer_id in order
        )
        chosen = [self.layers[index] for index in indices]
        if not chosen:
            return hv.Overlay([])
        return self._reconcile_view(
            chosen, {"alpha": op}, basemap, basemap_context="layer_control basemap"
        )

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
                "attribute_table(linked=True) is not implemented — this Tabulator is disabled=True, so it "
                "cannot emit a selection. For brushing that links the map to its data use "
                "cross_filter(...), whose link_selections instance carries selection_expr back into Python, "
                "or annotate(...) for an editable attribute table over a drawn layer. Pass linked=False "
                "(the default) for the read-only attribute view."
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

    def share_url(self, base_url: str = "") -> str:
        """Encode the **whole figure** into a shareable URL, not four hardcoded names (IN-14).

        Where :meth:`share` syncs a fixed handful of view parameters into a running server's URL, this
        serialises the entire :attr:`~digitalearth.interactive.base.InteractiveMapBase.figure_spec` — every
        layer, its source, its symbology and the view — into a URL-safe ``?state=`` blob, so a link
        reconstructs the map exactly through
        :meth:`~digitalearth.interactive.base.InteractiveMapBase.from_share_url`. It needs no server, which is
        what makes it the general answer the four-name ``share`` is not.

        The figure must be one that can be *written*: a layer built from an in-memory ``FeatureCollection`` or
        dataset has an ``object:`` source that only this process can open, and
        :meth:`~digitalearth.base.spec.FigureSpec.to_dict` refuses it — give the builders a path or URL for a
        figure that can be shared (the same limit :meth:`save`-to-figure has).

        Args:
            base_url: An optional base to hang the query on, e.g. ``"https://host/app"``. Empty returns the
                query string alone (``"?state=…"``).

        Returns:
            The URL carrying the encoded figure.

        Raises:
            ValueError: when the figure holds an in-memory (``object:``) source that cannot be serialised.

        Examples:
            - A sourceless map round-trips through the URL it produces:
                ```python
                >>> from digitalearth.interactive import InteractiveMap
                >>> url = InteractiveMap(crs=4326).share_url()
                >>> url.startswith("?state=")
                True
                >>> InteractiveMap.from_share_url(url).crs
                4326

                ```
        """
        import base64
        import json

        state = self.figure_spec.to_dict()
        blob = base64.urlsafe_b64encode(json.dumps(state).encode("utf-8")).decode(
            "ascii"
        )
        # Strip a trailing separator first, so a base already ending in `?` or `&` does not produce `?&state=`
        # or `&&state=` (L1); the separator is then `&` only when a real query already follows the `?`.
        base = base_url.rstrip("?&")
        separator = "&" if "?" in base else "?"
        return f"{base}{separator}state={blob}"
