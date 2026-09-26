"""ProjectionMixin — non-Mercator projections for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``projection`` and ``graticule`` (DI.9). Bokeh renders Web-Mercator only, so arbitrary map
projections (orthographic **globe**, Robinson, Mollweide, polar stereographic) go through HoloViews'
**matplotlib** backend: ``projection(name)`` records the target, and ``render()`` (base) switches the
backend and passes ``projection=`` through. GeoViews builds the cartopy CRS object *internally* from the
name/EPSG — this package never imports cartopy (DX.3), exactly the §Projection-note option-B path.

The trade is deliberate and documented: the matplotlib path is **static** (no live pan/zoom/tiles). Tile
basemaps auto-disable under a non-Mercator projection (a tile call raises via the Web-Mercator guard).
"""

from math import isfinite
from typing import TYPE_CHECKING, Any, Optional, Self, Sequence, Tuple, Union

from digitalearth.base.spec import Bounds, LayerSpec, Symbology, Viewport
from digitalearth.interactive.base import (
    _require_holoviz,
    describe_opts,
    held_props,
)

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.interactive.base import InteractiveMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object

#: Graticule spacings (degrees) :meth:`ProjectionMixin.graticule` can honour. GeoViews draws the grid from
#: Natural Earth's pre-cut ``graticules_<n>`` line layers, which exist only at these steps and are always
#: **symmetric** (the same spacing in longitude and latitude) — so anything else is refused rather than
#: silently rounded to the nearest shipped layer.
_GRATICULE_STEPS = (1, 5, 10, 15, 20, 30)


def _padded(box: Bounds, fraction: float) -> Bounds:
    """Return `box` grown by `fraction` of its own span, or `box` itself for no padding.

    Args:
        box: The rectangle to grow.
        fraction: How much to grow it by, as a proportion of its width and height.

    Returns:
        The padded rectangle.

    Raises:
        ValueError: from :meth:`~digitalearth.base.spec.bounds.Bounds.padded`, for a fraction below ``-0.5``.
    """
    return box if not fraction else box.padded(fraction)


def _element_extent(element: Any) -> Optional[Tuple[float, float, float, float]]:
    """Return the region one HoloViews element covers, as ``(xmin, ymin, xmax, ymax)``.

    Asked of the element rather than of the data behind it: ``range`` is HoloViews' own answer for every
    element type, so an image, a scatter, a path and a caller's own object are all measured the same way and
    each reports where it was actually placed.

    Args:
        element: The element a layer composed into.

    Returns:
        The rectangle it covers, or ``None`` when it has no finite range to give — an element built from no
        data ranges to ``(None, None)``, and a frame cannot be set from that.
    """
    if element is None:
        return None
    xmin, xmax = element.range(0)
    ymin, ymax = element.range(1)
    edges = (xmin, ymin, xmax, ymax)
    if any(edge is None for edge in edges):
        return None
    floats = tuple(float(edge) for edge in edges)
    return floats if all(isfinite(edge) for edge in floats) else None


def draw_graticule(interactive_map: Any, _data: Any, layer: LayerSpec) -> Any:
    """Build the GeoViews graticule element for a described grid.

    A graticule draws from no data: GeoViews cuts it from Natural Earth's pre-made line layers, chosen by
    the step the caller asked for, which is why `_data` is unused.

    Args:
        interactive_map: The map being drawn, whose held values carry the caller's own keywords.
        _data: Unused — a graticule has no source.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer` holding the element.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    gv, _ = _require_holoviz()
    props = held_props(interactive_map, layer)
    step = int(props["step"])
    element = gv.feature.grid.clone()
    if step != 30:
        # cartopy's NaturalEarthFeature, reached through the element GeoViews already built rather
        # than through an `import cartopy` (DX.3 — the tier never imports cartopy itself).
        source = element.data
        element = element.clone(
            type(source)(source.category, f"graticules_{step}", source.scale)
        )
    opts = dict(props.get("opts") or {})
    if opts:
        element = element.opts(**opts)
    return DrawnLayer(element=element, style=opts)


class ProjectionMixin(_MixinBase):
    """Projection builders (DI.9): arbitrary display projections via the matplotlib backend.

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

    #: The region the last :meth:`set_bounds` call asked for, in the display CRS, or ``None`` while nothing
    #: has framed the map. Declared on the class rather than set in a constructor, because this mixin is
    #: composed into :class:`~digitalearth.interactive.map.InteractiveMap` and owns no ``__init__`` — the
    #: class attribute is the default every instance reads until :meth:`set_bounds` writes one of its own.
    _frame_bounds: Optional[Bounds] = None

    @property
    def viewport(self) -> Viewport:
        """Where the map is looking, as a value — including the region it has been framed on.

        Returns:
            The view :class:`~digitalearth.interactive.base.InteractiveMapBase` builds from the display CRS,
            carrying the region :meth:`set_bounds` asked for when there has been one. It still carries no
            centre or zoom: this tier is framed by an extent, not by a camera position.
        """
        view = super().viewport
        framed = self._frame_bounds
        return view if framed is None else view.framed(framed)

    def set_bounds(
        self,
        bounds: Optional[Union[Bounds, Sequence[float]]] = None,
        *,
        padding: float = 0.0,
    ) -> Self:
        """Frame the map on a region, or on everything it draws.

        Bokeh pans and zooms, so before this the view was whatever HoloViews autoranged to and there was no
        call a caller could make to open the map on a region — the tier had no framing method under any
        spelling. It still pans: this sets the *initial* view rather than locking it.

        Args:
            bounds: A :class:`~digitalearth.base.spec.bounds.Bounds` in **any** CRS — it is reprojected to
                the display CRS, which is the point of passing one — or a bare
                ``(west, south, east, north)`` sequence taken to be in the display CRS. That is the bbox
                order :meth:`~digitalearth.base.spec.bounds.Bounds.as_bbox`, the web tier's ``set_bounds``
                and pyramids all use; the static tier's matplotlib ordering is its own history, not a
                cross-tier convention.

                ``None`` (the default) **fits the map to its data**: the union of what the panel's data
                layers cover, read from the elements themselves. Only the ``data`` band counts — a tile
                basemap, a graticule and a coastline are drawn around the subject rather than being it, and
                GeoViews reports a ``Feature``'s range in degrees whatever the display CRS is, so unioning
                one in would be wrong in units as well as in extent.
            padding: Breathing room around the frame, as a fraction of its own span — ``0.05`` leaves a 5%
                margin. Note the **units are this tier's**: a HoloViews figure is framed by data-coordinate
                limits, so a proportion of the span is the only padding that means anything, where the web
                tier's ``padding`` is a number of screen pixels.

        Returns:
            The same map instance, so builder calls chain — which is what the Core declares and what the web
            and 3-D tiers already answer.

        Raises:
            ValueError: for a sequence that is not four values; when ``bounds`` is ``None`` and the map
                draws nothing with an extent, because silently doing nothing is indistinguishable from the
                call being dropped; or, from `Bounds`, for a non-finite edge or a ``padding`` below
                ``-0.5``, which would turn the frame inside out.

        Examples:
            - Open the map on a region rather than on whatever the data autoranges to:
                ```python
                >>> from digitalearth.interactive import InteractiveMap       # doctest: +SKIP
                >>> m = InteractiveMap(crs=4326).set_bounds((3.0, 50.0, 7.0, 54.0))  # doctest: +SKIP
                >>> m.viewport.bounds.as_bbox()                              # doctest: +SKIP
                [3.0, 50.0, 7.0, 54.0]

                ```
        """
        self._frame_bounds = self._frame_asked(bounds, padding)
        return self

    def _frame_asked(
        self,
        bounds: Optional[Union[Bounds, Sequence[float]]],
        padding: float,
    ) -> Bounds:
        """Resolve what a caller asked for into one rectangle in the display CRS, padded.

        Split from :meth:`set_bounds` because the two halves answer different questions: this one is *which
        rectangle*, its caller is *what to do with it*. Each of the three spellings pads exactly once, inside
        its own branch, so no route can pad twice and none can skip it.

        Args:
            bounds: What the caller passed — a `Bounds`, a ``(west, south, east, north)`` sequence, or
                ``None`` to fit the data.
            padding: The fraction to grow the rectangle by.

        Returns:
            The region to frame on, in the display CRS.

        Raises:
            ValueError: as described on :meth:`set_bounds`.
        """
        if bounds is None:
            return self._fitted_box(padding)
        if isinstance(bounds, Bounds):
            # to_crs is a no-op when the CRSs already match. Without it a rectangle that carries its CRS
            # would be trusted to be in the display one, which is the mistake Bounds exists to stop.
            return _padded(bounds.to_crs(self.crs), padding)
        values = [float(value) for value in bounds]
        if len(values) != 4:
            raise ValueError(
                f"set_bounds(bounds=...) takes (west, south, east, north); got {len(values)} values"
            )
        return _padded(Bounds(*values, self.crs), padding)

    def _fitted_box(self, padding: float) -> Bounds:
        """Return the region the panel's own data layers cover, padded.

        Args:
            padding: The fraction to grow the union by.

        Returns:
            The union, in the display CRS. The panel does the merging
            (:meth:`~digitalearth.base.spec.figure.PanelSpec.bounds_of`): which layers count is its layer
            list and which CRS they meet in is its view, and neither is this tier's to decide.

        Raises:
            ValueError: when no data layer had an extent to give.
        """
        panel = self.figure_spec.panels[0]
        box = panel.bounds_of(self._data_extents(), padding=padding)
        if box is None:
            raise ValueError(
                "set_bounds() has nothing to frame on: this map draws no data layer with an extent — a "
                "basemap, graticule or coastline is drawn around the subject rather than being it, and a "
                "hidden layer is not drawn at all. Add the data first, or pass (west, south, east, north)."
            )
        return box

    def _data_extents(self) -> dict:
        """Return what each of the panel's data layers covers, by layer id, in the display CRS.

        Read from the **elements**, not from the sources: a layer is placed where the reprojection left it,
        and that is the region the frame has to hold. An element HoloViews cannot range — one with no data
        in it — simply has no entry, which
        :meth:`~digitalearth.base.spec.figure.PanelSpec.bounds_of` reads as "no extent to give".

        Returns:
            ``{layer_id: Bounds}`` for the visible layers of the ``data`` band.
        """
        measured = {}
        for layer in self._layer_tree.layers:
            if self._renderer.band_for(layer) != "data":
                continue
            if not self._layer_tree.is_visible(layer.id):
                continue
            drawn = self._renderer.drawn.get(layer.id)
            covered = None if drawn is None else _element_extent(drawn.element)
            if covered is not None:
                measured[layer.id] = Bounds(*covered, self.crs)
        return measured

    def _projected(self, obj: Any) -> Any:
        """Return `obj` drawn in the view this map was asked for — its projection, and its frame.

        The one hook both render paths share (``render`` and the dashboard's widget builders each call it),
        which is why the frame goes on here rather than in ``render``: a framed map has to stay framed when
        it is composed into a dashboard.

        Args:
            obj: The composed figure.

        Returns:
            `obj` carrying the projection, from the base implementation, plus the ``xlim``/``ylim`` of the
            region :meth:`set_bounds` asked for. Unframed, it is whatever the base returned — this tier's
            default is the view the engine autoranges to, and an unasked-for limit would take that away.
        """
        projected = super()._projected(obj)
        framed = self._frame_bounds
        if framed is None:
            return projected
        xmin, xmax, ymin, ymax = framed.as_mpl()
        return projected.opts(xlim=(xmin, xmax), ylim=(ymin, ymax))

    def projection(self, name: Any, **opts: Any) -> Self:
        """Set the display projection, rendering through the matplotlib backend.

        Args:
            name: A cartopy-projection name GeoViews resolves internally (e.g. ``"Orthographic"``,
                ``"Robinson"``, ``"Mollweide"``), an EPSG code, or a pre-built cartopy CRS. Pass
                ``None`` to return to the default Bokeh Web-Mercator path.
            **opts: Reserved for future projection options (currently unused).

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when a tile basemap was already requested (tiles are Web-Mercator only and
                cannot compose with a non-Mercator projection).
            ImportError: when the ``interactive`` extra is not installed.
        """
        # Called for its actionable ImportError; the cartopy projection itself is resolved through the
        # modules GeoViews already imported (`_resolve_projection`).
        _require_holoviz()
        if name is None:
            self._projection = None
            return self
        if self._tiles_provider is not None or any(
            type(layer).__name__ in ("WMTS", "Tiles") for layer in self.layers
        ):
            raise ValueError(
                "projection() cannot compose with tile basemaps — Bokeh tiles render in "
                "Web-Mercator only. Drop tiles() to use a non-Mercator projection."
            )
        self._projection = self._resolve_projection(name)
        return self

    @staticmethod
    def _resolve_projection(name: Any) -> Any:
        """Resolve a projection ``name``/EPSG to a cartopy projection (no cartopy import here, DX.3).

        A pre-built cartopy CRS (or any non-str/int) passes straight through — the recommended path is
        to pass ``cartopy.crs.Robinson()`` directly. A **name** (``"Robinson"``/``"Orthographic"``) is
        resolved by ``getattr`` on the cartopy ``crs`` module that **GeoViews already imported**
        (reached via ``geoviews.util``/``hvplot.util`` — *not* an ``import cartopy`` in our code, so the
        DX.3 guard stays green). An **EPSG int** goes through ``hvplot.util.proj_to_cartopy``.

        Args:
            name: A projection name, an EPSG code, or a pre-built cartopy CRS.

        Returns:
            A cartopy projection object.

        Raises:
            ValueError: when a name cannot be resolved to a cartopy projection.
        """
        if not isinstance(name, (str, int)):
            return name
        if isinstance(name, int):
            import geoviews as gv

            return gv.util.process_crs(name)  # process_crs accepts EPSG codes
        import geoviews.util as _gvutil
        import hvplot.util as _hvutil

        ccrs = getattr(_gvutil, "ccrs", None) or getattr(_hvutil, "ccrs", None)
        factory = getattr(ccrs, name, None) if ccrs is not None else None
        if factory is None:
            raise ValueError(
                f"unknown projection {name!r}; pass a cartopy.crs name (e.g. 'Robinson', "
                "'Orthographic', 'Mollweide') or a pre-built cartopy projection object"
            )
        return factory()

    def graticule(
        self,
        lon_step: float = 30.0,
        lat_step: float = 30.0,
        *,
        spacing: Optional[float] = None,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Self:
        """Add a longitude/latitude graticule at the requested spacing (parity with ``Map.graticule``).

        The spacing arguments mirror the static ``Map.graticule(lon_step, lat_step)`` so the same call
        means the same thing on both tiers. GeoViews draws the grid from Natural Earth's pre-cut
        ``graticules_<n>`` line layers rather than generating meridians on the fly, so only the shipped,
        **symmetric** steps can be honoured: ``1``, ``5``, ``10``, ``15``, ``20`` and ``30`` degrees, with
        ``lon_step == lat_step``. Any other request raises instead of being quietly ignored.

        Args:
            lon_step: Meridian spacing in degrees; one of :data:`_GRATICULE_STEPS`.
            lat_step: Parallel spacing in degrees; must equal ``lon_step``.
            spacing: One step for both, for a caller who wants a square grid; it overrides the two
                above. The same **keyword** the static and web tiers take, so one call draws one grid on
                every tier that draws a graticule at all (#324) — but its **value** domain is this
                engine's, not theirs: only the shipped steps listed above are honoured, so
                ``spacing=7.5`` raises here while it draws on both of them (review R-L10).
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the map is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden — before, the flag fell through ``**opts`` to HoloViews, which hid the
                element while the figure went on calling it visible (#327).
            **opts: Extra HoloViews style options applied to the grid feature.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``lon_step`` and ``lat_step`` differ, or when the step is not one of the
                Natural-Earth graticule spacings GeoViews can draw.

        Examples:
            - A 10-degree graticule instead of the 30-degree default:
                ```python
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> m = InteractiveMap().graticule(lon_step=10, lat_step=10)    # doctest: +SKIP
                >>> m.layers[-1].data.name                                      # doctest: +SKIP
                'graticules_10'

                ```
        """
        _require_holoviz()
        # Validated here, because the message names the caller's own arguments. The grid itself is built by
        # `draw_graticule` from exactly what this records, so the figure describes the graticule rather
        # than holding one somebody else built.
        if spacing is not None:
            lon_step = lat_step = spacing
        step = self._graticule_step(lon_step, lat_step)
        held: dict = {}
        described_opts = describe_opts(held, opts)
        return self.add_layer(
            None,
            name=name,
            visible=visible,
            kind="graticule",
            held=held,
            symbology=Symbology(
                props={
                    "via": "graticule",
                    "step": int(step),
                    "opts": described_opts,
                }
            ),
        )

    @staticmethod
    def _graticule_step(lon_step: float, lat_step: float) -> int:
        """Validate the requested graticule spacing and return it as a Natural-Earth step.

        Args:
            lon_step: Meridian spacing in degrees.
            lat_step: Parallel spacing in degrees.

        Returns:
            int: the spacing naming the ``graticules_<n>`` Natural-Earth layer to draw.

        Raises:
            ValueError: when the two steps differ, or the step is not a shipped Natural-Earth spacing.
        """
        shipped = {float(known) for known in _GRATICULE_STEPS}
        for name, value in (("lon_step", lon_step), ("lat_step", lat_step)):
            if float(value) not in shipped:
                known = ", ".join(str(entry) for entry in _GRATICULE_STEPS)
                raise ValueError(
                    f"graticule() cannot honour {name}={value!r} — GeoViews draws Natural Earth's "
                    f"pre-cut graticule layers, which exist only at: {known} degrees"
                )
        if float(lon_step) != float(lat_step):
            raise ValueError(
                f"graticule() cannot honour lon_step={lon_step!r} with lat_step={lat_step!r} — GeoViews "
                "draws Natural Earth's pre-cut graticule layers, which are symmetric; pass the same "
                "spacing for both, or use the static tier for an asymmetric grid"
            )
        return int(float(lon_step))
