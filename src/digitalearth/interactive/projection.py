"""ProjectionMixin — non-Mercator projections for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``projection`` and ``graticule`` (DI.9). Bokeh renders Web-Mercator only, so arbitrary map
projections (orthographic **globe**, Robinson, Mollweide, polar stereographic) go through HoloViews'
**matplotlib** backend: ``projection(name)`` records the target, and ``render()`` (base) switches the
backend and passes ``projection=`` through. GeoViews builds the cartopy CRS object *internally* from the
name/EPSG — this package never imports cartopy (DX.3), exactly the §Projection-note option-B path.

The trade is deliberate and documented: the matplotlib path is **static** (no live pan/zoom/tiles). Tile
basemaps auto-disable under a non-Mercator projection (a tile call raises via the Web-Mercator guard).
"""

from typing import TYPE_CHECKING, Any, Self

from digitalearth.interactive.base import _require_holoviz

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.interactive.base import InteractiveMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object

#: Graticule spacings (degrees) :meth:`ProjectionMixin.graticule` can honour. GeoViews draws the grid from
#: Natural Earth's pre-cut ``graticules_<n>`` line layers, which exist only at these steps and are always
#: **symmetric** (the same spacing in longitude and latitude) — so anything else is refused rather than
#: silently rounded to the nearest shipped layer.
_GRATICULE_STEPS = (1, 5, 10, 15, 20, 30)


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
        """
        gv, hv = _require_holoviz()
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
        self, lon_step: float = 30.0, lat_step: float = 30.0, **opts: Any
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
        gv, hv = _require_holoviz()
        step = self._graticule_step(lon_step, lat_step)
        element = gv.feature.grid.clone()
        if step != 30:
            # cartopy's NaturalEarthFeature, reached through the element GeoViews already built rather
            # than through an `import cartopy` (DX.3 — the tier never imports cartopy itself).
            source = element.data
            element = element.clone(
                type(source)(source.category, f"graticules_{step}", source.scale)
            )
        if opts:
            element = element.opts(**opts)
        return self.add_element(
            element,
            kind="graticule",
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
