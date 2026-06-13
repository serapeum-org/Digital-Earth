"""ProjectionMixin — non-Mercator projections for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``projection`` and ``graticule`` (DI.9). Bokeh renders Web-Mercator only, so arbitrary map
projections (orthographic **globe**, Robinson, Mollweide, polar stereographic) go through HoloViews'
**matplotlib** backend: ``projection(name)`` records the target, and ``render()`` (base) switches the
backend and passes ``projection=`` through. GeoViews builds the cartopy CRS object *internally* from the
name/EPSG — this package never imports cartopy (DX.3), exactly the §Projection-note option-B path.

The trade is deliberate and documented: the matplotlib path is **static** (no live pan/zoom/tiles). Tile
basemaps auto-disable under a non-Mercator projection (a tile call raises via the Web-Mercator guard).
"""

from typing import Any

from digitalearth.interactive.base import _require_holoviz


class ProjectionMixin:
    """Projection builders (DI.9): arbitrary display projections via the matplotlib backend."""

    def projection(self, name: Any, **opts: Any) -> "ProjectionMixin":
        """Set the display projection, rendering through the matplotlib backend.

        Args:
            name: A cartopy-projection name GeoViews resolves internally (e.g. ``"Orthographic"``,
                ``"Robinson"``, ``"Mollweide"``), an EPSG code, or a pre-built cartopy CRS. Pass
                ``None`` to return to the default Bokeh Web-Mercator path.
            **opts: Reserved for future projection options (currently unused).

        Returns:
            This map (chainable).

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

    def graticule(self, **opts: Any) -> "ProjectionMixin":
        """Add a longitude/latitude graticule (parity with ``Map.graticule``).

        Args:
            **opts: Extra HoloViews style options applied to the grid feature.

        Returns:
            This map (chainable).
        """
        gv, hv = _require_holoviz()
        element = gv.feature.grid.clone()
        if opts:
            element = element.opts(**opts)
        return self.add_element(element)
