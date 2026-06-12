"""InteractiveMapBase — the core HoloViz plumbing the interactive capability mixins build on.

``InteractiveMapBase`` owns the layer registry (HoloViews elements in add order), the display-CRS
reproject-through-pyramids plumbing, and the render/save/show lifecycle. Capability mixins (raster, vector,
big-data, temporal, decoration, interaction, projection, animation, dashboard) live in sibling modules and add
``image()`` / ``points()`` / … builder methods that call ``self.add_element(...)``; the public
:class:`digitalearth.interactive.map.InteractiveMap` composes the base with those mixins — exactly mirroring the
2-D ``Map(GeoLayerBase, RasterMixin, …)`` and 3-D ``Scene3D(Scene3DBase, TerrainMixin, …)`` patterns.

HoloViz is a **renderer, not a GIS engine**: elements are built from pyramids-sourced numpy / GeoDataFrames
(never xarray/rasterio/cartopy — see the tier's HARD RULE, enforced by ``tests/test_no_competitor_imports.py``).
All CRS/reproject work happens upstream in pyramids (``Dataset.to_crs``) *before* an element is built, so the
elements are already in the display CRS (default EPSG:3857 — the only CRS Bokeh tiles render).

Unlike the 3-D tier, the engine import is **lazy**: ``import digitalearth.interactive`` works without the
``interactive`` extra installed; only calling a builder/render method raises an actionable ``ImportError``.
"""

from functools import reduce
from operator import mul
from typing import Any, List, Optional

from digitalearth.sources import get_source
from digitalearth.sources.source import Source

#: The pip extra / pixi env that provides the HoloViz engine, quoted in the lazy-import error.
_INSTALL_HINT = (
    "the interactive tier needs the HoloViz stack (geoviews/holoviews/datashader/panel). "
    "Install it with `pip install 'digitalearth[interactive]'` "
    "(or, in this repo, `pixi install -e interactive`)."
)


def _require_holoviz() -> tuple:
    """Import and return ``(geoviews, holoviews)``, raising an actionable error when absent.

    The single lazy-import choke point every engine-touching method calls first, so that
    ``import digitalearth.interactive`` itself never needs the optional ``interactive`` extra.

    Returns:
        tuple: the imported ``(geoviews, holoviews)`` modules.

    Raises:
        ImportError: when the HoloViz stack is not installed, with the install command in the message.
    """
    try:
        import geoviews as gv
        import holoviews as hv
    except ImportError as err:
        raise ImportError(_INSTALL_HINT) from err
    return gv, hv


class InteractiveMapBase:
    """Core host: ordered HoloViews-element registry + display CRS + render/save lifecycle.

    Args:
        crs: Display CRS as an EPSG integer. Default ``3857`` (Web Mercator — the only CRS Bokeh
            tile basemaps render); use ``4326`` for a non-tiled Plate-Carrée map.
        width: Frame width in pixels for the rendered Bokeh plot.
        height: Frame height in pixels for the rendered Bokeh plot.
        tiles: Optional tile-provider name drawn beneath the data layers (resolved by the
            decoration mixin at render time), or ``None`` for no basemap.
        title: Plot title.

    Attributes:
        layers: Registered HoloViews/GeoViews elements, in add (= overlay) order.
    """

    def __init__(
        self,
        *,
        crs: int = 3857,
        width: int = 700,
        height: int = 500,
        tiles: Optional[str] = None,
        title: str = "",
    ):
        self.crs = crs
        self.width = width
        self.height = height
        self.tiles = tiles
        self.title = title
        self.layers: List[Any] = []

    def add_element(self, element: Any) -> "InteractiveMapBase":
        """Register a HoloViews/GeoViews ``element`` as a layer and return ``self`` (chainable).

        The low-level entry point the capability mixins build on — every builder method ends here.

        Args:
            element: Any HoloViews/GeoViews element (or overlay-able object).

        Returns:
            This map, so builder calls chain: ``m.image(dem).tiles().coastlines()``.
        """
        self.layers.append(element)
        return self

    def _needs_reproject(self, data: Any) -> bool:
        """Whether ``data`` must be reprojected (via pyramids) to the display CRS.

        Args:
            data: A pyramids object exposing ``.epsg`` (``Dataset``/``FeatureCollection``).

        Returns:
            ``False`` only when the display CRS is an ``int`` equal to ``data.epsg``; ``True`` otherwise.
        """
        return not (
            isinstance(self.crs, int) and getattr(data, "epsg", None) == self.crs
        )

    def _to_display_source(self, data: Any, *, band: int = 1) -> Source:
        """Reproject ``data`` to the display CRS through pyramids and wrap it as a :class:`Source`.

        The single display-CRS choke point every raster/vector builder calls (settling the tier's
        projection decision, option A: **pre-reproject in pyramids** — no cartopy anywhere). Inputs already
        in the display CRS pass through untouched; everything else goes through ``data.to_crs(self.crs)``
        (pyramids' warp for rasters, pyramids' ``FeatureCollection`` reprojection for vectors).

        Args:
            data: A pyramids ``Dataset`` / ``FeatureCollection`` (anything ``get_source`` accepts).
                Bare numpy arrays / ``Source`` objects pass straight through to extraction.
            band: 1-based band to extract for raster inputs.

        Returns:
            Source: the display-CRS view (``z``/``x``/``y``/``crs``/``metadata``).
        """
        if isinstance(data, Source):
            return data
        if (
            hasattr(data, "epsg")
            and hasattr(data, "to_crs")
            and self._needs_reproject(data)
        ):
            data = data.to_crs(self.crs)
        return get_source(data, band=band)

    def render(self) -> Any:
        """Compose the registered layers into one HoloViews object (overlaid with ``*``).

        Returns:
            The single element when one layer is registered, an ``hv.Overlay`` of all layers in add
            order otherwise (an empty map renders as a blank ``hv.Overlay``).

        Raises:
            ImportError: when the ``interactive`` extra is not installed.
        """
        gv, hv = _require_holoviz()
        if not self.layers:
            return hv.Overlay([])
        if len(self.layers) == 1:
            return self.layers[0]
        return reduce(mul, self.layers)

    def save(self, path: str, **kwargs: Any) -> str:
        """Save the composed map — interactive HTML (Bokeh) or a raster via the matplotlib backend.

        Args:
            path: Output file. ``*.html`` writes a self-contained interactive Bokeh page; any other
                suffix (``.png``/``.svg``/…) renders through HoloViews' matplotlib backend (headless,
                no browser/selenium needed).
            **kwargs: Forwarded to :func:`holoviews.save` (e.g. ``fmt``, ``dpi``).

        Returns:
            The ``path`` written.

        Raises:
            ImportError: when the ``interactive`` extra is not installed.
        """
        gv, hv = _require_holoviz()
        obj = self.render()
        backend = "bokeh" if str(path).lower().endswith(".html") else "matplotlib"
        hv.save(obj, path, backend=backend, **kwargs)
        return str(path)

    def show(self) -> Any:
        """Render the map and display it inline when IPython is available.

        Returns:
            The composed HoloViews object (which a notebook front-end renders richly).

        Raises:
            ImportError: when the ``interactive`` extra is not installed.
        """
        obj = self.render()
        try:
            from IPython.display import display

            display(obj)
        except (
            ImportError
        ):  # plain-script use: returning the object is all there is to show
            pass
        return obj

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """Render the map inline in notebooks by delegating to the composed HoloViews object.

        Returns:
            The mimebundle of the rendered object, or an empty dict when the engine is missing
            (so a bare repr in a notebook degrades gracefully instead of raising).
        """
        try:
            obj = self.render()
        except ImportError:
            return {}
        hook = getattr(obj, "_repr_mimebundle_", None)
        return hook(include, exclude) if hook is not None else {}
