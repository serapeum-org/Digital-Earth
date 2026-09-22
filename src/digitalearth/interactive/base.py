"""InteractiveMapBase — the core HoloViz plumbing the interactive capability mixins build on.

``InteractiveMapBase`` owns the layer registry — what the map draws, described as layers, sources and a view
(``figure_spec``), and the HoloViews elements those descriptions were drawn into, in draw order — the display-CRS
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

from functools import reduce, wraps
from operator import mul
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Self

from loguru import logger

from digitalearth.base.bigdata import (
    DEFAULT_BIG_DATA_THRESHOLD,
    validate_big_data_threshold,
)
from digitalearth.base.crs import OffLimbError
from digitalearth.base.custom import custom_kind
from digitalearth.base.deprecation import renamed_parameter
from digitalearth.base.display import (
    auto_cmap,
    needs_reproject,
    to_display_source,
)
from digitalearth.base.registry import (
    forget_namespace,
    forget_object,
    object_namespace,
)
from digitalearth.base.sources import get_source
from digitalearth.base.sources.source import Source
from digitalearth.base.spec import (
    DataRef,
    FigureSpec,
    LayerSpec,
    LayerTree,
    PanelSpec,
    Symbology,
    Viewport,
)
from digitalearth.base.spec.bounds import same_crs

# `DEFAULT_BIG_DATA_THRESHOLD` is imported above rather than declared here: the row/face count above which a
# vector builder auto-routes to its tier's big-data renderer (#250) lives in `digitalearth.base.bigdata`, so
# this tier and the web tier cut over at the same size. One number for the whole tier either way: the map
# carries it as an attribute, and every big-data builder takes a per-call override of the same name, so a map
# configured once keeps the setting for every later layer. It stays importable from this module because that
# is where this tier's callers and tests already reach for it.

#: The id of the one panel this tier draws into. A map is a single view; the constant is named so a
#: reader of `figure_spec` sees the same panel id every tier writes.
PANEL_ID: str = "main"

#: The pip extra / pixi env that provides the HoloViz engine, quoted in the lazy-import error.
_INSTALL_HINT = (
    "the interactive tier needs the HoloViz stack (geoviews/holoviews/datashader/panel). "
    "Install it with `pip install 'digitalearth[interactive]'` "
    "(or, in this repo, `pixi install -e interactive`)."
)


def _new_renderer(interactive_map: Any) -> Any:
    """Return the renderer a map draws through.

    Imported here rather than at module scope: :mod:`digitalearth.interactive.renderer` resolves its
    drawers from the builder modules, and every one of those imports this module, so a top-level import
    would close a cycle.

    Args:
        interactive_map: The map the renderer serves.

    Returns:
        A :class:`~digitalearth.interactive.renderer.Renderer` bound to it.
    """
    from digitalearth.interactive.renderer import Renderer

    return Renderer(interactive_map)


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
    if (
        "bokeh" not in hv.Store.renderers
    ):  # register Bokeh options once so backend="bokeh" opts apply
        hv.renderer("bokeh")
    return gv, hv


def _masked_to_nan(values: Any) -> Any:
    """Return ``values`` as a float array with masked/nodata cells as ``NaN``.

    HoloViews/Bokeh render ``NaN`` cells transparent, which is the tier's NoData contract
    (pyramids hands rasters over as masked arrays).

    Args:
        values: A numpy (possibly masked) array.

    Returns:
        numpy.ndarray: a plain float array, masked entries filled with ``NaN``.
    """
    import numpy as np

    if np.ma.isMaskedArray(values):
        return values.astype(float).filled(np.nan)
    return np.asarray(values)


def _skips_off_limb(builder: Callable) -> Callable:
    """Wrap a layer builder so data the display CRS cannot place skips the layer instead of raising (#257).

    The cross-tier rule: a layer whose data lands nowhere in the display CRS is **skipped with a warning
    naming the layer**, and the rest of the map still renders — unless the map was built with
    ``strict=True``, in which case the :class:`~digitalearth.base.crs.OffLimbError` propagates. The guard
    sits on the builder rather than on the reprojection choke point so the log line can name the public
    method the caller actually invoked.

    Args:
        builder: A ``-> Self`` builder method whose body may reproject through pyramids.

    Returns:
        The same method, with an off-limb reprojection turned into a skipped layer.
    """

    @wraps(builder)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Run the builder, answering an off-limb reprojection with a skipped layer.

        Args:
            self: The composed map the builder is bound to.
            *args: The builder's positional arguments.
            **kwargs: The builder's keyword arguments.

        Returns:
            Whatever the builder returns, or the map itself when the layer was skipped.
        """
        try:
            return builder(self, *args, **kwargs)
        except OffLimbError as error:
            return self._skip_off_limb(builder.__name__, error)

    return guarded


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
        strict: How a layer whose data the display CRS cannot place is answered (#257). ``False``
            (default) skips that layer with a warning and carries on; ``True`` re-raises the
            :class:`~digitalearth.base.crs.OffLimbError` instead.
        big_data_threshold: Row/face count above which a vector builder auto-routes through Datashader
            (#250). Set once here and every later layer on this map honours it; each builder also takes
            a per-call ``big_data_threshold=`` override.

    Attributes:
        layers: The HoloViews/GeoViews elements this map overlays, in **draw** order rather than call
            order: each is placed where its layer's band puts it, so a basemap added last still sits
            under the data and a graticule added first still sits over it. This is a view of what is
            drawn, not the map's state — the map is described by :attr:`figure_spec`, and its layers are
            addressed by id through :attr:`layer_ids`. It is kept because it is the shape this tier's
            callers and its own tests have always read.

    Examples:
        - Construct and inspect the display configuration (needs no HoloViz engine):
            ```python
            >>> from digitalearth.interactive.base import InteractiveMapBase
            >>> m = InteractiveMapBase(crs=4326, width=800, height=400, title="rain")
            >>> m.crs
            4326
            >>> (m.width, m.height, m.title)
            (800, 400, 'rain')
            >>> m.layers
            []

            ```
        - Defaults target the Web-Mercator tile contract:
            ```python
            >>> from digitalearth.interactive.base import InteractiveMapBase
            >>> m = InteractiveMapBase()
            >>> m.crs
            3857
            >>> m._tiles_provider is None  # no basemap requested at construction
            True

            ```
        - Off-limb data is skipped by default, and the big-data cutoff is a plain attribute:
            ```python
            >>> from digitalearth.interactive.base import InteractiveMapBase
            >>> m = InteractiveMapBase()
            >>> m.strict
            False
            >>> m.big_data_threshold
            50000
            >>> m.big_data_threshold = 1_000  # every later layer honours it
            >>> m.big_data_threshold
            1000

            ```

    See Also:
        digitalearth.interactive.map.InteractiveMap: the public composition of this base
            with the capability mixins.
    """

    def __init__(
        self,
        *,
        crs: int = 3857,
        width: int = 700,
        height: int = 500,
        tiles: Optional[str] = None,
        title: str = "",
        strict: bool = False,
        big_data_threshold: int = DEFAULT_BIG_DATA_THRESHOLD,
    ):
        """Set up an empty map: display CRS, Bokeh frame, and the two tier-wide policies.

        Nothing here touches HoloViz — construction is deliberately engine-free, so a map can be
        built (and its configuration read back) without the ``interactive`` extra installed; the
        lazy import fires on the first builder/render call instead.

        Both policy arguments are stored as **plain public attributes**, so a map configured once
        keeps the setting for every later layer, and either can be changed after construction.

        Args:
            crs: Display CRS as an EPSG integer. Default ``3857`` (Web Mercator — the only CRS
                Bokeh tile basemaps render); ``4326`` gives a non-tiled Plate-Carrée map.
            width: Frame width in pixels for the rendered Bokeh plot.
            height: Frame height in pixels for the rendered Bokeh plot.
            tiles: Tile-provider name drawn beneath the data layers, applied once at render time,
                or ``None`` for no basemap. Stored privately because the public ``tiles`` name is
                the decoration mixin's builder method.
            title: Plot title, applied to every styled element's Bokeh frame.
            strict: How a layer whose data the display CRS cannot place is answered (#257).
                ``False`` (default) skips that layer with a warning; ``True`` re-raises the
                :class:`~digitalearth.base.crs.OffLimbError`.
            big_data_threshold: Row/face count above which a vector builder auto-routes through
                Datashader (#250). Every big-data builder takes a per-call override of the same
                name that falls back to this value.
        """
        self.crs = crs
        self.width = width
        self.height = height
        # Off-limb policy (#257): skip-and-warn by default, raise under strict.
        self.strict = strict
        # Big-data cutoff (#250): the map-wide default every builder's per-call override falls back to.
        self.big_data_threshold = big_data_threshold
        # Stored privately: the public name `tiles` is the DecorationMixin builder method.
        self._tiles_provider = tiles
        self.title = title
        # Display projection for the matplotlib-backend path (DI.9); None = Bokeh Web-Mercator.
        self._projection: Any = None
        # Draw-tool stream (DI.8), set by the interaction mixin's draw(); None until a draw tool is added.
        self._draw_stream: Any = None
        # `(data, value_column, mesh)` while a `trimesh()` call is drawing: the builder builds the mesh to
        # count its faces and hands it to the drawer rather than have it built twice. None outside that call.
        self._built_mesh: Optional[tuple] = None
        # Class breaks / categories from the most recent choropleth, for building a legend out-of-band
        # (web-tier parity). None until a categorical choropleth runs; the continuous ramp resets it to None.
        self.last_breaks: Optional[List[Any]] = None
        self.layers: List[Any] = []
        # Style record written by `_styled`, keyed by `id()` of the element it returned (the object the
        # builders register, so the key stays alive for as long as the layer does). Read back through the
        # public `style_of` / `layer_styles` accessors — the tier owns its styling state instead of
        # delegating it to HoloViews' global option Store.
        self._styles: Dict[int, dict] = {}
        # What the map draws, as data. `self.layers` holds the built HoloViews elements — the drawing —
        # and this holds the description each was built from, which is what a figure can be written to and
        # read back from (#300).
        self._layer_tree = LayerTree()
        # Created once and kept: what it holds is what a converted kind composes into, so a layer drawn
        # when its builder ran is still there at render time.
        self._renderer = _new_renderer(self)
        self._sources: Dict[str, DataRef] = {}
        self._objects_ns: str = object_namespace()
        self._id_counter: int = 0
        self._issued_ids: set = set()
        # A keyed basemap's credential, by the id of the layer that needs it. Deliberately not in the
        # symbology: a figure is written to JSON and read back, and a key written into one leaks with it.
        self._layer_keys: Dict[str, Any] = {}
        # The layer the caller added last — what `colorbar`, `legend`, `hover` and `on_tap` act on by
        # default. Tracked by id because `layers` is in draw order, so its last entry is whatever sits in the
        # highest band, not the layer the caller just added (review H5). The web tier's `_last_layer_id`.
        self._last_layer_id: Optional[str] = None

    def _raster_element(
        self, x: Any, y: Any, arr: Any, name: str, bounds: Any = None
    ) -> Any:
        """Build the raster element: plain ``hv.Image`` (Bokeh path) or ``gv.Image`` under a projection.

        With no display projection set the element is a plain ``hv.Image`` already in the display CRS
        (the interactive Bokeh path — option A, no re-projection). When a non-Mercator projection is
        active (DI.9), it is a ``gv.Image`` carrying ``crs=self.crs`` so GeoViews' matplotlib backend
        reprojects the display-CRS coordinates to the target projection at render time.

        Args:
            x: 1-D x / longitude cell-centre coordinates (display CRS).
            y: 1-D y / latitude cell-centre coordinates (display CRS).
            arr: 2-D value array (masked nodata already filled with ``NaN``).
            name: Value-dimension name.
            bounds: The rectangle the cells cover, as ``(west, south, east, north)``, when the reader knows
                it. Given, the element is built from the array and placed on it; left out, HoloViews derives
                the placement from the axes — which it cannot do for an axis of one cell, because one sample
                has no spacing (it returns `nan` bounds and then raises on the next frame).

        Returns:
            An ``hv.Image`` or ``gv.Image`` element.
        """
        gv, hv = _require_holoviz()
        # The axes are the better answer wherever they have one: they carry the grid's *direction*, which
        # `SourceView._axes` derives from the geotransform's step signs, and `hv.Image(arr, bounds=...)`
        # assumes row 0 is north and column 0 is west. Placing every read by its bounds mirrored a south-up
        # or east-left raster, silently, because a flipped raster draws perfectly happily (review H4).
        # Bounds are the fallback for the one case the axes cannot answer: a single cell has no spacing.
        degenerate = len(getattr(x, "ravel", lambda: x)()) < 2 or (
            len(getattr(y, "ravel", lambda: y)()) < 2
        )
        use_bounds = bounds is not None and degenerate
        data = arr if use_bounds else (x, y, arr)
        placement = {"bounds": tuple(bounds)} if use_bounds else {}
        if self._projection is None:
            return hv.Image(data, kdims=["x", "y"], vdims=[name], **placement)
        return gv.Image(
            data,
            kdims=["x", "y"],
            vdims=[name],
            crs=gv.util.process_crs(self.crs),
            **placement,
        )

    def _layer_id(self, prefix: str, name: Optional[str] = None) -> str:
        """Return a unique layer id: the caller's name when they gave one, else a generated one.

        Args:
            prefix: What a generated id counts — the kind, usually.
            name: The caller's own name for the layer, used as its id when it is free.

        Returns:
            The id. A caller's name that collides with one already issued is suffixed, because two layers
            sharing an id makes the second unaddressable.
        """
        candidate = name or ""
        while not candidate or candidate in self._issued_ids:
            self._id_counter += 1
            candidate = (
                f"{name}-{self._id_counter}" if name else f"{prefix}-{self._id_counter}"
            )
        self._issued_ids.add(candidate)
        return candidate

    def _index_layer(
        self,
        layer_id: str,
        label: Optional[str],
        *,
        kind: str,
        visible: bool = True,
        band: Optional[str] = None,
        source: Any = None,
        symbology: Any = None,
    ) -> None:
        """Record a layer so the figure describes it rather than only holding its element.

        Args:
            layer_id: The layer's id, as `layer_ids` reports it.
            label: What a layer switcher should call it; `None` falls back to the id.
            kind: The registered, engine-neutral kind — `"raster"`, `"points"`, `"choropleth"` — not the
                HoloViews element type, which cannot tell a choropleth from a plain polygon draw.
            visible: Whether the layer was built visible.
            band: Where it is drawn, when its kind does not say — what a caller's own object needs, since
                `custom:holoviews` names the engine rather than what it draws.
            source: What the layer draws, recorded under its id. `None` for a layer drawn from no data.
            symbology: How it looks, as values rather than as the applied HoloViews options.
        """
        if source is not None:
            self._sources[layer_id] = DataRef.of(
                source, name=f"{self._objects_ns}:{layer_id}"
            )
        self._layer_tree = self._layer_tree.add(
            LayerSpec(
                layer_id,
                kind,
                source_id=layer_id if source is not None else None,
                symbology=Symbology() if symbology is None else symbology,
                label=label or layer_id,
                visible=bool(visible),
                band=band,
            )
        )

    @property
    def layer_ids(self) -> List[str]:
        """The ids of the layers this map draws, in draw order, bottom first.

        Returns:
            One id per described layer. A layer the caller named carries that name; an unnamed one gets a
            generated id counting the layers of its kind.

        Examples:
            - A named layer keeps its name; an unnamed one is numbered after what it is:
                ```python
                >>> from digitalearth.interactive import InteractiveMap
                >>> InteractiveMap().add_element("dem", name="elevation").add_element("obs").layer_ids
                ['elevation', 'holoviews-1']

                ```
            - Draw order, not call order: a layer in the reference band added second is listed first,
              because that is where it is drawn:
                ```python
                >>> from digitalearth.interactive import InteractiveMap
                >>> m = InteractiveMap().add_element("obs", name="obs")
                >>> m.add_element("grid", name="grid", band="reference").layer_ids
                ['grid', 'obs']

                ```
            - A map that has drawn nothing has no ids:
                ```python
                >>> from digitalearth.interactive import InteractiveMap
                >>> InteractiveMap().layer_ids
                []

                ```
        """
        return list(self._layer_tree.ids)

    @property
    def figure_spec(self) -> FigureSpec:
        """What the map draws, as data: its layers, their sources and the view.

        Returns:
            A :class:`~digitalearth.base.spec.FigureSpec` with one panel, `"main"`, whose layers are the
            tree in draw order and whose sources are what each builder was given.

        Examples:
            - The panel names the layers it draws, and each layer says what it is:
                ```python
                >>> from digitalearth.interactive import InteractiveMap
                >>> figure = InteractiveMap().add_element("dem", name="elevation").figure_spec
                >>> figure.panels[0].id, figure.panels[0].layers
                ('main', ('elevation',))
                >>> figure.layers.get("elevation").kind
                'custom:holoviews'

                ```
            - A builder describes its layer by the engine-neutral kind and the recipe it drew by, and
              files what it drew as the layer's source:
                ```python
                >>> import geopandas as gpd                                       # doctest: +SKIP
                >>> from shapely.geometry import Point                            # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap           # doctest: +SKIP
                >>> gdf = gpd.GeoDataFrame(geometry=[Point(4.9, 52.4)], crs=4326) # doctest: +SKIP
                >>> figure = InteractiveMap().points(gdf).figure_spec             # doctest: +SKIP
                >>> layer = figure.layers.get("points-1")                         # doctest: +SKIP
                >>> layer.kind, layer.symbology.props["via"], layer.source_id     # doctest: +SKIP
                ('points', 'geometry', 'points-1')

                ```
        """
        tree = self._layer_tree
        panel = PanelSpec(
            PANEL_ID,
            self.viewport,
            layers=tuple(tree.ids),
        )
        return FigureSpec(
            panels=(panel,),
            layers=tree,
            sources={
                key: ref for key, ref in self._sources.items() if key in set(tree.ids)
            },
        )

    @property
    def viewport(self) -> Viewport:
        """Where the map is looking, as a value.

        Returns:
            A :class:`~digitalearth.base.spec.Viewport` in the CRS this tier places data in. It carries no
            centre, zoom or bounds: a Bokeh figure is framed by what the viewer pans and zooms to rather
            than by a region set before it is drawn, which is what
            :data:`~digitalearth.interactive.capabilities.CAPABILITIES` records under `absent["domain"]`.

        Examples:
            - The view reports the CRS the map places data in, Web Mercator by default:
                ```python
                >>> from digitalearth.interactive import InteractiveMap
                >>> view = InteractiveMap().viewport
                >>> view.crs, view.center, view.zoom
                (3857, None, None)

                ```
            - A map built in another display CRS says so:
                ```python
                >>> from digitalearth.interactive import InteractiveMap
                >>> InteractiveMap(crs=4326).viewport.crs
                4326

                ```
        """
        return Viewport(crs=self.crs)

    def add_element(
        self,
        element: Any,
        *,
        kind: Optional[str] = None,
        name: Optional[str] = None,
        visible: bool = True,
        band: Optional[str] = None,
        source: Any = None,
        symbology: Any = None,
        key: Any = None,
    ) -> Self:
        """Register a HoloViews/GeoViews ``element`` as a layer and return ``self`` (chainable).

        The low-level entry point the capability mixins build on — every builder method ends here.

        A builder of a drawn kind passes ``element=None`` and a ``symbology``: the layer is described first
        and its drawer builds the element from that description, which is what "render from the description"
        means here.

        An object you build yourself is a **custom layer**: what a figure keeps is its description — an id, the
        kind `custom:holoviews`, a label, a band and whether it is visible — and never the object, which has no
        description to write. The object stays with the scene that was handed it, so a figure saved and loaded
        again names the layer but cannot rebuild it, and another backend cannot draw it at all
        (:mod:`digitalearth.base.custom` says which case a reader is in).

        Args:
            element: Any HoloViews/GeoViews element (or overlay-able object), or ``None`` for a layer of a
                kind this tier draws — that one is built by its drawer from ``symbology``.
            kind: The registered, engine-neutral kind the layer is — ``"raster"``, ``"points"``, … A kind
                listed in :data:`~digitalearth.interactive.renderer.DRAWN_KINDS` is drawn from its
                description; ``None`` makes it a custom layer (``custom:holoviews``).
            name: The caller's own name for the layer, used as its id when it is free and as its label.
            visible: Whether the layer is described as visible.
            band: The draw-order band, for a kind that does not imply one — what a custom layer needs,
                since the engine name says nothing about what it draws.
            source: What the layer draws, recorded under its id; ``None`` for a layer drawn from no data.
            symbology: How it looks, as values — the description its drawer reads.
            key: A credential the layer's drawer needs, held on the map rather than in ``symbology`` so a
                figure written to JSON carries no API key. ``None`` for every layer that needs none.

        Returns:
            The same map instance, so builder calls chain: ``m.image(dem).tiles().coastlines()``. A drawer
            that declined to draw the layer leaves the map as it was — the description, the registered
            source, the held key and the id are all let go again, so nothing names a layer that was never
            drawn.

        Raises:
            KeyError: when ``kind`` is one this tier draws but ``symbology`` records no ``via`` it has a
                recipe for — a builder routed to a kind without saying how it drew it. The refusal names
                the recipes that kind is drawn by.
            Exception: whatever the layer's drawer raises — an
                :class:`~digitalearth.base.crs.OffLimbError` from a warp that places nothing, a ``KeyError``
                for a column the data does not have — re-raised after the layer is let go exactly as a
                declined one is.

        Examples:
            - Two layers of one band register in call order and return the map for chaining (any object
              can stand in for a HoloViews element here — the registry does not inspect it):
                ```python
                >>> from digitalearth.interactive import InteractiveMap
                >>> m = InteractiveMap()
                >>> m.add_element("raster-layer").add_element("vector-layer") is m
                True
                >>> m.layers
                ['raster-layer', 'vector-layer']

                ```
            - A ``band`` decides where the element lands, not the order it was added in: a reference layer
              registered second is drawn beneath the data layer registered first:
                ```python
                >>> from digitalearth.interactive import InteractiveMap
                >>> m = InteractiveMap().add_element("observations")
                >>> m = m.add_element("grid", name="grid", band="reference")
                >>> m.layers
                ['grid', 'observations']
                >>> m.layer_ids
                ['grid', 'holoviews-1']

                ```
            - With the engine installed, real elements register the same way:
                ```python
                >>> import holoviews as hv                       # doctest: +SKIP
                >>> m = InteractiveMap()                         # doctest: +SKIP
                >>> m.add_element(hv.Points([(0, 0)])).layers    # doctest: +SKIP
                [:Points   [x,y]]

                ```
        """
        # A builder that says what it drew is described; one that does not is a caller's own object,
        # which has no description to write beyond the fact that it exists
        # (:mod:`digitalearth.base.custom`).
        from digitalearth.interactive.renderer import DRAWN_KINDS

        resolved = kind or custom_kind("holoviews")
        layer_id = self._layer_id(resolved.split(":")[-1], name)
        # Everything from here to the draw is undone together if any of it raises. The layer is described
        # before its drawer runs, so a drawer that refuses — an off-limb warp, `strict=True`, a column that
        # is not there — would otherwise leave the figure naming a layer nothing drew, its data registered
        # for the life of the process, and every later layer inserted at an index counted past the ghost.
        try:
            if key is not None:
                self._layer_keys[layer_id] = key
            self._index_layer(
                layer_id,
                name,
                kind=resolved,
                visible=visible,
                band=band,
                source=source,
                symbology=symbology,
            )
            if resolved in DRAWN_KINDS:
                # The element is built from the description rather than handed in: the builder passed `None`
                # and its drawer makes the real one, which is what "render from the description" means.
                drawn = self._renderer.draw_layer(self.figure_spec, layer_id)
                if drawn is None:
                    self._forget_layer(layer_id)
                    return self
                element = drawn.element
        except BaseException:
            self._forget_layer(layer_id)
            raise
        # Placed where the description puts it, not where the call happened to arrive. The tree orders by
        # draw-order band, so a basemap added last still goes under the data and a graticule added first
        # still goes over it — and the list `_compose` overlays cannot disagree with the figure the map
        # reports, which is what it did when a builder had to remember to insert at the front itself.
        self.layers.insert(self._layer_tree.ids.index(layer_id), element)
        self._note_last_layer(layer_id)
        return self

    def _note_last_layer(self, layer_id: str) -> None:
        """Make a just-added layer the one the toggles act on, unless it is an underlay added over data.

        A basemap, land or ocean is drawn beneath the data and carries no colorbar, legend or hover of its
        own, so `image(dem).tiles().colorbar(False)` means the raster. Before the layer tree, an underlay was
        inserted at the front of `layers` and so never became the layer these acted on; the web tier's
        `_last_layer_id` likewise counts data layers only. An underlay still takes the slot while nothing
        above the underlay band has been added, so a map of tiles alone can be configured at all.

        Args:
            layer_id: The layer just added.
        """
        held = self._last_layer_id
        if (
            held is not None
            and held in self._layer_tree
            and self._band_of(layer_id) == "underlay"
            and self._band_of(held) != "underlay"
        ):
            return
        self._last_layer_id = layer_id

    def _band_of(self, layer_id: str) -> str:
        """Return the draw-order band one of the map's layers sits in.

        Args:
            layer_id: A layer the tree holds.

        Returns:
            The band, as the renderer places it: the layer's own when it declared one, else its kind's.
        """
        band: str = self._renderer.band_for(self._layer_tree.get(layer_id))
        return band

    def _last_layer_index(self, method: str) -> int:
        """Return where in :attr:`layers` the layer the caller added last is drawn.

        Args:
            method: The public method asking, named in the refusal.

        Returns:
            The index of that layer's element. `layers` is kept in the tree's order — each element is
            inserted at its layer's tree index — so the tree index is the element's index too.

        Raises:
            ValueError: when the map has no layer yet.
        """
        if self._last_layer_id is None:
            raise ValueError(
                f"{method}() needs at least one layer — add a builder call first"
            )
        return self._layer_tree.ids.index(self._last_layer_id)

    def _forget_layer(self, layer_id: str) -> None:
        """Drop a layer that was described but never drawn, and everything it registered.

        What :meth:`add_element` calls for a layer its drawer declined or refused. Taking the entry out of the tree is
        not enough on its own: the source sits in the process-global object table, which holds a strong
        reference for the life of the process, and `figure_spec.sources` is filtered to the tree's ids, so
        nothing a caller reads would show it was still there. The id goes back to the pool too, so a caller
        who names a layer, watches it decline, and names it again gets the name they asked for; and a
        credential held for a layer nothing draws is a secret held for nothing.

        Args:
            layer_id: The layer to forget. A layer the tree no longer holds is ignored, so a caller can
                forget one whose description was never finished.
        """
        if layer_id in self._layer_tree:
            self._layer_tree = self._layer_tree.remove(layer_id)
        self._issued_ids.discard(layer_id)
        ref = self._sources.pop(layer_id, None)
        if ref is not None:
            forget_object(ref.uri)
        self._layer_keys.pop(layer_id, None)

    def _needs_reproject(self, data: Any) -> bool:
        """Whether `data` must be reprojected (via pyramids) to the display CRS.

        Args:
            data: A pyramids object exposing `.crs` and/or `.epsg` (`Dataset`/`FeatureCollection`).

        Returns:
            `False` when the data's CRS and the display CRS name the same reference system, however either
            is spelled; `True` otherwise.

        Note:
            This is a thin alias for :func:`digitalearth.base.display.needs_reproject`, kept because tier code
            and tests call it. :meth:`_to_display_source` calls the shared function **directly**, so overriding
            this method no longer changes what *that* method reprojects. The tier's other callers — `rgb`,
            `large_image`, `_display_gdf` and `tap_profile` — still consult it, so an override now applies to
            them and not to `_to_display_source`. Nothing in-tree overrides it.
        """
        return needs_reproject(data, self.crs)

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
        return to_display_source(data, self.crs, band=band)

    def _skip_off_limb(self, layer: str, error: OffLimbError) -> Self:
        """Answer a layer whose data the display CRS cannot place: skip it, or raise under ``strict``.

        The default is a **warning**, not a debug line: unlike the static tier there is no clipped globe
        here — the display CRS is Web Mercator or another unclipped projection — so a warp that places
        *none* of the data almost always means the source's CRS or geo-transform is wrong, and the only
        other symptom would be a layer that silently never appears.

        Args:
            layer: The public builder that drew nothing, named in the log line.
            error: The off-limb error the reprojection raised.

        Returns:
            The same map instance, so the skipped call still chains.

        Raises:
            OffLimbError: when the map was built with ``strict=True``.
        """
        if self.strict:
            raise error
        logger.warning(
            f"{layer}: none of the data could be placed in crs={self.crs!r}, so the layer was skipped "
            f"({error}). Build the map with InteractiveMap(strict=True) to raise instead."
        )
        return self

    def _resolve_big_data_threshold(
        self,
        big_data_threshold: Optional[int] = None,
        rasterize_threshold: Optional[int] = None,
        *,
        caller: str,
    ) -> int:
        """Resolve a builder's big-data cutoff: per-call value, else the map's attribute (#250).

        ``rasterize_threshold`` is the tier's **deprecated** spelling of the same number, resolved through
        :func:`~digitalearth.base.deprecation.renamed_parameter` — the one rename rule every backend shares,
        so this tier refuses two spellings of one cutoff exactly as static, web and 3-D do.

        A per-call cutoff is checked by :func:`~digitalearth.base.bigdata.validate_big_data_threshold`, the
        shared guard the web tier applies too — so a negative cutoff is refused identically on both rather
        than raising on one tier and routing every layer, empty ones included, on the other.

        Args:
            big_data_threshold: The per-call override, or ``None`` to use the map's attribute.
            rasterize_threshold: The deprecated alias of ``big_data_threshold``.
            caller: The builder the keywords were written on, named in the warning and the error.

        Returns:
            The row/face count above which the calling builder routes through Datashader.

        Raises:
            TypeError: if both spellings are passed — they name one cutoff, so two values for it cannot
                both be honoured.
            ValueError: when the per-call cutoff is negative.

        Warns:
            DeprecationWarning: when ``rasterize_threshold`` is passed.
        """
        threshold = renamed_parameter(
            new="big_data_threshold",
            value=big_data_threshold,
            old="rasterize_threshold",
            alias=rasterize_threshold,
            caller=caller,
            # renamed_parameter -> here -> the builder -> @_skips_off_limb's wrapper -> its caller.
            # The wrapper is the frame the old inline warning stopped at, so it named this module
            # instead of the notebook cell that wrote the deprecated keyword.
            stacklevel=5,
        )
        if threshold is None:
            return int(self.big_data_threshold)
        return validate_big_data_threshold(threshold, caller=caller)

    def _auto_style(self, source: Source) -> Dict[str, Any]:
        """Return the :func:`~digitalearth.base.autostyle.auto_style` record for ``source``.

        The tier's single autostyle lookup — the same variable→style table (incl. the ECMWF-Magics match)
        the static ``Map`` consults, so a recognised field is coloured, contoured and labelled the same way
        on every tier. Imported lazily so importing the tier never pulls the style library in.

        Args:
            source: The display-CRS source whose variable drives the lookup.

        Returns:
            dict: the style record — always a ``cmap``, plus ``levels`` / ``units`` for a recognised field.
        """
        from digitalearth.base.autostyle import auto_style

        return auto_style(source)

    def _auto_levels(self, source: Source, levels: Any) -> Any:
        """Resolve contour levels: the caller's ``levels`` if given, else the autostyle levels (#230).

        Args:
            source: The display-CRS source whose variable drives the lookup.
            levels: The caller-supplied level count / edges, or ``None`` to auto-resolve.

        Returns:
            The levels to contour with, or ``None`` when the variable is unrecognised (the caller then
            applies its own fallback).
        """
        if levels is not None:
            return levels
        return self._auto_style(source).get("levels")

    def _auto_clabel(self, source: Source, clabel: Optional[str]) -> Optional[str]:
        """Resolve the colorbar label: the caller's ``clabel`` if given, else the autostyle units (#230).

        Never guesses: an unrecognised variable carries no units, and the colorbar is then left unlabelled
        exactly as before.

        Args:
            source: The display-CRS source whose variable drives the lookup.
            clabel: The caller-supplied colorbar label, or ``None`` to auto-resolve.

        Returns:
            The colorbar label, or ``None`` when neither the caller nor the style library supplies one.
        """
        if clabel is not None:
            return clabel
        return self._auto_style(source).get("units")

    def _auto_cmap(self, source: Source, cmap: Optional[str]) -> str:
        """Resolve a colormap: the caller's ``cmap`` if given, else the autostyle default (DI.12).

        Defers to :func:`digitalearth.base.autostyle.auto_style` (the same variable→style lookup the static
        ``Map`` uses, incl. the ECMWF-Magics match) so a variable looks the same across tiers; falls
        back to ``"viridis"`` for an unrecognised field.

        Goes through :meth:`_auto_style`, like :meth:`_auto_levels` and :meth:`_auto_clabel`, so the three
        readers see one lookup and a subclass overriding it changes all three together.

        Args:
            source: The display-CRS source whose variable drives the lookup.
            cmap: The caller-supplied colormap, or ``None`` to auto-resolve.

        Returns:
            The colormap name to use.
        """
        return auto_cmap(source, cmap, lookup=self._auto_style)

    def _auto_cmap_for_band(self, dataset: Any, band: int, cmap: Optional[str]) -> str:
        """Resolve a colormap from a raster band's **name**, without reading the band (#249).

        The windowed reader (:meth:`~digitalearth.interactive.raster.RasterMixin.large_image`) must not
        materialise the array just to style it, so the autostyle lookup is fed a metadata-only
        :class:`Source`: :func:`~digitalearth.base.autostyle.auto_style` reads the variable name (and CF
        attributes), never the values, so a 1×1 placeholder array is enough to drive it.

        Args:
            dataset: A pyramids ``Dataset`` whose ``band_names`` name the variable.
            band: The 1-based band being rendered.
            cmap: The caller-supplied colormap, or ``None`` to auto-resolve.

        Returns:
            The colormap name to use.
        """
        if cmap is not None:
            return cmap
        import numpy as np

        names = list(getattr(dataset, "band_names", None) or [])
        variable = names[band - 1] if 1 <= band <= len(names) else ""
        placeholder = get_source(
            np.zeros((1, 1)), metadata={"variable": variable or ""}
        )
        return self._auto_cmap(placeholder, None)

    @staticmethod
    def _vdim_name(source: Source) -> str:
        """Return the value-dimension name for ``source`` (variable → z name → ``"value"``).

        The single naming recipe the raster/temporal builders share, so the value dimension is
        labelled consistently across ``image`` / ``quadmesh`` / ``timecube`` frames.

        Args:
            source: The display-CRS :class:`Source` whose value dimension is being named.

        Returns:
            The value-dimension name.
        """
        return source.metadata("variable", None) or source.z.name or "value"

    def _styled(
        self,
        element: Any,
        common: Optional[dict] = None,
        bokeh: Optional[dict] = None,
        owner: Any = None,
    ) -> Any:
        """Apply backend-agnostic style opts plus Bokeh-only frame opts to ``element``.

        Backend-agnostic options (``cmap``/``clim``/``alpha``/…) apply to whichever backend renders;
        the Bokeh-only frame (``width``/``height``/``tools``/``title``) is recorded for the Bokeh
        backend specifically, so the matplotlib save path ignores it instead of erroring.

        The applied options are also **recorded on the map**, keyed by the returned element, so a caller can
        read a layer's styling back through :meth:`style_of` / :attr:`layer_styles` instead of reaching into
        HoloViews' global option ``Store`` (which is where ``.opts()`` alone would leave them).

        Args:
            element: The HoloViews/GeoViews element to style.
            common: Backend-agnostic options; ``None``-valued entries are dropped.
            bokeh: Extra Bokeh-only options merged over the default frame.
            owner: The registered layer this element is a *frame* of, for a dynamic layer whose frames are
                drawn one per viewport. `None` for an element that is itself the layer.

        Returns:
            The styled element.
        """
        _require_holoviz()  # called for its actionable ImportError; no module name is needed here
        style = self._style_record(common, bokeh)
        if style["common"]:
            element = element.opts(**style["common"])
        element = element.opts(backend="bokeh", **style["bokeh"])
        self._record_style(element, style, owner)
        return element

    def _style_record(
        self, common: Optional[dict] = None, bokeh: Optional[dict] = None
    ) -> dict:
        """Return the style :meth:`_styled` applies for these options, without an element to apply it to.

        Split out so a layer can file its style before it has drawn anything: a dynamic layer's frames are
        drawn lazily, and drawing one only to read its style back cost a full window read per layer.

        Args:
            common: Backend-agnostic options; ``None``-valued entries are dropped.
            bokeh: Extra Bokeh-only options merged over the default frame.

        Returns:
            dict: ``{"common": {...}, "bokeh": {...}}`` — the shape :meth:`style_of` reads back.
        """
        kept = {
            key: value for key, value in (common or {}).items() if value is not None
        }
        frame: dict = {"width": self.width, "height": self.height}
        if self.title:
            frame["title"] = self.title
        frame.update(bokeh or {})
        return {"common": kept, "bokeh": frame}

    def _record_style(self, element: Any, style: dict, owner: Any = None) -> None:
        """File the style of a just-drawn element, and of the layer it is a frame of.

        A dynamic layer is registered as a `DynamicMap` and redrawn as a fresh element per frame, so the
        style recorded against the frame answered for nothing a caller holds: `style_of` on a
        `large_image(dynamic=True)` layer read empty, and the dashboard's widgets — which look the style up
        to merge over it — passed such a layer by (review M17). The style is filed against the registered
        layer as well, when the frame says which one it belongs to.

        It has to *say*. Reading `self.layers[-1]` instead was a guess, and a wrong one: every builder
        styles its element before registering it, so while a builder styles, the last registered layer is
        still the **previous** one — a static layer drawn after a dynamic one overwrote the dynamic layer's
        style, and with it the widget values merged over it (review H3).

        The table is then trimmed to the registered layers plus this element. Every frame is a new object
        under a new `id()`, so the old keying grew one dead entry per pan — sixteen after fifteen of them —
        keyed by the id of a collected object, which another object may later be handed (review M18).

        Args:
            element: The element that was styled.
            style: What was applied to it.
            owner: The registered layer `element` is a frame of, or `None` when it is the layer itself.
        """
        self._styles[id(element)] = style
        if owner is not None:
            self._styles[id(owner)] = style
        live = {id(layer) for layer in self.layers} | {id(element)}
        if owner is not None:
            live.add(id(owner))
        self._styles = {key: held for key, held in self._styles.items() if key in live}

    def style_of(self, layer: Any) -> dict:
        """Return the style options :meth:`_styled` applied to ``layer``.

        The public read-back for the tier's styling: ``.opts()`` writes into HoloViews' global option
        ``Store`` and returns nothing a caller can inspect, so every builder's styling is recorded here as
        it is applied. The result is the *requested* styling, which is what a dashboard override has to be
        merged over; the resolved, HoloViews-side view of the same values stays available through
        ``holoviews.Store.lookup_options(backend, element, group)``.

        Args:
            layer: A registered layer — either the element itself, or its integer index in
                :attr:`layers` (negative indices count from the end).

        Returns:
            dict: ``{"common": {...}, "bokeh": {...}}`` — the backend-agnostic options and the Bokeh-only
            frame. Both are empty for a layer that never went through :meth:`_styled` (a tile basemap, a
            Natural-Earth feature, a raw element passed to :meth:`add_element`).

        Raises:
            IndexError: when ``layer`` is an integer outside the registered layer range.

        Examples:
            - A builder's styling is readable straight off the map:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().image(dem, cmap="magma", alpha=0.5)    # doctest: +SKIP
                >>> m.style_of(0)["common"]["cmap"]                             # doctest: +SKIP
                'magma'

                ```
        """
        element = self.layers[layer] if isinstance(layer, int) else layer
        recorded = self._styles.get(id(element))
        if recorded is None:
            return {"common": {}, "bokeh": {}}
        return {"common": dict(recorded["common"]), "bokeh": dict(recorded["bokeh"])}

    @property
    def layer_styles(self) -> List[dict]:
        """The recorded style of every registered layer, in draw order.

        The whole-map view of :meth:`style_of`: one entry per layer, positionally aligned with
        :attr:`layers`, so the styling a builder *requested* can be read back off the map
        instead of being looked up in HoloViews' global option ``Store``. This is what the
        dashboard's widget overrides are merged over.

        Returns:
            list: one ``{"common": {...}, "bokeh": {...}}`` dict per entry of :attr:`layers`, as
            :meth:`style_of` returns it.

        Examples:
            - There is exactly one entry per layer, and a layer that never went through the styling
              path records nothing (the registry does not inspect what it is handed):
                ```python
                >>> from digitalearth.interactive import InteractiveMap
                >>> m = InteractiveMap().add_element("raster-layer").add_element("vector-layer")
                >>> len(m.layer_styles) == len(m.layers)
                True
                >>> m.layer_styles
                [{'common': {}, 'bokeh': {}}, {'common': {}, 'bokeh': {}}]

                ```
            - An empty map has no styles to report:
                ```python
                >>> from digitalearth.interactive import InteractiveMap
                >>> InteractiveMap().layer_styles
                []

                ```
            - With the engine installed, each builder's requested options show up in draw order:
                ```python
                >>> from pyramids.dataset import Dataset                       # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")       # doctest: +SKIP
                >>> m = InteractiveMap().image(dem, cmap="magma", alpha=0.5)   # doctest: +SKIP
                >>> [style["common"]["cmap"] for style in m.layer_styles]      # doctest: +SKIP
                ['magma']

                ```

        See Also:
            style_of: the same record for one layer, by element or by index.
        """
        return [self.style_of(layer) for layer in self.layers]

    def _require_web_mercator(self, method: str) -> None:
        """Raise when a Web-Mercator-only decoration is requested on a non-3857 map.

        Bokeh tile basemaps (and GeoViews' Bokeh feature rendering) are EPSG:3857-only; on any
        other display CRS they would silently misalign with the pre-reprojected data layers, so
        the tier fails loudly instead.

        Args:
            method: The calling method's name (quoted in the error).

        Raises:
            ValueError: when ``self.crs`` is not ``3857``.
        """
        if not same_crs(self.crs, 3857):
            raise ValueError(
                f"{method}() needs the Web-Mercator display CRS (crs=3857) — Bokeh renders tiles/"
                f"features in EPSG:3857 only, and this map uses crs={self.crs!r}. Either build the "
                "map with InteractiveMap(crs=3857) (the default) or drop the decoration."
            )

    def _compose(self, layers: Any) -> Any:
        """Overlay layers in draw order.

        The one place the tier turns a sequence of elements into one figure, so a caller that renders a subset
        — the layer switcher, a dashboard widget restyling each layer on its own (#300) — composes exactly as
        `render` does rather than growing a second reduce beside it.

        Args:
            layers: The elements to overlay, bottom first.

        Returns:
            An empty `hv.Overlay` for none, the element itself for one, and their product otherwise.
        """
        _, hv = _require_holoviz()
        drawn = list(layers)
        if not drawn:
            return hv.Overlay([])
        if len(drawn) == 1:
            return drawn[0]
        return reduce(mul, drawn)

    def render(self) -> Any:
        """Compose the registered layers into one HoloViews object (overlaid with ``*``).

        Returns:
            The single element when one layer is registered, an ``hv.Overlay`` of all layers in draw
            order otherwise (an empty map renders as a blank ``hv.Overlay``). The overlay's order is the
            order :attr:`figure_spec` describes, because each element was placed where its layer's band
            put it rather than where the builder call happened to arrive.

        Raises:
            ImportError: when the ``interactive`` extra is not installed.

        Examples:
            - Two registered layers compose into an overlay in draw order (needs the engine):
                ```python
                >>> import holoviews as hv                                   # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap      # doctest: +SKIP
                >>> m = InteractiveMap()                                     # doctest: +SKIP
                >>> _ = m.add_element(hv.Points([(0, 0)]))                   # doctest: +SKIP
                >>> _ = m.add_element(hv.Points([(1, 1)]))                   # doctest: +SKIP
                >>> overlay = m.render()                                     # doctest: +SKIP
                >>> len(overlay)                                             # doctest: +SKIP
                2

                ```
            - An empty map renders as a blank overlay rather than raising:
                ```python
                >>> from digitalearth.interactive import InteractiveMap      # doctest: +SKIP
                >>> len(InteractiveMap().render())                           # doctest: +SKIP
                0

                ```
        """
        self._flush_deferred_tiles()
        return self._projected(self._compose(self.layers))

    # Note: the dashboard's widget paths call `_flush_deferred_tiles`, `_compose` and `_projected`
    # themselves rather than this method, because they compose from a *restyled* or *chosen* subset of the
    # layers rather than from all of them. A subclass overriding `render` to change how the figure is put
    # together should override `_compose`, which both routes share (review N10).

    def _flush_deferred_tiles(self) -> None:
        """Draw a basemap that was asked for before there was a figure to draw it on.

        `InteractiveMap(tiles=...)` records the provider and defers it, because the tile layer has to sit
        under layers that do not exist yet. This is where it is applied, once, and it has to run before the
        layers are read — it adds one.
        """
        if self._tiles_provider is not None and hasattr(self, "tiles"):
            provider, self._tiles_provider = self._tiles_provider, None  # apply once
            self.tiles(provider)

    def _projected(self, obj: Any) -> Any:
        """Return `obj` drawn in the map's projection, when one was asked for.

        Args:
            obj: The composed figure.

        Returns:
            `obj` unchanged when no projection was set, else the same figure carrying it. The matplotlib
            renderer is registered first, since that is the backend whose options declare `projection`.
        """
        _, hv = _require_holoviz()
        if self._projection is None:
            return obj
        if (
            "matplotlib" not in hv.Store.renderers
        ):  # register mpl opts before applying them
            hv.renderer("matplotlib")
        return obj.opts(projection=self._projection, backend="matplotlib")

    def save(self, path: Any, **kwargs: Any) -> Path:
        """Save the composed map — interactive HTML (Bokeh) or a raster via the matplotlib backend.

        Args:
            path: Output file (``str`` or ``pathlib.Path``). ``*.html`` writes a self-contained
                interactive Bokeh page; any other suffix (``.png``/``.svg``/…) renders through
                HoloViews' matplotlib backend (headless, no browser/selenium needed).
            **kwargs: Forwarded to :func:`holoviews.save` (e.g. ``fmt``, ``dpi``).

        Returns:
            pathlib.Path: the file written — every tier's ``save`` returns a ``Path`` (#248), so the
            result can be opened, moved or asserted on without re-wrapping it.

        Raises:
            ImportError: when the ``interactive`` extra is not installed.

        Examples:
            - The suffix picks the backend — ``.html`` is interactive Bokeh, ``.png`` renders
              headless through matplotlib (needs the engine):
                ```python
                >>> import holoviews as hv                                   # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap      # doctest: +SKIP
                >>> m = InteractiveMap().add_element(hv.Points([(0, 0)]))    # doctest: +SKIP
                >>> m.save("map.html").name                                  # doctest: +SKIP
                'map.html'
                >>> m.save("map.png").suffix                                 # doctest: +SKIP
                '.png'

                ```
        """
        _, hv = _require_holoviz()
        obj = self.render()
        backend = "bokeh" if str(path).lower().endswith(".html") else "matplotlib"
        hv.save(obj, path, backend=backend, **kwargs)
        return Path(path)

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

    def close(self) -> None:
        """Let go of the in-memory data this map registered.

        The object registry is process-global and holds strong references, so a session that builds maps
        keeps every dataset they drew until something says otherwise. This is the caller saying so. Closing
        twice is harmless, and a map that registered nothing has nothing to forget.

        An `object:` source in a `figure_spec` captured from this map cannot be opened afterwards — which is
        what "closed" means for a figure whose data lived only in this process. Save the data and reference
        it by path to keep such a figure readable.

        The credentials held for keyed basemaps go too: they are kept off the figure so it can be written
        down without them, and a closed map has nothing left to draw with them.

        Examples:
            - A map that drew nothing closes quietly, with no engine installed:
                ```python
                >>> from digitalearth.interactive import InteractiveMap
                >>> InteractiveMap().close()

                ```

        See Also:
            __exit__: calls this on the way out of a ``with`` block.
        """
        forget_namespace(self._objects_ns)
        self._layer_keys.clear()

    def __enter__(self) -> Self:
        """Enter the runtime context, returning the map.

        Returns:
            The same map, so ``with InteractiveMap() as m:`` binds this object and :meth:`close` runs on
            the way out.
        """
        return self

    def __exit__(self, *exc: Any) -> None:
        """Let the map's in-memory data go on the way out of a ``with`` block.

        Args:
            *exc: The exception triple, ignored — closing is unconditional, as it is for a file.
        """
        self.close()

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
