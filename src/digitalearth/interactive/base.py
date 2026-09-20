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
from digitalearth.base.deprecation import renamed_parameter
from digitalearth.base.display import (
    auto_cmap,
    needs_reproject,
    to_display_source,
)
from digitalearth.base.sources import get_source
from digitalearth.base.sources.source import Source
from digitalearth.base.spec.bounds import same_crs

# `DEFAULT_BIG_DATA_THRESHOLD` is imported above rather than declared here: the row/face count above which a
# vector builder auto-routes to its tier's big-data renderer (#250) lives in `digitalearth.base.bigdata`, so
# this tier and the web tier cut over at the same size. One number for the whole tier either way: the map
# carries it as an attribute, and every big-data builder takes a per-call override of the same name, so a map
# configured once keeps the setting for every later layer. It stays importable from this module because that
# is where this tier's callers and tests already reach for it.

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
        layers: Registered HoloViews/GeoViews elements, in add (= overlay) order.

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
        # Class breaks / categories from the most recent choropleth, for building a legend out-of-band
        # (web-tier parity). None until a categorical choropleth runs; the continuous ramp resets it to None.
        self.last_breaks: Optional[List[Any]] = None
        self.layers: List[Any] = []
        # Style record written by `_styled`, keyed by `id()` of the element it returned (the object the
        # builders register, so the key stays alive for as long as the layer does). Read back through the
        # public `style_of` / `layer_styles` accessors — the tier owns its styling state instead of
        # delegating it to HoloViews' global option Store.
        self._styles: Dict[int, dict] = {}

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

    def add_element(self, element: Any) -> Self:
        """Register a HoloViews/GeoViews ``element`` as a layer and return ``self`` (chainable).

        The low-level entry point the capability mixins build on — every builder method ends here.

        An object you build yourself is a **custom layer**: what a figure keeps is its description — an id, the
        kind `custom:holoviews`, a label, a band and whether it is visible — and never the object, which has no
        description to write. The object stays with the scene that was handed it, so a figure saved and loaded
        again names the layer but cannot rebuild it, and another backend cannot draw it at all
        (:mod:`digitalearth.base.custom` says which case a reader is in). The tier records custom layers in its
        layer tree as its seam lands (#300); until then the object is drawn and nothing else is kept.

        Args:
            element: Any HoloViews/GeoViews element (or overlay-able object).

        Returns:
            The same map instance, so builder calls chain: ``m.image(dem).tiles().coastlines()``.

        Examples:
            - Registration appends in order and returns the map for chaining (any object can
              stand in for a HoloViews element here — the registry does not inspect it):
                ```python
                >>> from digitalearth.interactive import InteractiveMap
                >>> m = InteractiveMap()
                >>> m.add_element("raster-layer").add_element("vector-layer") is m
                True
                >>> m.layers
                ['raster-layer', 'vector-layer']

                ```
            - With the engine installed, real elements register the same way:
                ```python
                >>> import holoviews as hv                       # doctest: +SKIP
                >>> m = InteractiveMap()                         # doctest: +SKIP
                >>> m.add_element(hv.Points([(0, 0)])).layers    # doctest: +SKIP
                [:Points   [x,y]]

                ```
        """
        self.layers.append(element)
        return self

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
        common = {
            key: value for key, value in (common or {}).items() if value is not None
        }
        if common:
            element = element.opts(**common)
        frame: dict = {"width": self.width, "height": self.height}
        if self.title:
            frame["title"] = self.title
        frame.update(bokeh or {})
        element = element.opts(backend="bokeh", **frame)
        self._record_style(
            element, {"common": dict(common), "bokeh": dict(frame)}, owner
        )
        return element

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
        """The recorded style of every registered layer, in add (= overlay) order.

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
            - With the engine installed, each builder's requested options show up in add order:
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
            The single element when one layer is registered, an ``hv.Overlay`` of all layers in add
            order otherwise (an empty map renders as a blank ``hv.Overlay``).

        Raises:
            ImportError: when the ``interactive`` extra is not installed.

        Examples:
            - Two registered layers compose into an overlay in add order (needs the engine):
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
        gv, hv = _require_holoviz()
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
