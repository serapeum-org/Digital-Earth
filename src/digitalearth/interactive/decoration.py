"""DecorationMixin — tiles & cartographic features for :class:`~digitalearth.interactive.map.InteractiveMap`.

Owns ``tiles`` / ``features`` + the six named Natural-Earth layers (``coastlines`` / ``borders`` /
``land`` / ``ocean`` / ``lakes`` / ``rivers`` — the canonical cross-tier surface, #253), the
``legend``/``colorbar`` colour keys (DI.1c) and the ``text`` / ``labels`` annotations (DI.5); the full
provider catalog + custom WMTS lands in DI.10.

``legend`` and ``colorbar`` are **guides on a layer's colour encoding** since order 24 (#261), not ``show=``
toggles on whichever layer was added last: each records a
:class:`~digitalearth.base.spec.encoding.Guide` through
:meth:`~digitalearth.base.spec.style.Symbology.with_guide` and then applies the HoloViews options from that
record, so a key moves with its layer, goes away with it, and is carried by the figure. A layer whose colour
varies with nothing — a flat fill, a basemap, a coastline — carries no colour encoding and is refused a key
rather than given an empty one.

Tile basemaps (``gv.tile_sources``) and Natural-Earth features (``gv.feature``) are Web-Mercator-only in
Bokeh, so every decoration guards on the default ``crs=3857`` display CRS (``_require_web_mercator``) —
on any other CRS they would silently misalign with the pre-reprojected data layers. Constructing these
elements touches no network; tiles/coastline geometry is fetched by the renderer at display time.
"""

import warnings
from dataclasses import replace as with_fields
from typing import TYPE_CHECKING, Any, Callable, Optional, Self, Sequence

from loguru import logger

# DEFAULT_BASEMAP_PROVIDER is the shared cross-tier default (#247): one constant in base/, read by this
# tier and the web tier, so neither spells "CartoLight" for itself.
from digitalearth.base.basemaps import (
    DEFAULT_BASEMAP_PROVIDER,
    get_keyed_basemap,
    is_keyed_basemap,
)
from digitalearth.base.spec import Encoding, Guide, LayerSpec, Symbology
from digitalearth.base.spec.bounds import same_crs
from digitalearth.interactive.base import (
    _require_holoviz,
    _skips_off_limb,
    describe_opts,
    held_props,
)
from digitalearth.interactive.style_fold import (
    GUIDE_KIND_KEY,
    GUIDE_LABELS_KEY,
    fold_guide,
    split_guide_options,
)

# Note (#247): `tiles()` takes a provider *name*, a keyed preset and a raw XYZ *URL* through the one
# `provider=` argument. Splitting that collision (a `tiles(url=...)` vs `basemap(provider=...)` rename)
# is Core-contract work and is deliberately out of this batch.

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.interactive.base import InteractiveMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


def _attribution_hook(attribution: str) -> Callable[[Any, Any], None]:
    """Build a Bokeh plot hook that writes ``attribution`` onto the rendered tile source.

    The element's ``label`` is not an attribution slot — it is the element's display name, which Bokeh
    renders as the **plot title**. That is wrong twice over: on a bare tile map it silently replaces the
    caller's title (and changes the ``(group, label)`` key that ``.opts``/``.select`` match on), and the
    moment the tiles are overlaid with data the title comes from the overlay instead, so the attribution
    is displayed nowhere at all — which is the only way these tiles are ever actually used.

    Args:
        attribution: The text the licence requires the map to display.

    Returns:
        A hook that assigns the attribution to every tile source in the rendered figure, leaving the
        title untouched.
    """

    def hook(plot: Any, element: Any) -> None:
        """Assign the attribution to the Bokeh tile renderers of one rendered plot.

        Args:
            plot: The Bokeh plot HoloViews is building.
            element: The element being rendered; unused, but part of the hook signature.
        """
        for renderer in plot.handles["plot"].renderers:
            tile_source = getattr(renderer, "tile_source", None)
            if tile_source is not None:
                tile_source.attribution = attribution

    return hook


def _coverage_hook(bounds: tuple) -> Callable[[Any, Any], None]:
    """Build a Bokeh plot hook declaring a keyed provider's coverage as the pannable range (#233).

    A keyed service that covers only part of the world (NICFI's tropics band, say) serves 404s outside it.
    MapLibre takes that coverage as the source's ``bounds``; Bokeh's tile source has no such slot — its
    engine-level equivalent is the plot range's ``bounds``, which is what stops a viewer panning off the
    covered area into blank tiles. This is a **declaration**, not validation: nothing is refused here, and
    a pannable map is never rejected for opening outside the coverage (that guard belongs to the static
    tier, which renders one fixed extent).

    Args:
        bounds: ``(x0, y0, x1, y1)`` in the **display** CRS — the lon/lat coverage already reprojected.

    Returns:
        A hook that assigns the covered range to the rendered figure's x and y ranges.
    """

    def hook(plot: Any, element: Any) -> None:
        """Assign the covered extent to one rendered plot's axis ranges.

        Args:
            plot: The Bokeh plot HoloViews is building.
            element: The element being rendered; unused, but part of the hook signature.
        """
        x0, y0, x1, y1 = bounds
        figure = plot.handles["plot"]
        figure.x_range.bounds = (x0, x1)
        figure.y_range.bounds = (y0, y1)

    return hook


def _upper_placeholders(url: str) -> str:
    """Return ``url`` with ``{z}``/``{x}``/``{y}`` upper-cased, the spelling GeoViews' ``WMTS`` expects.

    This lives in the interactive backend rather than in :mod:`digitalearth.base.basemaps` because the
    casing is Bokeh's convention, not a property of the tile service — the web and static tiers use the
    lower-case template unchanged.

    Args:
        url: A tile URL template using the lower-case placeholders.

    Returns:
        The same string with every occurrence of the exact tokens ``{z}``, ``{x}`` and ``{y}`` upper-cased
        — wherever they appear, including a second copy inside the query string. Nothing else is touched,
        and no other placeholder is: a ``{mosaic}`` or ``{api_key}`` still reads as it did. Substitution
        happens before this, so a value that itself contained one of the three tokens would be rewritten
        too; :meth:`~digitalearth.base.basemaps.KeyedTileSource.tile_url` is what keeps such values out.
    """
    for lower, upper in (("{z}", "{Z}"), ("{x}", "{X}"), ("{y}", "{Y}")):
        url = url.replace(lower, upper)
    return url


def _provider_description(provider: Any) -> Optional[str]:
    """Return a tile provider in a spelling a figure can be written with, carrying no credential.

    A provider name is already one. An `xyzservices.TileProvider` is a `dict` whose fields include the
    service's API key, so the object itself is never written down: its `url` **template** is, which still
    names the service and which `_build_tiles` draws through the raw-URL path. A provider whose own fields
    appear verbatim in that template — one built with the key substituted in — is described as nothing,
    and a reader without the held object draws the tier's default basemap.

    Args:
        provider: The caller's `provider` argument.

    Returns:
        The name, the URL template, or `None`.
    """
    if isinstance(provider, str):
        return provider
    # Every string field but the four public ones is treated as a credential: `apikey`, `accessToken`,
    # whatever a service calls it. What is written must not depend on knowing each service's spelling.
    public = {"url", "name", "attribution", "html_attribution"}
    secrets = [
        value
        for key, value in dict(provider).items()
        if key not in public and isinstance(value, str) and value
    ]
    for spelling in (getattr(provider, "url", None), getattr(provider, "name", None)):
        if isinstance(spelling, str) and not any(
            secret in spelling for secret in secrets
        ):
            return spelling
    return None


#: What `xyzservices` writes into a field the caller has to fill in — `'<insert your apiKey here>'`. It is
#: the catalog's own marker for a credential, and `TileProvider.requires_token` reads it the same way.
TOKEN_PLACEHOLDER = "<insert your"


def _token_fields(provider: Any) -> list:
    """Return the field names a provider needs a credential in before its URL can be built.

    Args:
        provider: An `xyzservices.TileProvider`.

    Returns:
        The names, in catalog order — usually one (`apiKey`, `accessToken`, `apikey`), and empty for a
        service that needs none.
    """
    return [
        field
        for field, value in dict(provider).items()
        if isinstance(value, str) and TOKEN_PLACEHOLDER in value
    ]


def _catalogued_tiles(name: str, api_key: Any, *, known: Any) -> Any:
    """Resolve a provider name GeoViews' own catalog does not carry, through the `xyzservices` catalog.

    The second half of :func:`_provider_description`: that one decides what a figure may write a provider
    down as, and this one reads it back. A provider whose `url` repeats one of its own field values —
    Esri's polar maps (`variant="Arctic_Imagery"`), HERE's styles (`variant="explore.day"`) — is described
    by **name**, because the guard cannot tell such a template from one a key was substituted into. Only
    32 of the 880 catalogued services are in GeoViews' catalog, and it spells them without the dot, so
    those figures named a provider nothing here could resolve (review M9). The guard itself is unchanged:
    it leaks no credential across the catalog, and what was wrong was the resolution, not the writing.

    Args:
        name: The provider name a description carries, as `xyzservices` spells it.
        api_key: The credential held for this layer, or `None`.
        known: The GeoViews catalog, named in the refusal when nothing resolves the name — a caller
            reading it wants the names this tier offers, not all 880.

    Returns:
        The tile element. A keyed service is built from its filled-in template and carries its attribution
        as a plot hook, as a keyed preset does; a keyless one is handed over as the provider object, which
        is how GeoViews reads its attribution for itself.

    Raises:
        ValueError: when neither catalog knows the name — a typo must stay a refusal rather than become a
            blank basemap.
        ImportError: when the service needs a credential and none is held, matching the answer a keyed
            preset and the Stadia catalog entry already give.
    """
    import xyzservices.providers as xyz

    gv, _ = _require_holoviz()
    try:
        provider = xyz.query_name(name)
    except ValueError:
        # `xyzservices`' own message names its 880 services; this one names the catalog a caller chooses
        # from, which is the answer the tier has always given for an unknown name.
        raise ValueError(
            f"unknown tile provider {name!r} — choose one of: {', '.join(sorted(known))}"
        ) from None
    secrets = _token_fields(provider)
    if not secrets:
        return gv.WMTS(provider)
    if api_key is None:
        raise ImportError(
            f"tile provider {name!r} needs an api_key ({', '.join(secrets)}); pass "
            "tiles(provider, api_key=...)"
        )
    # Built rather than handed over, because the credential has to reach the template; the attribution
    # GeoViews would have read off the object is re-attached the way a keyed preset's is.
    url = provider.build_url(**dict.fromkeys(secrets, api_key))
    return gv.WMTS(_upper_placeholders(url)).opts(
        hooks=[_attribution_hook(provider.attribution)]
    )


def draw_tiles(interactive_map: Any, _data: Any, layer: LayerSpec) -> Any:
    """Build the tile basemap a description asks for.

    Args:
        interactive_map: The map being drawn.
        _data: Unused — tiles are fetched from a provider, not from a caller's source.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.

    Raises:
        ValueError: when the recorded provider names no tile service :meth:`DecorationMixin._build_tiles`
            knows, or carries preset keywords a non-keyed provider takes none of.
        ImportError: when the recorded provider needs a credential and the map holds none for this layer.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    _require_holoviz()
    props = held_props(interactive_map, layer)
    element = interactive_map._build_tiles(
        # `None` is a figure whose provider had no JSON spelling and whose reader holds no object: the
        # shared default basemap is what such a layer draws.
        props.get("provider") or DEFAULT_BASEMAP_PROVIDER,
        # The credential is the map's, not the figure's: a description that carried it would write it
        # out with the figure.
        interactive_map._layer_keys.get(layer.id),
        **dict(props.get("preset") or {}),
    )
    opts = dict(props.get("opts") or {})
    if opts:
        element = element.opts(**opts)
    element = element.opts(level=props["level"])
    return DrawnLayer(element=element, style=opts)


def draw_natural_earth(interactive_map: Any, _data: Any, layer: LayerSpec) -> Any:
    """Build one piece of Natural-Earth reference geography at the described resolution.

    Args:
        interactive_map: The map being drawn, whose held values carry the half of the caller's own
            keywords a description cannot spell; the plain half is read off the description itself.
        _data: Unused — reference geography is cut from Natural Earth, not from a caller's source.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    gv, _ = _require_holoviz()
    props = held_props(interactive_map, layer)
    # clone: .opts() would otherwise restyle the shared gv.feature singleton
    element = (
        getattr(gv.feature, props["feature"]).clone().opts(scale=props["resolution"])
    )
    opts = dict(props.get("opts") or {})
    if opts:
        element = element.opts(**opts)
    return DrawnLayer(element=element, style=opts)


def draw_labels(interactive_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the per-feature text labels a description asks for.

    Args:
        interactive_map: The map being drawn.
        data: The layer's source — the features whose column is drawn.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.

    Raises:
        KeyError: when the recorded column is absent from the reprojected frame. The builder refuses a
            column it can see is missing, but a features object that exposes no ``columns`` — a pyramids
            ``FeatureCollection`` rather than a GeoDataFrame — is only read here.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    gv, _ = _require_holoviz()
    props = held_props(interactive_map, layer)
    column = props["column"]
    gdf = interactive_map._display_gdf(data)
    # Build from explicit display-CRS x/y/text columns rather than handing GeoViews the
    # geometry-bearing GeoDataFrame: gv.Labels mis-projects a GeoDataFrame's point geometry at
    # render time (a boolean-mask length mismatch). The geometry is already in the display CRS
    # (reprojected by _display_gdf), so its x/y are the label anchors directly.
    element = gv.Labels(
        (gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy(), gdf[column].to_numpy()),
        kdims=["x", "y"],
        vdims=[column],
        crs=gv.util.process_crs(interactive_map.crs),
    )
    opts = dict(props.get("opts") or {})
    if opts:
        element = element.opts(**opts)
    return DrawnLayer(element=element, style=opts)


def draw_text(interactive_map: Any, _data: Any, layer: LayerSpec) -> Any:
    """Build the GeoViews text element for a described annotation.

    Args:
        interactive_map: The map being drawn.
        _data: Unused — an annotation has no source.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    gv, _ = _require_holoviz()
    props = held_props(interactive_map, layer)
    element = gv.Text(
        props["x"],
        props["y"],
        props["s"],
        crs=gv.util.process_crs(interactive_map.crs),
    )
    opts = dict(props.get("opts") or {})
    if opts:
        element = element.opts(**opts)
    return DrawnLayer(element=element, style=opts)


def draw_coastlines(interactive_map: Any, _data: Any, layer: LayerSpec) -> Any:
    """Build the GeoViews coastline feature at the described resolution.

    Args:
        interactive_map: The map being drawn, whose held values carry the half of the caller's own
            keywords a description cannot spell; the plain half is read off the description itself.
        _data: Unused — reference geography is cut from Natural Earth, not from a caller's source.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.interactive.renderer.DrawnLayer`.
    """
    from digitalearth.interactive.renderer import DrawnLayer

    gv, _ = _require_holoviz()
    props = held_props(interactive_map, layer)
    # clone: .opts() would otherwise restyle the shared gv.feature.coastline singleton
    element = gv.feature.coastline.clone().opts(scale=props["resolution"])
    opts = dict(props.get("opts") or {})
    if opts:
        element = element.opts(**opts)
    return DrawnLayer(element=element, style=opts)


class DecorationMixin(_MixinBase):
    """Decoration builders (DI.1c): tile basemaps, Natural-Earth features, legend/colorbar toggles.

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

    def tiles(
        self,
        provider: Any = DEFAULT_BASEMAP_PROVIDER,
        *,
        level: str = "underlay",
        api_key: Any = None,
        preset: Optional[dict] = None,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Self:
        """Add a web-tile basemap beneath the data layers (DI.1c + DI.10 catalog / custom WMTS).

        A keyed preset's coverage is **not** checked here, unlike the static tier. A Bokeh plot is pannable
        and zoomable, so there is no one extent to check against — refusing a NICFI basemap because the
        opening view sits outside the tropics would block a map the viewer can pan into. The static tier
        renders one fixed extent, where an out-of-coverage basemap is a dead end, which is where the guard
        lives.

        Args:
            provider: A ``geoviews.tile_sources`` provider name (``"CartoLight"``/``"OSM"``/
                ``"EsriImagery"``/…), defaulting to the shared
                :data:`~digitalearth.base.basemaps.DEFAULT_BASEMAP_PROVIDER` every tier reads (#247);
                a **keyed** preset name such as ``"Planet.NICFI"`` (see
                :mod:`digitalearth.base.basemaps`), whose credential is read from the environment; a raw
                XYZ/WMTS URL template (``"https://…/{Z}/{X}/{Y}.png"``); or an
                ``xyzservices.TileProvider``. A name GeoViews does not hold falls back to the
                `xyzservices` catalog, which is why :meth:`list_tile_providers` lists fewer names than
                this accepts — 32 against 880, measured.
            level: ``"underlay"`` (default) keeps tiles behind the data layers; ``"overlay"`` puts
                them on top (rare — e.g. a labels overlay).
            api_key: Credential for a keyed provider. For a keyed **preset** (``"Planet.NICFI"``) it is the
                service's API key and ``None`` reads the preset's environment variable; for a provider in
                GeoViews' own catalog that needs one (Stadia) it is only checked for presence, while one
                resolved from the `xyzservices` catalog has it substituted into the URL template, under
                whichever field that provider declares.
            preset: A keyed preset's own keywords, e.g. ``{"date": "2024-01", "flavour": "visual"}``. A
                dict rather than ``**kwargs`` so it cannot collide with a HoloViews style option.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the map is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden — before, the flag fell through ``**opts`` to HoloViews, which hid the
                element while the figure went on calling it visible (#327).
            **opts: Extra HoloViews style options applied to the tile element.

        Returns:
            The same map instance, so builder calls chain — the tile layer is drawn *beneath* the data,
            because the basemap band puts it there, whenever in the chain it was added.

        Examples:
            - Put a light Carto basemap beneath a raster:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().field(dem).tiles("CartoLight")         # doctest: +SKIP
                >>> type(m.layers[0]).__name__                                  # doctest: +SKIP
                'WMTS'

                ```
            - Use a keyed preset, whose credential comes from the environment:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> m = InteractiveMap().tiles(                                # doctest: +SKIP
                ...     "Planet.NICFI", preset={"date": "2024-01", "flavour": "visual"}
                ... )

                ```
            - Use a custom XYZ URL template:
                ```python
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> url = "https://a.tile.example/{Z}/{X}/{Y}.png"              # doctest: +SKIP
                >>> InteractiveMap().tiles(url).layers[0]                       # doctest: +SKIP

                ```

        Raises:
            ValueError: when the display CRS is not Web Mercator (Bokeh tiles are 3857-only), when a
                non-Mercator projection is active, or when a provider name is unknown.
            ImportError: when a keyed provider is requested without an ``api_key``.
        """
        _require_holoviz()
        # Both refusals stay with the builder: each answers the call the caller made.
        self._require_web_mercator("tiles")
        if getattr(self, "_projection", None) is not None:
            raise ValueError(
                "tiles() cannot compose with a non-Mercator projection() — Bokeh tiles render in "
                "Web-Mercator only. Drop the projection to use a tile basemap."
            )
        # An explicit tiles() call supersedes any provider passed to the constructor, so render()'s
        # one-shot hook does not also prepend a second basemap (L2).
        self._tiles_provider = None
        # A provider object is an `xyzservices.TileProvider`: a `dict` subclass whose fields include the
        # service's API key, so freezing it into the description wrote the key into every saved figure
        # (review R-C1) and thawed it back as a plain dict the engine cannot draw from (review R-H2). It is
        # held beside the layer, and described by its key-free URL template.
        held: dict = {"provider": provider}
        described_opts = describe_opts(held, opts)
        return self.add_layer(
            None,
            name=name,
            visible=visible,
            kind="basemap",
            # `level="overlay"` asks for a basemap drawn *over* the data — a labels-and-roads layer on top
            # of imagery, say — so it declares the band that overrides its kind's.
            band="overlay" if level == "overlay" else None,
            key=api_key,
            held=held,
            symbology=Symbology(
                props={
                    "via": "tiles",
                    "provider": _provider_description(provider),
                    "level": level,
                    "preset": dict(preset or {}),
                    "opts": described_opts,
                }
            ),
        )

    def _build_tiles(self, provider: Any, api_key: Any, **preset: Any) -> Any:
        """Resolve ``provider`` to a ``gv.WMTS``/``gv.Tiles`` element (name, URL, or xyzservices).

        A name is looked up in GeoViews' own catalog first and in the wider ``xyzservices`` one after it,
        so every name a figure can be **described** by is a name that can be **drawn** — which is what a
        provider written down as its name needs (review M9; see :func:`_catalogued_tiles`).

        Args:
            provider: A ``geoviews.tile_sources`` name, any ``xyzservices`` provider name
                (``"Esri.ArcticImagery"``), a keyed preset name, a raw ``{Z}/{X}/{Y}`` URL, or an
                ``xyzservices.TileProvider``.
            api_key: Credential for a keyed provider or preset.
            **preset: A keyed preset's own keywords, e.g. ``date`` / ``flavour``.

        Returns:
            The tile element (a fresh clone for catalog names — the shared instance is never mutated).

        Raises:
            ValueError: for a provider name neither catalog knows, or when preset keywords are passed with
                a provider that is not a keyed preset.
            ImportError: when a keyed provider needs an ``api_key`` that was not supplied.
        """
        gv, _ = _require_holoviz()
        if is_keyed_basemap(provider):
            # A keyed preset resolves to a URL with the credential already substituted; GeoViews wants the
            # tile placeholders upper-cased.
            keyed = get_keyed_basemap(str(provider), **preset)
            # NICFI is non-commercial-only, so the attribution is a licence obligation the other two
            # tiers already carry. Bokeh's slot for it is the tile source's own `attribution`, which is
            # what its attribution control renders; reaching it needs a plot hook.
            hooks = [_attribution_hook(keyed.attribution)]
            covered = self._display_bounds(keyed.bounds)
            if (
                covered is not None
            ):  # declare a partial-coverage service to the engine (#233)
                logger.info(
                    f"tiles({provider!r}): the provider covers {keyed.bounds} (lon/lat) only, so that "
                    "box is declared as the map's pannable range — data outside it will be framed by "
                    "the limit rather than by the data"
                )
                hooks.append(_coverage_hook(covered))
            return gv.WMTS(_upper_placeholders(keyed.tile_url(api_key))).opts(
                hooks=hooks
            )
        if preset:
            raise ValueError(
                f"tiles({provider!r}) takes no preset keywords; {sorted(preset)} apply only to a keyed "
                f"preset such as 'Planet.NICFI'"
            )
        if isinstance(provider, str) and "://" in provider:  # raw XYZ/WMTS URL template
            return gv.WMTS(provider)
        if isinstance(provider, str):
            sources = gv.tile_sources.tile_sources
            if provider not in sources:
                # Not in GeoViews' own catalog, which spells `Esri.ArcticImagery` as `EsriArcticImagery`
                # and carries 32 of the 880 services `xyzservices` does. A figure describes a provider by
                # name whenever its URL is one the credential guard refuses, so a name this tier cannot
                # resolve is a figure that writes cleanly and then raises when it is drawn back (review M9).
                return _catalogued_tiles(provider, api_key, known=sources)
            if "stadia" in provider.lower() and api_key is None:
                raise ImportError(
                    f"tile provider {provider!r} needs an api_key (Stadia/Stamen require a key for "
                    "non-local use); pass tiles(provider, api_key=...)"
                )
            return sources[provider].clone()
        return gv.WMTS(
            provider
        )  # an xyzservices.TileProvider (or any gv.WMTS-accepted object)

    def _display_bounds(self, bounds: Optional[tuple]) -> Optional[tuple]:
        """Reproject a keyed provider's lon/lat coverage into the display CRS (#233).

        Args:
            bounds: ``(west, south, east, north)`` in lon/lat, or ``None`` for a global service.

        Returns:
            ``(x0, y0, x1, y1)`` in the display CRS, or ``None`` when the service declares no coverage —
            a global service needs no declaration, so nothing is forwarded for it.
        """
        if not bounds:
            return None
        west, south, east, north = bounds
        (x0, x1), (y0, y1) = self._to_display_xy([west, east], [south, north], 4326)
        return (x0, y0, x1, y1)

    def list_tile_providers(self) -> list:
        """Return the sorted catalog of named ``geoviews.tile_sources`` providers.

        Returns:
            GeoViews' own catalog — 32 names, measured against the installed version. It is a *subset* of
            what :meth:`tiles` accepts, not the whole of it: a name GeoViews does not hold falls back to
            the `xyzservices` catalog, which carries 880. The narrower list is deliberate — a caller
            reading it wants the names this tier offers rather than every service in the world — so read
            it as "the catalogued names", not as "the accepted names".
        """
        gv, _ = _require_holoviz()
        return sorted(gv.tile_sources.tile_sources)

    def coastlines(
        self,
        resolution: str = "110m",
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Self:
        """Add the Natural-Earth coastline on top of the data layers.

        Args:
            resolution: Natural-Earth scale — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the map is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden — before, the flag fell through ``**opts`` to HoloViews, which hid the
                element while the figure went on calling it visible (#327).
            **opts: Extra HoloViews style options applied to the feature element.

        Returns:
            The same map instance, so builder calls chain.

        Examples:
            - Add a medium-resolution coastline on top of the data:
                ```python
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> m = InteractiveMap().coastlines(resolution="50m")           # doctest: +SKIP
                >>> len(m.layers)                                               # doctest: +SKIP
                1

                ```

        Raises:
            ValueError: when the display CRS is not Web Mercator.
        """
        _require_holoviz()
        # Refused here, because the message names the builder the caller called.
        self._require_web_mercator("coastlines")
        held: dict = {}
        described_opts = describe_opts(held, opts)
        return self.add_layer(
            None,
            name=name,
            visible=visible,
            kind="coastlines",
            held=held,
            symbology=Symbology(
                props={
                    "via": "coastlines",
                    "resolution": resolution,
                    "opts": described_opts,
                }
            ),
        )

    def features(
        self,
        *,
        land: bool = False,
        ocean: bool = False,
        borders: bool = False,
        rivers: bool = False,
        lakes: bool = False,
        resolution: str = "110m",
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Self:
        """Add Natural-Earth context layers (land/ocean/lakes beneath the data, borders/rivers on top).

        The flag-per-layer form, kept as the way to request several layers in one call. The six named
        methods (:meth:`coastlines`, :meth:`borders`, :meth:`land`, :meth:`ocean`, :meth:`lakes`,
        :meth:`rivers`) are the canonical cross-tier surface (#253) and delegate here.

        Args:
            land: Draw the land polygons (underlay).
            ocean: Draw the ocean polygons (underlay).
            borders: Draw country borders (overlay).
            rivers: Draw river centerlines (overlay).
            lakes: Draw lake polygons (underlay — the band the registry files ``lakes`` under, which is
                where :meth:`lakes` draws them: beneath the data, with land and ocean).
            resolution: Natural-Earth scale — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from each feature's kind. This call draws **one layer per
                requested feature**, so a single name is shared by all of them and the second and later
                take ``-2``, ``-3``, … — which is what :meth:`land`, :meth:`ocean` and their four
                siblings avoid, each asking for one feature and so taking the name exactly (#321).
            visible: Whether the requested features are drawn. ``False`` builds them hidden **and**
                describes them hidden — before, the flag fell through ``**opts`` to HoloViews, which hid
                the elements while the figure went on calling them visible (#327).
            **opts: Extra HoloViews style options applied to every requested feature element.

        Returns:
            The same map instance, so builder calls chain.

        Examples:
            - Underlay land and overlay country borders around a raster:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().field(dem).features(land=True, borders=True)  # doctest: +SKIP
                >>> len(m.layers)                                               # doctest: +SKIP
                3

                ```

        Raises:
            ValueError: when the display CRS is not Web Mercator.
        """
        _require_holoviz()
        self._require_web_mercator("features")
        held: dict = {}
        described_opts = describe_opts(held, opts)
        # `feature`, not `name`: `name` is the caller's own name for the layer, and this loop's variable
        # is the piece of reference geography — two different words that were spelled the same (#321).
        for feature, requested in (
            ("land", land),
            ("ocean", ocean),
            ("borders", borders),
            ("rivers", rivers),
            ("lakes", lakes),
        ):
            if requested:
                # Each is its own kind, so a figure says which piece of reference geography it drew —
                # and the registry, not this loop, decides where each is drawn.
                self.add_layer(
                    None,
                    name=name,
                    visible=visible,
                    kind=feature,
                    held=held,
                    symbology=Symbology(
                        props={
                            "via": "natural_earth",
                            "feature": feature,
                            "resolution": resolution,
                            "opts": described_opts,
                        }
                    ),
                )
        return self

    def borders(
        self,
        resolution: str = "110m",
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Self:
        """Overlay Natural-Earth country borders (one of the six named feature layers, #253).

        Args:
            resolution: Natural-Earth scale — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **opts: Extra HoloViews style options applied to the feature element.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when the display CRS is not Web Mercator.

        Examples:
            - Name the layer instead of spelling out a ``features`` flag:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> len(InteractiveMap().borders().layers)                     # doctest: +SKIP
                1

                ```
        """
        return self.features(
            borders=True, resolution=resolution, name=name, visible=visible, **opts
        )

    def land(
        self,
        resolution: str = "110m",
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Self:
        """Fill Natural-Earth land polygons beneath the data (one of the six named layers, #253).

        Args:
            resolution: Natural-Earth scale — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **opts: Extra HoloViews style options applied to the feature element.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when the display CRS is not Web Mercator.

        Examples:
            - Name the layer instead of spelling out a ``features`` flag:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> len(InteractiveMap().land().layers)                        # doctest: +SKIP
                1

                ```
            - Land is an **underlay**: its band draws it beneath the data, so a data layer added before
              it still ends up drawn on top:
                ```python
                >>> from pyramids.dataset import Dataset                       # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")       # doctest: +SKIP
                >>> m = InteractiveMap().field(dem).land()                     # doctest: +SKIP
                >>> [layer.group for layer in m.layers]                        # doctest: +SKIP
                ['Land', 'Image']

                ```
            - A finer Natural-Earth scale is one argument, and style options ride along:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> m = InteractiveMap().land(resolution="50m", alpha=0.3)     # doctest: +SKIP
                >>> len(m.layers)                                              # doctest: +SKIP
                1

                ```
        """
        return self.features(
            land=True, resolution=resolution, name=name, visible=visible, **opts
        )

    def ocean(
        self,
        resolution: str = "110m",
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Self:
        """Fill Natural-Earth ocean polygons beneath the data (one of the six named layers, #253).

        Args:
            resolution: Natural-Earth scale — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **opts: Extra HoloViews style options applied to the feature element.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when the display CRS is not Web Mercator.

        Examples:
            - Name the layer instead of spelling out a ``features`` flag:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> len(InteractiveMap().ocean().layers)                       # doctest: +SKIP
                1

                ```
            - Ocean is an **underlay** too, so it sits beneath data added before it — a background
              for a land-only raster rather than a mask over it:
                ```python
                >>> from pyramids.dataset import Dataset                       # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")       # doctest: +SKIP
                >>> m = InteractiveMap().field(dem).ocean()                    # doctest: +SKIP
                >>> [layer.group for layer in m.layers]                        # doctest: +SKIP
                ['Ocean', 'Image']

                ```
            - The two underlays compose, in the order they were asked for, both beneath the data:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> m = InteractiveMap().ocean().land()                        # doctest: +SKIP
                >>> [layer.group for layer in m.layers]                        # doctest: +SKIP
                ['Ocean', 'Land']

                ```
        """
        return self.features(
            ocean=True, resolution=resolution, name=name, visible=visible, **opts
        )

    def lakes(
        self,
        resolution: str = "110m",
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Self:
        """Fill Natural-Earth lake polygons beneath the data (one of the six named layers, #253).

        Args:
            resolution: Natural-Earth scale — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **opts: Extra HoloViews style options applied to the feature element.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when the display CRS is not Web Mercator.

        Examples:
            - Name the layer instead of spelling out a ``features`` flag:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> len(InteractiveMap().lakes().layers)                       # doctest: +SKIP
                1

                ```
            - Lakes are an **underlay**, like land and ocean: the registry files all three as ground
              cover, so a lake polygon is drawn beneath a data layer added before it rather than over it.
              Inland water that must read on top of an opaque raster is :meth:`rivers` — the line
              geography is the half of the pair drawn above the data:
                ```python
                >>> from pyramids.dataset import Dataset                       # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")       # doctest: +SKIP
                >>> m = InteractiveMap().field(dem).lakes()                    # doctest: +SKIP
                >>> [layer.group for layer in m.layers]                        # doctest: +SKIP
                ['Lakes', 'Image']

                ```
            - Chain the named layers to build the hydrography context in one line:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> len(InteractiveMap().lakes().rivers().coastlines().layers)  # doctest: +SKIP
                3

                ```
        """
        return self.features(
            lakes=True, resolution=resolution, name=name, visible=visible, **opts
        )

    def rivers(
        self,
        resolution: str = "110m",
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Self:
        """Overlay Natural-Earth river centerlines (one of the six named layers, #253).

        Args:
            resolution: Natural-Earth scale — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **opts: Extra HoloViews style options applied to the feature element.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when the display CRS is not Web Mercator.

        Examples:
            - Name the layer instead of spelling out a ``features`` flag:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> len(InteractiveMap().rivers().layers)                      # doctest: +SKIP
                1

                ```
            - Rivers are an **overlay**: their band draws them over the data, so the centerlines are drawn
              over the raster they describe:
                ```python
                >>> from pyramids.dataset import Dataset                       # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")       # doctest: +SKIP
                >>> m = InteractiveMap().field(dem).rivers(resolution="50m")   # doctest: +SKIP
                >>> [layer.group for layer in m.layers]                        # doctest: +SKIP
                ['Image', 'Rivers']

                ```
            - Style options reach the element, so the centerlines can be toned down:
                ```python
                >>> from digitalearth.interactive import InteractiveMap        # doctest: +SKIP
                >>> m = InteractiveMap().rivers(line_width=0.5, alpha=0.6)     # doctest: +SKIP
                >>> len(m.layers)                                              # doctest: +SKIP
                1

                ```
        """
        return self.features(
            rivers=True, resolution=resolution, name=name, visible=visible, **opts
        )

    def _to_display_xy(self, lon: Any, lat: Any, crs: Any) -> tuple:
        """Reproject ``(lon, lat)`` from ``crs`` to the display CRS via pyramids.

        Uses ``pyramids.feature.geometry.reproject_coordinates`` (pyramids owns the pyproj call —
        this package never imports pyproj/cartopy). A no-op when ``crs`` already equals the display
        CRS.

        Args:
            lon: X / longitude in ``crs`` (scalar or 1-D).
            lat: Y / latitude in ``crs`` (scalar or 1-D).
            crs: The CRS of ``lon``/``lat`` (EPSG int / string).

        Returns:
            ``(x, y)`` lists in the display CRS.
        """
        import numpy as np

        xs = np.atleast_1d(lon).astype(float).tolist()
        ys = np.atleast_1d(lat).astype(float).tolist()
        if same_crs(crs, self.crs):
            return xs, ys
        from pyramids.feature.geometry import reproject_coordinates

        return reproject_coordinates(xs, ys, from_crs=crs, to_crs=self.crs)

    def text(
        self,
        lon: Any,
        lat: Any,
        s: str,
        *,
        crs: Any = 4326,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Self:
        """Add a single text annotation at ``(lon, lat)`` (reprojected to the display CRS).

        Args:
            lon: X / longitude of the label anchor, in ``crs``.
            lat: Y / latitude of the label anchor, in ``crs``.
            s: The text to draw.
            crs: CRS of ``lon``/``lat`` (default EPSG:4326); reprojected to the display CRS via
                pyramids.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the map is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden — before, the flag fell through ``**opts`` to HoloViews, which hid the
                element while the figure went on calling it visible (#327).
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            The same map instance, so builder calls chain.
        """
        _require_holoviz()
        # Reprojected here, because `crs=` is the caller's own argument and this is where it is answered;
        # the element itself is built by `draw_text` from the display coordinates this records.
        (x,), (y,) = self._to_display_xy(lon, lat, crs)
        held: dict = {}
        described_opts = describe_opts(held, opts)
        return self.add_layer(
            None,
            name=name,
            visible=visible,
            kind="text",
            held=held,
            symbology=Symbology(
                props={
                    "via": "text",
                    "x": float(x),
                    "y": float(y),
                    "s": s,
                    "opts": described_opts,
                }
            ),
        )

    @_skips_off_limb
    def labels(
        self,
        features: Any,
        column: str,
        *,
        crs: Any = 4326,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Self:
        """Add per-feature text labels from a point ``FeatureCollection`` column.

        Args:
            features: A pyramids ``FeatureCollection`` of point geometries; reprojected to the
                display CRS through pyramids when needed.
            column: The attribute column whose values are drawn as labels.
            crs: Unused when ``features`` carries its own CRS (kept for signature symmetry with
                :meth:`text`); reprojection goes through ``FeatureCollection.to_crs``.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the map is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden — before, the flag fell through ``**opts`` to HoloViews, which hid the
                element while the figure went on calling it visible (#327).
            **opts: Extra HoloViews style options applied to the element.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            KeyError: when ``column`` is not a column of ``features``.
        """
        _require_holoviz()
        # Refused here, because the message names the column the caller asked for.
        if column not in getattr(features, "columns", [column]):
            raise KeyError(
                f"labels column {column!r} not found in the feature attributes"
            )
        held: dict = {}
        described_opts = describe_opts(held, opts)
        return self.add_layer(
            None,
            name=name,
            visible=visible,
            kind="labels",
            source=features,
            held=held,
            symbology=Symbology(
                props={"via": "labels", "column": column, "opts": described_opts}
            ),
        )

    def _colour_binding(self, layer_id: str) -> Optional[Encoding]:
        """Return what drives one layer's colour, when something varies it.

        Args:
            layer_id: A layer this map describes.

        Returns:
            Its ``color`` :class:`~digitalearth.base.spec.encoding.Encoding`, or `None` both when the layer
            publishes none and when the one it publishes is a **constant**. A flat colour is not something a
            key can explain — there are no values to label and no scale to sample — and the two answer as one
            here because the question a guide asks is whether there is anything to explain at all. The
            distinction matters: a caller's own ``color="#f00"`` is lifted onto the channel as a constant by
            :func:`~digitalearth.interactive.style_fold.portable_encodings`, so "publishes a colour encoding"
            on its own would have let a key onto a layer painted one colour.
        """
        encoding = self._layer_tree.get(layer_id).symbology.encoding("color")
        return None if encoding is None or encoding.is_constant else encoding

    def _guided_layer(self, method: str, layer_id: Optional[str]) -> str:
        """Resolve which layer a colour key belongs to, and refuse a layer that can carry none.

        Args:
            method: The public method asking, named in every refusal.
            layer_id: The caller's own ``layer_id``, or `None` for the default.

        Returns:
            The id of the layer the key describes.

            ``None`` means **the most recent layer that carries a colour encoding**. The care
            :meth:`~digitalearth.interactive.base.InteractiveMapBase._last_layer_index` took is kept — the
            layer the last builder call added, not the one drawn on top, so a raster added after a coastline
            is still the one this reaches, and an underlay added after data does not take it over — and the
            rule is tightened by exactly one thing: a layer with no colour encoding cannot take it over
            either. So `field(dem).coastlines().colorbar()` titles the raster, where the old `show=` toggle
            wrote a `colorbar` option onto the coastline.

            Where the layer added last carries no colour, the search falls back to the **topmost** layer in
            draw order that does. Within the data band draw order *is* the order the layers were added, and
            every kind that publishes a colour encoding is a data layer, so the fallback picks the
            last-added coloured layer for every figure this tier can build; it is written as a draw-order
            walk because that is the order the tree keeps and the only one it can be asked for.

        Raises:
            KeyError: when `layer_id` names no layer on this map, naming the ids that are.
            ValueError: when the named layer's colour is not driven by anything — a flat fill, a basemap, a
                coastline — and, for `None`, when no layer on the map has a colour a key could explain. An
                empty box would be worse than an error, which is the position the web tier's `legend` takes
                for the same question.
        """
        if layer_id is not None:
            self._require_layer(layer_id)
            if self._colour_binding(layer_id) is None:
                raise ValueError(
                    f"{method}(): layer {layer_id!r} draws no colour that varies with its data, so there is "
                    f"nothing for a colour key to explain; the layers that do are {self._guidable()}"
                )
            return layer_id
        held = self._last_layer_id
        # `held in self._layer_tree` before it is read, the guard `_note_last_layer` already applies: the
        # pointer is resolved by looking the id up, and a stale one raised `KeyError` out of the tree where
        # these methods document a refusal naming themselves (review N8).
        if (
            held is not None
            and held in self._layer_tree
            and self._colour_binding(held) is not None
        ):
            return held
        guidable = self._guidable()
        if not guidable:
            raise ValueError(
                f"{method}() needs a layer whose colour varies with its data — add a field(), a contours() "
                "or a builder given column=, or name one with layer_id="
            )
        return guidable[-1]

    def _guidable(self) -> list:
        """Return the ids of the layers a colour key could explain, in draw order.

        Returns:
            One id per layer whose colour is driven by a field, bottom of the draw order first. Read by
            :meth:`_guided_layer` both to pick the default and to name the alternatives in its refusals, so
            the answer and the message cannot describe different sets.
        """
        return [
            layer_id
            for layer_id in self._layer_tree.ids
            if self._colour_binding(layer_id) is not None
        ]

    def _record_guide(
        self,
        layer_id: str,
        guide: Guide,
        *,
        kind: str,
        labels: Optional[Sequence[str]] = None,
    ) -> LayerSpec:
        """Attach `guide` to one layer's colour encoding, and hand back the layer as it now reads.

        **The record is the point of order 24.** A colour key used to be a `show=` option written onto the
        element of whichever layer was added last and then forgotten: it could not move with its layer, could
        not go away with it, and was in no figure. Attached to the layer's own
        :class:`~digitalearth.base.spec.style.Symbology` it does all three for free, because the symbology is
        already what travels — `remove_layer` takes the guide with the layer, `set_visible` leaves it on a
        layer that comes back, and `to_dict()`/`from_dict()` carry it.

        Written **straight into the tree** rather than through
        :meth:`~digitalearth.interactive.base.InteractiveMapBase.replace_layer`. A guide is description and
        nothing else — the drawn key is the `.opts()` the caller's method applies next — so there is no
        difference for a renderer to reconcile, and going through `_change` would restyle the layer, redraw
        it (re-reading its source), and hand back a *new* element: the styles
        :attr:`~digitalearth.interactive.base.InteractiveMapBase._styles` files by ``id()`` and any stream a
        `hover()` attached are both keyed on the object this would have replaced. The panel `figure_spec`
        reports is derived from the tree on every read, so it follows this write without being told.

        **What is recorded beside the guide is what the redraw needs.** A `Guide` says whether, what the key
        is called and where — never which furniture, because the tiers that draw one kind of key need no such
        field. This tier draws two, so the kind and a caller's own row labels go into the symbology's
        ``props`` (:data:`~digitalearth.interactive.style_fold.GUIDE_KIND_KEY`,
        :data:`~digitalearth.interactive.style_fold.GUIDE_LABELS_KEY`), which is what lets
        :func:`~digitalearth.interactive.style_fold.fold_guide` build the same options again when the layer
        is drawn again — the static tier records the same two under the same names.

        Args:
            layer_id: The layer to explain. Already resolved and checked by :meth:`_guided_layer`.
            guide: What to say about its colour.
            kind: Which key it asks for — ``"colorbar"`` or ``"legend"``
                (:data:`~digitalearth.interactive.style_fold.GUIDE_KINDS`).
            labels: The caller's own row labels, or `None` to leave the derived one. `None` also *clears*
                labels an earlier call recorded, so one call's override does not outlive it.

        Returns:
            The layer as the tree now holds it, so the options applied next are folded **from the record**
            and cannot say something the figure does not.
        """
        layer = self._layer_tree.get(layer_id)
        props = dict(layer.symbology.props)
        props[GUIDE_KIND_KEY] = kind
        if labels is None:
            props.pop(GUIDE_LABELS_KEY, None)
        else:
            props[GUIDE_LABELS_KEY] = tuple(str(label) for label in labels)
        symbology = with_fields(layer.symbology, props=props).with_guide(guide)
        described = with_fields(layer, symbology=symbology)
        self._layer_tree = self._layer_tree.replace(described)
        return self._layer_tree.get(layer_id)

    def _draw_guide(self, layer_id: str, method: str, **opts: Any) -> None:
        """Apply one layer's colour-key options to the element it is drawn as.

        Args:
            layer_id: The layer whose element takes the options.
            method: The public method asking, named in the warning below.
            **opts: The HoloViews options the recorded guide resolved to.

        Warns:
            UserWarning: for an option this layer's element does not accept. Both keys are real: an
                ``hv.Image`` takes ``colorbar``, ``clabel`` and ``show_legend`` but **not** ``legend_opts``
                or ``legend_labels`` (measured against the registered Bokeh options), because a Bokeh legend
                is built from an overlay's labelled members and a raster contributes no label to name. Saying
                so is the choice :func:`~digitalearth.interactive.renderer._show` already made for the same
                shape of question — a caller who asked for something the engine cannot draw hears about it
                rather than watching it vanish.
        """
        index = self._layer_tree.ids.index(layer_id)
        element = self.layers[index]
        taken, refused = split_guide_options(element, opts)
        if refused:
            warnings.warn(
                f"{method}(): layer {layer_id!r} is drawn as a {type(element).__name__}, which Bokeh gives "
                f"no {', '.join(refused)} to draw; the rest of the key is applied and the guide is recorded",
                UserWarning,
                stacklevel=3,
            )
        if taken:
            self.layers[index] = element.opts(**taken, backend="bokeh")

    def colorbar(
        self,
        layer_id: Optional[str] = None,
        *,
        label: Optional[str] = None,
        visible: bool = True,
    ) -> Self:
        """Show the continuous colour key of a layer (the Core name, #261, order 24).

        A guide on the layer's own colour encoding rather than a ``show=`` toggle on whichever layer was
        added last: it is recorded first and the HoloViews options are applied **from the record**, so the key
        moves with its layer, goes away with it, and is carried by the figure.

        Bokeh draws **one** colorbar per plot, whatever the opts say. Measured: two guided rasters overlaid,
        each with ``colorbar=True`` and its own ``clabel``, render a single ``ColorBar`` — the one belonging
        to the layer lowest in draw order. So guiding a second layer does not stack a second bar; it decides
        which layer's bar is the one drawn. That is the same one-key-per-panel limit the web tier states from
        the other direction ("a second legend replaces the first"), and it is the engine's, not this tier's.

        Args:
            layer_id: Which layer's key to show. `None` (default) takes the most recent layer whose colour
                varies with its data — see :meth:`_guided_layer` for exactly what "most recent" means, which
                is the care the old toggle took plus one tightening.
            label: What to call the key — the variable and its units, usually. It reaches the drawn bar as
                HoloViews' ``clabel``, and is recorded as the guide's title. `None` leaves whatever the
                builder resolved (``field`` takes its own ``clabel=``, defaulting to the variable's units).
            visible: Whether the key is drawn. `False` records the guide switched **off** and applies
                ``colorbar=False``, so a caller passing a flag through does not have to branch — and the
                layer is resolved and checked before the flag is read, so one spelling of a call is not valid
                only half the time (the web tier's review L7).

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            KeyError: when `layer_id` names no layer on this map.
            ValueError: when the named layer's colour varies with nothing, or — for `None` — when no layer's
                does. Also for a ``label`` that is not a non-empty string, which
                :class:`~digitalearth.base.spec.encoding.Guide` refuses.
            ImportError: when the ``interactive`` extra is not installed.

        Examples:
            - Title the raster's colour key:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().field(dem).colorbar(label="Flow (m³/s)")  # doctest: +SKIP
                >>> m.get_layer(m.layer_ids[0]).symbology.guide().title         # doctest: +SKIP
                'Flow (m³/s)'

                ```
            - Drop it again, by name:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().field(dem, name="flow")                # doctest: +SKIP
                >>> m.colorbar("flow", visible=False).save("m.html").name       # doctest: +SKIP
                'm.html'

                ```

        See Also:
            digitalearth.base.spec.style.Symbology.with_guide: where the guide is attached.
        """
        # Called for its actionable ImportError before anything else: every option below is applied for the
        # Bokeh backend, and `_require_holoviz` is what registers it.
        _require_holoviz()
        target = self._guided_layer("colorbar", layer_id)
        described = self._record_guide(
            target, Guide(show=bool(visible), title=label), kind="colorbar"
        )
        self._draw_guide(
            target, "colorbar", **fold_guide(described.symbology, described.label)
        )
        return self

    def legend(
        self,
        layer_id: Optional[str] = None,
        *,
        title: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
        visible: bool = True,
    ) -> Self:
        """Show the keyed colour list of a layer (the Core name, #261, order 24).

        The same guide :meth:`colorbar` records, drawn as Bokeh's keyed box instead of a continuous bar. Both
        methods explain one layer's ``color`` encoding and so write one guide: calling both on a layer leaves
        the last title and the last ``visible``, and draws both keys.

        Bokeh builds **one** legend per plot, from the labelled members of the overlay, so ``show_legend``
        decides whether this layer *contributes a row* rather than whether it gets a box of its own —
        measured: a single element with ``show_legend=True`` renders no legend at all, because there is no
        overlay to build one from. A layer's row is captioned by its ``name=``, which is what ``labels``
        renames.

        Args:
            layer_id: Which layer's key to show. `None` (default) is the most recent layer whose colour
                varies with its data, exactly as :meth:`colorbar` resolves it.
            title: Heading above the key. It reaches the drawn box through Bokeh's own ``Legend.title``, and
                is recorded as the guide's title.
            labels: Explicit row labels, replacing the derived one. A layer contributes **one** row to a
                Bokeh legend — its own label — so this takes one entry; a count that does not match is
                refused rather than zipped, which would drop the difference and leave rows out of the key
                (the web tier's rule, for the same reason).
            visible: Whether the layer contributes to the key. `False` records the guide switched off and
                applies ``show_legend=False``. Checked after the layer is resolved, never before.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            KeyError: when `layer_id` names no layer on this map.
            ValueError: when the named layer's colour varies with nothing, or — for `None` — when no layer's
                does; when ``labels`` does not hold one entry per row; and for a ``title`` that is not a
                non-empty string.
            ImportError: when the ``interactive`` extra is not installed.

        Examples:
            - Key a contour layer, and title the box:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().contours(dem).legend(title="Contours")  # doctest: +SKIP
                >>> m.get_layer(m.layer_ids[0]).symbology.guide().show          # doctest: +SKIP
                True

                ```
            - Or take it off again:
                ```python
                >>> from pyramids.dataset import Dataset                        # doctest: +SKIP
                >>> from digitalearth.interactive import InteractiveMap         # doctest: +SKIP
                >>> dem = Dataset.read_file("examples/data/acc4000.tif")        # doctest: +SKIP
                >>> m = InteractiveMap().contours(dem).legend(visible=False)    # doctest: +SKIP
                >>> m.save("m.html").name                                       # doctest: +SKIP
                'm.html'

                ```

        See Also:
            digitalearth.base.spec.style.Symbology.with_guide: where the guide is attached.
        """
        _require_holoviz()
        target = self._guided_layer("legend", layer_id)
        rows = None if labels is None else list(labels)
        derived = [self._layer_tree.get(target).label]
        if rows is not None and len(rows) != len(derived):
            # Refused before the guide is recorded, so a rejected call leaves the description untouched.
            raise ValueError(
                f"legend(labels=...) has {len(rows)} entries and layer {target!r} contributes "
                f"{len(derived)} row to a Bokeh legend ({derived[0]!r}); zip would drop the difference and "
                "leave rows out of the key"
            )
        described = self._record_guide(
            target, Guide(show=bool(visible), title=title), kind="legend", labels=rows
        )
        self._draw_guide(
            target, "legend", **fold_guide(described.symbology, described.label)
        )
        return self
