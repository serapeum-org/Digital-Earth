"""RasterMixin — raster field rendering: imshow/contourf/contour/pcolormesh/block, composites, spaghetti.

Wires a pyramids ``Dataset`` (reprojected to the display CRS by the base) into cleopatra ``ArrayGlyph`` field
renders, plus the RGB/HSV composites and the ensemble spaghetti overlay.
"""

from math import isfinite
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph, RgbBands
from matplotlib import colormaps

from digitalearth.base.autostyle import auto_style
from digitalearth.base.display import auto_cmap
from digitalearth.base.preprocess import add_cyclic_column
from digitalearth.base.sources import get_stack
from digitalearth.base.spec import Bounds, LayerSpec, Symbology
from digitalearth.base.spec._serial import thawed_value
from digitalearth.base.stretch import (
    DEFAULT_COMPOSITE_BANDS,
    ChannelLimits,
    require_three_bands,
    stretch_to_unit,
)
from digitalearth.static.maps.base import OffLimbError
from digitalearth.static.render_compat import relocate_flat_style
from digitalearth.static.renderer import DrawnLayer, drawing_opts
from digitalearth.static.scene import LayerRecord

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.static.maps.base import GeoLayerBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object

#: Colormap a field falls back to when ``cmap=None`` and ``auto_style`` resolves none — the tier's literal,
#: kept *behind* the lookup rather than in a signature, so one variable-driven default decides the colour of
#: every backend's render of the same data.
DEFAULT_FIELD_CMAP = "viridis"

#: cleopatra's render kind -> the registered layer kind that render *is* (#303). The two vocabularies are not
#: the same word for the same thing: ``"contour"`` names a matplotlib call, ``"contours"`` names iso-value
#: lines traced from a band, and ``block`` draws a ``pcolormesh`` while being its own entry point. The figure
#: records the second, so a description written here is readable by a tier that has never heard of cleopatra.
FIELD_KINDS = {
    "imshow": "raster",
    "pcolormesh": "mesh",
    "contour": "contours",
    "contourf": "filled_contours",
}

#: Render kinds that draw contour lines/bands, i.e. the ones a resolved ``levels`` describes. ``auto_style``
#: contributes canonical contour levels for the operational fields it recognises (ECMWF Magics), and those
#: belong to a contour render — applying them to ``imshow``/``pcolormesh`` would quietly band a continuous
#: field the caller asked to see continuously.
_CONTOUR_KINDS = frozenset({"contour", "contourf"})


def _described_cmap(cmap: Any) -> Tuple[Any, Any]:
    """Split a colormap argument into the name a description records and the object held beside the layer.

    Args:
        cmap: What the caller asked for — a name, a ``Colormap``, or ``None`` to resolve one from the data.

    Returns:
        ``(recorded, held)``. A registered colormap is recorded by **name**: the name resolves to the same
        colours wherever the figure is read, and a `Colormap` object has no JSON form. One the caller built
        themselves resolves nowhere, so it is recorded as ``None`` — a figure read back elsewhere draws with
        the variable's own colormap — and held beside the layer, which is what colours this drawing.
    """
    if cmap is None or isinstance(cmap, str):
        return cmap, None
    name = getattr(cmap, "name", None)
    registered = (
        colormaps[name] if isinstance(name, str) and name in colormaps else None
    )
    if registered is not None and registered == cmap:
        return name, None
    return None, cmap


def _described_limits(limits: Optional[ChannelLimits]) -> Any:
    """Return stretch limits in the spelling a description carries: an unmeasurable bound as ``None``.

    :func:`~digitalearth.base.stretch.channel_limits` documents ``(nan, nan)`` for a channel it could not
    measure, and JSON has no spelling for ``nan`` — so a composite frozen on such a stack could not be
    written at all. ``None`` says the same thing in a form the description carries.

    Args:
        limits: The caller's per-channel ``(lo, hi)`` pairs, or ``None`` for a per-call scan.

    Returns:
        The pairs with every non-finite bound as ``None``, or ``None``.
    """
    if limits is None:
        return None
    return tuple(
        tuple(
            None if value is None or not isfinite(float(value)) else float(value)
            for value in pair
        )
        for pair in limits
    )


def _drawing_limits(limits: Any) -> Any:
    """Return recorded limits in the spelling ``stretch_to_unit`` reads, ``None`` back as ``nan``.

    Args:
        limits: The recorded pairs, or ``None``.

    Returns:
        The pairs as floats, a recorded ``None`` bound back as ``nan`` — which is what tells the stretch to
        fall back to this frame's own percentile for that channel.
    """
    if limits is None:
        return None
    return [
        tuple(float("nan") if value is None else float(value) for value in pair)
        for pair in limits
    ]


def draw_field(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Render the raster field a described layer asks for, through ``cleopatra.ArrayGlyph``.

    The one drawer behind ``imshow``/``pcolormesh``/``contour``/``contourf`` — four registered kinds drawn
    by one recipe, told apart by the ``via`` the builder recorded, which is cleopatra's own render kind.

    Args:
        scene: The map being drawn on.
        data: The pyramids ``Dataset`` the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the glyph's mappable.

    Raises:
        OffLimbError: when the data lies entirely outside what the display CRS shows. The builder answers
            it by skipping the layer (or, under ``strict``, letting it out).
    """
    props = dict(layer.symbology.props)
    kind = props["via"]
    opts = drawing_opts(scene, layer)
    src = scene._prepare(data, props["band"])
    z_values, x_values, y_values = src.z.values, src.x.values, src.y.values
    if opts.pop(
        "cyclic", False
    ):  # close the antimeridian seam for global fields (T5.2)
        z_values, x_values = add_cyclic_column(z_values, x_values)
    # Per-variable defaults (T6.2): the colormap, the canonical contour levels, and the units that label a
    # colorbar the caller did not label itself.
    style = auto_style(src)
    # A colormap the caller built themselves has no name a description can carry, so it is held beside the
    # layer; a named one was recorded. Either way the caller's choice wins over the variable's own.
    requested = opts.pop("cmap", props["cmap"])
    opts["cmap"] = auto_cmap(
        src, requested, props["default_cmap"], lookup=lambda _: style
    )
    levels = thawed_value(props["levels"])
    if levels is None and kind in _CONTOUR_KINDS:
        levels = style.get("levels")  # the variable's canonical contour levels
    if levels is not None:
        opts["levels"] = levels
    # Geo-reference the data. cleopatra honours `extent` only for imshow (bbox order
    # [xmin, ymin, xmax, ymax]); contour/contourf/pcolormesh plot in array-index space unless given
    # `coords=(x, y)`. The two are mutually exclusive, so pick by kind — otherwise the field lands at the
    # wrong location/zoom.
    if kind == "imshow":
        placement = {"extent": scene._extent_of(x_values, y_values)}
    else:
        placement = {"coords": (x_values, y_values)}
    # cleopatra's glyph constructors reject the regrouped styling keys (levels/style/color_scale/…);
    # relocate them onto the plot() call, where `_render_glyph` folds them into their group objects.
    plot_style = relocate_flat_style(opts)
    glyph = ArrayGlyph(
        z_values,
        exclude_value=[float("nan")],
        ax=scene.ax,
        fig=scene.fig,
        **placement,
        **opts,
    )
    drawn: DrawnLayer = scene._render_glyph(
        glyph,
        kind=kind,
        add_colorbar=props["add_colorbar"],
        label=style.get("units"),  # the Scene's colorbar labels itself with it (T6.2)
        **plot_style,
    )
    zorder = props["zorder"]
    if zorder is not None and drawn.artist is not None:
        # Recorded rather than set by the builder afterwards: a backdrop drawn again from its
        # description has to land behind the data again, and the builder is not there the second time.
        drawn.artist.set_zorder(zorder)
    return drawn


def _composite_bands(scene: Any, data: Any, props: Dict[str, Any]) -> tuple:
    """Return the reprojected dataset and its three stretched channels, band-last.

    The half ``rgb_composite`` and ``hsv_composite`` share: reproject once, read the three bands the
    description names, and stretch them onto ``0-1`` with whatever limits it froze.

    Args:
        scene: The map being drawn on.
        data: The multiband pyramids ``Dataset``.
        props: The layer's recorded properties, holding ``bands``, ``mask_nodata`` and ``limits``.

    Returns:
        ``(dataset, stretched)`` — the display-CRS dataset and a ``(rows, cols, 3)`` float array in ``0-1``.

    Raises:
        OffLimbError: when the data lies entirely outside what the display CRS shows.
    """
    bands = list(props["bands"])
    ds = scene._reproject(data)
    # (rows, cols, n); nodata -> NaN unless mask_nodata=False
    stack = get_stack(ds, bands, mask=props["mask_nodata"])
    return ds, stretch_to_unit(stack, _drawing_limits(props["limits"]))


def _composite_glyph(scene: Any, ds: Any, band_first: Any, opts: Dict[str, Any]) -> Any:
    """Build the ``ArrayGlyph`` that draws a three-channel composite as one image.

    Args:
        scene: The map being drawn on.
        ds: The display-CRS dataset, whose cells place the image.
        band_first: The channels as ``(3, rows, cols)`` — cleopatra's ``RgbBands`` path transposes them
            back to band-last for ``imshow``.
        opts: The constructor keywords, already stripped of the regrouped styling keys.

    Returns:
        The glyph, ready to plot.
    """
    return ArrayGlyph(
        band_first,
        rgb_bands=RgbBands(list(range(band_first.shape[0]))),
        extent=scene._extent(ds),
        ax=scene.ax,
        fig=scene.fig,
        **opts,
    )


def draw_rgb_composite(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Compose the three bands a described true/false-colour layer asks for into one image.

    Args:
        scene: The map being drawn on.
        data: The multiband pyramids ``Dataset`` the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the image mappable.

    Raises:
        OffLimbError: when the data lies entirely outside what the display CRS shows.
    """
    props = dict(layer.symbology.props)
    opts = drawing_opts(scene, layer)
    ds, stretched = _composite_bands(scene, data, props)
    # cleopatra's RgbBands path is band-FIRST: it does array[indices].transpose(1, 2, 0), so feed
    # (n, rows, cols) and let it transpose back to (rows, cols, n) for imshow.
    band_first = np.moveaxis(stretched, -1, 0)
    plot_style = relocate_flat_style(opts)
    drawn: DrawnLayer = scene._render_glyph(
        _composite_glyph(scene, ds, band_first, opts), **plot_style
    )
    return drawn


def draw_hsv_composite(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Read the three bands a described HSV layer asks for as hue/saturation/value and draw the RGB.

    Args:
        scene: The map being drawn on.
        data: The multiband pyramids ``Dataset`` the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the image mappable.

    Raises:
        OffLimbError: when the data lies entirely outside what the display CRS shows.
    """
    from matplotlib.colors import hsv_to_rgb

    props = dict(layer.symbology.props)
    opts = drawing_opts(scene, layer)
    ds, stretched = _composite_bands(scene, data, props)
    rgb = hsv_to_rgb(stretched)  # (rows, cols, 3) RGB
    # band-FIRST for cleopatra's RgbBands path (see draw_rgb_composite); it transposes back to band-last.
    band_first = np.moveaxis(rgb, -1, 0)
    plot_style = relocate_flat_style(opts)
    drawn: DrawnLayer = scene._render_glyph(
        _composite_glyph(scene, ds, band_first, opts), **plot_style
    )
    return drawn


class RasterMixin(_MixinBase):
    """Raster field renders and composites for :class:`~digitalearth.static.map.Map`."""

    def _field(
        self,
        dataset: Any,
        *,
        kind: str,
        band: int = 1,
        cmap: Optional[str] = None,
        levels: Any = None,
        add_colorbar: bool = False,
        default_cmap: str = DEFAULT_FIELD_CMAP,
        draw_band: Optional[str] = None,
        zorder: Optional[float] = None,
        **opts,
    ) -> Any:
        """Render a raster ``dataset`` on the shared axes via ``cleopatra.ArrayGlyph`` (the canonical recipe).

        The colour treatment is resolved here rather than defaulted in a signature: with ``cmap=None`` the
        field's own metadata decides it, through :func:`~digitalearth.base.autostyle.auto_style`, and
        ``default_cmap`` is what answers when that lookup has no opinion. The same lookup supplies the
        contour ``levels`` and the colorbar label (``units``) a caller did not pass — a caller-supplied
        value always wins.

        Args:
            dataset: A pyramids ``Dataset`` (reprojected to :attr:`crs` first).
            kind: ArrayGlyph render kind (``auto``/``imshow``/``pcolormesh``/``contour``/``contourf``).
            band: 1-based band index.
            cmap: Colormap name, or ``None`` (default) to resolve one from the variable via ``auto_style``.
            levels: Discrete levels (int or sequence of edges), or ``None`` to take the canonical levels
                ``auto_style`` resolved for the variable — on a contour render only.
            add_colorbar: When ``False`` (default) the Scene owns the colorbar, not the glyph.
            default_cmap: Colormap used when ``cmap`` is ``None`` *and* ``auto_style`` resolves none.
            draw_band: Where the layer is drawn in the figure's description, overriding the band its kind
                declares — ``"underlay"`` for a backdrop, which is what
                :meth:`~digitalearth.static.maps.decoration.DecorationMixin.stock_img` draws. ``None``
                (default) takes the kind's own band.
            zorder: The matplotlib draw order to give the render, or ``None`` (default) for whichever
                one the glyph leaves it with. The companion of `draw_band` on the engine's side, and
                recorded for the same reason: a backdrop drawn again from its description — a redraw, a
                rollback, a figure read back — has to come back *behind* the data rather than at the
                default 0.
            **opts: Extra styling kwargs; filtered to ``ArrayGlyph``'s accepted options.

        Returns:
            The glyph's mappable (also registered as a Scene layer), or ``None`` when the data lies
            entirely outside what the display CRS shows — an off-limb frame draws nothing rather than
            raising, so a rotation past the far side of a globe still renders.
        """
        recorded_cmap, held_cmap = _described_cmap(cmap)
        if held_cmap is not None:
            opts["cmap"] = held_cmap
        record = LayerRecord(
            FIELD_KINDS[kind],
            source=dataset,
            band=draw_band,
            symbology=Symbology(
                props={
                    "via": kind,
                    "band": band,
                    "cmap": recorded_cmap,
                    "levels": levels,
                    "add_colorbar": add_colorbar,
                    "default_cmap": default_cmap,
                    "zorder": zorder,
                }
            ),
            # What the figure records is the call the caller made, not the call cleopatra receives:
            # `draw_field` resolves the colormap and the levels from it and hands cleopatra the result. The
            # caller's own engine keywords travel beside the description, exactly as passed.
            opts=opts,
        )
        try:
            return self._draw(record)
        except OffLimbError:
            self._skipped_off_limb(kind)
            return None

    def imshow(self, dataset: Any, **kwargs) -> Any:
        """Render a raster as a pixel grid (``ArrayGlyph`` ``kind="imshow"``).

        Returns:
            The image mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.
        """
        return self._field(dataset, kind="imshow", **kwargs)

    def contourf(self, dataset: Any, **kwargs) -> Any:
        """Render a raster as filled contours (``ArrayGlyph`` ``kind="contourf"``).

        Returns:
            The filled-contour mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.
        """
        return self._field(dataset, kind="contourf", **kwargs)

    def contour(self, dataset: Any, **kwargs) -> Any:
        """Render a raster as line contours (``ArrayGlyph`` ``kind="contour"``).

        Returns:
            The line-contour mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.
        """
        return self._field(dataset, kind="contour", **kwargs)

    def pcolormesh(self, dataset: Any, **kwargs) -> Any:
        """Render a raster as a quadrilateral mesh (``ArrayGlyph`` ``kind="pcolormesh"``).

        Returns:
            The ``QuadMesh`` mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.
        """
        return self._field(dataset, kind="pcolormesh", **kwargs)

    def block(self, dataset: Any, **kwargs) -> Any:
        """Render a raster as a filled cell mesh — currently an alias of :meth:`pcolormesh`.

        ``block`` is meant for *discrete* per-cell rectangles aligned to cell **edges**. cleopatra's
        ``pcolormesh`` path accepts cell-**centre** coordinates only (it rejects edge arrays of length
        ``n + 1``), so true edge-aligned blocks need an upstream cleopatra option; until that lands ``block``
        draws the same filled mesh as :meth:`pcolormesh`. It is kept as a distinct, stable entry point so
        callers and examples can switch to true blocks transparently once cleopatra supports them.

        Returns:
            The ``QuadMesh`` mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.
        """
        return self._field(dataset, kind="pcolormesh", **kwargs)

    @staticmethod
    def _extent_of(x: Any, y: Any) -> List[float]:
        """Return the bbox-order extent of the cells 1-D x/y centres describe, as cleopatra takes it.

        The coordinates name the middle of each cell, and the extent is the rectangle those cells *cover* —
        half the outermost spacing further out on every side (:meth:`Bounds.cell_edges`). Taken from the
        smallest and largest centre instead, the image would be drawn one cell narrower and one shorter than
        the data, with every pixel at ``(n - 1) / n`` of its size (#301).

        Args:
            x: X coordinates, already in the display CRS.
            y: Y coordinates, already in the display CRS.

        Returns:
            ``[xmin, ymin, xmax, ymax]``. The ordering comes from
            :meth:`~digitalearth.base.spec.bounds.Bounds.as_bbox` rather than a list written out here, so it
            cannot drift from the ordering the axes path uses. The `Bounds` is built only to name that
            ordering and is unwrapped immediately, so it carries no CRS: the caller's coordinates are already
            in the display one, and a CRS that changes nothing is a parameter every caller must think about
            for no benefit.
        """
        return Bounds.cell_edges(x, y, crs=None).as_bbox()

    def _extent(self, ds: Any) -> List[float]:
        """Return the bbox-order extent of the cells a dataset's coordinates describe, as cleopatra takes it.

        Args:
            ds: The display-CRS source whose ``x``/``y`` cell centres bound the image.

        Returns:
            ``[xmin, ymin, xmax, ymax]`` — the rectangle the cells cover, as :meth:`_extent_of` places it.
        """
        return self._extent_of(ds.x, ds.y)

    def rgb_composite(
        self,
        dataset: Any,
        bands: Sequence[int] = DEFAULT_COMPOSITE_BANDS,
        *,
        mask_nodata: bool = True,
        limits: Optional[ChannelLimits] = None,
        **opts,
    ) -> Any:
        """Render three raster bands as a true/false-colour RGB image (``ArrayGlyph`` RGB path).

        Args:
            dataset: A multiband pyramids ``Dataset`` (reprojected to the display CRS first).
            bands: Three 1-based band indices mapped to R, G, B. Defaults to ``(1, 2, 3)``.
            mask_nodata: When ``True`` (default) each band's nodata cells are excluded from the 2-98
                percentile stretch (and render transparent). Pass ``False`` for the raw values (the
                pre-mask behaviour, where the nodata sentinel participates in the stretch).
            limits: Optional frozen ``(lo, hi)`` stretch bounds, one pair per channel — skips the per-call
                percentile scan. :meth:`~digitalearth.static.maps.animation.AnimationMixin.animate` fills
                these from the whole stack so a composite animation holds one stretch across its frames.
            **opts: Styling kwargs, filtered to ``ArrayGlyph``'s accepted options.

        Returns:
            The image mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: when ``bands`` does not hold exactly three indices, or ``limits`` is given without
                one ``(lo, hi)`` pair per channel.

        Examples:
            - Composite three bands of a multiband raster into one RGB image:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> base = np.nan_to_num(ds.read_array(band=0).astype("float32"))
                >>> rgb = Dataset.from_array(arr=np.stack([base, base, base]),
                ...                          geo_ref=GeoReference(geo=ds.geotransform, epsg=ds.epsg))
                >>> m = Map(crs=rgb.epsg)
                >>> _ = m.rgb_composite(rgb)
                >>> len(m.ax.images)
                1

                ```
            - Frozen ``limits`` replace the per-call stretch: a white point far above the data renders it
                black, which is what holds a sequence of frames on one scale:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> base = np.nan_to_num(ds.read_array(band=0).astype("float32"))
                >>> rgb = Dataset.from_array(arr=np.stack([base, base, base]),
                ...                          geo_ref=GeoReference(geo=ds.geotransform, epsg=ds.epsg))
                >>> m = Map(crs=rgb.epsg)
                >>> _ = m.rgb_composite(rgb, limits=[(0.0, 1e6)] * 3)
                >>> round(float(np.nanmax(m.ax.images[-1].get_array())), 3)
                0.0

                ```

        See Also:
            hsv_composite: The same three bands read as hue/saturation/value instead.
            digitalearth.base.stretch.channel_limits: Derives the ``limits`` this accepts.
        """
        require_three_bands("rgb_composite", bands)
        return self._composite(
            "rgb_composite", dataset, bands, mask_nodata, limits, opts
        )

    def _composite(
        self,
        via: str,
        dataset: Any,
        bands: Sequence[int],
        mask_nodata: bool,
        limits: Optional[ChannelLimits],
        opts: dict,
    ) -> Any:
        """Record a three-band composite and draw it, answering an off-limb warp with a skipped layer.

        The half :meth:`rgb_composite` and :meth:`hsv_composite` share once the band count is validated —
        they differ only in the recipe they record, which is what picks the drawer.

        Args:
            via: The recipe the figure records, ``"rgb_composite"`` or ``"hsv_composite"``.
            dataset: The multiband pyramids ``Dataset``.
            bands: The three 1-based band indices.
            mask_nodata: Whether each band's nodata cells are excluded from the stretch.
            limits: Frozen per-channel ``(lo, hi)`` stretch bounds, or ``None`` for a per-call scan.
            opts: The caller's styling keywords, held beside the layer as they were written.

        Returns:
            The image mappable, or ``None`` when the data lies outside what the display CRS shows.
        """
        record = LayerRecord(
            "rgb",
            source=dataset,
            symbology=Symbology(
                props={
                    "via": via,
                    "bands": tuple(bands),
                    "mask_nodata": mask_nodata,
                    "limits": _described_limits(limits),
                }
            ),
            opts=opts,
        )
        try:
            return self._draw(record)
        except OffLimbError:
            self._skipped_off_limb(via)
            return None

    def hsv_composite(
        self,
        dataset: Any,
        bands: Sequence[int] = DEFAULT_COMPOSITE_BANDS,
        *,
        mask_nodata: bool = True,
        limits: Optional[ChannelLimits] = None,
        **opts,
    ) -> Any:
        """Render three raster bands as an HSV composite (hue/sat/value → RGB → image).

        Args:
            dataset: A multiband pyramids ``Dataset`` (reprojected to the display CRS first).
            bands: Three 1-based band indices mapped to H, S, V. Defaults to ``(1, 2, 3)``.
            mask_nodata: When ``True`` (default) each band's nodata cells are excluded from the 2-98
                percentile stretch (and render transparent). Pass ``False`` for the raw pre-mask behaviour.
            limits: Optional frozen ``(lo, hi)`` stretch bounds, one pair per channel (H, S, V) — skips the
                per-call percentile scan, so an animation can hold one stretch across its frames.
            **opts: Styling kwargs, filtered to ``ArrayGlyph``'s accepted options.

        Returns:
            The image mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: when ``bands`` does not hold exactly three indices, or ``limits`` is given without
                one ``(lo, hi)`` pair per channel.

        Examples:
            - Read three bands as hue/saturation/value and render the resulting RGB image:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> base = np.nan_to_num(ds.read_array(band=0).astype("float32"))
                >>> hsv = Dataset.from_array(arr=np.stack([base, base * 0.5, base * 0.25]),
                ...                          geo_ref=GeoReference(geo=ds.geotransform, epsg=ds.epsg))
                >>> m = Map(crs=hsv.epsg)
                >>> _ = m.hsv_composite(hsv)
                >>> len(m.ax.images)
                1

                ```
            - The rendered image is band-last RGB, one value per channel per cell:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> from digitalearth.static import Map
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> base = np.nan_to_num(ds.read_array(band=0).astype("float32"))
                >>> hsv = Dataset.from_array(arr=np.stack([base, base * 0.5, base * 0.25]),
                ...                          geo_ref=GeoReference(geo=ds.geotransform, epsg=ds.epsg))
                >>> m = Map(crs=hsv.epsg)
                >>> _ = m.hsv_composite(hsv)
                >>> m.ax.images[-1].get_array().shape[-1]
                3

                ```

        See Also:
            rgb_composite: The same three bands mapped straight to red/green/blue.
            digitalearth.base.stretch.channel_limits: Derives the ``limits`` this accepts.
        """
        require_three_bands("hsv_composite", bands)
        return self._composite(
            "hsv_composite", dataset, bands, mask_nodata, limits, opts
        )

    def spaghetti(self, collection: Any, band: int = 1, **opts) -> List[Any]:
        """Overlay each member of a ``DatasetCollection`` as line contours on one axes (ensemble spaghetti).

        Args:
            collection: A pyramids ``DatasetCollection`` whose members share a grid.
            band: 1-based band read from each member.
            **opts: Styling kwargs forwarded to the per-member contour call.

        Returns:
            The list of per-member contour mappables (each also registered as a Scene layer). Members
            lying outside what the display CRS shows draw nothing and are absent from the list, so it
            stays one entry per *drawn* member and never contains ``None``. That means the list cannot be
            zipped against ``collection.datasets`` when some members are hidden — pair by drawing members
            individually if a per-member legend needs to know which is which.
        """
        drawn = [
            self._field(member, kind="contour", band=band, add_colorbar=False, **opts)
            for member in collection.datasets
        ]
        return [artist for artist in drawn if artist is not None]
