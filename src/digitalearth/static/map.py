"""Map — a Scene with a display CRS, pyramids reprojection, and basemap/coastline decoration (no Cartopy).

``Map`` reprojects each input to a chosen display CRS via **pyramids** (``Dataset.to_crs``), renders on a plain
matplotlib axes in that projected space, and decorates it with an XYZ-tile basemap (``cleopatra.basemap.tiles``) and
Natural-Earth vector features (``cleopatra.basemap.reference``). There is deliberately **no Cartopy**: the
projection is applied to the *data* upstream, not to the axes.

This module is a **thin composition and nothing else**: it lists the six capability mixins `Map` is built
from and adds the two figure-replay entry points, `draw_figure` and `from_figure`. No render method is
defined here — `field`, `contours`, `pcolormesh` and `block` are all `RasterMixin`
(`digitalearth.static.maps.raster`), the vector builders are `VectorMixin`, and so on. A new capability
joins the mixin it belongs to, or arrives as a seventh mixin composed in below; it does not land in this
file.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Self

from digitalearth.static.maps.animation import AnimationMixin
from digitalearth.static.maps.base import GeoLayerBase
from digitalearth.static.maps.decoration import DecorationMixin
from digitalearth.static.maps.inset import InsetMixin
from digitalearth.static.maps.projection import ProjectionMixin
from digitalearth.static.maps.raster import RasterMixin
from digitalearth.static.maps.vector import VectorMixin

if TYPE_CHECKING:
    from digitalearth.base.spec import FigureSpec

__all__ = ["Map"]


class Map(
    RasterMixin,
    VectorMixin,
    DecorationMixin,
    InsetMixin,
    ProjectionMixin,
    AnimationMixin,
    GeoLayerBase,
):
    """A geospatial :class:`~digitalearth.static.scene.Scene` that reprojects to a display CRS.

    The tier's workhorse, and a **composition rather than a class with methods of its own**: every
    capability lives in one of the six mixins under `digitalearth.static.maps` — `RasterMixin` (fields,
    composites, spaghetti), `VectorMixin` (points, polygons, unstructured, flow, u/v fields),
    `DecorationMixin` (text, annotate, basemap, Natural-Earth layers), `InsetMixin` (`inset`,
    `mark_extent`), `ProjectionMixin` (extent, domain, globe frame, save/show) and `AnimationMixin`
    (`animate`, `rotate`) — and `map.py` only lists them. `tests/static/test_map_composition.py` pins that
    MRO, so a seventh mixin fails it until the list is updated.

    **Every data builder returns `Self`, so calls chain** (`m.field(ds).coastlines().colorbar()`), and the
    matplotlib artist a layer drew is reached by name through :meth:`~digitalearth.static.scene.Scene.artist`
    — never through the renderer's private record.

    Args:
        crs: Display CRS as an EPSG int / string / CRS (anything ``Dataset.to_crs`` accepts). Default 3857.
        domain: A registered region name (`"europe"`, `"africa"`, …) or an explicit
            `(west, south, east, north)` bbox in EPSG:4326. **Stored, not applied**: constructing with one
            does not frame the axes, and it is not validated either — it becomes the fallback a bare
            :meth:`~digitalearth.static.maps.projection.ProjectionMixin.set_domain` call resolves, and the
            `domain` the panel's `Viewport` carries. `None` (the default) leaves the view to fit the data.
        ax: Existing axes to draw on (a new figure/axes is created when ``None``). One axes holds one map;
            two maps sharing one is unsupported (see :class:`~digitalearth.static.scene.Scene`).
        fig: Figure owning ``ax``. Taken from `ax` (its root figure) when only `ax` is given.
        figsize: New-figure size in inches when one is created; `(8, 8)` by default.
        globe: Draw the projection boundary/graticule and clip layers to it (an orthographic globe).
        strict: When ``True`` a layer whose data cannot be placed in the display CRS raises
            :class:`~digitalearth.base.crs.OffLimbError` instead of being skipped with a warning.

    Attributes:
        crs: The display CRS every layer is reprojected to, exactly as it was passed.
        domain: The configured domain (or ``None``). Never rewritten by a framing call.
        strict: Whether a layer that would draw nothing raises instead of being skipped.

    Examples:
        - Create a map in Web Mercator and read its display CRS; it describes nothing yet:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static import Map
            >>> m = Map(crs=3857)
            >>> m.crs
            3857
            >>> m.layers
            []
            >>> m.close()

            ```
        - Render a reprojected raster and chain decoration onto it — each builder hands the map back, so
            the layer ids accumulate in draw order:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from pyramids.dataset import Dataset
            >>> from digitalearth.static import Map
            >>> ds = Dataset.read_file("examples/data/acc4000.tif")
            >>> m = Map(crs=ds.epsg)          # same CRS -> no reprojection
            >>> m.field(ds).coastlines(name="coast") is m
            True
            >>> m.layer_ids
            ['raster-1', 'coast']
            >>> m.close()

            ```
        - A domain is remembered rather than applied: the axes is framed only once `set_domain()` asks for
            it, and the attribute survives the framing:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static import Map
            >>> m = Map(crs=4326, domain="europe")
            >>> [float(v) for v in m.ax.get_xlim()]
            [0.0, 1.0]
            >>> _ = m.set_domain()
            >>> [float(v) for v in m.ax.get_xlim()], m.domain
            ([-25.0, 45.0], 'europe')
            >>> m.close()

            ```
    """

    def draw_figure(self, figure: FigureSpec) -> Self:
        """Draw a figure into this map, bringing it from whatever it showed before.

        The 2-D counterpart of the 3-D tier's ``draw_figure``, with one difference in the plumbing: this
        tier's :meth:`~digitalearth.static.scene.Scene._change` derives a candidate's sources from
        ``self._sources`` (filtered to the tree), not from the incoming figure, so the figure's own sources
        are installed here first — otherwise :meth:`~digitalearth.static.renderer.Renderer.apply` would have
        nothing to open. The 3-D tier replays its incoming figure's sources directly and needs no such step.

        **The two titles land in their own slots, and the round trip is idempotent.** This tier describes a
        heading in two places — the panel's title, which `set_title` paints on the axes, and the figure's
        heading, which is its `suptitle` — so each is restored to the one it was described from. Measured on
        a figure carrying both: described `('rainfall, 2020', 'January')` as
        `(figure.title, figure.panels[0].title)`, restored to the same pair as
        `(fig.get_suptitle(), ax.get_title())`, and redescribed unchanged; `to_dict()` of the figure before
        and after is equal, and equal again on a third pass.

        **The heading is figure-wide, so it is restored only onto a figure this map owns.** A `grid` panel
        is built on an axes :func:`~digitalearth.static.figure.grid` laid out, and every panel of that grid
        shares one figure — so writing the described heading through ``Figure.suptitle`` from one panel
        reaches the whole figure and its siblings, which then describe the incoming heading as their own.
        Measured on a two-panel grid headed `'grid heading'`, drawing a description headed `'INCOMING'`
        into `maps[1]`: before the guard `fig.get_suptitle()` became `'INCOMING'` and `maps[0]` described
        `'INCOMING'`; now both still read `'grid heading'` and the declined heading is named in a
        ``UserWarning``. A map that made its own figure — `Map(crs=4326)`, and so every
        :meth:`from_figure` round trip — restores it as before. The panel's title is not guarded, because
        `set_title` paints the axes and that is panel-local.

        **A title is written only when the incoming figure carries one**, so drawing an untitled figure does
        not blank a heading the map already had. The layers are reconciled, not merely added — that part
        really does bring the map from whatever it showed before.

        One asymmetry is worth knowing, because the two titles live in different places. The figure's
        heading is on the figure and always survives. The panel's title is painted on the **axes**, and a
        cleopatra glyph clears the axes on a scene's *first* render — so on a map that has not drawn a glyph
        yet, an axes title set beforehand is wiped by the incoming figure's first layer while the scene goes
        on describing it. Measured, drawing an untitled figure into a map titled
        `('stale heading', 'stale panel')`: a map that had **already** drawn a layer reports
        `('stale heading', 'stale panel')` from `(fig.get_suptitle(), ax.get_title())`, while a map that had
        drawn nothing reports `('stale heading', '')` — and `figure_spec` still says `'stale panel'` for
        both, because the panel's title is held on the scene rather than read back off the axes.

        Args:
            figure: The figure to draw. Its one panel's :class:`~digitalearth.base.spec.Viewport` set the
                display CRS/domain/globe at construction; here its layers are drawn in order, each title it
                carries is restored to its own slot, and any `set_bounds` framing on the panel's view is
                reapplied (a fit-to-data view carries no bounds and is left to re-fit). An ``object:``
                (in-memory) source is replayed in process, so the scene that registered it must stay alive
                until this returns; a path or URL source has no such constraint.

        Returns:
            This map, so the call chains.

        Raises:
            KeyError: when a layer names a kind this tier does not draw, or a recipe it does not know —
                `"the static tier does not draw 'terrain' layers; it draws [...]"`.
            OffLimbError: when a layer's data cannot be placed and the map is ``strict``.

        Warns:
            UserWarning: when the incoming figure carries a heading and this map does not own its figure —
                naming the heading that was not drawn, and the ``fig.suptitle`` call that would draw it.

        Examples:
            - A figure drawn into a fresh map comes back describing exactly what it described, each title
                in its own place, and stays there on a second pass:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.base.spec import FigureSpec
                >>> from digitalearth.static import Map
                >>> first = Map(crs=4326)
                >>> _ = first.field("examples/data/acc4000.tif")
                >>> _ = first.set_title("January")
                >>> _ = first.fig.suptitle("rainfall, 2020")
                >>> stored = first.figure_spec.to_dict()
                >>> first.close()
                >>> second = Map(crs=4326).draw_figure(FigureSpec.from_dict(stored))
                >>> second.fig.get_suptitle(), second.ax.get_title()
                ('rainfall, 2020', 'January')
                >>> second.figure_spec.to_dict() == stored
                True
                >>> second.close()

                ```
            - A figure that carries no title of its own does not blank the headings already on the map —
                and the panel title survives on the axes because this map had already drawn a layer:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.base.spec import FigureSpec
                >>> from digitalearth.static import Map
                >>> blank = Map(crs=4326)
                >>> _ = blank.field("examples/data/acc4000.tif")
                >>> untitled = FigureSpec.from_dict(blank.figure_spec.to_dict())
                >>> blank.close()
                >>> m = Map(crs=4326)
                >>> _ = m.field("examples/data/acc4000.tif")
                >>> _ = m.set_title("stale panel")
                >>> _ = m.fig.suptitle("stale heading")
                >>> _ = m.draw_figure(untitled)
                >>> m.fig.get_suptitle(), m.ax.get_title()
                ('stale heading', 'stale panel')
                >>> m.close()

                ```
            - On a map that has drawn **nothing** yet, the first layer's render clears the axes and takes a
                pre-existing axes title with it, while the description keeps it:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.base.spec import FigureSpec
                >>> from digitalearth.static import Map
                >>> blank = Map(crs=4326)
                >>> _ = blank.field("examples/data/acc4000.tif")
                >>> untitled = FigureSpec.from_dict(blank.figure_spec.to_dict())
                >>> blank.close()
                >>> m = Map(crs=4326)
                >>> _ = m.set_title("stale panel")
                >>> _ = m.draw_figure(untitled)
                >>> m.ax.get_title(), m.figure_spec.panels[0].title
                ('', 'stale panel')
                >>> m.close()

                ```
            - A kind this tier has no drawer for is refused by name, and the map is left as it was:
                ```python
                >>> import copy
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.base.spec import FigureSpec
                >>> from digitalearth.static import Map
                >>> source = Map(crs=4326)
                >>> _ = source.field("examples/data/acc4000.tif")
                >>> stored = copy.deepcopy(source.figure_spec.to_dict())
                >>> source.close()
                >>> stored["layers"]["layers"][0]["kind"] = "terrain"
                >>> m = Map(crs=4326)
                >>> try:
                ...     m.draw_figure(FigureSpec.from_dict(stored))
                ... except KeyError as error:
                ...     print(str(error).split(";")[0].strip('"'))
                the static tier does not draw 'terrain' layers
                >>> m.layer_ids
                []
                >>> m.close()

                ```
        """
        # Normalise recipe names before drawing, so a figure the web or interactive tier described draws here:
        # a foreign recipe is retargeted to this tier's own, and the field drawer fills the style a portable
        # figure omits from its own defaults. A same-tier figure already knows every recipe, so this is a no-op.
        from digitalearth.static.renderer import retarget_via

        figure = retarget_via(figure)
        self._sources = dict(figure.sources)
        self._change(figure)
        # The layer diff does not re-read either title, so both are set here — each into the slot it was
        # described from. This tier describes two: the **panel's** title, which `set_title` paints on the
        # axes (ST-18), and the **figure's** heading, which is its `suptitle` (M4). Restoring them through
        # one `set_title` — `figure.title or figure.panels[0].title` — put the heading on the axes and
        # dropped the panel's title with it, so the round trip was not even idempotent. Measured on a
        # one-panel grid carrying both: described `('rainfall, 2020', 'January')` as
        # `(figure.title, panels[0].title)`, restored `('', 'rainfall, 2020')` as
        # `(fig.get_suptitle(), ax.get_title())`, and redescribed `(None, 'rainfall, 2020')`.
        if figure.panels[0].title:
            self.set_title(figure.panels[0].title)
        # Only onto a figure this map owns. `Figure.suptitle` is figure-level, and in a `grid` every panel
        # shares one figure, so an unguarded restore is a per-panel call rewriting the whole figure's
        # heading: measured on a two-panel grid headed `'grid heading'`, drawing a description headed
        # `'INCOMING'` into `maps[1]` left `fig.get_suptitle()` as `'INCOMING'`, and `maps[0]` — never
        # drawn into — then described `'INCOMING'` as its own (R2-M1). The panel's title above is
        # unguarded because `set_title` paints the **axes**, which is panel-local.
        if figure.title:
            if self._owns_fig:
                self.fig.suptitle(figure.title)
            else:
                warnings.warn(
                    f"draw_figure() did not restore the figure heading {figure.title!r}: this map was "
                    f"built on an axes somebody else laid out, and the heading belongs to the figure "
                    f"that axes is in, which its siblings share. Call "
                    f"`map.fig.suptitle({figure.title!r})` yourself if the whole figure should carry it.",
                    UserWarning,
                    stacklevel=2,
                )
        # The view's explicit framing is not a layer, so `_change` does not carry it: restore any `set_bounds`
        # region from the panel's viewport, or a map framed on a subregion replays showing the full data
        # extent. `set_bounds` reprojects a `Bounds` in any CRS, so a figure another tier described frames
        # correctly too. A fit-to-data view (no bounds) is left to re-fit.
        bounds = getattr(figure.panels[0].view, "bounds", None)
        if bounds is not None:
            self.set_bounds(bounds)
        return self

    @classmethod
    def from_figure(cls, figure: FigureSpec, **scene_kwargs: Any) -> Self:
        """Build a map and draw a figure into it — the round trip the description seam exists for.

        A map describes itself with :attr:`~digitalearth.static.scene.Scene.figure_spec`, the description
        survives ``to_dict()``/``from_dict()``, and this draws it again. :func:`digitalearth.api.to_backend`
        dispatches here for ``backend="matplotlib"``.

        Args:
            figure: The figure to draw.
            **scene_kwargs: Passed to the constructor — ``figsize``, ``strict``, ``ax``/``fig``; a ``crs``,
                ``domain`` or ``globe`` here overrides the one the figure's viewport carries.

        Returns:
            The map, with every layer drawn.

        Raises:
            KeyError: from :meth:`draw_figure`, when a layer names a kind this tier does not draw or a
                recipe it does not know. The map has been constructed by then, so its figure is left open.
            OffLimbError: from :meth:`draw_figure`, when a layer's data cannot be placed and `strict=True`
                was passed through `**scene_kwargs`.

        Examples:
            - A figure a map describes draws again into a fresh map, carrying its display CRS:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.static import Map
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> first = Map(crs=ds.epsg)
                >>> _ = first.field(ds)
                >>> second = Map.from_figure(first.figure_spec)
                >>> second.layer_ids == first.layer_ids
                True
                >>> first.close()
                >>> second.close()

                ```
        """
        view = figure.panels[0].view
        kwargs: dict[str, Any] = {
            "domain": getattr(view, "domain", None),
            "globe": bool(getattr(view, "globe", False)),
        }
        crs = getattr(view, "crs", None)
        if crs is not None:
            kwargs["crs"] = crs
        scene = cls(**{**kwargs, **scene_kwargs})
        scene.draw_figure(figure)
        return scene
