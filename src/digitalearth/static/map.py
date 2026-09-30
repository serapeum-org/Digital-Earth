"""Map — a Scene with a display CRS, pyramids reprojection, and basemap/coastline decoration (no Cartopy).

``Map`` reprojects each input to a chosen display CRS via **pyramids** (``Dataset.to_crs``), renders on a plain
matplotlib axes in that projected space, and decorates it with an XYZ-tile basemap (``cleopatra.basemap.tiles``) and
Natural-Earth vector features (``cleopatra.basemap.reference``). There is deliberately **no Cartopy**: the
projection is applied to the *data* upstream, not to the axes (see plan §2.4).

The field methods here (``field`` and the private ``_field`` recipe) are the foundation T1.1 extends with
``contours``/``pcolormesh``/``block``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from digitalearth.static.maps.animation import AnimationMixin
from digitalearth.static.maps.base import GeoLayerBase
from digitalearth.static.maps.decoration import DecorationMixin
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
    ProjectionMixin,
    AnimationMixin,
    GeoLayerBase,
):
    """A geospatial :class:`~digitalearth.static.scene.Scene` that reprojects to a display CRS.

    Args:
        crs: Display CRS as an EPSG int / string / CRS (anything ``Dataset.to_crs`` accepts). Default 3857.
        domain: Optional named region / bbox used to set the extent (resolved in T5.1); ``None`` uses data bounds.
        ax: Existing axes to draw on (a new figure/axes is created when ``None``). One axes holds one map;
            two maps sharing one is unsupported (see :class:`~digitalearth.static.scene.Scene`).
        fig: Figure owning ``ax``.
        figsize: New-figure size when one is created.
        globe: Draw the projection boundary/graticule and clip layers to it (an orthographic globe).
        strict: When ``True`` a layer whose data cannot be placed in the display CRS raises
            :class:`~digitalearth.base.crs.OffLimbError` instead of being skipped with a warning.

    Attributes:
        crs: The display CRS every layer is reprojected to.
        domain: The configured domain (or ``None``).
        strict: Whether a layer that would draw nothing raises instead of being skipped.

    Examples:
        - Create a map in Web Mercator and read its display CRS:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static import Map
            >>> m = Map(crs=3857)
            >>> m.crs
            3857
            >>> m.layers
            []

            ```
        - Render a reprojected raster, then iterate the registered layers:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from pyramids.dataset import Dataset
            >>> from digitalearth.static import Map
            >>> ds = Dataset.read_file("examples/data/acc4000.tif")
            >>> m = Map(crs=ds.epsg)          # same CRS -> no reprojection
            >>> _ = m.field(ds)
            >>> len(m.layers)
            1

            ```
    """

    def draw_figure(self, figure: FigureSpec) -> Self:
        """Draw a figure into this map, bringing it from whatever it showed before.

        The 2-D counterpart of the 3-D tier's ``draw_figure``, with one difference in the plumbing: this
        tier's :meth:`~digitalearth.static.scene.Scene._change` derives a candidate's sources from
        ``self._sources`` (filtered to the tree), not from the incoming figure, so the figure's own sources
        are installed here first — otherwise :meth:`~digitalearth.static.renderer.Renderer.apply` would have
        nothing to open. The 3-D tier replays its incoming figure's sources directly and needs no such step.

        Args:
            figure: The figure to draw. Its one panel's :class:`~digitalearth.base.spec.Viewport` set the
                display CRS/domain/globe at construction; here its layers are drawn in order. An ``object:``
                (in-memory) source is replayed in process, so the scene that registered it must stay alive
                until this returns; a path or URL source has no such constraint.

        Returns:
            This map (chainable).

        Raises:
            KeyError: when a layer names a kind this tier does not draw, or a recipe it does not know.
            OffLimbError: when a layer's data cannot be placed and the map is ``strict``.
        """
        # Normalise recipe names before drawing. For a same-tier figure — the only case this tier draws (a
        # foreign figure's drawers read non-portable `props`; see `retarget_via` and `to_backend`) — every
        # recipe is already known, so this is a no-op. Kept symmetric with the interactive tier's retarget.
        from digitalearth.static.renderer import retarget_via

        figure = retarget_via(figure)
        self._sources = dict(figure.sources)
        self._change(figure)
        # The layer diff does not re-read the figure's title, so a title carried across from a tier whose
        # figure_spec records one would be lost; set it best-effort. This tier's own figure_spec emits no
        # panel title, so a same-tier round trip has nothing to restore here.
        title = figure.title or figure.panels[0].title
        if title:
            self.set_title(title)
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
