"""ProjectionMixin — extent/domain, the globe projection frame, and render/save/show hooks.

Sets the axes extent from a bbox or named domain, builds and caches the projection boundary/graticule for a
globe map, and overrides ``save``/``show`` to apply that frame before output.
"""

import os
from dataclasses import replace as with_fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Union

from cleopatra.basemap.projection import apply_projection_frame

from digitalearth.base.domains import DomainLike, resolve_domain
from digitalearth.base.spec import Bounds, LayerSpec, Symbology
from digitalearth.static import projections
from digitalearth.static.renderer import DrawnLayer, artists_added
from digitalearth.static.scene import LayerRecord

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.static.maps.base import GeoLayerBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


def draw_graticule(scene: Any, _data: Any, layer: LayerSpec) -> DrawnLayer:
    """Build the lon/lat graticule a described layer asks for, at the spacing it recorded.

    The one drawer here that leaves no artist behind. A graticule's lines are drawn by
    ``apply_projection_frame`` when the globe frame goes on, which is after every data layer — so what
    "drawing" it means is computing the lines and handing them to the map, and the frame picks them up.

    Args:
        scene: The map being drawn on.
        _data: The source slot every drawer takes, unread here — a graticule is computed from the display
            CRS and two spacings, not from data.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` carrying the projected lines, with no artists
        **yet**: there is nothing on the axes to hide or remove until the frame goes on. It is
        :meth:`ProjectionMixin._apply_frame` that then hands the layer the line artists it drew, so hiding
        or removing the graticule reaches them like any other decoration layer's.
    """
    props = layer.symbology.props
    scene._graticule_lines = projections.graticule(
        scene.crs, lon_step=props["lon_step"], lat_step=props["lat_step"]
    )
    return DrawnLayer(artist=scene._graticule_lines)


class ProjectionMixin(_MixinBase):
    """Extent/domain and globe projection-frame behaviour for :class:`~digitalearth.static.map.Map`.

    A capability mixin of :class:`~digitalearth.static.map.Map`: it is only ever composed into that map class, never
    instantiated or subclassed on its own. Its methods reach the shared figure/axes, the layer registry and the
    display CRS — and the sibling mixins' methods — through ``self``, and only the composition supplies those.

    The ``if TYPE_CHECKING`` base declared above the class is what records that contract for a type checker: it
    resolves each ``self.<attr>`` against :class:`~digitalearth.static.maps.base.GeoLayerBase`, the state ``Map``
    inherits. At runtime that base is plain ``object``, so composing this mixin leaves the ``Map`` MRO exactly what
    it was before the annotation.

    See Also:
        digitalearth.static.map.Map: the composition that supplies the state these methods use.
        digitalearth.static.maps.base.GeoLayerBase: the typing-only base declared above the class.
    """

    def set_extent(self, bbox: Union[Bounds, Sequence[float]]) -> None:
        """Set the axes extent.

        Args:
            bbox: A :class:`~digitalearth.base.spec.bounds.Bounds` in **any** CRS — it is reprojected to the
                display CRS, which is the point of passing one — or a bare ``[xmin, xmax, ymin, ymax]``
                sequence in matplotlib axes order, assumed to be in the display CRS already. The sequence
                form is accepted because that ordering was this method's contract; prefer `Bounds`, which
                states both the ordering and the CRS instead of leaving them to position and assumption.

        Raises:
            ValueError: if the sequence form does not hold exactly four values.

        Notes:
            A **flipped** pair is honoured in the sequence form: ``[10, 0, 0, 10]`` inverts the x axis, which
            is how matplotlib expresses ``invert_xaxis`` through the limits. A `Bounds` cannot express that —
            it refuses corners the wrong way round, because for a rectangle handed to pyramids or cleopatra
            that is a defect rather than an intent — so invert the axis directly if you need both.

        Examples:
            - A rectangle in another CRS is converted, so the frame lands where the data is:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> from digitalearth.base.spec import Bounds
                >>> m = Map(crs=3857)
                >>> m.set_extent(Bounds(0.0, 0.0, 1.0, 1.0, crs=4326))
                >>> round(m.ax.get_xlim()[1])
                111319

                ```
            - The bare sequence is matplotlib's own ordering, in the display CRS:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> m = Map(crs=3857)
                >>> m.set_extent([0.0, 100.0, 0.0, 50.0])
                >>> [float(v) for v in m.ax.get_xlim()], [float(v) for v in m.ax.get_ylim()]
                ([0.0, 100.0], [0.0, 50.0])

                ```
        """
        if isinstance(bbox, Bounds):
            # to_crs is a no-op when the CRSs already match. Without it a rectangle that carries its CRS
            # would be trusted to be in the display one, which is exactly the mistake Bounds exists to stop.
            xmin, xmax, ymin, ymax = bbox.to_crs(self.crs).as_mpl()
        else:
            # Not routed through Bounds: axes limits may legitimately run backwards, and Bounds refuses that.
            values = [float(value) for value in bbox]
            if len(values) != 4:
                raise ValueError(
                    f"set_extent needs exactly 4 values as [xmin, xmax, ymin, ymax]; got {len(values)}"
                )
            xmin, xmax, ymin, ymax = values
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)

    def set_domain(self, domain: Optional[DomainLike] = None) -> None:
        """Set the axes extent from a named region or bbox, reprojected to the display CRS via pyramids.

        Args:
            domain: A registered region name (e.g. ``"Europe"``), an explicit ``(west, south, east, north)``
                bbox in EPSG:4326, or ``None`` to fall back to the domain passed at construction. A no-op
                when neither resolves to a domain.

        Raises:
            ValueError: if a caller-supplied bbox has its corners the wrong way round — including one
                crossing the antimeridian, which a single rectangle cannot express.

        Examples:
            - In a geographic CRS the axes limits equal the named region's bounds:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.static import Map
                >>> m = Map(crs=4326)
                >>> m.set_domain("europe")
                >>> [float(v) for v in m.ax.get_xlim()]
                [-25.0, 45.0]
                >>> [float(v) for v in m.ax.get_ylim()]
                [34.0, 72.0]

                ```
        """
        bbox = resolve_domain(domain if domain is not None else self.domain)
        if bbox is None:
            return
        # A resolved domain is always EPSG:4326 (see `static/domains.py`), which is exactly the assumption
        # a bare 4-tuple used to carry implicitly. Bounds makes it a value, and does the warp through pyramids.
        try:
            box = Bounds.from_bbox(bbox, crs=4326)
        except ValueError as error:
            # Bounds refuses corners the wrong way round, and its message names only the numbers. A caller
            # who wrote a bbox crossing the antimeridian needs to hear which argument and which ordering.
            raise ValueError(
                f"set_domain got a bbox whose corners are the wrong way round: {tuple(bbox)}. It takes "
                "(west, south, east, north) in EPSG:4326, and cannot express a region crossing the "
                f"antimeridian — split it into two, or set the extent directly ({error})"
            ) from error
        self.set_extent(box.to_crs(self.crs))

    # ------------------------------------------------------------------ globe / projection frame

    def graticule(self, lon_step: float = 30.0, lat_step: float = 30.0) -> None:
        """Add a lon/lat graticule to a projected map (drawn when the frame is applied).

        A second call **replaces** the first — the map holds one set of graticule lines, so it draws one
        graticule — and the description follows: the layer keeps its id and its place in the tree and only
        its spacing changes. Describing the second call as a second layer would say the map draws two grids
        where it draws one.

        Args:
            lon_step: Meridian spacing in degrees.
            lat_step: Parallel spacing in degrees.

        Raises:
            Exception: whatever computing the grid raises — a spacing of zero divides by zero in the
                projection — after the description has been put back as it was. A figure must not name a
                layer that was not drawn, and a refused *replacement* must not restyle the graticule the
                map is still drawing (round 2, M1).
        """
        symbology = Symbology(
            props={"via": "graticule", "lon_step": lon_step, "lat_step": lat_step}
        )
        pointer = self._graticule_id
        # `_reset_layers` clears the tree between animation frames while the lines themselves survive, so the
        # remembered id can outlive its layer; the membership test is what keeps that from raising.
        was: Optional[LayerSpec] = None
        if pointer is not None and pointer in self._layer_tree.ids:
            was = self._layer_tree.get(pointer)
        if was is not None:
            held = was.id
            self._layer_tree = self._layer_tree.replace(
                with_fields(was, symbology=symbology)
            )
        else:
            held = self._describe_layer(LayerRecord("graticule", symbology=symbology))
            self._graticule_id = held
        # Described first, then drawn — but through the renderer directly rather than through
        # `Scene._draw`, because a second call replaces the layer it already has rather than adding one,
        # and the funnel only knows how to add. The undo the funnel owns is therefore spelled here, in the
        # two shapes this method has: an added layer is forgotten, a replaced one is put back as it was.
        try:
            self._renderer.draw_layer(self.figure_spec, held)
        except BaseException:
            if was is None:
                self._forget_layer(held)
                self._graticule_id = pointer
            else:
                self._layer_tree = self._layer_tree.replace(was)
            raise

    def _frame(self) -> tuple:
        """Return the cached ``(boundary, xlim, ylim)`` for the display CRS (computed once per CRS).

        ``projection_frame`` reprojects a dense lon/lat sample of the whole sphere, so it is memoised here to
        avoid recomputing it for both ``set_global`` and ``_apply_frame``. The cache is keyed on the display
        CRS and recomputed only when the CRS changes.

        Returns:
            The ``(boundary_xy, (xmin, xmax), (ymin, ymax))`` tuple from
            :func:`digitalearth.static.projections.projection_frame` for the current display CRS — a closed
            ``(N, 2)`` boundary ring plus the projected x/y limits.
        """
        if self._frame_cache is None or self._frame_cache[0] != self.crs:
            self._frame_cache = (self.crs, projections.projection_frame(self.crs))
        return self._frame_cache[1]

    def set_global(self) -> None:
        """Set the axes extent to the full projection domain (the whole globe/world)."""
        _, xlim, ylim = self._frame()
        self.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]])

    def _apply_frame(self) -> Any:
        """Draw the projection boundary + graticule and clip the layers to it (once, at render time).

        This is also where the graticule layer is handed the artists it owns. Its lines are computed when
        the layer is described and only put on the axes here, so the record its drawer returned carried
        none — and hiding or removing the layer reached nothing (round 2, N5).

        Returns:
            The boundary patch the frame put on the axes, or ``None`` when there was nothing to do — the
            map is flat, or the frame has already been applied. It is idempotent for that reason: a scene
            that is rendered, saved and shown frames itself once.
        """
        if not self.globe or self._framed:
            return None
        boundary, xlim, ylim = self._frame()
        with artists_added(self.ax) as drawn_by_frame:
            patch = apply_projection_frame(
                self.ax,
                boundary_xy=boundary,
                xlim=xlim,
                ylim=ylim,
                graticule_lines=self._graticule_lines,
            )
        self._framed = True
        # Everything but the patch: `apply_projection_frame` adds the boundary and then one line per
        # graticule polyline, and the boundary is the *frame's* — hiding the grid must not take the globe's
        # outline with it.
        self._own_the_graticule(
            tuple(artist for artist in drawn_by_frame if artist is not patch)
        )
        return patch

    def _own_the_graticule(self, artists: Tuple[Any, ...]) -> None:
        """Give the described graticule layer the artists the projection frame drew for it.

        Args:
            artists: The line artists the frame added, in the order it added them.

        Note:
            This writes into the renderer's record of what it drew, which no public method reaches — every
            other layer's artists are known to its drawer, and a graticule's are not. A
            ``Renderer.attach_artists`` would be the tidier home for it.
        """
        layer_id = self._graticule_id
        if layer_id is None or not artists:
            return
        drawn = self._renderer.drawn.get(layer_id)
        if drawn is None:
            return
        self._renderer._drawn[layer_id] = with_fields(drawn, artists=artists)

    def render(self) -> None:
        """Apply the projection frame if this is a globe map (idempotent). Call before showing/saving."""
        self._apply_frame()

    def save(self, path: Union[str, "os.PathLike[str]"], **kwargs) -> Path:
        """Apply the projection frame (for a globe map) then save the figure.

        Args:
            path: Destination file path; the extension picks the format matplotlib writes.
            **kwargs: Forwarded to :meth:`~digitalearth.static.scene.Scene.save` / ``Figure.savefig``.

        Returns:
            The path that was written, as a :class:`pathlib.Path`.
        """
        self._apply_frame()
        return super().save(path, **kwargs)

    def show(self) -> None:
        """Apply the projection frame (for a globe map) then show the figure."""
        self._apply_frame()
        super().show()
