"""ProjectionMixin — extent/domain, the globe projection frame, and render/save/show hooks.

Sets the axes extent from a bbox or named domain, builds and caches the projection boundary/graticule for a
globe map, and overrides ``save``/``show`` to apply that frame before output.
"""

import os
import warnings
from dataclasses import replace as with_fields
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Optional,
    Self,
    Sequence,
    Tuple,
    Union,
)

from cleopatra.basemap.projection import apply_projection_frame

from digitalearth.base.deprecation import renamed_method
from digitalearth.base.domains import DomainLike, resolve_domain
from digitalearth.base.spec import Bounds, LayerSpec, Symbology
from digitalearth.static import projections
from digitalearth.static.renderer import DrawnLayer, artists_added
from digitalearth.static.scene import LayerRecord

#: The meridian and parallel spacing a graticule is drawn at when the caller names neither, in degrees —
#: the same default the interactive and web tiers take (#263). The two arguments default to ``None`` rather
#: than to this so that a step the caller wrote can be told from one they did not, which is what lets
#: :meth:`ProjectionMixin.graticule` say when ``spacing=`` has just discarded one (review R2-L3).
DEFAULT_GRATICULE_STEP: float = 30.0

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.static.maps.base import GeoLayerBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


class _GraticuleSteps(NamedTuple):
    """The meridian and parallel spacing one :meth:`ProjectionMixin.graticule` call settles on, in degrees.

    Three arguments, one decision. ``lon_step``, ``lat_step`` and ``spacing`` are the only values that travel
    together through the front of that method, and the only thing done to them is to collapse them into these
    two — with a warning when the caller wrote arguments the collapse throws away. Lifting them out with that
    rule is most of what took `graticule` under the cognitive-complexity bar (`python:S3776`), and it makes
    the rule testable without a map: the collapse is a pure function of what the caller wrote.

    :meth:`symbology` is here for the same reason — the props a graticule layer is described by are these two
    steps and nothing else, and `draw_graticule` reads exactly the keys this writes.

    Attributes:
        lon: Meridian spacing in degrees.
        lat: Parallel spacing in degrees.
    """

    lon: float
    lat: float

    @classmethod
    def asked(
        cls,
        lon_step: Optional[float],
        lat_step: Optional[float],
        spacing: Optional[float],
    ) -> "_GraticuleSteps":
        """Collapse what a caller wrote into the two steps a graticule is drawn at.

        Args:
            lon_step: Meridian spacing the caller named, or ``None``.
            lat_step: Parallel spacing the caller named, or ``None``.
            spacing: One step for both, which outranks the other two.

        Returns:
            The two steps, each falling back to :data:`DEFAULT_GRATICULE_STEP`.

        Warns:
            UserWarning: when ``spacing`` is given beside either step, because the call has then had two of
                its own arguments thrown away (review R2-L3). ``stacklevel=3`` so it still names the line
                that called ``graticule()``: this frame and ``graticule``'s both sit under it.
        """
        if spacing is not None:
            if lon_step is not None or lat_step is not None:
                warnings.warn(
                    f"graticule() was given spacing={spacing!r} together with "
                    f"lon_step={lon_step!r}/lat_step={lat_step!r}; spacing sets both, so those two are "
                    "discarded. Pass one or the other.",
                    UserWarning,
                    stacklevel=3,
                )
            return cls(spacing, spacing)
        lon = DEFAULT_GRATICULE_STEP if lon_step is None else lon_step
        lat = DEFAULT_GRATICULE_STEP if lat_step is None else lat_step
        return cls(lon, lat)

    def symbology(self) -> Symbology:
        """Return the symbology a graticule layer is described by.

        Returns:
            A :class:`~digitalearth.base.spec.Symbology` carrying the drawer key and the two steps — the
            props :func:`draw_graticule` reads back.
        """
        return Symbology(
            props={"via": "graticule", "lon_step": self.lon, "lat_step": self.lat}
        )


class _GraticuleEdit(NamedTuple):
    """What one :meth:`ProjectionMixin.graticule` call did to the description, and how to put it back.

    A graticule is described *before* it is drawn, and a draw that raises must leave no layer the map is not
    drawing behind (round 2, M1). These three values are what the undo needs and the only reason any of them
    outlives the describe step — so they travel as one value with :meth:`undo` on it, rather than as three
    locals threaded through the method that would otherwise have to hold all three branches at once.

    Attributes:
        held: The id the graticule is described under now, and the one the drawer is handed.
        was: The layer as it stood before this call, or ``None`` when this call created it.
        pointer: The id the map pointed at before this call. Not always ``was.id``: ``_reset_layers`` clears
            the tree between animation frames while the lines survive, so a remembered id can outlive its
            layer, and then the call creates a layer while ``pointer`` still names the dead one.
    """

    held: str
    was: Optional[LayerSpec]
    pointer: Optional[str]

    def undo(self, scene: Any) -> None:
        """Put the description back as this call found it.

        Args:
            scene: The map whose description was edited.
        """
        if self.was is None:
            scene._forget_layer(self.held)
            scene._graticule_id = self.pointer
        else:
            scene._layer_tree = scene._layer_tree.replace(self.was)


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

    def set_bounds(self, bbox: Union[Bounds, Sequence[float]]) -> Self:
        """Frame the figure on a region — the Core contract's name for setting the axes extent.

        Args:
            bbox: A :class:`~digitalearth.base.spec.bounds.Bounds` in **any** CRS — it is reprojected to the
                display CRS, which is the point of passing one — or a bare ``[xmin, xmax, ymin, ymax]``
                sequence in matplotlib axes order, assumed to be in the display CRS already. The sequence
                form is accepted because that ordering was this method's contract; prefer `Bounds`, which
                states both the ordering and the CRS instead of leaving them to position and assumption.

        Returns:
            This map, so the call chains (``Map(crs=3857).set_bounds(bbox).coastlines()``). The Core
            declares ``returns="self"`` for this name and the web and 3-D tiers already answer that way;
            ``set_extent`` returned ``None``, so the same line worked on one tier and raised on another.

        Raises:
            ValueError: if the sequence form does not hold exactly four values.

        Notes:
            **This takes no** ``padding`` **and no** ``None`` **that fits the data**, which the Core
            declares alongside the name. Those are auto-framing and arrive at order 26; order 27a settled
            the *spelling*, and `KEYWORD_SHORTFALLS` in `tests/test_contract_names.py` records the gap
            against that order so the rename cannot be mistaken for the capability.

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
                >>> _ = m.set_bounds(Bounds(0.0, 0.0, 1.0, 1.0, crs=4326))
                >>> round(m.ax.get_xlim()[1])
                111319

                ```
            - The bare sequence is matplotlib's own ordering, in the display CRS:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> m = Map(crs=3857)
                >>> _ = m.set_bounds([0.0, 100.0, 0.0, 50.0])
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
                    f"set_bounds needs exactly 4 values as [xmin, xmax, ymin, ymax]; got {len(values)}"
                )
            xmin, xmax, ymin, ymax = values
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        return self

    #: The tier's own spelling of :meth:`set_bounds`, kept working for one release. It returns what
    #: `set_bounds` returns, so a caller who ignored the old ``None`` is unaffected and one who chains gets
    #: the map.
    set_extent = renamed_method(new="set_bounds", old="set_extent", owner="Map")

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
        self.set_bounds(box.to_crs(self.crs))

    # ------------------------------------------------------------------ globe / projection frame

    def graticule(
        self,
        lon_step: Optional[float] = None,
        lat_step: Optional[float] = None,
        *,
        spacing: Optional[float] = None,
        name: Optional[str] = None,
        visible: Optional[bool] = None,
    ) -> None:
        """Add a lon/lat graticule to a projected map (drawn when the frame is applied).

        A second call **replaces** the first — the map holds one set of graticule lines, so it draws one
        graticule — and the description follows: the layer keeps its id and its place in the tree and only
        what the replacing call named changes. Describing the second call as a second layer would say the
        map draws two grids where it draws one — and that is also why a ``name`` on a *replacing* call
        names nothing: the layer already has its id, and taking a new one would break every caller holding
        the old one. ``visible`` is the same story from the other side: a call that does not name it is
        asking for a different spacing, not for a hidden grid to come back (review R-L5).

        Args:
            lon_step: Meridian spacing in degrees; ``None`` (default) means
                :data:`DEFAULT_GRATICULE_STEP`.
            lat_step: Parallel spacing in degrees; ``None`` (default) means
                :data:`DEFAULT_GRATICULE_STEP`.
            spacing: One step for both, for a caller who wants a square grid; it overrides the two
                above, and **warns** when it does, because a call that names all three has had two of
                its own arguments thrown away (review R2-L3). The same **keyword** the web and
                interactive tiers take, so one call draws one grid on every tier that draws a graticule
                at all (#324). The **values** are each engine's own: this tier generates its meridians,
                so any positive step draws, and so does the web tier — while the interactive tier draws
                Natural Earth's pre-cut layers and honours only ``1``, ``5``, ``10``, ``15``, ``20`` and
                ``30``. ``spacing=7.5`` draws here and raises there (review R-L10).
            name: The caller's own name for the layer, used as its id and its label on the call that
                **creates** it; ``None`` (default) generates one from the kind (#321). A *replacing*
                call cannot rename the layer, so one that names a different name **warns** rather than
                dropping it in silence (review R2-L3).
            visible: Whether the graticule is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327). ``None`` (default)
                leaves the flag as it is: on the call that creates the layer that means drawn, and on a
                *replacing* call it means whatever the caller last chose, so restyling a hidden graticule
                does not put it back on screen (review R-L5).

        Warns:
            UserWarning: when ``spacing`` is given beside either step, which discards the step; and when a
                *replacing* call names a ``name`` the layer does not already carry, which is discarded
                because the id is the one thing a replacement cannot change (review R2-L3).

        Raises:
            Exception: whatever computing the grid raises — a spacing of zero divides by zero in the
                projection — after the description has been put back as it was. A figure must not name a
                layer that was not drawn, and a refused *replacement* must not restyle the graticule the
                map is still drawing (round 2, M1).
        """
        steps = _GraticuleSteps.asked(lon_step, lat_step, spacing)
        edit = self._describe_graticule(steps.symbology(), name=name, visible=visible)
        # Described first, then drawn — but through the renderer directly rather than through
        # `Scene._draw`, because a second call replaces the layer it already has rather than adding one,
        # and the funnel only knows how to add. The undo the funnel owns is therefore spelled by the edit,
        # which is the only thing that knows which of the two shapes this call took.
        try:
            self._renderer.draw_layer(self.figure_spec, edit.held)
        except BaseException:
            edit.undo(self)
            raise

    def _describe_graticule(
        self,
        symbology: Symbology,
        *,
        name: Optional[str],
        visible: Optional[bool],
    ) -> _GraticuleEdit:
        """Describe the one graticule layer, replacing the map's own rather than adding a second.

        Split from :meth:`graticule` because the two halves answer to different things: this one is the
        description — which layer the map already has, what a replacement may and may not change — while its
        caller is the draw and the undo. Keeping them in one method is what put four of that method's
        branches at a nesting level they did not need (`python:S3776`).

        Args:
            symbology: What the graticule is drawn at, from :meth:`_GraticuleSteps.symbology`.
            name: The caller's own name for the layer, honoured only on the call that creates it.
            visible: Whether the graticule is drawn, or ``None`` to leave the flag as it is — which means
                drawn on a creating call, and whatever the caller last chose on a replacing one.

        Returns:
            The edit, carrying what it takes to put the description back if the draw then raises.

        Warns:
            UserWarning: when a *replacing* call names a ``name`` the layer does not already carry, which is
                discarded because the id is the one thing a replacement cannot change (review R2-L3).
                ``stacklevel=3`` so it still names the line that called ``graticule()``.
        """
        pointer = self._graticule_id
        # `_reset_layers` clears the tree between animation frames while the lines themselves survive, so the
        # remembered id can outlive its layer; the membership test is what keeps that from raising.
        was: Optional[LayerSpec] = None
        if pointer is not None and pointer in self._layer_tree.ids:
            was = self._layer_tree.get(pointer)
        if was is None:
            held = self._describe_layer(
                LayerRecord(
                    "graticule",
                    name=name,
                    visible=True if visible is None else visible,
                    symbology=symbology,
                )
            )
            self._graticule_id = held
            return _GraticuleEdit(held, None, pointer)
        held = was.id
        if name is not None and name != held:
            warnings.warn(
                f"graticule(name={name!r}) is discarded: this call replaces the graticule already "
                f"described as {held!r}, and a replacement keeps the id every caller holding it "
                "knows. Name it on the call that creates it.",
                UserWarning,
                stacklevel=3,
            )
        self._layer_tree = self._layer_tree.replace(
            with_fields(
                was,
                symbology=symbology,
                visible=was.visible if visible is None else visible,
            )
        )
        return _GraticuleEdit(held, was, pointer)

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
        self.set_bounds([xlim[0], xlim[1], ylim[0], ylim[1]])

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
