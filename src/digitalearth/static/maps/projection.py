"""ProjectionMixin — extent/domain, the globe projection frame, and render/save/show hooks.

Sets the axes extent from a bbox or named domain, builds and caches the projection boundary/graticule for a
globe map, draws that graticule (with its degree labels) on a flat map, and overrides ``save``/``show`` to
apply the frame before output.
"""

import os
import warnings
from dataclasses import dataclass
from dataclasses import replace as with_fields
from math import isfinite
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Self,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from cleopatra.basemap.projection import apply_projection_frame
from matplotlib.collections import LineCollection
from matplotlib.text import Text
from pyramids.base.crs import reproject_coordinates

from digitalearth.base.domains import DomainLike, resolve_domain
from digitalearth.base.spec import Bounds, LayerSpec, Symbology, Viewport
from digitalearth.base.spec.bounds import same_crs
from digitalearth.static import projections
from digitalearth.static.renderer import DrawnLayer, artists_added
from digitalearth.static.scene import LayerRecord, drawing_style

#: The meridian and parallel spacing a graticule is drawn at when the caller names neither, in degrees —
#: the same default the interactive and web tiers take (#263). The two arguments default to ``None`` rather
#: than to this so that a step the caller wrote can be told from one they did not, which is what lets
#: :meth:`ProjectionMixin.graticule` say when ``spacing=`` has just discarded one (review R2-L3).
DEFAULT_GRATICULE_STEP: float = 30.0

#: How a flat map's grid looks when the caller asks for nothing else, in the **plural** keys a
#: ``LineCollection`` takes. Deliberately quiet: a graticule is a reference the reader consults, not a layer
#: they look at, and cleopatra's ``apply_projection_frame`` draws the globe's own grid in the same register.
_GRATICULE_STYLE: Dict[str, Any] = {
    "colors": "gray",
    "linewidths": 0.5,
    "linestyles": ":",
}

#: How a degree label looks. Small and grey for the same reason the lines are.
_GRATICULE_LABEL_STYLE: Dict[str, Any] = {"fontsize": 8, "color": "gray"}

#: The lon/lat window degree labels are placed in when nothing has framed the axes — the span the lines
#: themselves cover (:func:`digitalearth.static.projections.graticule` runs its meridians from -89.5 to
#: 89.5), so a label sits at the end of its own line rather than past it.
_LABEL_WORLD_WINDOW: Tuple[float, float, float, float] = (-180.0, -89.5, 180.0, 89.5)

#: How far into the window a label sits, as a fraction of the window's own span, measured from the lower
#: edge for a meridian and the left edge for a parallel. Non-zero so the text is inside the axes rather
#: than straddling its edge, and small enough to read as "at the edge".
_LABEL_INSET: float = 0.02

#: Samples per side of the grid the view's own lon/lat window is read from. A 5x5 grid over the *interior*
#: rather than the border, because an oblique projection can place the extreme longitude of a rectangle
#: inside it rather than on an edge.
_WINDOW_SAMPLES: int = 5

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.static.maps.base import GeoLayerBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


def _padded(box: Bounds, fraction: float) -> Bounds:
    """Return `box` grown by `fraction` of its own span, or `box` itself for no padding.

    One line, and a named one, because every spelling of :meth:`ProjectionMixin.set_bounds` has to pad the
    same way and a ``0.0`` must be a no-op rather than a rebuild.

    Args:
        box: The rectangle to grow.
        fraction: How much to grow it by, as a proportion of its width and height.

    Returns:
        The padded rectangle.

    Raises:
        ValueError: from :meth:`~digitalearth.base.spec.bounds.Bounds.padded`, for a fraction below ``-0.5``.
    """
    return box if not fraction else box.padded(fraction)


def _from_image(artist: Any, _axes: Any) -> Tuple[float, float, float, float]:
    """Return an image artist's extent, as ``(xmin, ymin, xmax, ymax)``.

    Args:
        artist: An ``AxesImage`` or anything else answering ``get_extent()``.
        _axes: Unused — an image states its extent in data coordinates already.

    Returns:
        The rectangle it covers. ``get_extent`` answers in matplotlib's ``(left, right, bottom, top)``, which
        runs backwards for a flipped image, so the pairs are ordered here.
    """
    left, right, bottom, top = artist.get_extent()
    return min(left, right), min(bottom, top), max(left, right), max(bottom, top)


def _from_collection(artist: Any, axes: Any) -> Tuple[float, float, float, float]:
    """Return a collection's data limits, as ``(xmin, ymin, xmax, ymax)``.

    Args:
        artist: A ``Collection`` — a scatter's paths, a mesh, a set of polygons.
        axes: The axes whose ``transData`` the limits are wanted in.

    Returns:
        The rectangle it covers. A collection with no paths answers a null box, whose edges are infinite and
        which :func:`_artist_extent` therefore drops.
    """
    xmin, ymin, xmax, ymax = artist.get_datalim(axes.transData).extents
    return float(xmin), float(ymin), float(xmax), float(ymax)


def _from_line(artist: Any, _axes: Any) -> Optional[Tuple[float, float, float, float]]:
    """Return a line artist's vertex extent, as ``(xmin, ymin, xmax, ymax)``.

    Args:
        artist: A ``Line2D`` or anything else answering ``get_xydata()``.
        _axes: Unused — the vertices are in data coordinates.

    Returns:
        The rectangle its vertices span, or ``None`` for a line with no vertices.
    """
    data = artist.get_xydata()
    if data is None or len(data) == 0:
        return None
    xs, ys = data[:, 0], data[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _from_point(artist: Any, _axes: Any) -> Tuple[float, float, float, float]:
    """Return a positioned artist's location as a zero-size rectangle.

    Args:
        artist: A ``Text`` or anything else answering ``get_position()``.
        _axes: Unused — the position is in data coordinates.

    Returns:
        ``(x, y, x, y)``. A label covers a point rather than an area, and a union with one still moves the
        frame to include it.
    """
    x, y = artist.get_position()
    return float(x), float(y), float(x), float(y)


#: How to read a data extent off an artist, in the order the readers are tried: the first whose method the
#: artist answers to wins. Written as a table rather than as a chain of ``isinstance`` checks because what
#: matters is the *question the artist answers*, not which matplotlib class it is — a third-party artist a
#: ``custom:matplotlib`` layer holds answers the same questions without inheriting from any of them.
_EXTENT_READERS: Tuple[Tuple[str, Any], ...] = (
    ("get_extent", _from_image),
    ("get_datalim", _from_collection),
    ("get_xydata", _from_line),
    ("get_position", _from_point),
)


def _artist_extent(
    artist: Any, axes: Any
) -> Optional[Tuple[float, float, float, float]]:
    """Return the region one artist covers, as ``(xmin, ymin, xmax, ymax)`` in data coordinates.

    Args:
        artist: The matplotlib artist to measure.
        axes: The axes it was drawn on, for the readers that need its ``transData``.

    Returns:
        The rectangle, or ``None`` for an artist that states no extent and for one whose extent is not finite
        — an empty collection answers an infinite null box, and a frame cannot be set from that.
    """
    for name, read in _EXTENT_READERS:
        if not callable(getattr(artist, name, None)):
            continue
        covered = read(artist, axes)
        if covered is None or not all(isfinite(edge) for edge in covered):
            return None
        return covered
    return None


def _drawn_extent(
    artists: Sequence[Any], axes: Any
) -> Optional[Tuple[float, float, float, float]]:
    """Return the region one layer's artists cover together.

    Args:
        artists: Every artist the layer owns.
        axes: The axes they were drawn on.

    Returns:
        The enclosing rectangle as ``(xmin, ymin, xmax, ymax)``, or ``None`` when none of them stated one —
        which is one layer having no extent to give rather than an error.
    """
    boxes = [
        covered
        for covered in (_artist_extent(artist, axes) for artist in artists)
        if covered is not None
    ]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


class _Frame(NamedTuple):
    """The rectangle one :meth:`ProjectionMixin.set_bounds` call settles on, and which way its axes run.

    Three spellings arrive at that method — a `Bounds` in any CRS, a bare ``(west, south, east, north)``
    sequence, and the ``None`` that fits the data — and exactly two things are done with whichever one came:
    the axes limits are set from it, and the rectangle the axes is then left holding is recorded as the region
    the view reports. A rectangle alone cannot carry both, because the sequence form may legitimately run
    backwards to invert an axis and a `Bounds` refuses corners the wrong way round. So the direction travels
    beside the rectangle rather than inside it.

    Attributes:
        box: The region, always corners-in-order, in the display CRS.
        flip_x: Whether the x axis runs from high to low.
        flip_y: Whether the y axis runs from high to low.
    """

    box: Bounds
    flip_x: bool = False
    flip_y: bool = False

    def limits(self) -> Tuple[float, float, float, float]:
        """Return the axes limits this frame sets, in matplotlib's own order.

        Returns:
            ``(xmin, xmax, ymin, ymax)``, with either pair swapped for an inverted axis.
        """
        xmin, xmax, ymin, ymax = self.box.as_mpl()
        if self.flip_x:
            xmin, xmax = xmax, xmin
        if self.flip_y:
            ymin, ymax = ymax, ymin
        return xmin, xmax, ymin, ymax


#: The axes limits matplotlib starts an unused axes at, read here as "nothing has framed this view yet".
#: :data:`digitalearth.static.maps.decoration._UNFRAMED_LIMITS` is the same pair for the same reason — a
#: replayed basemap meets the same unframed axes — and the two are spelled where they are read because
#: neither mixin imports the other.
_UNFRAMED_LIMITS: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)


@dataclass(frozen=True)
class _DegreeAxis:
    """One of the two directions a graticule runs in: the meridians, or the parallels.

    The pair of them (:data:`_MERIDIANS` and :data:`_PARALLELS`) is everything that differs between
    labelling a meridian and labelling a parallel — how far from zero its lines exist, the two hemisphere
    letters, where a label hangs, and which half of a lon/lat pair its degree is. Holding that as a value
    with its own behaviour is what keeps :class:`_Graticule` from asking "is this a meridian?" twice in
    every method.

    Attributes:
        limit: How far from zero this axis has lines at all, in degrees — ``180`` for meridians, ``90`` for
            parallels.
        positive: Hemisphere letter for the positive side (``"E"`` / ``"N"``).
        negative: Hemisphere letter for the negative side (``"W"`` / ``"S"``).
        ha: Horizontal alignment of a label on one of its lines.
        va: Vertical alignment of one.
        along_longitude: Whether this axis's own degree is the **longitude** of a label's anchor, which is
            true of a meridian (it runs along one longitude) and false of a parallel.

    Examples:
        - The two directions differ in how far they reach, which letters they use and where a label hangs
          — everything :class:`_Graticule` would otherwise ask "is this a meridian?" to find out:
            ```python
            >>> from digitalearth.static.maps.projection import _MERIDIANS, _PARALLELS
            >>> (_MERIDIANS.limit, _MERIDIANS.positive, _MERIDIANS.negative, _MERIDIANS.va)
            (180.0, 'E', 'W', 'bottom')
            >>> (_PARALLELS.limit, _PARALLELS.positive, _PARALLELS.negative, _PARALLELS.ha)
            (90.0, 'N', 'S', 'left')

            ```
        - So the same degree is read as a longitude on one and a latitude on the other:
            ```python
            >>> from digitalearth.static.maps.projection import _MERIDIANS, _PARALLELS
            >>> [line.text for line in _MERIDIANS.lines_within(45.0, -50.0, 50.0)]
            ['45°W', '0°', '45°E']
            >>> [line.text for line in _PARALLELS.lines_within(45.0, -50.0, 50.0)]
            ['45°S', '0°', '45°N']

            ```
    """

    limit: float
    positive: str
    negative: str
    ha: str
    va: str
    along_longitude: bool

    def lines_within(self, step: float, low: float, high: float) -> List["_GridLine"]:
        """Return this axis's lines inside ``[low, high]``, in ascending order.

        Anchored on zero rather than on ``low``, so the equator and the prime meridian are always among
        the lines — stepping up from an arbitrary edge misses them at any spacing that does not divide it.

        Args:
            step: Spacing between lines, in degrees; positive.
            low: Lower edge of the window to keep, in degrees.
            high: Upper edge of the window to keep.

        Returns:
            One :class:`_GridLine` per line the window holds; empty when it holds none.

        Examples:
            - A window 5 degrees clear of the next line either way holds three meridians:
                ```python
                >>> from digitalearth.static.maps.projection import _MERIDIANS
                >>> [line.text for line in _MERIDIANS.lines_within(30.0, -35.0, 35.0)]
                ['30°W', '0°', '30°E']

                ```
            - Nothing past ``limit`` is ever a line, however wide the window:
                ```python
                >>> from digitalearth.static.maps.projection import _PARALLELS
                >>> [line.value for line in _PARALLELS.lines_within(60.0, -400.0, 400.0)]
                [-60.0, 0.0, 60.0]

                ```
        """
        reach = int(self.limit // step)
        return [
            _GridLine(index, index * step, self)
            for index in range(-reach, reach + 1)
            if low <= index * step <= high
        ]

    def anchor(self, value: float, across: float) -> Tuple[float, float]:
        """Return the ``(lon, lat)`` a label sits at, from the line's degree and the across-line one.

        Args:
            value: The line's own degree.
            across: Where along the line the label goes, in the other coordinate.

        Returns:
            The anchor as a lon/lat pair, the right way round for this axis.

        Examples:
            - A meridian's degree is the longitude of its anchor, and a parallel's the latitude:
                ```python
                >>> from digitalearth.static.maps.projection import _MERIDIANS, _PARALLELS
                >>> (_MERIDIANS.anchor(30.0, -80.0), _PARALLELS.anchor(30.0, -80.0))
                ((30.0, -80.0), (-80.0, 30.0))

                ```
        """
        return (value, across) if self.along_longitude else (across, value)


#: The meridians of a graticule. A label on one hangs centred under it, so it reads as that meridian's
#: degree rather than as the nearest parallel's.
_MERIDIANS = _DegreeAxis(180.0, "E", "W", "center", "bottom", True)

#: The parallels. A label on one hangs to the right of its anchor, centred on the line.
_PARALLELS = _DegreeAxis(90.0, "N", "S", "left", "center", False)


class _GridLine(NamedTuple):
    """One meridian or parallel, named by the step index it sits at rather than by its degree.

    The index is carried beside the value because every question asked of a line — which hemisphere is it
    in, is it the zero line — is a question about *which* line it is, and asking that of a float would be
    an equality test on a computed product.

    Attributes:
        steps: Step count from the equator / prime meridian; its sign is the hemisphere. Named ``steps``
            and not ``index`` because a :class:`~typing.NamedTuple` already answers to ``index`` —
            ``tuple.index`` — and a field of that name shadows the method.
        value: The line's degree.
        axis: Which direction it runs in.
    """

    steps: int
    value: float
    axis: _DegreeAxis

    @property
    def hemisphere(self) -> str:
        """The hemisphere letter this line is in, or ``""`` for one that is in neither.

        Returns:
            The letter, empty for the zero line — the equator and the prime meridian belong to no
            hemisphere — and empty for the antimeridian, which is the same line from either side.

        Examples:
            - East, nowhere, and west:
                ```python
                >>> from digitalearth.static.maps.projection import _MERIDIANS, _GridLine
                >>> [
                ...     _GridLine(steps, steps * 30.0, _MERIDIANS).hemisphere
                ...     for steps in (2, 0, -2, -6)
                ... ]
                ['E', '', 'W', '']

                ```
        """
        if self.steps == 0 or abs(self.value) >= self.axis.limit:
            return ""
        return self.axis.positive if self.steps > 0 else self.axis.negative

    @property
    def text(self) -> str:
        """The degree label this line carries, e.g. ``"30°E"``.

        The format is the web tier's own (``digitalearth.web.decoration._graticule_line``), to the
        character, so one spacing reads the same way on both tiers.

        Returns:
            The degree with no trailing zeros, a degree sign, and the hemisphere letter where there is one.

        Examples:
            - A whole degree, a fractional one, and the prime meridian:
                ```python
                >>> from digitalearth.static.maps.projection import _MERIDIANS, _PARALLELS, _GridLine
                >>> (
                ...     _GridLine(1, 30.0, _MERIDIANS).text,
                ...     _GridLine(-1, -7.5, _PARALLELS).text,
                ...     _GridLine(0, 0.0, _MERIDIANS).text,
                ... )
                ('30°E', '7.5°S', '0°')

                ```
        """
        return f"{abs(self.value):g}°{self.hemisphere}"


class _DegreeLabel(NamedTuple):
    """One degree label, placed: where it goes in the display CRS, what it reads, and how it hangs.

    Placed before anything reaches the axes and drawn afterwards, because placing a label can *fail* — a
    projection need not have a finite image of the point one is anchored at — and a drawer that has already
    added artists when it finds that out is a drawer that has to take them off again.

    Attributes:
        x: Position in the display CRS.
        y: Position in the display CRS.
        line: The meridian or parallel it labels, which is what it reads and how it hangs.
    """

    x: float
    y: float
    line: _GridLine

    def draw(self, axes: Any) -> Text:
        """Put this label on an axes.

        Args:
            axes: The axes to draw on.

        Returns:
            The :class:`~matplotlib.text.Text` it added, which the layer then owns — so hiding or removing
            the graticule reaches its degrees as well as its lines.
        """
        return axes.text(
            self.x,
            self.y,
            self.line.text,
            ha=self.line.axis.ha,
            va=self.line.axis.va,
            **_GRATICULE_LABEL_STYLE,
        )


class _Graticule:
    """The grid one described graticule layer asks for, and how it goes on the axes.

    Built per draw from the scene and the layer's description and thrown away after, so it holds no state
    the map carries between draws — the one thing that outlives it is the projected lines, which the map
    keeps because a globe's projection frame draws them later.

    Two frames, two ways of putting the one grid on the axes:

    - a **globe** is clipped at its limb, and cleopatra's `apply_projection_frame` draws its grid after
      every data layer — so :meth:`draw` hands the lines over and leaves the axes alone;
    - a **flat** map has no such pass, so :meth:`draw` puts the grid on now, as one `LineCollection`
      plus one :class:`~matplotlib.text.Text` per labelled line.

    Until #221 the flat map had no pass at all: the lines were computed, stored, and nothing ever read
    them, so `Map(crs=4326).graticule()` — the default frame — registered a layer and drew nothing while
    the figure described the layer as drawn.

    Examples:
        - The grid a framed flat map asks for: the lines span the world, and the labels are only the
          degrees the *view* holds:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static import Map
            >>> from digitalearth.static.maps.projection import _Graticule
            >>> m = Map(crs=4326)
            >>> _ = m.set_bounds([-35.0, -5.0, 35.0, 65.0])
            >>> grid = _Graticule(m, {"lon_step": 30.0, "lat_step": 30.0, "labels": True})
            >>> len(grid.lines), tuple(round(v, 1) for v in grid.window())
            (18, (-35.0, -5.0, 35.0, 65.0))
            >>> [label.line.text for label in grid.labels()]
            ['30°W', '0°', '30°E', '0°', '30°N', '60°N']
            >>> m.close()

            ```
        - On a globe the same spacing gives the same lines, and the window falls back to the span the
          lines themselves cover, because nothing has framed the axes:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static import Map
            >>> from digitalearth.static.maps.projection import _Graticule
            >>> globe = Map(crs=4326, globe=True)
            >>> grid = _Graticule(globe, {"lon_step": 30.0, "lat_step": 30.0, "labels": False})
            >>> len(grid.lines), grid.window()
            (18, (-180.0, -89.5, 180.0, 89.5))
            >>> globe.close()

            ```
    """

    def __init__(self, scene: Any, props: Mapping[str, Any]) -> None:
        """Compute the grid a description asks for.

        Args:
            scene: The map being drawn on.
            props: The layer's symbology props, as :meth:`_GraticuleSteps.symbology` wrote them —
                ``lon_step``, ``lat_step`` and ``labels``. A figure stored before the labels existed
                carries no ``labels`` key and is read as asking for them, which is this tier's default.

        Raises:
            ZeroDivisionError: from the projection, for a step of zero. Raised here, before anything is
                drawn or recorded, which is what lets a refused *replacement* leave the grid the map is
                still drawing exactly as it was (round 2, M1).
        """
        self._scene = scene
        self._lon_step = props["lon_step"]
        self._lat_step = props["lat_step"]
        self._labelled = bool(props.get("labels", True))
        #: The projected polylines, meridians then parallels, split at the projection limb.
        self.lines: List[Any] = projections.graticule(
            scene.crs, lon_step=self._lon_step, lat_step=self._lat_step
        )

    def draw(self, layer: LayerSpec) -> DrawnLayer:
        """Put this grid where its frame draws it, and say what it left behind.

        Args:
            layer: The layer being drawn, for its style and for the id its previous drawing is under.

        Returns:
            A :class:`~digitalearth.static.renderer.DrawnLayer` whose ``artist`` is the projected lines —
            this builder's own value, the same on both frames — and whose ``artists`` are what is on the
            axes to hide and remove: the collection and its labels on a flat map, and **nothing yet** on a
            globe, where :meth:`ProjectionMixin._apply_frame` attaches them when the frame goes on.

        Warns:
            UserWarning: from :meth:`labels`, naming any degree this frame cannot place.
        """
        if self._scene.globe:
            self._scene._graticule_lines = self.lines
            return DrawnLayer(artist=self.lines)
        # Everything that can raise or warn runs before the first artist reaches the axes: the style the
        # description carries, and the label placement, which refuses rather than drawing half a grid.
        style = {**_GRATICULE_STYLE, **drawing_style(self._scene, layer)}
        labels = self.labels() if self._labelled else ()
        self._scene._graticule_lines = self.lines
        return self._on_flat_axes(layer, style, labels)

    def labels(self) -> Tuple[_DegreeLabel, ...]:
        """Place a degree label on every line the view holds.

        **Where they go, and why there.** Each label sits on its own line, just inside the lower edge of
        the view for a meridian and just inside the left edge for a parallel (:data:`_LABEL_INSET` of the
        window's span). That is one anchor per line rather than a tick on an axis, which is what lets the
        same rule serve a projected frame: the anchor is a lon/lat point, projected through pyramids like
        everything else here, so on a curved graticule the label follows its own line instead of sitting
        where a straight axis would have put it. The whole set is projected in **one** call, because a
        per-label call costs more in PROJ overhead than the projection itself.

        Returns:
            One :class:`_DegreeLabel` per line that could be placed, meridians first.

        Warns:
            UserWarning: naming every degree the display CRS or the view would not take. A graticule that
                quietly came back with fewer labels than lines is the defect this row exists to end, so
                the one case left — a line the frame genuinely cannot label — states itself.
        """
        window = self.window()
        if window is None:
            warnings.warn(
                f"graticule() drew no degree labels: no sample of this view has a longitude/latitude in "
                f"{self._scene.crs!r}, so there is no window to place them in.",
                UserWarning,
            )
            return ()
        lon_min, lat_min, lon_max, lat_max = window
        across = {
            _MERIDIANS: lat_min + _LABEL_INSET * (lat_max - lat_min),
            _PARALLELS: lon_min + _LABEL_INSET * (lon_max - lon_min),
        }
        asked = _MERIDIANS.lines_within(
            self._lon_step, lon_min, lon_max
        ) + _PARALLELS.lines_within(self._lat_step, lat_min, lat_max)
        if not asked:
            return ()
        anchors = [line.axis.anchor(line.value, across[line.axis]) for line in asked]
        x, y = reproject_coordinates(
            [lon for lon, _ in anchors],
            [lat for _, lat in anchors],
            from_crs=4326,
            to_crs=self._scene.crs,
        )
        return self._placed(asked, np.asarray(x, float), np.asarray(y, float))

    def window(self) -> Optional[Tuple[float, float, float, float]]:
        """Return the lon/lat window the view covers, as ``(lon_min, lat_min, lon_max, lat_max)``.

        What the labels are placed in: the lines span the world whatever the map is looking at, so the
        degrees worth drawing are the ones the *view* holds. The view is in the display CRS, so it is
        converted back to lon/lat through pyramids — over a grid of its interior rather than its border,
        because an oblique projection can place a rectangle's extreme longitude inside it.

        Returns:
            The window, clipped to the span the lines themselves cover; :data:`_LABEL_WORLD_WINDOW` when
            nothing has framed the axes yet, which is a view of the whole world once the grid autoscales
            it; or ``None`` when no sample of the view has a lon/lat at all, which is a view the display
            CRS cannot place and so a window nothing can be labelled in.
        """
        xmin, xmax = (float(value) for value in self._scene.ax.get_xlim())
        ymin, ymax = (float(value) for value in self._scene.ax.get_ylim())
        if (xmin, xmax, ymin, ymax) == _UNFRAMED_LIMITS:
            return _LABEL_WORLD_WINDOW
        xs, ys = np.meshgrid(
            np.linspace(min(xmin, xmax), max(xmin, xmax), _WINDOW_SAMPLES),
            np.linspace(min(ymin, ymax), max(ymin, ymax), _WINDOW_SAMPLES),
        )
        lon, lat = reproject_coordinates(
            xs.ravel().tolist(),
            ys.ravel().tolist(),
            from_crs=self._scene.crs,
            to_crs=4326,
        )
        lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
        seen = np.isfinite(lon) & np.isfinite(lat)
        if not seen.any():
            return None
        west, south, east, north = _LABEL_WORLD_WINDOW
        return (
            max(float(lon[seen].min()), west),
            max(float(lat[seen].min()), south),
            min(float(lon[seen].max()), east),
            min(float(lat[seen].max()), north),
        )

    def _placed(
        self, asked: List[_GridLine], x: np.ndarray, y: np.ndarray
    ) -> Tuple[_DegreeLabel, ...]:
        """Keep the labels whose projected anchor the view can show, and name the ones it cannot.

        Split from :meth:`labels` because the two answer different questions: that one is *which degrees,
        and where in lon/lat*, this one is *which of them the display CRS and the axes will take*.

        Args:
            asked: The lines a label was placed for, in order.
            x: The projected x of each anchor, parallel to ``asked``; non-finite where the projection has
                no image of it.
            y: The projected y of each anchor.

        Returns:
            The labels that can be drawn, in the order they were asked for.

        Warns:
            UserWarning: naming the degrees that were dropped.
        """
        xmin, xmax = sorted(float(value) for value in self._scene.ax.get_xlim())
        ymin, ymax = sorted(float(value) for value in self._scene.ax.get_ylim())
        # An unframed axes is about to be autoscaled to the grid itself, so its limits say nothing about
        # what will be visible and a label is not dropped for falling outside the unit square.
        framed = (xmin, xmax, ymin, ymax) != _UNFRAMED_LIMITS
        placed: List[_DegreeLabel] = []
        dropped: List[str] = []
        for line, px, py in zip(asked, x, y):
            outside = framed and not (xmin <= px <= xmax and ymin <= py <= ymax)
            if not (isfinite(px) and isfinite(py)) or outside:
                dropped.append(line.text)
            else:
                placed.append(_DegreeLabel(float(px), float(py), line))
        if dropped:
            warnings.warn(
                f"graticule() could not place {len(dropped)} degree label(s): {', '.join(dropped)}. The "
                f"display CRS {self._scene.crs!r} gives their anchor no position inside the view the "
                "figure is framed on; frame the map on a region that holds them, or pass labels=False.",
                UserWarning,
            )
        return tuple(placed)

    def _on_flat_axes(
        self,
        layer: LayerSpec,
        style: Dict[str, Any],
        labels: Tuple[_DegreeLabel, ...],
    ) -> DrawnLayer:
        """Put the grid on a flat axes, replacing whatever this layer drew before.

        A flat map **replaces** its own previous drawing rather than adding a second one beside it:
        ``graticule()`` describes one layer however often it is called, and ``Renderer.draw_layer`` does
        not take a previous drawing off. The collection is reused where it can be — it is then still the
        artist the previous record names, so a draw refused further downstream (a key this tier cannot
        draw) rolls that record back onto an artist the axes is still holding — and rebuilt when it cannot,
        which is any drawing an ``ax.clear()`` between animation frames has already detached.

        Args:
            layer: The layer being drawn, for the id its previous drawing is recorded under.
            style: The collection's keywords.
            labels: The labels, already placed.

        Returns:
            The record of what is now on the axes.

        Note:
            The view is held across the draw (``_preserve_view``) because the grid spans the world whatever
            the map is looking at — the same guard every global decoration here takes. On an axes nothing
            has framed yet the block is free to set the extent, so a bare ``Map().graticule()`` ends up
            looking at the world the grid covers.
        """
        held = self._previous_artists(layer)
        reused = next(
            (artist for artist in held if isinstance(artist, LineCollection)), None
        )
        for artist in held:
            if isinstance(artist, Text):
                artist.remove()
        with self._scene._preserve_view():
            if reused is None:
                collection = LineCollection(self.lines, **style)
                self._scene.ax.add_collection(collection)
            else:
                collection = reused
                collection.set_segments(self.lines)
                collection.update(style)
            drawn = tuple(label.draw(self._scene.ax) for label in labels)
        return DrawnLayer(artist=self.lines, artists=(collection, *drawn))

    def _previous_artists(self, layer: LayerSpec) -> Tuple[Any, ...]:
        """Return the artists this layer's previous drawing still has on *this* axes.

        Args:
            layer: The layer being drawn.

        Returns:
            Its previously drawn artists, filtered to the ones still attached to this scene's axes — an
            animation's ``ax.clear()`` detaches them while the record survives, and a detached collection
            reused as this grid would be a grid on no figure.
        """
        drawn = self._scene._renderer.drawn.get(layer.id)
        if drawn is None:
            return ()
        return tuple(
            artist for artist in drawn.artists if artist.axes is self._scene.ax
        )


class _GraticuleSteps(NamedTuple):
    """The meridian and parallel spacing one :meth:`ProjectionMixin.graticule` call settles on, in degrees.

    Three arguments, one decision. ``lon_step``, ``lat_step`` and ``spacing`` are the only values that travel
    together through the front of that method, and the only thing done to them is to collapse them into these
    two — with a warning when the caller wrote arguments the collapse throws away. Lifting them out with that
    rule is most of what took `graticule` under the cognitive-complexity bar (`python:S3776`), and it makes
    the rule testable without a map: the collapse is a pure function of what the caller wrote.

    :meth:`symbology` is here for the same reason — the props a graticule layer is described by are these two
    steps and whether it is labelled, and `draw_graticule` reads exactly the keys this writes.

    Attributes:
        lon: Meridian spacing in degrees.
        lat: Parallel spacing in degrees.
        labels: Whether each line carries its degree. Settled by
            :meth:`ProjectionMixin._labels_asked` before this value is built, because it is the one of the
            three that depends on the *frame* rather than on what the caller wrote.
    """

    lon: float
    lat: float
    labels: bool = True

    @classmethod
    def held_by(cls, layer: Optional[LayerSpec]) -> Optional["_GraticuleSteps"]:
        """Read back what a graticule layer the map already describes is drawn at.

        The other half of :meth:`symbology`, and the reason a *replacing*
        :meth:`ProjectionMixin.graticule` call can keep an option it was not given: the props are the only
        record of what the grid already on the axes was asked for.

        Args:
            layer: The described graticule layer, or ``None`` when the map has none yet.

        Returns:
            The steps and label choice that layer carries, or ``None`` for ``None`` — which is what the
            creating call is given, and why its defaults stay the published ones. A prop a layer somehow
            does not carry falls back the way a creating call would.
        """
        if layer is None:
            return None
        symbology = getattr(layer, "symbology", None)
        props: Dict[str, Any] = {} if symbology is None else dict(symbology.props)
        return cls(
            float(props.get("lon_step", DEFAULT_GRATICULE_STEP)),
            float(props.get("lat_step", DEFAULT_GRATICULE_STEP)),
            bool(props.get("labels", True)),
        )

    @classmethod
    def asked(
        cls,
        lon_step: Optional[float],
        lat_step: Optional[float],
        spacing: Optional[float],
        labels: bool = True,
        carried: Optional["_GraticuleSteps"] = None,
    ) -> "_GraticuleSteps":
        """Collapse what a caller wrote into the two steps a graticule is drawn at.

        Args:
            lon_step: Meridian spacing the caller named, or ``None``.
            lat_step: Parallel spacing the caller named, or ``None``.
            spacing: One step for both, which outranks the other two.
            labels: Whether the lines are labelled, already settled against the frame.
            carried: What the graticule being *replaced* is drawn at, from :meth:`held_by`, or ``None`` on
                the call that creates it. A step the caller does not name is taken from here: a call that
                names one step is asking for that step, not for the other one to be re-cut at the default
                (round 1, M7). ``spacing`` outranks it, since that argument names both steps.

        Returns:
            The two steps — each the caller's own, else ``carried``'s, else
            :data:`DEFAULT_GRATICULE_STEP` — and the label choice.

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
            return cls(spacing, spacing, labels)
        standing = (
            cls(DEFAULT_GRATICULE_STEP, DEFAULT_GRATICULE_STEP)
            if carried is None
            else carried
        )
        lon = standing.lon if lon_step is None else lon_step
        lat = standing.lat if lat_step is None else lat_step
        return cls(lon, lat, labels)

    def symbology(self) -> Symbology:
        """Return the symbology a graticule layer is described by.

        Returns:
            A :class:`~digitalearth.base.spec.Symbology` carrying the drawer key, the two steps and the
            label choice — the props :func:`draw_graticule` reads back. ``labels`` travels because the
            labels are drawn *from* the description: a stored figure read back elsewhere would otherwise
            come back labelled when it was not.
        """
        return Symbology(
            props={
                "via": "graticule",
                "lon_step": self.lon,
                "lat_step": self.lat,
                "labels": self.labels,
            }
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
    """Draw the lon/lat graticule a described layer asks for, at the spacing it recorded.

    The drawer the kind registry dispatches to; :class:`_Graticule` is what it is, and holds both frames'
    behaviour and the label placement.

    Args:
        scene: The map being drawn on.
        _data: The source slot every drawer takes, unread here — a graticule is computed from the display
            CRS and two spacings, not from data.
        layer: The layer's description.

    Returns:
        What :meth:`_Graticule.draw` left behind: the collection and its degree labels on a flat map, and
        the lines alone on a globe, whose artists the projection frame attaches later.

    Raises:
        ZeroDivisionError: from the projection, for a spacing of zero — before anything is drawn.

    Warns:
        UserWarning: naming any degree the frame cannot place.
    """
    return _Graticule(scene, layer.symbology.props).draw(layer)


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

    #: The region the last framing call left the axes holding, in the display CRS, or ``None`` while nothing
    #: has framed the axes. Declared on the class rather than set in a constructor, because this mixin is
    #: composed into :class:`~digitalearth.static.map.Map` and owns no ``__init__`` of its own — the class
    #: attribute is the default every instance reads until :meth:`set_bounds` writes one of its own.
    #:
    #: It is what :attr:`viewport` reports, and it is written by :meth:`set_bounds` from the limits it has
    #: just set (:meth:`_frame_held`) rather than from the rectangle it was handed, because matplotlib expands
    #: a singular limit and a zero-span frame was otherwise reported as a rectangle nothing was showing
    #: (review R2-M10). It is *not* the axes limits read on demand: those are whatever the last autoscale left
    #: behind, which on an unframed figure is matplotlib's unit square, and ``None`` there is the honest
    #: answer rather than that square.
    _frame_bounds: Optional[Bounds] = None

    @property
    def viewport(self) -> Viewport:
        """Where the map is looking, as a value — including the region it has been framed on.

        Returns:
            The view :class:`~digitalearth.static.maps.base.GeoLayerBase` builds from the display CRS, the
            declared domain and the globe flag, carrying the region the last framing call asked for when
            there has been one. A view holds a region **or** a named domain, never both, so a map that was
            built with a ``domain`` and then framed reports the rectangle it is actually showing — which is
            what :meth:`~digitalearth.base.spec.viewport.Viewport.framed` does, and what keeps a stale
            domain from sitting beside a live frame.

            Before order 26 this always answered ``None`` for `bounds`: a map framed by its own data
            described itself as unframed, which is why the basemap coverage guard had to record the axes
            extent on its own layer instead of reading it off the panel.

            A frame recorded in a CRS the map no longer draws in is **not** reported, and is not converted
            either. A rotation swaps the display CRS between frames, and the region a previous frame was
            globally framed on is often one the next projection cannot show at all — so the honest answer
            there is that nothing has framed *this* view yet, rather than a rectangle reprojected across a
            limb or a refusal raised from a property.
        """
        view = super().viewport
        framed = self._frame_bounds
        if framed is None or not same_crs(framed.crs, view.crs):
            return view
        # `framed()` rather than a field write: a view holds a region or a domain and never both, and that is
        # the method that drops the one for the other. Its reprojection is a no-op, the CRSs having matched.
        return view.framed(framed)

    def set_bounds(
        self,
        bounds: Optional[Union[Bounds, Sequence[float]]] = None,
        *,
        padding: float = 0.0,
    ) -> Self:
        """Frame the figure on a region, or on everything it draws — the Core contract's framing method.

        Args:
            bounds: A :class:`~digitalearth.base.spec.bounds.Bounds` in **any** CRS — it is reprojected to
                the display CRS, which is the point of passing one — or a bare
                ``(west, south, east, north)`` sequence taken to be in the display CRS already. That is bbox
                order: the order `Bounds` holds its own four numbers in, the order
                :meth:`~digitalearth.base.spec.bounds.Bounds.as_bbox` writes and
                :meth:`~digitalearth.base.spec.bounds.Bounds.from_bbox` reads, and the order the interactive
                and web tiers' ``set_bounds`` take. Prefer `Bounds` even so — it states the CRS as well as
                the ordering, instead of leaving either to position and assumption.

                ``None`` (the default) **fits the figure to its data**: the union of what the panel's data
                layers cover, measured from the artists they actually put on the axes. Only the ``data``
                band counts — a basemap, a graticule and a coastline are drawn around the subject rather
                than being it, and a global coastline would otherwise answer every such call with the world.
                A layer built hidden does not count either.
            padding: Breathing room around the frame, as a fraction of its own span — ``0.05`` leaves a 5%
                margin on every side. It applies to a rectangle you name exactly as it applies to a fitted
                one. Note the **units are this tier's**: a figure is framed in data coordinates here, so a
                proportion of the span is the only padding that means anything, where the web tier's
                ``padding`` is a number of screen pixels.

        Returns:
            This map, so the call chains (``Map(crs=3857).set_bounds(bounds).coastlines()``). The Core
            declares ``returns="self"`` for this name, and every tier answers that way: the spelling
            this tier used to carry returned ``None``, so one line worked on one tier and raised on another.

        Raises:
            ValueError: if the sequence form does not hold exactly four values; if ``bounds=None`` and the
                figure draws nothing with an extent, because silently doing nothing is indistinguishable
                from the call being dropped; or, from `Bounds`, for a non-finite edge or a ``padding``
                below ``-0.5``, which would turn the frame inside out.

        Notes:
            A **flipped** pair is honoured in the sequence form: an ``east`` west of its ``west`` inverts the
            x axis, which is how matplotlib expresses ``invert_xaxis`` through the limits, and ``padding``
            grows such an axis outwards rather than inwards. :attr:`viewport` reports the *region* either way
            round: a `Bounds` refuses corners the wrong way round, and the direction an axis runs in is a
            property of the axes rather than of the region shown.

            The sequence used to be read as matplotlib's own ``[xmin, xmax, ymin, ymax]``, which is the order
            the axes limits are set in and the order the method carried under its old name. It was the only
            spelling in the package that read four bare numbers that way, so one call framed two different
            rectangles across the 2-D tiers; the matplotlib ordering is reachable from `Bounds` alone now,
            through :meth:`~digitalearth.base.spec.bounds.Bounds.as_mpl`.

            What :attr:`viewport` reports is read **back off the axes** once the limits are set, so it is the
            rectangle the figure is holding rather than the one that was asked for. The two differ for a frame
            of zero span — a fit onto a single point, or ``padding=-0.5``, the bottom of the legal range:
            matplotlib refuses a singular limit and expands it (by 5% of the value, warning as it goes), and
            the region reported is that expanded one. It is read at framing time and not re-read, so a later
            builder that autoscales the axes moves the picture without moving what this reports.

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
            - The bare sequence is ``(west, south, east, north)``, in the display CRS, and the view reports it
              back in the order it was written:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> m = Map(crs=3857)
                >>> _ = m.set_bounds([0.0, 0.0, 100.0, 50.0])
                >>> [float(v) for v in m.ax.get_xlim()], [float(v) for v in m.ax.get_ylim()]
                ([0.0, 100.0], [0.0, 50.0])
                >>> m.viewport.bounds.as_bbox()
                [0.0, 0.0, 100.0, 50.0]

                ```
            - ``padding`` grows the frame by a fraction of its span:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> m = Map(crs=3857)
                >>> _ = m.set_bounds([0.0, 0.0, 10.0, 10.0], padding=0.1)
                >>> [float(v) for v in m.ax.get_xlim()]
                [-1.0, 11.0]

                ```
        """
        frame = self._frame_asked(bounds, padding)
        xmin, xmax, ymin, ymax = frame.limits()
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        # Read back, not `frame.box`: matplotlib will not hold a singular limit and expands one as it is set,
        # so a zero-span frame — a fit onto one point, or `padding=-0.5` — was recorded as a rectangle the
        # axes was not holding and the figure never showed (review R2-M10).
        self._frame_bounds = self._frame_held(frame.box.crs)
        return self

    def _frame_held(self, crs: Any) -> Bounds:
        """Return the rectangle the axes is holding, as a region in `crs`.

        Args:
            crs: The display CRS the limits are measured in, taken from the frame they were set from.

        Returns:
            The limits as a `Bounds`, corners in order: which way each axis runs is a property of the axes
            rather than of the region, so an inverted pair is read back as the same rectangle it frames.
        """
        left, right = (float(value) for value in self.ax.get_xlim())
        bottom, top = (float(value) for value in self.ax.get_ylim())
        return Bounds(
            min(left, right), min(bottom, top), max(left, right), max(bottom, top), crs
        )

    def _frame_asked(
        self,
        bounds: Optional[Union[Bounds, Sequence[float]]],
        padding: float,
    ) -> "_Frame":
        """Resolve what a caller asked for into one rectangle in the display CRS, padded.

        Split from :meth:`set_bounds` because the two halves answer different questions: this one is *which
        rectangle*, its caller is *what to do with it* — set the limits, and record the region the view
        reports. Each of the three spellings pads exactly once, inside its own branch, so no route can pad
        twice and none can skip it.

        Args:
            bounds: What the caller passed — a `Bounds`, a ``(west, south, east, north)`` sequence, or
                ``None`` to fit the data.
            padding: The fraction to grow the rectangle by.

        Returns:
            The frame, carrying the rectangle and which way each axis runs.

        Raises:
            ValueError: for a sequence that is not four values, for ``None`` with nothing to frame on, or
                from `Bounds` for a non-finite edge or a padding that would invert the rectangle.

        Notes:
            The sequence branch is where the ordering changed, and the flipped-axis contract was preserved
            through that change **for the two literals the suite frames with** rather than by construction.
            Going from the matplotlib reading this tier used to carry to this one swaps positions 2 and 3, and
            in ``[10, 0, 0, 10]`` and ``[0, 10, 10, 0]`` those two positions hold equal values — so the swap
            is a no-op on them before any normalisation runs, and both readings frame the same rectangle. A
            flipped literal whose middle pair differs frames two different rectangles. Executed over the four:

            | literal | this reading | the matplotlib reading | same |
            |---|---|---|---|
            | ``[10, 0, 0, 10]`` | x 10→0, y 0→10 | x 10→0, y 0→10 | yes |
            | ``[0, 10, 10, 0]`` | x 0→10, y 10→0 | x 0→10, y 10→0 | yes |
            | ``[10, 0, 5, 10]`` | x 10→5, y 0→10 | x 10→0, y 5→10 | no |
            | ``[0, 10, 20, 30]`` | x 0→20, y 10→30 | x 0→10, y 20→30 | no |

            So a flip is not order-agnostic, and the two palindromic literals are evidence that the *contract*
            survived rather than that the orderings agree. Both of the differing literals are pinned in
            ``tests/static/test_static_set_bounds.py``, so neither reading can be reintroduced quietly.
        """
        if bounds is None:
            return _Frame(self._fitted_box(padding))
        if isinstance(bounds, Bounds):
            # to_crs is a no-op when the CRSs already match. Without it a rectangle that carries its CRS
            # would be trusted to be in the display one, which is exactly the mistake Bounds exists to stop.
            return _Frame(_padded(bounds.to_crs(self.crs), padding))
        values = [float(value) for value in bounds]
        if len(values) != 4:
            raise ValueError(
                f"set_bounds needs exactly 4 values as (west, south, east, north); got {len(values)}"
            )
        west, south, east, north = values
        # Normalised into a Bounds so the padding and the recorded region are the same arithmetic every
        # other spelling gets; which way each axis runs travels beside it, because a Bounds cannot hold it.
        box = Bounds(
            min(west, east),
            min(south, north),
            max(west, east),
            max(south, north),
            self.crs,
        )
        return _Frame(_padded(box, padding), east < west, north < south)

    def _fitted_box(self, padding: float) -> Bounds:
        """Return the region the panel's own data layers cover, padded.

        Args:
            padding: The fraction to grow the union by.

        Returns:
            The union, in the display CRS. The panel does the merging
            (:meth:`~digitalearth.base.spec.figure.PanelSpec.bounds_of`): which layers count is its layer
            list and which CRS they meet in is its view, and neither is this tier's to decide.

        Raises:
            ValueError: when no data layer had an extent to give. Framing on nothing and doing nothing look
                identical from the outside, so the refusal says what to pass instead.
        """
        panel = self.figure_spec.panels[0]
        box = panel.bounds_of(self._data_extents(), padding=padding)
        if box is None:
            raise ValueError(
                "set_bounds() has nothing to frame on: this figure draws no data layer with an extent — "
                "a basemap, graticule or coastline is drawn around the subject rather than being it, and a "
                "hidden layer is not drawn at all. Add the data first, or pass a rectangle."
            )
        return box

    def _data_extents(self) -> dict:
        """Return what each of the panel's data layers covers, by layer id, in the display CRS.

        Read from the **artists**, not from the sources: a layer is drawn where the reprojection and the
        cell-edge maths left it, and that is the region the frame has to hold. A layer with nothing on the
        axes — a graticule before the frame goes on, a layer whose data fell off the limb — simply has no
        entry, which :meth:`~digitalearth.base.spec.figure.PanelSpec.bounds_of` reads as "no extent to
        give" rather than as an error.

        Returns:
            ``{layer_id: Bounds}`` for the visible layers of the ``data`` band.
        """
        measured = {}
        for layer in self._layer_tree.layers:
            if self._renderer.band_for(layer) != "data":
                continue
            if not self._layer_tree.is_visible(layer.id):
                continue
            drawn = self._renderer.drawn.get(layer.id)
            covered = None if drawn is None else _drawn_extent(drawn.artists, self.ax)
            if covered is not None:
                measured[layer.id] = Bounds(*covered, self.crs)
        return measured

    def set_domain(self, domain: Optional[DomainLike] = None) -> Self:
        """Set the axes extent from a named region or bbox, reprojected to the display CRS via pyramids.

        Args:
            domain: A registered region name (e.g. ``"Europe"``), an explicit ``(west, south, east, north)``
                bbox in EPSG:4326, or ``None`` to fall back to the domain passed at construction. A no-op
                when neither resolves to a domain.

        Returns:
            This map, so the call chains like :meth:`set_bounds`, which it frames through. **The no-op
            path answers the map as well**: a call with nothing to resolve frames nothing, and a chain
            must survive it (L6).

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
                >>> _ = m.set_domain("europe")
                >>> [float(v) for v in m.ax.get_xlim()]
                [-25.0, 45.0]
                >>> [float(v) for v in m.ax.get_ylim()]
                [34.0, 72.0]

                ```
        """
        bbox = resolve_domain(domain if domain is not None else self.domain)
        if bbox is None:
            return self
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
        return self.set_bounds(box.to_crs(self.crs))

    # ------------------------------------------------------------------ globe / projection frame

    def graticule(
        self,
        lon_step: Optional[float] = None,
        lat_step: Optional[float] = None,
        *,
        spacing: Optional[float] = None,
        labels: Optional[bool] = None,
        name: Optional[str] = None,
        visible: Optional[bool] = None,
    ) -> Self:
        """Add a lon/lat graticule to the map, with its degrees labelled.

        A second call **replaces** the first — the map holds one set of graticule lines, so it draws one
        graticule — and the description follows: the layer keeps its id and its place in the tree and only
        what the replacing call named changes. Describing the second call as a second layer would say the
        map draws two grids where it draws one — and that is also why a ``name`` on a *replacing* call
        names nothing: the layer already has its id, and taking a new one would break every caller holding
        the old one.

        **One rule for the rest of the arguments: a replacing call keeps every option it does not name.**
        A call that names a spacing is asking for a spacing, not for a hidden grid to come back
        (``visible``, review R-L5), nor for the degrees of a deliberately bare grid to come back
        (``labels``), nor for the step it did not mention to be re-cut at the default (``lon_step`` /
        ``lat_step``) — the last two used to fall back to their construction defaults, which relabelled
        and re-cut grids in silence (round 1, M7). ``spacing`` is the one argument that overrides what the
        grid carries without being asked twice, because it names both steps by definition. The frame still
        outranks all of it: ``labels=True`` on a globe is refused whatever the grid it replaces carried.

        **Which frame draws it, and when.** On a flat map — the default — the lines and their labels go on
        the axes as this call runs. On a ``globe=True`` map they are drawn later, by the projection frame
        (``render()``/``save()``/``show()``), because a globe's grid has to be clipped at the limb with
        everything else. Before #221 the *flat* map had no such pass, so this call computed a grid, stored
        it, and drew nothing at all while the figure described a layer that was drawn.

        Args:
            lon_step: Meridian spacing in degrees; ``None`` (default) means
                :data:`DEFAULT_GRATICULE_STEP` on the call that **creates** the graticule, and the step
                the grid already carries on a *replacing* one.
            lat_step: Parallel spacing in degrees, the same way round.
            spacing: One step for both, for a caller who wants a square grid; it overrides the two
                above, and **warns** when it does, because a call that names all three has had two of
                its own arguments thrown away (review R2-L3). The same **keyword** the web and
                interactive tiers take, so one call draws one grid on every tier that draws a graticule
                at all (#324). The **values** are each engine's own: this tier generates its meridians,
                so any positive step draws, and so does the web tier — while the interactive tier draws
                Natural Earth's pre-cut layers and honours only ``1``, ``5``, ``10``, ``15``, ``20`` and
                ``30``. ``spacing=7.5`` draws here and raises there (review R-L10).
            labels: Whether each line carries its degree — ``"30°E"``, ``"60°N"``, the web tier's own
                spelling. ``None`` (the default) labels the frames that can be labelled, which is every
                flat map and no globe — and on a *replacing* call means whatever the grid already
                carries, so restyling a bare grid leaves it bare; ``True`` asks for them and **raises** on
                a globe, where there is no edge to hang a degree on (see *Raises*); ``False`` draws the
                lines bare. Each label sits on
                its own line, just inside the lower edge of the view for a meridian and the left edge for a
                parallel, and a degree the display CRS cannot place there is **named in a warning** rather
                than dropped in silence.
            name: The caller's own name for the layer, used as its id and its label on the call that
                **creates** it; ``None`` (default) generates one from the kind (#321). A *replacing*
                call cannot rename the layer, so one that names a different name **warns** rather than
                dropping it in silence (review R2-L3).
            visible: Whether the graticule is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327). ``None`` (default)
                leaves the flag as it is: on the call that creates the layer that means drawn, and on a
                *replacing* call it means whatever the caller last chose, so restyling a hidden graticule
                does not put it back on screen (review R-L5).

        Returns:
            This map, so the call chains like every other layer builder on the tier
            (``m.graticule(spacing=30.0).set_global()``). It registers ``graticule-1`` and draws, which
            makes it a builder by every other test here; it used to hand back nothing, which broke the
            chain on the one tier whose `visible`/`name` contract is the most elaborate (L6). The lines
            themselves are reached by id through :meth:`~digitalearth.static.scene.Scene.artist`, which
            **raises** ``KeyError`` for a layer that drew nothing.

        Warns:
            UserWarning: when ``spacing`` is given beside either step, which discards the step; when a
                *replacing* call names a ``name`` the layer does not already carry, which is discarded
                because the id is the one thing a replacement cannot change (review R2-L3); and when a
                degree label cannot be placed, which names it.

        Raises:
            ValueError: for ``labels=True`` on a ``globe=True`` map, which cannot carry them — see
                :meth:`_labels_asked`. Raised before anything is described, so the refusal costs the figure
                nothing.
            Exception: whatever computing the grid raises — a spacing of zero divides by zero in the
                projection — after the description has been put back as it was. A figure must not name a
                layer that was not drawn, and a refused *replacement* must not restyle the graticule the
                map is still drawing (round 2, M1).

        Examples:
            - A flat map's grid is one collection, and every line in view carries its degree:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> m = Map(crs=4326)
                >>> m.set_bounds([-35.0, -5.0, 35.0, 65.0])  # doctest: +ELLIPSIS
                <digitalearth.static.map.Map object at ...>
                >>> _ = m.graticule(spacing=30.0)
                >>> sorted({text.get_text() for text in m.ax.texts})
                ['0°', '30°E', '30°N', '30°W', '60°N']
                >>> m.close()

                ```
            - The lines can be drawn bare:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> m = Map(crs=4326)
                >>> _ = m.graticule(spacing=60.0, labels=False)
                >>> len(m.ax.texts)
                0
                >>> m.close()

                ```
            - A replacing call keeps what it does not name: the grid is re-cut at 60 degrees and stays
              bare, and its parallels keep the step only the first call named:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> m = Map(crs=4326)
                >>> m.set_bounds([-35.0, -5.0, 35.0, 65.0])  # doctest: +ELLIPSIS
                <digitalearth.static.map.Map object at ...>
                >>> _ = m.graticule(lon_step=30.0, lat_step=45.0, labels=False)
                >>> _ = m.graticule(lon_step=60.0)
                >>> props = m._layer_tree.get(m._graticule_id).symbology.props
                >>> (props["lon_step"], props["lat_step"], props["labels"])
                (60.0, 45.0, False)
                >>> len(m.ax.texts)
                0
                >>> m.close()

                ```
            - A globe draws the grid unlabelled, and **refuses** `labels=True` rather than dropping them
              quietly — the default `labels=None` is what lets one line serve both frames:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> globe = Map(crs=4326, globe=True)
                >>> _ = globe.graticule(spacing=30.0)
                >>> len(globe.ax.texts)
                0
                >>> globe.graticule(spacing=30.0, labels=True)  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                ValueError: graticule(labels=True) cannot place degree labels on a globe frame...
                >>> globe.close()

                ```
            - A degree the frame cannot place is **named in a warning**, not silently missing: a polar
              projection framed on a square puts most meridians' anchors outside the view:
                ```python
                >>> import warnings
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> m = Map(crs=3413)
                >>> _ = m.set_bounds([-3.0e6, -3.0e6, 3.0e6, 3.0e6])
                >>> with warnings.catch_warnings(record=True) as caught:
                ...     warnings.simplefilter("always")
                ...     _ = m.graticule(spacing=30.0)
                >>> message = str(caught[0].message)
                >>> message.split(":")[0]
                'graticule() could not place 8 degree label(s)'
                >>> "150°W" in message, "120°E" in message
                (True, True)
                >>> len(m.ax.texts)
                5
                >>> m.close()

                ```
        """
        # Read before anything is described: a replacing call keeps every option it does not name, and
        # the layer it replaces is the only record of what those options were.
        carried = _GraticuleSteps.held_by(self._graticule_held())
        steps = _GraticuleSteps.asked(
            lon_step, lat_step, spacing, self._labels_asked(labels, carried), carried
        )
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
        return self

    def _labels_asked(
        self, labels: Optional[bool], carried: Optional["_GraticuleSteps"] = None
    ) -> bool:
        """Settle whether this graticule is labelled, refusing the frame that cannot carry labels.

        The one of :meth:`graticule`'s arguments that depends on the **frame** rather than on what the
        caller wrote, which is why it is settled here and not in :meth:`_GraticuleSteps.asked`.

        **Why a globe is refused rather than quietly left bare.** A globe's grid is drawn by cleopatra's
        ``apply_projection_frame``, which turns the axes off (``set_axis_off``) and gives a figure with no
        edge to hang a degree on: every meridian converges at the limb, where a label would sit on top of
        its neighbours, and the near side has no outer boundary a reader reads coordinates off. Placing a
        degree on a projected frame is generic matplotlib artistry and cleopatra already owns both the
        frame and the degree formatters it uses on its flat path, so that placement is the upstream ask
        recorded in #221 — not something to reimplement here. Until it lands, a call that asks for labels
        on a globe is told so by name; drawing the lines and dropping the labels in silence is the shape of
        defect this row exists to end.

        Args:
            labels: What the caller wrote — ``True``, ``False``, or ``None`` for "wherever this frame can".
            carried: What the graticule being *replaced* carries, or ``None`` on the call that creates it.
                A call that does not name ``labels`` is asking for a different spacing, not for the
                degrees of a deliberately bare grid to come back — the rule ``visible`` already follows
                (round 1, M7).

        Returns:
            Whether the lines are labelled: always ``False`` on a globe; otherwise what the caller asked
            for, else what the grid being replaced carried, else ``True``.

        Raises:
            ValueError: for ``labels=True`` on a ``globe=True`` map — whatever the replaced grid
                carried, because the refusal is about the frame and not about the layer.
        """
        if not self.globe:
            if labels is not None:
                return bool(labels)
            return True if carried is None else bool(carried.labels)
        if labels:
            raise ValueError(
                "graticule(labels=True) cannot place degree labels on a globe frame: the projection "
                "frame turns the axes off and its meridians converge at the limb, so there is no edge to "
                "place a degree on. Placing one there is cleopatra's to add (the upstream ask in "
                "Digital-Earth#221). Draw the globe's grid unlabelled — graticule() with no labels= does "
                "— or label a flat projected map instead (Map(crs=...) without globe=True)."
            )
        return False

    def _graticule_held(self) -> Optional[LayerSpec]:
        """Return the graticule layer a next call would replace, or ``None`` for a creating call.

        Read by :meth:`graticule` for the options a replacing call inherits and by
        :meth:`_describe_graticule` for the layer it edits, so the two cannot disagree about which call
        this is.

        Returns:
            The described layer :attr:`_graticule_id` points at, or ``None`` when the map has no graticule
            — including when the id outlived its layer: ``_reset_layers`` clears the tree between
            animation frames while the lines themselves survive, so the membership test is what keeps the
            remembered id from raising here.
        """
        pointer = self._graticule_id
        if pointer is None or pointer not in self._layer_tree.ids:
            return None
        return self._layer_tree.get(pointer)

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
        was = self._graticule_held()
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

    def set_global(self) -> Self:
        """Set the axes extent to the full projection domain (the whole globe/world).

        Returns:
            This map, so the call chains like :meth:`set_bounds`, which it frames through — it is that
            method with the whole projection domain for an argument.

        Examples:
            - The frame is the projection's own, and the call hands the map on:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth import Map
                >>> m = Map(crs=4326)
                >>> m.set_global() is m
                True
                >>> [float(value) for value in m.ax.get_xlim()]
                [-180.0, 180.0]
                >>> m.close()

                ```
        """
        _, xlim, ylim = self._frame()
        # `_frame` answers in matplotlib's pairs; `set_bounds` reads (west, south, east, north), so the
        # four values are re-ordered here rather than handed over in the order they arrived.
        return self.set_bounds([xlim[0], ylim[0], xlim[1], ylim[1]])

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
        """Give the described graticule layer the artists the projection frame drew for it, and its flag.

        Args:
            artists: The line artists the frame added, in the order it added them.

        Note:
            This writes into the renderer's record of what it drew, which no public method reaches — every
            other layer's artists are known to its drawer, and a graticule's are not. A
            ``Renderer.attach_artists`` would be the tidier home for it.

            **The flag is applied here because this is the only place it can be.** ``Renderer.draw_layer``
            applies a layer's ``visible`` the moment its drawer returns, which for every other layer is the
            moment its artists exist. A graticule's do not: they are drawn by ``apply_projection_frame``,
            after the funnel has run and found nothing on the axes to toggle — so a grid built
            ``visible=False`` was described hidden and drawn visible (#333). Adopting the artists and
            applying the flag are one step for that reason, and the flag is read from the tree, exactly as
            the funnel reads it, so the two cannot answer differently.
        """
        layer_id = self._graticule_id
        if layer_id is None or not artists:
            return
        drawn = self._renderer.drawn.get(layer_id)
        if drawn is None:
            return
        self._renderer._drawn[layer_id] = with_fields(drawn, artists=artists)
        # Guarded on membership rather than caught: `_reset_layers` clears the tree between animation frames
        # while `_graticule_id` survives it, so the remembered id can outlive its layer — and `is_visible`
        # answers a missing id with a `KeyError`, which is not a question about visibility.
        if layer_id in self._layer_tree.ids and not self._layer_tree.is_visible(
            layer_id
        ):
            self._renderer.set_visible(layer_id, False)

    def render(self) -> None:
        """Apply the projection frame if this is a globe map (idempotent). Call before showing/saving."""
        self._apply_frame()

    def save(self, path: Union[str, "os.PathLike[str]"], **kwargs: Any) -> Path:
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
