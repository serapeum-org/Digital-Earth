"""The view as a value: `Viewport` for a flat map, `Camera` for a 3-D scene.

On every tier the view is loose state on the facade. The static `Map` keeps ``crs``, ``domain`` and ``globe`` as three
attributes, and animates a rotation by assigning ``self.crs`` per frame and restoring it afterwards
(`static/maps/animation.py`). The 3-D tier has no camera value of its own: a view is framed through a `cpos=`
keyword forwarded to `show()`, or swept along `orbit`'s path, and the tier has no API of its own to read the camera
back (#204). None of it is one value that can be compared or stored — which is what export (#187, #194), camera
read-back (#204) and a figure description all need.

These are values: frozen, serialisable, renderer-free. A change of view is a new value, never a mutation of the old
one, which is what makes a rotation a sequence of views rather than a loop that edits one in place.

`Camera` keeps ``vertical_exaggeration`` because the 3-D tier already treats exaggeration as view state — one value
per scene, applied through the plotter's scale — rather than as something baked into mesh coordinates.
"""

from dataclasses import dataclass, replace
from math import atan2, cos, degrees, hypot, radians, sin, sqrt
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from digitalearth.base.spec._serial import (
    as_list,
    crs_to_json,
    finite_number,
    refuse_unknown,
    require,
)
from digitalearth.base.spec.bounds import Bounds, _same_crs

__all__ = ["Camera", "DEFAULT_VIEW_ANGLE", "Viewport"]

#: The camera's default vertical field of view, in degrees. It is VTK's own default, so a `Camera` built without one
#: frames a scene the way the 3-D tier already does.
DEFAULT_VIEW_ANGLE = 30.0

#: Below this, the view-up vector is treated as parallel to the line of sight. It compares the sine of the angle
#: between the two unit vectors, so it is independent of the scene's scale.
_PARALLEL_TOLERANCE = 1e-9

Vector3 = Tuple[float, float, float]

#: The owner named in `Camera.look_at`'s validation messages.
_LOOK_AT = "Camera.look_at"


def _vector(owner: str, name: str, value: Any) -> Vector3:
    """Return `value` as three finite floats.

    Args:
        owner: The type being built, for the message.
        name: The field.
        value: The candidate — any iterable of three numbers other than a string: a tuple, a list, a numpy array.

    Returns:
        The vector as a tuple of floats.

    Raises:
        ValueError: for a string or a non-iterable, an iterable that does not hold exactly three items, or an item
            that is not a finite number (a boolean included).
    """
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise ValueError(f"{owner} needs {name} as three numbers; got {value!r}")
    items = list(value)
    if len(items) != 3:
        raise ValueError(
            f"{owner} needs {name} as three numbers; got {len(items)}: {value!r}"
        )
    x, y, z = (
        finite_number(owner, f"{name}[{index}]", item)
        for index, item in enumerate(items)
    )
    return (x, y, z)


@dataclass(frozen=True)
class Viewport:
    """What a flat map shows: the CRS it is drawn in, and optionally the region.

    Attributes:
        crs: The display CRS every layer is drawn in — an EPSG integer or anything pyramids reads. Defaults to Web
            Mercator, as the static `Map` does.
        bounds: The region shown, in `crs`. ``None`` means the view is not framed yet and the renderer chooses.
        domain: A named region (``"europe"``) or a ``(west, south, east, north)`` box in degrees — the static
            tier's ``domain=`` — or ``None``.
        globe: Whether the map is drawn on a globe frame rather than a flat projection.

    Raises:
        ValueError: for a `None` or boolean CRS, `bounds` that is not a `Bounds` or is in a different CRS, a
            `domain` that is neither a non-empty name nor four finite numbers, or a non-boolean `globe`.

    Examples:
        - A map framed on a region, in the CRS it is drawn in:
            ```python
            >>> from digitalearth.base.spec import Bounds, Viewport
            >>> view = Viewport(4326, bounds=Bounds(-10.0, 35.0, 30.0, 60.0, crs=4326))
            >>> view.crs, view.bounds.as_bbox()
            (4326, [-10.0, 35.0, 30.0, 60.0])

            ```
        - A named domain, as the static tier spells it:
            ```python
            >>> from digitalearth.base.spec import Viewport
            >>> Viewport(domain="europe").domain
            'europe'

            ```
        - A region in another CRS is refused rather than drawn in the wrong place; `framed` reprojects it:
            ```python
            >>> from digitalearth.base.spec import Bounds, Viewport
            >>> Viewport(3857, bounds=Bounds(0.0, 0.0, 1.0, 1.0, crs=4326))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: Viewport bounds are in 4326 but the view is drawn in 3857; use Viewport.framed(bounds), ...

            ```
    """

    crs: Any = 3857
    bounds: Optional[Bounds] = None
    domain: Optional[Union[str, Tuple[float, float, float, float]]] = None
    globe: bool = False

    def __post_init__(self) -> None:
        """Refuse a view nothing could be drawn in.

        Raises:
            ValueError: as described on the class.
        """
        if self.crs is None or isinstance(self.crs, bool):
            raise ValueError(f"Viewport needs a display CRS; got {self.crs!r}")
        self._check_bounds()
        object.__setattr__(self, "domain", self._checked_domain(self.domain))
        if not isinstance(self.globe, bool):
            raise ValueError(
                f"Viewport globe must be True or False; got {self.globe!r}"
            )

    def _check_bounds(self) -> None:
        """Refuse bounds that are not a `Bounds`, or that are in a different CRS from the view.

        Raises:
            ValueError: for bounds that are not a `Bounds`; or, naming both CRSs and `framed`, which reprojects, for
                bounds in a CRS pyramids does not judge the same as the view's.
        """
        if self.bounds is None:
            return
        if not isinstance(self.bounds, Bounds):
            raise ValueError(
                f"Viewport bounds must be a Bounds; got {type(self.bounds).__name__}"
            )
        if not _same_crs(self.bounds.crs, self.crs):
            # A rectangle in one CRS read as another draws the wrong place, silently. `framed` reprojects; the
            # constructor refuses, so a view never holds a region its own CRS cannot place.
            raise ValueError(
                f"Viewport bounds are in {self.bounds.crs!r} but the view is drawn in {self.crs!r}; "
                "use Viewport.framed(bounds), which reprojects"
            )

    @staticmethod
    def _checked_domain(
        domain: Any,
    ) -> Optional[Union[str, Tuple[float, float, float, float]]]:
        """Return the domain as a name or a tuple of four floats, or ``None``.

        Args:
            domain: What the constructor was given.

        Returns:
            The name unchanged, the four edges as floats, or ``None``.

        Raises:
            ValueError: for an empty or whitespace-only name, a non-iterable, the wrong number of edges, or an edge
                that is not a finite number (a boolean included).
        """
        if domain is None:
            return None
        if isinstance(domain, str):
            if not domain.strip():
                raise ValueError(
                    "Viewport domain must be a region name or four numbers; got an empty name"
                )
            return domain
        edges = list(domain) if hasattr(domain, "__iter__") else []
        if len(edges) != 4:
            raise ValueError(
                f"Viewport domain must be a region name or (west, south, east, north); got {domain!r}"
            )
        west, south, east, north = (
            finite_number("Viewport", "domain", edge) for edge in edges
        )
        return west, south, east, north

    def framed(self, bounds: Bounds) -> "Viewport":
        """Return this view framed on `bounds`, reprojected into the view's CRS if they are in another.

        Args:
            bounds: The region to show, in any CRS.

        Returns:
            A new view. This one is unchanged — a change of view is a new value, never an edit of the old one.

        Raises:
            ValueError: if `bounds` is not a `Bounds`, or cannot be reprojected into the view's CRS.

        Examples:
            - Frame a view without changing the one it came from:
                ```python
                >>> from digitalearth.base.spec import Bounds, Viewport
                >>> view = Viewport(4326)
                >>> framed = view.framed(Bounds(0.0, 0.0, 10.0, 5.0, crs=4326))
                >>> framed.bounds.as_bbox(), view.bounds
                ([0.0, 0.0, 10.0, 5.0], None)

                ```
            - A region in another CRS is reprojected into the view's:
                ```python
                >>> from digitalearth.base.spec import Bounds, Viewport
                >>> framed = Viewport(3857).framed(Bounds(0.0, 0.0, 1.0, 1.0, crs=4326))
                >>> framed.bounds.crs, [round(edge) for edge in framed.bounds.as_bbox()]
                (3857, [0, 0, 111319, 111325])

                ```
        """
        if not isinstance(bounds, Bounds):
            raise ValueError(
                f"Viewport.framed needs a Bounds; got {type(bounds).__name__}"
            )
        return replace(self, bounds=bounds.to_crs(self.crs))

    def needs_reproject(self, data: Any) -> bool:
        """Whether `data` has to be reprojected to be drawn in this view.

        Args:
            data: A pyramids object exposing ``.epsg``.

        Returns:
            ``False`` only when the view's CRS is an EPSG integer equal to ``data.epsg`` — the rule every tier
            shares, from :func:`digitalearth.base.display.needs_reproject`.

        Examples:
            - Data already in the view's CRS needs no warp:
                ```python
                >>> from types import SimpleNamespace
                >>> from digitalearth.base.spec import Viewport
                >>> Viewport(4326).needs_reproject(SimpleNamespace(epsg=4326))
                False

                ```
            - A CRS spelled as a string always warps, even when it names the data's own system:
                ```python
                >>> from types import SimpleNamespace
                >>> from digitalearth.base.spec import Viewport
                >>> Viewport("EPSG:4326").needs_reproject(SimpleNamespace(epsg=4326))
                True

                ```
        """
        # Imported here: base.display reaches base.sources, which imports this package, so importing it at module
        # level would be circular.
        from digitalearth.base.display import needs_reproject

        return needs_reproject(data, self.crs)

    def to_dict(self) -> Dict[str, Any]:
        """Return the plain-dict form a figure stores.

        Returns:
            `crs`, plus `bounds` and `domain` when set and `globe` only when it is `True`. A CRS object is written
            as `"EPSG:<code>"` or WKT; an EPSG integer or a string is written as given, unchecked.

        Raises:
            TypeError: if the CRS is an object pyramids cannot read as a CRS.

        Examples:
            - A default view is its CRS:
                ```python
                >>> from digitalearth.base.spec import Viewport
                >>> Viewport().to_dict()
                {'crs': 3857}

                ```
            - A framed globe view writes its region as a dict of named edges:
                ```python
                >>> from digitalearth.base.spec import Bounds, Viewport
                >>> stored = Viewport(4326, bounds=Bounds(0.0, 0.0, 1.0, 1.0, crs=4326), globe=True).to_dict()
                >>> sorted(stored), stored["bounds"]
                (['bounds', 'crs', 'globe'], {'xmin': 0.0, 'ymin': 0.0, 'xmax': 1.0, 'ymax': 1.0, 'crs': 4326})

                ```
        """
        out: Dict[str, Any] = {"crs": crs_to_json(self.crs, "Viewport.crs")}
        if self.bounds is not None:
            out["bounds"] = self.bounds.to_dict()
        if self.domain is not None:
            out["domain"] = (
                self.domain if isinstance(self.domain, str) else list(self.domain)
            )
        if self.globe:
            out["globe"] = True
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Viewport":
        """Rebuild a view from its dict form.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The view, validated as the constructor validates it.

        Raises:
            TypeError: if `data`, or its `bounds`, is not a mapping.
            ValueError: for a missing CRS, an unknown key, or a view the constructor refuses.

        Examples:
            - A stored globe view reads back:
                ```python
                >>> from digitalearth.base.spec import Viewport
                >>> view = Viewport.from_dict({"crs": "+proj=ortho +lat_0=30 +lon_0=10", "globe": True})
                >>> view.globe, view.crs
                (True, '+proj=ortho +lat_0=30 +lon_0=10')

                ```
            - A stored domain box comes back as a tuple of floats:
                ```python
                >>> from digitalearth.base.spec import Viewport
                >>> Viewport.from_dict({"crs": 4326, "domain": [-10, 35, 30, 60]}).domain
                (-10.0, 35.0, 30.0, 60.0)

                ```
            - The CRS cannot be left out:
                ```python
                >>> from digitalearth.base.spec import Viewport
                >>> Viewport.from_dict({"globe": True})
                Traceback (most recent call last):
                    ...
                ValueError: Viewport.from_dict needs 'crs'; got keys ['globe']

                ```
        """
        refuse_unknown("Viewport", data, ("crs", "bounds", "domain", "globe"))
        bounds = data.get("bounds")
        domain = data.get("domain")
        return cls(
            crs=require("Viewport", data, "crs"),
            bounds=None if bounds is None else Bounds.from_dict(bounds),
            domain=tuple(domain) if isinstance(domain, list) else domain,
            globe=data.get("globe", False),
        )


@dataclass(frozen=True)
class Camera:
    """Where a 3-D scene is looked at from.

    Attributes:
        position: The camera's location, in the scene's coordinates.
        focal_point: The point it looks at.
        view_up: Which way is up on the screen. It may not be zero or parallel to the line of sight.
        view_angle: The vertical field of view in degrees, strictly between 0 and 180. Ignored by a parallel
            projection, which has no perspective.
        parallel: Whether the projection is parallel (orthographic) rather than perspective — for a figure whose
            distances should be measurable.
        vertical_exaggeration: The factor applied to the z axis, strictly positive. It lives on the view, as the 3-D
            tier already keeps it, rather than in the mesh coordinates.

    Raises:
        ValueError: for a vector that is not three finite numbers, a camera placed at its own focal point, a zero or
            parallel `view_up`, a view angle outside ``(0, 180)``, a non-positive exaggeration, or a non-boolean
            `parallel`.

    Examples:
        - A camera south-west of a scene, 30 degrees up:
            ```python
            >>> from digitalearth.base.spec import Camera
            >>> camera = Camera.look_at((0.0, 0.0, 0.0), azimuth=225.0, elevation=30.0, distance=100.0)
            >>> round(camera.azimuth, 6), round(camera.elevation, 6), round(camera.distance, 6)
            (225.0, 30.0, 100.0)

            ```
        - Fields left out look at the origin with `+z` up and VTK's 30-degree view angle; vectors become floats:
            ```python
            >>> from digitalearth.base.spec import Camera
            >>> camera = Camera((0, -10, 5))
            >>> camera.position, camera.focal_point, camera.view_up, camera.view_angle
            ((0.0, -10.0, 5.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 30.0)

            ```
        - Looking straight down with the default vertical `view_up` leaves "up" undefined:
            ```python
            >>> from digitalearth.base.spec import Camera
            >>> Camera((0.0, 0.0, 10.0))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: Camera view_up (0.0, 0.0, 1.0) is parallel to the line of sight, so 'up' on the screen is ...

            ```
    """

    position: Vector3
    focal_point: Vector3 = (0.0, 0.0, 0.0)
    view_up: Vector3 = (0.0, 0.0, 1.0)
    view_angle: float = DEFAULT_VIEW_ANGLE
    parallel: bool = False
    vertical_exaggeration: float = 1.0

    def __post_init__(self) -> None:
        """Refuse a camera that could not frame anything.

        Raises:
            ValueError: as described on the class.
        """
        position = _vector("Camera", "position", self.position)
        focal = _vector("Camera", "focal_point", self.focal_point)
        up = _vector("Camera", "view_up", self.view_up)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "focal_point", focal)
        object.__setattr__(self, "view_up", up)
        sight = tuple(p - f for p, f in zip(position, focal))
        sight_length = sqrt(sum(component * component for component in sight))
        if sight_length <= 0.0:
            raise ValueError(
                f"Camera position and focal_point are the same point {position}; it looks at nothing"
            )
        up_length = sqrt(sum(component * component for component in up))
        if up_length <= 0.0:
            raise ValueError(
                "Camera view_up is the zero vector, which names no direction"
            )
        cross = (
            sight[1] * up[2] - sight[2] * up[1],
            sight[2] * up[0] - sight[0] * up[2],
            sight[0] * up[1] - sight[1] * up[0],
        )
        sine = sqrt(sum(component * component for component in cross)) / (
            sight_length * up_length
        )
        if sine < _PARALLEL_TOLERANCE:
            raise ValueError(
                f"Camera view_up {up} is parallel to the line of sight, so 'up' on the screen is undefined. "
                "Looking straight down or up, pass a horizontal view_up such as (0, 1, 0)"
            )
        angle = finite_number("Camera", "view_angle", self.view_angle)
        if not 0.0 < angle < 180.0:
            raise ValueError(
                f"Camera view_angle must be strictly between 0 and 180 degrees; got {angle}"
            )
        object.__setattr__(self, "view_angle", angle)
        factor = finite_number(
            "Camera", "vertical_exaggeration", self.vertical_exaggeration
        )
        if factor <= 0.0:
            raise ValueError(
                f"Camera vertical_exaggeration must be positive; got {factor}"
            )
        object.__setattr__(self, "vertical_exaggeration", factor)
        if not isinstance(self.parallel, bool):
            raise ValueError(
                f"Camera parallel must be True or False; got {self.parallel!r}"
            )

    @classmethod
    def look_at(
        cls,
        focal_point: Sequence[float],
        *,
        azimuth: float,
        elevation: float,
        distance: float,
        view_up: Sequence[float] = (0.0, 0.0, 1.0),
        view_angle: float = DEFAULT_VIEW_ANGLE,
        parallel: bool = False,
        vertical_exaggeration: float = 1.0,
    ) -> "Camera":
        """Place a camera by the direction it looks from, in the terms a map reader uses.

        Args:
            focal_point: The point to look at.
            azimuth: The compass bearing, in degrees, **from the focal point to the camera** — clockwise from north,
                with north along ``+y`` and east along ``+x``. ``225`` puts the camera to the south-west, looking
                north-east.
            elevation: Degrees above the horizontal. ``90`` looks straight down, which needs a horizontal
                `view_up`.
            distance: How far the camera is from the focal point; strictly positive.
            view_up: Which way is up on the screen.
            view_angle: The vertical field of view, in degrees.
            parallel: Whether the projection is parallel rather than perspective.
            vertical_exaggeration: The z-axis factor.

        Returns:
            The camera.

        Raises:
            ValueError: for a non-positive distance, an azimuth, elevation or distance that is not a finite number,
                or any value the constructor refuses — including looking straight down with the default vertical
                `view_up`.

        Examples:
            - Looking at a point from due south, level with it:
                ```python
                >>> from digitalearth.base.spec import Camera
                >>> camera = Camera.look_at((10.0, 20.0, 0.0), azimuth=180.0, elevation=0.0, distance=5.0)
                >>> tuple(round(value, 6) for value in camera.position)
                (10.0, 15.0, 0.0)

                ```
            - Straight down needs a horizontal up vector:
                ```python
                >>> from digitalearth.base.spec import Camera
                >>> top = Camera.look_at((0, 0, 0), azimuth=0, elevation=90, distance=10, view_up=(0, 1, 0))
                >>> tuple(round(value, 6) for value in top.position)
                (0.0, 0.0, 10.0)

                ```
        """
        span = finite_number(_LOOK_AT, "distance", distance)
        if span <= 0.0:
            raise ValueError(f"Camera.look_at needs a positive distance; got {span}")
        bearing = radians(finite_number(_LOOK_AT, "azimuth", azimuth))
        tilt = radians(finite_number(_LOOK_AT, "elevation", elevation))
        focal = _vector(_LOOK_AT, "focal_point", focal_point)
        offset = (
            span * sin(bearing) * cos(tilt),
            span * cos(bearing) * cos(tilt),
            span * sin(tilt),
        )
        return cls(
            position=(focal[0] + offset[0], focal[1] + offset[1], focal[2] + offset[2]),
            focal_point=focal,
            view_up=_vector(_LOOK_AT, "view_up", view_up),
            view_angle=view_angle,
            parallel=parallel,
            vertical_exaggeration=vertical_exaggeration,
        )

    @property
    def distance(self) -> float:
        """How far the camera is from its focal point.

        Returns:
            The Euclidean distance, in scene units.

        Examples:
            - A 3-4-5 triangle:
                ```python
                >>> from digitalearth.base.spec import Camera
                >>> Camera((3.0, 4.0, 0.0)).distance
                5.0

                ```
        """
        return sqrt(sum((p - f) ** 2 for p, f in zip(self.position, self.focal_point)))

    @property
    def azimuth(self) -> float:
        """The compass bearing from the focal point to the camera, in degrees.

        Returns:
            A bearing in `[0, 360)`, clockwise from north (`+y`). A camera directly above or below its focal
            point has no horizontal direction, so its bearing means nothing: it comes out as `0`, or as `180` when
            the y offset is a negative zero, because `atan2` keeps the sign of zero.

        Examples:
            - A camera due east of its focal point:
                ```python
                >>> from digitalearth.base.spec import Camera
                >>> Camera((5.0, 0.0, 0.0)).azimuth
                90.0

                ```
            - Bearings run clockwise, so due west is 270 rather than -90:
                ```python
                >>> from digitalearth.base.spec import Camera
                >>> Camera((-5.0, 0.0, 0.0)).azimuth
                270.0

                ```
        """
        dx = self.position[0] - self.focal_point[0]
        dy = self.position[1] - self.focal_point[1]
        return degrees(atan2(dx, dy)) % 360.0

    @property
    def elevation(self) -> float:
        """The angle of the camera above the horizontal, seen from the focal point.

        Returns:
            Degrees in ``[-90, 90]``: positive above, negative below.

        Examples:
            - A camera level with its focal point:
                ```python
                >>> from digitalearth.base.spec import Camera
                >>> Camera((0.0, 5.0, 0.0)).elevation
                0.0

                ```
            - A camera straight above, which needs a horizontal up vector:
                ```python
                >>> from digitalearth.base.spec import Camera
                >>> Camera((0.0, 0.0, 5.0), view_up=(0.0, 1.0, 0.0)).elevation
                90.0

                ```
        """
        dx = self.position[0] - self.focal_point[0]
        dy = self.position[1] - self.focal_point[1]
        dz = self.position[2] - self.focal_point[2]
        return degrees(atan2(dz, hypot(dx, dy)))

    def to_dict(self) -> Dict[str, Any]:
        """Return the plain-dict form a figure stores.

        Returns:
            Every field. A camera is small, and a stored view is only reproducible if it records the settings
            that were in force rather than relying on today's defaults.

        Examples:
            - Every setting is written, defaults included:
                ```python
                >>> from digitalearth.base.spec import Camera
                >>> stored = Camera((0.0, -10.0, 5.0), parallel=True).to_dict()
                >>> stored["position"], stored["parallel"], stored["view_angle"]
                ([0.0, -10.0, 5.0], True, 30.0)

                ```
            - The dict survives a JSON round trip and rebuilds an equal camera:
                ```python
                >>> import json
                >>> from digitalearth.base.spec import Camera
                >>> text = json.dumps(Camera((0.0, -10.0, 5.0), vertical_exaggeration=3.0).to_dict())
                >>> Camera.from_dict(json.loads(text)).vertical_exaggeration
                3.0

                ```
        """
        return {
            "position": list(self.position),
            "focal_point": list(self.focal_point),
            "view_up": list(self.view_up),
            "view_angle": self.view_angle,
            "parallel": self.parallel,
            "vertical_exaggeration": self.vertical_exaggeration,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Camera":
        """Rebuild a camera from its dict form.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The camera, validated as the constructor validates it.

        Raises:
            TypeError: if `data` is not a mapping, or `position`, `focal_point` or `view_up` is not a list,
                naming the field.
            ValueError: for a missing position, an unknown key, or a camera the constructor refuses.

        Examples:
            - A stored camera reads back with its exaggeration:
                ```python
                >>> from digitalearth.base.spec import Camera
                >>> Camera.from_dict({"position": [0, -10, 5], "vertical_exaggeration": 3}).vertical_exaggeration
                3.0

                ```
            - Only the position is required; the rest takes the constructor's defaults:
                ```python
                >>> from digitalearth.base.spec import Camera
                >>> camera = Camera.from_dict({"position": [0, -10, 5]})
                >>> camera.focal_point, camera.parallel
                ((0.0, 0.0, 0.0), False)

                ```
            - A missing position is named:
                ```python
                >>> from digitalearth.base.spec import Camera
                >>> Camera.from_dict({"view_angle": 45})
                Traceback (most recent call last):
                    ...
                ValueError: Camera.from_dict needs 'position'; got keys ['view_angle']

                ```
        """
        refuse_unknown(
            "Camera",
            data,
            (
                "position",
                "focal_point",
                "view_up",
                "view_angle",
                "parallel",
                "vertical_exaggeration",
            ),
        )
        return cls(
            position=as_list("Camera", "position", require("Camera", data, "position")),
            focal_point=as_list(
                "Camera", "focal_point", data.get("focal_point", (0.0, 0.0, 0.0))
            ),
            view_up=as_list("Camera", "view_up", data.get("view_up", (0.0, 0.0, 1.0))),
            view_angle=data.get("view_angle", DEFAULT_VIEW_ANGLE),
            parallel=data.get("parallel", False),
            vertical_exaggeration=data.get("vertical_exaggeration", 1.0),
        )
