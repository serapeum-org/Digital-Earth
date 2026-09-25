"""The named views a 3-D scene can be looked from, as a table a caller can extend (order 26, TD-2).

The tier could already say *exactly* where to stand: :class:`~digitalearth.base.spec.Camera` carries a position,
a focal point, an up vector, a field of view, the projection and the vertical exaggeration, and the scene reads
and writes the whole record. What it had no way to say was **"from the top"** — the one-word request that is
most of what a figure needs, and the only part of the view API that was missing.

So this is a naming layer and nothing more. Each row points at one of PyVista's own ``view_*`` methods, which
already know how to frame the scene's bounds from a direction; none of the trigonometry is re-done here. That is
deliberate: a preset computing its own camera would drift from what the renderer does when a caller calls
``plotter.view_xy()`` directly, and there would then be two answers to "what is the top view".

**A view is data, not a subclass.** The table is a mapping of name to :class:`NamedView`, and
:func:`register_view` puts a caller's own name in it — so "north-east at 20 degrees, the way our report figures
are drawn" is a row somebody adds, not a method somebody has to get merged. What holds the table honest is
``tests/three_d/test_views3d.py``: every row's method must be a real ``view_*`` on :class:`pyvista.Plotter`,
every row's options must bind to that method's signature, and every shipped row is checked against the camera
the plotter really ends up with — not against the record the scene keeps of it.

**No PyVista at import.** The rows name their methods as strings, so reading the table costs no engine import
(``tests/three_d/test_lazy_engine.py``). PyVista is reached when a view is registered — where checking it is
worth the import, because the error then lands on the line that wrote the bad row — and when one is applied.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "NAMED_VIEWS",
    "NamedView",
    "forget_view",
    "named_view",
    "register_view",
    "view_names",
]

#: The prefix every PyVista viewpoint method carries. A registered view has to name one of those and not, say,
#: ``add_mesh``: a method that draws rather than aims would "work" and then leave the camera where it was.
_VIEW_PREFIX: str = "view_"


@dataclass(frozen=True)
class NamedView:
    """One row of the view table: a name, the PyVista method behind it, and what it shows.

    Attributes:
        name: The name a caller passes to :meth:`digitalearth.three_d.Scene3D.view`.
        method: The :class:`pyvista.Plotter` method that aims the camera — ``"view_xy"``, ``"view_isometric"``,
            …. Held as a string so the table can be read without importing PyVista.
        describes: What the viewer sees, in a phrase. A row nobody can read is a row nobody uses, and this is
            what a listing or a refusal prints.
        options: The keywords the method is called with — ``{"negative": True}`` for the opposite side of an
            axis. Read-only, and checked against the method's signature when the row is registered.

    Examples:
        - A row says which PyVista call it stands for:
            ```python
            >>> from digitalearth.three_d.views import named_view
            >>> top = named_view("top")
            >>> top.method, dict(top.options)
            ('view_xy', {})

            ```
        - The opposite side of the same axis is the same method with one option:
            ```python
            >>> from digitalearth.three_d.views import named_view
            >>> named_view("bottom").method, dict(named_view("bottom").options)
            ('view_xy', {'negative': True})

            ```
    """

    name: str
    method: str
    describes: str
    options: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def apply(self, plotter: Any) -> None:
        """Aim `plotter`'s camera the way this view says.

        Args:
            plotter: The :class:`pyvista.Plotter` to aim. The scene reads the camera back off it afterwards,
                which is what turns the preset into a `Camera` the tier can store and set again.

        Raises:
            AttributeError: if the plotter has no such method — which a registered row cannot be, but a
                stand-in plotter implementing only half the view API can.
        """
        getattr(plotter, self.method)(**dict(self.options))


def _shipped() -> tuple[NamedView, ...]:
    """Build the views the tier ships.

    The directions are PyVista 0.48.4's, measured rather than assumed: ``view_xy`` stands on ``+z`` looking
    down, ``view_xz`` on ``-y``, ``view_yz`` on ``+x``, and each takes ``negative=True`` for the far side. In a
    scene drawn with x east, y north and z up — which is how every builder in this tier lays its data out —
    those read as top, front (from the south) and right (from the east).

    Returns:
        The shipped rows, in the order they are listed.
    """
    return (
        NamedView("top", "view_xy", "straight down, north up — the plan view"),
        NamedView(
            "bottom",
            "view_xy",
            "straight up from underneath, for the underside of a surface",
            MappingProxyType({"negative": True}),
        ),
        NamedView("front", "view_xz", "from the south, looking north along the ground"),
        NamedView(
            "back",
            "view_xz",
            "from the north, looking south along the ground",
            MappingProxyType({"negative": True}),
        ),
        NamedView("right", "view_yz", "from the east, looking west along the ground"),
        NamedView(
            "left",
            "view_yz",
            "from the west, looking east along the ground",
            MappingProxyType({"negative": True}),
        ),
        NamedView(
            "isometric",
            "view_isometric",
            "the three-quarter view relief reads best from, and the scene's own default",
        ),
    )


#: The live table, keyed by name. Private because :func:`register_view` is the way in: it is the only path that
#: checks a row against PyVista before it can be used.
_VIEWS: dict[str, NamedView] = {view.name: view for view in _shipped()}

#: Every named view, by name — a read-only view of the live table, so a row registered later shows up here too.
NAMED_VIEWS: Mapping[str, NamedView] = MappingProxyType(_VIEWS)


def view_names() -> tuple[str, ...]:
    """Return every registered view name, in alphabetical order.

    Returns:
        The names, sorted — which is also the order a refusal lists them in.

    Examples:
        - The shipped set:
            ```python
            >>> from digitalearth.three_d.views import view_names
            >>> view_names()
            ('back', 'bottom', 'front', 'isometric', 'left', 'right', 'top')

            ```
    """
    return tuple(sorted(_VIEWS))


def named_view(name: str) -> NamedView:
    """Return the view registered under `name`.

    Args:
        name: The view's name, as :func:`view_names` lists it.

    Returns:
        The row.

    Raises:
        KeyError: for a name nothing registered. The message lists the names that do exist, because the way out
            of this error is one of them — the same shape :func:`~digitalearth.base.contract_clauses.clause`
            and :func:`~digitalearth.three_d.renderer.drawer_for` use.

    Examples:
        - A name nobody registered says which ones are there:
            ```python
            >>> from digitalearth.three_d.views import named_view
            >>> named_view("birdseye")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            KeyError: "'birdseye' is not a named view; the scene knows ['back', ...

            ```
    """
    try:
        return _VIEWS[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a named view; the scene knows {list(view_names())}. "
            "Register your own with digitalearth.three_d.views.register_view()"
        ) from None


def register_view(
    name: str,
    *,
    method: str,
    describes: str,
    replace: bool = False,
    **options: Any,
) -> NamedView:
    """Put a view of your own in the table, and check it against PyVista before anything uses it.

    The checks are the point. A row is only ever read when somebody asks for that view, so a typo in `method`
    or an option the method does not take would surface as an `AttributeError` or a `TypeError` from inside
    :meth:`NamedView.apply`, in a scene that was drawing fine a moment earlier. Binding both here means the
    error lands on the line that wrote the row.

    Args:
        name: The name callers will pass to :meth:`digitalearth.three_d.Scene3D.view`.
        method: The :class:`pyvista.Plotter` method that aims the camera. It has to be one of PyVista's own
            ``view_*`` methods: a method that draws instead of aiming would leave the camera untouched and the
            view would silently do nothing.
        describes: What the view shows, in a phrase — what a listing prints beside the name.
        replace: Whether to overwrite a name that is already in the table. `False` (the default) refuses,
            because two callers registering ``"top"`` would otherwise silently fight over it; pass `True` to
            override a shipped view deliberately.
        **options: The keywords `method` is called with — ``negative=True``, ``vector=(1, 1, 0)``, ….

    Returns:
        The row that was registered.

    Raises:
        ValueError: if `name` is blank, if `name` is taken and `replace` is `False`, if `method` is not a
            ``view_*`` method of :class:`pyvista.Plotter`, or if `options` do not bind to its signature.

    Examples:
        - Register a view and look at a scene from it:
            ```python
            >>> from digitalearth.three_d import Scene3D
            >>> from digitalearth.three_d.views import forget_view, register_view
            >>> _ = register_view(
            ...     "north_east",
            ...     method="view_vector",
            ...     describes="from the north-east, the way our report figures are drawn",
            ...     vector=(1.0, 1.0, 0.6),
            ... )
            >>> scene = Scene3D(off_screen=True)
            >>> scene.view("north_east").camera.position[2] > 0.0
            True
            >>> scene.close()
            >>> _ = forget_view("north_east")

            ```
        - A method PyVista does not have is refused where it was written:
            ```python
            >>> from digitalearth.three_d.views import register_view
            >>> register_view("nowhere", method="view_from_behind", describes="nowhere")
            Traceback (most recent call last):
                ...
            ValueError: view 'nowhere': pyvista.Plotter has no view method 'view_from_behind'

            ```
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"a view's name must be a non-empty string; got {name!r}")
    name = name.strip()
    if name in _VIEWS and not replace:
        raise ValueError(
            f"view {name!r} is already registered ({_VIEWS[name].describes}); "
            "pass replace=True to override it deliberately"
        )
    _check_method(name, method, options)
    view = NamedView(name, method, describes, MappingProxyType(dict(options)))
    _VIEWS[name] = view
    return view


def forget_view(name: str) -> NamedView:
    """Take a view back out of the table.

    The other half of :func:`register_view`. The table is process-global — like the object registry the layers
    use — so a caller who put a name in it, and a doctest or a test that did, needs a way to put the table back
    the way it was rather than leaking a name into everything that runs afterwards.

    Args:
        name: The view to forget.

    Returns:
        The row that was removed, so a caller can register it again later.

    Raises:
        KeyError: for a name nothing registered, naming the ones that are.

    Examples:
        - Register, use, forget:
            ```python
            >>> from digitalearth.three_d.views import forget_view, register_view, view_names
            >>> _ = register_view("underside", method="view_xy", describes="from below", negative=True)
            >>> "underside" in view_names()
            True
            >>> forget_view("underside").method
            'view_xy'
            >>> "underside" in view_names()
            False

            ```
    """
    view = named_view(name)
    del _VIEWS[view.name]
    return view


def _check_method(name: str, method: str, options: Mapping[str, Any]) -> None:
    """Refuse a row PyVista could not carry out.

    Args:
        name: The view being registered, named in the error.
        method: The `pyvista.Plotter` method the row points at.
        options: The keywords it would be called with.

    Raises:
        ValueError: if `method` is not one of PyVista's ``view_*`` methods, or `options` do not bind to it.
    """
    # Imported here, not at module level: reading the table costs no engine import (`test_lazy_engine.py`),
    # and registering is the one moment where reaching PyVista buys a real check.
    import inspect

    import pyvista as pv

    attribute = (
        getattr(pv.Plotter, method, None) if method.startswith(_VIEW_PREFIX) else None
    )
    if not callable(attribute):
        raise ValueError(
            f"view {name!r}: pyvista.Plotter has no view method {method!r}"
        )
    try:
        # `None` stands in for `self`: the signature is read off the unbound function, so the first
        # parameter has to be filled for the keywords after it to be checked at all.
        inspect.signature(attribute).bind(None, **dict(options))
    except TypeError as error:
        raise ValueError(
            f"view {name!r}: pyvista.Plotter.{method} does not take {dict(options)} — {error}"
        ) from error
