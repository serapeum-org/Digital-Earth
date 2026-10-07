"""Every 3-D builder that splats a caller's keywords into PyVista, held to refusing the ones it derives (R2-L3).

`point_cloud` got a keyword guard of its own in round 1, and the commit that added it audited the other builders
by hand: each was said to be covered by one of three mechanisms — PyVista's own unknown-keyword refusal,
``props.pop("cmap", ...)`` before a derived one is passed, or Python's duplicate-keyword ``TypeError`` from
``**style, **props``. That audit lived in a commit message, which is to say nowhere: nothing asserted any of it,
so a future builder splatting a derived style over the caller's props would reproduce the silent overwrite
`point_cloud` was fixed for, and no test would notice.

This is the audit as a guard, and it is written so the *set* of builders cannot drift either: the methods are
discovered from `Scene3D`'s own signatures rather than listed, and every one taking ``**kwargs`` has to appear on
one side of :data:`DRAWS_WITH_KWARGS` / :data:`NOT_A_DRAWING_CALL`. A builder added tomorrow joins the
parametrised checks below by existing.

**Measured while writing this, and worth recording**: review L3 counted nine such builders and named
``coastlines`` among them. There is no ``Scene3D.coastlines`` — coastlines are a *layer kind*, drawn through
``globe(coastlines=True)``, and the discovery below finds eight drawing builders rather than nine.
"""

# The package imports have to follow pytest.importorskip("pyvista") — importing digitalearth.three_d without
# pyvista is the very thing the skip exists to avoid — so E402 is expected throughout.
# ruff: noqa: E402
import inspect

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString, Polygon

from digitalearth.base.sources import get_source
from digitalearth.three_d import Scene3D
from digitalearth.three_d.base import classified_scalars

#: The keyword every derived colour style pins, and so the one a pinned caller value would be overwritten by.
#: Read out of `classified_scalars` rather than written here, which is what ties this to the code that derives
#: it: were the style to stop binding the array name, this guard would be about the wrong keyword.
DERIVED = "scalars"

#: The builders that splat the caller's ``**kwargs`` into a PyVista drawing call, with the mechanism that keeps a
#: pinned :data:`DERIVED` from being silently overwritten. Measured on PyVista 0.48.4, one call each.
DRAWS_WITH_KWARGS = {
    "terrain": "the caller's array name is honoured and VTK refuses it by name when the mesh has no such array",
    "point_cloud": "this drawer's own guard, which names the keyword and the colouring that derives it",
    "vectors": "`add_mesh(merged, scalars=VALUE, **style, **props)` — Python's duplicate-keyword TypeError",
    "extruded_polygons": "`_classify_or_refuse`, which names the keyword and the scheme that derives it",
    "lines": "`add_mesh(merged, scalars=VALUE, **style, **props)` — Python's duplicate-keyword TypeError",
    "streamlines": "`add_mesh(mesh, scalars=_SPEED, **props)` — Python's duplicate-keyword TypeError",
    "volume": "the caller's array name is honoured; VTK refuses one the grid does not carry",
    "isosurface": "`add_mesh(mesh, scalars=FIELD, **props)` — Python's duplicate-keyword TypeError",
    "globe": "geovista's own `add_mesh` call, reached with `scalars` already pinned",
    "text": "`add_point_labels(anchor, [label], **style, **props)` — it takes no array, so PyVista refuses one",
    "add_mesh": "the caller's own object and the caller's own array name; VTK refuses a missing array",
    "add_volume": "the caller's own grid and array name; VTK refuses a missing array",
}

#: The other methods whose signature ends in ``**kwargs`` and which draw nothing from a derived style, each with
#: the reason — the same "declared, with a reason" shape :data:`~digitalearth.three_d.bigdata.UNREDUCED` uses. A
#: method here is exempt because of what it does, not because nobody got round to it.
NOT_A_DRAWING_CALL = {
    "axes": "decoration: its keywords reach show_bounds, which takes no scalars and derives nothing from data",
    "orientation_axes": "decoration: its keywords reach add_axes, which draws a widget rather than the data",
    "from_figure": "a constructor: the keywords configure the new scene, and no style is derived from data",
    "orbit": "an animation loop over layers already drawn; its keywords reach the writer, not a drawer",
    "save": "writes the scene that was already drawn; its keywords reach the exporter",
    "screenshot": "captures the scene that was already drawn; its keywords reach the plotter's own call",
    "show": "opens the window on the scene that was already drawn",
    "clip_plane": "a live widget below the seam: its keywords reach add_mesh_clip_plane, not a derived style",
    "slice_planes": "a live widget below the seam: its keywords reach add_mesh_slice, not a derived style",
    "clip_box": "a live widget below the seam: its keywords reach add_mesh_clip_box, not a derived style",
    "threshold": "a live widget below the seam: its keywords reach add_mesh_threshold, not a derived style",
    "isovalue": "a live widget below the seam: its keywords reach add_mesh_isovalue, not a derived style",
    "slider": "a live widget below the seam: its keywords reach add_slider_widget, not a derived style",
    "enable_picking": "turns on a pick gesture; its keywords reach enable_*_picking, not a derived style",
    "serve": "opens a live trame view of the already-described scene; its keywords reach show_trame",
}


def _dem():
    """Return a small ramped DEM as a `Source`.

    Returns:
        The source a `terrain` or `globe` layer is built from.
    """
    return get_source(np.add.outer(np.arange(6.0), np.arange(7.0)))


def _cube():
    """Return a small scalar cube.

    Returns:
        The array a `volume` or `isosurface` layer is built from.
    """
    axis = np.linspace(-2.0, 2.0, 12)
    squared = np.add.outer(np.add.outer(axis**2, axis**2), axis**2)
    return np.exp(-squared)


def _points():
    """Return a small point table.

    Returns:
        The table a `point_cloud` or `vectors` layer is built from.
    """
    return np.random.default_rng(0).random((30, 3))


def _squares():
    """Return two unit squares with a numeric column.

    Returns:
        The frame an `extrusion` layer is built from.
    """
    return gpd.GeoDataFrame(
        {"pop": [10.0, 20.0]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        ],
    )


def _flow_field():
    """Return a small rotational vector field.

    Returns:
        The ``(nz, ny, nx, 3)`` field a `streamlines` layer is built from.
    """
    ax = np.linspace(-1.0, 1.0, 8)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.stack([-y, x, np.zeros_like(z)], axis=-1)


def _lines():
    """Return two line features with a numeric column.

    Returns:
        The frame a `lines` layer is built from.
    """
    return gpd.GeoDataFrame(
        {"flow": [3.0, 7.0]},
        geometry=[
            LineString([(0, 0), (1, 1), (2, 0)]),
            LineString([(0, 2), (2, 2)]),
        ],
    )


def _grid():
    """Return a small `ImageData` carrying a cell field.

    Returns:
        The grid `add_volume` is handed.
    """
    grid = pv.ImageData(dimensions=(7, 7, 7))
    grid.cell_data["v"] = np.linspace(0.0, 1.0, 216)
    return grid


def _draw(scene, builder: str, extra: dict):
    """Call one builder on `scene` with `extra` added to its keywords.

    Args:
        scene: The scene to draw into.
        builder: The method name, one of :data:`DRAWS_WITH_KWARGS`.
        extra: The keywords under test.

    Returns:
        Whatever the builder returns.
    """
    if builder == "globe":
        pytest.importorskip("geovista")
    calls = {
        "terrain": lambda: scene.terrain(_dem(), **extra),
        "point_cloud": lambda: scene.point_cloud(
            _points(), values=_points()[:, 0], **extra
        ),
        "vectors": lambda: scene.vectors(
            _points(), np.tile([1.0, 0.0, 0.0], (30, 1)), factor=0.1, **extra
        ),
        "extruded_polygons": lambda: scene.extruded_polygons(
            _squares(), column="pop", height=2.0, **extra
        ),
        "lines": lambda: scene.lines(_lines(), column="flow", **extra),
        "streamlines": lambda: scene.streamlines(_flow_field(), n_points=30, **extra),
        "volume": lambda: scene.volume(_cube(), **extra),
        "isosurface": lambda: scene.isosurface(_cube(), isosurfaces=[0.3], **extra),
        "globe": lambda: scene.globe(_dem(), coastlines=False, **extra),
        "text": lambda: scene.text(0.0, 0.0, "here", **extra),
        "add_mesh": lambda: scene.add_mesh(pv.Sphere(), **extra),
        "add_volume": lambda: scene.add_volume(_grid(), **extra),
    }
    return calls[builder]()


def _kwargs_methods() -> set:
    """Return every public `Scene3D` method whose signature ends in ``**kwargs``.

    Returns:
        The method names, discovered from the signatures rather than listed — which is what makes a builder
        added later join the checks below without anyone remembering to add it.
    """
    found = set()
    for name in dir(Scene3D):
        if name.startswith("_"):
            continue
        held = getattr(Scene3D, name)
        if not callable(held):
            continue
        try:
            signature = inspect.signature(held)
        except (
            TypeError,
            ValueError,
        ):  # pragma: no cover - every method here is introspectable
            continue
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            found.add(name)
    return found


class TestEveryKwargsMethodIsClassified:
    """The drift guard over the audit: a method taking ``**kwargs`` cannot be silently unexamined."""

    def test_the_two_halves_account_for_every_one_of_them(self):
        """A builder added tomorrow has to say which half it is in, or this fails naming it.

        Test scenario:
            Discovered from `Scene3D`'s signatures, so the set cannot fall behind the code. The audit that
            this replaces was a list in a commit message, and a list is exactly what goes stale.
        """
        classified = set(DRAWS_WITH_KWARGS) | set(NOT_A_DRAWING_CALL)
        assert classified == _kwargs_methods(), sorted(
            classified.symmetric_difference(_kwargs_methods())
        )

    def test_no_method_claims_both_halves(self):
        """ "It draws with them" and "it does not draw at all" cannot both be true of one method."""
        assert not set(DRAWS_WITH_KWARGS) & set(NOT_A_DRAWING_CALL)

    @pytest.mark.parametrize("method", sorted(NOT_A_DRAWING_CALL))
    def test_every_exempt_method_says_why(self, method):
        """An exemption nobody can read is an exemption nobody can check.

        Args:
            method: The exempt method under test.
        """
        assert NOT_A_DRAWING_CALL[method].strip(), (
            f"{method} is exempt without a reason"
        )

    def test_the_keyword_under_test_is_one_the_colouring_really_derives(self):
        """The premise: were the derived style to stop binding the array name, this guard would be idle.

        Test scenario:
            `DERIVED` is checked against `classified_scalars`' own output rather than trusted, so this suite
            cannot go on testing a keyword nothing sets any more.
        """
        style = classified_scalars([1.0, 5.0, 9.0], scheme=None, k=5, cmap="viridis")
        assert DERIVED in style, (
            f"the colouring no longer derives {DERIVED!r}; it derives {sorted(style)}"
        )


class TestAPinnedDerivedKeywordIsNeverSilentlyAccepted:
    """#294's rule, applied to the eight builders whose coverage was claimed rather than asserted."""

    @pytest.mark.parametrize("builder", sorted(DRAWS_WITH_KWARGS))
    def test_pinning_it_refuses_rather_than_drawing(self, builder):
        """Silently drawing is the one outcome that leaves the caller believing they were honoured.

        Args:
            builder: The builder under test.

        Test scenario:
            The class differs by mechanism — this tier's own guard, Python's duplicate-keyword `TypeError`, or
            VTK refusing an array the mesh does not carry — so what is asserted is that *something* refuses,
            not which. Asserting the class would make a PyVista upgrade look like a defect while leaving the
            claim itself unchecked.
        """
        scene = Scene3D(off_screen=True)
        try:
            with pytest.raises((TypeError, KeyError, ValueError)) as refused:
                _draw(scene, builder, {DERIVED: "no-such-array"})
        finally:
            scene.close()
        assert str(refused.value), f"{builder} refused with an empty message"

    @pytest.mark.parametrize("builder", sorted(DRAWS_WITH_KWARGS))
    def test_the_same_call_draws_without_it(self, builder):
        """Built the other way round: without this, a builder that refused everything would pass above.

        Args:
            builder: The builder under test.
        """
        scene = Scene3D(off_screen=True)
        try:
            drawn = _draw(scene, builder, {})
            ids = list(scene.layer_ids)
        finally:
            scene.close()
        assert (drawn is not None, ids != []) == (True, True), (
            f"{builder} drew nothing when its keywords were left alone: {ids}"
        )

    @pytest.mark.parametrize("builder", sorted(DRAWS_WITH_KWARGS))
    def test_a_keyword_nothing_takes_is_refused(self, builder):
        """The first mechanism the audit claimed: PyVista's own unknown-keyword guard, per builder.

        Args:
            builder: The builder under test.

        Test scenario:
            This is what makes ``**kwargs`` on a builder defensible at all — a keyword that reaches the engine
            and is dropped is how `reorder=` and `basemap_switch=` came to look implemented on another tier.
        """
        scene = Scene3D(off_screen=True)
        try:
            with pytest.raises(TypeError) as refused:
                _draw(scene, builder, {"not_a_keyword_at_all": 1})
        finally:
            scene.close()
        assert "not_a_keyword_at_all" in str(refused.value), (
            f"{builder} refused an unknown keyword without naming it: {refused.value}"
        )
