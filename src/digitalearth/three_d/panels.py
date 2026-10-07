"""grid — compose several 3-D scenes into one subplot window with linked cameras (TD-16).

A single :class:`~digitalearth.three_d.scene3d.Scene3D` owns one renderer and draws into one view. Putting two
scenes side by side — before/after, two variables, two exaggerations — is a *compositor's* job, not a method on
one scene, exactly as the 2-D tier's ``grid``/``facet``/``shared_colorbar`` are module functions that compose
across scenes rather than verbs on one map. :func:`grid` is that function for the 3-D tier.

It builds one :class:`pyvista.Plotter` with a ``(rows, cols)`` subplot layout and redraws each scene's layers
into its own cell through that scene's **own renderer** — so every layer keeps the styling its builder gave it,
rather than being re-added with defaults. With ``link=True`` the cameras are linked, so orbiting one cell
orbits them all, which is what makes a before/after read as one comparison.
"""

from typing import TYPE_CHECKING, Any, Sequence

from digitalearth.base.spec import FigureSpec, PanelSpec
from digitalearth.three_d.base import DEFAULT_CAMERA, PANEL_ID

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    import pyvista as pv

    from digitalearth.three_d.scene3d import Scene3D

__all__ = ["grid"]


def _layout(count: int, shape: tuple[int, int] | None) -> tuple[int, int]:
    """Resolve the ``(rows, cols)`` subplot layout for ``count`` scenes.

    Args:
        count: How many scenes are being composed.
        shape: The caller's explicit ``(rows, cols)``, or ``None`` for a single row.

    Returns:
        The ``(rows, cols)`` to build the plotter with.

    Raises:
        ValueError: if ``count`` is zero, or an explicit ``shape`` has fewer cells than there are scenes.
    """
    if count == 0:
        raise ValueError("grid() needs at least one scene to compose")
    if shape is None:
        return 1, count
    rows, cols = int(shape[0]), int(shape[1])
    if rows * cols < count:
        raise ValueError(
            f"grid(shape={shape!r}) has {rows * cols} cells for {count} scenes; give a larger shape"
        )
    return rows, cols


def grid(
    scenes: "Sequence[Scene3D]",
    *,
    shape: tuple[int, int] | None = None,
    link: bool = True,
    window_size: tuple[int, int] = (1024, 768),
    off_screen: bool | None = None,
) -> "pv.Plotter":
    """Compose several 3-D scenes into one subplot window, optionally with linked cameras.

    Each scene is redrawn into its own cell through its own renderer, so its layers keep their styling. The
    scenes are **consumed**: each is rebound to the shared plotter (as the 2-D ``grid`` consumes its maps), so
    use the returned plotter from here on, not the scenes.

    Args:
        scenes: The scenes to place, in reading order (left to right, top to bottom).
        shape: The ``(rows, cols)`` layout; ``None`` puts them in a single row.
        link: Link the cells' cameras so orbiting one orbits all — the point of a side-by-side comparison.
        window_size: The whole window's size in pixels.
        off_screen: Render without a window; ``None`` follows :data:`pyvista.OFF_SCREEN`.

    Returns:
        The composed :class:`pyvista.Plotter`, ready to ``show()`` or ``screenshot()``.

    Raises:
        ValueError: if ``scenes`` is empty, or ``shape`` has fewer cells than there are scenes.

    Examples:
        - Two DEMs side by side under one linked camera:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import get_source
            >>> from digitalearth.three_d import Scene3D
            >>> from digitalearth.three_d.panels import grid
            >>> left = Scene3D(off_screen=True)
            >>> _ = left.terrain(get_source(np.add.outer(np.arange(6.0), np.arange(6.0))))
            >>> right = Scene3D(off_screen=True)
            >>> _ = right.terrain(get_source(np.add.outer(np.arange(6.0), np.arange(6.0)[::-1])))
            >>> plotter = grid([left, right], shape=(1, 2), off_screen=True)
            >>> len(plotter.renderers)
            2
            >>> plotter.close()

            ```
    """
    import pyvista as pv

    scenes = list(scenes)
    rows, cols = _layout(len(scenes), shape)
    resolved_off_screen = pv.OFF_SCREEN if off_screen is None else off_screen
    plotter = pv.Plotter(
        shape=(rows, cols),
        off_screen=resolved_off_screen,
        window_size=list(window_size),
    )
    empty = FigureSpec(panels=(PanelSpec(PANEL_ID, DEFAULT_CAMERA),))
    for index, scene in enumerate(scenes):
        row, col = divmod(index, cols)
        plotter.subplot(row, col)
        _redraw_into(scene, plotter, empty)
    if link:
        plotter.link_views()
    return plotter


def _redraw_into(scene: Any, plotter: "pv.Plotter", empty: FigureSpec) -> None:
    """Rebind `scene` to `plotter`'s active cell and redraw its layers there through its own renderer.

    Args:
        scene: The scene to draw into the active subplot.
        plotter: The shared multi-panel plotter, with the target cell already selected via ``subplot``.
        empty: An empty figure to diff against, so the renderer treats every layer as newly added.
    """
    figure = scene._figure
    previous = scene._plotter
    scene._renderer._drawn.clear()
    # Bind through the setter so the scene's camera, vertical scale and decoration are applied to the active
    # cell; then draw every described layer into it through the scene's real renderer, which keeps each
    # layer's styling (a plain add_mesh here would lose the cmap/scalars the builder chose).
    scene.plotter = plotter
    scene._renderer.apply(empty, figure)
    # The scene built its own single-cell plotter when its first layer was drawn; now that it is rebound to the
    # shared one, close the orphan so the compositor does not leak a render window per scene.
    if previous is not None and previous is not plotter:
        previous.close()
