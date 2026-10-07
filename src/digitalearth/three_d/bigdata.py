"""The 3-D tier's big-data route: reduce what is too heavy to carry, and report what cannot be (#207).

Before this the tier had no large-data path at all — "every builder loads the whole array or table into one mesh
and hands it straight to VTK", as #207 put it, while both 2-D tiers had one. What that cost was not frame rate:
measured on a desktop GPU at 800x600, the per-frame draw of a triangulated terrain was **flat** from 79 k to
20 M cells, 2-3 ms either way. What it cost was everything that carries the geometry rather than drawing it —
the self-contained page `export_html` writes, and the host copy of the mesh:

| triangles | page (`export_html`) | host mesh |
|---|---|---|
| 100 000 | 3.4 MB | 5.0 MB |
| 200 000 | 5.7 MB | 9.9 MB |
| 500 000 | 12.6 MB | 24.8 MB |
| 1 000 000 | 24.0 MB | 49.6 MB |
| 2 000 000 | 46.7 MB | 99.2 MB |

Measured end to end through the tier, a full-resolution 1000x1000 DEM — 998 001 structured cells — exports a
**39.0 MB** page, and a 5.1 M-triangle surface **107.6 MB**: the "page too large to open" #207 names.
:data:`DEFAULT_CELL_BUDGET` is set from that table, and the reduction's own cost is the counterweight:
``decimate_pro`` runs at ~6.3 microseconds per **input** cell (6.3 s for 1 M, 28 s for 5 M, 74 s for 11.5 M, all
measured), so a budget low enough to fire on ordinary data would spend seconds to save nothing.
``ImageData.resample`` is output-driven and costs 0.01-0.06 s whatever it is given.

**Two routes, and a declaration for every other kind.** A route is only meaningful where the cells form one
sampled surface, and a filter that fits a `PolyData` can refuse or ruin another class, so the route is keyed by
**layer kind** rather than sniffed off the object: :data:`REDUCTIONS` says how a kind reduces,
:data:`UNREDUCED` says why a kind does not, and ``tests/three_d/test_bigdata3d.py`` fails if a drawn kind is in
neither or in both. Type-sniffing is what that avoids — measured on PyVista 0.48.4, ``triangulate()`` on a
vertex-only point cloud returns a mesh of **zero cells**, so a generic "triangulate, then decimate" path would
have deleted every point cloud it was asked to thin.

**The budget counts what is drawn, in the unit the route reduces in**: triangles for the surface route, where
a structured quad counts as the two triangles VTK draws it as, and voxels for the volume route. Both sides of
each comparison are therefore the same quantity as the table above — the gate used to read a terrain's quad
count against a triangle budget, which admitted twice the geometry the number was chosen for (review R2-M11).
:func:`report_unreduced` is the exception, and deliberately so: it compares the object's own cells, because a
kind with no route has no triangle count the budget could be about — a point cloud draws none, and a block of
arrows is one small solid per sample.

An object at or under the budget is passed through untouched — not triangulated, not copied — so nothing that
renders today renders differently: the two shapes this tier builds, an all-triangle `PolyData` and a
structured sheet, are both counted arithmetically rather than by triangulating to find out.
"""

import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_CELL_BUDGET",
    "REDUCTIONS",
    "UNREDUCED",
    "reduce_surface",
    "reduce_volume",
    "report_unreduced",
]

logger = logging.getLogger(__name__)

#: Cells above which a 3-D layer takes its tier's big-data route, unless the scene or the call overrides it.
#: **Cells, not rows** — this is deliberately not
#: :data:`~digitalearth.base.bigdata.DEFAULT_BIG_DATA_THRESHOLD`, the 50 000-row cutoff the two 2-D tiers share
#: for switching a vector layer to a density renderer. A row is one feature; a cell is one triangle or one
#: voxel, and a single DEM tile is hundreds of thousands of them, so one number could not mean both.
#:
#: 500 000 is the largest budget in the module docstring's table whose exported page stays in the low tens of
#: megabytes (12.6 MB) and whose host copy stays under 25 MB; the next step up doubles the page to 24 MB. At the
#: margin the reduction costs ~3 s of ``decimate_pro``, against ~6 s at a 1 M budget.
#:
#: What that means for a DEM, measured through :func:`reduce_surface` — and in triangles, since a structured
#: quad is drawn as two of them: a 501x501 one (250 000 quads, exactly 500 000 triangles) is drawn as it is, and
#: a 502x502 one (251 001 quads, 502 002 triangles) is reduced to 499 999. A 708x708 DEM, which the quad-counting
#: gate let through whole at 999 698 triangles, is now reduced like its 709x709 neighbour — the two come out at
#: 499 999 and 500 000, which is what the budget was chosen to mean. A 1000x1000 DEM goes from a 39.0 MB
#: exported page to a 12.3 MB one, as it did before: at 1 996 002 triangles it was already over either gate.
DEFAULT_CELL_BUDGET: int = 500_000

#: The PyVista filter that reduces a surface mesh: boundary-preserving, which is what terrain edges and
#: coastlines need — plain ``decimate`` moves them.
_DECIMATE: str = "decimate_pro"

#: The PyVista filter that reduces a uniform volume: it resamples to fewer voxels per axis.
_RESAMPLE: str = "resample"

#: How each drawn kind is reduced above the budget, by the name of the PyVista filter that does it. Keyed by
#: kind, not by class, so the route is a decision recorded here rather than one guessed from an object.
REDUCTIONS: Mapping[str, str] = MappingProxyType(
    {
        "terrain": _DECIMATE,
        "isosurface": _DECIMATE,
        "volume": _RESAMPLE,
    }
)

#: The drawn kinds with no reduction route, and why — the same shape as
#: :attr:`~digitalearth.base.capabilities.Capabilities.absent`: "we decided against it, because" is an answer,
#: and one a reader can act on. Above the budget these are reported by :func:`report_unreduced` and drawn
#: whole; nothing is silently thinned.
UNREDUCED: Mapping[str, str] = MappingProxyType(
    {
        "point_cloud": (
            "a cloud is vertices with no faces, and decimate_pro refuses a mesh that is not all triangles; "
            "triangulating one returns zero cells, so the surface route would delete the layer rather than "
            "thin it. Thinning a cloud is a stride or a random sample, which is #207's remaining ask"
        ),
        "vectors": (
            "arrow glyphs are one small solid per sample rather than one sampled surface, so simplifying the "
            "block merges neighbouring arrows into shapes that are no longer arrows; the field is thinned by "
            "passing fewer samples"
        ),
        "extrusion": (
            "prisms are combined into an UnstructuredGrid, which PyVista gives no decimate_pro at all, and "
            "simplifying their outer surface would merge separate footprints into one building"
        ),
        "raster": (
            "geovista builds the sphere and owns its resolution; decimating a sphere's quads flattens the "
            "curvature the globe exists to show"
        ),
        "coastlines": (
            "a coastline is lines rather than a surface, and geovista already takes the resolution to load "
            "them at"
        ),
        "text": (
            "a label is one anchor point and one string, so there is never anything to reduce; a scene with "
            "too many labels to read is a composition the caller chose rather than data to thin"
        ),
        "custom:pyvista": (
            "the caller built the object and chose what it is made of, so reducing it here would overrule "
            "that silently; reduce it before handing it over"
        ),
    }
)


def reduce_surface(mesh: Any, budget: int, *, kind: str) -> Any:
    """Return `mesh` drawing at most `budget` triangles, simplified with ``decimate_pro`` if it draws more.

    The route for a **sampled surface** — a terrain built from a DEM, a shell extracted from a cube.
    ``decimate_pro`` rather than ``decimate`` because it preserves boundaries, which is what keeps a terrain's
    edge and a coastline where they were.

    **The budget is counted in the triangles VTK draws, on both sides of the comparison.** The module
    docstring's page-size table is in triangles and ``decimate_pro`` is given a triangle target, but the gate
    read ``mesh.n_cells`` — and a terrain under the budget is a `StructuredGrid` of **quads**, each drawn as
    two triangles (measured: 499 849 quads render as 999 698 triangles). So a terrain was admitted at up to
    twice the geometry the budget was chosen for, and one input cell past the gate halved the exported page
    (review R2-M11). An all-triangle `PolyData` — what ``contour`` hands the isosurface route — counts as it
    is, so the two kinds sharing this function are now measured in one unit.

    A mesh that is not already triangles is triangulated first, because ``decimate_pro`` refuses anything else
    (``NotAllTrianglesError``, measured on PyVista 0.48.4) — and a structured grid has its surface extracted
    before that, since ``triangulate()`` on one yields an `UnstructuredGrid`, which has no ``decimate_pro``.
    Point and cell arrays survive both steps, so a layer drawn with ``scalars="elevation"`` still is.

    Args:
        mesh: The mesh the drawer built — a `pyvista.PolyData` or `pyvista.StructuredGrid`.
        budget: The most triangles to draw. A mesh at or under it is returned **as it is**: untriangulated,
            uncopied, so no render that fits the budget changes.
        kind: The layer kind, named in the log line.

    Returns:
        The mesh to draw: `mesh` itself, or a simplified copy of it.

    Examples:
        - A surface over the budget comes back at or under it:
            ```python
            >>> import numpy as np, pyvista as pv
            >>> from digitalearth.three_d.bigdata import reduce_surface
            >>> ax = np.arange(40.0)
            >>> xx, yy = np.meshgrid(ax, ax)
            >>> grid = pv.StructuredGrid(xx, yy, xx * 0.1)
            >>> grid.n_cells, grid.n_cells * 2
            (1521, 3042)
            >>> reduce_surface(grid, 400, kind="terrain").n_cells <= 400
            True

            ```
        - A sheet whose quads fit the budget but whose triangles do not is still reduced:
            ```python
            >>> import numpy as np, pyvista as pv
            >>> from digitalearth.three_d.bigdata import reduce_surface
            >>> ax = np.arange(41.0)
            >>> grid = pv.StructuredGrid(*np.meshgrid(ax, ax), np.zeros((41, 41)))
            >>> grid.n_cells
            1600
            >>> reduce_surface(grid, 2000, kind="terrain").n_cells <= 2000
            True

            ```
        - One whose triangles are under the budget is the very same object:
            ```python
            >>> import numpy as np, pyvista as pv
            >>> from digitalearth.three_d.bigdata import reduce_surface
            >>> grid = pv.StructuredGrid(*np.meshgrid(np.arange(4.0), np.arange(4.0)), np.zeros((4, 4)))
            >>> reduce_surface(grid, 500_000, kind="terrain") is grid
            True

            ```
    """
    cells = int(mesh.n_cells)
    rendered, triangles = _rendered_triangles(mesh)
    if rendered == 0:
        # `triangulate()` empties a vertex-only mesh (measured, PyVista 0.48.4). Nothing in `REDUCTIONS`
        # reaches here, but a direct call could, and deleting a layer is not a reduction of it. Reported
        # against the mesh's own cells, because a cloud has no triangles for the budget to be about.
        if cells > budget:
            logger.warning(
                f"{kind}: {cells:,} cells exceed big_data_threshold={budget:,}, but the mesh has no faces "
                f"to simplify — drawn whole"
            )
        return mesh
    if rendered <= budget:
        return mesh
    if triangles is None:
        triangles = _triangles(mesh)
    smaller = triangles.decimate_pro(max(0.0, 1.0 - budget / triangles.n_cells))
    _report_reduction(kind, rendered, int(smaller.n_cells), budget, _DECIMATE)
    return smaller


def reduce_volume(
    grid: Any, budget: int, *, kind: str, scalars: str, preference: str
) -> Any:
    """Return `grid` at or under `budget` voxels, resampled to fewer per axis if it has more.

    The route for a **uniform volume**. ``resample`` is output-driven — it costs what it produces, not what it
    is given (0.01-0.06 s from 2 M to 17 M input cells, measured) — which is why a volume needs no threshold on
    the cost side at all, only on the size side.

    The target is given as explicit **dimensions** rather than a sample rate: a rate is applied to the point
    dimensions, and the budget counts cells, so rounding could land a rate-resampled grid just over the number
    the caller asked for. Truncating the per-axis cell counts cannot — provided the rate is one every axis can
    actually take, which :func:`_resample_target` is what settles.

    Args:
        grid: The `pyvista.ImageData` the drawer built.
        budget: The most cells to draw. A grid at or under it is returned as it is.
        kind: The layer kind, named in the log line.
        scalars: The array to carry through the resample — a volume's field.
        preference: Whether `scalars` is ``"cell"`` or ``"point"`` data, which decides which of the two
            `ImageData` layouts is resampled.

    Returns:
        The grid to draw: `grid` itself, or a resampled copy of it. At or under `budget` whenever any grid can
        be — one cell per axis is the floor, so a budget below a single cell is unreachable, and
        :func:`_report_reduction` says so rather than reporting the overshoot as a reduction.

    Examples:
        - A cube over the budget comes back at or under it, still carrying its field:
            ```python
            >>> import numpy as np, pyvista as pv
            >>> from digitalearth.three_d.bigdata import reduce_volume
            >>> cube = np.random.default_rng(0).random((20, 20, 20))
            >>> grid = pv.ImageData(dimensions=np.array(cube.shape[::-1]) + 1)
            >>> grid.cell_data["field"] = cube.transpose(2, 1, 0).ravel(order="F")
            >>> smaller = reduce_volume(grid, 2000, kind="volume", scalars="field", preference="cell")
            >>> smaller.n_cells <= 2000, list(smaller.cell_data)
            (True, ['field'])

            ```
        - A volume with one thin axis is reduced by the axes that can shrink, so the budget still holds:
            ```python
            >>> import numpy as np, pyvista as pv
            >>> from digitalearth.three_d.bigdata import reduce_volume
            >>> grid = pv.ImageData(dimensions=(1001, 1001, 2))
            >>> grid.cell_data["field"] = np.zeros(1_000_000)
            >>> smaller = reduce_volume(grid, 500_000, kind="volume", scalars="field", preference="cell")
            >>> smaller.n_cells <= 500_000
            True

            ```
    """
    cells = int(grid.n_cells)
    if cells <= budget:
        return grid
    per_axis = np.maximum(np.asarray(grid.dimensions, dtype="int64") - 1, 1)
    target = _resample_target(per_axis, budget)
    smaller = grid.resample(
        dimensions=target + 1, scalars=scalars, preference=preference
    )
    _report_reduction(kind, cells, int(smaller.n_cells), budget, _RESAMPLE)
    return smaller


def report_unreduced(kind: str, drawn: Any, budget: int) -> None:
    """Say so when a layer with no reduction route is drawn over the budget.

    #207's second ask, and the cheap half of it: "a warning or error above the budget instead of a silent hang,
    which alone would be a large improvement over today". A kind in :data:`REDUCTIONS` has already been reduced
    by its drawer and is not reported; one in :data:`UNREDUCED` is drawn whole, and this is what stops that
    being silent — the reason comes from the declaration, so it cannot drift from it.

    Args:
        kind: The layer's kind, as its `LayerSpec` holds it.
        drawn: What was drawn for it — the mesh, grid or object. Anything without an `n_cells` is passed over,
            since there is nothing to compare.
        budget: The scene's cell budget.
    """
    reason = UNREDUCED.get(kind)
    cells = getattr(drawn, "n_cells", None)
    if reason is None or cells is None or int(cells) <= budget:
        return
    logger.warning(
        f"{kind}: {int(cells):,} cells exceed big_data_threshold={budget:,} and this tier has no reduction "
        f"for them — {reason}. Drawn whole; a heavy scene exports to a heavy page"
    )


def _resample_target(per_axis: np.ndarray, budget: int) -> np.ndarray:
    """Return the per-axis cell counts to resample to, whose product is at or under `budget`.

    One rate over every axis holds the budget only while no axis floors away under it. ``floor(p * r)`` is 0
    for a one-cell axis at any rate below 1, and raising that 0 back to 1 — which it has to be, since an axis
    of no cells has no cells to draw — multiplies the product back up by the very factor the rate had divided
    out. Measured on the clamp this replaces: a ``(1, 1000, 1000)`` volume at a 500 000 budget came back at
    **628 849** cells, and at a 1 000 budget at 10 000, ten times over (review R2-H6).

    An axis pinned at its floor is a known factor rather than a shrinking one, so the answer is to take it out
    of the rate and divide the budget by it, then recompute over the axes that can still shrink. That
    converges: each pass either pins an axis — there are three — or produces a target no axis floored away,
    and ``prod(floor(p_i * r))`` is then at most ``prod(p_i) * r ** k``, which is the room left by
    construction.

    **The floor is two cells, not one, wherever the axis has two to give.** ``ImageData.resample`` declines a
    target of two *points* on an axis and leaves that axis exactly as it arrived — measured on PyVista 0.48.4:
    a ``(801, 801, 3)`` grid asked for ``(585, 585, 2)`` comes back ``(585, 585, 3)``, and a 12-point cube
    asked for ``(2, 2, 2)`` comes back unchanged, while 3 and above are honoured. So an axis that arrives with
    two or more cells can be taken down to two and no further, and only an axis that already had one keeps it.

    Args:
        per_axis: Cells along each axis of the grid being reduced, each at least 1.
        budget: The most cells the resampled grid may have.

    Returns:
        The per-axis cell counts to resample to, each at its floor or above. Their product is at or under
        `budget` whenever the floors themselves are; a budget below the smallest grid ``resample`` can produce
        has no answer, so the floors come back and :func:`_report_reduction` says the budget was not reached.
    """
    target = np.asarray(per_axis, dtype="int64").copy()
    floors = np.minimum(target, 2)
    pinned = np.zeros(target.shape, dtype=bool)
    while True:
        free = ~pinned
        if not free.any():
            return target
        room = (
            budget / float(np.prod(target[pinned])) if pinned.any() else float(budget)
        )
        rate = (room / float(np.prod(target[free]))) ** (1.0 / int(free.sum()))
        scaled = np.floor(target * rate).astype("int64")
        thin = free & (scaled < floors)
        if not thin.any():
            target[free] = scaled[free]
            return target
        pinned |= thin
        target[thin] = floors[thin]


def _rendered_triangles(mesh: Any) -> tuple:
    """Return how many triangles `mesh` is drawn as, and the triangulated copy if one had to be built.

    The budget is a triangle count, so this is the quantity :func:`reduce_surface` gates on. Two shapes answer
    without triangulating anything, which is what keeps the promise that a surface inside the budget is passed
    through untouched:

    * an all-triangle `PolyData` — what ``contour`` hands the isosurface route — is its own count;
    * a **structured sheet** — one of whose three dimensions is a single point, which is how every terrain in
      this tier is built — is quads, and a quad is drawn as exactly two triangles (measured on PyVista 0.48.4:
      a 708x708 DEM's 499 849 cells render as 999 698, a ratio of 2.0 at every size tried).

    Anything else is triangulated and counted, which is exact for any class and is what a caller's own mesh
    arriving through a direct call gets.

    Args:
        mesh: The mesh the drawer built.

    Returns:
        tuple: ``(triangles, triangulated)`` — the count, and the all-triangle `PolyData` it was read off, or
        ``None`` when the count was arrived at without building one. A vertex-only mesh counts 0, since it
        draws no triangles at all.
    """
    import pyvista as pv

    if isinstance(mesh, pv.PolyData) and mesh.is_all_triangles:
        return int(mesh.n_cells), mesh
    if isinstance(mesh, pv.StructuredGrid) and 1 in tuple(mesh.dimensions):
        return 2 * int(mesh.n_cells), None
    triangulated = _triangles(mesh)
    return int(triangulated.n_cells), triangulated


def _triangles(mesh: Any) -> Any:
    """Return `mesh` as an all-triangle `PolyData`, which is the only thing ``decimate_pro`` accepts.

    Args:
        mesh: The mesh to convert.

    Returns:
        The triangulated surface. A `PolyData` that is already all triangles is returned unchanged.
    """
    import pyvista as pv

    if isinstance(mesh, pv.PolyData):
        surface = mesh
    else:
        # `pass_pointid`/`pass_cellid` off: the extraction otherwise adds `vtkOriginalPointIds` arrays that
        # travel with the mesh into the figure and mean nothing to a caller. `algorithm` is named because
        # PyVista warns that its default is changing, and today's behaviour is the one measured here.
        surface = mesh.extract_surface(
            algorithm="dataset_surface", pass_pointid=False, pass_cellid=False
        )
    return surface if surface.is_all_triangles else surface.triangulate()


def _report_reduction(
    kind: str, before: int, after: int, budget: int, filter_name: str
) -> None:
    """Log what a reduction actually did.

    #207 asks for a reduction that is "predictable and reported rather than magic", so the line carries the
    measured before and after rather than the reduction that was requested.

    Args:
        kind: The layer kind.
        before: Cells in the mesh the drawer built.
        after: Cells in the mesh that is drawn.
        budget: The budget that triggered it.
        filter_name: The PyVista filter that did it.
    """
    if after > budget:
        # A reduction that overshot and said "reduced to N" read as a success, so a caller who saw the line
        # believed the scene fitted (review R2-H6). What cannot be reached is reported as what it is: the
        # floor of one cell per axis is a grid of one cell, so a budget under that has no answer, and
        # `ImageData.resample` declines a target of two points on every axis outright (measured, PyVista
        # 0.48.4), which leaves the grid as it was.
        logger.warning(
            f"{kind}: {before:,} cells exceed big_data_threshold={budget:,} and {filter_name} could not "
            f"reach it — drawn at {after:,} cells. Raise the budget, or pass a coarser layer"
        )
        return
    logger.info(
        f"{kind}: {before:,} cells exceed big_data_threshold={budget:,} — reduced to {after:,} with "
        f"{filter_name} (raise the budget to draw it whole)"
    )
