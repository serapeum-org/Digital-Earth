"""The 3-D tier's big-data route — what is reduced, what is not, and what reached PyVista (order 25, #207).

#207's finding was that nothing on this tier reduces anything: every builder loads the whole array or table into
one mesh and hands it to VTK. These tests hold the answer to that in the only place it can be checked honestly —
**the cell count read off the PyVista object the plotter was given**, through `Scene3D.mesh_of`, not off the
record the layer keeps of the budget it was told about. The repeated shape this repo has shipped is a value that
was recorded and never reached the renderer, so recording is asserted *as well*, never *instead*.

Two routes, tested separately because a budget that works on a `PolyData` can be silently inert on an
`ImageData`: `decimate_pro` for a surface (terrain, isosurface) and `resample` for a volume. The kinds with no
route — a point cloud, a vector field, a prism block — are held to the other half of #207's ask: above the budget
they say so by name rather than hanging silently, and nothing is quietly dropped from them.
"""

# The package imports have to follow pytest.importorskip("pyvista") — importing digitalearth.three_d
# without pyvista is the very thing the skip exists to avoid — so E402 is expected throughout.
# ruff: noqa: E402
import logging

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import Polygon

from digitalearth.base.bigdata import DEFAULT_BIG_DATA_THRESHOLD
from digitalearth.base.sources import get_source
from digitalearth.three_d import Scene3D
from digitalearth.three_d.bigdata import (
    DEFAULT_CELL_BUDGET,
    REDUCTIONS,
    UNREDUCED,
)
from digitalearth.three_d.renderer import DRAWN_KINDS
from digitalearth.three_d.terrain import ELEVATION
from digitalearth.three_d.volume import FIELD

#: A budget small enough that the tiny fixtures below cross it, so a test is a second rather than a minute.
SMALL = 2_000


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test so no window opens in CI."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def _dem(n: int = 100):
    """Return a `Source` over an ``n x n`` ramped DEM — ``(n - 1) ** 2`` structured cells.

    Args:
        n: Cells per side.

    Returns:
        The source a `terrain` layer is built from.
    """
    ax = np.linspace(0.0, 1.0, n)
    return get_source(np.add.outer(ax, ax))


def _cube(side: int = 20):
    """Return a ``side ** 3`` scalar cube — ``side ** 3`` volume cells.

    Args:
        side: Samples per axis.

    Returns:
        The cube a `volume` or `isosurface` layer is built from.
    """
    ax = np.linspace(-2.0, 2.0, side)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.exp(-(xx**2 + yy**2 + zz**2))


def _points(count: int = 5_000):
    """Return an ``(N, 3)`` point table.

    Args:
        count: How many points.

    Returns:
        The table a `point_cloud` layer is built from.
    """
    rng = np.random.default_rng(0)
    return rng.random((count, 3))


def _squares(count: int = 60):
    """Return a GeoDataFrame of unit squares, for an extrusion big enough to cross a small budget.

    Args:
        count: How many squares.

    Returns:
        The frame an `extrusion` layer is built from.
    """
    return gpd.GeoDataFrame(
        {"pop": np.arange(float(count))},
        geometry=[
            Polygon([(x, 0), (x + 1, 0), (x + 1, 1), (x, 1)]) for x in range(count)
        ],
    )


class TestEveryDrawnKindDeclaresItsRoute:
    """The declaration, and the drift guard over it — the tier's `absent` pattern applied to reduction."""

    def test_every_drawn_kind_is_in_exactly_one_half(self):
        """A kind added to the tier has to say how it reduces, or why it does not."""
        declared = set(REDUCTIONS) | set(UNREDUCED)
        assert declared == set(DRAWN_KINDS), sorted(
            declared.symmetric_difference(DRAWN_KINDS)
        )

    def test_no_kind_claims_a_route_and_a_reason(self):
        """ "It reduces like this" and "it cannot be reduced, because" cannot both be true of one kind."""
        assert not set(REDUCTIONS) & set(UNREDUCED)

    @pytest.mark.parametrize("route", sorted(set(REDUCTIONS.values())))
    def test_every_route_is_a_filter_pyvista_offers(self, route):
        """The table is built over PyVista's own filters; a renamed one is a route that raises when used."""
        offered = [
            callable(getattr(pv.PolyData, route, None)),
            callable(getattr(pv.ImageData, route, None)),
        ]
        assert any(offered), f"neither PolyData nor ImageData offers {route!r}"

    @pytest.mark.parametrize("kind", sorted(UNREDUCED))
    def test_every_unreduced_kind_says_why(self, kind):
        """A kind listed as unreducible without a reason is a gap nobody can act on."""
        assert UNREDUCED[kind].strip()


class TestTheBudgetIsTheTiersOwnNumber:
    """C8's shape on this tier: one name, one default, an attribute and a per-call override."""

    def test_the_default_is_the_measured_cell_budget(self):
        """Measured, not guessed — the number the commit body's timings support."""
        assert DEFAULT_CELL_BUDGET == 500_000

    def test_a_fresh_scene_carries_it(self):
        """The attribute is the map-level reach of the same number."""
        scene = Scene3D(off_screen=True)
        budget = scene.big_data_threshold
        scene.close()
        assert budget == DEFAULT_CELL_BUDGET

    def test_it_is_not_the_two_d_tiers_row_count(self):
        """Cells are not rows, so the shared row default would be the wrong number here."""
        assert DEFAULT_CELL_BUDGET != DEFAULT_BIG_DATA_THRESHOLD


class TestATerrainOverTheBudgetIsDecimated:
    """The `decimate_pro` route, read off the mesh the plotter was handed."""

    def test_the_drawn_mesh_is_at_or_under_the_budget(self):
        """The assertion #207 asks for: the real PyVista object, after drawing, not the description."""
        scene = Scene3D(off_screen=True)
        scene.terrain(_dem(), big_data_threshold=SMALL)
        cells = scene.mesh_of("terrain-1").n_cells
        scene.close()
        assert cells <= SMALL, cells

    def test_it_really_was_reduced_rather_than_arriving_small(self):
        """Without this the test above would pass for a fixture that never crossed the budget."""
        scene = Scene3D(off_screen=True)
        scene.terrain(_dem())
        cells = scene.mesh_of("terrain-1").n_cells
        scene.close()
        assert cells > SMALL, cells

    def test_a_terrain_under_the_budget_is_drawn_untouched(self):
        """Under the budget nothing is triangulated or simplified, so no existing render moves."""
        source = _dem()
        expected = (len(source.y.values) - 1) * (len(source.x.values) - 1)
        scene = Scene3D(off_screen=True)
        scene.terrain(source)
        mesh = scene.mesh_of("terrain-1")
        cells, kind = mesh.n_cells, type(mesh).__name__
        scene.close()
        assert (cells, kind) == (expected, "StructuredGrid")

    def test_the_reduced_surface_still_carries_its_elevation(self):
        """`add_mesh(scalars="elevation")` would raise if the filter chain dropped it."""
        scene = Scene3D(off_screen=True)
        scene.terrain(_dem(), big_data_threshold=SMALL)
        arrays = list(scene.mesh_of("terrain-1").point_data)
        scene.close()
        assert ELEVATION in arrays, arrays

    def test_the_scene_attribute_moves_the_budget_for_every_later_layer(self):
        """Set once, honoured by the builders after it — not passed layer by layer."""
        scene = Scene3D(off_screen=True)
        scene.big_data_threshold = SMALL
        scene.terrain(_dem(), name="a")
        scene.terrain(_dem(), name="b")
        cells = [scene.mesh_of("a").n_cells, scene.mesh_of("b").n_cells]
        scene.close()
        assert max(cells) <= SMALL, cells

    def test_a_per_call_override_does_not_leak_onto_the_scene(self):
        """The same keyword per call, and the map's own setting is left where it was."""
        scene = Scene3D(off_screen=True)
        scene.terrain(_dem(), big_data_threshold=SMALL)
        budget = scene.big_data_threshold
        scene.close()
        assert budget == DEFAULT_CELL_BUDGET

    def test_the_budget_the_layer_used_is_on_its_description(self):
        """Recorded as well as applied, so a figure read back reduces the same way."""
        scene = Scene3D(off_screen=True)
        scene.terrain(_dem(), big_data_threshold=SMALL)
        recorded = scene.get_layer("terrain-1").symbology.props["big_data_threshold"]
        scene.close()
        assert recorded == SMALL

    def test_a_figure_read_back_draws_the_same_reduced_mesh(self):
        """The round trip: the description carries the budget, so the redraw is not full resolution."""
        scene = Scene3D(off_screen=True)
        scene.terrain(_dem(), big_data_threshold=SMALL)
        again = Scene3D.from_figure(scene.figure_spec, off_screen=True)
        cells = again.mesh_of("terrain-1").n_cells
        scene.close()
        again.close()
        assert cells <= SMALL, cells

    def test_the_reduction_is_logged(self, caplog):
        """#207: predictable and reported, never magic — a caller must be able to see it happened.

        Args:
            caplog: Captures the line the reduction logs.
        """
        scene = Scene3D(off_screen=True)
        with caplog.at_level(logging.INFO):
            scene.terrain(_dem(), big_data_threshold=SMALL)
        scene.close()
        assert "terrain" in caplog.text, caplog.text


class TestABadBudgetIsRefused:
    """The shared guard, so a nonsensical cutoff answers the same way it does on the 2-D tiers."""

    def test_a_negative_budget_is_refused(self):
        """Below zero would reduce every layer, an empty one included."""
        scene = Scene3D(off_screen=True)
        with pytest.raises(ValueError, match="must not be negative"):
            scene.terrain(_dem(8), big_data_threshold=-1)
        scene.close()

    def test_a_fractional_budget_is_refused(self):
        """A cutoff is a count of cells, so it is not truncated into one."""
        scene = Scene3D(off_screen=True)
        with pytest.raises(ValueError, match="whole number"):
            scene.terrain(_dem(8), big_data_threshold=1.9)
        scene.close()

    def test_the_refusal_names_the_call(self):
        """The message points at the builder the keyword was written on, not at the guard."""
        scene = Scene3D(off_screen=True)
        with pytest.raises(ValueError, match="Scene3D.volume") as refusal:
            scene.volume(_cube(8), big_data_threshold=-1)
        scene.close()
        assert "volume" in refusal.value.args[0]


class TestAVolumeOverTheBudgetIsResampled:
    """The `resample` route — a different filter on a different class, so it is proved separately."""

    def test_the_drawn_grid_is_at_or_under_the_budget(self):
        """Read off the `ImageData` the plotter was handed."""
        scene = Scene3D(off_screen=True)
        scene.volume(_cube(), big_data_threshold=SMALL)
        cells = scene.mesh_of("volume-1").n_cells
        scene.close()
        assert cells <= SMALL, cells

    def test_it_really_was_reduced_rather_than_arriving_small(self):
        """The unreduced cube crosses the budget, so the test above is not vacuous."""
        scene = Scene3D(off_screen=True)
        scene.volume(_cube())
        cells = scene.mesh_of("volume-1").n_cells
        scene.close()
        assert cells > SMALL, cells

    def test_the_resampled_grid_still_carries_its_field(self):
        """`add_volume` colours by the field; a resample that dropped it would render nothing."""
        scene = Scene3D(off_screen=True)
        scene.volume(_cube(), big_data_threshold=SMALL)
        arrays = list(scene.mesh_of("volume-1").cell_data)
        scene.close()
        assert FIELD in arrays, arrays

    def test_a_volume_under_the_budget_keeps_every_cell(self):
        """Under the budget the cube reaches VTK exactly as it was built."""
        scene = Scene3D(off_screen=True)
        scene.volume(_cube(8))
        cells = scene.mesh_of("volume-1").n_cells
        scene.close()
        assert cells == 8**3


class TestAnIsosurfaceOverTheBudgetIsDecimated:
    """The third drawn kind with a route — shells extracted from a cube, then simplified."""

    def test_the_drawn_shell_is_at_or_under_the_budget(self):
        """The budget applies to what is drawn, which for this kind is the extracted surface."""
        scene = Scene3D(off_screen=True)
        scene.isosurface(_cube(), isosurfaces=[0.3], big_data_threshold=200)
        cells = scene.mesh_of("isosurface-1").n_cells
        scene.close()
        assert cells <= 200, cells

    def test_it_really_was_reduced_rather_than_arriving_small(self):
        """The unreduced shell crosses that budget."""
        scene = Scene3D(off_screen=True)
        scene.isosurface(_cube(), isosurfaces=[0.3])
        cells = scene.mesh_of("isosurface-1").n_cells
        scene.close()
        assert cells > 200, cells

    def test_the_decimated_shell_still_carries_its_field(self):
        """It is drawn with `scalars=FIELD`, so losing the array would raise rather than look wrong."""
        scene = Scene3D(off_screen=True)
        scene.isosurface(_cube(), isosurfaces=[0.3], big_data_threshold=200)
        arrays = list(scene.mesh_of("isosurface-1").point_data)
        scene.close()
        assert FIELD in arrays, arrays


class TestTheKindsWithNoRouteSaySo:
    """#207's other half: above the budget it must not hang silently. It does not reduce — it reports."""

    def test_a_point_cloud_over_the_budget_is_reported_by_name(self, caplog):
        """`decimate_pro` refuses a vertex-only mesh, so this kind has no route — and says which it is.

        Args:
            caplog: Captures the warning the report logs.
        """
        scene = Scene3D(off_screen=True)
        scene.big_data_threshold = 100
        with caplog.at_level(logging.WARNING):
            scene.point_cloud(_points())
        scene.close()
        assert "point_cloud" in caplog.text, caplog.text

    def test_a_point_cloud_over_the_budget_keeps_every_point(self):
        """Nothing is quietly dropped: an unreduced kind is drawn whole, and the report is the answer."""
        scene = Scene3D(off_screen=True)
        scene.big_data_threshold = 100
        scene.point_cloud(_points())
        points = scene.mesh_of("point_cloud-1").n_points
        scene.close()
        assert points == 5_000

    def test_the_report_names_the_count_and_the_budget(self, caplog):
        """A report a caller cannot act on is not a report; the two numbers are what they act on.

        Args:
            caplog: Captures the warning the report logs.
        """
        scene = Scene3D(off_screen=True)
        scene.big_data_threshold = 100
        with caplog.at_level(logging.WARNING):
            scene.point_cloud(_points())
        scene.close()
        assert "5,000" in caplog.text, caplog.text

    def test_an_extrusion_over_the_budget_is_reported(self, caplog):
        """Prisms combine into an `UnstructuredGrid`, which PyVista gives no `decimate_pro` at all.

        Args:
            caplog: Captures the warning the report logs.
        """
        scene = Scene3D(off_screen=True)
        scene.big_data_threshold = 10
        with caplog.at_level(logging.WARNING):
            scene.extruded_polygons(_squares(), height=2.0)
        scene.close()
        assert "extrusion" in caplog.text, caplog.text

    def test_a_vector_field_over_the_budget_is_reported(self, caplog):
        """Arrow glyphs are separate solids, not one sampled surface, so simplifying them is meaningless.

        Args:
            caplog: Captures the warning the report logs.
        """
        ax = np.linspace(0.0, 1.0, 12)
        xx, yy = np.meshgrid(ax, ax)
        pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
        scene = Scene3D(off_screen=True)
        scene.big_data_threshold = 10
        with caplog.at_level(logging.WARNING):
            scene.vectors(pts, np.tile([1.0, 0.0, 0.0], (len(pts), 1)), factor=0.1)
        scene.close()
        assert "vectors" in caplog.text, caplog.text

    def test_nothing_is_reported_under_the_budget(self, caplog):
        """The report is about crossing the budget, not about the kind having no route.

        Args:
            caplog: Captures anything the draw logs.
        """
        scene = Scene3D(off_screen=True)
        with caplog.at_level(logging.WARNING):
            scene.point_cloud(_points(10))
        scene.close()
        assert "big_data_threshold" not in caplog.text, caplog.text

    def test_a_kind_with_a_route_is_not_reported_as_unreducible(self, caplog):
        """Terrain reduces, so it must not also warn that it cannot be reduced.

        Args:
            caplog: Captures anything the draw logs at warning level.
        """
        scene = Scene3D(off_screen=True)
        with caplog.at_level(logging.WARNING):
            scene.terrain(_dem(), big_data_threshold=SMALL)
        scene.close()
        assert "no reduction" not in caplog.text, caplog.text

    def test_a_custom_mesh_over_the_budget_is_reported(self, caplog):
        """An object the caller built and handed over: reported, never reduced behind their back.

        Args:
            caplog: Captures the warning the report logs.
        """
        scene = Scene3D(off_screen=True)
        scene.big_data_threshold = 10
        with caplog.at_level(logging.WARNING):
            scene.add_mesh(pv.Sphere(theta_resolution=30, phi_resolution=30))
        scene.close()
        assert "custom:pyvista" in caplog.text, caplog.text

    def test_a_globe_over_the_budget_is_reported(self, caplog):
        """geovista owns the sphere's resolution, so the tier says so rather than flattening it.

        Args:
            caplog: Captures the warning the report logs.
        """
        pytest.importorskip("geovista")
        scene = Scene3D(off_screen=True)
        scene.big_data_threshold = 10
        with caplog.at_level(logging.WARNING):
            scene.globe(_dem(40), coastlines=False)
        scene.close()
        assert "raster" in caplog.text, caplog.text
