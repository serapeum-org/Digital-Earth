"""Tests for the D3.1 terrain renderer (digitalearth.three_d.TerrainMixin.terrain).

Gated on the optional ``3d`` extra (pyvista). Includes the orientation guard the plan calls for: a DEM with a
known high corner must render with its peak at the geographically-correct location (not mirrored/upside-down).
"""

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

pv = pytest.importorskip("pyvista")

from digitalearth.base.autostyle import auto_style
from digitalearth.base.sources import get_source
from digitalearth.three_d import Scene3D
from digitalearth.three_d.terrain import (
    _METRES_PER_DEGREE,
    ELEVATION,
    _terrain_mesh,
    _vertical_unit_scale,
)


def _named_raster(variable: str) -> Dataset:
    """Build a tiny geographic raster whose single band is named ``variable``.

    Args:
        variable: The band name the autostyle lookup keys off (e.g. ``"t2m"``).

    Returns:
        A pyramids ``Dataset`` carrying that band name.
    """
    dataset = Dataset.from_array(
        np.add.outer(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8)).astype(
            "float32"
        ),
        geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0), epsg=4326),
    )
    dataset.band_names = [variable]
    return dataset


def _record_add_mesh(scene: Scene3D) -> dict:
    """Wrap the plotter's ``add_mesh`` so the styling kwargs a builder passes can be inspected afterwards.

    The scene's own ``add_mesh`` is the entry point for a caller's own mesh (#293); a builder describes its
    layer and the renderer draws it, so what a builder styles is read where the renderer draws (#295).

    Args:
        scene: The scene to instrument (still renders normally).

    Returns:
        The dict the recorded kwargs land in.
    """
    captured: dict = {}
    original = scene.plotter.add_mesh

    def recorder(mesh, **kwargs):
        captured.update(kwargs)
        return original(mesh, **kwargs)

    scene.plotter.add_mesh = recorder
    return captured


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def test_terrain_registers_a_layer_and_renders():
    """terrain() adds one layer and produces a non-empty off-screen frame."""
    dem = np.add.outer(np.linspace(0.0, 1.0, 12), np.linspace(0.0, 1.0, 12))
    scene = Scene3D(off_screen=True)
    actor = scene.terrain(get_source(dem), z_exaggeration=3.0)
    assert actor is not None and len(scene.layers) == 1
    img = scene.screenshot()
    assert img.ndim == 3 and bool(img.any())
    scene.close()


def test_terrain_mesh_is_not_upside_down():
    """A DEM peak at the north-east corner stays at the north-east point of the mesh.

    Guards the VTK Fortran-order gotcha: C-order ravel silently mirrors the surface. x ascending, y descending
    (north→south, as pyramids hands rasters); z high at row 0 (north) / last col (east).
    """
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([30.0, 20.0, 10.0, 0.0])  # descending: first row = north
    z = np.zeros((4, 4))
    z[0, 3] = 100.0  # north-east peak → expect the elevation max at point (x=3, y=30)

    mesh = _terrain_mesh(z, x, y, vertical_scale=1.0)
    peak = int(np.argmax(mesh.point_data[ELEVATION]))
    px, py, _ = mesh.points[peak]
    assert (px, py) == (3.0, 30.0)


def test_vertical_scale_stretches_the_mesh():
    """A larger vertical_scale (the CRS unit conversion) produces a taller surface (bigger z extent)."""
    dem = np.add.outer(np.linspace(0.0, 10.0, 6), np.zeros(6))
    flat = _terrain_mesh(dem, np.arange(6.0), np.arange(6.0), vertical_scale=1.0)
    tall = _terrain_mesh(dem, np.arange(6.0), np.arange(6.0), vertical_scale=5.0)
    flat_h = flat.bounds[5] - flat.bounds[4]
    tall_h = tall.bounds[5] - tall.bounds[4]
    assert tall_h == pytest.approx(5.0 * flat_h)


def test_terrain_handles_nan_nodata():
    """NaN (masked nodata) cells do not crash the mesh build and are filled to the surface floor."""
    dem = np.add.outer(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    dem[0, 0] = np.nan
    mesh = _terrain_mesh(dem, np.arange(5.0), np.arange(5.0), vertical_scale=1.0)
    assert np.isfinite(mesh.points).all()  # geometry has no NaN coordinates


def test_vertical_unit_scale_geographic_vs_projected():
    """A geographic CRS rescales metre elevation into degrees; a projected CRS leaves it alone."""
    assert _vertical_unit_scale(4326) == pytest.approx(
        1.0 / _METRES_PER_DEGREE
    )  # WGS84 lon/lat
    assert _vertical_unit_scale(3857) == 1.0  # Web Mercator (metres)
    assert _vertical_unit_scale(None) == 1.0  # unknown CRS → no rescaling


@pytest.mark.parametrize(
    "spelling",
    [4326, "EPSG:4326", "epsg:4326", "+proj=longlat +datum=WGS84 +no_defs"],
)
def test_every_geographic_spelling_rescales(spelling):
    """Each spelling ``Source.crs`` may carry must rescale, not just the bare EPSG int.

    Args:
        spelling: One of the CRS forms the ``Source`` contract allows.

    Test scenario:
        The scale was read by a helper that did ``int(crs)``, so anything but the int raised, was swallowed as
        "unknown", and fell back to ``1.0`` — a ~111 000x vertical error that renders the DEM as the invisible
        needle this very module documents. A geographic CRS must rescale however it is spelled.
    """
    assert _vertical_unit_scale(spelling) == pytest.approx(1.0 / _METRES_PER_DEGREE), (
        f"{spelling!r} is geographic, so metre elevation must be divided by metres-per-degree"
    )


def test_geographic_dem_is_not_an_invisible_needle():
    """A geographic DEM (degrees x/y, metre z) builds relief comparable to its footprint, not a spike.

    Regression guard via the exact math terrain() composes (``_vertical_unit_scale`` x ``_terrain_mesh``): left
    unscaled the metre elevation dwarfs the ~degree-wide footprint ~100 000:1, so the surface is an invisible
    vertical needle. The geographic rescale must bring the relief into the footprint's order of magnitude.
    """
    # ~0.1 deg footprint, ~1000 m relief — the realistic geographic-DEM mismatch.
    x = np.linspace(-9.2, -9.1, 16)
    y = np.linspace(38.8, 38.7, 16)
    dem = np.add.outer(np.linspace(0.0, 1000.0, 16), np.zeros(16))

    scaled = _vertical_unit_scale(
        4326
    )  # the geographic CRS unit conversion terrain() bakes in
    mesh = _terrain_mesh(dem, x, y, vertical_scale=scaled)
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    horizontal = max(xmax - xmin, ymax - ymin)
    vertical = zmax - zmin
    # Relief is brought into the same order of magnitude as the footprint (not ~10000x taller).
    assert vertical < horizontal * 5.0

    # And without the rescale it really is a needle — proving the guard is meaningful.
    needle = _terrain_mesh(dem, x, y, vertical_scale=1.0)
    assert (needle.bounds[5] - needle.bounds[4]) > horizontal * 1000.0


class TestVerticalExaggeration:
    """Tests for terrain()'s z_exaggeration — a scene-wide view scale, not a factor baked into the points."""

    def test_it_is_not_multiplied_into_the_mesh_points(self):
        """The mesh keeps the elevation the data carries; only the view is stretched.

        Test scenario:
            Regression guard: `terrain()` used to multiply z into the `StructuredGrid`, so the exaggeration
            could not be read back, two layers could disagree about it, and changing it meant rebuilding the
            mesh. The built mesh's z extent must now match the raw relief, with the factor living on the view.
        """
        dem = np.add.outer(np.linspace(0.0, 10.0, 6), np.zeros(6))
        source = get_source(dem, x=np.arange(6.0), y=np.arange(6.0))
        scene = Scene3D(off_screen=True)
        try:
            scene.terrain(source, z_exaggeration=4.0)
            mesh, actor = scene.layers[0]
            assert (mesh.bounds[5] - mesh.bounds[4]) == pytest.approx(10.0), (
                "The exaggeration must not be baked into the mesh points"
            )
            assert actor.scale[2] == pytest.approx(4.0), (
                "The actor carries the exaggeration as a view scale"
            )
            assert scene.vertical_exaggeration == pytest.approx(4.0), (
                "The value must be readable back off the scene"
            )
        finally:
            scene.close()

    def test_the_rendered_scene_is_still_exaggerated(self):
        """Moving the factor to the view keeps the public parameter working: the scene really is taller.

        Test scenario:
            The renderer's bounds include the view transform, so they must show the z extent multiplied by the
            exaggeration even though the mesh itself was left alone.
        """
        dem = np.add.outer(np.linspace(0.0, 10.0, 6), np.zeros(6))
        source = get_source(dem, x=np.arange(6.0), y=np.arange(6.0))
        scene = Scene3D(off_screen=True)
        try:
            scene.terrain(source, z_exaggeration=4.0)
            bounds = scene.plotter.renderer.bounds
            assert (bounds[5] - bounds[4]) == pytest.approx(40.0), (
                "The rendered scene must be exaggerated by the factor asked for"
            )
        finally:
            scene.close()

    def test_one_value_governs_every_layer(self):
        """A second terrain added without an exaggeration inherits the scene's, instead of disagreeing.

        Test scenario:
            The second consequence of baking the factor in: each layer carried its own. A view scale applies
            to actors added later, so both terrains are drawn at the same exaggeration.
        """
        dem = np.add.outer(np.linspace(0.0, 10.0, 6), np.zeros(6))
        source = get_source(dem, x=np.arange(6.0), y=np.arange(6.0))
        scene = Scene3D(off_screen=True)
        try:
            scene.terrain(source, z_exaggeration=3.0)
            scene.terrain(source)
            assert scene.vertical_exaggeration == pytest.approx(3.0), (
                "A later layer with no exaggeration must leave the scene's value alone"
            )
            assert [actor.scale[2] for _, actor in scene.layers] == pytest.approx(
                [3.0, 3.0]
            ), "Both layers must render at the one scene exaggeration"
        finally:
            scene.close()


class TestTerrainColormap:
    """Tests for terrain()'s colormap resolution — the cross-tier autostyle lookup."""

    def test_an_unspecified_cmap_is_resolved_through_auto_style(self):
        """A named variable picks the same colormap the other three tiers give it.

        Test scenario:
            Regression guard for the tier divergence: `terrain()` hard-coded `cmap="terrain"`, so a `t2m`
            raster came out in `terrain` here and `coolwarm` on the static, interactive and web tiers - the
            same lookup those tiers run is `auto_style`, so its answer is what this must match.
        """
        dataset = _named_raster("t2m")
        scene = Scene3D(off_screen=True)
        captured = _record_add_mesh(scene)
        try:
            scene.terrain(dataset)
        finally:
            scene.close()
        expected = auto_style(get_source(dataset))["cmap"]
        assert expected == "coolwarm", (
            "auto_style is the cross-tier lookup; t2m resolves to coolwarm"
        )
        assert captured["cmap"] == expected, (
            f"terrain() must colour a {dataset.band_names[0]} raster the way every other tier does"
        )

    def test_an_explicit_cmap_still_wins(self):
        """A caller-supplied colormap overrides the lookup, as on every other tier.

        Test scenario:
            The autostyle default only fills in for `cmap=None`; an explicit name must reach `add_mesh`
            untouched even when the variable is one `auto_style` recognises.
        """
        dataset = _named_raster("t2m")
        scene = Scene3D(off_screen=True)
        captured = _record_add_mesh(scene)
        try:
            scene.terrain(dataset, cmap="magma")
        finally:
            scene.close()
        assert captured["cmap"] == "magma", (
            "An explicit cmap must not be overridden by the lookup"
        )

    def test_the_terrain_literal_sits_behind_the_lookup(self, monkeypatch):
        """The tier's own `"terrain"` default is the last resort, reached only when the lookup yields nothing.

        Args:
            monkeypatch: Empties the autostyle answer so the fallback is reachable.

        Test scenario:
            `auto_style` normally always returns a colormap, so the literal is exercised by making the lookup
            come back empty - proving it is consulted after `auto_style`, not ahead of it.
        """
        monkeypatch.setattr("digitalearth.base.autostyle.auto_style", lambda source: {})
        dataset = _named_raster("t2m")
        scene = Scene3D(off_screen=True)
        captured = _record_add_mesh(scene)
        try:
            scene.terrain(dataset)
        finally:
            scene.close()
        assert captured["cmap"] == "terrain", (
            "The tier fallback applies only when the lookup yields none"
        )
