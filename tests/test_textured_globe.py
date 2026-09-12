"""Tests for digitalearth.static.textured_globe — the pyramids/cleopatra 3-D globe seam.

The cleopatra glyph is already tested upstream, so these cover the half Digital-Earth owns: turning geodata
into an equirectangular texture at the right lon/lat, and mapping lon/lat back onto the drawn sphere.
"""

import copy
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from cleopatra.styling.colors import resolve_colormap
from matplotlib import colors as mcolors
from matplotlib.animation import PillowWriter
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pyramids.dataset import Dataset, GeoReference
from pyramids.feature import FeatureCollection
from shapely.geometry import Point, Polygon

from digitalearth.static import TexturedGlobe
from digitalearth.static.textured_globe import (
    _clip_ring,
    _limb_arc,
    _limb_point,
    _texture_axes,
)


def _drawn(scatter, values: str = "sizes") -> np.ndarray:
    """Render the scatter's figure and return the values its painted markers were drawn with.

    A globe scatter keeps every point and paints the far-side ones at size zero, then puts the caller's
    sizes back once the render is over. What each visible marker was drawn with is therefore its own
    entry, taken for the points the render actually showed.

    Args:
        scatter: A scatter returned by ``TexturedGlobe.points``.
        values: ``"sizes"`` for marker sizes, ``"widths"`` for marker edge widths.

    Returns:
        np.ndarray: one value per painted marker, in the order the points were given.
    """
    scatter.axes.get_figure().canvas.draw()
    asked = scatter.get_sizes() if values == "sizes" else scatter.get_linewidths()
    per_point = np.resize(
        np.atleast_1d(np.asarray(asked, dtype=float)), scatter._shown.size
    )
    return per_point[scatter._shown]


@pytest.fixture(scope="module")
def flat_texture() -> np.ndarray:
    """A cheap two-tone equirectangular texture (north blue, south ochre)."""
    texture = np.zeros((90, 180, 3), dtype=np.uint8)
    texture[:45] = (40, 90, 180)
    texture[45:] = (180, 120, 40)
    return texture


@pytest.fixture
def globe(flat_texture) -> TexturedGlobe:
    """A low-resolution globe — the mesh is the render-cost driver, so keep it small in tests."""
    return TexturedGlobe(flat_texture, n_lon=24, n_lat=12)


@pytest.fixture(scope="module")
def points_fc() -> FeatureCollection:
    """The committed point fixture, as a pyramids FeatureCollection."""
    return FeatureCollection.read_file("tests/data/points.geojson")


class TestTextureAxes:
    """The texture's lon/lat convention must match the glyph's documented layout."""

    def test_rows_run_north_to_south(self):
        lat, _ = _texture_axes(5, 9)
        assert lat[0] == 90.0, f"row 0 should be +90, got {lat[0]}"
        assert lat[-1] == -90.0, f"the last row should be -90, got {lat[-1]}"

    def test_columns_run_west_to_east(self):
        _, lon = _texture_axes(5, 9)
        assert lon[0] == -180.0, f"column 0 should be -180, got {lon[0]}"
        assert lon[-1] == 180.0, f"the last column should be +180, got {lon[-1]}"


class TestFromDataset:
    """A raster is draped where it actually is, not stretched over the sphere."""

    def test_texture_is_a_global_rgba_canvas(self, dataset):
        globe = TexturedGlobe.from_dataset(dataset, shape=(360, 720))
        assert globe.glyph.texture.shape == (360, 720, 4)

    def test_drape_lands_on_the_datasets_own_lonlat_footprint(self, dataset):
        """The opaque cells must coincide with pyramids' own reprojected bounds."""
        shape = (1440, 2880)
        globe = TexturedGlobe.from_dataset(dataset, shape=shape)
        alpha = globe.glyph.texture[..., 3]
        rows, cols = np.nonzero(alpha > 0)
        assert rows.size, "the dataset should drape onto some cells"
        lat_axis, lon_axis = _texture_axes(*shape)
        lon, lat = lon_axis[cols], lat_axis[rows]

        lonlat = dataset.to_crs(4326)
        src_lon, src_lat = (
            np.asarray(lonlat.x, dtype=float),
            np.asarray(lonlat.y, dtype=float),
        )
        tol = 0.25  # a canvas cell (0.125 deg) plus the source's own half-cell
        assert src_lon.min() - tol <= lon.min(), (
            f"drape starts west of the source: {lon.min()}"
        )
        assert lon.max() <= src_lon.max() + tol, (
            f"drape ends east of the source: {lon.max()}"
        )
        assert src_lat.min() - tol <= lat.min(), (
            f"drape starts south of the source: {lat.min()}"
        )
        assert lat.max() <= src_lat.max() + tol, (
            f"drape ends north of the source: {lat.max()}"
        )

    def test_a_0_360_longitude_axis_drapes_the_whole_world(self):
        """0-360 is the usual climate/NWP convention; treating it as -180..180 loses half the globe.

        The band ramps with longitude rather than being constant, so a drape that covers the globe but is
        rotated by 180 degrees -- which a constant array cannot tell apart -- fails too.
        """
        lon = np.linspace(0.0, 359.0, 360, dtype="float32")
        arr = np.repeat(lon[None, :], 180, axis=0)
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 90.0, 0.0, -1.0), epsg=4326),
        )
        globe = TexturedGlobe.from_dataset(
            ds, cmap="viridis", vmin=0.0, vmax=359.0, shape=(180, 360)
        )
        texture = globe.glyph.texture
        assert (texture[..., 3] > 0).mean() == pytest.approx(1.0), (
            "should cover the whole globe"
        )

        # Each canvas longitude must carry the source value for that same place on Earth. The source is
        # stored on 0-360, so the equivalent source longitude is the canvas longitude modulo a full turn,
        # and the band's value at that longitude is the longitude itself.
        _, lon_axis = _texture_axes(*texture.shape[:2])
        cmap, norm = resolve_colormap("viridis"), Normalize(vmin=0.0, vmax=359.0)
        for col in (0, 90, 180, 270, 359):
            source_lon = lon_axis[col] % 360.0
            expected = cmap(norm(np.floor(source_lon)))
            assert np.allclose(texture[90, col, :3], expected[:3], atol=2.0 / 255), (
                f"canvas lon {lon_axis[col]:.2f} should carry source lon {source_lon:.2f}, not a rotated one"
            )

    def test_a_minus180_longitude_axis_still_drapes_the_whole_world(self):
        """The conventional frame must be unaffected by the 0-360 handling."""
        arr = np.ones((180, 360), dtype="float32")
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0), epsg=4326),
        )
        globe = TexturedGlobe.from_dataset(ds, shape=(180, 360))
        assert (globe.glyph.texture[..., 3] > 0).mean() == pytest.approx(1.0)

    def test_a_raster_beyond_the_antimeridian_lands_west(self):
        """Longitudes 200-210 in the 0-360 frame are -160..-150 on the canvas, not off the edge."""
        arr = np.ones((10, 10), dtype="float32")
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(200.0, 1.0, 0.0, 10.0, 0.0, -1.0), epsg=4326),
        )
        globe = TexturedGlobe.from_dataset(ds, shape=(720, 1440))
        opaque = globe.glyph.texture[..., 3] > 0
        assert opaque.any(), "a raster stored beyond the antimeridian must still drape"
        _, lon_axis = _texture_axes(*opaque.shape)
        lon = lon_axis[np.nonzero(opaque.any(axis=0))[0]]
        assert -161.0 <= lon.min(), f"drape starts too far west: {lon.min()}"
        assert lon.max() <= -149.0, f"drape ends too far east: {lon.max()}"

    def test_a_single_row_raster_drapes(self):
        """One row has no latitude spacing of its own; borrowing the column spacing keeps it visible."""
        arr = np.ones((1, 8), dtype="float32")
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(0.0, 5.0, 0.0, 10.0, 0.0, -5.0), epsg=4326),
        )
        globe = TexturedGlobe.from_dataset(ds, shape=(180, 360))
        assert (globe.glyph.texture[..., 3] > 0).any(), (
            "a single-row raster should still drape"
        )

    def test_a_single_column_raster_drapes(self):
        arr = np.ones((8, 1), dtype="float32")
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(0.0, 5.0, 0.0, 40.0, 0.0, -5.0), epsg=4326),
        )
        globe = TexturedGlobe.from_dataset(ds, shape=(180, 360))
        assert (globe.glyph.texture[..., 3] > 0).any(), (
            "a single-column raster should still drape"
        )

    def test_the_poles_and_the_antimeridian_are_not_left_empty(self):
        """Outer centres on the source's boundary returned nothing: a pole hole and an antimeridian seam."""
        arr = np.ones((180, 360), dtype="float32")
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0), epsg=4326),
        )
        opaque = (
            TexturedGlobe.from_dataset(ds, shape=(180, 360)).glyph.texture[..., 3] > 0
        )
        assert opaque[0].all(), "the north-pole row should be filled"
        assert opaque[-1].all(), (
            "the south-pole row should be filled, not left as a hole"
        )
        assert opaque[:, 0].all(), "the -180 column should be filled"
        assert opaque[:, -1].all(), (
            "the +180 column should be filled, not left as a seam"
        )

    def test_a_genuine_nodata_edge_stays_transparent(self):
        """The earlier edge-clamp filled this in, fabricating data that is not in the raster."""
        arr = np.ones((180, 360), dtype="float32")
        arr[-1, :] = -9999.0
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0), epsg=4326),
            no_data_value=-9999.0,
        )
        opaque = (
            TexturedGlobe.from_dataset(ds, shape=(180, 360)).glyph.texture[..., 3] > 0
        )
        assert not opaque[-1].any(), (
            "a nodata southern row must stay transparent, not be filled from above"
        )
        assert opaque[-2].any(), "the row above it is real data and should still drape"

    def test_a_regional_raster_keeps_its_transparent_margin(self, dataset):
        """Nothing may spread a regional raster to the poles."""
        opaque = (
            TexturedGlobe.from_dataset(dataset, shape=(360, 720)).glyph.texture[..., 3]
            > 0
        )
        assert opaque.mean() < 0.01, (
            f"a small raster should stay small, covered {opaque.mean():.4f}"
        )
        assert not opaque[-1].any(), (
            "the south-pole row should stay empty for a regional raster"
        )

    def test_uncovered_cells_stay_transparent(self, dataset):
        """A regional raster leaves the rest of the globe see-through rather than filling it."""
        alpha = TexturedGlobe.from_dataset(dataset, shape=(360, 720)).glyph.texture[
            ..., 3
        ]
        assert (alpha == 0).sum() > alpha.size * 0.9

    def test_nodata_cells_are_transparent(self):
        """The nodata cell must be transparent while its neighbours are opaque, at its own lon/lat."""
        arr = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype="float32")
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(0.0, 10.0, 0.0, 20.0, 0.0, -10.0), epsg=4326),
            no_data_value=-9999.0,
        )
        globe = TexturedGlobe.from_dataset(ds, shape=(720, 1440))
        alpha = globe.glyph.texture[..., 3]
        lat_axis, lon_axis = _texture_axes(*alpha.shape)

        def alpha_at(lon: float, lat: float) -> float:
            return float(
                alpha[
                    int(np.abs(lat_axis - lat).argmin()),
                    int(np.abs(lon_axis - lon).argmin()),
                ]
            )

        assert alpha_at(5.0, 15.0) > 0, "the top-left valid cell should be opaque"
        assert alpha_at(15.0, 15.0) == 0, (
            "the top-right cell is nodata and must be transparent"
        )
        assert alpha_at(5.0, 5.0) > 0, "the bottom-left valid cell should be opaque"
        assert alpha_at(15.0, 5.0) > 0, "the bottom-right valid cell should be opaque"

    def test_nodata_is_transparent_even_under_an_opaque_bad_colour(self):
        """matplotlib's default 'bad' colour is already transparent, so only an opaque one tests our own
        masking. A colormap with set_bad opaque must still leave nodata see-through."""
        cmap = resolve_colormap("viridis").copy()
        cmap.set_bad("red", alpha=1.0)
        arr = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype="float32")
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(0.0, 10.0, 0.0, 20.0, 0.0, -10.0), epsg=4326),
            no_data_value=-9999.0,
        )
        globe = TexturedGlobe.from_dataset(ds, cmap=cmap, shape=(720, 1440))
        alpha = globe.glyph.texture[..., 3]
        lat_axis, lon_axis = _texture_axes(*alpha.shape)
        r = int(np.abs(lat_axis - 15.0).argmin())
        c = int(np.abs(lon_axis - 15.0).argmin())
        assert alpha[r, c] == 0, (
            "the nodata cell must be transparent whatever the colormap's bad colour is"
        )

    def test_the_drape_is_not_transposed_or_shifted(self):
        """Pin the orientation: a distinctive value must land at its own lon/lat, not a mirrored one."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(0.0, 10.0, 0.0, 20.0, 0.0, -10.0), epsg=4326),
        )
        globe = TexturedGlobe.from_dataset(
            ds, cmap="viridis", vmin=1.0, vmax=4.0, shape=(720, 1440)
        )
        texture = globe.glyph.texture
        lat_axis, lon_axis = _texture_axes(*texture.shape[:2])
        expected = resolve_colormap("viridis")(Normalize(vmin=1.0, vmax=4.0)(arr))

        for row, lat in enumerate((15.0, 5.0)):
            for col, lon in enumerate((5.0, 15.0)):
                r = int(np.abs(lat_axis - lat).argmin())
                c = int(np.abs(lon_axis - lon).argmin())
                # The texture is 8-bit, so one colour step (1/255) is the tightest meaningful tolerance.
                assert np.allclose(
                    texture[r, c, :3], expected[row, col, :3], atol=1.0 / 255
                ), f"cell ({row}, {col}) should appear at lon {lon}, lat {lat}"

    def test_a_dataset_without_a_crs_is_refused(self, dataset, monkeypatch):
        """Without a CRS there is no way to place the raster, so fail loudly instead of guessing 4326."""
        monkeypatch.setattr(type(dataset), "epsg", property(lambda self: None))
        monkeypatch.setattr(type(dataset), "crs", property(lambda self: None))
        with pytest.raises(ValueError, match="has none"):
            TexturedGlobe.from_dataset(dataset)

    def test_a_dataset_whose_crs_has_no_epsg_code_is_accepted(
        self, dataset, monkeypatch
    ):
        """`.epsg` is None for geostationary/Mollweide too; those reproject fine and must not be rejected."""
        monkeypatch.setattr(type(dataset), "epsg", property(lambda self: None))
        globe = TexturedGlobe.from_dataset(dataset, n_lon=2880, n_lat=1440)
        assert (globe.glyph.texture[..., 3] > 0).any(), (
            "a CRS without an EPSG code should still drape"
        )

    def test_a_none_colormap_falls_back_to_viridis(self, dataset):
        """resolve_colormap returns None only for cmap=None, which must still produce a texture."""
        globe = TexturedGlobe.from_dataset(dataset, cmap=None, n_lon=2880, n_lat=1440)
        fallback = TexturedGlobe.from_dataset(
            dataset, cmap="viridis", n_lon=2880, n_lat=1440
        )
        assert np.array_equal(globe.glyph.texture, fallback.glyph.texture), (
            "cmap=None should fall back to viridis, not to some other colormap"
        )

    def test_reversed_colour_bounds_are_refused(self, dataset):
        """vmax <= vmin was silently discarded and replaced, hiding the caller's mistake."""
        with pytest.raises(ValueError, match="vmax must be greater than vmin"):
            TexturedGlobe.from_dataset(dataset, vmin=10.0, vmax=1.0)

    @pytest.mark.parametrize("band", [0, 2, -1])
    def test_an_out_of_range_band_is_refused(self, dataset, band):
        """The single-band shortcut skips select_bands, which used to be what validated the index."""
        with pytest.raises(ValueError, match="out of range"):
            TexturedGlobe.from_dataset(dataset, band=band, shape=(180, 360))

    @pytest.mark.parametrize("band, value", [(1, 1.0), (2, 2.0), (3, 3.0)])
    def test_the_requested_band_is_the_one_drawn(self, band, value):
        """Selecting a band before the warp must not change which band ends up on the globe."""
        arr = np.stack([np.full((40, 40), v, dtype="float32") for v in (1.0, 2.0, 3.0)])
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(0.0, 0.25, 0.0, 10.0, 0.0, -0.25), epsg=4326),
        )
        globe = TexturedGlobe.from_dataset(
            ds, band=band, cmap="viridis", vmin=1.0, vmax=3.0, n_lon=2880, n_lat=1440
        )
        texture = globe.glyph.texture
        opaque = texture[..., 3] > 0
        assert opaque.any(), f"band {band} should drape"
        expected = resolve_colormap("viridis")(Normalize(vmin=1.0, vmax=3.0)(value))
        assert np.allclose(texture[opaque][0][:3], expected[:3], atol=2.0 / 255), (
            f"band {band} should carry value {value}"
        )

    @pytest.mark.parametrize("kwargs", [{"vmin": float("nan")}, {"vmax": float("inf")}])
    def test_a_non_finite_bound_is_refused(self, dataset, kwargs):
        """vmin=nan slipped through Normalize and painted the whole globe the colormap's bad colour."""
        with pytest.raises(ValueError, match="must be a finite number"):
            TexturedGlobe.from_dataset(dataset, **kwargs)

    @pytest.mark.parametrize("kwargs", [{"vmin": 99.0}, {"vmax": -99.0}])
    def test_a_lone_bound_that_inverts_the_range_is_refused(self, kwargs):
        """One bound on the wrong side of the data leaves no range, just as passing both reversed does."""
        arr = np.arange(16, dtype="float32").reshape(4, 4)
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326),
        )
        with pytest.raises(ValueError, match="no range to colour"):
            TexturedGlobe.from_dataset(ds, **kwargs)

    def test_an_all_nodata_band_warns(self):
        """A fully transparent globe is indistinguishable from a broken one, so say which it is."""
        arr = np.full((4, 4), -9999.0, dtype="float32")
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326),
            no_data_value=-9999.0,
        )
        with pytest.warns(RuntimeWarning, match="fully transparent"):
            TexturedGlobe.from_dataset(ds)

    @pytest.mark.parametrize("shape", [(1, 2, 3), ("a", "b"), 5, None])
    def test_a_malformed_shape_is_refused_clearly(self, dataset, shape):
        """A 3-tuple used to fail with 'too many values to unpack', which names nothing useful."""
        with pytest.raises(ValueError, match="rows, columns"):
            TexturedGlobe.from_dataset(dataset, shape=shape)

    def test_a_degenerate_shape_is_refused(self, dataset):
        with pytest.raises(ValueError, match="at least"):
            TexturedGlobe.from_dataset(dataset, shape=(1, 1))

    def test_too_coarse_a_canvas_warns_instead_of_rendering_blank(self, dataset):
        with pytest.warns(RuntimeWarning, match="nothing was draped"):
            TexturedGlobe.from_dataset(dataset, shape=(90, 180))

    def test_data_finer_than_the_mesh_warns(self, dataset):
        """A fine texture is not enough: the glyph samples down to the mesh, so sub-mesh data vanishes."""
        with pytest.warns(RuntimeWarning, match="finer than the .* sphere mesh"):
            TexturedGlobe.from_dataset(dataset, n_lon=180, n_lat=90)

    def test_a_mesh_that_resolves_the_data_does_not_warn(self, dataset):
        """The warning must be about visibility, not merely about the data being small."""
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            TexturedGlobe.from_dataset(dataset, n_lon=2880, n_lat=1440)
        mesh_warnings = [w for w in caught if "sphere mesh" in str(w.message)]
        assert not mesh_warnings, (
            f"a mesh that resolves the data should not warn: {mesh_warnings}"
        )

    def test_the_mesh_warning_predicts_what_actually_paints(self, dataset):
        """Tie the warning to reality: when it fires, the drawn sphere really does carry no opaque face."""
        with pytest.warns(RuntimeWarning, match="sphere mesh"):
            globe = TexturedGlobe.from_dataset(dataset, n_lon=180, n_lat=90)
        globe.draw()
        painted = np.asarray(globe.glyph._facecolors)
        assert int((painted[..., 3] > 0).sum()) == 0, (
            "the warning fired but the data did paint"
        )

    def test_the_mesh_warning_matches_the_render_for_many_placements(self):
        """One fixture can agree by luck. Sweep placements and sizes and require exact agreement.

        The first version of this guard sampled the mesh's vertex grid while the glyph samples its face
        centres; that disagreed with the real render for about one small raster in eight, in both directions
        — silent blank globes and false alarms. A single-case test did not notice.
        """
        rng = np.random.default_rng(1337)
        disagreements = []
        for _ in range(60):
            lon0 = float(rng.uniform(-170.0, 160.0))
            lat0 = float(rng.uniform(-70.0, 70.0))
            size = float(rng.uniform(0.2, 4.0))
            ds = Dataset.from_array(
                arr=np.ones((4, 4), dtype="float32"),
                geo_ref=GeoReference(
                    geo=(lon0, size / 4, 0.0, lat0, 0.0, -size / 4), epsg=4326
                ),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                globe = TexturedGlobe.from_dataset(ds, n_lon=180, n_lat=90)
            warned = any("sphere mesh" in str(w.message) for w in caught)
            globe.draw()
            paints = bool((np.asarray(globe.glyph._facecolors)[..., 3] > 0).any())
            if warned == paints:  # warned yet painted, or silent yet blank
                disagreements.append(
                    (round(lon0, 2), round(lat0, 2), round(size, 2), warned, paints)
                )
            globe.close()
        assert not disagreements, (
            f"the warning disagreed with the render at {disagreements[:5]}"
        )

    def test_a_constant_band_does_not_divide_by_zero(self):
        """vmin == vmax has no range to normalise against; it must still produce a texture."""
        arr = np.full((4, 4), 5.0, dtype="float32")
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326),
        )
        globe = TexturedGlobe.from_dataset(ds, shape=(180, 360))
        opaque = globe.glyph.texture[..., 3] > 0
        assert opaque.any(), "a constant band should still drape"
        rgb = globe.glyph.texture[opaque][:, :3]
        assert np.allclose(rgb, rgb[0]), (
            "a constant band should map to one colour, not a gradient"
        )


class TestFromProvider:
    """Tile-basemap textures. The network call is cleopatra's; what is ours is the kwarg split."""

    @pytest.fixture
    def fetched(self, monkeypatch):
        """Capture the world_texture call instead of pulling thousands of tiles."""
        seen = {}

        def _fake(provider=None, **kwargs):
            seen["provider"], seen["kwargs"] = provider, kwargs
            return np.zeros((16, 32, 3), dtype=np.uint8)

        monkeypatch.setattr("digitalearth.static.textured_globe.world_texture", _fake)
        return seen

    def test_defaults_to_a_bulk_permitting_provider(self, fetched):
        """OpenStreetMap forbids whole-world fetches, so the default must not be it."""
        TexturedGlobe.from_provider()
        assert fetched["provider"] == "Esri.WorldImagery"

    def test_texture_options_go_to_the_fetcher(self, fetched):
        TexturedGlobe.from_provider(
            "Esri.WorldImagery",
            zoom=3,
            texture_n_lon=720,
            texture_n_lat=360,
            cache=False,
        )
        assert fetched["kwargs"] == {
            "zoom": 3,
            "n_lon": 720,
            "n_lat": 360,
            "cache": False,
        }

    def test_glyph_options_do_not_leak_into_the_fetcher(self, fetched):
        TexturedGlobe.from_provider(zoom=2, tilt_deg=0.0)
        assert "tilt_deg" not in fetched["kwargs"]

    def test_n_lon_means_the_mesh_here_as_everywhere_else(self, fetched):
        """The same keyword must not mean the mesh in one constructor and the texture in another."""
        globe = TexturedGlobe.from_provider(texture_n_lon=720, n_lon=48, n_lat=24)
        assert fetched["kwargs"]["n_lon"] == 720, (
            "texture_n_lon should size the fetched grid"
        )
        assert globe.glyph.n_lon == 48, "n_lon should size the sphere mesh"
        assert globe.glyph.n_lat == 24, "n_lat should size the sphere mesh"

    def test_mesh_size_is_not_leaked_to_the_fetcher(self, fetched):
        TexturedGlobe.from_provider(n_lon=48, n_lat=24)
        assert "n_lon" not in fetched["kwargs"], (
            "the mesh size must not reach the fetcher"
        )
        assert "n_lat" not in fetched["kwargs"], (
            "the mesh size must not reach the fetcher"
        )


class TestProject:
    """lon/lat -> sphere must agree with the glyph's own mesh, tilt included."""

    def test_points_land_on_the_unit_sphere(self, globe):
        world = globe.project([0.0, 45.0, -120.0], [0.0, 30.0, -60.0])
        assert np.allclose(np.linalg.norm(world, axis=1), 1.0)

    def test_altitude_lifts_points_off_the_surface(self, globe):
        lifted = globe.project(10.0, 20.0, altitude=0.05)
        assert np.allclose(np.linalg.norm(lifted, axis=1), 1.05)

    def test_untilted_cardinal_points_are_exact(self, flat_texture):
        """With no tilt the body frame is the world frame, so the maths is checkable by hand."""
        globe = TexturedGlobe(flat_texture, tilt_deg=0.0, n_lon=8, n_lat=4)
        assert np.allclose(globe.project(0.0, 0.0), [[1.0, 0.0, 0.0]], atol=1e-9)
        assert np.allclose(globe.project(90.0, 0.0), [[0.0, 1.0, 0.0]], atol=1e-9)
        assert np.allclose(globe.project(0.0, 90.0), [[0.0, 0.0, 1.0]], atol=1e-9)

    def test_projection_matches_the_glyphs_own_transform(self, globe):
        """The overlay must use the glyph's transform, not a re-derived one, or it drifts from the surface."""
        lon, lat = 33.0, -12.0
        expected = globe.glyph.transform(
            np.array(
                [
                    [
                        np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon)),
                        np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon)),
                        np.sin(np.deg2rad(lat)),
                    ]
                ]
            ),
            spin=40.0,
        )
        assert np.allclose(globe.project(lon, lat, spin=40.0), expected)

    def test_spin_moves_the_point(self, globe):
        assert not np.allclose(
            globe.project(0.0, 0.0, spin=0.0), globe.project(0.0, 0.0, spin=90.0)
        )

    def test_projection_defaults_to_the_drawn_spin(self, globe):
        """spin=0 by default silently placed overlays on a face the reader could not see."""
        globe.draw(spin=90.0)
        assert np.allclose(
            globe.project(10.0, 20.0), globe.project(10.0, 20.0, spin=90.0)
        ), "project() should default to the spin the globe was drawn at"

    def test_an_explicit_spin_still_wins(self, globe):
        globe.draw(spin=90.0)
        assert np.allclose(
            globe.project(10.0, 20.0, spin=0.0),
            TexturedGlobe(
                globe.glyph.texture, tilt_deg=globe.glyph.tilt_deg, n_lon=8, n_lat=4
            ).project(10.0, 20.0, spin=0.0),
        )

    def test_mismatched_lon_lat_shapes_are_refused(self, globe):
        with pytest.raises(ValueError, match="same shape"):
            globe.project([0.0, 1.0], [0.0])


class TestVisibility:
    """Far-side culling keeps overlays from being drawn through the sphere."""

    def test_visibility_needs_a_drawn_globe(self, globe):
        with pytest.raises(RuntimeError, match="draw"):
            globe.visible(np.array([[1.0, 0.0, 0.0]]))

    def test_the_camera_facing_hemisphere_is_visible(self, globe):
        globe.draw(elev=0.0, azim=0.0)
        near, far = np.array([[1.0, 0.0, 0.0]]), np.array([[-1.0, 0.0, 0.0]])
        assert globe.visible(near)[0], "the camera-facing point should be visible"
        assert not globe.visible(far)[0], "the point behind the globe should not be"


class TestPoints:
    """Vector overlays."""

    def test_points_need_a_drawn_globe(self, globe):
        with pytest.raises(RuntimeError, match="draw"):
            globe.points([0.0], lat=[0.0])

    def test_lon_lat_points_are_scattered(self, globe):
        globe.draw(elev=0.0, azim=0.0)
        assert globe.points([0.0, 10.0], lat=[0.0, 5.0]) is not None

    def test_far_side_points_are_dropped(self, globe):
        """A point behind the globe must not be drawn through it."""
        globe.draw(elev=0.0, azim=0.0)
        collection = globe.points([180.0], lat=[0.0], hide_far_side=True)
        assert _drawn(collection).size == 0, "a far-side marker should not be drawn"

    def test_far_side_points_are_kept_when_asked(self, globe):
        globe.draw(elev=0.0, azim=0.0)
        collection = globe.points([180.0], lat=[0.0], hide_far_side=False)
        assert _drawn(collection).size == 1, "hide_far_side=False should draw it"

    def test_per_point_colours_stay_with_their_points(self, globe):
        """The colour array is never cut down to the visible points, so it cannot fall out of step."""
        globe.draw(elev=0.0, azim=0.0)
        collection = globe.points(
            [0.0, 180.0], lat=[0.0, 0.0], c=[1.0, 2.0], hide_far_side=True
        )
        assert _drawn(collection).size == 1, "only the near-side marker should be drawn"
        assert list(collection.get_array()) == [1.0, 2.0], (
            f"the colour values should stay whole, got {list(collection.get_array())}"
        )

    @pytest.mark.parametrize(
        "key, values, expected",
        [
            pytest.param("s", [10.0, 20.0, 30.0, 40.0], [10.0, 30.0, 40.0], id="s"),
            pytest.param(
                "linewidths", [1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 4.0], id="linewidths"
            ),
        ],
    )
    def test_the_hidden_point_is_the_one_masked(self, globe, key, values, expected):
        """The far-side mask zeroes the hidden point's own value, not a neighbour's.

        Args:
            key: The per-point argument under test.
            values: One value per point; the second point is on the far side.
            expected: The values that must survive.

        Test scenario:
            Compared as a multiset, since matplotlib reorders markers by depth: if the mask landed on the
            wrong point, the hidden value would survive and a visible one would vanish.
        """
        globe.draw(elev=0.0, azim=0.0)
        collection = globe.points(
            [0.0, 180.0, 10.0, 20.0], lat=[0.0] * 4, **{key: values}
        )
        kept = list(_drawn(collection, "sizes" if key == "s" else "widths"))
        assert kept == expected, f"{key} should keep {expected}, got {kept}"

    @pytest.mark.parametrize(
        "colour",
        [[1.0, 0.0, 0.0, 1.0], (1.0, 0.0, 0.0, 1.0), np.array([1.0, 0.0, 0.0, 1.0])],
        ids=["list", "tuple", "ndarray"],
    )
    def test_a_single_rgba_colour_is_one_colour_whatever_its_container(
        self, globe, colour
    ):
        """A 4-element red is one colour for every point, not four values to map, in any container."""
        globe.draw(elev=0.0, azim=0.0)
        collection = globe.points([0.0, 180.0, 10.0, 20.0], lat=[0.0] * 4, color=colour)
        assert np.allclose(collection.get_facecolors()[0], [1.0, 0.0, 0.0, 1.0]), (
            f"expected red, got {collection.get_facecolors()[0]}"
        )

    def test_a_length_matching_c_is_value_mapped_whole(self, globe):
        """matplotlib value-maps a `c` whose length matches the point count; all four values survive."""
        globe.draw(elev=0.0, azim=0.0)
        collection = globe.points(
            [0.0, 180.0, 10.0, 20.0], lat=[0.0] * 4, c=(0.1, 0.2, 0.3, 0.4)
        )
        assert list(np.round(collection.get_array(), 2)) == [0.1, 0.2, 0.3, 0.4], (
            f"c should be kept whole, got {list(collection.get_array())}"
        )

    def test_a_pandas_series_colours_every_point(self, globe):
        """Colours often come straight from a dataframe column; the scale spans the whole column."""
        pd = pytest.importorskip("pandas")
        globe.draw(elev=0.0, azim=0.0)
        collection = globe.points(
            [0.0, 180.0, 10.0, 20.0], lat=[0.0] * 4, c=pd.Series([1.0, 2.0, 3.0, 4.0])
        )
        collection.axes.get_figure().canvas.draw()
        assert (collection.norm.vmin, collection.norm.vmax) == (1.0, 4.0), (
            f"the colour scale should span the column, got "
            f"{(collection.norm.vmin, collection.norm.vmax)}"
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"color": "red"}, id="scalar-name"),
            pytest.param({"s": 30}, id="scalar-size"),
        ],
    )
    def test_scalar_arguments_apply_to_every_near_side_point(self, globe, kwargs):
        """A single colour or size applies to every point; only the far-side one goes undrawn."""
        globe.draw(elev=0.0, azim=0.0)
        collection = globe.points([0.0, 180.0, 10.0, 20.0], lat=[0.0] * 4, **kwargs)
        assert _drawn(collection).size == 3, (
            f"{kwargs} should still draw the 3 near-side points"
        )

    def test_a_feature_collection_is_accepted(self, globe, points_fc):
        globe.draw()
        assert globe.points(points_fc) is not None

    def test_a_projected_feature_collection_is_reprojected(self, globe, points_fc):
        """The fixture is UTM 18N; its coordinates must become lon/lat before they reach the sphere."""
        assert points_fc.epsg == 32618, (
            "fixture precondition: the points are in a projected CRS"
        )
        lon, lat = TexturedGlobe._as_lonlat(points_fc, None)
        assert np.all(np.abs(lon) <= 180), (
            f"longitudes are still projected: {lon.min()}..{lon.max()}"
        )
        assert np.all(np.abs(lat) <= 90), (
            f"latitudes are still projected: {lat.min()}..{lat.max()}"
        )

    def test_a_lonlat_feature_collection_passes_through(self, points_fc):
        """Already in 4326, so the reprojection branch must be skipped rather than re-warping."""
        lonlat = points_fc.to_crs(4326)
        lon, lat = TexturedGlobe._as_lonlat(lonlat, None)
        assert np.allclose(lon, lonlat.geometry.x.to_numpy()), (
            "lon should be untouched for a 4326 collection"
        )
        assert np.allclose(lat, lonlat.geometry.y.to_numpy()), (
            "lat should be untouched for a 4326 collection"
        )

    def test_non_point_geometry_falls_back_to_centroids(self):
        """Polygons have no .x/.y, so they must be reduced to centroids rather than raising."""
        square = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
        frame = FeatureCollection(geometry=[square], crs="EPSG:4326")
        lon, lat = TexturedGlobe._as_lonlat(frame, None)
        assert lon[0] == pytest.approx(1.0), f"centroid lon should be 1.0, got {lon[0]}"
        assert lat[0] == pytest.approx(1.0), f"centroid lat should be 1.0, got {lat[0]}"

    def test_projected_polygons_are_centroided_before_reprojection(self):
        """A centroid taken on lon/lat degrees is not the centroid on the ground, and geopandas warns."""
        import warnings as _warnings

        square = Polygon(
            [(500000.0, 0.0), (501000.0, 0.0), (501000.0, 1000.0), (500000.0, 1000.0)]
        )
        frame = FeatureCollection(geometry=[square], crs="EPSG:32618")
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            lon, lat = TexturedGlobe._as_lonlat(frame, None)
        geographic = [w for w in caught if "geographic CRS" in str(w.message)]
        assert not geographic, (
            f"centroids should be taken in the projected CRS: {geographic}"
        )
        assert -180.0 <= lon[0] <= 180.0, f"longitude out of range: {lon[0]}"
        assert -90.0 <= lat[0] <= 90.0, f"latitude out of range: {lat[0]}"

    def test_a_crs_without_an_epsg_code_is_accepted(self):
        """`.epsg` is None for a valid CRS with no authority code; only a missing CRS is unplaceable."""
        square = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        mollweide = "+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        frame = FeatureCollection(geometry=[square], crs=mollweide)
        lon, lat = TexturedGlobe._as_lonlat(frame, None)
        assert np.isfinite(lon).all(), "longitudes should all be finite"
        assert np.isfinite(lat).all(), "latitudes should all be finite"

    def test_a_feature_collection_without_a_crs_is_refused(self):
        """With no CRS the coordinates cannot be placed on the sphere, so fail rather than assume 4326."""
        frame = FeatureCollection(geometry=[Point(1.0, 2.0)], crs=None)
        with pytest.raises(ValueError, match="no CRS"):
            TexturedGlobe._as_lonlat(frame, None)

    def test_input_without_geometry_is_refused(self, globe):
        globe.draw()
        with pytest.raises(ValueError, match="FeatureCollection"):
            globe.points(object())


class TestOverlaysFollowTheSpin:
    """Overlays turn with the sphere, stay on their own axes, and stay valid artists (#164)."""

    @staticmethod
    def _offsets(scatter):
        """The scatter's world-space coordinates as an ``(N, 3)`` array.

        Args:
            scatter: A scatter returned by ``TexturedGlobe.points``.

        Returns:
            np.ndarray: its ``(N, 3)`` offsets.
        """
        return np.asarray(scatter._offsets3d, dtype=float).T

    @staticmethod
    def _overlays_on(ax, globe):
        """Every collection on the axes other than a sphere surface.

        Counting these, rather than what the globe keeps track of, is what catches a copy left behind. A
        sphere is the exact ``Poly3DCollection`` ``plot_surface`` makes — every panel a globe is drawn on
        has its own, so it is matched by type rather than against the glyph's most recent surface.

        Args:
            ax: The axes to inspect.
            globe: The globe drawn on it.

        Returns:
            list: the overlay collections on ``ax``.
        """
        return [c for c in ax.collections if type(c) is not Poly3DCollection]

    @staticmethod
    def _axes():
        """A fresh 3-D axes on its own small figure.

        Returns:
            Axes3D: the axes.
        """
        return plt.figure(figsize=(3, 3)).add_subplot(projection="3d")

    def test_points_added_after_animate_turn_to_each_frames_spin(
        self, globe, mocker, tmp_path
    ):
        """Every rendered frame turns the overlay to the spin the sphere is drawn at.

        Test scenario:
            The defect was that animate() recorded only the start spin, so markers added afterwards stayed
            frozen while the sphere turned. Four frames over one revolution must turn the overlay through
            0, 90, 180 and 270 degrees. Consecutive repeats are collapsed: matplotlib primes the first
            frame before its save loop.
        """
        globe.animate(n_frames=4, interval=100, figsize=(3, 3), start_spin=0.0)
        globe.points([120.0, 130.0], lat=[10.0, -10.0], s=20, c="red")
        spy = mocker.spy(TexturedGlobe, "_turn_overlays")
        globe._animation.save(str(tmp_path / "spin.gif"), writer=PillowWriter(fps=2))
        spins = [call.args[2] for call in spy.call_args_list]
        stepped = [s for i, s in enumerate(spins) if i == 0 or s != spins[i - 1]]
        assert stepped == [0.0, 90.0, 180.0, 270.0], (
            f"the overlay should turn to each frame's spin, got {spins}"
        )
        globe.close()

    def test_a_marker_stays_on_the_ground_it_was_placed_on(self, globe):
        """After a turn each marker sits exactly where the surface point under it has moved to.

        Test scenario:
            Checked at 45 degrees on a tilted globe. At 180 degrees turning either way lands in the same
            place, which is how an overlay turning the wrong way once passed every placement test.
        """
        lon, lat = [30.0, -60.0], [10.0, -25.0]
        ax = self._axes()
        globe.draw(ax, spin=0.0)
        scatter = globe.points(lon, lat=lat, hide_far_side=False)
        globe.draw(ax, spin=45.0)
        ax.get_figure().canvas.draw()
        expected = globe.project(lon, lat, spin=45.0, altitude=0.01)
        assert np.allclose(self._offsets(scatter), expected), (
            f"the markers should sit on the ground at spin 45, got {self._offsets(scatter).tolist()} "
            f"instead of {expected.tolist()}"
        )
        plt.close(ax.get_figure())

    def test_after_an_animation_a_marker_sits_on_the_last_frames_ground(
        self, globe, tmp_path
    ):
        """The last rendered frame leaves each marker on the ground at that frame's spin.

        Test scenario:
            Three frames over a revolution end at 240 degrees, a spin where the two directions of turn
            disagree.
        """
        globe.animate(n_frames=3, interval=100, figsize=(3, 3), start_spin=0.0)
        scatter = globe.points([30.0], lat=[10.0], hide_far_side=False)
        globe._animation.save(str(tmp_path / "ground.gif"), writer=PillowWriter(fps=2))
        expected = globe.project([30.0], [10.0], spin=240.0, altitude=0.01)
        assert np.allclose(self._offsets(scatter), expected), (
            f"the marker should sit on the ground at spin 240, got {self._offsets(scatter).tolist()}"
        )
        globe.close()

    def test_the_returned_artist_is_the_one_that_moves(self, globe, tmp_path):
        """The scatter points() returned is still on the axes after the animation, and has turned.

        Test scenario:
            The overlay is one persistent artist, so the caller's handle is live for the whole animation.
        """
        globe.animate(n_frames=2, interval=100, figsize=(3, 3), start_spin=0.0)
        scatter = globe.points([120.0], lat=[0.0], s=20, c="red", hide_far_side=False)
        start = self._offsets(scatter)
        globe._animation.save(str(tmp_path / "half.gif"), writer=PillowWriter(fps=2))
        assert scatter in globe.ax.collections, (
            "the returned artist should still be drawn"
        )
        assert not np.allclose(self._offsets(scatter), start), (
            f"a half turn should move the marker, but it stayed at {start.tolist()}"
        )
        globe.close()

    def test_frames_do_not_stack_copies_of_an_overlay(self, globe, tmp_path):
        """After eight frames the axes holds exactly one overlay collection.

        Test scenario:
            Counts every collection on the axes other than the sphere, not only the ones the globe tracks,
            so an implementation that left a copy behind each frame would fail here.
        """
        globe.animate(n_frames=8, interval=100, figsize=(3, 3))
        globe.points([120.0], lat=[0.0], s=20, c="red", hide_far_side=False)
        globe._animation.save(str(tmp_path / "stack.gif"), writer=PillowWriter(fps=2))
        overlays = self._overlays_on(globe.ax, globe)
        assert len(overlays) == 1, (
            f"expected one overlay collection, got {len(overlays)}"
        )
        globe.close()

    def test_an_explicit_spin_pins_the_overlay(self, globe, tmp_path):
        """points(spin=...) opts out of following the globe.

        Test scenario:
            Driving the rotation by hand is the documented escape hatch, so a caller who names a spin must
            keep the markers exactly there while the sphere turns beneath them.
        """
        globe.animate(n_frames=4, interval=100, figsize=(3, 3), start_spin=0.0)
        scatter = globe.points(
            [120.0], lat=[0.0], spin=0.0, s=20, c="red", hide_far_side=False
        )
        pinned = self._offsets(scatter)
        globe._animation.save(str(tmp_path / "pinned.gif"), writer=PillowWriter(fps=2))
        assert globe._overlays == [], "a pinned overlay should not be registered"
        assert np.allclose(self._offsets(scatter), pinned), (
            "a pinned overlay should not move when the globe turns"
        )
        globe.close()

    def test_points_added_before_animate_on_the_same_axes_follow(self, globe, tmp_path):
        """draw(ax) then points() then animate(ax) turns the markers with the sphere.

        Test scenario:
            The overlay belongs to the axes it was drawn on, so an animation on that same axes turns it.
        """
        ax = self._axes()
        globe.draw(ax)
        scatter = globe.points([120.0], lat=[0.0], s=20, c="red", hide_far_side=False)
        start = self._offsets(scatter)
        globe.animate(ax, n_frames=2, interval=100, start_spin=0.0)
        globe._animation.save(str(tmp_path / "before.gif"), writer=PillowWriter(fps=2))
        assert not np.allclose(self._offsets(scatter), start), (
            "an overlay added before animate() on the same axes should follow the rotation"
        )
        plt.close(ax.get_figure())

    def test_an_animation_on_a_new_figure_does_not_bring_old_overlays_back(
        self, globe, tmp_path
    ):
        """An overlay stays with the figure it was drawn on, and goes when that figure is closed.

        Test scenario:
            A bare animate() makes its own figure and closes the one the globe drew before, so the overlay
            drawn there must neither appear in the animation nor stay registered to the dead figure.
        """
        globe.draw(figsize=(3, 3))
        globe.points([120.0], lat=[0.0], s=20, c="red", hide_far_side=False)
        globe.animate(n_frames=2, interval=100)
        globe._animation.save(str(tmp_path / "fresh.gif"), writer=PillowWriter(fps=2))
        assert self._overlays_on(globe.ax, globe) == [], (
            "the animation's figure should not gain the old figure's overlay"
        )
        assert globe._overlays == [], (
            "an overlay on a closed figure should be forgotten"
        )
        globe.close()

    def test_redrawing_at_a_new_spin_turns_the_overlay(self, globe):
        """A draw() at another spin on the same axes turns an overlay already on it.

        Test scenario:
            draw() turns the sphere for the same reason animate() does, so the overlay turns with it.
        """
        ax = self._axes()
        globe.draw(ax, spin=0.0)
        scatter = globe.points([120.0], lat=[0.0], s=20, c="red", hide_far_side=False)
        start = self._offsets(scatter)
        globe.draw(ax, spin=90.0)
        ax.get_figure().canvas.draw()
        assert scatter._spin == 90.0, (
            f"the overlay should be at spin 90, got {scatter._spin}"
        )
        assert not np.allclose(self._offsets(scatter), start), (
            "redrawing at a new spin should move the overlay with the sphere"
        )
        plt.close(ax.get_figure())

    def test_each_panel_of_a_contact_sheet_keeps_its_own_overlay(self, globe):
        """Turning one panel moves only that panel's overlay, and no panel loses or gains one.

        Test scenario:
            Three panels drawn at three spins, each given a marker. Redrawing the last panel used to strip
            the first two and stack three copies on the third.
        """
        fig = plt.figure(figsize=(6, 2))
        axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
        for ax, spin in zip(axes, (0.0, 90.0, 180.0)):
            globe.draw(ax, spin=spin)
            globe.points([0.0], lat=[0.0], hide_far_side=False)
        globe.draw(axes[2], spin=250.0)
        counts = [len(self._overlays_on(ax, globe)) for ax in axes]
        assert counts == [1, 1, 1], (
            f"every panel should keep exactly one overlay, got {counts}"
        )
        spins = [self._overlays_on(ax, globe)[0]._spin for ax in axes]
        assert spins == [0.0, 90.0, 250.0], (
            f"only the redrawn panel should turn, got {spins}"
        )
        plt.close(fig)

    def test_a_redraw_loop_that_clears_the_axes_keeps_one_overlay(self, globe):
        """A hand-written loop of clear, draw and points leaves one marker set per frame.

        Test scenario:
            ax.clear() takes the previous frame's scatter off the axes, so it has to be forgotten rather
            than turned and brought back — otherwise every frame adds one more copy.
        """
        ax = self._axes()
        for spin in np.arange(0.0, 360.0, 30.0):
            ax.clear()
            globe.draw(ax, spin=float(spin))
            globe.points([0.0], lat=[0.0], hide_far_side=False)
        overlays = self._overlays_on(ax, globe)
        assert len(overlays) == 1, (
            f"expected one scatter on the axes, got {len(overlays)}"
        )
        assert len(globe._overlays) == 1, (
            f"expected one registered overlay, got {len(globe._overlays)}"
        )
        plt.close(ax.get_figure())

    def test_a_value_keeps_its_colour_as_the_globe_turns(self, globe):
        """The colour scale spans every point's value at every spin.

        Test scenario:
            Thirty-six equator stations valued 0..35 with no vmin/vmax. Fitting the scale to whichever
            stations face the camera made one value change colour frame to frame.
        """
        ax = self._axes()
        globe.draw(ax, spin=0.0)
        scatter = globe.points(
            np.linspace(-175.0, 175.0, 36), lat=np.zeros(36), c=np.arange(36.0)
        )
        scales = []
        for spin in (0.0, 90.0, 180.0):
            globe.draw(ax, spin=spin)
            ax.get_figure().canvas.draw()
            scales.append((float(scatter.norm.vmin), float(scatter.norm.vmax)))
        assert scales == [(0.0, 35.0)] * 3, (
            f"the colour scale should not move, got {scales}"
        )
        plt.close(ax.get_figure())

    def test_removing_the_returned_artist_is_final(self, globe):
        """An overlay the caller removed stays removed when the globe turns.

        Test scenario:
            Its artist has left the axes, so the next draw forgets it rather than restoring it.
        """
        ax = self._axes()
        globe.draw(ax, spin=0.0)
        scatter = globe.points([0.0], lat=[0.0])
        scatter.remove()
        globe.draw(ax, spin=10.0)
        assert scatter not in ax.collections, "a removed overlay should not come back"
        assert globe._overlays == [], "a removed overlay should be forgotten"
        plt.close(ax.get_figure())

    def test_restyling_the_returned_artist_survives_a_turn(self, globe):
        """A size set on the returned artist is kept, with the far-side mask applied on top of it.

        Test scenario:
            The mask rewrites sizes on every render, so it must start from what the caller asked for.
        """
        ax = self._axes()
        globe.draw(ax, elev=0.0, azim=0.0, spin=0.0)
        scatter = globe.points([0.0, 10.0, 180.0], lat=[0.0] * 3, s=10)
        scatter.set_sizes([80.0])
        globe.draw(ax, elev=0.0, azim=0.0, spin=5.0)
        assert list(_drawn(scatter)) == [80.0, 80.0], (
            f"the two near-side markers should keep the new size, got {list(_drawn(scatter))}"
        )
        plt.close(ax.get_figure())

    def test_a_restyled_edge_width_survives_the_mask(self, globe):
        """An edge width set on the returned artist is kept for the markers that show.

        Test scenario:
            The far-side mask zeroes hidden markers' edges on every render; it must apply over the width
            the caller set, not over the one the scatter was created with.
        """
        ax = self._axes()
        globe.draw(ax, elev=0.0, azim=0.0, spin=0.0)
        scatter = globe.points([0.0, 10.0, 180.0], lat=[0.0] * 3, linewidths=1.0)
        scatter.set_linewidth(3.0)
        globe.draw(ax, elev=0.0, azim=0.0, spin=5.0)
        widths = list(_drawn(scatter, "widths"))
        assert widths == [3.0, 3.0], (
            f"the two near-side markers should keep the new width, got {widths}"
        )
        plt.close(ax.get_figure())

    def test_close_forgets_the_overlays(self, globe):
        """close() drops every overlay, so a later draw on any axes starts clean."""
        globe.draw(figsize=(3, 3))
        globe.points([0.0], lat=[0.0])
        globe.close()
        assert globe._overlays == [], "close() should forget the overlays"
        ax = self._axes()
        globe.draw(ax, spin=10.0)
        assert self._overlays_on(ax, globe) == [], (
            "nothing should be brought back after close()"
        )
        plt.close(ax.get_figure())

    def test_a_camera_move_re_decides_which_side_shows(self, globe):
        """Visibility is decided at draw time, against the camera the axes has then.

        Test scenario:
            A marker at lon 0 faces a camera at azim 0 and is hidden from one at azim 180 — the same
            thing a mouse drag of the 3-D axes does, with no call into the globe in between.
        """
        ax = self._axes()
        globe.draw(ax, elev=0.0, azim=0.0)
        scatter = globe.points([0.0], lat=[0.0], s=40)
        assert _drawn(scatter).size == 1, "the marker should show from the front"
        ax.view_init(elev=0.0, azim=180.0)
        assert _drawn(scatter).size == 0, (
            "the marker should hide once the camera goes round"
        )
        plt.close(ax.get_figure())

    @pytest.mark.parametrize(
        "setter, values",
        [("set_sizes", "sizes"), ("set_linewidth", "widths")],
        ids=["sizes", "widths"],
    )
    def test_clearing_a_style_falls_back_to_matplotlibs_default(
        self, globe, setter, values
    ):
        """Passing None asks matplotlib for its default, which the mask must not turn into NaN.

        Args:
            setter: The setter the caller clears the style with.
            values: Which drawn values that setter governs.

        Test scenario:
            The mask re-applies the remembered style on every render, so remembering the raw None left
            every marker at a NaN size or edge and they silently vanished.
        """
        ax = self._axes()
        globe.draw(ax, elev=0.0, azim=0.0)
        scatter = globe.points([0.0, 10.0, 180.0], lat=[0.0] * 3, s=40, linewidths=1.0)
        getattr(scatter, setter)(None)
        drawn = _drawn(scatter, values)
        assert np.isfinite(drawn).all() and (drawn > 0).all(), (
            f"clearing {setter} should fall back to a real default, got {list(drawn)}"
        )
        plt.close(ax.get_figure())

    def test_a_legend_built_after_a_render_shows_the_marker_as_asked(self, globe):
        """A legend made once the figure has been drawn takes the caller's size and edge, not the mask.

        Test scenario:
            matplotlib's legend handler reads the sizes and edge widths straight off the artist. While a
            render is under way those hold the far-side mask, which would size the legend's marker from a
            zero — half what was asked here — and leave it with no edge.
        """
        ax = self._axes()
        globe.draw(ax, elev=0.0, azim=0.0)
        globe.points(
            [0.0, 10.0, 180.0], lat=[0.0] * 3, s=40, linewidths=1.0, label="sites"
        )
        ax.get_figure().canvas.draw()
        handle = ax.legend().legend_handles[0]
        assert list(np.atleast_1d(handle.get_sizes())) == [40.0], (
            f"the legend marker should be the size asked for, got {handle.get_sizes()}"
        )
        assert list(np.atleast_1d(handle.get_linewidths())) == [1.0], (
            f"the legend marker should keep its edge, got {handle.get_linewidths()}"
        )
        plt.close(ax.get_figure())

    def test_the_callers_axes_keeps_its_depth_sort(self, globe):
        """Drawing a globe with overlays leaves the caller's axes depth-sorting its artists.

        Test scenario:
            Switching computed_zorder off would reorder every artist the caller drew on the axes; the
            overlays sort in front of the sphere by depth instead.
        """
        ax = self._axes()
        globe.draw(ax)
        globe.points([0.0], lat=[0.0])
        assert ax.computed_zorder is True, (
            "the caller's axes should keep matplotlib's depth sort"
        )
        plt.close(ax.get_figure())

    def test_an_overlay_near_the_limb_sorts_in_front_of_the_sphere(self, globe):
        """A near-side overlay reports a depth nearer than the sphere's, however close to the limb.

        Test scenario:
            matplotlib ranks each collection by one representative depth; a marker near the limb would
            rank behind the sphere's front pole and be painted over. The overlay reports a nearer one.
        """
        ax = self._axes()
        globe.draw(ax, elev=0.0, azim=0.0)
        scatter = globe.points([80.0], lat=[0.0])
        ax.get_figure().canvas.draw()
        surface = globe.glyph.surface.do_3d_projection()
        overlay = scatter.do_3d_projection()
        assert overlay < surface, (
            f"the overlay should sort nearer than the sphere, got {overlay} vs {surface}"
        )
        plt.close(ax.get_figure())

    def test_a_fresh_globe_survives_pickling_and_copying(self, flat_texture):
        """A globe carries no closures, so it pickles, and a copy turns its own overlays.

        Test scenario:
            Batch rendering hands globes to worker processes; a copy must be independent of the original.
        """
        globe = TexturedGlobe(flat_texture, n_lon=8, n_lat=4)
        for clone in (pickle.loads(pickle.dumps(globe)), copy.deepcopy(globe)):
            ax = self._axes()
            clone.draw(ax, spin=0.0)
            scatter = clone.points([0.0], lat=[0.0])
            clone.draw(ax, spin=45.0)
            assert scatter._spin == 45.0, "a copy's overlay should follow the copy"
            assert globe._overlays == [], "the original should be untouched by its copy"
            plt.close(ax.get_figure())

    def test_a_globe_with_no_overlays_is_untouched(self, globe, tmp_path):
        """A bare globe animates with nothing but its surface on the axes."""
        globe.animate(n_frames=3, interval=100, figsize=(3, 3))
        globe._animation.save(str(tmp_path / "bare.gif"), writer=PillowWriter(fps=2))
        assert self._overlays_on(globe.ax, globe) == [], "no overlay should appear"
        globe.close()


class TestReferenceGeography:
    """coastlines / borders / land on the 3-D globe, matching the 2-D Map spelling (#165)."""

    @pytest.fixture
    def drawn(self, flat_texture):
        """A globe with no tilt, drawn face-on, so the visible hemisphere is exactly |lon| < 90.

        Returns:
            TexturedGlobe: drawn at elev 0 / azim 0 with the polar axis upright.
        """
        globe = TexturedGlobe(flat_texture, n_lon=24, n_lat=12, tilt_deg=0.0)
        globe.draw(elev=0.0, azim=0.0, figsize=(3, 3))
        yield globe
        globe.close()

    @staticmethod
    def _arc(lon_from, lon_to, lat=0.0, n=61):
        """A lon/lat polyline along one parallel.

        Args:
            lon_from: Starting longitude in degrees.
            lon_to: Ending longitude in degrees.
            lat: The parallel to run along.
            n: Vertex count.

        Returns:
            np.ndarray: an ``(n, 2)`` lon/lat array shaped like natural_earth's parts.
        """
        lon = np.linspace(lon_from, lon_to, n)
        return np.column_stack([lon, np.full(n, lat)])

    @staticmethod
    def _ring(west, east, south, north, start=0):
        """A closed lon/lat rectangle, shaped like a natural_earth polygon part.

        Args:
            west: Western longitude.
            east: Eastern longitude.
            south: Southern latitude.
            north: Northern latitude.
            start: How many vertices to roll the ring by before closing it, to move its seam.

        Returns:
            np.ndarray: an ``(N, 2)`` closed ring.
        """
        lon = np.r_[
            np.linspace(west, east, 25),
            np.full(25, east),
            np.linspace(east, west, 25),
            np.full(25, west),
        ]
        lat = np.r_[
            np.full(25, south),
            np.linspace(south, north, 25),
            np.full(25, north),
            np.linspace(north, south, 25),
        ]
        ring = np.roll(np.column_stack([lon, lat]), -start, axis=0)
        return np.vstack([ring, ring[:1]])

    @staticmethod
    def _lonlat(world):
        """Back from world space to lon/lat degrees, for a globe with no tilt at spin 0.

        Args:
            world: ``(N, 3)`` world-space points.

        Returns:
            tuple: longitude and latitude arrays, in degrees.
        """
        radius = np.linalg.norm(world, axis=1)
        return (
            np.degrees(np.arctan2(world[:, 1], world[:, 0])),
            np.degrees(np.arcsin(world[:, 2] / radius)),
        )

    def _patch(self, mocker, parts):
        """Serve ``parts`` in place of the Natural-Earth download.

        Args:
            mocker: The pytest-mock fixture.
            parts: The lon/lat arrays natural_earth should return.
        """
        mocker.patch(
            "digitalearth.static.textured_globe.natural_earth", return_value=parts
        )

    def test_the_globe_offers_the_layers_the_flat_map_does(self):
        """TexturedGlobe answers coastlines/borders/land, as Map already did."""
        missing = [
            name
            for name in ("coastlines", "borders", "land")
            if not hasattr(TexturedGlobe, name)
        ]
        assert missing == [], (
            f"TexturedGlobe should offer these layers, missing {missing}"
        )

    def test_a_line_crossing_the_limb_is_split_into_near_side_arcs(self, drawn, mocker):
        """A polyline running off the visible hemisphere is broken rather than drawn through the planet.

        Test scenario:
            A parallel running east from 60 degrees round to 300 leaves the visible hemisphere at 90 and
            re-enters it at 270, so it must come back as two arcs with every vertex facing the camera (or
            sitting exactly on the limb, where each arc now ends).
        """
        self._patch(mocker, [self._arc(60.0, 300.0)])
        segments = drawn.coastlines()._near_side_segments()
        assert len(segments) == 2, (
            f"a line crossing the far side should split in two, got {len(segments)}"
        )
        for segment in segments:
            depth = segment @ np.array([1.0, 0.0, 0.0])
            assert depth.min() >= -1e-9, (
                f"no vertex should be behind the limb, min depth {depth.min()}"
            )

    def test_each_arc_is_carried_out_to_the_limb(self, drawn, mocker):
        """An arc ends on the horizon, not one vertex short of it, as a land fill does.

        Test scenario:
            A parallel from 0 to 150 degrees leaves the visible side at 90; the arc's last vertex must lie
            on the limb plane, not at the last visible data vertex some degrees inside it. Twelve vertices
            keep every data vertex off the limb itself, where the check would pass without the crossing.
        """
        self._patch(mocker, [self._arc(0.0, 150.0, n=12)])
        (segment,) = drawn.coastlines()._near_side_segments()
        assert abs(float(segment[-1] @ np.array([1.0, 0.0, 0.0]))) < 1e-9, (
            f"the arc should end on the limb, depth {float(segment[-1] @ np.array([1.0, 0.0, 0.0]))}"
        )

    def test_a_layer_entirely_on_the_far_side_draws_nothing(self, drawn, mocker):
        """Geography behind the globe produces no arcs."""
        self._patch(mocker, [self._arc(170.0, 190.0)])
        assert drawn.coastlines()._near_side_segments() == [], (
            "a hidden layer should draw no arcs"
        )

    @pytest.mark.parametrize("start", [0, 60], ids=["seam-visible", "seam-hidden"])
    def test_a_land_ring_stays_inside_its_polygon_or_on_the_limb(self, drawn, start):
        """A clipped ring's every vertex lies inside the land it outlines, wherever the ring's seam falls.

        Args:
            start: How far to roll the ring, putting its first vertex on the near side (0) or far side (60).

        Test scenario:
            A rectangle over 60..120 east crosses the limb at 90. A ring starting on a visible vertex used
            to be split at its seam and each half closed out to the horizon, adding a spur of vertices
            south of the polygon; every vertex must lie within the rectangle, clipped at the limb.
        """
        world = drawn.project(
            *self._ring(60.0, 120.0, -20.0, 20.0, start).T, altitude=0.001
        )
        face = _clip_ring(world, np.array([1.0, 0.0, 0.0]))
        lon, lat = self._lonlat(face)
        assert np.allclose(np.linalg.norm(face, axis=1), 1.001), (
            "the face should stay on one shell"
        )
        assert lon.min() >= 60.0 - 1e-6 and lon.max() <= 90.0 + 1e-6, (
            f"longitudes should run 60..90, got {lon.min()}..{lon.max()}"
        )
        # a limb crossing sits on the chord between two vertices of a parallel, a few hundredths of a
        # degree off it; the spur this guards against overshot the polygon by nearly three degrees
        assert lat.min() >= -20.1 and lat.max() <= 20.1, (
            f"latitudes should stay within -20..20, got {lat.min()}..{lat.max()}"
        )

    @pytest.mark.parametrize("layer", ["coastlines", "land"])
    def test_a_layer_with_no_usable_parts_draws_nothing(self, drawn, mocker, layer):
        """A layer whose source has no drawable parts is added and renders as nothing.

        Args:
            layer: The layer method under test.

        Test scenario:
            A part needs two vertices to be a line or a ring; a source offering only shorter ones (or
            none) must not break the render.
        """
        self._patch(mocker, [np.array([[0.0, 0.0]])])
        overlay = getattr(drawn, layer)()
        drawn.fig.canvas.draw()
        pieces = (
            overlay._near_side_faces()
            if layer == "land"
            else overlay._near_side_segments()
        )
        assert pieces == [], (
            f"{layer} should have nothing to draw, got {len(pieces)} pieces"
        )

    def test_a_degenerate_land_ring_is_skipped(self, drawn, mocker):
        """A land ring too short to enclose anything is dropped without disturbing its neighbours.

        Test scenario:
            A two-vertex sliver sits beside a proper rectangle; only the rectangle becomes a face.
        """
        sliver = np.array([[10.0, 0.0], [12.0, 1.0]])
        self._patch(mocker, [sliver, self._ring(-40.0, 40.0, -20.0, 20.0)])
        faces = drawn.land()._near_side_faces()
        assert len(faces) == 1, (
            f"only the rectangle should become a face, got {len(faces)}"
        )

    def test_a_long_land_ring_does_not_pad_every_face(self, drawn, mocker):
        """Rendering one very long ring beside many short ones stays within a few megabytes.

        Test scenario:
            matplotlib 3.11's Poly3DCollection pads every face to the longest, so 300 small islands next to
            one 20 000-vertex coast would cost ~150 MB per frame there. The fill projects its own faces, so
            the render's peak allocation must stay far below that.
        """
        tracemalloc = pytest.importorskip("tracemalloc")
        angle = np.linspace(0.0, 2.0 * np.pi, 20_000)
        coast = np.column_stack([20.0 * np.cos(angle), 20.0 * np.sin(angle)])
        islands = [
            np.array(
                [[lon, -60.0], [lon + 1.0, -60.0], [lon + 1.0, -59.0], [lon, -59.0]]
            )
            for lon in np.linspace(-80.0, 80.0, 300)
        ]
        self._patch(mocker, [coast, *islands])
        drawn.land()
        drawn.fig.canvas.draw()
        tracemalloc.start()
        drawn.fig.canvas.draw()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak < 40 * 1024 * 1024, (
            f"one render should stay small, peaked at {peak / 2**20:.0f} MB"
        )

    def test_a_land_layer_with_nothing_in_view_has_no_faces(self, drawn, mocker):
        """land() still returns its collection, which simply has nothing to fill at this camera."""
        self._patch(mocker, [self._ring(160.0, 200.0, -10.0, 10.0)])
        patch = drawn.land()
        assert patch in drawn.ax.collections, "the fill should be on the axes"
        assert patch._near_side_faces() == [], "no face should face the camera"

    def test_the_layers_follow_the_globe_as_it_turns(self, drawn, mocker):
        """A reference layer is registered, and a redraw at a new spin moves it.

        Test scenario:
            The sibling of the overlay fix: geography added to a turning globe has to turn with it.
        """
        self._patch(mocker, [self._arc(-20.0, 20.0)])
        lines = drawn.coastlines()
        assert drawn._overlays == [lines], (
            "a reference layer should register to follow the globe"
        )
        (before,) = lines._near_side_segments()
        drawn.draw(drawn.ax, elev=0.0, azim=0.0, spin=45.0)
        (after,) = lines._near_side_segments()
        assert before.shape == after.shape, (
            "this arc stays wholly visible at 45 degrees"
        )
        assert not np.allclose(before, after), (
            "the layer should move when the globe turns"
        )
        plt.close(
            drawn.fig
        )  # drawing onto the globe's own axes hands the figure to the caller

    def test_a_coastline_stays_on_the_ground_it_traces(self, drawn, mocker):
        """After a turn a coastline's vertices are exactly the turned surface points.

        Test scenario:
            A wholly visible arc, turned 45 degrees: every vertex must equal the projection of its lon/lat
            at that spin, not merely have moved.
        """
        arc = self._arc(-20.0, 20.0)
        self._patch(mocker, [arc])
        lines = drawn.coastlines()
        drawn.draw(drawn.ax, elev=0.0, azim=0.0, spin=45.0)
        (segment,) = lines._near_side_segments()
        expected = drawn.project(arc[:, 0], arc[:, 1], spin=45.0, altitude=0.002)
        assert np.allclose(segment, expected), (
            "the coastline should trace the ground at spin 45"
        )
        plt.close(drawn.fig)

    def test_a_land_fill_stays_on_the_ground_it_covers(self, drawn, mocker):
        """After a turn a wholly visible land ring's face is exactly the turned surface ring.

        Test scenario:
            The same check for the fill layer, on a small ring well inside the visible hemisphere.
        """
        ring = np.array(
            [
                [-10.0, -5.0],
                [10.0, -5.0],
                [15.0, 5.0],
                [0.0, 12.0],
                [-15.0, 5.0],
                [-10.0, -5.0],
            ]
        )
        self._patch(mocker, [ring])
        fill = drawn.land()
        drawn.draw(drawn.ax, elev=0.0, azim=0.0, spin=30.0)
        (face,) = fill._near_side_faces()
        expected = drawn.project(ring[:-1, 0], ring[:-1, 1], spin=30.0, altitude=0.001)
        assert np.allclose(face, expected), (
            "the fill should cover the ground at spin 30"
        )
        plt.close(drawn.fig)

    def test_a_pinned_layer_stays_where_it_was_put(self, drawn, mocker):
        """Naming a spin opts the layer out of following, exactly as it does for points()."""
        self._patch(mocker, [self._arc(-60.0, 60.0)])
        drawn.coastlines(spin=0.0)
        assert drawn._overlays == [], "a pinned layer should not be registered"

    def test_fill_lines_and_points_sort_in_that_order_in_front_of_the_sphere(
        self, drawn, mocker
    ):
        """Each layer kind reports a depth nearer than the one it should be drawn over.

        Test scenario:
            matplotlib paints collections far to near, so fill must rank nearer than the sphere, lines
            nearer than fill, and markers nearest.
        """
        self._patch(mocker, [self._ring(-40.0, 40.0, -20.0, 20.0)])
        fill = drawn.land()
        self._patch(mocker, [self._arc(-60.0, 60.0)])
        lines = drawn.coastlines()
        markers = drawn.points([0.0], lat=[0.0])
        drawn.fig.canvas.draw()
        depths = [
            artist.do_3d_projection()
            for artist in (drawn.glyph.surface, fill, lines, markers)
        ]
        assert depths == sorted(depths, reverse=True), (
            f"expected sphere > fill > lines > markers in depth, got {depths}"
        )

    def test_a_layer_near_the_limb_is_actually_painted(self, mocker):
        """A coastline hugging the horizon shows in the rendered image, not behind the planet.

        Test scenario:
            The pixel-level check behind the depth ordering: a red arc near the limb on a blue globe has
            to leave red pixels in the saved figure.
        """
        texture = np.zeros((45, 90, 3), dtype=np.uint8)
        texture[:] = (0, 40, 120)
        globe = TexturedGlobe(texture, n_lon=90, n_lat=45, tilt_deg=0.0)
        fig, _ = globe.draw(elev=0.0, azim=0.0, figsize=(3, 3))
        self._patch(mocker, [self._arc(60.0, 85.0, lat=-30.0)])
        globe.coastlines(color="red", linewidth=2.0)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int)
        red = (image[:, :, 0] > 150) & (image[:, :, 1] < 90) & (image[:, :, 2] < 90)
        assert red.sum() > 0, "the arc near the limb should be visible"
        globe.close()

    def test_styling_reaches_the_collection(self, drawn, mocker):
        """Layer styling is forwarded to matplotlib, over each layer's defaults."""
        self._patch(mocker, [self._arc(-60.0, 60.0)])
        lines = drawn.borders(color="white", linewidth=1.5)
        assert mcolors.to_hex(lines.get_color()[0]) == "#ffffff", (
            f"colour should be forwarded, got {lines.get_color()[0]}"
        )
        assert float(lines.get_linewidth()[0]) == 1.5, (
            f"linewidth should be forwarded, got {lines.get_linewidth()}"
        )

    @pytest.mark.parametrize(
        "layer, kwargs",
        [
            pytest.param("coastlines", {"lw": 1.2, "c": "white"}, id="line-aliases"),
            pytest.param("coastlines", {"zorder": 5}, id="line-zorder"),
            pytest.param("land", {"fc": "red", "ec": "blue"}, id="fill-aliases"),
            pytest.param("land", {"zorder": 5}, id="fill-zorder"),
        ],
    )
    def test_matplotlibs_own_spellings_are_accepted(self, drawn, mocker, layer, kwargs):
        """Short aliases and zorder are ordinary styling, and must not collide with the defaults.

        Args:
            layer: The layer method under test.
            kwargs: The styling to pass.

        Test scenario:
            ``lw=`` met the default ``linewidth`` and ``zorder=`` met an explicit one, each raising a
            TypeError; both spellings now resolve to one value.
        """
        self._patch(mocker, [self._ring(-40.0, 40.0, -20.0, 20.0)])
        overlay = getattr(drawn, layer)(**kwargs)
        assert overlay in drawn.ax.collections, (
            f"{layer}({kwargs}) should add its layer"
        )

    @pytest.mark.parametrize("layer", ["coastlines", "borders", "land"])
    def test_a_layer_needs_a_drawn_globe(self, flat_texture, layer):
        """Asking for geography before draw() is refused, since there is no axes to add it to.

        Args:
            layer: The reference layer method under test.
        """
        globe = TexturedGlobe(flat_texture, n_lon=8, n_lat=4)
        with pytest.raises(RuntimeError, match="draw"):
            getattr(globe, layer)()


class TestOverlayGeometryHelpers:
    """The small geometry helpers the overlay layers are built from, including their degenerate cases."""

    def test_limb_point_lands_on_the_limb_at_the_rings_radius(self):
        """The crossing between a visible and a hidden vertex sits on the limb, same shell.

        Test scenario:
            With the camera down +x, a vertex at 45 degrees east of the limb and its neighbour 45
            degrees behind it must cross exactly at the limb plane, keeping the ring's radius.
        """
        view = np.array([1.0, 0.0, 0.0])
        radius = 1.01
        inside = radius * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
        outside = radius * np.array([-np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
        crossing = _limb_point(inside, outside, view)
        assert abs(float(crossing @ view)) < 1e-12, (
            f"the crossing should sit on the limb, depth {float(crossing @ view)}"
        )
        assert np.isclose(np.linalg.norm(crossing), radius), (
            f"the crossing should keep the ring's radius, got {np.linalg.norm(crossing)}"
        )

    def test_limb_point_falls_back_when_the_chord_runs_through_the_centre(self):
        """A chord straight down the view axis has no limb direction, so the visible end is kept.

        Test scenario:
            Both vertices lie on the view axis, so their crossing is the sphere's centre and there is
            no tangential direction to push it out along. Returning the visible vertex keeps the ring
            finite instead of emitting a zero vector.
        """
        view = np.array([1.0, 0.0, 0.0])
        inside = np.array([0.5, 0.0, 0.0])
        crossing = _limb_point(inside, np.array([-0.5, 0.0, 0.0]), view)
        assert np.allclose(crossing, inside), (
            f"expected the visible vertex back, got {crossing}"
        )

    def test_limb_point_handles_two_vertices_at_the_same_depth(self):
        """Neighbours at equal depth give no interpolation span, and the visible one is used.

        Test scenario:
            The zero-span branch: the crossing cannot be interpolated, so the near-side vertex is
            projected onto the limb rather than dividing by zero.
        """
        view = np.array([1.0, 0.0, 0.0])
        inside = np.array([0.5, 0.5, 0.0])
        crossing = _limb_point(inside, np.array([0.5, -0.5, 0.0]), view)
        assert abs(float(crossing @ view)) < 1e-12, (
            f"the fallback should still land on the limb, depth {float(crossing @ view)}"
        )

    def test_limb_arc_walks_the_short_way_round(self):
        """The closing arc stays on the limb and spans the shorter of the two ways round.

        Test scenario:
            From 0 to 90 degrees around the limb, every sampled point must sit on the limb plane at
            the same radius, and the arc must not take the 270-degree route.
        """
        view = np.array([1.0, 0.0, 0.0])
        start = np.array([0.0, 1.0, 0.0])
        arc = _limb_arc(start, np.array([0.0, 0.0, 1.0]), view)
        assert len(arc) > 0, "a quarter turn should be sampled, not skipped"
        assert np.allclose(arc @ view, 0.0, atol=1e-12), (
            "every arc point should lie on the limb"
        )
        assert np.allclose(np.linalg.norm(arc, axis=1), 1.0), (
            "the arc should keep its radius"
        )
        assert arc[0][1] > arc[-1][1], (
            "the arc should run from the start toward the end"
        )

    def test_limb_arc_is_empty_when_it_starts_where_it_ends(self):
        """A ring that leaves and rejoins the limb at one point has no arc to add.

        Test scenario:
            Sampling a zero-length arc used to emit dozens of copies of one point into the ring.
        """
        view = np.array([1.0, 0.0, 0.0])
        point = np.array([0.0, 1.0, 0.0])
        assert _limb_arc(point, point.copy(), view).shape == (0, 3), (
            "a zero-length arc should add no points"
        )

    def test_clip_ring_accepts_a_ring_without_its_closing_repeat(self):
        """An open ring is clipped the same way, with nothing to strip first.

        Test scenario:
            A three-vertex ring with one vertex behind the limb: the face keeps the two visible vertices
            and gains the two limb crossings either side of the hidden one.
        """
        view = np.array([1.0, 0.0, 0.0])
        ring = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [-0.6, 0.0, 0.8]])
        face = _clip_ring(ring, view)
        assert face is not None, "a ring with visible vertices should clip to a face"
        assert len(face) > 4, (
            f"two vertices plus the crossings and the limb arc, got {len(face)}"
        )
        assert (face @ view >= -1e-9).all(), (
            "no vertex of the face should be behind the limb"
        )

    def test_clip_ring_of_a_hidden_ring_is_none(self):
        """A ring wholly behind the globe clips to nothing."""
        view = np.array([1.0, 0.0, 0.0])
        ring = np.array([[-1.0, 0.0, 0.0], [-0.8, 0.6, 0.0], [-0.8, 0.0, 0.6]])
        assert _clip_ring(ring, view) is None, "a hidden ring should clip to None"

    @pytest.mark.parametrize(
        "ring",
        [
            pytest.param([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]], id="two-vertices"),
            pytest.param(
                [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [1.0, 0.0, 0.0]], id="closed-pair"
            ),
        ],
    )
    def test_clip_ring_of_a_degenerate_ring_is_none(self, ring):
        """Fewer than three distinct vertices enclose nothing, visible or not.

        Args:
            ring: A ring too short to enclose an area, with or without its closing repeat.
        """
        assert _clip_ring(np.array(ring), np.array([1.0, 0.0, 0.0])) is None, (
            "a degenerate ring should clip to None"
        )

    def test_clip_ring_does_not_depend_on_where_the_ring_starts(self):
        """Rolling a ring's seam round does not change the face it clips to.

        Test scenario:
            The same polygon, stored starting on a visible vertex and on a hidden one, must enclose the
            same region: the same set of vertices, allowing for the order they come in.
        """
        view = np.array([1.0, 0.0, 0.0])
        lon = np.deg2rad(np.r_[np.linspace(60, 120, 13), np.linspace(120, 60, 13)])
        lat = np.deg2rad(np.r_[np.full(13, -20.0), np.full(13, 20.0)])
        ring = np.column_stack(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )
        faces = [
            _clip_ring(np.vstack([rolled, rolled[:1]]), view)
            for rolled in (ring, np.roll(ring, -13, axis=0))
        ]
        assert faces[0].shape == faces[1].shape, (
            f"both seams should give the same face, got {faces[0].shape} and {faces[1].shape}"
        )
        assert np.allclose(np.sort(faces[0], axis=0), np.sort(faces[1], axis=0)), (
            "both seams should give the same vertices"
        )

    def test_limb_arc_is_empty_when_the_start_faces_the_camera(self):
        """A start point on the view axis has no limb direction, so no arc is produced.

        Test scenario:
            The degenerate branch: the cross product that builds the arc's basis vanishes, and
            returning nothing lets the ring close on its own vertices rather than on NaNs.
        """
        view = np.array([1.0, 0.0, 0.0])
        arc = _limb_arc(view.copy(), np.array([0.0, 1.0, 0.0]), view)
        assert arc.shape == (0, 3), f"expected no arc points, got {arc.shape}"


class TestRenderLifecycle:
    """draw / animate / save / stamp."""

    def test_draw_records_the_figure_and_axes(self, globe):
        fig, ax = globe.draw(spin=15.0)
        assert globe.fig is fig, "draw() should record the figure"
        assert globe.ax is ax, "draw() should record the axes"
        assert ax.name == "3d", f"expected a 3-D axes, got {ax.name}"

    def test_animate_records_the_figure_axes_and_rate(self, globe):
        anim = globe.animate(n_frames=2, interval=100)
        assert anim is not None
        assert globe.ax is not None, "animate() should record the axes"
        assert globe.fig is not None, "animate() should record the figure"
        assert globe._animation_fps == pytest.approx(10.0)

    def test_close_releases_the_figure(self, globe):
        """pyplot holds every figure it creates, so a loop of globes leaks without an explicit close."""
        plt.close("all")
        globe.draw()
        assert len(plt.get_fignums()) == 1
        globe.close()
        assert plt.get_fignums() == [], "close() should release the figure"
        assert globe.fig is None, "close() should clear the figure"
        assert globe.ax is None, "close() should clear the axes"

    def test_close_leaves_a_caller_supplied_figure_alone(self, globe):
        """The caller may have other subplots on that figure; it is not ours to close."""
        plt.close("all")
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        globe.draw(ax=ax)
        globe.close()
        assert plt.fignum_exists(fig.number), (
            "a caller-supplied figure must survive close()"
        )

    @pytest.mark.parametrize("via", ["ax", "fig"])
    def test_close_leaves_a_constructor_supplied_figure_alone(self, flat_texture, via):
        """The glyph stores a ctor fig/ax and draws on it even when no later call passes one."""
        plt.close("all")
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        kwargs = {"ax": ax, "fig": fig} if via == "ax" else {"fig": fig}
        owned = TexturedGlobe(flat_texture, n_lon=8, n_lat=4, **kwargs)
        owned.draw()
        assert owned.fig is fig, (
            "the globe should have drawn on the constructor's figure"
        )
        assert not owned._owns_fig, "a constructor-supplied figure is the caller's"
        owned.close()
        assert plt.fignum_exists(fig.number), (
            "a constructor-supplied figure must survive close()"
        )

    def test_switching_to_a_caller_axes_does_not_orphan_our_figure(self, globe):
        """Our own figure must be released when draw() rebinds, or close() can never reach it."""
        plt.close("all")
        globe.draw()
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        globe.draw(ax=ax)
        globe.close()
        assert plt.get_fignums() == [fig.number], (
            f"only the caller's figure should remain, got {plt.get_fignums()}"
        )

    def test_animate_works_when_the_constructor_supplied_the_axes(self, flat_texture):
        """A bare animate() must reuse the ctor's axes, not crash on a None it never created."""
        plt.close("all")
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        globe = TexturedGlobe(flat_texture, n_lon=8, n_lat=4, fig=fig, ax=ax)
        assert globe.animate(n_frames=2, interval=100) is not None, (
            "animate() should return the animation"
        )
        assert globe.ax is ax, "animate() should have used the constructor's axes"

    def test_redrawing_onto_the_same_figure_does_not_close_it(self, globe):
        """Rebinding to another axes on the figure we are already drawing on must keep it open."""
        plt.close("all")
        fig, _ = globe.draw()
        second = fig.add_subplot(222, projection="3d")
        globe.draw(ax=second)
        assert plt.fignum_exists(fig.number), "the figure being drawn on was closed"

    def test_animating_onto_a_caller_axes_releases_our_figure(self, globe):
        """The release-on-rebind rule has to apply to animate(), not only to draw()."""
        plt.close("all")
        globe.draw()
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        globe.animate(ax, n_frames=2, interval=100)
        globe.close()
        assert plt.get_fignums() == [fig.number], (
            f"only the caller's figure should remain, got {plt.get_fignums()}"
        )

    def test_close_leaves_a_caller_supplied_animation_axes_alone(self, globe):
        plt.close("all")
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        globe.animate(ax, n_frames=2, interval=100)
        globe.close()
        assert plt.fignum_exists(fig.number), (
            "a caller-supplied figure must survive close()"
        )

    def test_redrawing_does_not_leak_the_previous_figure(self, globe):
        """draw() rebinds fig/ax, so the one it replaces would otherwise be held by pyplot forever."""
        plt.close("all")
        globe.draw()
        globe.draw()
        globe.draw()
        assert len(plt.get_fignums()) == 1, (
            f"expected 1 open figure, got {len(plt.get_fignums())}"
        )

    def test_close_is_safe_before_drawing_and_twice(self, globe):
        globe.close()
        globe.draw()
        globe.close()
        globe.close()
        assert globe.fig is None

    def test_the_context_manager_closes_the_figure(self, flat_texture):
        plt.close("all")
        with TexturedGlobe(flat_texture, n_lon=8, n_lat=4) as globe:
            globe.draw()
            assert len(plt.get_fignums()) == 1
        assert plt.get_fignums() == [], "leaving the block should close the figure"

    def test_the_context_manager_propagates_errors(self, flat_texture):
        """It must not swallow an exception raised inside the block, and must still close the figure."""
        plt.close("all")
        globe = TexturedGlobe(flat_texture, n_lon=8, n_lat=4)
        globe.draw()

        def raise_inside() -> None:
            with globe:
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            raise_inside()
        assert plt.get_fignums() == [], (
            "the figure should be closed even when the block raised"
        )

    def test_animating_twice_does_not_leak_the_first_figure(self, globe):
        """animate() creates its own figure when given no axes, so the previous one must be released."""
        plt.close("all")
        globe.animate(n_frames=2, interval=100)
        globe.animate(n_frames=2, interval=100)
        assert len(plt.get_fignums()) == 1, (
            f"expected 1 open figure, got {len(plt.get_fignums())}"
        )

    def test_animate_records_its_starting_spin(self, globe):
        """Otherwise an overlay added after animate() is placed at whatever spin draw() last used."""
        globe.animate(n_frames=2, interval=100, start_spin=42.0)
        assert globe._spin == pytest.approx(42.0)

    def test_a_zero_animation_interval_is_refused(self, globe):
        """interval=0 used to raise ZeroDivisionError from the frame-rate bookkeeping."""
        with pytest.raises(ValueError, match="positive number of milliseconds"):
            globe.animate(n_frames=2, interval=0)

    @pytest.mark.parametrize("frames", [0, -3], ids=["zero", "negative"])
    def test_an_animation_with_no_frames_is_refused(self, globe, frames):
        """A rotation has to have at least one frame to show.

        Args:
            frames: A frame count below one.
        """
        with pytest.raises(ValueError, match="n_frames must be at least 1"):
            globe.animate(n_frames=frames, interval=100)

    def test_saving_before_drawing_is_refused(self, globe, tmp_path):
        with pytest.raises(RuntimeError, match="draw"):
            globe.save(str(tmp_path / "globe.png"))

    def test_save_writes_a_file(self, globe, tmp_path):
        globe.draw()
        out = tmp_path / "globe.png"
        globe.save(str(out))
        assert out.exists(), f"{out} was not written"
        assert out.stat().st_size > 0, f"{out} is empty"

    def test_saving_an_animation_before_animating_is_refused(self, globe, tmp_path):
        with pytest.raises(RuntimeError, match="no animation"):
            globe.save_animation(str(tmp_path / "globe.mp4"))

    def test_save_animation_forwards_to_the_shared_saver(self, globe, monkeypatch):
        """The globe delegates to digitalearth.static.animation rather than reimplementing the encode."""
        seen = {}
        monkeypatch.setattr(
            "digitalearth.static.textured_globe.save_animation",
            lambda anim, path, **kw: seen.update(anim=anim, path=path, **kw) or path,
        )
        globe.animate(n_frames=2, interval=125)
        globe.save_animation("globe.mp4", gif="globe.gif")
        assert seen["path"] == "globe.mp4", "the video path should be forwarded"
        assert seen["gif"] == "globe.gif", "the gif path should be forwarded"
        assert seen["anim"] is globe._animation, (
            "the saver must receive this globe's animation"
        )

    def test_save_animation_defaults_to_the_animations_own_rate(
        self, globe, monkeypatch
    ):
        """interval=125 ms is 8 fps; the saved clip should match what animate() was built for."""
        seen = {}
        monkeypatch.setattr(
            "digitalearth.static.textured_globe.save_animation",
            lambda anim, path, **kw: seen.update(kw) or path,
        )
        globe.animate(n_frames=2, interval=125)
        globe.save_animation("globe.mp4")
        assert seen["fps"] == pytest.approx(8.0), (
            f"expected 8 fps from a 125 ms interval, got {seen['fps']}"
        )

    def test_an_explicit_rate_overrides_the_animations(self, globe, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            "digitalearth.static.textured_globe.save_animation",
            lambda anim, path, **kw: seen.update(kw) or path,
        )
        globe.animate(n_frames=2, interval=125)
        globe.save_animation("globe.mp4", fps=24)
        assert seen["fps"] == 24, f"an explicit fps must win, got {seen['fps']}"

    def test_animate_accepts_an_existing_axes(self, globe):
        """Passing an axes must reuse it rather than opening a second figure."""
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(projection="3d")
        globe.animate(ax, n_frames=2, interval=100)
        assert globe.ax is ax, "the supplied axes should be adopted"
        assert globe.fig is fig, "fig should follow the supplied axes"

    def test_stamping_before_drawing_is_refused(self, globe):
        with pytest.raises(RuntimeError, match="draw"):
            globe.stamp(np.zeros((4, 4, 4), dtype=np.uint8))

    def test_stamp_adds_an_axes_to_the_figure(self, globe):
        fig, _ = globe.draw()
        before = len(fig.axes)
        globe.stamp(np.full((8, 8, 4), 255, dtype=np.uint8), frac=0.1, shadow=False)
        assert len(fig.axes) == before + 1
