"""Tests for digitalearth.scene.textured_globe — the pyramids/cleopatra 3-D globe seam.

The cleopatra glyph is already tested upstream, so these cover the half Digital-Earth owns: turning geodata
into an equirectangular texture at the right lon/lat, and mapping lon/lat back onto the drawn sphere.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from cleopatra.styling.colors import resolve_colormap
from matplotlib.colors import Normalize
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from shapely.geometry import Point, Polygon

from digitalearth.scene import TexturedGlobe
from digitalearth.scene.textured_globe import _cull_per_point, _texture_axes


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


class TestCullPerPoint:
    """Which scatter arguments are treated as one-value-per-point."""

    def test_a_shorter_sequence_is_left_alone(self):
        """Only a sequence matching the point count is per-point; anything else is the caller's business."""
        keep = np.array([True, False, True])
        assert _cull_per_point({"s": [1, 2]}, keep)["s"] == [1, 2]

    def test_an_unknown_keyword_is_left_alone(self):
        keep = np.array([True, False])
        assert _cull_per_point({"zorder": [1, 2]}, keep)["zorder"] == [1, 2]


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
        src_lon, src_lat = np.asarray(lonlat.x, dtype=float), np.asarray(lonlat.y, dtype=float)
        tol = 0.25  # a canvas cell (0.125 deg) plus the source's own half-cell
        assert src_lon.min() - tol <= lon.min(), f"drape starts west of the source: {lon.min()}"
        assert lon.max() <= src_lon.max() + tol, f"drape ends east of the source: {lon.max()}"
        assert src_lat.min() - tol <= lat.min(), f"drape starts south of the source: {lat.min()}"
        assert lat.max() <= src_lat.max() + tol, f"drape ends north of the source: {lat.max()}"

    def test_a_0_360_longitude_axis_drapes_the_whole_world(self):
        """0-360 is the usual climate/NWP convention; treating it as -180..180 loses half the globe.

        The band ramps with longitude rather than being constant, so a drape that covers the globe but is
        rotated by 180 degrees -- which a constant array cannot tell apart -- fails too.
        """
        lon = np.linspace(0.0, 359.0, 360, dtype="float32")
        arr = np.repeat(lon[None, :], 180, axis=0)
        ds = Dataset.create_from_array(arr=arr, geo=(0.0, 1.0, 0.0, 90.0, 0.0, -1.0), epsg=4326)
        globe = TexturedGlobe.from_dataset(ds, cmap="viridis", vmin=0.0, vmax=359.0, shape=(180, 360))
        texture = globe.glyph.texture
        assert (texture[..., 3] > 0).mean() == pytest.approx(1.0), "should cover the whole globe"

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
        ds = Dataset.create_from_array(arr=arr, geo=(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0), epsg=4326)
        globe = TexturedGlobe.from_dataset(ds, shape=(180, 360))
        assert (globe.glyph.texture[..., 3] > 0).mean() == pytest.approx(1.0)

    def test_a_raster_beyond_the_antimeridian_lands_west(self):
        """Longitudes 200-210 in the 0-360 frame are -160..-150 on the canvas, not off the edge."""
        arr = np.ones((10, 10), dtype="float32")
        ds = Dataset.create_from_array(arr=arr, geo=(200.0, 1.0, 0.0, 10.0, 0.0, -1.0), epsg=4326)
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
        ds = Dataset.create_from_array(arr=arr, geo=(0.0, 5.0, 0.0, 10.0, 0.0, -5.0), epsg=4326)
        globe = TexturedGlobe.from_dataset(ds, shape=(180, 360))
        assert (globe.glyph.texture[..., 3] > 0).any(), "a single-row raster should still drape"

    def test_a_single_column_raster_drapes(self):
        arr = np.ones((8, 1), dtype="float32")
        ds = Dataset.create_from_array(arr=arr, geo=(0.0, 5.0, 0.0, 40.0, 0.0, -5.0), epsg=4326)
        globe = TexturedGlobe.from_dataset(ds, shape=(180, 360))
        assert (globe.glyph.texture[..., 3] > 0).any(), "a single-column raster should still drape"


    def test_the_poles_and_the_antimeridian_are_not_left_empty(self):
        """The outer cell centres sit on the source's boundary, so the warp drops them without a clamp."""
        arr = np.ones((180, 360), dtype="float32")
        ds = Dataset.create_from_array(arr=arr, geo=(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0), epsg=4326)
        opaque = TexturedGlobe.from_dataset(ds, shape=(180, 360)).glyph.texture[..., 3] > 0
        assert opaque[0].all(), "the north-pole row should be filled"
        assert opaque[-1].all(), "the south-pole row should be filled, not left as a hole"
        assert opaque[:, 0].all(), "the -180 column should be filled"
        assert opaque[:, -1].all(), "the +180 column should be filled, not left as a seam"

    def test_a_regional_raster_keeps_its_transparent_margin(self, dataset):
        """The edge clamp must not spread a regional raster to the poles."""
        opaque = TexturedGlobe.from_dataset(dataset, shape=(360, 720)).glyph.texture[..., 3] > 0
        assert opaque.mean() < 0.01, f"a small raster should stay small, covered {opaque.mean():.4f}"
        assert not opaque[-1].any(), "the south-pole row should stay empty for a regional raster"

    def test_uncovered_cells_stay_transparent(self, dataset):
        """A regional raster leaves the rest of the globe see-through rather than filling it."""
        alpha = TexturedGlobe.from_dataset(dataset, shape=(360, 720)).glyph.texture[..., 3]
        assert (alpha == 0).sum() > alpha.size * 0.9

    def test_nodata_cells_are_transparent(self):
        """The nodata cell must be transparent while its neighbours are opaque, at its own lon/lat."""
        arr = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype="float32")
        ds = Dataset.create_from_array(arr=arr, geo=(0.0, 10.0, 0.0, 20.0, 0.0, -10.0), epsg=4326,
                                       no_data_value=-9999.0)
        globe = TexturedGlobe.from_dataset(ds, shape=(720, 1440))
        alpha = globe.glyph.texture[..., 3]
        lat_axis, lon_axis = _texture_axes(*alpha.shape)

        def alpha_at(lon: float, lat: float) -> float:
            return float(alpha[int(np.abs(lat_axis - lat).argmin()), int(np.abs(lon_axis - lon).argmin())])

        assert alpha_at(5.0, 15.0) > 0, "the top-left valid cell should be opaque"
        assert alpha_at(15.0, 15.0) == 0, "the top-right cell is nodata and must be transparent"
        assert alpha_at(5.0, 5.0) > 0, "the bottom-left valid cell should be opaque"
        assert alpha_at(15.0, 5.0) > 0, "the bottom-right valid cell should be opaque"

    def test_nodata_is_transparent_even_under_an_opaque_bad_colour(self):
        """matplotlib's default 'bad' colour is already transparent, so only an opaque one tests our own
        masking. A colormap with set_bad opaque must still leave nodata see-through."""
        cmap = resolve_colormap("viridis").copy()
        cmap.set_bad("red", alpha=1.0)
        arr = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype="float32")
        ds = Dataset.create_from_array(arr=arr, geo=(0.0, 10.0, 0.0, 20.0, 0.0, -10.0), epsg=4326,
                                       no_data_value=-9999.0)
        globe = TexturedGlobe.from_dataset(ds, cmap=cmap, shape=(720, 1440))
        alpha = globe.glyph.texture[..., 3]
        lat_axis, lon_axis = _texture_axes(*alpha.shape)
        r = int(np.abs(lat_axis - 15.0).argmin())
        c = int(np.abs(lon_axis - 15.0).argmin())
        assert alpha[r, c] == 0, "the nodata cell must be transparent whatever the colormap's bad colour is"

    def test_the_drape_is_not_transposed_or_shifted(self):
        """Pin the orientation: a distinctive value must land at its own lon/lat, not a mirrored one."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        ds = Dataset.create_from_array(arr=arr, geo=(0.0, 10.0, 0.0, 20.0, 0.0, -10.0), epsg=4326)
        globe = TexturedGlobe.from_dataset(ds, cmap="viridis", vmin=1.0, vmax=4.0, shape=(720, 1440))
        texture = globe.glyph.texture
        lat_axis, lon_axis = _texture_axes(*texture.shape[:2])
        expected = resolve_colormap("viridis")(Normalize(vmin=1.0, vmax=4.0)(arr))

        for row, lat in enumerate((15.0, 5.0)):
            for col, lon in enumerate((5.0, 15.0)):
                r = int(np.abs(lat_axis - lat).argmin())
                c = int(np.abs(lon_axis - lon).argmin())
                # The texture is 8-bit, so one colour step (1/255) is the tightest meaningful tolerance.
                assert np.allclose(texture[r, c, :3], expected[row, col, :3], atol=1.0 / 255), (
                    f"cell ({row}, {col}) should appear at lon {lon}, lat {lat}"
                )

    def test_a_dataset_without_a_crs_is_refused(self, dataset, monkeypatch):
        """Without a CRS there is no way to place the raster, so fail loudly instead of guessing 4326."""
        monkeypatch.setattr(type(dataset), "epsg", property(lambda self: None))
        monkeypatch.setattr(type(dataset), "crs", property(lambda self: None))
        with pytest.raises(ValueError, match="has none"):
            TexturedGlobe.from_dataset(dataset)

    def test_a_dataset_whose_crs_has_no_epsg_code_is_accepted(self, dataset, monkeypatch):
        """`.epsg` is None for geostationary/Mollweide too; those reproject fine and must not be rejected."""
        monkeypatch.setattr(type(dataset), "epsg", property(lambda self: None))
        globe = TexturedGlobe.from_dataset(dataset, n_lon=2880, n_lat=1440)
        assert (globe.glyph.texture[..., 3] > 0).any(), "a CRS without an EPSG code should still drape"

    def test_a_none_colormap_falls_back_to_viridis(self, dataset):
        """resolve_colormap returns None only for cmap=None, which must still produce a texture."""
        globe = TexturedGlobe.from_dataset(dataset, cmap=None, n_lon=2880, n_lat=1440)
        fallback = TexturedGlobe.from_dataset(dataset, cmap="viridis", n_lon=2880, n_lat=1440)
        assert np.array_equal(globe.glyph.texture, fallback.glyph.texture), (
            "cmap=None should fall back to viridis, not to some other colormap"
        )

    def test_reversed_colour_bounds_are_refused(self, dataset):
        """vmax <= vmin was silently discarded and replaced, hiding the caller's mistake."""
        with pytest.raises(ValueError, match="vmax must be greater than vmin"):
            TexturedGlobe.from_dataset(dataset, vmin=10.0, vmax=1.0)

    @pytest.mark.parametrize("band, value", [(1, 1.0), (2, 2.0), (3, 3.0)])
    def test_the_requested_band_is_the_one_drawn(self, band, value):
        """Selecting a band before the warp must not change which band ends up on the globe."""
        arr = np.stack([np.full((40, 40), v, dtype="float32") for v in (1.0, 2.0, 3.0)])
        ds = Dataset.create_from_array(arr=arr, geo=(0.0, 0.25, 0.0, 10.0, 0.0, -0.25), epsg=4326)
        globe = TexturedGlobe.from_dataset(ds, band=band, cmap="viridis", vmin=1.0, vmax=3.0,
                                           n_lon=2880, n_lat=1440)
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
        ds = Dataset.create_from_array(arr=arr, geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326)
        with pytest.raises(ValueError, match="no range to colour"):
            TexturedGlobe.from_dataset(ds, **kwargs)

    def test_an_all_nodata_band_warns(self):
        """A fully transparent globe is indistinguishable from a broken one, so say which it is."""
        arr = np.full((4, 4), -9999.0, dtype="float32")
        ds = Dataset.create_from_array(arr=arr, geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326,
                                       no_data_value=-9999.0)
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
        assert not mesh_warnings, f"a mesh that resolves the data should not warn: {mesh_warnings}"

    def test_the_mesh_warning_predicts_what_actually_paints(self, dataset):
        """Tie the warning to reality: when it fires, the drawn sphere really does carry no opaque face."""
        with pytest.warns(RuntimeWarning, match="sphere mesh"):
            globe = TexturedGlobe.from_dataset(dataset, n_lon=180, n_lat=90)
        globe.draw()
        painted = np.asarray(globe.glyph._facecolors)
        assert int((painted[..., 3] > 0).sum()) == 0, "the warning fired but the data did paint"

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
            ds = Dataset.create_from_array(
                arr=np.ones((4, 4), dtype="float32"),
                geo=(lon0, size / 4, 0.0, lat0, 0.0, -size / 4),
                epsg=4326,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                globe = TexturedGlobe.from_dataset(ds, n_lon=180, n_lat=90)
            warned = any("sphere mesh" in str(w.message) for w in caught)
            globe.draw()
            paints = bool((np.asarray(globe.glyph._facecolors)[..., 3] > 0).any())
            if warned == paints:  # warned yet painted, or silent yet blank
                disagreements.append((round(lon0, 2), round(lat0, 2), round(size, 2), warned, paints))
            globe.close()
        assert not disagreements, f"the warning disagreed with the render at {disagreements[:5]}"

    def test_a_constant_band_does_not_divide_by_zero(self):
        """vmin == vmax has no range to normalise against; it must still produce a texture."""
        arr = np.full((4, 4), 5.0, dtype="float32")
        ds = Dataset.create_from_array(arr=arr, geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326)
        globe = TexturedGlobe.from_dataset(ds, shape=(180, 360))
        opaque = globe.glyph.texture[..., 3] > 0
        assert opaque.any(), "a constant band should still drape"
        rgb = globe.glyph.texture[opaque][:, :3]
        assert np.allclose(rgb, rgb[0]), "a constant band should map to one colour, not a gradient"


class TestFromProvider:
    """Tile-basemap textures. The network call is cleopatra's; what is ours is the kwarg split."""

    @pytest.fixture
    def fetched(self, monkeypatch):
        """Capture the world_texture call instead of pulling thousands of tiles."""
        seen = {}

        def _fake(provider=None, **kwargs):
            seen["provider"], seen["kwargs"] = provider, kwargs
            return np.zeros((16, 32, 3), dtype=np.uint8)

        monkeypatch.setattr("digitalearth.scene.textured_globe.world_texture", _fake)
        return seen

    def test_defaults_to_a_bulk_permitting_provider(self, fetched):
        """OpenStreetMap forbids whole-world fetches, so the default must not be it."""
        TexturedGlobe.from_provider()
        assert fetched["provider"] == "Esri.WorldImagery"

    def test_texture_options_go_to_the_fetcher(self, fetched):
        TexturedGlobe.from_provider("Esri.WorldImagery", zoom=3, texture_n_lon=720, texture_n_lat=360,
                                    cache=False)
        assert fetched["kwargs"] == {"zoom": 3, "n_lon": 720, "n_lat": 360, "cache": False}

    def test_glyph_options_do_not_leak_into_the_fetcher(self, fetched):
        TexturedGlobe.from_provider(zoom=2, tilt_deg=0.0)
        assert "tilt_deg" not in fetched["kwargs"]

    def test_n_lon_means_the_mesh_here_as_everywhere_else(self, fetched):
        """The same keyword must not mean the mesh in one constructor and the texture in another."""
        globe = TexturedGlobe.from_provider(texture_n_lon=720, n_lon=48, n_lat=24)
        assert fetched["kwargs"]["n_lon"] == 720, "texture_n_lon should size the fetched grid"
        assert globe.glyph.n_lon == 48, "n_lon should size the sphere mesh"
        assert globe.glyph.n_lat == 24, "n_lat should size the sphere mesh"

    def test_mesh_size_is_not_leaked_to_the_fetcher(self, fetched):
        TexturedGlobe.from_provider(n_lon=48, n_lat=24)
        assert "n_lon" not in fetched["kwargs"], "the mesh size must not reach the fetcher"
        assert "n_lat" not in fetched["kwargs"], "the mesh size must not reach the fetcher"


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
            np.array([[np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon)),
                       np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon)),
                       np.sin(np.deg2rad(lat))]]),
            spin=40.0,
        )
        assert np.allclose(globe.project(lon, lat, spin=40.0), expected)

    def test_spin_moves_the_point(self, globe):
        assert not np.allclose(globe.project(0.0, 0.0, spin=0.0), globe.project(0.0, 0.0, spin=90.0))

    def test_projection_defaults_to_the_drawn_spin(self, globe):
        """spin=0 by default silently placed overlays on a face the reader could not see."""
        globe.draw(spin=90.0)
        assert np.allclose(globe.project(10.0, 20.0), globe.project(10.0, 20.0, spin=90.0)), (
            "project() should default to the spin the globe was drawn at"
        )

    def test_an_explicit_spin_still_wins(self, globe):
        globe.draw(spin=90.0)
        assert np.allclose(globe.project(10.0, 20.0, spin=0.0),
                           TexturedGlobe(globe.glyph.texture, tilt_deg=globe.glyph.tilt_deg,
                                         n_lon=8, n_lat=4).project(10.0, 20.0, spin=0.0))

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
        assert len(collection.get_offsets()) == 0

    def test_far_side_points_are_kept_when_asked(self, globe):
        globe.draw(elev=0.0, azim=0.0)
        collection = globe.points([180.0], lat=[0.0], hide_far_side=False)
        assert len(collection.get_offsets()) == 1

    def test_per_point_colours_are_culled_with_their_points(self, globe):
        """Dropping a far-side point must drop its colour too, or the arrays desynchronise."""
        globe.draw(elev=0.0, azim=0.0)
        collection = globe.points([0.0, 180.0], lat=[0.0, 0.0], c=[1.0, 2.0], hide_far_side=True)
        assert len(collection.get_offsets()) == 1

    @pytest.mark.parametrize(
        "key, values, expected",
        [
            pytest.param("s", [10, 20, 30, 40], [10, 30, 40], id="s"),
            pytest.param("linewidths", [1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 4.0], id="linewidths"),
            pytest.param("alpha", [0.1, 0.2, 0.3, 0.4], [0.1, 0.3, 0.4], id="alpha"),
            pytest.param("c", [1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 4.0], id="c"),
            pytest.param("linestyles", ["-", "--", ":", "-."], ["-", ":", "-."], id="linestyles"),
        ],
    )
    def test_per_point_values_are_culled_to_the_kept_points(self, globe, key, values, expected):
        """Assert the surviving values, not the point count: a count passes even with the cull removed."""
        globe.draw(elev=0.0, azim=0.0)
        culled = _cull_per_point({key: values}, np.array([True, False, True, True]))
        assert list(np.asarray(culled[key])) == list(np.asarray(expected)), (
            f"{key} should keep {expected}, got {list(np.asarray(culled[key]))}"
        )

    def test_a_far_side_point_does_not_shift_the_linewidths(self, globe):
        """End to end: the second point is hidden, so its width must not land on the third."""
        globe.draw(elev=0.0, azim=0.0)
        collection = globe.points([0.0, 180.0, 10.0, 20.0], lat=[0.0] * 4,
                                  linewidths=[1.0, 2.0, 3.0, 4.0])
        assert list(np.atleast_1d(collection.get_linewidths())) == [1.0, 3.0, 4.0]

    @pytest.mark.parametrize(
        "colour", [[1.0, 0.0, 0.0, 1.0], (1.0, 0.0, 0.0, 1.0), np.array([1.0, 0.0, 0.0, 1.0])],
        ids=["list", "tuple", "ndarray"],
    )
    def test_a_single_rgba_colour_survives_the_cull_whatever_its_container(self, globe, colour):
        """Culling a 4-element red to 3 elements renders magenta, silently and in any container type."""
        globe.draw(elev=0.0, azim=0.0)
        collection = globe.points([0.0, 180.0, 10.0, 20.0], lat=[0.0] * 4, color=colour)
        assert np.allclose(collection.get_facecolors()[0], [1.0, 0.0, 0.0, 1.0]), (
            f"expected red, got {collection.get_facecolors()[0]} (magenta means the RGBA was culled)"
        )

    def test_c_is_culled_even_as_a_four_element_sequence(self, globe):
        """matplotlib value-maps a length-matching `c`, so it is per-point data, not one RGBA colour."""
        culled = _cull_per_point({"c": (0.1, 0.2, 0.3, 0.4)}, np.array([True, False, True, True]))
        assert list(np.asarray(culled["c"])) == [0.1, 0.3, 0.4]

    def test_a_sized_but_unindexable_value_is_left_alone(self, globe):
        """A set has a length but cannot be sliced; it must be passed through, not crash the cull."""
        culled = _cull_per_point({"s": {1, 2, 3, 4}}, np.array([True, False, True, True]))
        assert culled["s"] == {1, 2, 3, 4}

    def test_a_value_with_no_length_is_left_alone(self, globe):
        """An arbitrary object with no len() is a scalar as far as the cull is concerned."""
        sentinel = object()
        culled = _cull_per_point({"s": sentinel}, np.array([True, False]))
        assert culled["s"] is sentinel

    def test_a_pandas_series_is_culled(self, globe):
        """Colours often come straight from a dataframe column."""
        pd = pytest.importorskip("pandas")
        culled = _cull_per_point({"c": pd.Series([1.0, 2.0, 3.0, 4.0])}, np.array([True, False, True, True]))
        assert list(np.asarray(culled["c"])) == [1.0, 3.0, 4.0]

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"color": "red"}, id="scalar-name"),
            pytest.param({"s": 30}, id="scalar-size"),
        ],
    )
    def test_scalar_arguments_pass_through_unculled(self, globe, kwargs):
        """A single colour or size applies to every point; slicing it would change what was asked for."""
        globe.draw(elev=0.0, azim=0.0)
        collection = globe.points([0.0, 180.0, 10.0, 20.0], lat=[0.0] * 4, **kwargs)
        assert len(collection.get_offsets()) == 3, f"{kwargs} should still draw the 3 near-side points"

    def test_a_feature_collection_is_accepted(self, globe, points_fc):
        globe.draw()
        assert globe.points(points_fc) is not None

    def test_a_projected_feature_collection_is_reprojected(self, globe, points_fc):
        """The fixture is UTM 18N; its coordinates must become lon/lat before they reach the sphere."""
        assert points_fc.epsg == 32618, "fixture precondition: the points are in a projected CRS"
        lon, lat = TexturedGlobe._as_lonlat(points_fc, None)
        assert np.all(np.abs(lon) <= 180), f"longitudes are still projected: {lon.min()}..{lon.max()}"
        assert np.all(np.abs(lat) <= 90), f"latitudes are still projected: {lat.min()}..{lat.max()}"

    def test_a_lonlat_feature_collection_passes_through(self, points_fc):
        """Already in 4326, so the reprojection branch must be skipped rather than re-warping."""
        lonlat = points_fc.to_crs(4326)
        lon, lat = TexturedGlobe._as_lonlat(lonlat, None)
        assert np.allclose(lon, lonlat.geometry.x.to_numpy()), "lon should be untouched for a 4326 collection"
        assert np.allclose(lat, lonlat.geometry.y.to_numpy()), "lat should be untouched for a 4326 collection"

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

        square = Polygon([(500000.0, 0.0), (501000.0, 0.0), (501000.0, 1000.0), (500000.0, 1000.0)])
        frame = FeatureCollection(geometry=[square], crs="EPSG:32618")
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            lon, lat = TexturedGlobe._as_lonlat(frame, None)
        geographic = [w for w in caught if "geographic CRS" in str(w.message)]
        assert not geographic, f"centroids should be taken in the projected CRS: {geographic}"
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

    def test_animate_works_when_the_constructor_supplied_the_axes(self, flat_texture):
        """The guard that skipped creating an axes left ax=None and animate crashed on ax.get_figure()."""
        plt.close("all")
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        owned = TexturedGlobe(flat_texture, n_lon=8, n_lat=4, fig=fig, ax=ax)
        assert owned.animate(n_frames=2, interval=100) is not None
        assert owned.ax is ax, "animate should reuse the constructor's axes"

    def test_redrawing_onto_the_same_figure_does_not_close_it(self, globe):
        """draw(ax=) closed self.fig before drawing — including when the new axes lives on that figure."""
        plt.close("all")
        fig, _ = globe.draw()
        second = fig.add_subplot(222, projection="3d")
        globe.draw(ax=second)
        assert plt.fignum_exists(fig.number), "the figure being drawn on must not be closed"

    def test_animating_onto_a_caller_axes_releases_our_figure(self, globe):
        """The release was only wired into draw(), so animate() leaked the figure draw() had created."""
        plt.close("all")
        globe.draw()
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        globe.animate(ax, n_frames=2, interval=100)
        globe.close()
        assert plt.get_fignums() == [fig.number], (
            f"only the caller's figure should remain, got {plt.get_fignums()}"
        )

    def test_close_leaves_a_caller_supplied_figure_alone(self, globe):
        """The caller may have other subplots on that figure; it is not ours to close."""
        plt.close("all")
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        globe.draw(ax=ax)
        globe.close()
        assert plt.fignum_exists(fig.number), "a caller-supplied figure must survive close()"

    @pytest.mark.parametrize("via", ["ax", "fig"])
    def test_close_leaves_a_constructor_supplied_figure_alone(self, flat_texture, via):
        """The glyph stores a ctor fig/ax and draws on it even when no later call passes one."""
        plt.close("all")
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        kwargs = {"ax": ax, "fig": fig} if via == "ax" else {"fig": fig}
        owned = TexturedGlobe(flat_texture, n_lon=8, n_lat=4, **kwargs)
        owned.draw()
        owned.close()
        assert plt.fignum_exists(fig.number), "a constructor-supplied figure must survive close()"

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
        globe.animate(n_frames=2, interval=100)
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
        assert plt.fignum_exists(fig.number), "a caller-supplied figure must survive close()"

    def test_redrawing_does_not_leak_the_previous_figure(self, globe):
        """draw() rebinds fig/ax, so the one it replaces would otherwise be held by pyplot forever."""
        plt.close("all")
        globe.draw()
        globe.draw()
        globe.draw()
        assert len(plt.get_fignums()) == 1, f"expected 1 open figure, got {len(plt.get_fignums())}"

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
        assert plt.get_fignums() == [], "the figure should be closed even when the block raised"

    def test_animating_twice_does_not_leak_the_first_figure(self, globe):
        """animate() creates its own figure when given no axes, so the previous one must be released."""
        plt.close("all")
        globe.animate(n_frames=2, interval=100)
        globe.animate(n_frames=2, interval=100)
        assert len(plt.get_fignums()) == 1, f"expected 1 open figure, got {len(plt.get_fignums())}"

    def test_animate_records_its_starting_spin(self, globe):
        """Otherwise an overlay added after animate() is placed at whatever spin draw() last used."""
        globe.animate(n_frames=2, interval=100, start_spin=42.0)
        assert globe._spin == pytest.approx(42.0)

    def test_a_zero_animation_interval_is_refused(self, globe):
        """interval=0 used to raise ZeroDivisionError from the frame-rate bookkeeping."""
        with pytest.raises(ValueError, match="positive number of milliseconds"):
            globe.animate(n_frames=2, interval=0)

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
        """The globe delegates to digitalearth.animation rather than reimplementing the encode."""
        seen = {}
        monkeypatch.setattr("digitalearth.scene.textured_globe.save_animation",
                            lambda anim, path, **kw: seen.update(anim=anim, path=path, **kw) or path)
        globe.animate(n_frames=2, interval=125)
        globe.save_animation("globe.mp4", gif="globe.gif")
        assert seen["path"] == "globe.mp4", "the video path should be forwarded"
        assert seen["gif"] == "globe.gif", "the gif path should be forwarded"
        assert seen["anim"] is globe._animation, "the saver must receive this globe's animation"

    def test_save_animation_defaults_to_the_animations_own_rate(self, globe, monkeypatch):
        """interval=125 ms is 8 fps; the saved clip should match what animate() was built for."""
        seen = {}
        monkeypatch.setattr("digitalearth.scene.textured_globe.save_animation",
                            lambda anim, path, **kw: seen.update(kw) or path)
        globe.animate(n_frames=2, interval=125)
        globe.save_animation("globe.mp4")
        assert seen["fps"] == pytest.approx(8.0), f"expected 8 fps from a 125 ms interval, got {seen['fps']}"

    def test_an_explicit_rate_overrides_the_animations(self, globe, monkeypatch):
        seen = {}
        monkeypatch.setattr("digitalearth.scene.textured_globe.save_animation",
                            lambda anim, path, **kw: seen.update(kw) or path)
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
