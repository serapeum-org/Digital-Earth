"""ST-5 — a static field render reads the band at the resolution the figure will draw it.

``draw_field`` decoded every cell of every raster through ``Scene._prepare``, then handed the whole array to
matplotlib to be resampled down onto a few hundred thousand device pixels. The interactive tier has read
through :class:`~digitalearth.base.sources.view.SourceView` since its viewport loop existed — the window, the
canvas and the budget as a :class:`~digitalearth.base.spec.target.RenderTarget` request, answered through
pyramids' windowed read — and this tier had no reference to any of it.

The decimation is deliberately *not* applied to every raster: below twice the canvas on a side the decimated
read and the full read differ by less than one output pixel of detail, so a raster that size is still read
whole and every figure already drawn stays exactly as it was.
"""

import numpy as np
import pytest
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset

from digitalearth import Map

#: Degrees the test rasters span on each side, whatever their cell count — so a 1200-cell raster is a fine grid
#: over a small region rather than a grid running off the end of the world.
_SPAN = 20.0


def _raster(side, *, path=None, name="elevation"):
    """Build a square single-band raster spanning :data:`_SPAN` degrees in EPSG:4326.

    Args:
        side: Cells on each side.
        path: Where to write it, or ``None`` to keep it in memory.
        name: The band's name, which is the identity the style lookup matches on.

    Returns:
        The pyramids ``Dataset``.
    """
    values = np.arange(float(side * side)).reshape(side, side)
    dataset = Dataset.from_array(
        arr=values,
        geo_ref=GeoReference(
            top_left_corner=(0.0, _SPAN), cell_size=_SPAN / side, epsg=4326
        ),
        no_data_value=-9999.0,
    )
    dataset.band_names = [name]
    if path is None:
        return dataset
    dataset.to_file(str(path))
    return Dataset.read_file(str(path))


def _drawn_shape(scene, layer_id):
    """Return the shape of the array a drawn field put on the axes.

    Args:
        scene: The map that drew it.
        layer_id: The layer to look up.

    Returns:
        The ``(rows, columns)`` the artist holds.
    """
    artist = scene._renderer.drawn[layer_id].artist
    return tuple(np.asarray(artist.get_array()).shape)


class TestASmallRasterIsStillReadWhole:
    """Every figure already drawn came from a full read, and none of them may move."""

    def test_a_raster_under_the_threshold_keeps_every_cell(self, tmp_path):
        """A raster the canvas can show at full detail is drawn at full detail.

        Args:
            tmp_path: pytest's temporary directory, holding the raster on disk.
        """
        dataset = _raster(64, path=tmp_path / "small.tif")
        with Map(crs=4326, figsize=(4, 4)) as scene:
            scene.field(dataset, name="small")
            assert _drawn_shape(scene, "small") == (64, 64)


class TestALargeRasterIsReadThroughAWindow:
    """ST-5 — the read is sized by the figure, not by the file."""

    def test_the_read_is_decimated_rather_than_fully_decoded(self, tmp_path):
        """A raster far larger than the canvas is not fully decoded to fill it.

        Args:
            tmp_path: pytest's temporary directory, holding the raster on disk.

        Test scenario:
            The drawn array has to be smaller than the file and no smaller than the canvas — bounded on both
            sides, since a read that returned a single cell would also be "smaller than the file".
        """
        dataset = _raster(1200, path=tmp_path / "big.tif")
        with Map(crs=4326, figsize=(4, 4)) as scene:
            scene.field(dataset, name="big")
            rows, columns = _drawn_shape(scene, "big")
        assert rows < 1200, f"the whole raster was decoded: {rows}x{columns}"
        assert rows >= 400, f"the read fell below the canvas it fills: {rows}x{columns}"

    def test_a_bigger_figure_reads_more_of_the_same_raster(self, tmp_path):
        """The read follows the canvas, which is what makes it view-dependent.

        Args:
            tmp_path: pytest's temporary directory.

        Test scenario:
            Two canvases over one file, compared against each other: a constant decimation would satisfy any
            single expected shape, and only the comparison shows the read is sized by the figure.
        """
        dataset = _raster(1200, path=tmp_path / "big.tif")
        with Map(crs=4326, figsize=(2, 2)) as small:
            small.field(dataset, name="f")
            fewer = _drawn_shape(small, "f")
        with Map(crs=4326, figsize=(4, 4)) as large:
            large.field(dataset, name="f")
            more = _drawn_shape(large, "f")
        assert fewer[0] < more[0], f"{fewer} vs {more}"

    def test_the_decimated_read_is_still_placed_where_the_data_is(self, tmp_path):
        """A read at another resolution covers the same ground, or the field lands in the wrong place.

        Args:
            tmp_path: pytest's temporary directory.
        """
        dataset = _raster(1200, path=tmp_path / "big.tif")
        with Map(crs=4326, figsize=(4, 4)) as scene:
            scene.field(dataset, name="big")
            west, east, south, north = scene._renderer.drawn["big"].artist.get_extent()
        assert (round(west), round(east)) == (0, round(_SPAN)), (west, east)
        assert (round(south), round(north)) == (0, round(_SPAN)), (south, north)

    def test_a_raster_in_another_crs_is_warped_a_window_at_a_time(self, tmp_path):
        """Warping the whole raster first would decode every cell to throw most of them away.

        Args:
            tmp_path: pytest's temporary directory.

        Test scenario:
            The display CRS is Web Mercator and the raster is lon/lat, so the read goes through pyramids'
            lazily warped view — each window warps only itself.
        """
        dataset = _raster(1200, path=tmp_path / "big.tif")
        with Map(crs=3857, figsize=(4, 4)) as scene:
            scene.field(dataset, name="big")
            rows, _ = _drawn_shape(scene, "big")
        assert rows < 1200, f"the whole raster was warped and decoded: {rows}"

    def test_the_bands_identity_survives_the_decimated_read(self, tmp_path):
        """A windowed read comes back as a bare array, and a band with no name has no style to match.

        Args:
            tmp_path: pytest's temporary directory.

        Test scenario:
            ``auto_style`` matches on the band's name, so a decimated read that lost it would colour a
            recognised variable with the fallback colormap. The two reads of the same variable are compared
            with each other rather than against a literal, so the test says "the same", not "coolwarm".
        """
        small = _raster(64, path=tmp_path / "small.tif", name="t2m")
        big = _raster(1200, path=tmp_path / "big.tif", name="t2m")
        with Map(crs=4326, figsize=(4, 4)) as whole:
            whole.field(small, name="f")
            undecimated = whole._renderer.drawn["f"].artist.get_cmap().name
        with Map(crs=4326, figsize=(4, 4)) as part:
            part.field(big, name="f")
            decimated = part._renderer.drawn["f"].artist.get_cmap().name
        assert decimated == undecimated, (
            f"the decimated read lost the band's identity: {decimated!r} vs {undecimated!r}"
        )


class TestTheCanvasMeasurement:
    """The budget comes from the figure, so what it measures is worth pinning."""

    def test_the_canvas_is_measured_in_device_pixels(self):
        """Inches alone would ask for a ten-thousandth of the cells a figure actually draws.

        Test scenario:
            The same figure size at two resolutions must ask for different canvases; measuring only the inches
            would answer the same for both.
        """
        from digitalearth.static.maps.raster import _canvas_pixels

        class Figure:
            """The two attributes the measurement reads off a matplotlib figure."""

            def __init__(self, dpi):
                """Hold the resolution.

                Args:
                    dpi: Device pixels per inch.
                """
                self.dpi = dpi

            @staticmethod
            def get_size_inches():
                """Return a four-by-three figure.

                Returns:
                    The width and height in inches.
                """
                return (4.0, 3.0)

        coarse = _canvas_pixels(type("S", (), {"fig": Figure(50)})())
        fine = _canvas_pixels(type("S", (), {"fig": Figure(200)})())
        assert coarse == (200, 150), coarse
        assert fine == (800, 600), fine

    def test_a_figure_that_cannot_be_measured_falls_back_rather_than_failing(self):
        """A scene with no figure yet still has to yield a budget, not a traceback."""
        from digitalearth.static.maps.raster import _FALLBACK_CANVAS, _canvas_pixels

        assert _canvas_pixels(type("S", (), {"fig": None})()) == _FALLBACK_CANVAS

    @pytest.mark.parametrize("cells,expected", [(1_000_000, True), (100, False)])
    def test_only_a_raster_well_past_the_canvas_is_decimated(self, cells, expected):
        """Below twice the canvas on a side the resample costs more than it saves.

        Args:
            cells: How many cells the raster reports.
            expected: Whether it should be decimated.
        """
        from digitalearth.base.spec import RenderTarget
        from digitalearth.static.maps.raster import _worth_decimating

        side = int(cells**0.5)
        raster = type(
            "R", (), {"rows": side, "columns": side, "read_part": lambda *a, **k: None}
        )()
        target = RenderTarget("image", width=400, height=400)
        assert _worth_decimating(raster, target) is expected

    def test_a_source_with_no_windowed_read_is_never_decimated(self):
        """Without pyramids' ``read_part`` there is no window to read, whatever the size says."""
        from digitalearth.base.spec import RenderTarget
        from digitalearth.static.maps.raster import _worth_decimating

        raster = type("R", (), {"rows": 10_000, "columns": 10_000})()
        assert (
            _worth_decimating(raster, RenderTarget("image", width=400, height=400))
            is False
        )

    def test_a_grid_that_reports_no_integer_size_is_never_decimated(self):
        """A size that cannot be measured is not one to decide a read on."""
        from digitalearth.base.spec import RenderTarget
        from digitalearth.static.maps.raster import _worth_decimating

        raster = type(
            "R", (), {"rows": "many", "columns": 3, "read_part": lambda *a, **k: None}
        )()
        assert (
            _worth_decimating(raster, RenderTarget("image", width=400, height=400))
            is False
        )


class TestTheOffLimbDecisionStaysWithTheFullRead:
    """A raster hidden by the projection must still be skipped, however it was going to be read."""

    @staticmethod
    def _hiding_crs():
        """Return a display CRS that shows none of the test rasters' ground.

        Returns:
            An orthographic projection centred on the far side of the globe.
        """
        from digitalearth import projections

        return projections.orthographic(lon=-175, lat=-70)

    def test_a_large_off_limb_raster_draws_nothing(self, tmp_path):
        """The lazy warp reports a failed transform, which only the full read reads as "nothing to draw".

        Args:
            tmp_path: pytest's temporary directory.

        Test scenario:
            A raster past the canvas takes the windowed read, and pyramids' lazily warped view raises a bare
            ``RuntimeError`` about points that would not transform. Left as it came, that surfaced from
            ``imshow`` as an error where the tier has always skipped the layer with a warning instead.
        """
        dataset = _raster(1200, path=tmp_path / "big.tif")
        with Map(crs=self._hiding_crs(), globe=True, figsize=(4, 4)) as scene:
            assert scene.imshow(dataset) is None, "an off-limb field must draw nothing"
            assert not scene.ax.images, "no raster should have been drawn"

    def test_strict_still_refuses_a_large_off_limb_raster(self, tmp_path):
        """``strict=True`` asks for the refusal, and the windowed read must not turn it into a skip.

        Args:
            tmp_path: pytest's temporary directory.
        """
        from digitalearth.base.crs import OffLimbError

        dataset = _raster(1200, path=tmp_path / "big.tif")
        with Map(
            crs=self._hiding_crs(), globe=True, figsize=(4, 4), strict=True
        ) as scene:
            with pytest.raises(OffLimbError):
                scene.imshow(dataset)


class TestTheReadLeavesNothingBehind:
    """A draw happens many times over a figure's life, so what it registers is what it leaks."""

    def test_no_object_is_pinned_in_the_process_registry_per_draw(self, tmp_path):
        """A ``DataRef`` minted per draw would hold every warped view alive for the session.

        Args:
            tmp_path: pytest's temporary directory.

        Test scenario:
            ``DataRef.to_object`` keys the process-global object table by a fresh id and keeps a strong
            reference, so registering the warped view here — once per *draw*, not once per layer — would pin
            one per frame of an animation. The view is never re-read on this tier, so it needs no reference at
            all; the count is compared before and after two draws of a raster large enough to be decimated.
        """
        from digitalearth.base.registry import _OBJECTS

        dataset = _raster(1200, path=tmp_path / "big.tif")
        before = len(_OBJECTS)
        for _ in range(2):
            with Map(crs=4326, figsize=(4, 4)) as scene:
                scene.field(dataset, name="f")
        assert len(_OBJECTS) == before, (
            f"the read pinned {len(_OBJECTS) - before} object(s) in the registry"
        )
