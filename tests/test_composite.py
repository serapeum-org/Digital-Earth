"""Tests for T7.1 — Map RGB/HSV composites from a multiband pyramids raster."""

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

from digitalearth.static import Map
from digitalearth.static.maps.raster import _stretch_to_unit, channel_limits


@pytest.fixture
def rgb_dataset(dataset):
    """A 3-band dataset built from the acc4000 grid (three scaled copies).

    Returns:
        Dataset: a 3-band pyramids Dataset in the acc4000 CRS.
    """
    base = np.nan_to_num(dataset.read_array(band=0).astype("float32"))
    arr3 = np.stack([base, base * 0.5, base * 0.25])  # (3, rows, cols)
    return Dataset.from_array(arr=arr3, geo_ref=GeoReference(geo=dataset.geotransform, epsg=dataset.epsg))


def test_stretch_to_unit_range():
    """_stretch_to_unit maps each channel into [0, 1]."""
    stack = np.dstack([np.arange(100.0).reshape(10, 10) for _ in range(3)])
    out = _stretch_to_unit(stack)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.shape == stack.shape


def test_stretch_to_unit_constant_band():
    """A constant channel (equal percentiles) stretches to all-zeros without dividing by zero."""
    stack = np.dstack([np.full((4, 4), 7.0), np.zeros((4, 4)), np.ones((4, 4))])
    out = _stretch_to_unit(stack)
    assert np.all(out[..., 0] == 0.0)
    assert np.isfinite(out).all()


def test_rgb_composite(rgb_dataset):
    """rgb_composite renders a 3-band raster as a single RGB image with correct (rows, cols, 3) shape."""
    m = Map(crs=rgb_dataset.epsg)
    m.rgb_composite(rgb_dataset)
    assert len(m.layers) == 1
    assert len(m.ax.images) == 1
    # band-first must be transposed back to band-last (rows, cols, 3), not a garbled (cols, 3, 3)
    assert m.ax.images[-1].get_array().shape == (rgb_dataset.rows, rgb_dataset.columns, 3)


def test_hsv_composite(rgb_dataset):
    """hsv_composite renders a 3-band raster as an HSV-derived RGB image with correct shape."""
    m = Map(crs=rgb_dataset.epsg)
    m.hsv_composite(rgb_dataset)
    assert len(m.layers) == 1
    assert len(m.ax.images) == 1
    assert m.ax.images[-1].get_array().shape == (rgb_dataset.rows, rgb_dataset.columns, 3)


def test_rgb_composite_custom_band_order(rgb_dataset):
    """A custom band order still produces one RGB image."""
    m = Map(crs=rgb_dataset.epsg)
    m.rgb_composite(rgb_dataset, bands=(3, 2, 1))
    assert len(m.ax.images) == 1


def test_rgb_composite_mask_flag_controls_nodata(rgb_dataset):
    """rgb_composite(mask_nodata=...) toggles whether nodata cells are excluded from the stretch (review L2).

    The default masks nodata (cells can be NaN/transparent); mask_nodata=False keeps the raw values so the
    rendered RGB array is fully finite.
    """
    import numpy as np

    m = Map(crs=rgb_dataset.epsg)
    m.rgb_composite(rgb_dataset, mask_nodata=False)
    arr = np.asarray(m.ax.images[-1].get_array(), dtype="float64")
    assert np.isfinite(arr).all(), "mask_nodata=False should keep every cell finite (raw stretch)"


def test_hsv_composite_accepts_mask_flag(rgb_dataset):
    """hsv_composite accepts the mask_nodata flag and still renders one image (review L2)."""
    m = Map(crs=rgb_dataset.epsg)
    m.hsv_composite(rgb_dataset, mask_nodata=False)
    assert len(m.ax.images) == 1, "hsv_composite should still render with mask_nodata=False"


def test_channel_limits_one_pair_per_channel():
    """channel_limits returns the 2-98 percentile (lo, hi) of every channel, in channel order."""
    stack = np.dstack([np.arange(100.0).reshape(10, 10) * scale for scale in (1.0, 2.0, 3.0)])
    limits = channel_limits(stack)
    assert len(limits) == 3, f"expected one pair per channel, got {limits!r}"
    assert all(lo < hi for lo, hi in limits), f"each channel needs a real span: {limits!r}"
    assert limits[1][1] == pytest.approx(limits[0][1] * 2.0), "channel 1 is twice channel 0"


def test_stretch_to_unit_uses_given_limits():
    """Passing limits replaces the per-call percentile scan, so the same values map to a fixed output."""
    stack = np.dstack([np.arange(100.0).reshape(10, 10) for _ in range(3)])
    limits = [(0.0, 200.0)] * 3
    out = _stretch_to_unit(stack, limits)
    assert out.max() == pytest.approx(99.0 / 200.0), "the given hi (200) must set the white point"
    assert out.min() == pytest.approx(0.0), "the given lo (0) must set the black point"
    assert out.max() < _stretch_to_unit(stack).max(), "the per-call stretch would push the max to 1.0"


def test_stretch_to_unit_given_degenerate_limits():
    """Degenerate (lo == hi) limits are widened instead of dividing by zero."""
    stack = np.dstack([np.full((4, 4), 7.0) for _ in range(3)])
    out = _stretch_to_unit(stack, [(7.0, 7.0)] * 3)
    assert np.isfinite(out).all(), "a zero-span limit must not produce NaN/inf"


def test_frozen_limits_preserve_relative_brightness():
    """Two exposures of one scene keep their brightness ratio under shared limits, not under per-frame ones."""
    base = np.dstack([np.arange(100.0).reshape(10, 10) for _ in range(3)])
    bright, dim = base, base * 0.4
    limits = channel_limits(bright)
    assert np.nanmean(_stretch_to_unit(dim, limits)) < np.nanmean(_stretch_to_unit(bright, limits)) * 0.75
    assert np.nanmean(_stretch_to_unit(dim)) == pytest.approx(np.nanmean(_stretch_to_unit(bright)))


def test_rgb_composite_accepts_frozen_limits(rgb_dataset):
    """rgb_composite(limits=...) stretches on the given bounds rather than its own percentiles."""
    wide = Map(crs=rgb_dataset.epsg)
    wide.rgb_composite(rgb_dataset, limits=[(0.0, 1e6)] * 3)
    own = Map(crs=rgb_dataset.epsg)
    own.rgb_composite(rgb_dataset)
    wide_mean = np.nanmean(np.asarray(wide.ax.images[-1].get_array(), dtype="float64"))
    own_mean = np.nanmean(np.asarray(own.ax.images[-1].get_array(), dtype="float64"))
    assert wide_mean < own_mean, "a far wider white point must render darker than the per-call stretch"


def test_hsv_composite_accepts_frozen_limits(rgb_dataset):
    """hsv_composite takes the same frozen limits and still renders one image."""
    m = Map(crs=rgb_dataset.epsg)
    m.hsv_composite(rgb_dataset, limits=[(0.0, 1e6)] * 3)
    assert len(m.ax.images) == 1, "hsv_composite should render with explicit limits"


def test_channel_limits_all_nodata_channel_is_nan_not_a_warning():
    """A channel with no finite cell yields (nan, nan) quietly — no All-NaN RuntimeWarning escapes."""
    import warnings

    stack = np.dstack([np.arange(100.0).reshape(10, 10), np.full((10, 10), np.nan), np.ones((10, 10))])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        limits = channel_limits(stack)
    assert np.isnan(limits[1]).all(), f"an all-nodata channel should report (nan, nan), got {limits[1]}"
    assert np.isfinite(limits[0]).all(), "a live channel next to a dead one must keep real bounds"


def test_channel_limits_ignores_infinities():
    """Infinities are dropped alongside NaN, so one inf cell cannot become the white point."""
    band = np.arange(100.0).reshape(10, 10).copy()
    band[0, 0] = np.inf
    limits = channel_limits(np.dstack([band, band, band]))
    assert np.isfinite(limits[0]).all(), f"inf leaked into the bounds: {limits[0]}"
    assert limits[0][1] < 100.0, f"the white point should come from the finite values, got {limits[0][1]}"


def test_stretch_to_unit_survives_non_finite_limits():
    """Non-finite limits fall back instead of dividing by nan, leaving live channels usable."""
    stack = np.dstack([np.arange(100.0).reshape(10, 10), np.full((10, 10), np.nan), np.ones((10, 10))])
    out = _stretch_to_unit(stack, [(np.nan, np.nan)] * 3)
    assert np.isnan(out[..., 1]).all(), "an all-nodata channel stays NaN (it renders transparent)"
    assert np.isfinite(out[..., 0]).all(), "a live channel must not be poisoned by the fallback"
    assert out[..., 0].min() >= 0.0 and out[..., 0].max() <= 1.0, "the fallback must still clip into [0, 1]"


@pytest.mark.parametrize("method", ["rgb_composite", "hsv_composite"])
@pytest.mark.parametrize("bands", [(1,), (1, 2), (1, 2, 3, 1)])
def test_composite_rejects_a_wrong_band_count(rgb_dataset, method, bands):
    """Both composites refuse a band list that is not exactly three, naming themselves in the message."""
    m = Map(crs=rgb_dataset.epsg)
    with pytest.raises(ValueError, match=f"{method}\\(\\) needs exactly three bands"):
        getattr(m, method)(rgb_dataset, bands=bands)
