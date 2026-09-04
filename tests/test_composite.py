"""Tests for T7.1 — Map RGB/HSV composites from a multiband pyramids raster."""

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

from digitalearth.scene import Map
from digitalearth.scene.maps.raster import _stretch_to_unit


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
