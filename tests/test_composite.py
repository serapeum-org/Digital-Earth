"""Tests for T7.1 — Map RGB/HSV composites from a multiband pyramids raster."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalearth.scene import Map
from digitalearth.scene.map import _stretch_to_unit


@pytest.fixture
def rgb_dataset(dataset):
    """A 3-band dataset built from the acc4000 grid (three scaled copies).

    Returns:
        Dataset: a 3-band pyramids Dataset in the acc4000 CRS.
    """
    base = np.nan_to_num(dataset.read_array(band=0).astype("float32"))
    arr3 = np.stack([base, base * 0.5, base * 0.25])  # (3, rows, cols)
    return Dataset.create_from_array(arr=arr3, geo=dataset.geotransform, epsg=dataset.epsg)


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
    """rgb_composite renders a 3-band raster as a single RGB image layer."""
    m = Map(crs=rgb_dataset.epsg)
    m.rgb_composite(rgb_dataset)
    assert len(m.layers) == 1
    assert len(m.ax.images) == 1


def test_hsv_composite(rgb_dataset):
    """hsv_composite renders a 3-band raster as an HSV-derived RGB image layer."""
    m = Map(crs=rgb_dataset.epsg)
    m.hsv_composite(rgb_dataset)
    assert len(m.layers) == 1
    assert len(m.ax.images) == 1


def test_rgb_composite_custom_band_order(rgb_dataset):
    """A custom band order still produces one RGB image."""
    m = Map(crs=rgb_dataset.epsg)
    m.rgb_composite(rgb_dataset, bands=(3, 2, 1))
    assert len(m.ax.images) == 1
