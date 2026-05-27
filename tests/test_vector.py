"""Tests for T3.2 — Map vector methods (quiver/barbs/streamplot) from two pyramids rasters."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalearth.scene import Map


@pytest.fixture
def uv():
    """Two small synthetic u/v rasters (no nodata) on an increasing-y grid.

    Returns:
        tuple[Dataset, Dataset]: (u, v) single-band datasets in EPSG:4326.
    """
    ny, nx = 6, 8
    u = np.ones((ny, nx), dtype="float32")
    v = np.linspace(-1.0, 1.0, ny, dtype="float32")[:, None] * np.ones((1, nx), "float32")
    geo = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)  # increasing y (good for streamplot)
    u_ds = Dataset.create_from_array(arr=u, geo=geo, epsg=4326)
    v_ds = Dataset.create_from_array(arr=v, geo=geo, epsg=4326)
    return u_ds, v_ds


def test_quiver(uv):
    """quiver renders arrows from the (u, v) pair on the shared axes."""
    u_ds, v_ds = uv
    m = Map(crs=4326)
    m.quiver(u_ds, v_ds)
    assert len(m.layers) == 1
    assert m.ax.collections


def test_barbs(uv):
    """barbs renders wind barbs from the (u, v) pair."""
    u_ds, v_ds = uv
    m = Map(crs=4326)
    m.barbs(u_ds, v_ds)
    assert len(m.layers) == 1


def test_streamplot(uv):
    """streamplot renders streamlines from the (u, v) pair."""
    u_ds, v_ds = uv
    m = Map(crs=4326)
    m.streamplot(u_ds, v_ds)
    assert len(m.layers) == 1
