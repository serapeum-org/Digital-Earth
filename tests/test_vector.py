"""Tests for T3.2 — Map vector methods (quiver/barbs/streamplot) from two pyramids rasters."""

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

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
    u_ds = Dataset.from_array(arr=u, geo_ref=GeoReference(geo=geo, epsg=4326))
    v_ds = Dataset.from_array(arr=v, geo_ref=GeoReference(geo=geo, epsg=4326))
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


def test_quiverkey_after_quiver(uv):
    """quiverkey adds a reference arrow keyed to the most recent quiver layer."""
    from matplotlib.quiver import QuiverKey

    u_ds, v_ds = uv
    m = Map(crs=4326)
    m.quiver(u_ds, v_ds)
    key = m.quiverkey(1.0, "1 m/s")
    assert isinstance(key, QuiverKey), f"expected a QuiverKey, got {type(key)}"


def test_quiverkey_without_quiver_raises():
    """quiverkey before any quiver layer raises a clear error."""
    m = Map(crs=4326)
    with pytest.raises(ValueError, match="needs a prior quiver"):
        m.quiverkey(1.0, "1 m/s")


def test_quiverkey_rejects_barbs_streamplot(uv):
    """barbs/streamplot have no quiver key, so quiverkey raises after them."""
    u_ds, v_ds = uv
    m = Map(crs=4326)
    m.barbs(u_ds, v_ds)
    with pytest.raises(ValueError, match="needs a prior quiver"):
        m.quiverkey(1.0, "1 m/s")


@pytest.fixture
def uv_descending_y():
    """Two u/v rasters on the usual north->south (descending-y) raster grid.

    Exercises the streamplot axis-flip path: a descending y axis (and its data rows) must be
    reversed to a strictly-increasing grid before streamplot.

    Returns:
        tuple[Dataset, Dataset]: (u, v) single-band datasets in EPSG:4326 with descending y.
    """
    ny, nx = 6, 8
    u = np.ones((ny, nx), dtype="float32")
    v = np.linspace(-1.0, 1.0, ny, dtype="float32")[:, None] * np.ones((1, nx), "float32")
    geo = (0.0, 1.0, 0.0, 6.0, 0.0, -1.0)  # ymax=6, negative dy -> y runs north->south
    u_ds = Dataset.from_array(arr=u, geo_ref=GeoReference(geo=geo, epsg=4326))
    v_ds = Dataset.from_array(arr=v, geo_ref=GeoReference(geo=geo, epsg=4326))
    return u_ds, v_ds


def test_streamplot_flips_descending_y(uv_descending_y):
    """streamplot reverses a descending-y grid (and data) so the field renders right-side-up."""
    u_ds, v_ds = uv_descending_y
    m = Map(crs=4326)
    m.streamplot(u_ds, v_ds)
    assert len(m.layers) == 1
    assert m.ax.collections  # streamlines drawn


def test_streamplot_flips_descending_x(uv, mocker):
    """_vector reverses a descending-x axis (and data columns) to a strictly increasing grid.

    Real rasters always hand back an ascending x / descending y axis, so the x-flip branch is fed a
    synthetic ``_prepare`` result whose x axis runs east->west and y axis runs south->north — the one
    orientation that exercises the x-reversal and skips the y-reversal in a single pass.
    """
    from types import SimpleNamespace

    ny, nx = 6, 8
    x = np.arange(nx, 0, -1, dtype="float64")  # descending x: 8..1
    y = np.arange(ny, dtype="float64")         # ascending y: 0..5
    z = np.ones((ny, nx), dtype="float32")
    src = SimpleNamespace(x=SimpleNamespace(values=x), y=SimpleNamespace(values=y),
                          z=SimpleNamespace(values=z))
    mocker.patch.object(Map, "_prepare", return_value=src)
    m = Map(crs=4326)
    m.streamplot(*uv)
    assert len(m.layers) == 1 and m.ax.collections
