"""Tests for T5.2 — longitude wrap, cyclic column, and the cyclic field option."""

import numpy as np
from pyramids.dataset import Dataset, GeoReference

from digitalearth.preprocess import add_cyclic_column, wrap_longitude
from digitalearth.scene import Map


class TestAddCyclicColumn:
    """Tests for add_cyclic_column."""

    def test_adds_one_column(self):
        """A wrap-around column is appended and the longitude is extended by one step."""
        z = np.arange(6.0).reshape(2, 3)
        x = np.array([0.0, 120.0, 240.0])
        z2, x2 = add_cyclic_column(z, x)
        assert z2.shape == (2, 4)
        assert x2[-1] == 360.0
        np.testing.assert_array_equal(z2[:, -1], z[:, 0])

    def test_single_column_uses_360_step(self):
        """With a single column the wrap step defaults to 360 degrees."""
        z = np.array([[5.0]])
        x = np.array([10.0])
        _, x2 = add_cyclic_column(z, x)
        assert x2.tolist() == [10.0, 370.0]


class TestWrapLongitude:
    """Tests for wrap_longitude."""

    def test_rolls_0_360_to_signed(self):
        """A 0-360 dataset is rolled to -180..180 via pyramids wrap_longitude."""
        arr = np.arange(8 * 4, dtype="float32").reshape(4, 8)
        geo = (0.0, 45.0, 0.0, 90.0, 0.0, -45.0)
        ds = Dataset.from_array(arr=arr, geo_ref=GeoReference(geo=geo, epsg=4326))
        wrapped = wrap_longitude(ds)
        assert wrapped.x.min() < 0.0
        assert wrapped.x.max() <= 180.0


def test_field_cyclic_option_widens_extent():
    """contourf(cyclic=True) closes the seam, widening the x-extent by one grid step."""
    arr = np.arange(8 * 4, dtype="float32").reshape(4, 8)
    geo = (-180.0, 45.0, 0.0, 90.0, 0.0, -45.0)
    ds = Dataset.from_array(arr=arr, geo_ref=GeoReference(geo=geo, epsg=4326))
    plain = Map(crs=4326)
    plain.contourf(ds)
    wide = Map(crs=4326)
    wide.contourf(ds, cyclic=True)
    assert wide.ax.get_xlim()[1] > plain.ax.get_xlim()[1]
