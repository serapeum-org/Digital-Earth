"""Tests for digitalearth.sources.get_stack — the multiband band-stack reader (PC-5)."""
import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

from digitalearth.sources import get_stack


@pytest.fixture
def three_band():
    """A 3-band EPSG:4326 dataset (distinct per-band values, nodata -9999)."""
    arr = np.stack([
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
        np.array([[10.0, 20.0], [30.0, 40.0]], dtype="float32"),
        np.array([[100.0, 200.0], [300.0, 400.0]], dtype="float32"),
    ])  # (3, 2, 2)
    return Dataset.from_array(
        arr=arr,
        geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
        no_data_value=-9999.0,
    )


class TestGetStack:
    """Tests for get_stack."""

    def test_shape_is_band_last(self, three_band):
        """get_stack returns an (rows, cols, n) float64 array in band order.

        Test scenario:
            Three 2x2 bands stack to (2, 2, 3) float64 with band 1 first along the last axis.
        """
        stack = get_stack(three_band, (1, 2, 3))
        assert stack.shape == (2, 2, 3), f"expected (2,2,3), got {stack.shape}"
        assert stack.dtype == np.float64, f"expected float64, got {stack.dtype}"
        assert stack[0, 0, 0] == 1.0 and stack[0, 0, 1] == 10.0 and stack[0, 0, 2] == 100.0, (
            f"band order/values wrong: {stack[0, 0]}"
        )

    def test_custom_band_order(self, three_band):
        """get_stack honours an arbitrary band order.

        Test scenario:
            Requesting (3, 1) returns those two bands in that order along the last axis.
        """
        stack = get_stack(three_band, (3, 1))
        assert stack.shape == (2, 2, 2), f"expected (2,2,2), got {stack.shape}"
        assert stack[0, 0, 0] == 100.0 and stack[0, 0, 1] == 1.0, f"order wrong: {stack[0, 0]}"

    def test_masks_nodata_by_default(self):
        """get_stack nulls nodata cells per band when mask=True (the default).

        Test scenario:
            A band whose cell equals the sentinel comes back as NaN at that cell.
        """
        arr = np.stack([np.array([[5.0, -9999.0]], dtype="float32")])  # (1, 1, 2)
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
            no_data_value=-9999.0,
        )
        stack = get_stack(ds, (1,))  # shape (rows=1, cols=2, n=1)
        assert stack[0, 0, 0] == 5.0, "real value should be preserved"
        assert np.isnan(stack[0, 1, 0]), "nodata cell should be NaN"

    def test_mask_false_keeps_raw_values(self):
        """get_stack(mask=False) returns the raw cast values (sentinel not nulled).

        Test scenario:
            With masking off the nodata sentinel stays as its numeric value.
        """
        arr = np.stack([np.array([[5.0, -9999.0]], dtype="float32")])
        ds = Dataset.from_array(
            arr=arr,
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
            no_data_value=-9999.0,
        )
        stack = get_stack(ds, (1,), mask=False)  # shape (rows=1, cols=2, n=1)
        assert stack[0, 1, 0] == -9999.0, f"raw sentinel should be kept, got {stack[0, 1, 0]}"
        assert np.isfinite(stack).all(), "no cell should be NaN when mask=False"
