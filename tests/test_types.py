"""Tests for digitalearth._types — structural Protocols for pyramids inputs (PC-1)."""
import numpy as np
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

from digitalearth._types import PlottableData, RasterLike, VectorLike


class TestRasterLike:
    """Tests for the RasterLike protocol."""

    def test_dataset_satisfies_rasterlike(self):
        """A real pyramids Dataset structurally satisfies RasterLike.

        Test scenario:
            The runtime-checkable protocol recognises a Dataset (it exposes epsg/x/y/to_crs/read_array/…).
        """
        ds = Dataset.read_file("examples/data/acc4000.tif")
        assert isinstance(ds, RasterLike), "Dataset should satisfy RasterLike"

    def test_plain_object_is_not_rasterlike(self):
        """An object missing the raster surface is not RasterLike.

        Test scenario:
            A bare object lacks read_array/to_crs/etc., so the runtime check is False.
        """
        assert not isinstance(object(), RasterLike), "a bare object must not satisfy RasterLike"


class TestVectorLike:
    """Tests for the VectorLike protocol."""

    def test_feature_collection_satisfies_vectorlike(self):
        """A real pyramids FeatureCollection structurally satisfies VectorLike.

        Test scenario:
            The runtime-checkable protocol recognises a FeatureCollection (epsg/crs/geometry/to_crs/len).
        """
        fc = FeatureCollection.read_file("tests/data/points.geojson")
        assert isinstance(fc, VectorLike), "FeatureCollection should satisfy VectorLike"

    def test_plain_object_is_not_vectorlike(self):
        """An object missing the vector surface is not VectorLike.

        Test scenario:
            A bare object lacks geometry/to_crs/etc., so the runtime check is False.
        """
        assert not isinstance(object(), VectorLike), "a bare object must not satisfy VectorLike"


class TestPlottableData:
    """Tests for the PlottableData union alias."""

    def test_union_members(self):
        """PlottableData is the union of RasterLike, VectorLike and numpy arrays.

        Test scenario:
            The alias advertises exactly the three accepted input kinds (so the API surface documents them).
        """
        import typing

        args = set(typing.get_args(PlottableData))
        assert args == {RasterLike, VectorLike, np.ndarray}, f"unexpected union members: {args}"
