"""Tests for digitalearth.sources — one per input type, plus the no-competitor-imports guard.

Run from the repository root (data paths are repo-root-relative); ``MPLBACKEND=Agg`` is set in pytest config.
"""
from pathlib import Path

import numpy as np
import pytest

from digitalearth.sources import DimensionInfo, Source, get_source


def test_raster_source(dataset):
    """A pyramids Dataset becomes a NaN-masked raster Source with 1-D axis coords."""
    src = get_source(dataset)
    assert isinstance(src, Source)
    # z is the 2-D grid; x/y are 1-D vectors of length columns/rows
    assert src.z.values.ndim == 2
    assert src.z.values.shape == (dataset.rows, dataset.columns)
    assert src.x.values.shape == (dataset.columns,)
    assert src.y.values.shape == (dataset.rows,)
    # nodata has been replaced by NaN (acc4000 has masked cells)
    assert np.isnan(src.z.values).any()
    assert src.crs == dataset.epsg
    assert src.metadata("kind") == "raster"
    assert src.metadata("variable") == dataset.band_names[0]


def test_raster_band_is_one_based(dataset):
    """band=1 selects the first band (DE is 1-based even though pyramids read_array is 0-based)."""
    src = get_source(dataset, band=1)
    np.testing.assert_allclose(
        np.nan_to_num(src.z.values),
        np.nan_to_num(get_source(dataset).z.values),
    )


def test_numpy_source_pixel_axes():
    """A raw 2-D array gets pixel-index axes when x/y are not supplied."""
    arr = np.arange(12, dtype="float64").reshape(3, 4)
    src = get_source(arr)
    assert src.z.values.shape == (3, 4)
    np.testing.assert_array_equal(src.x.values, np.arange(4))
    np.testing.assert_array_equal(src.y.values, np.arange(3))
    assert src.crs is None
    assert src.metadata("kind") == "raster"


def test_numpy_source_explicit_axes():
    """Explicit x/y coordinates are passed through."""
    arr = np.zeros((2, 2))
    x, y = np.array([10.0, 20.0]), np.array([5.0, 6.0])
    src = get_source(arr, x=x, y=y)
    np.testing.assert_array_equal(src.x.values, x)
    np.testing.assert_array_equal(src.y.values, y)


def test_numpy_source_rejects_non_2d():
    """A non-2-D array raises a clear error."""
    with pytest.raises(ValueError, match="2-D"):
        get_source(np.arange(5))


def test_feature_source():
    """A pyramids FeatureCollection of points becomes a vector Source."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    src = get_source(fc)
    n = len(fc)
    assert src.metadata("kind") == "vector"
    assert src.x.values.shape == (n,)
    assert src.y.values.shape == (n,)
    assert src.crs == 32618
    # points.geojson has a numeric "fid" column -> used as z
    assert src.z is not None
    assert src.z.values.shape == (n,)


def test_collection_source():
    """One member of a DatasetCollection becomes a raster Source with member metadata."""
    from pyramids.dataset.collection import DatasetCollection

    dc = DatasetCollection.from_files(["examples/data/acc4000.tif"])
    src = get_source(dc)
    assert src.metadata("kind") == "raster"
    assert src.metadata("member") == 0
    assert src.metadata("n_members") == 1
    assert src.z.values.ndim == 2


def test_netcdf_source():
    """A NetCDF variable becomes a raster Source (synthesised from the test raster)."""
    from pyramids.dataset import Dataset
    from pyramids.netcdf import NetCDF

    ds = Dataset.read_file("examples/data/acc4000.tif")
    arr = ds.read_array(band=0).astype("float32")[np.newaxis, ...]
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=ds.geotransform,
        epsg=ds.epsg,
        no_data_value=ds.no_data_value[0],
        variable_name="acc",
    )
    src = get_source(nc)
    assert src.metadata("kind") == "raster"
    assert src.metadata("variable") == "acc"
    assert src.z.values.ndim == 2
    assert src.x.values.ndim == 1 and src.y.values.ndim == 1


def test_unsupported_type_raises():
    """An unsupported input type raises TypeError."""
    with pytest.raises(TypeError, match="cannot build a Source"):
        get_source("not a dataset")


def test_dimension_info_dataclass():
    """DimensionInfo carries values/name/units."""
    di = DimensionInfo(np.array([1.0, 2.0]), "x", "m")
    assert di.name == "x" and di.units == "m"
    np.testing.assert_array_equal(di.values, [1.0, 2.0])


class TestExtractorHelpers:
    """Tests for the private helpers in digitalearth.sources.extractors."""

    def test_band_item_normal(self):
        """_band_item returns seq[index] for a valid index."""
        from digitalearth.sources.extractors import _band_item

        assert _band_item(("a", "b"), 1) == "b"

    def test_band_item_empty_returns_default(self):
        """_band_item returns the default for an empty/None sequence."""
        from digitalearth.sources.extractors import _band_item

        assert _band_item((), 0, default="x") == "x"
        assert _band_item(None, 0, default="x") == "x"

    def test_band_item_out_of_range_returns_default(self):
        """_band_item returns the default when the index is out of range."""
        from digitalearth.sources.extractors import _band_item

        assert _band_item(("a",), 5, default=None) is None

    def test_mask_nodata_passthrough_when_none(self):
        """mask_nodata returns the array unchanged (as float) when nodata is None."""
        from digitalearth._arrays import mask_nodata

        out = mask_nodata(np.array([1, 2, 3]), None)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_mask_nodata_replaces_with_nan(self):
        """mask_nodata replaces cells matching nodata with NaN."""
        from digitalearth._arrays import mask_nodata

        out = mask_nodata(np.array([1.0, -9999.0, 3.0]), -9999.0)
        assert np.isnan(out[1]) and not np.isnan(out[0])

    def test_from_netcdf_no_variables_raises(self):
        """_from_netcdf raises ValueError when the NetCDF exposes no variables."""
        from digitalearth.sources.extractors import _from_netcdf

        class _StubNetCDF:
            variable_names: list = []

        with pytest.raises(ValueError, match="no variables"):
            _from_netcdf(_StubNetCDF(), None, None)


def test_feature_source_polygon_uses_centroid():
    """A non-point FeatureCollection falls back to geometry centroids for x/y."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    polys = fc.copy()
    polys["geometry"] = fc.geometry.buffer(100.0)  # points -> polygons
    src = get_source(polys)
    assert src.metadata("kind") == "vector"
    assert src.x.values.shape == (len(fc),)


def test_no_competitor_imports():
    """The sources package must not import xarray/rasterio/fiona/etc. (CLAUDE.md: pyramids is the only GIS dep)."""
    forbidden = ("xarray", "rasterio", "rioxarray", "fiona", "netCDF4", "cfgrib", "osgeo", "cartopy")
    pkg = Path("src/digitalearth/sources")
    offenders = []
    for py in pkg.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        for mod in forbidden:
            if f"import {mod}" in text or f"from {mod}" in text:
                offenders.append(f"{py.name}: {mod}")
    assert not offenders, f"competitor imports found: {offenders}"
