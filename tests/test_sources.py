"""Tests for digitalearth.base.sources — one per input type, plus the no-competitor-imports guard.

Run from the repository root (data paths are repo-root-relative); ``MPLBACKEND=Agg`` is set in pytest config.
"""

from pathlib import Path

import numpy as np
import pytest

from digitalearth.base.sources import DimensionInfo, Source, get_source


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
    from pyramids.dataset import Dataset, GeoReference
    from pyramids.netcdf import NetCDF

    ds = Dataset.read_file("examples/data/acc4000.tif")
    arr = ds.read_array(band=0).astype("float32")[np.newaxis, ...]
    nc = NetCDF.from_array(
        arr,
        geo_ref=GeoReference(geo=ds.geotransform, epsg=ds.epsg),
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
    """Tests for the private helpers in digitalearth.base.sources.extractors."""

    def test_band_item_normal(self):
        """_band_item returns seq[index] for a valid index."""
        from digitalearth.base.sources.extractors import _band_item

        assert _band_item(("a", "b"), 1) == "b"

    def test_band_item_empty_returns_default(self):
        """_band_item returns the default for an empty/None sequence."""
        from digitalearth.base.sources.extractors import _band_item

        assert _band_item((), 0, default="x") == "x"
        assert _band_item(None, 0, default="x") == "x"

    def test_band_item_out_of_range_returns_default(self):
        """_band_item returns the default when the index is out of range."""
        from digitalearth.base.sources.extractors import _band_item

        assert _band_item(("a",), 5, default=None) is None

    def test_mask_nodata_passthrough_when_none(self):
        """mask_nodata returns the array unchanged (as float) when nodata is None."""
        from digitalearth.base.arrays import mask_nodata

        out = mask_nodata(np.array([1, 2, 3]), None)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_mask_nodata_replaces_with_nan(self):
        """mask_nodata replaces cells matching nodata with NaN."""
        from digitalearth.base.arrays import mask_nodata

        out = mask_nodata(np.array([1.0, -9999.0, 3.0]), -9999.0)
        assert np.isnan(out[1]) and not np.isnan(out[0])

    def test_from_netcdf_no_variables_raises(self):
        """_from_netcdf raises ValueError when the NetCDF exposes no variables."""
        from digitalearth.base.sources.extractors import _from_netcdf

        class _StubNetCDF:
            variable_names: list = []

        with pytest.raises(ValueError, match="no variables"):
            _from_netcdf(_StubNetCDF(), None, None)


@pytest.mark.parametrize("dtype", ["string", "boolean"])
def test_feature_source_survives_pandas_extension_dtypes(dtype):
    """A pandas extension dtype anywhere in the frame must not take the whole render down.

    Test scenario:
        The value-column scan asked numpy to classify the dtype, and np.issubdtype *raises* on a pandas
        extension dtype instead of answering False. One nullable or string column — even one nobody is
        plotting — therefore killed every path through get_source. Under pandas 3 a plain list of strings
        is already a StringDtype, so this is the common case, not an exotic one. Both dtypes here are
        non-numeric, so the float column beside them stays the z (a nullable *integer* column is numeric
        and is a legitimate z — covered separately).
    """
    import geopandas as gpd
    import pandas as pd
    from pyramids.feature import FeatureCollection
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {
            "other": pd.array(["a", "b"] if dtype == "string" else [1, 0], dtype=dtype),
            "score": [1.5, 2.5],
        },
        geometry=[Point(0, 0), Point(1, 1)],
        crs=4326,
    )
    src = get_source(FeatureCollection(gdf))
    assert src.z is not None, (
        f"a numeric column should still be found alongside a {dtype} column"
    )
    assert list(src.z.values) == [1.5, 2.5], (
        f"the {dtype} column must not be chosen as z"
    )


def test_feature_source_reads_a_nullable_numeric_column():
    """A nullable integer column is a legitimate z, and its NA becomes NaN rather than an object array."""
    import geopandas as gpd
    import pandas as pd
    from pyramids.feature import FeatureCollection
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"score": pd.array([1, None, 3], dtype="Int64")},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs=4326,
    )
    src = get_source(FeatureCollection(gdf))
    assert src.z is not None, "a nullable Int64 column should be usable as z"
    assert np.isnan(src.z.values[1]), (
        f"the missing value should read as NaN, got {src.z.values[1]!r}"
    )


def test_feature_source_still_ignores_booleans():
    """Bools are numeric to pandas but were never a value column here; that must not change."""
    import geopandas as gpd
    import pandas as pd
    from pyramids.feature import FeatureCollection
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"flag": pd.array([True, False], dtype="boolean")},
        geometry=[Point(0, 0), Point(1, 1)],
        crs=4326,
    )
    src = get_source(FeatureCollection(gdf))
    assert src.z is None, f"a boolean-only frame should yield no z, got {src.z}"


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
    forbidden = (
        "xarray",
        "rasterio",
        "rioxarray",
        "fiona",
        "netCDF4",
        "cfgrib",
        "osgeo",
        "cartopy",
    )
    pkg = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "digitalearth"
        / "base"
        / "sources"
    )
    modules = sorted(pkg.rglob("*.py"))
    assert modules, f"no modules found under {pkg} — has the package moved again?"
    offenders = []
    for py in modules:
        text = py.read_text(encoding="utf-8")
        for mod in forbidden:
            if f"import {mod}" in text or f"from {mod}" in text:
                offenders.append(f"{py.name}: {mod}")
    assert not offenders, f"competitor imports found: {offenders}"


def test_raster_source_nodata_is_exact_not_tolerant():
    """get_source masks nodata by exact match, keeping values merely *near* the sentinel (review L1).

    Regression for the tolerant->exact change: a cell equal to the nodata sentinel becomes NaN, but a cell
    0.01% away from it is preserved (the old isclose(rtol=1e-3) rule would have nulled it).
    """
    from pyramids.dataset import Dataset, GeoReference

    nodata = -9999.0
    near = nodata * (1 + 1e-4)  # within 0.1% of the sentinel, but not equal
    arr = np.array([[1.0, nodata], [near, 4.0]], dtype="float64")
    ds = Dataset.from_array(
        arr=arr,
        geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
        no_data_value=nodata,
    )
    z = get_source(ds).z.values
    assert np.isnan(z[0, 1]), "exact nodata cell should be masked"
    assert not np.isnan(z[1, 0]), (
        "a value near (but != ) nodata must be kept under exact-compare"
    )
    assert z[0, 0] == 1.0 and z[1, 1] == 4.0, f"real values changed: {z}"


@pytest.mark.parametrize(
    "values, dtype, expected_numeric",
    [
        ([1, 2], "int64", True),
        ([1.5, 2.5], "float64", True),
        ([True, False], "bool", False),
        (["1 days", "2 days"], "timedelta64[ns]", False),
        (["2020-01-01", "2020-01-02"], "datetime64[ns]", False),
    ],
)
def test_value_column_matches_the_numpy_rule_it_replaced(
    values, dtype, expected_numeric
):
    """The pandas classification answers what np.issubdtype did, wherever numpy could answer.

    Test scenario:
        Two dtypes differ from what np.issubdtype answered, both on purpose. Bools it already called
        non-numeric. Timedeltas it called numeric, but a timedelta column cannot be rendered — float()
        rejects the values — so selecting one only moved the crash downstream.
    """
    import geopandas as gpd
    import pandas as pd
    from pyramids.feature import FeatureCollection
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"candidate": pd.Series(values).astype(dtype)},
        geometry=[Point(0, 0), Point(1, 1)],
        crs=4326,
    )
    src = get_source(FeatureCollection(gdf))
    chosen = src.z is not None
    assert chosen is expected_numeric, (
        f"a {dtype} column should {'' if expected_numeric else 'not '}be picked as z"
    )


def test_a_timedelta_column_is_skipped_for_a_renderable_one():
    """A timedelta column is passed over rather than chosen and then failing to draw."""
    import geopandas as gpd
    import pandas as pd
    from pyramids.feature import FeatureCollection
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"elapsed": pd.to_timedelta([1, 2], unit="D"), "score": [1.5, 2.5]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs=4326,
    )
    src = get_source(FeatureCollection(gdf))
    assert list(src.z.values) == [1.5, 2.5], (
        f"the renderable column should win over the timedelta, got {src.z.values}"
    )
