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


def _netcdf_container():
    """Build a one-variable pyramids ``NetCDF`` from the committed test raster.

    Returns:
        A ``NetCDF`` holding a single ``acc`` variable whose attribute list declares no ``units``.
    """
    from pyramids.dataset import Dataset, GeoReference
    from pyramids.netcdf import NetCDF

    ds = Dataset.read_file("examples/data/acc4000.tif")
    values = ds.read_array(band=0).astype("float32")[np.newaxis, ...]
    return NetCDF.from_array(
        values,
        geo_ref=GeoReference(geo=ds.geotransform, epsg=ds.epsg),
        no_data_value=ds.no_data_value[0],
        variable_name="acc",
    )


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

    def test_from_netcdf_no_variables_raises(self):
        """_from_netcdf raises ValueError when the NetCDF exposes no variables."""
        from digitalearth.base.sources.extractors import _from_netcdf

        class _StubNetCDF:
            variable_names: list = []

        with pytest.raises(ValueError, match="no variables"):
            _from_netcdf(_StubNetCDF(), None, None)

    def test_attr_answers_none_for_anything_that_is_not_a_mapping(self):
        """A CF lookup handed something that is not an attribute mapping answers ``None``.

        Test scenario:
            The identity lookups are fed whatever a driver reports as metadata — ``None`` from a container
            carrying none, a bare string from one that reports it as text. Both have to mean "no such
            attribute" rather than fail on the ``.items()`` call the mapping branch makes.
        """
        from digitalearth.base.sources.extractors import _attr

        assert _attr(None, "units") is None, (
            "a container with no metadata mapping declares no units"
        )
        assert _attr("units=mm", "units") is None, (
            "a non-mapping metadata blob must not be parsed for attributes"
        )

    def test_variable_attributes_is_empty_without_variable_metadata(self):
        """A container exposing no per-variable metadata reports no attributes at all.

        Test scenario:
            ``_variable_attributes`` reaches ``meta_data.variables`` through ``getattr``, so a container
            with neither must answer with an empty mapping — the CF lookups then simply find nothing,
            instead of the extractor failing on data it can still read values from.
        """
        from digitalearth.base.sources.extractors import _variable_attributes

        class _NoMetadata:
            """A container that exposes values but no ``meta_data``."""

        assert _variable_attributes(_NoMetadata(), "acc") == {}, (
            "a container with no meta_data must report no attributes"
        )

    def test_variable_attributes_is_empty_for_a_variable_it_does_not_describe(self):
        """Asking for a variable the container's metadata does not list yields no attributes.

        Test scenario:
            The lookup walks the container's per-variable metadata by name. A name that is not in it has
            to fall out with an empty mapping rather than hand back whichever variable happened to be
            last, which would label the source with another variable's units.
        """
        from digitalearth.base.sources.extractors import _variable_attributes

        assert _variable_attributes(_netcdf_container(), "no_such_variable") == {}, (
            "an unlisted variable must report no attributes"
        )

    def test_the_cf_unit_slot_fills_in_for_a_missing_units_attribute(self):
        """A variable's CF units reach the ``Source`` from the unit slot, not the attribute list.

        Test scenario:
            GDAL normalises a CF ``units`` attribute onto the variable's own unit and drops it from the
            attribute list, so reading the attribute list alone left the source unitless for exactly the
            input type CF units live in.
        """
        from dataclasses import replace

        nc = _netcdf_container()
        metadata = nc.meta_data
        variables = dict(metadata.variables)
        variables["acc"] = replace(variables["acc"], unit="m3/s")
        nc.meta_data = replace(metadata, variables=variables)
        assert "units" not in variables["acc"].attributes, (
            "the case under test is a unit slot with no units attribute beside it"
        )
        source = get_source(nc)
        assert source.units == "m3/s", (
            f"expected the unit slot to be read, got {source.units!r}"
        )


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
        Path(__file__).resolve().parents[2]
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


def test_reproject_reports_an_empty_view_as_off_limb():
    """The shared reprojection turns GDAL's wording into the typed signal every backend reads."""
    import numpy as np
    from pyramids.dataset import Dataset, GeoReference

    from digitalearth.base.crs import OffLimbError, reproject
    from digitalearth.static import projections

    ds = Dataset.from_array(
        np.ones((20, 20), "float32"),
        geo_ref=GeoReference(geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326),
        no_data_value=-9999.0,
    )
    hidden = projections.orthographic(lon=-175, lat=15)
    with pytest.raises(OffLimbError, match="too few sample points"):
        reproject(ds, hidden)


def test_reproject_passes_other_failures_through():
    """Only the "too few points survived" message becomes OffLimbError; the rest stay as they came."""
    from digitalearth.base.crs import OffLimbError, reproject

    class Broken:
        def to_crs(self, crs):
            raise RuntimeError("PROJ: proj_create: unrecognized format / unknown name")

    broken = Broken()
    with pytest.raises(RuntimeError) as caught:
        reproject(broken, 3857)
    assert not isinstance(caught.value, OffLimbError), (
        "a real projection failure must not be reported as an empty view"
    )


#: An orthographic projection has no EPSG code, so pyramids reports ``epsg is None`` for a dataset warped
#: into it — the case that used to leave ``Source.crs`` saying "unknown" while holding orthographic metres.
ORTHOGRAPHIC = "+proj=ortho +lat_0=53 +lon_0=4 +datum=WGS84 +units=m +no_defs"


def _small_raster():
    """Build a 20x20 lon/lat raster around (4E, 53N).

    Returns:
        pyramids.dataset.Dataset: the raster, in EPSG:4326.
    """
    from pyramids.dataset import Dataset, GeoReference

    return Dataset.from_array(
        np.ones((20, 20), "float32"),
        geo_ref=GeoReference(geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326),
    )


class TestSourceCrsContract:
    """Regression tests for #235 — ``Source.crs`` was write-only and wrong after a warp."""

    def test_warped_source_reports_the_display_crs_it_was_given(self):
        """A caller that warped the data names the CRS, and the Source reports it verbatim.

        Test scenario:
            The reported repro: after ``to_crs(orthographic)`` the source said ``None`` while its x/y held
            orthographic metres. Now the display CRS the caller warped to is what comes back.
        """
        warped = _small_raster().to_crs(ORTHOGRAPHIC)
        src = get_source(warped, band=1, crs=ORTHOGRAPHIC)
        assert src.crs == ORTHOGRAPHIC, (
            f"the display CRS should be stored as given, got {src.crs!r}"
        )
        assert src.crs != 4326, (
            f"a warped source must not report its pre-warp CRS, got {src.crs!r}"
        )
        assert src.crs is not None, (
            f"a warped source must report a CRS rather than 'unknown', got {src.crs!r}"
        )

    def test_code_less_projection_still_reports_its_definition(self):
        """Without a caller-supplied CRS the projection definition is used, never ``None``.

        Test scenario:
            ``epsg is None`` means "no authority code", not "no CRS" — the two questions used to share one
            slot, and the answer to the wrong one was being stored.
        """
        warped = _small_raster().to_crs(ORTHOGRAPHIC)
        src = get_source(warped, band=1)
        assert warped.epsg is None, (
            "the fixture must be a code-less projection to be the case under test"
        )
        assert src.crs, (
            f"a code-less projection must still report its definition, got {src.crs!r}"
        )
        assert "ortho" in str(src.crs).lower(), (
            f"expected the orthographic definition, got {src.crs!r}"
        )

    def test_epsg_answers_the_code_question_separately(self):
        """``Source.epsg`` is the code-or-None question; ``crs`` keeps saying where the coordinates are.

        Test scenario:
            An EPSG-coded raster answers both; the orthographic one answers only ``crs``.
        """
        coded = get_source(_small_raster(), band=1)
        warped = get_source(
            _small_raster().to_crs(ORTHOGRAPHIC), band=1, crs=ORTHOGRAPHIC
        )
        assert (coded.crs, coded.epsg) == (4326, 4326), (
            f"an EPSG raster should report 4326 twice, got {(coded.crs, coded.epsg)}"
        )
        assert warped.epsg is None, (
            f"an orthographic source has no EPSG code, got {warped.epsg!r}"
        )

    def test_none_means_genuinely_unknown(self):
        """``None`` is reserved for an input that declares no CRS at all.

        Test scenario:
            A raw numpy array is the only source of a ``None`` CRS now that a code-less projection reports
            its definition.
        """
        assert get_source(np.zeros((2, 2))).crs is None, (
            "a bare numpy array has no CRS, so None is the right answer there"
        )

    def test_a_frame_declaring_no_crs_reports_an_unknown_one(self):
        """A vector frame with no CRS at all is the unknown case, not a code-less projection.

        Test scenario:
            The vector fallback goes EPSG code, then CRS definition. A frame carrying neither has to end
            at ``None`` rather than at whatever the ``crs`` attribute happened to hold, which is what
            keeps ``None`` meaning "genuinely unknown" for vectors as well as for raw arrays.
        """
        import geopandas as gpd
        from pyramids.feature import FeatureCollection
        from shapely.geometry import Point

        frame = gpd.GeoDataFrame(
            {"value": [1.0, 2.0]},
            geometry=[Point(0.0, 0.0), Point(1.0, 1.0)],
            crs=None,
        )
        source = get_source(FeatureCollection(frame))
        assert source.crs is None, (
            f"a CRS-less frame must report None, got {source.crs!r}"
        )

    def test_an_authority_prefixed_crs_string_still_answers_its_code(self):
        """``crs="EPSG:3857"`` answers the code question the same way ``crs=3857`` does.

        Test scenario:
            A caller that warped the data names the CRS however their stack spells it, and the authority
            prefix is the common spelling. Stripping it is what stops ``epsg`` reporting ``None`` for a
            CRS that plainly has a code, while ``crs`` keeps the spelling it was given.
        """
        source = get_source(np.zeros((2, 2)), crs="EPSG:3857")
        assert source.epsg == 3857, f"expected the code 3857, got {source.epsg!r}"
        assert source.crs == "EPSG:3857", (
            f"crs must keep the spelling it was given, got {source.crs!r}"
        )

    def test_a_written_out_definition_still_answers_its_code(self):
        """A CRS spelled as WKT is read down to the authority block it carries.

        Test scenario:
            ``crs`` is documented to hold a full definition whenever pyramids reports no code of its own,
            and a warp into a *coded* CRS reports its WKT — which ends on ``AUTHORITY["EPSG", ...]``. A
            reader that only stripped an ``"EPSG:"`` prefix answered "no code" for those, so ``epsg`` said
            ``None`` for a source whose CRS names its authority outright.
        """
        from pyramids.base.crs import crs_from_user_input

        wkt = crs_from_user_input(3857).to_wkt()
        source = get_source(np.zeros((2, 2)), crs=wkt)
        assert source.epsg == 3857, (
            f"the authority block names EPSG:3857, got {source.epsg!r}"
        )
        assert source.crs == wkt, (
            "crs must keep the definition it was given, not the code read out of it"
        )

    def test_epsg_is_none_when_there_is_no_code_to_report(self):
        """An unknown CRS has no code, and a boolean is never mistaken for one.

        Test scenario:
            ``epsg`` narrows ``crs`` to "is there an authority code". ``None`` has none; ``True`` is an
            ``int`` subclass in Python, so without the boolean guard a stray flag would be reported as
            the EPSG code 1.
        """
        assert get_source(np.zeros((2, 2))).epsg is None, (
            "a bare numpy array has no CRS, so it has no EPSG code either"
        )
        axis = DimensionInfo(np.array([0.0]), "x")
        assert Source(None, axis, axis, crs=True).epsg is None, (
            "a boolean must not be read as the EPSG code 1"
        )
        assert Source(None, axis, axis, crs=ORTHOGRAPHIC).epsg is None, (
            "a projection that names no authority must still report no code"
        )


class TestRasterIdentity:
    """Regression tests for #231 — no extractor wrote the ``standard_name`` the style matcher reads."""

    def test_band_standard_name_reaches_the_source(self):
        """A band declaring a CF ``standard_name`` carries it into ``Source.metadata``.

        Test scenario:
            The key ``auto_style`` reads was written by nothing, so the Magics standard-name step was dead
            code for every real input.
        """
        ds = _small_raster()
        ds.bands.set_metadata({"standard_name": "air_temperature"}, band=0)
        src = get_source(ds, band=1)
        assert src.metadata("standard_name") == "air_temperature", (
            f"expected the band's standard_name, got {src.metadata('standard_name')!r}"
        )

    def test_dataset_level_metadata_is_read_case_insensitively(self):
        """A dataset-level ``STANDARD_NAME`` is found too, whatever case the writer used.

        Test scenario:
            GDAL metadata keys survive round-trips in whatever case a writer chose, so the lookup cannot be
            case-sensitive.
        """
        ds = _small_raster()
        ds.meta_data = {"STANDARD_NAME": "air_pressure_at_mean_sea_level"}
        src = get_source(ds, band=1)
        assert src.metadata("standard_name") == "air_pressure_at_mean_sea_level", (
            f"an upper-cased key should still be read, got {src.metadata('standard_name')!r}"
        )

    def test_caller_metadata_still_wins(self):
        """An explicit ``metadata=`` overrides the derived identity.

        Test scenario:
            Deriving the key must not take the override away from callers that pass one.
        """
        ds = _small_raster()
        ds.bands.set_metadata({"standard_name": "air_temperature"}, band=0)
        src = get_source(
            ds, band=1, metadata={"standard_name": "sea_surface_temperature"}
        )
        assert src.metadata("standard_name") == "sea_surface_temperature", (
            "a caller-supplied standard_name must win over the derived one"
        )
