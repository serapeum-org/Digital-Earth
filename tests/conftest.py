import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from geopandas.geodataframe import GeoDataFrame
from pyramids.dataset import Dataset


def pytest_addoption(parser):
    """Register the options that switch the 3-D image baselines between rendering, comparing and generating.

    Args:
        parser: pytest's option parser.
    """
    group = parser.getgroup("image3d", "3-D image baselines (tests marked image3d)")
    group.addoption(
        "--image3d-compare",
        action="store_true",
        default=False,
        help="compare each image3d render with its baseline in tests/baseline3d",
    )
    group.addoption(
        "--image3d-generate",
        action="store_true",
        default=False,
        help="write each image3d render as its new baseline in tests/baseline3d",
    )
    group.addoption(
        "--image3d-results",
        default=None,
        help="directory to write the renders and baselines of image3d comparisons that fail",
    )


#: Committed Natural-Earth 110m assets (``ne_110m_*.geojson.gz``) that back the
#: coastline/border/land/ocean/lake/river overlay tests.
_NATURAL_EARTH_FIXTURES = Path(__file__).parent / "data" / "naturalearth"


@pytest.fixture(scope="session", autouse=True)
def _seed_natural_earth_cache(tmp_path_factory):
    """Point ``cleopatra.basemap.reference`` at the committed Natural-Earth assets so overlay tests run offline.

    ``cleopatra.basemap.reference.natural_earth`` / ``add_features`` download each layer from a GitHub release on
    first use and cache it under ``CLEOPATRA_CACHE_DIR``; if the file is already there they read it without
    touching the network. We copy the committed ``ne_110m_*`` fixtures into a throwaway cache dir (a copy, so
    cleopatra's corrupt-cache cleanup can never unlink the originals) and aim ``CLEOPATRA_CACHE_DIR`` at it.
    The overlay tests therefore exercise the real geometry deterministically and **fail** — rather than skip —
    if an asset is missing or unreadable.
    """
    cache = tmp_path_factory.mktemp("cleopatra-cache")
    for asset in _NATURAL_EARTH_FIXTURES.glob("*.geojson.gz"):
        shutil.copy(asset, cache / asset.name)
    import os

    os.environ["CLEOPATRA_CACHE_DIR"] = str(cache)
    yield


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to avoid the >20-open-figures warning/leak."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return Dataset.read_file("examples/data/acc4000.tif")


def _grib_like_dataset(bands: int = 1, element: str = "TMP") -> Dataset:
    """Build an in-memory pyramids ``Dataset`` whose bands carry GRIB message metadata.

    The real GRIB *driver* (``libgdal-grib``) is not installed in the test environment — so GRIB files cannot
    be opened here, and pyramids' own GRIB doctests are ``+SKIP``. But ``pyramids.grib.grib_band_metadata``
    only *decodes* the ``GRIB_*`` band-metadata keys off ``dataset.raster``; a plain MEM-backed GDAL dataset
    can carry those keys, so this gives a real, driver-free fixture for the GRIB-metadata extraction path.
    Each band's forecast horizon / valid time steps by 6 h, as a short forecast stack would.

    Args:
        bands: Number of forecast steps (bands) to build.
        element: The GRIB ``element`` code every band carries.

    Returns:
        A pyramids ``Dataset`` of the requested band count, EPSG:4326, with GRIB message metadata per band.
    """
    from osgeo import gdal, osr

    gdal.UseExceptions()
    mem = gdal.GetDriverByName("MEM").Create("", 4, 3, bands, gdal.GDT_Float32)
    spatial = osr.SpatialReference()
    spatial.ImportFromEPSG(4326)
    mem.SetProjection(spatial.ExportToWkt())
    mem.SetGeoTransform([0.0, 1.0, 0.0, 3.0, 0.0, -1.0])
    base = 1609459200  # 2021-01-01T00:00:00Z, in GRIB's "<seconds> sec UTC" spelling
    for index in range(1, bands + 1):
        band = mem.GetRasterBand(index)
        band.WriteArray(np.arange(12, dtype="float32").reshape(3, 4))
        step = index * 21600  # 6 h per step
        for key, value in {
            "GRIB_ELEMENT": element,
            "GRIB_SHORT_NAME": "2-HTGL",
            "GRIB_UNIT": "[C]",
            "GRIB_COMMENT": "Temperature",
            "GRIB_DISCIPLINE": "0",
            "GRIB_REF_TIME": f"{base} sec UTC",
            "GRIB_VALID_TIME": f"{base + step} sec UTC",
            "GRIB_FORECAST_SECONDS": f"{step} sec",
        }.items():
            band.SetMetadataItem(key, value)
    return Dataset(mem)


@pytest.fixture
def grib_dataset() -> Dataset:
    """A single-band in-memory Dataset carrying GRIB message metadata (see :func:`_grib_like_dataset`)."""
    return _grib_like_dataset(bands=1)


@pytest.fixture
def grib_stack() -> Dataset:
    """A three-band in-memory GRIB Dataset: one band per 6-hour forecast step, each with its own valid time."""
    return _grib_like_dataset(bands=3)


@pytest.fixture(scope="module")
def points() -> GeoDataFrame:
    return gpd.read_file("tests/data/points.geojson")
