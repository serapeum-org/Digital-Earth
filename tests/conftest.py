import shutil
from pathlib import Path

import geopandas as gpd
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


@pytest.fixture(scope="module")
def points() -> GeoDataFrame:
    return gpd.read_file("tests/data/points.geojson")
