import shutil
from pathlib import Path

import pytest
import geopandas as gpd
from geopandas.geodataframe import GeoDataFrame
from pyramids.dataset import Dataset

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
def display_cellvalue() -> bool:
    return True


@pytest.fixture(scope="module")
def background_color_threshold():
    return None


@pytest.fixture(scope="module")
def num_size() -> int:
    return 8


@pytest.fixture(scope="module")
def ticks_spacing() -> int:
    return 500


@pytest.fixture(scope="module")
def points() -> GeoDataFrame:
    return gpd.read_file("tests/data/points.geojson")


@pytest.fixture(scope="module")
def pid_size() -> int:
    return 20


@pytest.fixture(scope="module")
def pid_color() -> str:
    return "green"


@pytest.fixture(scope="module")
def point_size() -> int:
    return 100


@pytest.fixture(scope="module")
def point_color() -> str:
    return "blue"
