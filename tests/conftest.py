import matplotlib
import pytest
import geopandas as gpd
from geopandas.geodataframe import GeoDataFrame
from pyramids.dataset import Dataset

matplotlib.use("Agg")


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
