import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from geopandas.geodataframe import GeoDataFrame
from pyramids.dataset import Dataset, GeoReference


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


def _grib_like_dataset(
    bands: int = 1, *, element: str = "TMP", named: bool = True
) -> Dataset:
    """Build an in-memory pyramids ``Dataset`` whose bands carry GRIB message metadata.

    Built entirely through pyramids — ``Dataset.from_array`` for the raster and the ``band_meta_data`` setter
    for the per-band metadata — so no raster/GDAL handling leaks into the tests. The real GRIB *driver*
    (``libgdal-grib``) is absent in CI, so GRIB files cannot be opened here (pyramids' own GRIB doctests are
    ``+SKIP``); but ``pyramids.grib.grib_band_metadata`` only *decodes* the ``GRIB_*`` band-metadata keys, so
    a Dataset carrying those keys exercises the real extraction path driver-free. Each band's forecast horizon
    / valid time steps by 6 h, as a short forecast stack would.

    Args:
        bands: Number of forecast steps (bands) to build.
        element: The GRIB ``element`` code every band carries (when ``named``).
        named: When ``True`` each band carries the identity keys (element/short_name/unit/comment/discipline);
            when ``False`` only the reference/valid time and forecast horizon, to exercise the unnamed path.

    Returns:
        A pyramids ``Dataset`` of the requested band count, EPSG:4326, with GRIB message metadata per band.
    """
    cells = np.arange(12, dtype="float32").reshape(3, 4)
    array = cells if bands == 1 else np.stack([cells] * bands)
    reference = GeoReference(geo=(0.0, 1.0, 0.0, 3.0, 0.0, -1.0), epsg=4326)
    grid = Dataset.from_array(array, geo_ref=reference)
    base = 1609459200  # 2021-01-01T00:00:00Z, in GRIB's "<seconds> sec UTC" spelling
    per_band = []
    for index in range(1, bands + 1):
        step = index * 21600  # 6 h per step
        message = {
            "GRIB_REF_TIME": f"{base} sec UTC",
            "GRIB_VALID_TIME": f"{base + step} sec UTC",
            "GRIB_FORECAST_SECONDS": f"{step} sec",
        }
        if named:
            message |= {
                "GRIB_ELEMENT": element,
                "GRIB_SHORT_NAME": "2-HTGL",
                "GRIB_UNIT": "[C]",
                "GRIB_COMMENT": "Temperature",
                "GRIB_DISCIPLINE": "0",
            }
        per_band.append(message)
    grid.band_meta_data = per_band
    return grid


@pytest.fixture
def grib_dataset() -> Dataset:
    """A single-band Dataset carrying GRIB message metadata (see :func:`_grib_like_dataset`)."""
    return _grib_like_dataset(bands=1)


@pytest.fixture
def grib_stack() -> Dataset:
    """A three-band GRIB Dataset: one band per 6-hour forecast step, each with its own valid time."""
    return _grib_like_dataset(bands=3)


@pytest.fixture
def grib_dataset_timeonly() -> Dataset:
    """A GRIB Dataset whose message carries only a valid time + forecast horizon (no element/short_name)."""
    return _grib_like_dataset(bands=1, named=False)


@pytest.fixture
def grib_dataset_unitonly() -> Dataset:
    """A GRIB band that declares only a unit (``GRIB_UNIT``) — no identity keys and no time."""
    reference = GeoReference(geo=(0.0, 1.0, 0.0, 3.0, 0.0, -1.0), epsg=4326)
    grid = Dataset.from_array(
        np.arange(12, dtype="float32").reshape(3, 4), geo_ref=reference
    )
    grid.band_meta_data = [{"GRIB_UNIT": "[K]"}]
    return grid


@pytest.fixture
def grib_collection():
    """A DatasetCollection of two single-band GRIB members, each carrying its own message metadata."""
    from pyramids.dataset.collection import DatasetCollection

    members = [_grib_like_dataset(bands=1), _grib_like_dataset(bands=1)]
    return DatasetCollection(members[0], len(members), datasets=members)


@pytest.fixture(scope="module")
def points() -> GeoDataFrame:
    return gpd.read_file("tests/data/points.geojson")


@pytest.fixture
def ugrid_mesh():
    """Return a factory for a two-face UGRID mesh: a quad and a triangle.

    The quad fan-triangulates into two triangles and the triangle into one, so a face-located value that
    is spread correctly reads back as ``[10, 10, 20]`` — the mesh's own faces rather than a Delaunay guess.
    Shared by the base, static and quickmap suites, which all need the same mesh.

    Returns:
        Callable: ``build(location="face", node_x=None, steps=1, epsg=None)`` → a pyramids
        ``UgridDataset``. ``location`` is ``"face"`` or ``"node"``; ``node_x`` overrides the node x
        coordinates (pass non-finite ones to make a mesh no projection can place); ``steps`` > 1 gives the
        variable a leading time axis, which is the shape UGRID model output normally has; ``epsg`` makes
        the mesh declare a CRS, which is what puts it on the reprojecting path.
    """
    import numpy as np
    from pyramids.netcdf.ugrid.connectivity import Connectivity
    from pyramids.netcdf.ugrid.dataset import UgridDataset
    from pyramids.netcdf.ugrid.mesh import Mesh2d
    from pyramids.netcdf.ugrid.models import MeshVariable

    def build(location: str = "face", node_x=None, steps: int = 1, epsg=None):
        mesh = Mesh2d(
            node_x=(
                np.array([0.0, 1.0, 1.0, 0.0, 2.0])
                if node_x is None
                else np.asarray(node_x)
            ),
            node_y=np.array([0.0, 0.0, 1.0, 1.0, 0.5]),
            face_node_connectivity=Connectivity(
                data=np.array([[0, 1, 2, 3], [1, 4, 2, -1]], dtype=np.intp),
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
        )
        name = "depth" if location == "face" else "zs"
        per_step = (
            np.array([10.0, 20.0])
            if location == "face"
            else np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        # steps > 1 gives the variable the leading time axis real UGRID output carries, each step offset
        # by 100 so the step that was drawn is identifiable from its values alone.
        data = (
            per_step
            if steps == 1
            else np.stack([per_step + 100.0 * step for step in range(steps)])
        )
        held = {
            name: MeshVariable(
                name=name,
                location=location,
                mesh_name="m",
                shape=data.shape,
                _data=data,
            )
        }
        wkt = None
        if epsg is not None:
            from pyramids.base.crs import crs_from_user_input

            wkt = crs_from_user_input(epsg).to_wkt()
        return UgridDataset(
            mesh=mesh, data_variables=held, global_attributes={}, crs_wkt=wkt
        )

    return build
