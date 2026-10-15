"""Verify model-grid fields render once they are regridded to a regular Dataset (gap #459).

HEALPix / reduced-Gaussian / ORCA recognition lives upstream in ``earthlens.grids`` (per pyramids#1307's
resolution — out of scope for pyramids), and each ``from_healpix`` / ``from_octahedral`` / ``from_orca``
returns a **regular** ``pyramids.dataset.Dataset``. So there is no exotic geometry left by the time a field
reaches digitalearth, and no new render path is needed here — the regridded Dataset is an ordinary raster.

These tests pin that contract without importing earthlens (not a dependency, and not installed in CI): they
stand in for the earthlens step with pyramids' own ``Dataset.from_points`` — the same scattered-points ->
regular-grid regridding earthlens.grids performs internally — and prove the result ingests and renders through
the normal field path.
"""

import geopandas as gpd
import numpy as np
from matplotlib.image import AxesImage
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection
from shapely.geometry import Point

from digitalearth.base.sources import Source, get_source
from digitalearth.static import Map


def _regridded_field() -> Dataset:
    """A regular Dataset interpolated from scattered lon/lat samples — the shape earthlens.grids returns."""
    rng = np.random.default_rng(0)
    lon = rng.uniform(-10.0, 10.0, 300)
    lat = rng.uniform(40.0, 60.0, 300)
    frame = gpd.GeoDataFrame(
        {
            "t": np.cos(np.radians(lat)) * 20.0,
            "geometry": [Point(x, y) for x, y in zip(lon, lat)],
        },
        crs="EPSG:4326",
    )
    return Dataset.from_points(
        FeatureCollection(frame),
        value_column="t",
        cell_size=1.0,
        bbox=(-10.0, 40.0, 10.0, 60.0),
        epsg=4326,
    )


class TestRegriddedModelGridField:
    """A field regridded off a model grid is an ordinary raster to digitalearth."""

    def test_regridded_result_is_a_regular_raster(self):
        """from_points yields a regular, CRS-bearing Dataset (what earthlens.grids hands back)."""
        ds = _regridded_field()
        assert (ds.rows, ds.columns) == (20, 20)
        assert ds.epsg == 4326

    def test_it_ingests_as_a_normal_raster_source(self):
        """get_source reads the regridded Dataset as a plain raster Source — no special grid handling."""
        src = get_source(_regridded_field())
        assert isinstance(src, Source)
        assert src.z.values.shape == (20, 20)
        assert src.metadata("kind") == "raster"

    def test_it_renders_through_the_field_path(self):
        """Map.field draws the regridded Dataset through the ordinary raster path, no new code required."""
        canvas = Map(crs=4326)
        canvas.field(_regridded_field())
        assert isinstance(canvas.artist(), AxesImage)
