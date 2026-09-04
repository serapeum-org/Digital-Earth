"""Tests for digitalearth.geostatistics — visualizing geostatista results in the map tiers.

Wires geostatista's spatial-statistics + kriging outputs into Digital-Earth's map composition. geostatista owns
the computation; these tests only assert the *visualization* (categorical LISA/hotspot maps with the
conventional palette, and the kriged-surface composition).
"""
import geopandas as gpd
import matplotlib
import pytest
from shapely.geometry import Point, box

matplotlib.use("Agg")

from matplotlib import colormaps  # noqa: E402

from geostatista import Weights, getis_ord_gi, local_morans  # noqa: E402
from pyramids.dataset import Dataset  # noqa: E402
from pyramids.feature import FeatureCollection  # noqa: E402

from digitalearth._symbology import _categories  # noqa: E402
from digitalearth.geostatistics import (  # noqa: E402
    HOTSPOT_COLORS,
    LISA_COLORS,
    hotspot_map,
    kriging_map,
    lisa_map,
)


@pytest.fixture
def clustered_polygons() -> FeatureCollection:
    """A 6×6 grid of unit squares with a sharp low/high west–east split — yields real HH/LL clusters."""
    polys = [box(i, j, i + 1, j + 1) for j in range(6) for i in range(6)]
    vals = [(0.0 if i < 3 else 10.0) for _ in range(6) for i in range(6)]
    return FeatureCollection(gpd.GeoDataFrame({"v": vals}, geometry=polys, crs="EPSG:32631"))


class TestLisaMap:
    """lisa_map — LISA (local Moran) clusters drawn as a categorical choropleth."""

    def test_draws_one_categorical_layer(self, clustered_polygons):
        """A local_morans result renders as a single categorical Scene layer."""
        lm = local_morans(clustered_polygons, "v", Weights.queen(clustered_polygons))
        scene = lisa_map(lm)
        assert len(scene.layers) == 1

    def test_uses_conventional_palette(self, clustered_polygons):
        """Each present LISA class is pinned to its GeoDa colour (HH red, LL blue, ns grey), not an arbitrary hue."""
        lm = local_morans(clustered_polygons, "v", Weights.queen(clustered_polygons))
        lisa_map(lm)
        cats = _categories(lm["cluster"])
        palette = dict(zip(cats, colormaps["_de_geostat_lisa"].colors))
        for label, hexcolor in palette.items():
            assert hexcolor.lower() == LISA_COLORS[str(label)].lower()

    def test_explicit_cmap_is_honoured(self, clustered_polygons):
        """Passing an explicit cmap overrides the conventional palette (no semantic registration relied on)."""
        lm = local_morans(clustered_polygons, "v", Weights.queen(clustered_polygons))
        scene = lisa_map(lm, cmap="Set2")
        assert len(scene.layers) == 1

    def test_missing_column_raises(self, clustered_polygons):
        """A FeatureCollection without the cluster column is a clear error, not a silent empty map."""
        with pytest.raises(KeyError):
            lisa_map(clustered_polygons, column="cluster")


class TestHotspotMap:
    """hotspot_map — Getis-Ord Gi* hot/cold spots drawn as a categorical choropleth."""

    def test_draws_one_categorical_layer(self, clustered_polygons):
        """A getis_ord_gi result renders as a single categorical Scene layer."""
        go = getis_ord_gi(clustered_polygons, "v", Weights.queen(clustered_polygons))
        scene = hotspot_map(go)
        assert len(scene.layers) == 1

    def test_uses_conventional_palette(self, clustered_polygons):
        """Hot spots are red, cold spots blue, non-significant grey."""
        go = getis_ord_gi(clustered_polygons, "v", Weights.queen(clustered_polygons))
        hotspot_map(go)
        cats = _categories(go["hotspot"])
        palette = dict(zip(cats, colormaps["_de_geostat_hotspot"].colors))
        for label, hexcolor in palette.items():
            assert hexcolor.lower() == HOTSPOT_COLORS[str(label)].lower()


class TestKrigingMap:
    """kriging_map — a kriged surface draped as a raster field, optionally with the sample overlay."""

    def test_composition_with_stand_in_surface(self, dataset):
        """The surface-draping composition works on a pyramids Dataset (a KrigedSurface is one).

        Uses a real Dataset as a stand-in because geostatista 0.2.0 cannot *produce* a KrigedSurface on
        pyramids 0.59 (see ``test_end_to_end_via_krige_blocked_upstream``); this isolates and verifies the
        Digital-Earth composition itself.
        """
        scene = kriging_map(dataset)
        assert len(scene.layers) == 1

    def test_composition_overlays_samples(self, dataset):
        """Passing samples adds a second (scatter) layer over the surface."""
        cx, cy = dataset.bbox[0] + 1000.0, dataset.bbox[1] + 1000.0
        samples = FeatureCollection(
            gpd.GeoDataFrame({"v": [1.0, 2.0]}, geometry=[Point(cx, cy), Point(cx + 4000.0, cy + 4000.0)],
                             crs=f"EPSG:{dataset.epsg}")
        )
        scene = kriging_map(dataset, samples=samples)
        assert len(scene.layers) == 2

    @pytest.mark.xfail(
        raises=AttributeError,
        strict=False,
        reason="geostatista 0.2.0 KrigedSurface.from_arrays calls Dataset.create_from_array, removed in "
        "pyramids 0.59 (now from_array/create). krige/predict_grid raise before a surface exists. Upstream "
        "geostatista fix needed; remove xfail once released.",
    )
    def test_end_to_end_via_krige_blocked_upstream(self):
        """Full path (samples → krige → kriging_map) — xfail until geostatista is fixed for pyramids 0.59."""
        from geostatista import Samples

        rng = [(i * 1.7 % 10, i * 2.3 % 10) for i in range(30)]
        pts = [Point(x, y) for x, y in rng]
        fc = Samples(gpd.GeoDataFrame({"v": [x + y for x, y in rng]}, geometry=pts, crs="EPSG:32631"))
        surface = fc.krige("v", "spherical", cell_size=0.5)  # raises AttributeError on pyramids 0.59
        scene = kriging_map(surface, samples=fc)
        assert len(scene.layers) == 2
