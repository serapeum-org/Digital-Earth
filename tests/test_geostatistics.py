"""Tests for digitalearth.geostatistics — visualizing geostatista results in the map tiers.

Wires geostatista's spatial-statistics + kriging outputs into Digital-Earth's map composition. geostatista owns
the computation; these tests only assert the *visualization* (categorical LISA/hotspot maps with the
conventional palette, and the kriged-surface composition).
"""
import geopandas as gpd
import matplotlib
import pytest
from matplotlib.colors import to_hex
from shapely.geometry import Point, box

matplotlib.use("Agg")

from geostatista import Weights, getis_ord_gi, local_morans  # noqa: E402
from pyramids.feature import FeatureCollection  # noqa: E402

from digitalearth._symbology import _categories  # noqa: E402
from digitalearth.geostatistics import (  # noqa: E402
    HOTSPOT_COLORS,
    LISA_COLORS,
    hotspot_map,
    kriging_map,
    lisa_map,
)


def _rendered_colors(scene, categories):
    """Read the hex colour the drawn choropleth actually renders for each sorted category."""
    collection = scene.ax.collections[-1]
    cmap, norm = collection.get_cmap(), collection.norm
    return {cat: to_hex(cmap(norm(i))) for i, cat in enumerate(categories)}


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

    def test_renders_conventional_palette(self, clustered_polygons):
        """Each present LISA class is *rendered* in its GeoDa colour (HH red, LL blue, ns grey)."""
        lm = local_morans(clustered_polygons, "v", Weights.queen(clustered_polygons))
        scene = lisa_map(lm)
        cats = _categories(lm["cluster"])
        rendered = _rendered_colors(scene, cats)
        assert set(cats) == set(rendered)
        for label, hexcolor in rendered.items():
            assert hexcolor.lower() == LISA_COLORS[str(label)].lower()

    def test_explicit_cmap_is_honoured(self, clustered_polygons):
        """Passing an explicit cmap overrides the conventional palette (no semantic palette relied on)."""
        lm = local_morans(clustered_polygons, "v", Weights.queen(clustered_polygons))
        scene = lisa_map(lm, cmap="Set2")
        assert len(scene.layers) == 1

    def test_scheme_override_rejected(self, clustered_polygons):
        """A non-categorical scheme is a clear error, not a confusing keyword collision."""
        lm = local_morans(clustered_polygons, "v", Weights.queen(clustered_polygons))
        with pytest.raises(ValueError, match="categorical"):
            lisa_map(lm, scheme="quantiles")

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

    def test_renders_conventional_palette(self, clustered_polygons):
        """Hot spots render red, cold spots blue, non-significant grey."""
        go = getis_ord_gi(clustered_polygons, "v", Weights.queen(clustered_polygons))
        scene = hotspot_map(go)
        cats = _categories(go["hotspot"])
        rendered = _rendered_colors(scene, cats)
        assert set(cats) == set(rendered)
        for label, hexcolor in rendered.items():
            assert hexcolor.lower() == HOTSPOT_COLORS[str(label)].lower()

    def test_missing_column_raises(self, clustered_polygons):
        """A FeatureCollection without the hotspot column is a clear error."""
        with pytest.raises(KeyError):
            hotspot_map(clustered_polygons, column="hotspot")


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

    def test_invalid_field_rejected(self, dataset):
        """An unknown field name is rejected before a figure is created, not via a confusing AttributeError."""
        with pytest.raises(ValueError, match="field must be one of"):
            kriging_map(dataset, field="save")

    def test_variance_without_variance_attribute_raises(self, dataset):
        """variance=True on a plain Dataset (no .variance) is a clear error before any figure is built."""
        with pytest.raises(AttributeError, match="variance"):
            kriging_map(dataset, variance=True)

    @pytest.mark.xfail(
        raises=AttributeError,
        strict=True,
        reason="geostatista 0.2.0 KrigedSurface.from_arrays calls Dataset.create_from_array, removed in "
        "pyramids 0.59 (now from_array/create). krige/predict_grid raise before a surface exists. When "
        "geostatista is fixed this XPASSES (strict) and fails the suite — the signal to remove this marker.",
    )
    def test_end_to_end_via_krige_blocked_upstream(self):
        """Full path (samples → krige → kriging_map) — xfail until geostatista is fixed for pyramids 0.59."""
        from geostatista import Samples

        coords = [(i * 1.7 % 10, i * 2.3 % 10) for i in range(30)]
        pts = [Point(x, y) for x, y in coords]
        fc = Samples(gpd.GeoDataFrame({"v": [x + y for x, y in coords]}, geometry=pts, crs="EPSG:32631"))
        surface = fc.krige("v", "spherical", cell_size=0.5)  # raises AttributeError on pyramids 0.59
        scene = kriging_map(surface, samples=fc)
        assert len(scene.layers) == 2
