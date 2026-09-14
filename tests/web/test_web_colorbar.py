"""``quickmap(colorbar=...)`` means the same thing on the web tier as everywhere else (#254).

The web tier's colour key is ``WebMap.legend``, a *builder* that reads what the last classified layer
recorded — not the boolean toggle the interactive tier exposes. ``colorbar=`` used to be refused here for that
reason, so one call written the same way worked on three backends and raised on the fourth. It is now
translated in ``api.py`` instead.
"""

import pytest

import digitalearth.api as qp
from digitalearth.api import BACKEND_CAPABILITIES


@pytest.fixture(autouse=True)
def _need_engine():
    """Skip the module when the web extra is absent."""
    pytest.importorskip("maplibre")


@pytest.fixture()
def polygons_gdf():
    """Four triangles in lon/lat (EPSG:4326) with a ``pop`` value ramp."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    geoms = [
        Polygon([(0, 0), (1, 0), (1, 1)]),
        Polygon([(2, 2), (3, 2), (3, 3)]),
        Polygon([(4, 4), (5, 4), (5, 5)]),
        Polygon([(6, 6), (7, 6), (7, 7)]),
    ]
    return gpd.GeoDataFrame({"pop": [1.0, 5.0, 9.0, 3.0]}, geometry=geoms, crs=4326)


class TestEveryBackendHonoursColorbar:
    """The capability table is what ``quickmap`` refuses by, so every tier must declare it."""

    def test_no_backend_refuses_colorbar(self):
        """All four backends accept ``colorbar=``, so the same call works everywhere.

        Test scenario:
            The divergence in #254: ``web`` alone lacked the capability, so ``quickmap(ds, backend="web",
            colorbar=True)`` raised while the other three drew a key. Asserting over the whole table is what
            stops a fifth backend arriving without one.
        """
        missing = sorted(
            name
            for name, caps in BACKEND_CAPABILITIES.items()
            if "colorbar" not in caps
        )
        assert not missing, f"these backends still refuse colorbar=: {missing}"


class TestTheWebKeyFollowsTheFlag:
    """``colorbar=True`` builds the key when there is one to build, and is tolerated when there is not."""

    def test_a_classified_map_gets_its_key(self, polygons_gdf):
        """A choropleth built through ``quickmap`` carries a legend panel.

        Args:
            polygons_gdf: Four triangles with a ``pop`` ramp to classify.

        Test scenario:
            ``colorbar=True`` is the default, so the one-call path must produce the key a thematic map is
            unreadable without — the same thing ``backend="matplotlib"`` does with a colorbar.
        """
        scene = qp.quickmap(polygons_gdf, backend="web", column="pop", colorbar=True)
        assert "legend" in scene._panels, (
            f"a classified web map must carry a key, panels present: {sorted(scene._panels)}"
        )

    def test_colorbar_false_leaves_no_key(self, polygons_gdf):
        """``colorbar=False`` suppresses the key on a map that could have carried one.

        Args:
            polygons_gdf: Four triangles with a ``pop`` ramp to classify.

        Test scenario:
            The flag has to be a real choice on this tier, not just an accepted no-op — otherwise "every
            backend honours it" is only half true.
        """
        scene = qp.quickmap(polygons_gdf, backend="web", column="pop", colorbar=False)
        assert "legend" not in scene._panels, (
            "colorbar=False must leave the map without a key"
        )

    def test_a_map_with_nothing_to_describe_is_tolerated(self, polygons_gdf):
        """An unclassified layer under the default ``colorbar=True`` warns rather than raising.

        Args:
            polygons_gdf: The same triangles, drawn without a ``column`` so nothing is classified.

        Test scenario:
            ``WebMap.legend`` refuses a map with no classification to describe. Under the default that is not
            a caller error — they asked for a key *if there is one* — so it is skipped the same way the
            matplotlib path skips an outline-only layer, rather than turning the default into a crash.
        """
        scene = qp.quickmap(polygons_gdf, backend="web", colorbar=True)
        assert "legend" not in scene._panels, (
            "an unclassified map has no key to draw, so none should appear"
        )
