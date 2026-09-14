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
def polygon_fc():
    """A polygon ``FeatureCollection`` carrying an ``fid`` column to classify.

    ``quickmap`` dispatches on pyramids types, so this is a ``FeatureCollection`` rather than the bare
    ``GeoDataFrame`` the tier builders also accept.
    """
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    fc["geometry"] = fc.geometry.buffer(500.0)
    return fc


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

    def test_a_classified_map_gets_its_key(self, polygon_fc):
        """A choropleth built through ``quickmap`` carries a legend panel.

        Args:
            polygon_fc: A polygon collection with an ``fid`` column to classify.

        Test scenario:
            ``colorbar=True`` is the default, so the one-call path must produce the key a thematic map is
            unreadable without — the same thing ``backend="matplotlib"`` does with a colorbar.
        """
        scene = qp.quickmap(polygon_fc, backend="web", column="fid", colorbar=True)
        assert "legend" in scene._panels, (
            f"a classified web map must carry a key, panels present: {sorted(scene._panels)}"
        )

    def test_colorbar_false_leaves_no_key(self, polygon_fc):
        """``colorbar=False`` suppresses the key on a map that could have carried one.

        Args:
            polygon_fc: A polygon collection with an ``fid`` column to classify.

        Test scenario:
            The flag has to be a real choice on this tier, not just an accepted no-op — otherwise "every
            backend honours it" is only half true.
        """
        scene = qp.quickmap(polygon_fc, backend="web", column="fid", colorbar=False)
        assert "legend" not in scene._panels, (
            "colorbar=False must leave the map without a key"
        )

    def test_a_map_with_nothing_to_describe_is_tolerated(self, polygon_fc):
        """An unclassified layer under the default ``colorbar=True`` is inert rather than raising.

        Args:
            polygon_fc: The same collection, drawn without a ``column`` so nothing is classified.

        Test scenario:
            ``WebMap.legend`` refuses a map with no classification to describe. Under the default that is not
            a caller error — they asked for a key *if there is one* — so it is skipped silently, rather
            than turning the default into a crash on unclassified input. Nothing is logged: the guard
            answers from recorded state and never reaches the builder.
        """
        scene = qp.quickmap(polygon_fc, backend="web", colorbar=True)
        assert "legend" not in scene._panels, (
            "an unclassified map has no key to draw, so none should appear"
        )
