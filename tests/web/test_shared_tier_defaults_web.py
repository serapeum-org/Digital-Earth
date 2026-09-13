"""M4/M8/M9 — the web tier's half of the shared-constant regressions, with MapLibre present.

``tests/base/test_shared_tier_defaults.py`` pins the agreement between the tiers; this file watches the web
tier honour it through its own public builders in the ``web`` environment — the basemap source a bare
``basemap()`` adds, the rate ``animate`` encodes at, and the cutoff a per-call override is checked against.

Run it with ``pixi run -e web test-web``; without the extra the whole module skips.
"""

import pytest

pytest.importorskip("maplibre")

import geopandas as gpd  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from digitalearth.base.animation import DEFAULT_FPS  # noqa: E402
from digitalearth.base.basemaps import DEFAULT_BASEMAP_PROVIDER  # noqa: E402
from digitalearth.base.bigdata import DEFAULT_BIG_DATA_THRESHOLD  # noqa: E402
from digitalearth.web import WebMap  # noqa: E402
from digitalearth.web.decoration import _BASEMAP_PROVIDERS  # noqa: E402

#: The Carto tile set every tier requests for the shared default provider — see the base-tier test module.
CARTO_LIGHT = "light_all"


@pytest.fixture
def points():
    """Three points in lon/lat, the smallest input the vector builders accept."""
    return gpd.GeoDataFrame(
        {"value": [1.0, 2.0, 3.0]},
        geometry=[Point(0.0, 0.0), Point(1.0, 1.0), Point(2.0, 2.0)],
        crs=4326,
    )


class TestTheDefaultBasemapIsTheSharedOne:
    """M4 — an unnamed basemap fetches the same tiles here as on the static and interactive tiers."""

    def test_the_default_provider_maps_to_the_carto_light_tiles(self):
        """The tier's table resolves the shared name to the Carto light tile set.

        Test scenario:
            Each tier spells the provider for its own engine, so the only thing that can be compared across
            them is the tile set actually requested. This is the web tier's end of that comparison.
        """
        url = _BASEMAP_PROVIDERS[DEFAULT_BASEMAP_PROVIDER.lower()][0]
        assert CARTO_LIGHT in url, url

    def test_a_bare_basemap_call_adds_a_layer(self):
        """``basemap()`` with no argument resolves and draws, rather than raising on an unknown name."""
        assert WebMap().basemap().layers, "the shared default added no basemap"


class TestTheAnimationRateIsTheSharedOne:
    """M9 — the tier's ``DEFAULT_FPS`` is the object ``base/animation.py`` declares, not a copy."""

    def test_the_tier_constant_is_the_shared_object(self):
        """``web.export.DEFAULT_FPS`` *is* the shared constant.

        Test scenario:
            The tier declared its own ``DEFAULT_FPS = 3.0``. ``is`` rather than ``==`` is what tells a shared
            constant from a re-declared literal that happens to match today.
        """
        from digitalearth.web import export

        assert export.DEFAULT_FPS is DEFAULT_FPS

    def test_animate_falls_back_to_the_shared_rate(self):
        """``animate`` with no ``fps`` encodes at the shared rate.

        Test scenario:
            ``fps`` is ``None`` in the signature — the "not passed" sentinel the deprecated ``duration=`` is
            resolved against — so the number a caller actually gets is the ``default=`` handed to
            ``renamed_parameter``, which is what has to be the shared one.
        """
        import inspect

        from digitalearth.web import export

        source = inspect.getsource(export.ExportMixin.animate)
        assert "default=DEFAULT_FPS," in source


class TestTheBigDataCutoffIsTheSharedOne:
    """M8 — the map's cutoff is the shared constant, and a bad override is refused as on the interactive tier."""

    def test_the_map_carries_the_shared_default(self):
        """A freshly built map routes at the number every tier reads."""
        assert WebMap().big_data_threshold == DEFAULT_BIG_DATA_THRESHOLD

    def test_a_negative_override_is_refused_by_a_real_builder(self, points):
        """The shared guard still fires through the public builder, and names the call.

        Args:
            points: A three-row point frame.

        Test scenario:
            The check moved out of this tier into ``base/bigdata.py`` so the interactive tier could apply it
            too; this pins that the move did not soften what the web tier already refused.
        """
        scene = WebMap()
        with pytest.raises(ValueError, match=r"points\(\): big_data_threshold"):
            scene.points(points, big_data_threshold=-1)
