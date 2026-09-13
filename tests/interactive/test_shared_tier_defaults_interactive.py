"""M4/M8/M9 — the interactive tier's half of the shared-constant regressions, with GeoViews present.

``tests/base/test_shared_tier_defaults.py`` pins the agreement itself and runs engine-free; what it cannot do
is watch a tile element actually get built. This file does that in the ``interactive`` environment: the
basemap the tier requests when the caller named none, the frame rate its animation entry points start from,
and the big-data cutoff its builders route on.

Run it with ``pixi run -e interactive test-interactive``; without the extra the whole module skips.
"""

import pytest

pytest.importorskip("geoviews")
pytest.importorskip("holoviews")

from digitalearth.base.animation import DEFAULT_FPS  # noqa: E402
from digitalearth.base.bigdata import DEFAULT_BIG_DATA_THRESHOLD  # noqa: E402
from digitalearth.interactive import InteractiveMap  # noqa: E402

#: The Carto tile set every tier requests for the shared default provider — see the base-tier test module.
CARTO_LIGHT = "light_all"


@pytest.fixture
def point_fc():
    """The repo's point fixture as a pyramids ``FeatureCollection`` (EPSG:32618, numeric ``fid``)."""
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


class TestTheDefaultBasemapIsTheSharedOne:
    """M4 — an unnamed basemap fetches the same tiles here as on the static and web tiers."""

    def test_tiles_with_no_provider_requests_the_carto_light_set(self):
        """The built WMTS element points at the Carto light tiles.

        Test scenario:
            The constant only settles the *name*; each tier still resolves it through its own engine. Reading
            the URL off the element is what proves this tier's resolution lands on the same tile set the
            static tier's ``xyzservices`` path and the web tier's URL template do.
        """
        element = InteractiveMap().tiles().layers[0]
        assert CARTO_LIGHT in element.data, element.data


class TestTheAnimationRateIsTheSharedOne:
    """M9 — both entry points start from the one rate, and it is the object from ``base/``."""

    @pytest.mark.parametrize("method", ["play", "save_animation"])
    def test_the_entry_point_defaults_to_the_shared_rate(self, method):
        """Each signature default is the shared constant itself.

        Args:
            method: The animation entry point to inspect.

        Test scenario:
            Both were written as a bare ``fps: float = 3.0``, so the tier agreed with the others only by
            coincidence. ``is`` is the check that a literal of the same value would fail.
        """
        import inspect

        default = (
            inspect.signature(getattr(InteractiveMap, method)).parameters["fps"].default
        )
        assert default is DEFAULT_FPS


class TestTheBigDataCutoffIsTheSharedOne:
    """M8 — the map's cutoff is the shared constant, and a bad override is refused as on the web tier."""

    def test_the_map_carries_the_shared_default(self):
        """A freshly built map routes at the number every tier reads."""
        assert InteractiveMap().big_data_threshold == DEFAULT_BIG_DATA_THRESHOLD

    def test_a_negative_override_is_refused_by_a_real_builder(self, point_fc):
        """The guard fires through the public builder, not only through the resolver.

        Args:
            point_fc: A small point ``FeatureCollection`` fixture.

        Test scenario:
            This tier accepted ``big_data_threshold=-1`` and quietly routed every layer through Datashader,
            while the web tier raised on the same call. The builder is where a user meets it.
        """
        fresh_map = InteractiveMap()
        with pytest.raises(ValueError, match="must not be negative"):
            fresh_map.points(point_fc, big_data_threshold=-1)
