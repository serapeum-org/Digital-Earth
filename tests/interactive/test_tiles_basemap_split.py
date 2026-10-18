"""The ``basemap()`` / ``tiles()`` split on the interactive tier (#268).

``tiles()`` once took a provider name, a keyed preset and a raw XYZ URL through one overloaded ``provider=``.
That is split now: :meth:`~digitalearth.interactive.decoration.decoration.DecorationMixin.basemap` takes a provider name or
keyed preset and :meth:`~digitalearth.interactive.decoration.decoration.DecorationMixin.tiles` takes a raw XYZ/WMTS URL
template, each refusing the other's input. These tests pin that both halves draw and that each refuses the other.

Web-Mercator (``crs="EPSG:3857"``) throughout, so the tiles' projection guard does not trip and the only refusal
under test is the name/URL split itself.
"""

import pytest

from digitalearth.interactive import InteractiveMap

pytest.importorskip("holoviews")
pytest.importorskip("geoviews")
pytest.importorskip("panel")


class TestTilesBasemapSplit:
    """``basemap(name)`` and ``tiles(url)`` each draw; each refuses the other's argument."""

    def test_basemap_draws_from_a_provider_name(self):
        """A provider name resolves through ``basemap`` and adds a layer.

        Test scenario:
            ``basemap("CartoLight")`` on a Web-Mercator map draws a tile layer, so ``.layers`` is non-empty.
        """
        m = InteractiveMap(crs="EPSG:3857").basemap("CartoLight")
        assert m.layers, "basemap(provider=...) drew no layer"

    def test_tiles_draws_from_a_raw_url_template(self):
        """A raw XYZ URL template resolves through ``tiles`` and adds a layer.

        Test scenario:
            ``tiles("https://.../{Z}/{X}/{Y}.png")`` draws a tile layer, so ``.layers`` is non-empty.
        """
        m = InteractiveMap(crs="EPSG:3857").tiles(
            "https://a.tile.example/{Z}/{X}/{Y}.png"
        )
        assert m.layers, "tiles(url=...) drew no layer"

    def test_basemap_refuses_a_url_template(self):
        """``basemap`` is the named-provider half, so a URL template is refused with a pointer to ``tiles``.

        Test scenario:
            ``basemap("https://.../{Z}/{X}/{Y}.png")`` raises ``ValueError`` telling the caller it takes a
            provider name, not a URL template.
        """
        m = InteractiveMap(crs="EPSG:3857")
        with pytest.raises(ValueError, match=r"not a URL template"):
            m.basemap("https://a.tile.example/{Z}/{X}/{Y}.png")

    def test_tiles_refuses_a_provider_name(self):
        """``tiles`` is the raw-URL half, so a provider name is refused with a pointer to ``basemap``.

        Test scenario:
            ``tiles("CartoLight")`` raises ``ValueError`` telling the caller to use ``basemap(provider=...)``.
        """
        m = InteractiveMap(crs="EPSG:3857")
        with pytest.raises(ValueError, match=r"use basemap\(provider="):
            m.tiles("CartoLight")
