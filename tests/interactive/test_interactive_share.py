"""Shareable URL state beyond four hardcoded names (IN-14).

``share_url`` encodes the whole figure into a ``?state=`` blob that ``from_share_url`` reconstructs, with no
server — the general answer the four-name live ``share`` is not.
"""

import pytest

pytest.importorskip("geoviews")

from digitalearth.interactive import InteractiveMap  # noqa: E402


class TestShareUrl:
    """The whole figure round-trips through the URL it produces."""

    def test_round_trip_preserves_the_display_crs(self):
        """A map encoded to a URL and rebuilt from it keeps its display CRS."""
        url = InteractiveMap(crs=4326).share_url()
        assert InteractiveMap.from_share_url(url).crs == 4326

    def test_base_url_is_prefixed_with_the_right_separator(self):
        """A base that already has a query gets ``&``; a bare base gets ``?``."""
        plain = InteractiveMap(crs=3857).share_url("https://host/app")
        joined = InteractiveMap(crs=3857).share_url("https://host/app?tab=1")
        assert plain.startswith("https://host/app?state=")
        assert "&state=" in joined

    def test_a_trailing_separator_in_the_base_url_is_not_doubled(self):
        """L1 — a base ending in ``?`` or ``&`` must not yield ``?&state=`` / ``&&state=``."""
        q = InteractiveMap(crs=3857).share_url("https://host/app?")
        amp = InteractiveMap(crs=3857).share_url("https://host/app?tab=1&")
        assert q.startswith("https://host/app?state="), q
        assert amp.startswith("https://host/app?tab=1&state="), amp

    def test_a_url_without_state_is_refused(self):
        """Decoding a URL that carries no ``state=`` is a clear error, not a crash."""
        with pytest.raises(ValueError, match="no shareable state"):
            InteractiveMap.from_share_url("https://host/app?tab=1")
