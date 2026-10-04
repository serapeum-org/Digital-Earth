"""OGC WMS and WMTS basemaps through ``Map.basemap`` (ST-24).

cleopatra 0.38 added ``cleopatra.basemap.ogc.WMSProvider`` and ``WMTSProvider``: provider objects that build a
``GetMap``/``GetTile`` URL per tile, which ``add_tiles`` accepts beside an ``xyzservices`` provider. These pin
that both pass straight through the static tier's ``basemap()``, and that the URLs actually requested are the
OGC requests the provider describes.

Nothing here reaches the network: the HTTP call underneath cleopatra's tile fetch is replaced by a stand-in
that records each request and answers with one PNG, so the real fetch, stitch and draw all run.
"""

import io
import json
import warnings
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from cleopatra.basemap.ogc import WMSProvider, WMTSProvider
from PIL import Image

from digitalearth.api import to_backend
from digitalearth.base.spec import FigureSpec
from digitalearth.static import Map

WMS_URL = "https://example.test/geoserver/wms"
WMTS_URL = "https://example.test/wmts"

#: The Netherlands in Web Mercator, (xmin, xmax, ymin, ymax).
NETHERLANDS_3857 = (350_000.0, 800_000.0, 6_550_000.0, 7_100_000.0)


class _Response:
    """The one method of an HTTP response cleopatra's tile fetch reads."""

    def __init__(self, payload: bytes):
        """Hold the body to hand back.

        Args:
            payload: The response body.
        """
        self._payload = payload

    def read(self) -> bytes:
        """Return the body.

        Returns:
            The bytes given at construction.
        """
        return self._payload


@pytest.fixture
def requested(monkeypatch):
    """Answer every tile request from memory and record the URL asked for.

    Args:
        monkeypatch: pytest's patcher, which restores the real HTTP call afterwards.

    Returns:
        The list of URLs requested, one per tile.
    """
    from cleopatra.basemap import tiles as cleo_tiles

    buffer = io.BytesIO()
    Image.fromarray(np.full((256, 256, 4), 180, "uint8")).save(buffer, format="PNG")
    payload = buffer.getvalue()
    urls = []

    def answer(request, timeout=None):
        """Record the request and answer with the PNG.

        Args:
            request: The ``urllib.request.Request`` cleopatra built.
            timeout: Ignored.

        Returns:
            A response whose body is the PNG.
        """
        urls.append(request.full_url)
        return _Response(payload)

    monkeypatch.setattr(cleo_tiles, "urlopen_http", answer)
    return urls


def _framed_map() -> Map:
    """Return a Web Mercator map framed on the Netherlands, ready to tile.

    Returns:
        The map.
    """
    canvas = Map(crs=3857)
    xmin, xmax, ymin, ymax = NETHERLANDS_3857
    canvas.ax.set_xlim(xmin, xmax)
    canvas.ax.set_ylim(ymin, ymax)
    return canvas


def _query(url: str) -> dict:
    """Return a URL's query as upper-cased keys mapped to single values.

    Args:
        url: The requested URL.

    Returns:
        The query parameters.
    """
    return {
        key.upper(): values[0] for key, values in parse_qs(urlsplit(url).query).items()
    }


class TestWms:
    """A WMS service is tiled as ``GetMap`` requests, one per tile."""

    def test_a_wms_provider_draws_tiles(self, requested):
        """The provider object reaches ``add_tiles`` and an image lands on the axes.

        Args:
            requested: The recorded tile requests.
        """
        canvas = _framed_map()
        canvas.basemap(WMSProvider(WMS_URL, "topp:states"))
        assert requested, "no tile was requested"
        assert canvas.ax.images, "no tile image was drawn"

    def test_every_request_is_a_getmap_for_the_layer(self, requested):
        """Each tile is a WMS ``GetMap`` for the layer the provider names.

        Args:
            requested: The recorded tile requests.
        """
        _framed_map().basemap(WMSProvider(WMS_URL, "topp:states"))
        queries = [_query(url) for url in requested]
        assert {q["SERVICE"] for q in queries} == {"WMS"}, queries
        assert {q["REQUEST"] for q in queries} == {"GetMap"}, queries
        assert {q["LAYERS"] for q in queries} == {"topp:states"}, queries

    def test_every_request_goes_to_the_service(self, requested):
        """The host and path are the service's own.

        Args:
            requested: The recorded tile requests.
        """
        _framed_map().basemap(WMSProvider(WMS_URL, "topp:states"))
        bases = {url.split("?")[0] for url in requested}
        assert bases == {WMS_URL}, bases

    def test_each_request_is_a_box_inside_the_web_mercator_world(self, requested):
        """A tile's BBOX is a Web Mercator box, so it spans no more than the world's width.

        Args:
            requested: The recorded tile requests.
        """
        _framed_map().basemap(WMSProvider(WMS_URL, "topp:states"))
        world = 2 * 20037508.342789244
        for url in requested:
            west, south, east, north = (
                float(v) for v in _query(url)["BBOX"].split(",")
            )
            assert 0 < east - west <= world, url
            assert 0 < north - south <= world, url

    def test_the_layer_is_hidden_by_its_id(self, requested):
        """A WMS basemap is an addressable layer like a named one.

        Args:
            requested: The recorded tile requests.
        """
        canvas = _framed_map()
        canvas.basemap(WMSProvider(WMS_URL, "topp:states"), name="states")
        canvas.set_visible("states", False)
        assert not any(image.get_visible() for image in canvas.ax.images), (
            "hiding the basemap must hide its tiles"
        )


class TestWmts:
    """A WMTS service is tiled as ``GetTile`` requests on its tile matrix set."""

    def test_a_wmts_provider_draws_tiles(self, requested):
        """The provider object reaches ``add_tiles`` and an image lands on the axes.

        Args:
            requested: The recorded tile requests.
        """
        canvas = _framed_map()
        canvas.basemap(WMTSProvider(WMTS_URL, "basemap"))
        assert requested, "no tile was requested"
        assert canvas.ax.images, "no tile image was drawn"

    def test_every_request_is_a_gettile_for_the_layer(self, requested):
        """Each tile is a KVP ``GetTile`` naming the layer and a tile column and row.

        Args:
            requested: The recorded tile requests.
        """
        _framed_map().basemap(WMTSProvider(WMTS_URL, "basemap"))
        queries = [_query(url) for url in requested]
        assert {q["REQUEST"] for q in queries} == {"GetTile"}, queries
        assert {q["LAYER"] for q in queries} == {"basemap"}, queries
        assert all("TILECOL" in q and "TILEROW" in q for q in queries), queries

    def test_a_restful_template_is_filled_per_tile(self, requested):
        """A RESTful WMTS template gets each tile's matrix, row and column substituted.

        Args:
            requested: The recorded tile requests.
        """
        template = WMTS_URL + "/{TileMatrix}/{TileRow}/{TileCol}.png"
        _framed_map().basemap(WMTSProvider(template, "basemap"))
        assert requested, "no tile was requested"
        assert not any("{" in url for url in requested), requested


class TestRoundTrip:
    """A figure that recorded an OGC provider draws the same service when it is read back (M10)."""

    def test_a_wms_figure_read_back_asks_the_same_service(self, requested):
        """A stored WMS figure redrawn elsewhere must reach the service, not the shared default.

        Args:
            requested: The recorded tile requests.

        Test scenario:
            A provider object is held beside the layer and never written into the figure, because it is an
            engine object and an ``xyzservices.TileProvider`` carries the caller's key. ``WMSProvider`` has
            no ``name``, so the description recorded ``source: None`` — the shared default's spelling — and
            a figure read back silently drew CartoDB Positron instead of the service, with no warning and
            no marker that anything was lost. The description must carry enough to rebuild the provider.
        """
        canvas = _framed_map()
        canvas.basemap(WMSProvider(WMS_URL, "topp:states"), name="bm")
        stored = canvas.figure_spec.to_dict()
        requested.clear()
        to_backend(FigureSpec.from_dict(stored), "matplotlib")
        assert requested, "the replayed figure requested no tile at all"
        assert {urlsplit(url).netloc for url in requested} == {"example.test"}, (
            requested
        )
        assert {_query(url)["LAYERS"] for url in requested} == {"topp:states"}, (
            requested
        )

    def test_a_wmts_figure_read_back_asks_the_same_service(self, requested):
        """The same for WMTS, whose recorded fields name a layer rather than a layer list.

        Args:
            requested: The recorded tile requests.

        Test scenario:
            ``WMSProvider`` and ``WMTSProvider`` are separate dataclasses with different fields
            (``layers``/``styles``/``transparent`` against ``layer``/``tile_matrix_set``/``style``), so a
            description that round-trips one need not round-trip the other.
        """
        canvas = _framed_map()
        canvas.basemap(WMTSProvider(WMTS_URL, "basemap"), name="bm")
        stored = canvas.figure_spec.to_dict()
        requested.clear()
        to_backend(FigureSpec.from_dict(stored), "matplotlib")
        assert requested, "the replayed figure requested no tile at all"
        assert {_query(url)["REQUEST"] for url in requested} == {"GetTile"}, requested
        assert {_query(url)["LAYER"] for url in requested} == {"basemap"}, requested

    def test_a_providers_extra_params_stay_out_of_the_stored_figure(self, requested):
        """A service credential lives in ``extra_params``, and that field is never recorded.

        Args:
            requested: The recorded tile requests, so drawing the basemap reaches no network.

        Test scenario:
            Recording the provider is what makes the round trip faithful, and a figure is written to JSON
            and read back, so whatever is recorded travels with it. ``extra_params`` is where a service's
            token goes, so it is the one field held beside the layer and left out of the description — the
            same rule that keeps an ``xyzservices`` ``apikey`` out of a figure.
        """
        canvas = _framed_map()
        canvas.basemap(
            WMSProvider(WMS_URL, "topp:states", extra_params={"token": "s3cr3t"}),
            name="bm",
        )
        stored = json.dumps(canvas.figure_spec.to_dict())
        assert WMS_URL in stored, "the service URL is what makes the figure faithful"
        assert "s3cr3t" not in stored, "a credential must never reach a stored figure"
        assert "token" not in stored, "nor the name of the parameter carrying it"

    def test_a_credential_in_the_service_url_stays_out_of_the_stored_figure(
        self, requested
    ):
        """A token in the service URL's query string is dropped from the record, and the drop is named.

        Args:
            requested: The recorded tile requests, so drawing the basemap reaches no network.

        Test scenario:
            ``extra_params`` being the one field left out protected only a credential passed *that* way.
            A query-string token is an ordinary way for an OGC service to authenticate (``token=``,
            ``api_key=``, ``SERVICE_KEY=``), and the ``url`` field was copied into the figure whole — so
            the same value this tier treats as a secret in ``_quiet_tile_urls`` was written to JSON two
            functions away. Only the base URL is recorded now; the whole query string goes, whatever it
            holds, because nothing here can tell a credential from a parameter. The warning has to name
            the parameters it dropped, since a service that needs one has to be told to pass it through
            ``extra_params`` instead.
        """
        canvas = _framed_map()
        service = WMSProvider(f"{WMS_URL}?token=s3cr3t&api_key=k2", "topp:states")
        with pytest.warns(UserWarning, match="token") as caught:
            canvas.basemap(service, name="bm")
        stored = json.dumps(canvas.figure_spec.to_dict())
        assert "s3cr3t" not in stored, (
            "a URL-borne credential must never reach a figure"
        )
        assert "k2" not in stored, "nor a second one beside it"
        assert WMS_URL in stored, "the base URL still names the service"
        assert "api_key" in str(caught[0].message), (
            f"the warning must name every dropped parameter, got {caught[0].message}"
        )

    def test_a_service_url_with_no_query_is_recorded_unchanged_and_unremarked(
        self, requested
    ):
        """Stripping the query string must not disturb the ordinary case or warn about it.

        Args:
            requested: The recorded tile requests, so drawing the basemap reaches no network.

        Test scenario:
            The sibling test above pins what is dropped; this pins that nothing else is. A plain service
            URL has no query string to lose, so the recorded ``url`` must be the one given — byte for byte,
            not a reassembled near-miss with a trailing ``?`` — and a warning about a credential that was
            never there would train the reader to ignore the one that matters.
        """
        canvas = _framed_map()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            canvas.basemap(WMSProvider(WMS_URL, "topp:states"), name="bm")
        recorded = canvas.figure_spec.layers.get("bm").symbology.props["source"]
        assert recorded["url"] == WMS_URL, (
            f"a query-free URL must be recorded as given, got {recorded['url']!r}"
        )

    def test_a_recorded_kind_this_version_cannot_rebuild_is_refused_by_name(
        self, requested
    ):
        """A figure naming an OGC kind this release has never heard of is refused, not quietly defaulted.

        Args:
            requested: The recorded tile requests, so nothing here reaches the network.

        Test scenario:
            A stored figure carries the provider's kind as an ``ogc`` tag, and a figure written by a newer
            release may name one this version cannot build — here a ``"wcs"`` tag written over the
            recorded ``"wms"`` one. Treating an unknown tag as "no name" would walk straight back into the
            silent substitution the recording exists to end: CartoDB Positron drawn under a figure that
            asked for a service. The refusal names the kind it was handed and the kinds it does know, and
            no tile is requested.
        """
        canvas = _framed_map()
        canvas.basemap(WMSProvider(WMS_URL, "topp:states"), name="bm")
        stored = canvas.figure_spec.to_dict()
        stored["layers"]["layers"][0]["symbology"]["props"]["source"]["ogc"] = "wcs"
        spec = FigureSpec.from_dict(stored)
        requested.clear()
        with pytest.raises(ValueError, match="'wcs'") as refusal:
            to_backend(spec, "matplotlib")
        assert "'wms'" in str(refusal.value) and "'wmts'" in str(refusal.value), (
            f"the refusal must name the kinds it does know, got {refusal.value}"
        )
        assert requested == [], (
            f"a refused figure must request no tile, got {requested}"
        )
