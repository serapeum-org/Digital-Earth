"""What ``Map.basemap`` hands back, and what reaches the tile image, against the real ``add_tiles``.

Round 2, M5. ``basemap()``'s ``Returns:`` claimed "the tile artist ``add_tiles`` added to the axes is
reached through ``Scene.artist``, by this layer's id", which the drawer's own docstring in the same module
denies: cleopatra's ``GeoMixin.add_tiles`` (which the static tier's ``decoration.add_tiles`` wrapper calls)
documents ``Returns: matplotlib.axes.Axes: The same axes, for chaining``, so ``artist(id)`` is the axes. The
test that was supposed to prove the claim stubbed
``add_tiles`` with ``return_value=<object()>`` and asserted the layer's artist was that sentinel — a
tautology about the plumbing, true whatever cleopatra returns.

So nothing here mocks ``add_tiles``. The seam moved one level down to
``cleopatra.basemap.tiles.fetch_single_tile``, the HTTP call: it is handed a synthetic PNG instead of
reaching the network, and the **real** ``add_tiles`` then decodes, stitches and ``imshow``s it. What the
tests below assert is therefore cleopatra's real return value and a real ``AxesImage`` on the axes, with no
network — which is the whole of what is observable offline, and all of it is what the claim was about.
"""

import io

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

import matplotlib

matplotlib.use("Agg")

from matplotlib.image import AxesImage  # noqa: E402

from digitalearth import Map  # noqa: E402

#: A frame over Berlin, in degrees, so one zoom level needs only a handful of tiles.
BERLIN = (13.0, 52.3, 13.7, 52.7)


@pytest.fixture
def offline_tiles(monkeypatch):
    """Hand cleopatra's tile fetch a synthetic PNG, leaving the rest of ``add_tiles`` real.

    Args:
        monkeypatch: pytest's patcher, which puts the real fetch back afterwards.

    The patch is the whole effect; ``monkeypatch`` undoes it at teardown, so the fixture needs no yield body.
    """
    from cleopatra.basemap import tiles as cleo_tiles
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGBA", (256, 256), (31, 119, 180, 255)).save(buffer, "PNG")
    png = buffer.getvalue()

    def fetch(tile, provider, timeout, retries, user_agent="test"):
        return tile, png

    monkeypatch.setattr(cleo_tiles, "fetch_single_tile", fetch)


@pytest.fixture
def tiled(offline_tiles):
    """Yield a framed map with a real basemap layer named ``tiles`` drawn on it.

    Args:
        offline_tiles: The stubbed HTTP fetch, so the real ``add_tiles`` runs without the network.

    Yields:
        A :class:`~digitalearth.static.map.Map` carrying one drawn basemap layer.
    """
    scene = Map(domain=BERLIN)
    scene.ax.set_xlim(BERLIN[0], BERLIN[2])
    scene.ax.set_ylim(BERLIN[1], BERLIN[3])
    scene.basemap("OSM", name="tiles", zoom=8)
    yield scene
    scene.close()


class TestWhatABasemapLayerHandsBack:
    """The `Returns:` claim, measured against cleopatra rather than against a sentinel.

    The claim as it stood was written out as a test and run against the real engine first:

    ```
    >       assert isinstance(tiled.artist("tiles"), AxesImage), type(tiled.artist("tiles"))
    E       AssertionError: <class 'matplotlib.axes._axes.Axes'>
    E        +  where False = isinstance(<Axes: >, AxesImage)
    1 failed in 2.32s
    ```
    """

    def test_the_layer_s_artist_is_the_axes_add_tiles_handed_back(self, tiled):
        """What `artist(id)` really answers, which the `Returns:` now says.

        Args:
            tiled: A map with a real basemap layer drawn on it.

        Test scenario:
            `add_tiles` documents `Returns: matplotlib.axes.Axes: The same axes, for chaining`, and the
            drawer files whatever it returned as the layer's artist. Identity against `tiled.ax` rather
            than an `isinstance(..., Axes)` check, because the weaker form would also pass for some other
            axes — and because "it is the axes it drew onto" is the sentence a caller needs.
        """
        assert tiled.artist("tiles") is tiled.ax, type(tiled.artist("tiles"))

    def test_the_tile_image_is_on_those_axes_even_though_it_is_not_what_artist_answers(
        self, tiled
    ):
        """The tiles really were drawn, so the first test is about the return value and not about a no-op.

        Args:
            tiled: A map with a real basemap layer drawn on it.

        Test scenario:
            This is the half the mocked test could never reach: with `add_tiles` replaced by a sentinel,
            nothing is decoded, stitched or shown, so an axes with no image at all would have passed. Here
            the real `add_tiles` ran — only cleopatra's HTTP call was replaced — and left exactly one
            `AxesImage` behind.
        """
        images = [image for image in tiled.ax.images if isinstance(image, AxesImage)]
        assert len(images) == 1, tiled.ax.images

    def test_hiding_the_layer_hides_the_tile_image_the_artist_does_not_name(
        self, tiled
    ):
        """The public route to the tile image is the layer, not `artist()` — which is what the fix says.

        Args:
            tiled: A map with a real basemap layer drawn on it.

        Test scenario:
            The drawer records the images that appeared on the axes while `add_tiles` ran as the layer's
            `artists`, and that is what visibility and removal act on. So `set_visible(id, False)` reaching
            the real `AxesImage` is the observable proof that the layer owns it, with nothing private read.
        """
        tiled.set_visible("tiles", False)
        assert not tiled.ax.images[0].get_visible(), "the tile image stayed visible"

    def test_the_call_still_chains(self, tiled):
        """`Returns: This map (chainable)` is the half of the sentence that was always true.

        Args:
            tiled: A map with a real basemap layer drawn on it.
        """
        assert tiled.basemap("OSM", name="second", zoom=8) is tiled, (
            "basemap did not chain"
        )


class TestCleopatra046ImportDrift:
    """Regression guard for #545 — the static tier must import and tile under the installed cleopatra.

    cleopatra 0.46 removed the free function `cleopatra.basemap.tiles.add_tiles` the static tier draped
    tiles with, so `import digitalearth.static` failed at a fresh install while the shipped `pixi.lock`
    (pinned to the older cleopatra) hid it from CI. These run against whatever cleopatra the environment
    resolves, so a future tile-API move breaks the suite here rather than at a downstream install.
    """

    def test_the_static_tier_imports_under_the_installed_cleopatra(self):
        """`from digitalearth.static import Map` — the exact import #545 reported broken — must succeed.

        Test scenario:
            The break was an `ImportError` raised at module load of `digitalearth.static.builders.decoration`
            (a top-level `from cleopatra.basemap.tiles import add_tiles`), which propagated to `Map`. Importing
            the tier's public entry point here fails loudly if that surface drifts again.
        """
        from digitalearth.static import Map as StaticMap

        assert StaticMap is Map, (
            "the static tier's Map export is not the package-root Map"
        )


class TestAddTilesWrapper:
    """The static tier's `decoration.add_tiles` wrapper over cleopatra 0.46's `GeoMixin.add_tiles` (#545).

    These pin the wrapper's forwarding contract directly, without the network: cleopatra's own
    `_add_tiles` (the single implementation `GeoMixin.add_tiles` delegates to) is replaced with a recorder,
    so what the static tier actually hands cleopatra is observable.
    """

    @staticmethod
    def _recording_tiler(monkeypatch):
        """Replace cleopatra's `_add_tiles` with a recorder and return the captured-call dict.

        Args:
            monkeypatch: pytest's patcher, which restores the real `_add_tiles` at teardown.

        Returns:
            dict: Populated with ``ax``/``source``/``crs``/``kwargs`` when the wrapper calls the tiler.
        """
        from cleopatra.basemap import tiles as cleo_tiles

        captured: dict = {}

        def fake_add_tiles(ax, source=None, *, crs=None, **kwargs):
            captured.update(ax=ax, source=source, crs=crs, kwargs=kwargs)
            return ax

        monkeypatch.setattr(cleo_tiles, "_add_tiles", fake_add_tiles)
        return captured

    def test_forwards_axes_source_crs_and_kwargs_and_returns_the_axes(
        self, monkeypatch
    ):
        """Axes, source, crs and extra keywords reach cleopatra unchanged, and the axes is returned.

        Args:
            monkeypatch: pytest's patcher for the recording tiler.

        Test scenario:
            The wrapper binds a `_TileHost` to the axes and forwards every argument straight through to
            cleopatra's tiler, returning whatever it returns (the axes). The axes is a plain sentinel so the
            identity checks cannot pass by coincidence.
        """
        from digitalearth.static.builders.decoration import add_tiles

        captured = self._recording_tiler(monkeypatch)
        sentinel_ax = object()
        result = add_tiles(sentinel_ax, source="OSM", crs=3857, zoom=5, alpha=0.5)
        assert result is sentinel_ax, f"the wrapper did not return the axes: {result!r}"
        assert captured["ax"] is sentinel_ax, f"axes not forwarded: {captured['ax']!r}"
        assert captured["source"] == "OSM", (
            f"source not forwarded: {captured['source']!r}"
        )
        assert captured["crs"] == 3857, f"crs not forwarded: {captured['crs']!r}"
        assert captured["kwargs"] == {"zoom": 5, "alpha": 0.5}, (
            f"extra keywords not forwarded verbatim: {captured['kwargs']!r}"
        )

    def test_a_crs_object_passes_through_without_being_validated(self, monkeypatch):
        """A pyproj CRS object reaches the tiler untouched — GeoMixin's int/str/None validator is bypassed.

        Args:
            monkeypatch: pytest's patcher for the recording tiler.

        Test scenario:
            The removed free function took `crs` as-is, so a CRS object worked. `GeoMixin.crs`'s setter
            rejects anything but an int/str/None, so routing the display CRS through `self.crs` would break
            that. The wrapper leaves `self.crs` unset and passes `crs=` explicitly, so the exact object
            arrives at cleopatra — this guards that path.
        """
        from pyramids.base.crs import crs_from_user_input

        from digitalearth.static.builders.decoration import add_tiles

        captured = self._recording_tiler(monkeypatch)
        crs_object = crs_from_user_input("+proj=robin")
        add_tiles(object(), source=None, crs=crs_object)
        assert captured["crs"] is crs_object, (
            f"the CRS object was converted or dropped instead of passed through: {captured['crs']!r}"
        )
        assert captured["source"] is None, (
            f"an omitted source should forward as None: {captured['source']!r}"
        )

    def test_a_pyproj_crs_object_renders_end_to_end(self, offline_tiles):
        """A `Map` built with a pyproj CRS object draws a basemap through the real cleopatra tiler (#545 L5).

        Args:
            offline_tiles: Stubs only cleopatra's HTTP fetch, so the real tile path runs offline.

        Test scenario:
            The boundary test proves the wrapper forwards a CRS object unchanged; this proves the whole
            render survives one. `Map(crs=<pyproj CRS>)` frames a Web-Mercator extent over Berlin and draws a
            basemap, so cleopatra's real `_add_tiles` stringifies and reprojects the object (not a stub) and
            leaves exactly one `AxesImage` — the end-to-end path `scene.crs` can take in real use.
        """
        from pyramids.base.crs import crs_from_user_input

        scene = Map(crs=crs_from_user_input(3857))
        scene.ax.set_xlim(1_447_000, 1_525_000)
        scene.ax.set_ylim(6_880_000, 6_950_000)
        scene.basemap("OSM", zoom=8)
        images = [image for image in scene.ax.images if isinstance(image, AxesImage)]
        scene.close()
        assert len(images) == 1, scene.ax.images
