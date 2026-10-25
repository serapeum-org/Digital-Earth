"""The five map features the 3-D tier used to declare absent, now built (#294 follow-up).

`domain`, `basemap`, `scale_bar`, `north_arrow` and `attribution` were entries in
`three_d/capabilities.py`'s `absent` dict — each a reasoned decision rather than a gap. This covers the four
that became **features** (`domain`, `scale_bar`, `north_arrow`, `attribution`) and the one that became a drawn
**kind** (`basemap`): that each is declared where it should be, that the method behaves, and that it draws.

`basemap` fetches web-map tiles over HTTP; every test here replaces the headless seam
`decoration.fetch_basemap_image` with a synthetic image so nothing touches the network.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.base.registry import band_of  # noqa: E402
from digitalearth.base.sources import get_source  # noqa: E402
from digitalearth.three_d import Scene3D  # noqa: E402
from digitalearth.three_d.capabilities import CAPABILITIES  # noqa: E402
from digitalearth.three_d.decoration import decoration as deco  # noqa: E402


def _terrain(scene: Scene3D) -> Scene3D:
    """Add a small lon/lat terrain so the scene has a drawn layer and a display CRS.

    Args:
        scene: The scene to draw into.

    Returns:
        The same scene, for chaining in a test.
    """
    lon = np.linspace(-30.0, 50.0, 8)
    lat = np.linspace(30.0, 75.0, 8)
    z = np.add.outer(np.cos(np.radians(lat)), np.zeros_like(lon))
    scene.terrain(get_source(z, x=lon, y=lat))
    return scene


@pytest.fixture
def scene():
    """Yield an off-screen lon/lat scene, closed on the way out.

    Yields:
        The scene under test.
    """
    built = Scene3D(off_screen=True, crs=4326)
    yield built
    built.close()


class TestTheDeclarationMoved:
    """The five names left `absent`; four are features, one is a kind."""

    def test_none_of_the_five_is_absent_any_more(self):
        """The `absent` dict no longer names any of the five covered capabilities."""
        covered = {"domain", "basemap", "scale_bar", "north_arrow", "attribution"}
        assert covered.isdisjoint(CAPABILITIES.absent), sorted(CAPABILITIES.absent)

    def test_the_four_features_are_declared(self):
        """`domain`, `scale_bar`, `north_arrow` and `attribution` are declared features."""
        for name in ("domain", "scale_bar", "north_arrow", "attribution"):
            assert name in CAPABILITIES.features, (name, sorted(CAPABILITIES.features))

    def test_basemap_is_a_declared_underlay_kind(self):
        """`basemap` is a drawn kind, banded under the data."""
        assert "basemap" in CAPABILITIES.kinds, sorted(CAPABILITIES.kinds)
        assert band_of("basemap") == "underlay"


class TestAttribution:
    """A credit line is recorded on the scene and drawn in a corner."""

    def test_it_records_and_chains(self, scene):
        """The call records the text and returns the scene."""
        assert scene.attribution("© OSM", anchor="bottom-left") is scene
        assert scene._decoration["attribution"]["text"] == "© OSM"
        assert scene._decoration["attribution"]["anchor"] == "bottom-left"

    def test_it_draws_one_named_actor_that_replaces(self, scene):
        """Building the plotter draws the credit under one actor name; a second call replaces it."""
        scene.attribution("first")
        _ = (
            scene.plotter
        )  # building the plotter dresses it, drawing the recorded furniture
        assert deco.ATTRIBUTION_ACTOR in scene.plotter.actors
        scene.attribution("second")
        # One actor name, not two: the credit replaced rather than stacked.
        assert scene._decoration["attribution"]["text"] == "second"

    def test_a_non_string_is_refused(self, scene):
        """A credit that is not a string is refused, as the title is."""
        with pytest.raises(TypeError):
            scene.attribution(42)

    def test_an_unknown_anchor_is_refused(self, scene):
        """Only the four panel corners are accepted."""
        with pytest.raises(ValueError, match="anchor"):
            scene.attribution("x", anchor="middle")


class TestScaleBar:
    """A scale bar is drawn, and forces the orthographic projection it is only honest in."""

    def test_it_turns_on_parallel_projection(self, scene):
        """The bar is true only in parallel projection, so the method switches it on."""
        _terrain(scene)
        assert scene.scale_bar() is scene
        _ = scene.plotter
        assert scene.plotter.camera.parallel_projection is True

    def test_it_draws_one_named_actor(self, scene):
        """The legend-scale ruler is filed under one actor name."""
        _terrain(scene)
        scene.scale_bar()
        _ = scene.plotter
        assert deco.SCALE_BAR_ACTOR in scene.plotter.actors


class TestNorthArrow:
    """A camera-aware north arrow is held on the scene as an orientation widget."""

    def test_it_records_and_chains(self, scene):
        """The call records the anchor and returns the scene."""
        assert scene.north_arrow(anchor="top-left") is scene
        assert scene._decoration["north_arrow"]["anchor"] == "top-left"

    def test_it_holds_a_widget_once_drawn(self, scene):
        """Building the plotter creates the orientation widget and the scene keeps a reference to it."""
        _terrain(scene)
        scene.north_arrow()
        _ = scene.plotter
        assert getattr(scene, "_north_arrow_widget", None) is not None

    def test_an_unknown_anchor_is_refused(self, scene):
        """Only the four panel corners are accepted."""
        with pytest.raises(ValueError, match="anchor"):
            scene.north_arrow(anchor="centre")


class TestDomain:
    """`domain` frames the camera on a named region or a box, reprojecting into the display CRS."""

    def test_a_named_region_frames_the_camera_on_it(self, scene):
        """`domain("europe")` centres the camera's focal point inside Europe's lon/lat box."""
        _terrain(scene)
        assert scene.domain("europe") is scene
        focal = scene.plotter.camera.focal_point
        # Europe is (-25, 34, 45, 72) in EPSG:4326; reset_camera centres on the box, so ~ (10, 53).
        assert -25.0 <= focal[0] <= 45.0, focal
        assert 34.0 <= focal[1] <= 72.0, focal

    def test_an_explicit_box_frames_on_its_centre(self, scene):
        """A numeric box frames on its own centre."""
        _terrain(scene)
        scene.domain([0.0, 40.0, 20.0, 60.0])
        focal = scene.plotter.camera.focal_point
        assert 0.0 <= focal[0] <= 20.0, focal
        assert 40.0 <= focal[1] <= 60.0, focal

    def test_a_box_in_another_crs_is_reprojected_into_the_display_crs(self):
        """A 4326 box reprojects into a projected display CRS, turning degrees into metres."""
        scene = Scene3D(off_screen=True, crs=3857)
        try:
            west, south, east, north = scene._domain_bounds(
                -25.0, 34.0, 45.0, 72.0, 4326
            )
            # Web-Mercator metres, not degrees: Europe's west edge is far below -1e6 m, its east far above.
            assert west < -1_000_000.0, west
            assert east > 1_000_000.0, east
        finally:
            scene.close()

    def test_an_unknown_region_is_refused(self, scene):
        """A name no region is registered under is refused, pointing at the catalogue."""
        with pytest.raises(KeyError, match="not a registered region"):
            scene.domain("atlantis")


class TestBasemap:
    """`basemap` drapes stitched tiles under the scene as an underlay layer (fetch stubbed, no network)."""

    @pytest.fixture
    def stub_tiles(self, monkeypatch):
        """Replace the tile-fetch seam with a synthetic RGBA image and a Web-Mercator extent.

        Args:
            monkeypatch: pytest's patcher, scoped to the test.

        Returns:
            The synthetic extent, so a test can assert the plane spans it.
        """
        extent = (-1_000_000.0, 4_000_000.0, 3_000_000.0, 8_000_000.0)
        image = np.full((4, 4, 4), 128, dtype="uint8")
        monkeypatch.setattr(
            deco, "fetch_basemap_image", lambda bbox, provider, zoom: (image, extent)
        )
        return extent

    def test_it_records_a_basemap_layer_and_draws_it(self, stub_tiles):
        """The call registers a `basemap` layer, draws an actor, and stores the box in EPSG:4326."""
        scene = Scene3D(off_screen=True, crs=3857)
        try:
            actor = scene.basemap(
                "CartoDB.Positron", bbox=(-10.0, 35.0, 30.0, 60.0), zoom=3
            )
            assert actor is not None
            assert "basemap" in scene.layer_ids
            layer = scene.get_layer("basemap")
            assert layer.kind == "basemap"
            props = layer.symbology.props
            assert props["bbox"] == (-10.0, 35.0, 30.0, 60.0)
            assert props["provider"] == "CartoDB.Positron"
        finally:
            scene.close()

    def test_a_box_in_another_crs_is_stored_in_4326(self, stub_tiles):
        """A box given in metres (EPSG:3857) is reprojected to the degrees the tile math needs."""
        scene = Scene3D(off_screen=True, crs=3857)
        try:
            scene.basemap(
                bbox=(-1_000_000.0, 4_000_000.0, 3_000_000.0, 8_000_000.0), crs=3857
            )
            stored = scene.get_layer("basemap").symbology.props["bbox"]
            assert all(-180.0 <= stored[i] <= 180.0 for i in (0, 2)), stored
            assert all(-90.0 <= stored[i] <= 90.0 for i in (1, 3)), stored
        finally:
            scene.close()

    def test_a_bad_box_is_refused(self, stub_tiles):
        """A box that is not four numbers is refused before anything is drawn."""
        scene = Scene3D(off_screen=True)
        try:
            with pytest.raises(ValueError, match="west, south, east, north"):
                scene.basemap(bbox=(1.0, 2.0, 3.0))
        finally:
            scene.close()
