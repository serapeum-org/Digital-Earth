"""Tests for the D3.5 textured-globe renderer (digitalearth.three_d.GlobeMixin).

The globe path is gated on the optional, lazily-imported ``geovista``: the render tests skip when geovista is
absent, but the lazy-import **error** path is always tested (it must raise a clear, actionable message).
"""

import builtins

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

pv = pytest.importorskip("pyvista")

from digitalearth.base.sources import get_source
from digitalearth.three_d import Scene3D
from digitalearth.three_d import globe as globe_mod


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def _global_field():
    """A cos(latitude) field on a 1-D lon/lat grid (a smooth global field to drape)."""
    lon = np.linspace(-180.0, 180.0, 37)
    lat = np.linspace(-90.0, 90.0, 19)
    field = np.add.outer(np.cos(np.deg2rad(lat)), np.zeros(len(lon)))
    return get_source(field, x=lon, y=lat)


def test_globe_missing_geovista_raises_actionable_error(monkeypatch):
    """When geovista is not importable, globe() raises ImportError naming the install command."""
    real_import = builtins.__import__

    def _no_geovista(name, *args, **kwargs):
        if name == "geovista" or name.startswith("geovista."):
            raise ImportError("No module named 'geovista'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_geovista)
    scene = Scene3D(off_screen=True)
    with pytest.raises(ImportError, match=r"digitalearth\[3d\]"):
        scene.globe(_global_field(), coastlines=False)
    scene.close()


def _utm_raster() -> Dataset:
    """Build an ordinary projected (UTM 33N) raster — the input globe() used to refuse.

    Returns:
        A pyramids ``Dataset`` in EPSG:32633.
    """
    return Dataset.from_array(
        np.ones((20, 20), "float32"),
        geo_ref=GeoReference(
            geo=(400000.0, 100.0, 0.0, 5700000.0, 0.0, -100.0), epsg=32633
        ),
    )


def _projected_source_near_the_origin(crs):
    """A Web-Mercator field whose metres are small enough to pass for lon/lat, in a given CRS spelling.

    Args:
        crs: How the caller spelled the source's CRS (an EPSG int, ``"EPSG:<code>"``, proj4, ...).

    Returns:
        A :class:`~digitalearth.base.sources.Source` in that CRS whose coordinates the magnitude fallback
        cannot tell from degrees — so only reading the CRS can refuse it.
    """
    return get_source(
        np.zeros((5, 7)),
        x=np.linspace(-300.0, 300.0, 7),
        y=np.linspace(-80.0, 80.0, 5),
        crs=crs,
    )


class TestGeographicSource:
    """Tests for GlobeMixin._to_geographic_source — the globe's pyramids reprojection choke point."""

    def test_a_projected_dataset_is_reprojected_through_pyramids(self):
        """A UTM Dataset is warped to EPSG:4326 instead of being rejected.

        Test scenario:
            Regression guard: globe() used to raise "the coordinates look projected" for any ordinary
            projected raster, telling the caller to run the very `Dataset.to_crs(4326)` the other three tiers
            run for them. The extracted source must now be geographic, with lon/lat-range coordinates.
        """
        scene = Scene3D(off_screen=True)
        try:
            source = scene._to_geographic_source(_utm_raster())
        finally:
            scene.close()
        assert source.crs == 4326, (
            "The source must come back in the globe's geographic CRS"
        )
        assert float(np.nanmax(np.abs(source.x.values))) <= 360.0, (
            "x must be longitudes after the warp"
        )
        assert float(np.nanmax(np.abs(source.y.values))) <= 90.0, (
            "y must be latitudes after the warp"
        )

    def test_an_already_geographic_dataset_is_not_warped(self):
        """Geographic input passes through untouched — no needless resampling.

        Test scenario:
            Reprojection is asked for only when pyramids reports the CRS as projected, so an EPSG:4326 raster
            must reach extraction as the very object it was handed.
        """
        dataset = Dataset.from_array(
            np.ones((6, 6), "float32"),
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0), epsg=4326),
        )
        calls = []
        original = dataset.to_crs

        def spy(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        dataset.to_crs = spy
        scene = Scene3D(off_screen=True)
        try:
            source = scene._to_geographic_source(dataset)
        finally:
            scene.close()
        assert calls == [], "An already-geographic dataset must not be reprojected"
        assert source.crs == 4326, "It is still extracted as a geographic source"

    def test_a_source_in_a_projected_crs_is_named_by_its_epsg(self):
        """A bare Source cannot be reprojected, so it is refused — by its CRS, not by coordinate magnitude.

        Test scenario:
            `get_source(utm_dataset).crs` is 32633, which is decisive evidence; the error must quote that code
            rather than inferring "projected" from how large the numbers are.
        """
        scene = Scene3D(off_screen=True)
        try:
            with pytest.raises(ValueError, match="EPSG:32633"):
                scene._to_geographic_source(get_source(_utm_raster()))
        finally:
            scene.close()

    @pytest.mark.parametrize(
        "crs, named",
        [
            (3857, "EPSG:3857"),
            ("EPSG:3857", "EPSG:3857"),
            ("epsg:3857", "EPSG:3857"),
            ("+proj=ortho +lat_0=53 +lon_0=4", "proj=ortho"),
        ],
    )
    def test_a_projected_source_is_refused_however_its_crs_is_spelled(self, crs, named):
        """Every spelling the ``Source.crs`` contract blesses is read, not just the bare EPSG int.

        Test scenario:
            The reported defect: the CRS reader ran ``int(crs)``, so ``"EPSG:3857"`` — a spelling the
            contract explicitly allows — resolved to "unknown" and fell through to the coordinate-magnitude
            guard, which accepts Web-Mercator metres near the origin as degrees. The source was then draped
            on the sphere as if it were lon/lat.
        """
        scene = Scene3D(off_screen=True)
        try:
            with pytest.raises(ValueError, match="projected") as raised:
                scene._to_geographic_source(_projected_source_near_the_origin(crs))
        finally:
            scene.close()
        assert named in str(raised.value), (
            f"the refusal should name the CRS as {named!r}, got {raised.value}"
        )

    def test_the_warp_names_the_crs_it_warped_to(self, monkeypatch):
        """The globe passes ``crs=`` to ``get_source`` after warping, instead of re-deriving it.

        Test scenario:
            ``crs=`` exists so a caller that has already placed the data says where it put it (#235). This
            is the globe's end of that contract: after ``Dataset.to_crs(4326)`` the extraction is told the
            display CRS, so the source is geographic by construction rather than by a second lookup that a
            code-less projection would answer with ``None``.
        """
        seen = {}
        original = globe_mod.get_source

        def spy(data, **kwargs):
            seen.update(kwargs)
            return original(data, **kwargs)

        monkeypatch.setattr(globe_mod, "get_source", spy)
        scene = Scene3D(off_screen=True)
        try:
            source = scene._to_geographic_source(_utm_raster())
        finally:
            scene.close()
        assert seen.get("crs") == globe_mod.GEOGRAPHIC_EPSG, (
            f"the warped extraction must be told the CRS it was warped to, got {seen.get('crs')!r}"
        )
        assert source.crs == globe_mod.GEOGRAPHIC_EPSG, (
            f"and the source must report it, got {source.crs!r}"
        )

    def test_an_unwarped_input_is_not_told_a_crs_it_is_not_in(self, monkeypatch):
        """Input that was never warped keeps deriving its own CRS — the globe must not assert one.

        Test scenario:
            The complement of the wiring above: claiming EPSG:4326 for a raw array that declares no CRS
            would silence the coordinate-range guard, the only check such input has.
        """
        seen = {}
        original = globe_mod.get_source

        def spy(data, **kwargs):
            seen.update(kwargs)
            return original(data, **kwargs)

        monkeypatch.setattr(globe_mod, "get_source", spy)
        scene = Scene3D(off_screen=True)
        try:
            source = scene._to_geographic_source(np.zeros((4, 4)))
        finally:
            scene.close()
        assert seen.get("crs") is None, (
            f"an unwarped input must not be handed a CRS, got {seen.get('crs')!r}"
        )
        assert source.crs is None, "so its CRS stays genuinely unknown"

    def test_a_crs_less_projected_source_still_raises(self):
        """With no CRS to reproject from, the coordinate-range guard is still the last resort.

        Test scenario:
            A raw numpy source carries no CRS, so nothing can be warped; metre-scale coordinates must raise
            the actionable "reproject in pyramids first" error rather than silently mis-placing the field.
        """
        source = get_source(
            np.zeros((5, 5)), x=np.linspace(0, 5e5, 5), y=np.linspace(0, 5e5, 5)
        )
        scene = Scene3D(off_screen=True)
        try:
            with pytest.raises(ValueError, match="to_crs"):
                scene._to_geographic_source(source)
        finally:
            scene.close()


# --- real-render tests below need geovista installed ---
geovista = pytest.importorskip("geovista")


def test_globe_renders_field_sphere():
    """globe() drapes the field on a sphere and registers a layer (no coastlines = offline-safe)."""
    scene = Scene3D(off_screen=True)
    actor = scene.globe(_global_field(), coastlines=False)
    assert actor is not None and len(scene.layers) == 1
    assert bool(scene.screenshot().any())
    scene.close()


def test_globe_with_coastlines_adds_a_second_layer():
    """With coastlines=True the globe adds the field sphere plus a coastline overlay (two layers)."""
    scene = Scene3D(off_screen=True)
    scene.globe(_global_field(), coastlines=True, coastline_resolution="110m")
    assert len(scene.layers) == 2
    scene.close()


def test_globe_accepts_a_source_directly():
    """globe() takes an already-built Source without re-sourcing it."""
    scene = Scene3D(off_screen=True)
    scene.globe(_global_field(), coastlines=False, cmap="cividis")
    assert globe_mod._GEOVISTA_DATA in scene.layers[0][0].point_data
    scene.close()


def test_globe_rejects_a_crs_less_projected_source():
    """globe() raises a clear error for input it cannot reproject: a Source with no CRS and projected coords.

    Projected *datasets* are reprojected through pyramids now (see ``TestGeographicSource``); this covers what
    is left — a bare ``Source`` carrying no CRS at all, where the coordinate range is the only evidence and
    the caller really must reproject upstream.
    """
    proj = get_source(
        np.zeros((5, 5)), x=np.linspace(0, 5e5, 5), y=np.linspace(0, 5e5, 5)
    )
    scene = Scene3D(off_screen=True)
    with pytest.raises(ValueError, match="to_crs"):
        scene.globe(proj, coastlines=False)
    scene.close()


def test_globe_refuses_a_projected_source_spelled_with_its_authority_prefix():
    """globe() refuses a projected ``Source`` spelled ``"EPSG:<code>"`` instead of draping it on the sphere.

    The end-to-end reproduction of the defect: a Web-Mercator source whose coordinates are small enough to
    look like degrees was accepted and rendered, because the ``"EPSG:3857"`` spelling read as "no CRS".
    """
    scene = Scene3D(off_screen=True)
    try:
        with pytest.raises(ValueError, match="EPSG:3857"):
            scene.globe(
                _projected_source_near_the_origin("EPSG:3857"), coastlines=False
            )
        assert len(scene.layers) == 0, "nothing may be drawn for a refused source"
    finally:
        scene.close()


def test_globe_renders_a_projected_dataset():
    """A projected Dataset renders on the globe, reprojected to EPSG:4326 on the way in.

    The end-to-end version of ``TestGeographicSource``: the whole call must succeed and register a layer where
    it used to raise.
    """
    scene = Scene3D(off_screen=True)
    actor = scene.globe(_utm_raster(), coastlines=False)
    assert actor is not None and len(scene.layers) == 1
    scene.close()


def test_globe_resolves_an_unspecified_cmap_through_auto_style():
    """globe() colours a named variable the way the other three tiers do, instead of always `viridis`.

    Regression guard for the cross-tier divergence: the colormap came from a hard-coded ``cmap="viridis"``
    default that ignored the ``Source`` the method had just built.
    """
    from digitalearth.base.autostyle import auto_style

    dataset = Dataset.from_array(
        np.ones((19, 37), "float32"),
        geo_ref=GeoReference(geo=(-180.0, 10.0, 0.0, 90.0, 0.0, -10.0), epsg=4326),
    )
    dataset.band_names = ["t2m"]
    captured = {}
    scene = Scene3D(off_screen=True)
    original = scene.add_mesh

    def recorder(mesh, **kwargs):
        captured.update(kwargs)
        return original(mesh, **kwargs)

    scene.add_mesh = recorder
    try:
        scene.globe(dataset, coastlines=False)
    finally:
        scene.close()
    assert captured["cmap"] == auto_style(get_source(dataset))["cmap"] == "coolwarm"
