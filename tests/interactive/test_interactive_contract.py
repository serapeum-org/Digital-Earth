"""The Wave-0 cross-tier contract, as the interactive (HoloViz) tier implements it.

One class per contract point (#230, #233, #246–#257): the four backends agreed on one spelling and one
default for every shared concept, and these tests pin this tier's half of each. They are deliberately
signature- and behaviour-level rather than render-level — the point is that ``fps``, ``size``, ``scheme``/
``k``, ``cmap=None``, ``big_data_threshold`` and ``save() -> Path`` mean here exactly what they mean on
static / web / three_d.

Every rename in the batch keeps its old spelling working for one release, so each deprecated alias is
tested three ways: that it still does its job, that it warns while doing it, and that passing it
alongside the new spelling is a ``TypeError`` naming both — the one answer all four tiers give, from
:func:`~digitalearth.base.deprecation.renamed_parameter`.

Runs in the ``interactive`` pixi env (``pixi run -e interactive test-interactive``).
"""

import inspect
import pathlib
import warnings

import numpy as np
import pytest
from loguru import logger

from digitalearth.base.crs import OffLimbError
from digitalearth.base.sources import Source
from digitalearth.base.sources.dimension import DimensionInfo
from digitalearth.interactive import InteractiveMap
from digitalearth.interactive import base as interactive_base
from digitalearth.interactive import decoration as interactive_decoration
from digitalearth.interactive import raster as interactive_raster
from digitalearth.interactive import vector as interactive_vector

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")


@pytest.fixture
def m() -> InteractiveMap:
    """A fresh Web-Mercator map for each test."""
    return InteractiveMap()


@pytest.fixture
def point_fc():
    """The point fixture as a pyramids FeatureCollection (EPSG:32618, numeric 'fid')."""
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


@pytest.fixture
def polygon_fc(point_fc):
    """A polygon FeatureCollection built by buffering the point fixture (same CRS, 'fid' column)."""
    fc = point_fc.copy()
    fc["geometry"] = fc.geometry.buffer(500.0)
    return fc


@pytest.fixture
def off_limb(monkeypatch):
    """Make every reprojection report the data as off-limb.

    Real off-limb data is a property of the warp, which GDAL decides: an orthographic display CRS that
    hides one fixture still produces output for another, so a test built on real geometry pins the
    fixture's coordinates rather than the guard. Patching the tier's single reprojection choke point is
    what actually exercises the skip/raise policy.

    Args:
        monkeypatch: pytest's attribute patcher.
    """

    def _raise(dataset, crs):
        """Stand in for ``digitalearth.base.crs.reproject`` on data the display CRS cannot place.

        Args:
            dataset: The pyramids object being warped.
            crs: The display CRS it is warped into.

        Raises:
            OffLimbError: always — that is the point of the fixture.
        """
        raise OffLimbError(f"the data lies outside what {crs!r} can show")

    # base.display is where the warp now happens: DE-17 (#277) lifted _to_display_source out of the tier
    # base classes, so the tier modules no longer import `reproject` themselves. The two tier modules that
    # still call it directly are patched as well.
    from digitalearth.base import display as base_display

    # No hasattr guard: if a later refactor moves the warp again, a silently-skipped patch would leave these
    # tests passing while simulating nothing. monkeypatch.setattr raises on a missing attribute, which is the
    # signal we want. base_display is where _to_display_source warps; the other two call reproject directly.
    for module in (base_display, interactive_raster, interactive_vector):
        monkeypatch.setattr(module, "reproject", _raise)


def _source(variable: str, values: np.ndarray = None) -> Source:
    """A tiny EPSG:3857 raster ``Source`` carrying a variable name (drives the autostyle lookup).

    Args:
        variable: The variable name ``auto_style`` matches on.
        values: Optional 6x10 value array; defaults to a plain ramp.

    Returns:
        Source: the display-CRS source a raster builder accepts unchanged.
    """
    array = np.arange(60.0).reshape(6, 10) if values is None else values
    z = DimensionInfo(array, "z")
    x = DimensionInfo(np.arange(10.0), "x")
    y = DimensionInfo(np.arange(6.0), "y")
    return Source(z, x, y, crs=3857, metadata={"variable": variable})


class TestSaveReturnsPath:
    """C1 (#248) — every tier's ``save`` hands back the ``pathlib.Path`` it wrote."""

    def test_save_returns_a_path(self, m, tmp_path):
        """``save`` returns a ``Path``, not the string it was handed.

        Test scenario:
            A caller chaining ``save(...).stat()`` or ``.unlink()`` breaks the moment one tier answers
            with a string, which is why the return type is part of the contract rather than a detail.
        """
        out = tmp_path / "m.html"
        written = m.add_element(hv.Points([(0.0, 0.0)])).save(str(out))
        assert isinstance(written, pathlib.Path), (
            f"expected a Path, got {type(written)}"
        )
        assert written == out, f"expected the path it was handed, got {written}"
        assert written.exists(), f"the writer must leave a file behind: {written}"

    def test_save_animation_and_save_app_agree(self, m, dataset, tmp_path):
        """The tier's other two writers hand back the ``Path`` they wrote, so the rule has no exceptions.

        Args:
            m: A fresh Web-Mercator map (the ``save_app`` subject, once it carries a layer).
            dataset: The raster fixture, stacked into a two-step cube for the animation writer.
            tmp_path: pytest's per-test directory.

        Test scenario:
            This used to compare ``inspect.signature(...).return_annotation`` against ``Path``, which an
            implementation returning the ``str`` it was handed passes unchanged — leaving C1 proven
            behaviourally for exactly one of the tier's writers. Both are called for real here; the
            animation goes out as the scrubber HTML, the cheap writer down the same return path as the
            GIF.
        """
        from pyramids.dataset.collection import DatasetCollection

        cube = DatasetCollection.from_files(["examples/data/acc4000.tif"] * 2)
        written = {
            tmp_path / "anim.html": InteractiveMap()
            .timecube(cube)
            .save_animation(str(tmp_path / "anim.html")),
            tmp_path / "app.html": m.image(dataset).save_app(
                str(tmp_path / "app.html"), widgets=("cmap",)
            ),
        }
        for expected, actual in written.items():
            assert isinstance(actual, pathlib.Path), (
                f"{expected.name}: expected a Path, got {type(actual)}"
            )
            assert actual == expected, (
                f"{expected.name}: expected that path back, got {actual}"
            )
            assert actual.exists(), f"{expected.name}: nothing written at {actual}"


class TestFrameRate:
    """C2 (#256) — the animation entry points take ``fps: float`` with the shared default ``3.0``."""

    @pytest.mark.parametrize(
        "method",
        [InteractiveMap.play, InteractiveMap.save_animation],
        ids=lambda f: f.__name__,
    )
    def test_fps_defaults_to_three_point_zero(self, method):
        """``fps`` is a float defaulting to ``3.0`` on both animation entry points.

        Args:
            method: The animation method to inspect.

        Test scenario:
            ``3`` and ``3.0`` compare equal, so an ``int`` default would slip past an ``== 3.0``
            assertion while still differing from every other tier's annotation; the type is checked too.
        """
        parameter = inspect.signature(method).parameters["fps"]
        assert parameter.default == 3.0, parameter.default
        assert isinstance(parameter.default, float), (
            f"{method.__name__} fps default must be a float, got {type(parameter.default)}"
        )
        assert parameter.annotation is float, parameter.annotation


class TestMarkerSize:
    """C3 (#251) — ``size`` means the marker size, and nothing else, anywhere in the tier."""

    def test_points_size_reaches_the_style(self, m, point_fc):
        """``points(size=...)`` is recorded as the marker size the glyph draws with."""
        m.points(point_fc, size=12.0)
        assert m.style_of(0)["common"]["size"] == 12.0, m.style_of(0)

    def test_no_other_builder_takes_a_size_parameter(self):
        """``points`` is the only public method with a parameter spelled exactly ``size``.

        Test scenario:
            The contract's whole point is that ``size`` never means two things. A text- or node-size
            parameter added later under the bare name is the regression this catches; ``gridsize`` /
            ``max_size`` and friends are compound names and are left alone.
        """
        offenders = []
        for name, method in inspect.getmembers(InteractiveMap, inspect.isfunction):
            if name.startswith("_") or name == "points":
                continue
            if "size" in inspect.signature(method).parameters:
                offenders.append(name)
        assert not offenders, (
            f"`size` must mean the marker size only; also found on {offenders} — rename those "
            "(text_size=, node_size=, …) per #251"
        )


class TestClassification:
    """C4 (#246) — ``scheme: str | None = None`` + ``k: int = 5`` wherever a layer is classified."""

    def test_signature_defaults(self):
        """``scheme`` defaults to ``None`` (continuous) and ``k`` to 5, as on every other tier."""
        parameters = inspect.signature(InteractiveMap.choropleth).parameters
        assert parameters["scheme"].default is None, parameters["scheme"].default
        assert parameters["k"].default == 5, parameters["k"].default

    def test_scheme_none_is_a_continuous_ramp(self, m, polygon_fc):
        """``scheme=None`` colours continuously: one colormap name, and no recorded class breaks.

        Test scenario:
            The graduated path sets ``last_breaks`` and hands HoloViews a *list* of per-class colours,
            so both are checked — a continuous layer that left stale breaks behind would build a legend
            for classes it never drew.
        """
        m.choropleth(polygon_fc, "fid", cmap="plasma")
        style = hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs
        assert style["cmap"] == "plasma", style["cmap"]
        assert m.last_breaks is None, (
            f"a continuous ramp has no breaks: {m.last_breaks}"
        )

    def test_k_drives_the_class_count(self, m, polygon_fc):
        """A non-``None`` scheme classifies into exactly ``k`` flat classes."""
        m.choropleth(polygon_fc, "fid", scheme="quantiles", k=3)
        cmap = hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs["cmap"]
        assert isinstance(cmap, list), f"a classified cmap must be a list, got {cmap}"
        assert len(cmap) == 3, f"k=3 must give 3 flat classes, got {cmap}"


class TestAutoCmap:
    """C5 (#249) — ``cmap=None`` on a raster builder resolves through ``auto_style``."""

    @pytest.mark.parametrize(
        "builder", ["image", "quadmesh", "large_image", "timecube"]
    )
    def test_no_raster_builder_hard_codes_a_colormap(self, builder):
        """Every raster builder's ``cmap`` default is ``None``, never a bare string.

        Args:
            builder: The raster builder to inspect.

        Test scenario:
            A literal default silently wins over the variable's canonical colormap, so the same field
            would be coloured differently depending on which builder drew it.
        """
        default = (
            inspect.signature(getattr(InteractiveMap, builder))
            .parameters["cmap"]
            .default
        )
        assert default is None, (
            f"{builder}(cmap=...) must default to None, got {default!r}"
        )

    def test_image_resolves_the_variables_colormap(self, m):
        """A recognised variable picks up its canonical colormap with no ``cmap`` given."""
        from digitalearth.base.autostyle import auto_style

        src = _source("t2m")
        m.image(src)
        assert m.style_of(0)["common"]["cmap"] == auto_style(src)["cmap"]

    def test_the_callers_colormap_always_wins(self, m):
        """An explicit ``cmap`` is never overridden by the lookup."""
        m.image(_source("t2m"), cmap="magma")
        assert m.style_of(0)["common"]["cmap"] == "magma"

    def test_large_image_resolves_from_the_band_name(self, m):
        """The windowed reader resolves from the band's *name*, without reading the band.

        Test scenario:
            ``large_image`` exists so a multi-GB raster is never materialised; feeding the autostyle
            lookup a real ``Source`` would have read the whole array just to pick a colour.
        """
        from digitalearth.base.autostyle import auto_style

        dataset = type("Ds", (), {"band_names": ["t2m", "other"]})()
        resolved = m._auto_cmap_for_band(dataset, 1, None)
        assert resolved == auto_style(_source("t2m"))["cmap"], resolved
        assert m._auto_cmap_for_band(dataset, 1, "magma") == "magma"
        assert m._auto_cmap_for_band(dataset, 9, None) == "viridis", (
            "an unnamed band falls back to the tier's previous literal"
        )


class TestAutoLevelsAndUnits:
    """C6 (#230) — ``auto_style``'s ``levels`` and ``units`` are consumed, not just its ``cmap``."""

    def test_units_label_the_colorbar(self, m):
        """A recognised variable's ``units`` become the colorbar label when the caller gave none."""
        m.image(_source("msl"))
        assert m.style_of(0)["common"]["clabel"] == "hPa", m.style_of(0)

    def test_a_caller_supplied_label_wins(self, m):
        """``clabel=`` is never overridden by the style library."""
        m.image(_source("msl"), clabel="millibar")
        assert m.style_of(0)["common"]["clabel"] == "millibar"

    def test_an_unknown_variable_is_left_unlabelled(self, m):
        """Units are never guessed: an unrecognised field labels exactly as before (not at all)."""
        m.image(_source("totally-unknown-field"))
        assert "clabel" not in m.style_of(0)["common"], m.style_of(0)

    def test_levels_come_from_the_style_library(self, m):
        """``contours`` with no ``levels`` uses the variable's canonical contour levels.

        Test scenario:
            The values are put in the field's real range (hPa around 1000) so the canonical levels
            actually cut the data; the drawn contour values must all come from that canonical list.
        """
        from digitalearth.base.autostyle import auto_style

        pressures = np.linspace(960.0, 1050.0, 60).reshape(6, 10)
        src = _source("msl", pressures)
        expected = auto_style(src)["levels"]
        m.contours(src)
        drawn = {
            float(value)
            for value in m.layers[0].dimension_values("msl", expanded=False)
            if np.isfinite(value)
        }
        assert drawn, "no contours were drawn at all"
        assert drawn <= {float(level) for level in expected}, (
            f"contour levels {sorted(drawn)} are not the canonical {expected}"
        )

    def test_caller_levels_still_win(self, m):
        """An explicit ``levels`` count is used even for a variable the library knows."""
        pressures = np.linspace(960.0, 1050.0, 60).reshape(6, 10)
        m.contours(_source("msl", pressures), levels=3)
        drawn = {
            float(value)
            for value in m.layers[0].dimension_values("msl", expanded=False)
            if np.isfinite(value)
        }
        assert len(drawn) <= 3, (
            f"expected at most 3 contour levels, got {sorted(drawn)}"
        )


class TestOffLimbIsSkipped:
    """C7 (#257) — data the display CRS cannot place skips the layer and warns; ``strict`` raises."""

    def test_the_layer_is_skipped_and_the_map_still_chains(self, dataset, off_limb):
        """A skipped layer registers nothing and still returns the map.

        Args:
            dataset: The raster fixture.
            off_limb: Makes the reprojection report the data as off-limb.
        """
        m = InteractiveMap()
        assert m.image(dataset) is m, "a skipped builder must still chain"
        assert m.layers == [], f"nothing may be registered: {m.layers}"

    def test_the_skip_is_warned_about_and_names_the_layer(self, dataset, off_limb):
        """The warning names the builder that drew nothing — a silent skip is a blank map.

        Args:
            dataset: The raster fixture.
            off_limb: Makes the reprojection report the data as off-limb.
        """
        messages = []
        handle = logger.add(messages.append, level="WARNING")
        try:
            InteractiveMap().image(dataset)
        finally:
            logger.remove(handle)
        assert any("image" in text for text in messages), messages
        assert any("skipped" in text for text in messages), messages

    def test_strict_raises_instead(self, dataset, off_limb):
        """``strict=True`` re-raises the ``OffLimbError`` rather than skipping.

        Args:
            dataset: The raster fixture.
            off_limb: Makes the reprojection report the data as off-limb.
        """
        strict_map = InteractiveMap(strict=True)
        with pytest.raises(OffLimbError):
            strict_map.image(dataset)

    def test_a_vector_builder_is_guarded_too(self, point_fc, off_limb):
        """The policy is the tier's, not one builder's — a vector layer skips the same way.

        Args:
            point_fc: The point fixture.
            off_limb: Makes the reprojection report the data as off-limb.
        """
        m = InteractiveMap()
        assert m.points(point_fc) is m
        assert m.layers == []

    @pytest.mark.parametrize(
        "builder",
        [
            "image",
            "rgb",
            "quadmesh",
            "contours",
            "filled_contours",
            "large_image",
            "points",
            "path",
            "polygons",
            "choropleth",
            "trimesh",
            "hexbin",
            "kde",
            "graph",
            "timecube",
            "labels",
        ],
    )
    def test_every_reprojecting_builder_carries_the_guard(self, builder):
        """Each builder that reprojects is wrapped by ``_skips_off_limb``.

        Args:
            builder: The builder to inspect.

        Test scenario:
            The guard is a decorator, so a new builder that forgets it fails only for the user whose
            data happens to fall off the limb. ``functools.wraps`` leaves ``__wrapped__`` behind, which
            is what this looks for.
        """
        method = getattr(InteractiveMap, builder)
        assert getattr(method, "__wrapped__", None) is not None, (
            f"{builder}() reprojects but is not decorated with @_skips_off_limb (#257)"
        )


class TestBigDataThreshold:
    """C8 (#250) — one ``big_data_threshold``: an attribute, a per-call override, one deprecated alias."""

    def test_default_and_constructor(self):
        """The cutoff is 50 000 by default and settable at construction."""
        assert InteractiveMap().big_data_threshold == 50_000
        assert InteractiveMap(big_data_threshold=7).big_data_threshold == 7

    def test_setting_the_attribute_affects_later_layers(self, m, point_fc):
        """Setting it once governs every subsequent layer — the point of making it an attribute.

        Test scenario:
            The fixture has a handful of rows, so a cutoff of 1 forces the Datashader route and a
            ``DynamicMap`` layer; the same call at the default cutoff stays raw glyphs.
        """
        m.big_data_threshold = 1
        m.points(point_fc)
        assert isinstance(m.layers[0], hv.DynamicMap), type(m.layers[0])
        assert isinstance(InteractiveMap().points(point_fc).layers[0], gv.Points)

    def test_the_per_call_override_wins(self, m, point_fc):
        """A per-call ``big_data_threshold`` overrides the map's attribute for that layer only."""
        m.big_data_threshold = 1
        m.points(point_fc, big_data_threshold=10_000)
        assert isinstance(m.layers[0], gv.Points), type(m.layers[0])
        assert m.big_data_threshold == 1, (
            "a per-call override must not rewrite the attribute"
        )

    @pytest.mark.parametrize("builder", ["points", "polygons", "trimesh"])
    def test_every_big_data_builder_takes_both_names(self, builder):
        """All three call sites carry the new name plus the deprecated alias.

        Args:
            builder: The builder to inspect.
        """
        parameters = inspect.signature(getattr(InteractiveMap, builder)).parameters
        assert parameters["big_data_threshold"].default is None, builder
        assert parameters["rasterize_threshold"].default is None, builder


class TestDeprecatedAliases:
    """Every rename keeps the old spelling working for one release — and says so while it does."""

    def test_rasterize_threshold_still_works(self, m, point_fc):
        """The deprecated alias routes exactly as ``big_data_threshold`` would."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            m.points(point_fc, rasterize_threshold=1)
        assert isinstance(m.layers[0], hv.DynamicMap), type(m.layers[0])

    @pytest.mark.parametrize("builder", ["points", "polygons", "trimesh"])
    def test_rasterize_threshold_warns(self, builder, point_fc, polygon_fc):
        """Using it emits a ``DeprecationWarning`` naming the replacement.

        Args:
            builder: The builder to call with the deprecated alias.
            point_fc: The point fixture (``points``/``trimesh``).
            polygon_fc: The polygon fixture (``polygons``).
        """
        data = polygon_fc if builder == "polygons" else point_fc
        build = getattr(InteractiveMap(), builder)
        with pytest.warns(DeprecationWarning, match="big_data_threshold"):
            build(data, rasterize_threshold=10_000)

    def test_the_new_name_does_not_warn(self, m, point_fc):
        """Passing only the new name is silent — otherwise the warning trains users to ignore it."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            m.points(point_fc, big_data_threshold=10_000)

    @pytest.mark.parametrize("builder", ["points", "polygons", "trimesh"])
    def test_both_spellings_at_once_is_refused(self, builder, point_fc, polygon_fc):
        """Both spellings of the cutoff at once is a ``TypeError`` naming both, on every builder.

        Args:
            builder: The builder called with the contradictory pair.
            point_fc: The point fixture (``points``/``trimesh``).
            polygon_fc: The polygon fixture (``polygons``).

        Test scenario:
            This tier used to resolve the pair silently in favour of the new name, which meant a caller who
            set both got one of their two numbers honoured and no hint that the other was dropped. All four
            tiers now refuse it through the shared
            :func:`~digitalearth.base.deprecation.renamed_parameter`, and the message names both spellings
            plus the one to keep.
        """
        data = polygon_fc if builder == "polygons" else point_fc
        build = getattr(InteractiveMap(), builder)
        with pytest.raises(TypeError) as excinfo:
            build(data, big_data_threshold=10_000, rasterize_threshold=1)
        message = str(excinfo.value)
        assert "both big_data_threshold= and the deprecated rasterize_threshold=" in (
            message
        ), message
        assert "pass only big_data_threshold=" in message, message


class TestBasemapDefault:
    """C9 (#247) — the default provider is read from ``base/basemaps.py``, not spelled here."""

    def test_tiles_defaults_to_the_shared_constant(self):
        """``tiles()``'s default *is* the shared constant object, not a copy of its value."""
        default = inspect.signature(InteractiveMap.tiles).parameters["provider"].default
        assert default == interactive_decoration.DEFAULT_BASEMAP_PROVIDER, default

    def test_the_constant_comes_from_base(self):
        """The tier reads ``base.basemaps.DEFAULT_BASEMAP_PROVIDER``, which must exist to be read.

        Test scenario:
            This skipped when ``base/`` did not declare the constant, which was defensive while it was
            landing and is a hole now that it has: deleting the constant would turn the guard green by
            skipping rather than red. A missing constant is the bug, so it is asserted, not skipped.
        """
        from digitalearth.base import basemaps

        shared = getattr(basemaps, "DEFAULT_BASEMAP_PROVIDER", None)
        assert shared is not None, (
            "base/basemaps.py must declare DEFAULT_BASEMAP_PROVIDER — it is the one cross-tier default "
            "(#247) this tier imports"
        )
        assert interactive_decoration.DEFAULT_BASEMAP_PROVIDER is shared

    def test_the_dashboard_offers_the_default_first(self):
        """The basemap switcher's first (=selected) choice is the same shared default."""
        from digitalearth.interactive.dashboard import _BASEMAP_CHOICES

        assert _BASEMAP_CHOICES[0] == interactive_decoration.DEFAULT_BASEMAP_PROVIDER
        assert len(_BASEMAP_CHOICES) == len(set(_BASEMAP_CHOICES)), _BASEMAP_CHOICES

    def test_default_tiles_still_render(self, m):
        """Calling ``tiles()`` bare still resolves a real provider."""
        assert m.tiles().layers, "the default provider resolved to nothing"


class TestKeyedBasemapCoverage:
    """C10 (#233) — a keyed provider's ``bounds`` are declared to the engine."""

    @pytest.fixture(autouse=True)
    def _fake_key(self, monkeypatch):
        """Give the keyed preset a credential that is not a real key.

        Args:
            monkeypatch: pytest's environment patcher.
        """
        monkeypatch.setenv("PLANET_API_KEY", "FAKE-KEY-NOT-REAL")

    def test_partial_coverage_becomes_the_pannable_range(self, m):
        """NICFI's tropics band reaches Bokeh as the plot's range bounds, in display-CRS metres.

        Test scenario:
            Bokeh's tile source has no ``bounds`` slot (MapLibre's does), so the declaration has to land
            on the ranges. ±30° latitude is ≈ ±3.5e6 m in Web Mercator, which is what pins that the
            lon/lat box was reprojected rather than passed through as degrees.
        """
        m.tiles("Planet.NICFI", preset={"date": "2024-01"})
        figure = hv.render(m.layers[0])
        assert figure.x_range.bounds is not None, "no coverage was declared"
        assert figure.x_range.bounds[1] == pytest.approx(20_037_508.34, rel=1e-6)
        assert figure.y_range.bounds[1] == pytest.approx(3_503_549.84, rel=1e-6)
        assert figure.y_range.bounds[0] == pytest.approx(-3_503_549.84, rel=1e-6)

    def test_the_coverage_limit_is_logged(self, m):
        """The limit is never silent: a declared coverage says so, because it can frame the map.

        Test scenario:
            Bokeh clamps a view to a range's bounds, so data outside a keyed provider's coverage would
            be framed by the limit instead of by the data — visible only as a map that will not pan.
        """
        messages = []
        handle = logger.add(messages.append, level="INFO")
        try:
            m.tiles("Planet.NICFI", preset={"date": "2024-01"})
        finally:
            logger.remove(handle)
        assert any("pannable range" in text for text in messages), messages

    def test_a_global_provider_declares_nothing(self, m):
        """A provider with no declared coverage leaves the ranges free — no invented limits."""
        m.tiles("CartoLight")
        figure = hv.render(m.layers[0])
        assert figure.x_range.bounds is None, figure.x_range.bounds
        assert figure.y_range.bounds is None, figure.y_range.bounds

    def test_a_keyed_provider_with_no_coverage_declares_nothing(self, m, monkeypatch):
        """A keyed service that covers the world leaves the ranges free — no invented limit.

        Args:
            m: The map under test.
            monkeypatch: pytest's patcher, registering a global keyed preset beside the tropics one.

        Test scenario:
            ``bounds`` is optional on a keyed source: a worldwide service simply declares none. Turning
            that absence into a reprojected box would clamp the map to nothing, so the coverage hook is
            skipped entirely rather than fed an empty extent.
        """
        from digitalearth.base.basemaps import KEYED_BASEMAPS, KeyedTileSource

        def _global_service() -> KeyedTileSource:
            """A keyed service covering the whole world, and therefore declaring no bounds."""
            return KeyedTileSource(
                name="Example.Global",
                url_template="https://tiles.example/{z}/{x}/{y}.png?api_key={api_key}",
                attribution="© Example",
                credential_env="PLANET_API_KEY",
            )

        monkeypatch.setitem(KEYED_BASEMAPS, "example.global", _global_service)
        m.tiles("Example.Global")
        figure = hv.render(m.layers[0])
        assert figure.x_range.bounds is None, (
            f"a global service must not limit panning, got {figure.x_range.bounds}"
        )
        assert figure.y_range.bounds is None, (
            f"a global service must not limit panning, got {figure.y_range.bounds}"
        )

    def test_the_map_is_not_refused_for_opening_outside_coverage(self, m, dataset):
        """Coverage is declared, never validated: a pannable map outside the box still builds.

        Args:
            m: The map under test.
            dataset: A raster fixture well outside the tropics band.
        """
        m.image(dataset).tiles("Planet.NICFI", preset={"date": "2024-01"})
        assert len(m.layers) == 2, m.layers


class TestNamedNaturalEarthFeatures:
    """C12 (#253) — the six named layers, delegating to the ``features(...)`` flags that stay."""

    @pytest.mark.parametrize(
        "name, flag",
        [
            ("borders", "borders"),
            ("land", "land"),
            ("ocean", "ocean"),
            ("lakes", "lakes"),
            ("rivers", "rivers"),
        ],
    )
    def test_each_named_method_delegates_to_its_flag(self, m, name, flag, monkeypatch):
        """The named method is a thin alias of the matching ``features`` flag.

        Args:
            m: The map under test.
            name: The named method.
            flag: The ``features`` keyword it must set.
            monkeypatch: pytest's attribute patcher.
        """
        seen = {}

        def _record(**kwargs):
            """Record the flags ``features`` was called with.

            Args:
                **kwargs: The forwarded feature flags.

            Returns:
                The map, as the real ``features`` does.
            """
            seen.update(kwargs)
            return m

        monkeypatch.setattr(m, "features", _record)
        assert getattr(m, name)(resolution="50m") is m
        assert seen == {flag: True, "resolution": "50m"}, seen

    def test_coastlines_is_one_of_the_six_and_still_draws(self, m):
        """``coastlines`` completes the six and keeps its own element (no Natural-Earth flag for it)."""
        assert m.coastlines().layers, "coastlines drew nothing"

    @pytest.mark.parametrize(
        "name", ["coastlines", "borders", "land", "ocean", "lakes", "rivers"]
    )
    def test_each_named_layer_registers_an_element(self, name):
        """Every named layer registers exactly one element on a fresh map.

        Args:
            name: The named layer to call.
        """
        assert len(getattr(InteractiveMap(), name)().layers) == 1

    def test_features_still_takes_several_flags_at_once(self, m):
        """``features()`` keeps working — the named methods are an addition, not a replacement."""
        m.features(land=True, borders=True, rivers=True)
        assert len(m.layers) == 3, m.layers
