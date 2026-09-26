"""Colour composites on the web tier (#190, #266).

The encoder is a pure helper, tested without the engine (numpy/matplotlib are core deps via cleopatra);
the builders themselves ``importorskip`` maplibre and run on the shared ``dataset`` fixture. The stretch
is asserted to be the *same* one the static tier applies, which is the point of sharing
``digitalearth.base.stretch``.

``hsv_composite`` (#266) is the second composite, and what it draws is checked against a composite whose
answer can be **written down**: hues 0, 1/3 and 2/3 at full saturation and value are pure red, green and
blue, whatever code produces them. The same fixture is then drawn on the static tier and the two images
compared, because "the tiers agree" is a claim about output, not about both calling the same helper.
"""

import base64
import inspect
import io
import json

import numpy as np
import pytest
from matplotlib import image as mpimage

from digitalearth.base.spec import DEFAULT_BAND, LayerSpec, Symbology
from digitalearth.base.stretch import DEFAULT_COMPOSITE_BANDS
from digitalearth.web import WebMap
from digitalearth.web import raster as web_raster

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: Stretch limits that pass the band values through unchanged, one pair per channel. The 2-98 percentile
#: default would rewrite the hue fixture's values into something no reader could check by hand.
_UNIT_LIMITS = [(0.0, 1.0)] * 3

#: What the hue fixture *is*, in RGB: hue 0, 1/3 and 2/3 at full saturation and value are pure red, green
#: and blue. Written out rather than derived, so the checks below cannot agree with the code by construction.
_PURE_HUES = np.array([[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]] * 2)

#: The three bands every composite check reads, one per channel so a swapped channel shows.
_BANDS = (1, 2, 3)

#: 8-bit tolerance: the tier encodes a PNG, so every comparison against float channels allows one step.
_ONE_LEVEL = 1 / 255


def _hue_dataset(hue):
    """Return a 2 x 3 lon/lat raster whose bands are hue, saturation and value.

    Already EPSG:4326, so the display warp leaves the grid alone and the encoded image is the cells as
    given — the comparison is then about colour, with no resampling in it.

    Args:
        hue: The hue band, as a ``(2, 3)`` array.

    Returns:
        A three-band pyramids ``Dataset`` with saturation and value at 1.0 and ``-9999`` as NoData.
    """
    from pyramids.dataset import Dataset, GeoReference

    ones = np.ones((2, 3), dtype="float32")
    return Dataset.from_array(
        arr=np.stack([np.asarray(hue, dtype="float32"), ones, ones]),
        geo_ref=GeoReference(geo=(4.0, 0.5, 0.0, 52.0, 0.0, -0.5), epsg=4326),
        no_data_value=-9999.0,
    )


@pytest.fixture
def hue_wheel():
    """A raster whose HSV composite is pure red, green and blue, hand-derivable cell by cell."""
    return _hue_dataset([[0.0, 1.0 / 3.0, 2.0 / 3.0]] * 2)


@pytest.fixture
def hue_wheel_with_a_gap():
    """The same raster with its first cell's **hue** missing, which is the channel a NaN ruins quietly."""
    return _hue_dataset([[-9999.0, 1.0 / 3.0, 2.0 / 3.0], [0.0, 1.0 / 3.0, 2.0 / 3.0]])


def _decode(uri):
    """Decode a PNG data-URI into an RGBA array.

    Args:
        uri: A ``data:image/png;base64,…`` string.

    Returns:
        The decoded image as a float array in ``[0, 1]``.
    """
    assert uri.startswith("data:image/png;base64,"), uri[:40]
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert raw[:8] == _PNG_MAGIC, "decoded payload must be a real PNG"
    return mpimage.imread(io.BytesIO(raw))


class TestTheBandDefaultsAreTheSharedOnes:
    """#266 — a builder called without bands means what ``base/`` says it means, not its own literal.

    The defaults are read off the signature rather than from a call: a caller who names no bands gets
    exactly the object written there, and a literal that happens to equal the shared constant today is
    precisely what lets the two drift tomorrow. That drift is the finding.
    """

    def test_the_composite_default_is_the_shared_tuple(self):
        """``rgb_composite``'s ``bands`` default *is* :data:`DEFAULT_COMPOSITE_BANDS`.

        Test scenario:
            ``is`` rather than ``==``: the tier wrote ``(1, 2, 3)`` inline, which compares equal to the
            shared tuple while being a different object, so only identity tells the shared constant from a
            copy of today's value. The static and interactive tiers already answer ``True`` here.
        """
        default = inspect.signature(WebMap.rgb_composite).parameters["bands"].default
        assert default is DEFAULT_COMPOSITE_BANDS, (
            f"rgb_composite defaults to its own {default!r}, not to the shared constant"
        )

    def test_the_module_reads_the_shared_single_band(self):
        """``web/raster.py`` reads :data:`DEFAULT_BAND` — the sibling instance of the same finding.

        Test scenario:
            ``field(band=1)`` hardcodes the number ``base/spec/selection.py`` declares as "the band a
            builder reads when the caller names none". Identity cannot catch this one — ``DEFAULT_BAND`` is
            ``1`` and CPython caches small integers, so ``default is DEFAULT_BAND`` holds for a hardcoded
            literal too. What is checkable is whether the module reads the name at all: the attribute
            exists only if it was imported.
        """
        assert web_raster.DEFAULT_BAND is DEFAULT_BAND, (
            "web/raster.py does not import the shared DEFAULT_BAND"
        )

    def test_the_field_signature_defaults_to_that_name(self):
        """The imported name is what ``field`` defaults to, rather than an import nothing uses.

        Test scenario:
            The pair to the check above, and the reason it is read from the source: with the integers
            identical there is no runtime difference between reading the constant and rewriting its value,
            so the only evidence is the signature itself.
        """
        source = inspect.getsource(web_raster.RasterMixin.field)
        assert "band: int = DEFAULT_BAND," in source, (
            "field's band default is a literal, not the shared constant it imports"
        )


class TestTheCompositeEncoder:
    """Three stretched channels become one RGBA PNG, with NoData transparent."""

    def test_a_stretched_stack_becomes_a_png(self):
        """The encoder is what puts the composite on the map, so it must produce a real PNG."""
        stack = np.random.default_rng(1337).random((4, 5, 3))
        assert _decode(WebMap()._composite_png_datauri(stack)).shape == (4, 5, 4)

    def test_channel_order_is_preserved(self):
        """R, G and B are not interchangeable — a swap makes a false-colour image silently wrong."""
        stack = np.zeros((1, 3, 3))
        stack[0, 0] = (1.0, 0.0, 0.0)
        stack[0, 1] = (0.0, 1.0, 0.0)
        stack[0, 2] = (0.0, 0.0, 1.0)
        image = _decode(WebMap()._composite_png_datauri(stack))
        assert image[0, 0, 0] == pytest.approx(1.0, abs=0.01), "red channel moved"
        assert image[0, 1, 1] == pytest.approx(1.0, abs=0.01), "green channel moved"
        assert image[0, 2, 2] == pytest.approx(1.0, abs=0.01), "blue channel moved"

    def test_nodata_in_any_channel_is_transparent(self):
        """A pixel missing one band has no colour, so drawing it would invent one."""
        stack = np.ones((1, 2, 3))
        stack[0, 1, 1] = np.nan
        image = _decode(WebMap()._composite_png_datauri(stack))
        assert image[0, 0, 3] == pytest.approx(1.0), "a complete pixel went transparent"
        assert image[0, 1, 3] == pytest.approx(0.0), "an incomplete pixel was drawn"

    def test_a_stack_with_no_complete_pixel_is_refused(self):
        """An all-transparent image looks like a rendering failure rather than an empty input."""
        web_map = WebMap()
        full = np.full((2, 2, 3), np.nan)
        with pytest.raises(ValueError, match="no pixel finite"):
            web_map._composite_png_datauri(full)

    def test_values_outside_the_unit_range_are_clipped(self):
        """`stretch_to_unit` can overshoot with explicit limits, and a PNG has no room for it."""
        stack = np.array([[[-0.5, 0.5, 1.5]]])
        image = _decode(WebMap()._composite_png_datauri(stack))
        assert image[0, 0, 0] == pytest.approx(0.0, abs=0.01)
        assert image[0, 0, 2] == pytest.approx(1.0, abs=0.01)


class TestTheCompositeOnTheMap:
    """The builder wires the encoder to a MapLibre image source."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    def test_three_bands_are_required(self, dataset):
        """Two or four bands is a caller error the renderer would report far less clearly.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        web_map = WebMap()
        with pytest.raises(ValueError):
            web_map.rgb_composite(dataset, bands=(1, 1))

    def test_it_registers_an_image_layer(self, dataset):
        """A composite is an image source like the single-band path, not a new kind of layer.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        m = WebMap().basemap().rgb_composite(dataset, bands=(1, 1, 1))
        html = m.to_html()
        payload = html[html.rfind("var data = ") :]
        assert '"type": "image"' in payload
        assert "data:image/png;base64," in payload

    def test_it_frames_the_map_like_any_other_layer(self, dataset):
        """A composite is data, so a map showing only one should open on it.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        m = WebMap().basemap().rgb_composite(dataset, bands=(1, 1, 1))
        assert m._data_bounds is not None

    def test_it_is_an_addressable_layer(self, dataset):
        """A viewer should be able to switch imagery off to see what is underneath.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        m = (
            WebMap()
            .basemap()
            .rgb_composite(dataset, bands=(1, 1, 1), name="True colour")
        )
        assert m.layer_ids == ["tiles-1", "True colour"], m.layer_ids
        html = m.layer_control().to_html()
        payload = html[html.rfind("var data = ") :]
        assert '"layerIds": ["True colour"]' in payload, (
            "the switcher would caption this row with a generated id"
        )

    def test_the_stretch_matches_the_static_tier(self, dataset):
        """Two tiers that disagree about the same composite would be worse than one lacking it.

        Test scenario:
            Both call ``digitalearth.base.stretch.stretch_to_unit`` over the same
            ``digitalearth.base.sources.get_stack`` output, so the encoded pixels must equal a stretch
            computed independently here.
        """
        from digitalearth.base.sources import get_stack
        from digitalearth.base.stretch import stretch_to_unit

        bands = (1, 1, 1)
        m = WebMap()
        expected = stretch_to_unit(
            get_stack(m._to_display_raster(dataset), bands, mask=True)
        )
        uri = m._composite_png_datauri(expected)
        m.basemap().rgb_composite(dataset, bands=bands)
        html = m.to_html()
        # The whole base64 payload, not a prefix: a PNG's first bytes are the header and the IHDR
        # dimensions, which two different images of the same size share.
        assert uri.split(",", 1)[1] in html, (
            "the drawn composite is not the shared stretch"
        )


class TestTheBandCountGuardIsTheSharedOne:
    """#266 — one helper answers "is this three bands?", on both ways into a composite.

    ``base/stretch.py``'s ``require_three_bands`` says why it is checked up front: any other count
    otherwise "surfaces deep inside the renderer". A builder is not the only way in — a figure is drawn from
    its description, by a redraw onto another view or from JSON written elsewhere, and that path has no
    builder in front of it.
    """

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    @staticmethod
    def _described_with(bands):
        """Return a composite description naming ``bands``, as a stored figure would carry it.

        Args:
            bands: The band tuple to record.

        Returns:
            A `LayerSpec` of kind ``rgb`` with the four props the drawer reads.
        """
        return LayerSpec(
            "rgb",
            "rgb",
            symbology=Symbology(
                props={
                    "bands": bands,
                    "mask_nodata": True,
                    "limits": None,
                    "opacity": 1.0,
                }
            ),
        )

    def test_the_builder_refuses_in_the_shared_words(self, dataset):
        """The builder's refusal is the shared helper's message, not one this tier writes.

        Args:
            dataset: The shared pyramids raster fixture.

        Test scenario:
            The half that was already right, pinned so the extraction below cannot quietly replace it with a
            locally worded check again.
        """
        m = WebMap()
        with pytest.raises(ValueError, match=r"needs exactly three bands, got 2"):
            m.rgb_composite(dataset, bands=(1, 2))

    def test_the_drawer_refuses_in_the_same_words(self, mercator_rgb):
        """A described composite with two bands is refused by count, not by array shape.

        Args:
            mercator_rgb: A three-band EPSG:3857 raster, so the drawer really warps and reads it.

        Test scenario:
            Without the guard the bands reach ``get_stack``, which fails with numpy's ``could not broadcast
            input array from shape (4,9,2) into shape (4,9,3)`` — a message about an array the caller never
            named, from inside a renderer they did not call.
        """
        m = WebMap()
        described = self._described_with((1, 2))
        with pytest.raises(ValueError, match=r"needs exactly three bands, got 2"):
            web_raster.draw_rgb_composite(m, mercator_rgb, described)


class TestACompositeTheWarpReshapes:
    """Review H1: the image and the corners it is placed on come from one warp of the same pixels."""

    #: The three bands every case below draws, one per channel so a swap would show.
    BANDS = (1, 2, 3)

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    def test_the_fixture_is_one_the_warp_reshapes(self, mercator_rgb):
        """A grid the warp leaves alone cannot tell the warped pixels from the unwarped ones.

        Args:
            mercator_rgb: A three-band EPSG:3857 raster.

        Test scenario:
            `acc4000.tif` keeps its shape and values through the warp, which is why the defect passed the
            suite. This pins that the fixture below does not, so the next test asks a real question.
        """
        warped = WebMap()._to_display_raster(mercator_rgb)
        assert (warped.rows, warped.columns) != (
            mercator_rgb.rows,
            mercator_rgb.columns,
        ), "the warp kept the grid, so the test below cannot fail"

    def test_the_drawn_image_is_the_warped_grid(self, mercator_rgb):
        """The PNG placed on the lon/lat corners is the stretch of the pixels those corners bound.

        Args:
            mercator_rgb: A three-band EPSG:3857 raster.

        Test scenario:
            The expected image is built here from the dataset warped to the display CRS — the grid the
            corners are taken from. A drawer that stacked the unwarped EPSG:3857 grid encodes a 6 x 8 image
            and stretches it over the 4 x 9 grid's extent; the whole payload is compared, so a different
            shape or different pixels both fail.
        """
        from digitalearth.base.sources import get_source, get_stack
        from digitalearth.base.stretch import stretch_to_unit

        m = WebMap()
        warped = m._to_display_raster(mercator_rgb)
        stack = stretch_to_unit(get_stack(warped, self.BANDS, mask=True))
        y = np.asarray(get_source(warped).y.values, dtype=float)
        if y.size > 1 and y[0] < y[-1]:
            stack = stack[::-1]
        expected = m._composite_png_datauri(stack)

        m.rgb_composite(mercator_rgb, bands=self.BANDS, name="rgb")
        drawn = m._renderer.drawn["rgb"].source_spec["url"]
        assert _decode(drawn).shape[:2] == (warped.rows, warped.columns), (
            "the image is not the warped grid"
        )
        assert drawn == expected, "the drawn pixels are not the warped stretch"

    def test_a_redraw_from_the_figure_draws_the_same_image(self, mercator_rgb):
        """A figure handed straight to the renderer is placed by the drawer, not by the builder.

        Args:
            mercator_rgb: A three-band EPSG:3857 raster.

        Test scenario:
            The figure records the caller's own, unwarped dataset. The builder's call warps it before the
            first draw; a redraw from the figure has no builder, so the drawer has to warp it itself — and
            what it draws must be the image the build drew.
        """
        m = WebMap().rgb_composite(mercator_rgb, bands=self.BANDS, name="rgb")
        built = m._renderer.drawn["rgb"].source_spec
        redrawn = m._renderer.draw_layer(m.figure_spec, "rgb").source_spec
        assert redrawn["url"] == built["url"], "the redraw encoded a different image"
        assert redrawn["coordinates"] == built["coordinates"], (
            "the redraw placed the image elsewhere"
        )


class TestFrozenLimitsAChannelCouldNotMeasure:
    """A description holds only what a figure can be written as, and NaN has no JSON form."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    @staticmethod
    def _frozen_limits():
        """Return per-channel limits whose third channel could not be measured.

        Returns:
            What `channel_limits` answers for a stack whose third channel is all nodata — `(nan, nan)`,
            which it documents as "this channel contributes no bound". A caller freezing a composite series
            on its first frame gets exactly this, and passes it to every later frame.
        """
        from digitalearth.base.stretch import channel_limits

        measured = np.arange(16.0).reshape(4, 4)
        return channel_limits(
            np.dstack([measured, measured * 2, np.full((4, 4), np.nan)])
        )

    def test_the_fixture_is_a_channel_with_no_bound(self):
        """The limits under test have to carry the NaN pair, or the checks below ask nothing."""
        assert np.isnan(self._frozen_limits()[2]).all(), self._frozen_limits()

    def test_the_figure_can_be_written(self, mercator_rgb):
        """A composite drawn with frozen limits must still produce a figure that can be saved.

        Args:
            mercator_rgb: A three-band raster.

        Test scenario:
            Freezing the stretch is the documented way to keep a series comparable, and a channel the
            freeze could not measure is a documented, normal result. Recorded as NaN, the whole figure
            stopped being writable — `Symbology.to_dict()` refuses it — which is the property the seam
            exists to provide.
        """
        m = WebMap().rgb_composite(
            mercator_rgb, bands=(1, 2, 3), limits=self._frozen_limits(), name="rgb"
        )
        written = m.figure_spec.layers.get("rgb").symbology.to_dict()
        json.dumps(written)
        assert written["props"]["limits"][2] == [None, None], written["props"]["limits"]

    def test_the_unmeasured_channel_still_falls_back_to_this_frame(self, mercator_rgb):
        """What is drawn must not change: an unmeasured bound still means "use this frame's own".

        Args:
            mercator_rgb: A three-band raster.
        """
        from digitalearth.base.sources import get_source, get_stack
        from digitalearth.base.stretch import stretch_to_unit

        m = WebMap()
        warped = m._to_display_raster(mercator_rgb)
        stack = stretch_to_unit(
            get_stack(warped, (1, 2, 3), mask=True), self._frozen_limits()
        )
        y = np.asarray(get_source(warped).y.values, dtype=float)
        if y.size > 1 and y[0] < y[-1]:
            stack = stack[::-1]
        expected = m._composite_png_datauri(stack)

        m.rgb_composite(
            mercator_rgb, bands=(1, 2, 3), limits=self._frozen_limits(), name="rgb"
        )
        assert m._renderer.drawn["rgb"].source_spec["url"] == expected, (
            "the recorded limits no longer draw the stretch they were given"
        )


class TestOneBuildWarpsOnce:
    """Review M8: a builder must not warp the raster its drawer then warps again."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(lambda m, ds: m.field(ds, band=1), id="field"),
            pytest.param(
                lambda m, ds: m.rgb_composite(ds, bands=(1, 2, 3)), id="rgb_composite"
            ),
        ],
    )
    def test_the_raster_is_warped_once(self, mercator_rgb, warp_counter, build):
        """The comment on the builder promised to spare "a multi-second GDAL warp"; it paid for two.

        Args:
            mercator_rgb: A raster that is not in the display CRS, so every build has to warp it.
            warp_counter: Counts the warps made on either of the tier's two warp paths.
            build: The builder call under test.
        """
        build(WebMap(), mercator_rgb)
        assert warp_counter["warps"] == 1, (
            f"one build warped the raster {warp_counter['warps']} times"
        )


def _source_with(y_values):
    """Build a stand-in source exposing only the coordinate arrays the raster builders read.

    Args:
        y_values: The y coordinates, ascending or descending.

    Returns:
        An object with ``.x.values`` and ``.y.values``.
    """
    import numpy as np

    axis = type("Axis", (), {})
    x, y = axis(), axis()
    x.values = np.array([0.0, 1.0])
    y.values = np.asarray(y_values, dtype=float)
    return type("Source", (), {"x": x, "y": y})()


class TestTheCompositeSizeCeiling:
    """The inline data-URI path has a ceiling, and past it a composite is refused rather than warned about."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    def test_an_oversized_composite_is_refused(self, dataset, monkeypatch):
        """A warning was all that stood between a caller and a hundreds-of-MB page, and it stopped nothing.

        Args:
            dataset: The shared pyramids raster fixture.
            monkeypatch: pytest's patcher, used to lower the ceiling rather than build a huge raster.

        Test scenario:
            Three bands inline as three times the data of one, so the check is on the stack's size. The
            refusal names the builder and both tiled routes, since a caller who is refused has to be told
            which parameter to reach for (#189).
        """
        from digitalearth.web import raster as raster_module

        monkeypatch.setattr(raster_module, "_LARGE_RASTER_PIXELS", 1)
        m = WebMap().basemap()
        with pytest.raises(ValueError, match="rgb_composite") as refusal:
            m.rgb_composite(dataset, bands=(1, 1, 1))
        message = str(refusal.value)
        assert '"xyz"' in message, message
        assert '"cog"' in message, message

    def test_a_small_composite_is_drawn_as_it_always_was(self, dataset):
        """A ceiling that fired on ordinary input would break every map already built.

        Args:
            dataset: The shared pyramids raster fixture, far below the ceiling.
        """
        m = WebMap().basemap().rgb_composite(dataset, bands=(1, 1, 1))
        spec = m._renderer.drawn[m._last_layer_id].source_spec
        assert spec["type"] == "image", spec

    def test_the_refusal_belongs_to_the_call_and_not_to_every_draw(
        self, dataset, monkeypatch
    ):
        """Review N7 — a drawer runs again on every redraw; the size of the input was chosen once.

        Args:
            dataset: The shared pyramids raster fixture.
            monkeypatch: pytest's patcher, used to lower the ceiling rather than build a huge raster.

        Test scenario:
            A figure written before the ceiling existed already holds its pixels, so drawing it back must not
            fail on a size its builder accepted. The check therefore stays with the builder — where it can
            still offer a route — and the drawer never re-runs it.
        """
        from digitalearth.web import raster as raster_module

        built = WebMap().rgb_composite(dataset, bands=(1, 1, 1))
        figure = built.figure_spec
        monkeypatch.setattr(raster_module, "_LARGE_RASTER_PIXELS", 1)
        redrawn = WebMap()._renderer.draw_layer(figure, figure.layers.ids[0])
        assert redrawn is not None, "a stored composite must still draw back"

    def test_an_input_with_no_integer_grid_is_not_guessed_at(self, monkeypatch):
        """A size that cannot be measured must produce no number and no refusal.

        Args:
            monkeypatch: pytest's patcher, used to lower the ceiling to one pixel.

        Test scenario:
            The composite builder reads the grid rather than a band, because reading a band only to
            measure it costs a whole band read. Anything that does not report `rows`/`columns` as plain
            integers therefore has no measurable size — and a fallback that guessed one would refuse a page
            weight nobody measured. With the ceiling at one pixel, any guess at all refuses.
        """
        from digitalearth.web import raster as raster_module

        monkeypatch.setattr(raster_module, "_LARGE_RASTER_PIXELS", 1)
        unmeasurable = type("Grid", (), {"rows": "many", "columns": 3})()
        measured = raster_module._grid_pixels(unmeasurable)
        assert measured is None, (
            f"an unmeasurable grid must answer None; got {measured}"
        )
        raster_module._refuse_if_large("rgb_composite", "composite", measured)


class TestTheTiledCompositeRoutes:
    """#189 — the composites share the inline path, so they share the way out of it."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    def test_rgb_composite_writes_a_pyramid_the_widget_reads_as_a_raster_source(
        self, dataset, tmp_path
    ):
        """What the map draws is the source spec, not what the figure records.

        Args:
            dataset: The shared pyramids raster fixture.
            tmp_path: pytest's temporary directory, where the pyramid is written.
        """
        import io

        from matplotlib import image as mpimage

        m = WebMap().rgb_composite(
            dataset,
            bands=(1, 1, 1),
            tiles="xyz",
            tiles_path=tmp_path / "scene",
            zooms=(9, 9),
        )
        spec = m._renderer.drawn[m._last_layer_id].source_spec
        assert spec["tiles"] == ["scene/{z}/{x}/{y}.png"], spec
        written = sorted((tmp_path / "scene").rglob("*.png"))
        assert written, "the route has to leave a pyramid behind"
        tile = mpimage.imread(io.BytesIO(written[0].read_bytes()))
        assert tile.shape == (256, 256, 4), tile.shape

    def test_a_tiled_composite_clears_the_ceiling(self, dataset, tmp_path, monkeypatch):
        """The ceiling is the inline path's, so naming a route has to clear it for a composite too.

        Args:
            dataset: The shared pyramids raster fixture.
            tmp_path: pytest's temporary directory.
            monkeypatch: pytest's patcher, used to lower the ceiling.
        """
        from digitalearth.web import raster as raster_module

        monkeypatch.setattr(raster_module, "_LARGE_RASTER_PIXELS", 1)
        m = WebMap().hsv_composite(
            dataset,
            bands=(1, 1, 1),
            tiles="cog",
            tiles_path=tmp_path / "scene.tif",
            limits=[(0.0, 90.0)] * 3,
        )
        spec = m._renderer.drawn[m._last_layer_id].source_spec
        assert spec["url"] == "cog://scene.tif", spec

    def test_the_stretch_is_measured_once_for_the_whole_pyramid(
        self, dataset, tmp_path
    ):
        """Every tile has to be stretched on the same bounds, or neighbours disagree at their shared edge.

        Args:
            dataset: The shared pyramids raster fixture.
            tmp_path: pytest's temporary directory.

        Test scenario:
            The recorded ``limits`` are what the description hands every later draw, so they are the evidence
            the stretch was frozen rather than re-derived per tile. A caller's own freeze is kept as given;
            an absent one is measured from a decimated read, and either way it is one set of bounds.
        """
        m = WebMap().rgb_composite(
            dataset,
            bands=(1, 1, 1),
            tiles="xyz",
            tiles_path=tmp_path / "scene",
            zooms=(9, 9),
        )
        figure = m.figure_spec
        recorded = figure.layers.get(figure.layers.ids[0]).symbology.props["limits"]
        assert len(recorded) == 3, recorded
        assert len(set(recorded)) == 1, (
            f"three copies of one band must freeze to one pair: {recorded}"
        )

    def test_a_composite_tile_off_the_raster_is_left_out_of_the_pyramid(
        self, dataset, tmp_path, monkeypatch
    ):
        """An extent's bounding box is not the extent, so a composite pyramid legitimately has holes.

        Args:
            dataset: The shared pyramids raster fixture.
            tmp_path: pytest's temporary directory, holding both pyramids written below.
            monkeypatch: Used to make one window miss the raster, which one this small never does.

        Test scenario:
            The same range is written twice — once whole, once with the first window standing in a miss —
            and the two are counted against each other rather than against a literal tile count, which
            depends on where the fixture's extent falls in the tile grid. One hole must leave exactly one
            tile out; a blank tile written for it would leave the counts equal.
        """
        from digitalearth.web import raster as raster_module

        real_stack = raster_module._tile_stack
        missed: list = []

        def _the_first_window_misses(opened, zoom, x, y, *, bands):
            """Answer ``None`` for the first window asked for, then read as usual.

            Args:
                opened: The dataset the real reader is given.
                zoom: The zoom level.
                x: The tile column.
                y: The tile row.
                bands: The three 1-based bands the composite reads.

            Returns:
                ``None`` the first time, then whatever the real reader answers.
            """
            if not missed:
                missed.append((zoom, x, y))
                return None
            return real_stack(opened, zoom, x, y, bands=bands)

        WebMap().rgb_composite(
            dataset,
            bands=(1, 1, 1),
            tiles="xyz",
            tiles_path=tmp_path / "whole",
            zooms=(8, 9),
        )
        monkeypatch.setattr(raster_module, "_tile_stack", _the_first_window_misses)
        WebMap().rgb_composite(
            dataset,
            bands=(1, 1, 1),
            tiles="xyz",
            tiles_path=tmp_path / "holed",
            zooms=(8, 9),
        )
        complete = sorted((tmp_path / "whole").rglob("*.png"))
        holed = sorted((tmp_path / "holed").rglob("*.png"))
        assert missed, "the stand-in never saw a window"
        assert holed, "the windows that do meet the raster must still be written"
        assert len(holed) == len(complete) - 1, (
            f"one hole must leave one tile out: {len(holed)} against {len(complete)}"
        )

    def test_a_declined_tiled_composite_is_not_the_maps_last_layer(
        self, dataset, tmp_path, monkeypatch
    ):
        """The description is written before the drawer runs, so a decline has to take the record back.

        Args:
            dataset: The shared pyramids raster fixture.
            tmp_path: pytest's temporary directory, where the pyramid is written.
            monkeypatch: Used to make the composite drawer decline, which the tiled route's own never does.

        Test scenario:
            The tiled drawer places a reference rather than pixels, so nothing in it can decline — but the
            builder shares the registration check with the inline path, where an unplaceable raster can. A
            declined layer that still became ``_last_layer_id`` would have a later ``colorbar()`` label
            itself from a layer no widget holds.
        """
        from digitalearth.web import raster as raster_module

        def _declines(web_map, data, layer):
            """Behave like a drawer that found nothing to draw.

            Args:
                web_map: Unused — the drawer declines before it would read the map.
                data: Unused — the raster it would have placed.
                layer: Unused — the description it would have drawn.

            Returns:
                ``None``, the shape a drawer answers in when it declines.
            """
            return None

        m = WebMap().rgb_composite(dataset, bands=(1, 1, 1), name="inline")
        monkeypatch.setattr(raster_module, "draw_rgb_composite", _declines)
        m.hsv_composite(
            dataset,
            bands=(1, 1, 1),
            tiles="xyz",
            tiles_path=tmp_path / "scene",
            zooms=(9, 9),
            name="tiled",
        )
        assert m.layer_ids == ["inline"], m.layer_ids
        assert m._last_layer_id == "inline", (
            f"a declined tiled composite became the map's last layer: {m._last_layer_id!r}"
        )


class TestNorthUpOrientation:
    """PNG row 0 is the northern edge, so a source whose rows run south-first has to be flipped."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    @pytest.mark.parametrize(
        "y_values, flipped",
        [([1.0, 0.0], False), ([0.0, 1.0], True)],
    )
    def test_rows_are_ordered_north_first(
        self, dataset, monkeypatch, y_values, flipped
    ):
        """An ascending y means row 0 is the south edge, which renders the image upside down.

        Args:
            dataset: The shared pyramids raster fixture.
            monkeypatch: pytest's patcher, used to hand the builder a source with a chosen y order.
            y_values: The coordinate order the source reports.
            flipped: Whether the stack should be reversed before encoding.

        Test scenario:
            The encoded pixels are compared against the orientation that *should* be drawn, so the test
            fails both when the flip is missing and when it happens for a north-up source.
        """
        import numpy as np

        from digitalearth.base.sources import get_stack
        from digitalearth.base.stretch import stretch_to_unit

        source = _source_with(y_values)
        monkeypatch.setattr(
            WebMap, "_to_display_source", lambda self, data, band=1: source
        )

        captured = {}
        encoder = WebMap._composite_png_datauri

        def spy(unit_stack, *, caller=web_raster._RGB_RECIPE):
            """Record the stack handed to the encoder, then encode it normally.

            Args:
                unit_stack: The stretched channels the drawer built.
                caller: The builder the drawer names for the encoder's own refusal, forwarded as given.
            """
            captured["stack"] = np.array(unit_stack, copy=True)
            return encoder(unit_stack, caller=caller)

        monkeypatch.setattr(WebMap, "_composite_png_datauri", staticmethod(spy))

        m = WebMap()
        stack = stretch_to_unit(
            get_stack(m._to_display_raster(dataset), (1, 1, 1), mask=True)
        )
        m.basemap().rgb_composite(dataset, bands=(1, 1, 1))
        expected = stack[::-1] if flipped else stack
        assert np.allclose(captured["stack"], expected, equal_nan=True), (
            f"rows are not north-first for y={y_values}"
        )


class TestLimitsShapedLikeSomethingElse:
    """Both limit translators hand back what they cannot read, so the stretch refuses it in its words."""

    @pytest.mark.parametrize(
        "limits",
        [
            pytest.param(7.5, id="not-a-sequence-at-all"),
            pytest.param(((1.0, 2.0, 3.0),), id="a-triple-where-a-pair-belongs"),
        ],
    )
    @pytest.mark.parametrize(
        "translate",
        [
            pytest.param("_recorded_limits", id="writing-it-down"),
            pytest.param("_stretch_limits", id="reading-it-back"),
        ],
    )
    def test_it_is_handed_back_untouched(self, translate, limits):
        """Neither translator is the place a malformed ``limits=`` is diagnosed.

        Args:
            translate: The name of the translator under test.
            limits: A value shaped like neither `None` nor a sequence of pairs.

        Test scenario:
            ``stretch_to_unit`` owns the message that tells a caller what per-channel limits look like,
            and it names the argument. A translator that instead fails on the value first — iterating a
            float, or unpacking a triple — replaces that message with a `TypeError` from inside the web
            tier, about a private helper the caller never called.
        """
        from digitalearth.web import raster as web_raster

        assert getattr(web_raster, translate)(limits) is limits, (
            f"{translate} rewrote a value it cannot read: {limits!r}"
        )

    def test_a_bound_that_is_not_a_number_is_handed_back_too(self):
        """A pair of the right *shape* can still hold something `float()` refuses.

        Test scenario:
            The length check passes for ``("a", "b")``, so only the conversion can catch it — and the
            conversion is inside a comprehension whose `ValueError` would otherwise escape from
            `rgb_composite` as if the builder itself were broken.
        """
        from digitalearth.web.raster import _recorded_limits

        limits = (("a", "b"),)
        assert _recorded_limits(limits) is limits, (
            "a pair holding something that is not a number must not be rewritten"
        )


class TestACompositeTheViewCannotPlace:
    """The drawer warps for itself, so it meets the off-limb answer the builder already handled."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    def test_the_drawer_skips_it_instead_of_stacking_nothing(
        self, mercator_rgb, monkeypatch
    ):
        """A described composite can be re-drawn onto a view its data no longer reaches.

        Args:
            mercator_rgb: A three-band EPSG:3857 raster.
            monkeypatch: Used to make the warp answer "off-limb" the way it does for real data.

        Test scenario:
            `rgb_composite` refuses an off-limb dataset before it records anything, so the drawer's own
            copy of that guard is only reached on a re-draw — a figure captured on one view and applied to
            another. Without it the `None` goes straight into `get_stack`, and the layer fails with an
            `AttributeError` on `NoneType` rather than being skipped like every other off-limb layer.
        """
        web_map = WebMap().rgb_composite(mercator_rgb, bands=(1, 2, 3), name="rgb")
        figure = web_map.figure_spec
        monkeypatch.setattr(
            WebMap,
            "_display_raster_or_skip",
            lambda self, dataset, *, layer: None,
        )
        assert web_map._renderer.draw_layer(figure, "rgb") is None, (
            "a composite whose data cannot be placed must be skipped, not drawn"
        )


class TestTheHsvComposite:
    """#266 — the tier's second composite: the three bands read as hue, saturation and value.

    The static tier has had both composites since it had one; this tier had only RGB, and did not declare
    HSV absent either — so it was missing rather than refused. It arrives as the same concept through this
    tier's own engine: three bands, each stretched by ``base/stretch.py``, composed in HSV space and encoded
    as the one PNG a MapLibre image source takes.
    """

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    @staticmethod
    def _drawn(web_map, layer_id="hsv"):
        """Return the image the widget was handed for one layer.

        The renderer's record is what ``render()`` builds the widget from, so this is the drawn image rather
        than the description of it — two different questions, and only the second is cheap to get right.

        Args:
            web_map: The map that drew it.
            layer_id: Which layer to read.

        Returns:
            The decoded RGBA image.
        """
        return _decode(web_map._renderer.drawn[layer_id].source_spec["url"])

    def test_the_image_the_widget_receives_is_the_hue_wheel(self, hue_wheel):
        """Three bands read as hue/saturation/value draw pure red, green and blue.

        Args:
            hue_wheel: The raster whose HSV composite is written down in :data:`_PURE_HUES`.

        Test scenario:
            The expectation is the colour theory, not a second run of the tier's own code path: hue 0, 1/3
            and 2/3 at full saturation and value are the primaries. A composite that read the bands as RGB,
            or that mapped them to the wrong channels, lands somewhere else.
        """
        web_map = WebMap().hsv_composite(
            hue_wheel, bands=_BANDS, limits=_UNIT_LIMITS, name="hsv"
        )
        image = self._drawn(web_map)[..., :3]
        assert np.allclose(image, _PURE_HUES, atol=_ONE_LEVEL), image

    def test_it_draws_something_other_than_the_rgb_composite(self, hue_wheel):
        """The same bands through the two builders must not encode the same pixels.

        Args:
            hue_wheel: The three-band raster both builders read.

        Test scenario:
            The failure this catches is an HSV builder that records its recipe and is then drawn by the RGB
            path anyway — every other check about the layer still passes, and the image is a plausible false
            colour nobody can spot.
        """
        hsv = self._drawn(
            WebMap().hsv_composite(
                hue_wheel, bands=_BANDS, limits=_UNIT_LIMITS, name="hsv"
            )
        )
        rgb = self._drawn(
            WebMap().rgb_composite(
                hue_wheel, bands=_BANDS, limits=_UNIT_LIMITS, name="hsv"
            )
        )
        assert not np.allclose(hsv, rgb, atol=_ONE_LEVEL), (
            "the HSV composite encoded the RGB image, so its recipe was not read"
        )

    def test_a_cell_missing_its_hue_stays_transparent(self, hue_wheel_with_a_gap):
        """NoData in any channel means no colour, in HSV space as in RGB.

        Args:
            hue_wheel_with_a_gap: The hue wheel with its first cell's hue set to NoData.

        Test scenario:
            This is the trap the colour space sets. ``matplotlib.colors.hsv_to_rgb`` reads the hue with
            ``(h * 6).astype(int)``, and casting NaN gives a garbage index rather than NaN, so fed the masked
            stack directly it answers a **finite** triple for that cell — measured, ``(0.0, 0.0, 0.0)`` for
            this fixture's full saturation and value, ``(0.5, 1.04e-311, 1.04e-311)`` for a half-saturated
            one — and a finite triple is drawn opaque, with only a ``RuntimeWarning`` to say anything
            happened.
        """
        web_map = WebMap().hsv_composite(
            hue_wheel_with_a_gap, bands=_BANDS, limits=_UNIT_LIMITS, name="hsv"
        )
        alpha = self._drawn(web_map)[..., 3]
        assert alpha[0, 0] == pytest.approx(0.0), (
            f"a cell with no hue was drawn, at alpha {alpha[0, 0]}"
        )
        assert alpha[1, 0] == pytest.approx(1.0), (
            "the complete cell below it went transparent too"
        )

    def test_the_figure_records_the_recipe_under_the_shared_kind(self, hue_wheel):
        """The layer is an ``rgb`` layer — the vocabulary's word for a multi-band colour composite.

        Args:
            hue_wheel: The three-band raster.

        Test scenario:
            ``base/registry.py`` defines ``rgb`` as "a multi-band colour composite", naming static's
            ``hsv_composite`` among the builders that draw it, and the static tier records both composites
            under it with the recipe in ``via``. A private new kind here would be a fifth vocabulary.
        """
        web_map = WebMap().hsv_composite(
            hue_wheel, bands=_BANDS, limits=_UNIT_LIMITS, name="hsv"
        )
        layer = web_map.figure_spec.layers.get("hsv")
        assert layer.kind == "rgb", layer.kind
        assert layer.symbology.props["via"] == "hsv_composite", layer.symbology.props

    def test_the_kind_is_declared_and_drawn(self, hue_wheel):
        """The kind the builder records is in the tier's declaration and in its drawer table.

        Args:
            hue_wheel: The three-band raster.

        Test scenario:
            ``drawer_for`` refuses a kind that is not in ``DRAWN_KINDS``, and the tier's capability check
            reads ``CAPABILITIES.kinds``: a composite recorded under a kind missing from either is a layer
            that cannot be drawn back from its own figure.
        """
        from digitalearth.web.capabilities import CAPABILITIES
        from digitalearth.web.renderer import DRAWN_KINDS, drawer_for

        web_map = WebMap().hsv_composite(
            hue_wheel, bands=_BANDS, limits=_UNIT_LIMITS, name="hsv"
        )
        kind = web_map.figure_spec.layers.get("hsv").kind
        assert kind in CAPABILITIES.kinds, sorted(CAPABILITIES.kinds)
        assert kind in DRAWN_KINDS, sorted(DRAWN_KINDS)
        assert drawer_for(kind) is web_raster.draw_rgb_composite

    def test_the_figure_can_be_written_and_redrawn(self, hue_wheel):
        """A stored HSV figure comes back as the same image, through the drawer alone.

        Args:
            hue_wheel: The three-band raster.

        Test scenario:
            The builder warps and records; a redraw has no builder, so it warps and composes for itself. If
            the recipe were held on the map rather than in the description, the redraw would fall back to
            RGB — and the only symptom is a different image.
        """
        web_map = WebMap().hsv_composite(
            hue_wheel, bands=_BANDS, limits=_UNIT_LIMITS, name="hsv"
        )
        figure = web_map.figure_spec
        json.dumps(figure.layers.get("hsv").symbology.to_dict())
        redrawn = web_map._renderer.draw_layer(figure, "hsv").source_spec["url"]
        assert redrawn == web_map._renderer.drawn["hsv"].source_spec["url"], (
            "the redraw encoded a different image"
        )

    def test_a_description_with_no_recipe_still_draws_the_rgb_composite(
        self, hue_wheel
    ):
        """A figure written before this tier had two composites means the one it had.

        Args:
            hue_wheel: The three-band raster.

        Test scenario:
            Every stored ``rgb`` layer predating ``hsv_composite`` carries no ``via`` at all. Read as
            "unknown recipe" it would refuse figures that used to draw; read as HSV it would silently redraw
            them in another colour space.
        """
        described = LayerSpec(
            "rgb",
            "rgb",
            symbology=Symbology(
                props={
                    "bands": _BANDS,
                    "mask_nodata": True,
                    "limits": _UNIT_LIMITS,
                    "opacity": 1.0,
                }
            ),
        )
        without_a_recipe = web_raster.draw_rgb_composite(
            WebMap(), hue_wheel, described
        ).source_spec["url"]
        built = WebMap().rgb_composite(
            hue_wheel, bands=_BANDS, limits=_UNIT_LIMITS, name="rgb"
        )
        assert without_a_recipe == built._renderer.drawn["rgb"].source_spec["url"]

    def test_a_recipe_this_tier_does_not_draw_is_refused(self, hue_wheel):
        """A description naming a composite the tier has no path for says so, rather than drawing RGB.

        Args:
            hue_wheel: The three-band raster.

        Test scenario:
            The same answer the static renderer gives an unknown ``via``. Silently drawing the default
            instead would make a figure from a newer version render as something it does not claim to be.
        """
        described = LayerSpec(
            "rgb",
            "rgb",
            symbology=Symbology(
                props={
                    "bands": _BANDS,
                    "mask_nodata": True,
                    "limits": None,
                    "opacity": 1.0,
                    "via": "cmyk_composite",
                }
            ),
        )
        m = WebMap()
        with pytest.raises(ValueError, match="cmyk_composite"):
            web_raster.draw_rgb_composite(m, hue_wheel, described)

    def test_three_bands_are_required_here_too(self, hue_wheel):
        """The shared guard names this builder, not its sibling.

        Args:
            hue_wheel: The three-band raster.
        """
        m = WebMap()
        with pytest.raises(ValueError, match=r"hsv_composite\(\) needs exactly three"):
            m.hsv_composite(hue_wheel, bands=(1, 2))

    def test_an_empty_composite_names_this_builder(self):
        """A stack with nothing to draw is refused in the name of the call that was made.

        Test scenario:
            The encoder's refusal read ``rgb_composite got a stack with no pixel finite …`` whichever
            builder reached it. Naming the wrong sibling sends the reader to the wrong call.
        """
        empty = _hue_dataset([[-9999.0] * 3] * 2)
        m = WebMap()
        with pytest.raises(
            ValueError, match=r"hsv_composite got a stack with no pixel"
        ):
            m.hsv_composite(empty, bands=_BANDS)

    def test_an_oversized_composite_is_refused_in_its_own_name(
        self, hue_wheel, monkeypatch
    ):
        """The size refusal names the builder the caller used.

        Args:
            hue_wheel: The three-band raster.
            monkeypatch: pytest's patcher, used to lower the ceiling rather than build a huge raster.

        Test scenario:
            The message is built from the builder's name, which the sibling passed as a literal. Left as it
            was, a caller of ``hsv_composite`` would be sent to ``rgb_composite``.
        """
        monkeypatch.setattr(web_raster, "_LARGE_RASTER_PIXELS", 1)
        m = WebMap()
        with pytest.raises(ValueError, match="hsv_composite"):
            m.hsv_composite(hue_wheel, bands=_BANDS, limits=_UNIT_LIMITS)

    def test_the_two_tiers_agree_on_the_shared_parameters(self):
        """Every parameter both tiers' ``hsv_composite`` take is spelled and defaulted the same.

        Test scenario:
            The issue's first condition is that the tiers agree on the name and the parameter names. The
            defaults are compared too, since ``mask_nodata=False`` on one tier and ``True`` on the other
            would be the same divergence one level down. What each tier adds of its own — this tier's
            ``opacity``, the static tier's ``**opts`` — is outside the shared set and left alone.
        """
        from digitalearth.static.maps.raster import RasterMixin as StaticRaster

        here = inspect.signature(WebMap.hsv_composite).parameters
        there = inspect.signature(StaticRaster.hsv_composite).parameters
        shared = [name for name in there if name in here and name != "self"]
        assert shared == [
            "dataset",
            "bands",
            "mask_nodata",
            "limits",
            "name",
            "visible",
        ]
        assert [(name, here[name].default) for name in shared] == [
            (name, there[name].default) for name in shared
        ]

    def test_the_two_tiers_draw_the_same_composite(self, hue_wheel):
        """The same call on both tiers produces the same pixels, run side by side.

        Args:
            hue_wheel: A raster already in lon/lat, so neither tier resamples and the only difference left
                would be the composite itself.

        Test scenario:
            "The tiers agree" is a claim about output. Both are handed the same dataset, the same bands and
            the same frozen limits; the static tier's image comes off its axes and this tier's out of the
            PNG the widget was given, and the two are compared to one 8-bit level.
        """
        from digitalearth.static import Map

        with Map(crs=4326) as static_map:
            static_map.hsv_composite(hue_wheel, bands=_BANDS, limits=_UNIT_LIMITS)
            drawn_there = np.asarray(static_map.ax.images[-1].get_array())
        drawn_here = self._drawn(
            WebMap().hsv_composite(
                hue_wheel, bands=_BANDS, limits=_UNIT_LIMITS, name="hsv"
            )
        )[..., :3]
        assert np.allclose(drawn_there, drawn_here, atol=_ONE_LEVEL), (
            f"static drew\n{drawn_there}\nweb drew\n{drawn_here}"
        )
