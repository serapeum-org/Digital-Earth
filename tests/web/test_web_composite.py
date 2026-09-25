"""RGB composites on the web tier (#190).

The encoder is a pure helper, tested without the engine (numpy/matplotlib are core deps via cleopatra);
``rgb_composite`` itself ``importorskip``s maplibre and runs on the shared ``dataset`` fixture. The stretch
is asserted to be the *same* one the static tier applies, which is the point of sharing
``digitalearth.base.stretch``.
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
        assert m.layer_ids == ["True colour"], m.layer_ids
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
        with pytest.raises(ValueError, match=r"needs exactly three bands, got 2"):
            WebMap().rgb_composite(dataset, bands=(1, 2))

    def test_the_drawer_refuses_in_the_same_words(self, mercator_rgb):
        """A described composite with two bands is refused by count, not by array shape.

        Args:
            mercator_rgb: A three-band EPSG:3857 raster, so the drawer really warps and reads it.

        Test scenario:
            Without the guard the bands reach ``get_stack``, which fails with numpy's ``could not broadcast
            input array from shape (4,9,2) into shape (4,9,3)`` — a message about an array the caller never
            named, from inside a renderer they did not call.
        """
        with pytest.raises(ValueError, match=r"needs exactly three bands, got 2"):
            web_raster.draw_rgb_composite(
                WebMap(), mercator_rgb, self._described_with((1, 2))
            )


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


class TestTheCompositeSizeWarning:
    """The inline data-URI path has a ceiling, and a caller past it needs to hear about it."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    @staticmethod
    def _capture():
        """Attach a loguru sink for warnings.

        Returns:
            ``(records, sink_id)`` — the list the sink fills, and the handle to remove it afterwards.
        """
        from loguru import logger

        records = []
        return records, logger.add(records.append, level="WARNING")

    def test_an_oversized_composite_warns(self, dataset, monkeypatch):
        """The warning is all that stands between a caller and a hundreds-of-MB page.

        Args:
            dataset: The shared pyramids raster fixture.
            monkeypatch: pytest's patcher, used to lower the ceiling rather than build a huge raster.

        Test scenario:
            Three bands inline as three times the data of one, so the check is on the stack's size.
        """
        from loguru import logger

        from digitalearth.web import raster as raster_module

        monkeypatch.setattr(raster_module, "_LARGE_RASTER_PIXELS", 1)
        records, sink_id = self._capture()
        try:
            WebMap().basemap().rgb_composite(dataset, bands=(1, 1, 1))
        finally:
            logger.remove(sink_id)
        assert any("rgb_composite" in str(r) for r in records), records

    def test_a_small_composite_stays_quiet(self, dataset):
        """A warning on every composite would train the caller to ignore it.

        Args:
            dataset: The shared pyramids raster fixture, far below the ceiling.
        """
        from loguru import logger

        records, sink_id = self._capture()
        try:
            WebMap().basemap().rgb_composite(dataset, bands=(1, 1, 1))
        finally:
            logger.remove(sink_id)
        assert not [r for r in records if "rgb_composite" in str(r)], records

    def test_the_warning_belongs_to_the_call_and_not_to_every_draw(
        self, dataset, monkeypatch
    ):
        """Review N7 — a drawer runs again on every redraw; the size of the input was chosen once.

        Args:
            dataset: The shared pyramids raster fixture.
            monkeypatch: pytest's patcher, used to lower the ceiling rather than build a huge raster.

        Test scenario:
            `field` warns from its builder and `rgb_composite` warned from its drawer, so drawing a stored
            composite back repeated the advice once per draw — and the two sibling builders answered the
            same question from two different places.
        """
        from loguru import logger

        from digitalearth.web import raster as raster_module

        monkeypatch.setattr(raster_module, "_LARGE_RASTER_PIXELS", 1)
        records, sink_id = self._capture()
        try:
            built = WebMap().rgb_composite(dataset, bands=(1, 1, 1))
            after_building = len([r for r in records if "rgb_composite" in str(r)])
            figure = built.figure_spec
            WebMap()._renderer.draw_layer(figure, figure.layers.ids[0])
            after_redrawing = len([r for r in records if "rgb_composite" in str(r)])
        finally:
            logger.remove(sink_id)
        assert (after_building, after_redrawing) == (1, 1), records

    def test_an_input_with_no_integer_grid_is_not_guessed_at(self, monkeypatch):
        """A size that cannot be measured must produce no number and no warning.

        Args:
            monkeypatch: pytest's patcher, used to lower the ceiling to one pixel.

        Test scenario:
            The composite builder reads the grid rather than a band, because reading a band only to
            measure it costs a whole band read. Anything that does not report `rows`/`columns` as plain
            integers therefore has no measurable size — and a fallback that guessed one would warn about
            a page weight nobody measured. With the ceiling at one pixel, any guess at all warns.
        """
        from loguru import logger

        from digitalearth.web import raster as raster_module

        monkeypatch.setattr(raster_module, "_LARGE_RASTER_PIXELS", 1)
        unmeasurable = type("Grid", (), {"rows": "many", "columns": 3})()
        records, sink_id = self._capture()
        try:
            measured = raster_module._grid_pixels(unmeasurable)
            raster_module._warn_if_large("rgb_composite", "composite", measured)
        finally:
            logger.remove(sink_id)
        assert measured is None, (
            f"an unmeasurable grid must answer None; got {measured}"
        )
        assert records == [], (
            f"nothing measurable means nothing to warn about: {records}"
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

        def spy(unit_stack):
            """Record the stack handed to the encoder, then encode it normally."""
            captured["stack"] = np.array(unit_stack, copy=True)
            return encoder(unit_stack)

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
