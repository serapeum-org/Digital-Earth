"""DW.1 — web-tier raster builder (``add_raster``) + basemap/tiles encoding helpers.

The PNG/coordinate encoding lives in pure helpers (``_rgba_png_datauri`` / ``_image_coordinates``) tested
without the engine (numpy/matplotlib are core deps via cleopatra). ``add_raster`` itself ``importorskip``s
maplibre and exercises the reproject → image-source → render → save path on the shared ``dataset`` fixture.
"""

import base64

import numpy as np
import pytest

from digitalearth.base.crs import OffLimbError
from digitalearth.web import WebMap

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _Grid:
    """A stand-in dataset that reports nothing but its column count.

    ``_zoom_range`` derives a range by reading ``columns`` off whatever it is handed, so the derivation can
    be exercised without opening a raster of each resolution it is compared at.
    """

    def __init__(self, columns: int) -> None:
        """Hold the column count the stand-in reports.

        Args:
            columns: How many columns it claims to have.
        """
        self.columns = columns


class _OutOfBoundsReport(Exception):
    """Stands in for pyramids' out-of-bounds report, which the tier matches by **type name**.

    ``_off_tile`` deliberately neither imports the real exception nor searches its message, so a stand-in
    named like it is exactly what the matching is written against.
    """


class _TileReader:
    """A stand-in dataset whose ``read_tile`` answers — or raises — whatever a test hands it.

    The windowed tile reads are the one part of the tiled route a small in-memory raster cannot exercise:
    every tile of one meets the raster, so neither the missed-tile answer nor the re-raise of a read that
    really broke is ever reached through a real ``Dataset``.
    """

    def __init__(self, answer, *, no_data_value=None):
        """Hold what the reader answers, and the sentinels the dataset reports.

        Args:
            answer: The array ``read_tile`` returns, or an exception instance for it to raise.
            no_data_value: The per-band sentinels, or ``None`` for a dataset that reports none.
        """
        self._answer = answer
        self.no_data_value = no_data_value

    def read_tile(self, zoom, x, y, *, tilesize, band=None):
        """Answer one windowed tile read.

        Args:
            zoom: Unused — the stand-in answers the same window at every level.
            x: Unused — the tile column.
            y: Unused — the tile row.
            tilesize: Unused — the tile edge the caller asks for.
            band: Unused — the zero-based band a single-band read names.

        Returns:
            Whatever the test handed the constructor.

        Raises:
            BaseException: the exception the test handed the constructor, when it handed one.
        """
        if isinstance(self._answer, BaseException):
            raise self._answer
        return self._answer


class TestRgbaPngDataUri:
    """``_rgba_png_datauri`` colour-maps a 2-D array to a transparent-NoData PNG data-URI (no engine)."""

    def test_returns_a_png_data_uri(self):
        arr = np.arange(12.0).reshape(3, 4)
        uri = WebMap()._rgba_png_datauri(arr, "viridis")
        assert uri.startswith("data:image/png;base64,"), "must be a PNG data-URI"
        raw = base64.b64decode(uri.split(",", 1)[1])
        assert raw[:8] == _PNG_MAGIC, "decoded payload must be a real PNG"

    def test_nodata_is_transparent(self):
        """A masked / NaN cell becomes fully transparent in the encoded RGBA PNG.

        Decoded via matplotlib's PNG reader (a core dep) so the test needs no extra image library.
        """
        import io

        from matplotlib import image as mpimage

        arr = np.ma.masked_array(
            np.arange(12.0).reshape(3, 4), mask=np.zeros((3, 4), dtype=bool)
        )
        arr.mask[0, 0] = True
        uri = WebMap()._rgba_png_datauri(arr, "viridis")
        rgba = mpimage.imread(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        assert rgba[0, 0, 3] == 0.0, "masked cell must be transparent"
        assert rgba[2, 3, 3] > 0.0, "a valid cell must be opaque"

    def test_all_nonfinite_raises(self):
        full = np.full((2, 2), np.nan)
        webMap = WebMap()
        with pytest.raises(ValueError, match="no finite values"):
            webMap._rgba_png_datauri(full, "viridis")

    def test_constant_band_does_not_crash(self):
        uri = WebMap()._rgba_png_datauri(np.full((2, 2), 5.0), "viridis")
        assert uri.startswith("data:image/png;base64,")


class TestImageCoordinates:
    """``_image_coordinates`` returns the MapLibre image corners ``[TL, TR, BR, BL]`` in ``[lng, lat]``."""

    def test_corner_order(self):
        """The corners run top-left, top-right, bottom-right, bottom-left, around the cells.

        Test scenario:
            Since #301 the corners are the edges of the rectangle the cells cover, not their centres: three
            one-degree cells centred at 10/11/12 span 9.5 to 12.5, so an image source is no longer drawn half
            a cell inside its own data on every side.
        """
        x = np.array([10.0, 11.0, 12.0])
        y = np.array([50.0, 51.0, 52.0])
        corners = WebMap()._image_coordinates(x, y)
        assert corners == [[9.5, 52.5], [12.5, 52.5], [12.5, 49.5], [9.5, 49.5]], (
            corners
        )


class TestAddRasterNeedsEngine:
    """``add_raster`` reprojects through pyramids and builds an image source (engine required)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_add_raster_registers_layer_and_renders(self, dataset):
        from maplibre.ipywidget import MapWidget

        m = WebMap().add_raster(dataset)
        assert len(m.layers) == 1, "add_raster should register one layer"
        assert m._last_layer_id is not None
        assert isinstance(m.render(), MapWidget)

    def test_add_raster_then_basemap_saves(self, tmp_path, dataset):
        out = tmp_path / "raster.html"
        WebMap().add_raster(dataset, opacity=0.7).basemap("CartoLight").save(str(out))
        assert out.stat().st_size > 1_000


def _source_with(y_values, values):
    """Build a stand-in display source exposing the coordinate and value arrays `add_raster` reads.

    Args:
        y_values: The y coordinates, ascending or descending.
        values: The 2-D band array, in the row order the source reports.

    Returns:
        An object with `.x.values`, `.y.values`, `.z.values` and the `metadata`/`units` accessors
        `auto_style` reads — a real `Source` has all five, and the builder now consults the style
        library for the band's units as well as its colormap.
    """
    axis = type("Axis", (), {})
    x, y, z = axis(), axis(), axis()
    x.values = np.array([0.0, 1.0])
    y.values = np.asarray(y_values, dtype=float)
    z.values = np.asarray(values, dtype=float)
    return type(
        "Source",
        (),
        {
            "x": x,
            "y": y,
            "z": z,
            "units": None,
            "metadata": staticmethod(lambda key=None, default=None: None),
        },
    )()


class TestAddRasterDrawsNorthFirst:
    """PNG row 0 is the northern edge, so a source whose rows run south-first has to be flipped.

    `rgb_composite` has this test; `add_raster` — the older and far more used builder — did not, and an
    upside-down overlay is silently plausible: it still lines up with its bounding box.
    """

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    @pytest.mark.parametrize(
        "y_values, flipped", [([1.0, 0.0], False), ([0.0, 1.0], True)]
    )
    def test_rows_are_ordered_north_first(
        self, dataset, monkeypatch, y_values, flipped
    ):
        """An ascending y means row 0 is the south edge, which draws the image upside down.

        Args:
            dataset: The shared pyramids raster fixture.
            monkeypatch: pytest's patcher, used to hand the builder a source with a chosen y order.
            y_values: The coordinate order the source reports.
            flipped: Whether the band should be reversed before encoding.

        Test scenario:
            The pixels handed to the encoder are compared against the orientation that *should* be drawn,
            so the test fails both when the flip is missing and when it happens for a north-up source.
        """
        band = np.array([[1.0, 2.0], [3.0, 4.0]])
        source = _source_with(y_values, band)
        monkeypatch.setattr(
            WebMap, "_to_display_source", lambda self, data, band=1: source
        )

        captured = {}
        encoder = WebMap._rgba_png_datauri

        def spy(values, cmap_name, vmin=None, vmax=None):
            """Record the array handed to the encoder, then encode it normally."""
            captured["values"] = np.array(values, copy=True)
            return encoder(values, cmap_name, vmin=vmin, vmax=vmax)

        monkeypatch.setattr(WebMap, "_rgba_png_datauri", staticmethod(spy))
        WebMap().basemap().add_raster(dataset, cmap="viridis")

        expected = band[::-1] if flipped else band
        assert np.allclose(captured["values"], expected), (
            f"rows are not north-first for y={y_values}"
        )


class TestARasterThatCannotBeGeoreferencedIsRefused:
    """A MapLibre image source is positioned by four lon/lat corners; there is no drawing it without them."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip when the web extra is absent."""
        pytest.importorskip("maplibre")

    @staticmethod
    def _call(m, builder, dataset):
        """Run `builder` on `m`, so the two parametrised cases read the same either side of the guard.

        Args:
            m: The map under test.
            builder: The raster builder name.
            dataset: The shared pyramids raster fixture.

        Returns:
            Whatever the builder returns (the map itself, since both are fluent).
        """
        return {
            "add_raster": lambda: m.add_raster(dataset),
            "rgb_composite": lambda: m.rgb_composite(dataset, bands=(1, 1, 1)),
        }[builder]()

    @pytest.mark.parametrize("builder", ["add_raster", "rgb_composite"])
    def test_unplaceable_corners_skip_the_layer_and_warn(
        self, dataset, monkeypatch, builder, warning_log
    ):
        """Falling back to the projected numbers would place the image somewhere off the planet.

        Args:
            dataset: The shared pyramids raster fixture.
            monkeypatch: pytest's patcher.
            builder: The raster builder under test.
            warning_log: The tier's loguru warnings, proving the skip is announced rather than silent.

        Test scenario:
            `_as_lonlat` answers `None` for a CRS that cannot be resolved or a reprojection that comes
            back non-finite; both builders must stop there — adding no layer, saying why, and leaving the
            rest of the chain to draw (C7).
        """
        monkeypatch.setattr(WebMap, "_as_lonlat", lambda self, *bounds: None)
        m = WebMap().basemap()
        before = len(m.layers)
        assert self._call(m, builder, dataset) is m, "the builder must stay chainable"
        assert len(m.layers) == before, "the unplaceable raster was added anyway"
        assert any("cannot be expressed in lon/lat" in line for line in warning_log), (
            f"the skip has to name why the layer is missing; got {warning_log!r}"
        )

    @pytest.mark.parametrize("builder", ["add_raster", "rgb_composite"])
    def test_strict_raises_instead_of_skipping(self, dataset, monkeypatch, builder):
        """`strict=True` is for a pipeline that must not publish a map with a layer quietly missing.

        Args:
            dataset: The shared pyramids raster fixture.
            monkeypatch: pytest's patcher.
            builder: The raster builder under test.
        """
        monkeypatch.setattr(WebMap, "_as_lonlat", lambda self, *bounds: None)
        m = WebMap(strict=True).basemap()
        # OffLimbError, not a bare ValueError: one exception type across the four tiers (M6).
        with pytest.raises(OffLimbError, match="cannot be expressed in lon/lat"):
            self._call(m, builder, dataset)


class TestTheContractsColourLimits:
    """#299 / review L12 — `limits=` is the spelling every tier answers to."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip the class when the web extra is not installed."""
        pytest.importorskip("maplibre")

    def test_limits_colours_the_same_band_as_vmin_and_vmax(self, dataset):
        """The contract's name and this tier's own reach the same colouring.

        Args:
            dataset: The raster fixture.
        """
        by_pair = WebMap().field(dataset, limits=(0.0, 10.0)).to_html()
        by_ends = WebMap().field(dataset, vmin=0.0, vmax=10.0).to_html()
        other = WebMap().field(dataset, limits=(0.0, 50.0)).to_html()
        assert by_pair == by_ends, "the two spellings must colour the same raster alike"
        assert by_pair != other, "and different limits must colour it differently"

    def test_naming_the_limits_twice_is_refused(self, dataset):
        """They are one thing, and there is no right answer to being given two.

        Args:
            dataset: The raster fixture.
        """
        m = WebMap()
        with pytest.raises(ValueError, match="takes limits= or vmin=/vmax=, not both"):
            m.field(dataset, limits=(0.0, 10.0), vmin=1.0)

    @pytest.mark.parametrize("given", [10.0, "ab", (0.0, 1.0, 2.0), ("a", "b")])
    def test_limits_that_are_not_a_pair_of_numbers_name_the_shape(self, dataset, given):
        """A pair of numbers, checked as one — several plausible mistakes unpack just as happily.

        Args:
            dataset: The raster fixture.
            given: What the caller passed.

        Test scenario:
            `low, high = limits` accepts a two-character string and a two-element iterator, and a triple
            silently loses its third value; `"ab"` reached numpy and came back as *"could not convert
            string to float: 'a'"* (review L9/L11).
        """
        m = WebMap()
        with pytest.raises(ValueError, match=r"limits must be a \(vmin, vmax\) pair"):
            m.field(dataset, limits=given)


class TestAColormapObjectIsAcceptedWhereItsNameIs:
    """Review H3 — `field(cmap=)` takes a `Colormap`, as the classified builders already do (#315)."""

    def test_the_encoder_takes_the_object_a_name_resolves_to(self):
        """A registered name and the object it names colour one band identically.

        Test scenario:
            The two sides are built differently on purpose: one hands the encoder the string a keyword
            argument usually carries, the other the object `colormaps[...]` hands back. Before the fix the
            object side raised `TypeError: unhashable type` from `colormaps[cmap]`, because a `Colormap`
            defines `__eq__` and so cannot be a dict key.
        """
        from matplotlib import colormaps

        band = np.arange(12.0).reshape(3, 4)
        by_name = WebMap()._rgba_png_datauri(band, "magma")
        by_object = WebMap()._rgba_png_datauri(band, colormaps["magma"])
        assert by_object == by_name, "the object and its name must encode the same PNG"

    def test_a_homemade_ramp_colours_the_band_it_was_given(self):
        """A ramp built on the spot has no registered name, and must still colour the image.

        Test scenario:
            The three-colour ramp is deliberately unlike any registered map, so the first cell decodes to
            pure red — which proves the caller's own object reached the encoder rather than a default.
        """
        import io

        from matplotlib import image as mpimage
        from matplotlib.colors import ListedColormap

        ramp = ListedColormap(["#ff0000", "#00ff00", "#0000ff"], name="homemade-ramp")
        uri = WebMap()._rgba_png_datauri(np.arange(12.0).reshape(3, 4), ramp)
        rgba = mpimage.imread(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        assert tuple(rgba[0, 0, :3]) == (1.0, 0.0, 0.0), "the lowest cell must be red"

    def test_the_builder_takes_the_object_too(self, dataset):
        """`field(cmap=<Colormap>)` builds a layer, as `choropleth(cmap=<Colormap>)` already does.

        Args:
            dataset: The raster fixture.

        Test scenario:
            This is the asymmetry the finding names: the classified paths resolve a colormap through
            `as_colormap`, the unclassified raster path looked it up as a dict key and raised.
        """
        pytest.importorskip("maplibre")
        from matplotlib.colors import ListedColormap

        ramp = ListedColormap(["#ff0000", "#00ff00", "#0000ff"], name="homemade-ramp")
        built = WebMap().field(dataset, cmap=ramp, name="dem")
        assert built.layer_ids == ["dem"], "the layer must be registered"


class TestTheInlineCeilingIsARefusal:
    """#189 — above the inline ceiling a page is hundreds of MB, which a warning does not prevent."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip the class when the web extra is absent."""
        pytest.importorskip("maplibre")

    def test_an_oversized_band_is_refused_and_both_routes_are_named(
        self, dataset, monkeypatch
    ):
        """A warning let a national-scale raster inline itself; the refusal has to offer a way out.

        Args:
            dataset: The shared pyramids raster fixture.
            monkeypatch: pytest's patcher, used to lower the ceiling rather than build a huge raster.

        Test scenario:
            4 M pixels is a 2000x2000 raster, so anything national-scale was warned about and then inlined
            anyway. The refusal names both tiled routes, because a caller who is refused has to be told
            which parameter to reach for.
        """
        from digitalearth.web import raster as raster_module

        monkeypatch.setattr(raster_module, "_LARGE_RASTER_PIXELS", 1)
        m = WebMap()
        with pytest.raises(ValueError, match=r"tiles=") as refusal:
            m.field(dataset)
        message = str(refusal.value)
        assert '"xyz"' in message, message
        assert '"cog"' in message, message

    def test_a_band_under_the_ceiling_still_inlines(self, dataset):
        """Every map built so far embeds its pixels, and none of them may move.

        Args:
            dataset: The shared pyramids raster fixture, far below the ceiling.
        """
        m = WebMap().field(dataset)
        spec = m._renderer.drawn[m._last_layer_id].source_spec
        assert spec["type"] == "image", spec

    def test_a_tiled_route_is_not_refused_above_the_ceiling(
        self, dataset, tmp_path, monkeypatch
    ):
        """The ceiling is the inline path's, so naming a route has to clear it.

        Args:
            dataset: The shared pyramids raster fixture.
            tmp_path: pytest's temporary directory, where the pyramid is written.
            monkeypatch: pytest's patcher, used to lower the ceiling.
        """
        from digitalearth.web import raster as raster_module

        monkeypatch.setattr(raster_module, "_LARGE_RASTER_PIXELS", 1)
        m = WebMap().field(
            dataset, tiles="xyz", tiles_path=tmp_path / "acc", zooms=(9, 9)
        )
        assert m.layer_ids, "a tiled raster must still register its layer"


class TestTheTiledRasterRoutes:
    """#189 — ``tiles=`` writes the pixels beside the page instead of inside it."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        """Skip the class when the web extra is absent."""
        pytest.importorskip("maplibre")

    def test_an_unknown_route_is_refused_by_name(self, dataset, tmp_path):
        """A misspelled route must not fall back to inlining a raster the caller asked to tile.

        Args:
            dataset: The shared pyramids raster fixture.
            tmp_path: pytest's temporary directory.
        """
        m = WebMap()
        with pytest.raises(ValueError, match="pmtiles"):
            m.field(dataset, tiles="pmtiles", tiles_path=tmp_path / "out")

    def test_a_route_with_no_destination_is_refused(self, dataset):
        """A tiled route writes files, so it has to be told where.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        m = WebMap()
        with pytest.raises(ValueError, match="tiles_path"):
            m.field(dataset, tiles="xyz")

    def test_xyz_writes_a_pyramid_the_widget_reads_as_a_raster_source(
        self, dataset, tmp_path
    ):
        """The widget is built from the source spec, so that is what a tiled route has to change.

        Args:
            dataset: The shared pyramids raster fixture.
            tmp_path: pytest's temporary directory, where the pyramid is written.

        Test scenario:
            The figure recording ``tiles="xyz"`` proves nothing — this tier has shipped values recorded and
            never drawn. What the map draws is the ``source_spec`` the renderer holds, and the pyramid itself
            has to be on disk and decode as a PNG.
        """
        import io

        from matplotlib import image as mpimage

        m = WebMap().field(
            dataset, tiles="xyz", tiles_path=tmp_path / "acc", zooms=(9, 9)
        )
        spec = m._renderer.drawn[m._last_layer_id].source_spec
        assert spec["type"] == "raster", spec
        assert spec["tiles"] == ["acc/{z}/{x}/{y}.png"], spec
        written = sorted((tmp_path / "acc").rglob("*.png"))
        assert written, "the route has to leave a pyramid behind"
        tile = mpimage.imread(io.BytesIO(written[0].read_bytes()))
        assert tile.shape == (256, 256, 4), tile.shape

    def test_the_template_carries_no_local_path_into_the_page(self, dataset, tmp_path):
        """A saved page is shared, and an absolute path names the machine that made it.

        Args:
            dataset: The shared pyramids raster fixture.
            tmp_path: pytest's temporary directory, whose own name must not reach the page.
        """
        out = tmp_path / "tiled.html"
        page = (
            WebMap()
            .field(dataset, tiles="xyz", tiles_path=tmp_path / "acc", zooms=(9, 9))
            .save(str(out))
            .read_text("utf-8")
        )
        assert "acc/{z}/{x}/{y}.png" in page, (
            "the page must address the pyramid it sits beside"
        )
        assert str(tmp_path) not in page, (
            "an absolute path names the machine that made the page"
        )

    def test_cog_writes_a_cloud_optimized_geotiff_the_page_addresses(
        self, dataset, tmp_path
    ):
        """The COG route hands the browser a file to read windows from, through a ``cog://`` protocol.

        Args:
            dataset: The shared pyramids raster fixture.
            tmp_path: pytest's temporary directory, where the COG is written.
        """
        from pyramids.dataset import Dataset

        m = WebMap().field(dataset, tiles="cog", tiles_path=tmp_path / "acc.tif")
        spec = m._renderer.drawn[m._last_layer_id].source_spec
        assert spec["url"] == "cog://acc.tif", spec
        assert Dataset.read_file(str(tmp_path / "acc.tif")).is_cog, "must be a real COG"

    def test_only_the_inline_path_builds_a_display_source_of_the_whole_band(
        self, dataset, tmp_path, monkeypatch
    ):
        """Materialising the warped band is the cost a tiled route exists to avoid.

        Args:
            dataset: The shared pyramids raster fixture.
            tmp_path: pytest's temporary directory.
            monkeypatch: pytest's patcher, which restores the spied method.

        Test scenario:
            The two paths are counted through the same spy, so the contrast is measured rather than asserted
            of one side alone: the inline path goes through the tier's display choke point — a warp plus a
            read of every cell — and the tiled path must not reach it at all.
        """
        from digitalearth.web.base import WebMapBase

        visits = []
        original = WebMapBase._display_source_or_skip

        def counted(self, data, **kwargs):
            """Record one display-source build, then perform it.

            Args:
                self: The map being built.
                data: The raster being read.
                **kwargs: The band and layer name, passed on.

            Returns:
                Whatever the real choke point answers.
            """
            visits.append(kwargs.get("layer"))
            return original(self, data, **kwargs)

        monkeypatch.setattr(WebMapBase, "_display_source_or_skip", counted)
        WebMap().field(dataset)
        # Twice: the builder reads the band to colour-limit and style it, and the drawer reads it again to
        # encode the image — both through the choke point, and both on the whole warped band.
        assert visits == ["field", "field"], visits
        WebMap().field(dataset, tiles="xyz", tiles_path=tmp_path / "acc", zooms=(9, 9))
        assert visits == ["field", "field"], (
            f"the tiled route built a display source of the whole band: {visits}"
        )

    def test_the_colour_scan_reads_no_more_than_its_budget(
        self, dataset, tmp_path, monkeypatch
    ):
        """The limits every tile shares are measured from a decimated read, not from the whole band.

        Args:
            dataset: The shared pyramids raster fixture.
            tmp_path: pytest's temporary directory.
            monkeypatch: pytest's patcher, used to lower the scan budget below the fixture's own size.

        Test scenario:
            With the budget under the raster's 13x14 cells the scan has to decimate, which is the behaviour a
            national-scale raster relies on; at the real budget a raster this small is simply read whole.
        """
        import numpy as np

        from digitalearth.web import raster as raster_module

        monkeypatch.setattr(raster_module, "_LIMIT_SCAN_BUDGET", 16)
        reads = []
        original = type(dataset).read_part

        def counted(self, *args, **kwargs):
            """Record how many cells one windowed read returned.

            Args:
                self: The dataset being read.
                *args: Positional arguments, passed on.
                **kwargs: Keyword arguments, passed on.

            Returns:
                Whatever the real reader answers.
            """
            result = original(self, *args, **kwargs)
            reads.append(int(np.asarray(result).size))
            return result

        monkeypatch.setattr(type(dataset), "read_part", counted)
        WebMap().field(dataset, tiles="xyz", tiles_path=tmp_path / "acc", zooms=(9, 9))
        assert reads, "the scan has to reach the windowed reader"
        assert max(reads) <= 16, f"the scan read {max(reads)} cells for a budget of 16"

    def test_the_map_frames_itself_on_the_tiled_extent(self, dataset, tmp_path):
        """A tiled raster still has to hand the map an extent, or the page opens on the whole world.

        Args:
            dataset: The shared pyramids raster fixture.
            tmp_path: pytest's temporary directory.
        """
        m = WebMap().field(
            dataset, tiles="xyz", tiles_path=tmp_path / "acc", zooms=(9, 9)
        )
        assert m.set_bounds() is m, "the tiled route must leave an extent to frame on"

    def test_a_tiled_layer_survives_a_figure_written_as_json(self, tmp_path):
        """A description is the other way into the drawer, so the tile props have to be writable.

        Args:
            tmp_path: pytest's temporary directory, holding the pyramid the figure points at.

        Test scenario:
            The tiled route is described by a reference rather than by pixels, which is what lets a figure
            carry it at all — but only if every prop has a JSON form. A tuple of bounds or a NaN would refuse
            the whole figure, and the layer would come back undrawable on the far side.
        """
        import json

        from digitalearth.base.spec import FigureSpec

        built = WebMap().field(
            "examples/data/acc4000.tif",
            tiles="xyz",
            tiles_path=tmp_path / "acc",
            zooms=(9, 9),
            name="acc",
        )
        reloaded = FigureSpec.from_dict(
            json.loads(json.dumps(built.figure_spec.to_dict()))
        )
        drawn = WebMap()._renderer.draw_layer(reloaded, "acc")
        assert drawn.source_spec["tiles"] == ["acc/{z}/{x}/{y}.png"], drawn.source_spec

    def test_a_description_recording_no_extent_is_still_drawn(self):
        """A description is the other way in, and one written without an extent has to draw without it.

        Test scenario:
            ``tiles_bounds`` is what lets a tiled layer frame the map on pixels it never reads, and the
            prop is optional on the way in — a description built by hand, or written before the route
            recorded an extent, carries none. The source then simply has no ``bounds`` key, which is how
            MapLibre reads "no limit", rather than the drawer raising on the absent prop.
        """
        from digitalearth.base.spec import LayerSpec, Symbology
        from digitalearth.web import raster as web_raster

        described = LayerSpec(
            "acc",
            "raster",
            symbology=Symbology(
                props={
                    "band": 1,
                    "cmap": "viridis",
                    "vmin": 0.0,
                    "vmax": 1.0,
                    "opacity": 1.0,
                    "tiles": "xyz",
                    "tiles_url": "acc/{z}/{x}/{y}.png",
                }
            ),
        )
        drawn = web_raster.draw_field(WebMap(), None, described)
        assert drawn.source_spec["tiles"] == ["acc/{z}/{x}/{y}.png"], drawn.source_spec
        assert "bounds" not in drawn.source_spec, drawn.source_spec

    def test_a_declined_tiled_layer_is_not_the_maps_last_layer(
        self, dataset, tmp_path, monkeypatch
    ):
        """The description is written before the drawer runs, so a decline has to take the record back.

        Args:
            dataset: The shared pyramids raster fixture.
            tmp_path: pytest's temporary directory, where the pyramid is written.
            monkeypatch: Used to make the raster drawer decline, which the tiled route's own never does.

        Test scenario:
            The tiled drawer places a reference rather than pixels, so nothing in it can decline — but the
            builder shares the registration check with the inline path, where an unplaceable raster can. A
            declined layer that still became ``_last_layer_id`` would have a later ``colorbar()`` label
            itself from a layer no widget holds.
        """
        from digitalearth.web import raster as web_raster

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

        m = WebMap().field(dataset, name="inline")
        monkeypatch.setattr(web_raster, "draw_field", _declines)
        m.field(
            dataset,
            tiles="xyz",
            tiles_path=tmp_path / "acc",
            zooms=(9, 9),
            name="tiled",
        )
        assert m.layer_ids == ["inline"], m.layer_ids
        assert m._last_layer_id == "inline", (
            f"a declined tiled layer became the map's last layer: {m._last_layer_id!r}"
        )


class TestTheTileAddressing:
    """The slippy-map arithmetic a ``{z}/{x}/{y}`` template is written in (no engine needed)."""

    def test_zoom_zero_is_one_tile_for_the_whole_world(self):
        """At zoom 0 every coordinate is in tile ``(0, 0)``, which is what the level means."""
        from digitalearth.web.raster import _tile_index

        assert _tile_index(-179.0, 84.0, 0) == (0, 0)

    def test_the_prime_meridian_splits_the_grid_in_half(self):
        """Longitude 0 opens the eastern half, and the degree before it closes the western one.

        Test scenario:
            Built from both sides rather than asserted of one, so an off-by-one in the scale factor shows up:
            at zoom 1 the world is two columns, and ``-0.001`` and ``0.0`` must not land in the same one.
        """
        from digitalearth.web.raster import _tile_index

        west = _tile_index(-0.001, 0.0, 1)[0]
        east = _tile_index(0.0, 0.0, 1)[0]
        assert (west, east) == (0, 1), (west, east)

    def test_a_latitude_past_web_mercator_is_clamped_into_the_grid(self):
        """Mercator runs to infinity at the pole, so the row has to be clamped rather than computed.

        Test scenario:
            Without the clamp ``tan(pi/2)`` overflows and the row indexes off the end of the level.
        """
        from digitalearth.web.raster import _tile_index

        assert _tile_index(0.0, 90.0, 3)[1] == 0

    def test_a_tile_range_covers_the_extent_north_west_first(self):
        """The addresses a pyramid is written in run across then down, from the north-west corner.

        Test scenario:
            A box spanning two columns at zoom 1 yields both, in west-to-east order.
        """
        from digitalearth.base.spec import Bounds
        from digitalearth.web.raster import _tile_addresses

        box = Bounds(-10.0, -10.0, 10.0, 10.0, crs=4326)
        assert list(_tile_addresses(box, 1)) == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_the_native_zoom_grows_with_the_resolution(self):
        """A finer raster over the same extent needs more zoom levels, not the same number.

        Test scenario:
            Two column counts over one extent, compared against each other — a constant returned for both
            would look right against any single expected number.
        """
        from digitalearth.base.spec import Bounds
        from digitalearth.web.raster import _native_zoom

        box = Bounds(0.0, 0.0, 1.0, 1.0, crs=4326)
        assert _native_zoom(box, 256) < _native_zoom(box, 4096)

    def test_an_unmeasurable_grid_asks_for_no_zoom_at_all(self):
        """A column count that is not a number has no native zoom to derive, and none is guessed."""
        from digitalearth.base.spec import Bounds
        from digitalearth.web.raster import _native_zoom

        assert _native_zoom(Bounds(0.0, 0.0, 1.0, 1.0, crs=4326), "many") == 0

    @pytest.mark.parametrize(
        "west, east",
        [(1.0, 1.0), (-1.5e308, 1.5e308)],
        ids=["no-width", "wider-than-a-float"],
    )
    def test_an_extent_with_no_measurable_width_asks_for_no_zoom_at_all(
        self, west, east
    ):
        """A width of zero has no cell size to match, and one that overflows has none either.

        Args:
            west: The extent's western edge.
            east: Its eastern edge.

        Test scenario:
            ``Bounds`` guarantees each edge is finite, but not that their *difference* is: an extent from
            -1.5e308 to 1.5e308 is two finite numbers whose span is infinity, which measured gives a cell
            size of infinity and asks for ``log2(0)``. Both ends of the guard answer zero instead of
            raising from inside the arithmetic.
        """
        from digitalearth.base.spec import Bounds
        from digitalearth.web.raster import _native_zoom

        box = Bounds(west, 0.0, east, 1.0, crs=4326)
        assert _native_zoom(box, 256) == 0, (
            "an unmeasurable width must ask for no tiles"
        )

    def test_a_derived_range_opens_at_the_world_and_closes_at_the_native_zoom(self):
        """``zooms=None`` derives a range, and its low levels are what make a tiled raster findable.

        Test scenario:
            The derived range is compared against a *finer* grid's rather than against a literal pair — a
            derivation returning one constant range would satisfy any single expected pair. Both must open
            at zoom 0, one tile for the whole world, which is what a page zoomed out shows; and the finer
            grid must close later than the coarser one.
        """
        from digitalearth.base.spec import Bounds
        from digitalearth.web.raster import _zoom_range

        box = Bounds(0.0, 0.0, 1.0, 1.0, crs=4326)
        coarse = _zoom_range(box, _Grid(256), None, "field()")
        fine = _zoom_range(box, _Grid(4096), None, "field()")
        assert coarse[0] == 0, f"a derived range must open at the world; got {coarse}"
        assert fine[0] == 0, f"a derived range must open at the world; got {fine}"
        assert coarse[1] < fine[1], (
            f"a finer grid must close at a higher zoom: {coarse} against {fine}"
        )

    @pytest.mark.parametrize("given", [(3, 1), (-1, 2), (1, 2, 3), 4, (1.5, 2.0)])
    def test_a_range_that_writes_nothing_is_refused(self, given):
        """A reversed or malformed range would write an empty pyramid and look like a silent failure.

        Args:
            given: A ``zooms=`` value no pyramid can be written for.
        """
        from digitalearth.base.spec import Bounds
        from digitalearth.web.raster import _zoom_range

        box = Bounds(0.0, 0.0, 1.0, 1.0, crs=4326)
        with pytest.raises(ValueError, match="zooms"):
            _zoom_range(box, None, given, "field()")


class TestTheSharedTileEncoder:
    """One encoder colours the inlined image and every written tile, so both obey the same limits."""

    def test_explicit_limits_win_over_the_values_own_range(self):
        """A tile normalised over its own range disagrees with its neighbour at their shared edge.

        Test scenario:
            The same two cells encoded against two different explicit spans must not produce the same bytes;
            if the limits were ignored, both would be the tile's own 0-to-1 stretch.
        """
        import numpy as np

        from digitalearth.web.raster import _coloured_png

        cells = np.array([[0.0, 1.0]])
        against_ten = _coloured_png(cells, "viridis", vmin=0.0, vmax=10.0)
        against_one = _coloured_png(cells, "viridis", vmin=0.0, vmax=1.0)
        assert against_ten != against_one, "the explicit limits were not applied"

    def test_a_nodata_sentinel_is_drawn_transparent(self):
        """A windowed tile read hands back the raster's sentinel, not NaN, so it has to be named.

        Test scenario:
            Without the sentinel the NoData padding around an edge tile is coloured as though it were data.
        """
        import io

        import numpy as np
        from matplotlib import image as mpimage

        from digitalearth.web.raster import _coloured_png

        cells = np.array([[-9999.0, 1.0]])
        payload = _coloured_png(cells, "viridis", vmin=0.0, vmax=1.0, nodata=-9999.0)
        rgba = mpimage.imread(io.BytesIO(payload))
        assert rgba[0, 0, 3] == 0.0, "the sentinel must be transparent"
        assert rgba[0, 1, 3] > 0.0, "a real value must be opaque"

    def test_a_tile_with_nothing_in_it_is_not_written(self):
        """An extent's bounding box is not the extent, so a pyramid legitimately has holes."""
        import numpy as np

        from digitalearth.web.raster import _coloured_png

        assert _coloured_png(np.full((2, 2), np.nan), "viridis") is None


class TestTheColourLimitScan:
    """One pair of limits colours a whole pyramid, so where that pair comes from is what this checks."""

    def test_limits_the_caller_named_are_used_without_reading_the_band(self):
        """Reading the band to measure what the caller already said is the cost the route exists to avoid.

        Test scenario:
            The dataset handed in has no reader at all, so a scan that ran would raise ``AttributeError``
            rather than quietly cost a windowed read. What comes back is the caller's own pair, as the
            floats every tile is then normalised on.
        """
        from digitalearth.web.raster import _scan_limits

        measured = _scan_limits(object(), 1, vmin=-2.0, vmax=8.0)
        assert measured == (-2.0, 8.0), measured


class TestTheWrittenPyramid:
    """A pyramid is written tile by tile, and a tile the raster does not reach is left out of it."""

    def test_a_tile_with_nothing_in_it_is_left_out_rather_than_written_blank(
        self, tmp_path
    ):
        """An extent's bounding box is not the extent, so a pyramid legitimately has holes.

        Args:
            tmp_path: pytest's temporary directory, where the pyramid is written.

        Test scenario:
            Zoom 1 over a box straddling both the equator and the prime meridian has four addresses. An
            encoder answering for the western column only must leave two of them on disk and none at all
            for the other two — a blank tile written for a hole draws an opaque square over the basemap.
        """
        from digitalearth.base.spec import Bounds
        from digitalearth.web.raster import _written_pyramid

        def _western_only(zoom, x, y):
            """Colour the western column of the level and nothing else.

            Args:
                zoom: Unused — the level being written.
                x: The tile column, which decides whether this tile has anything in it.
                y: Unused — the tile row.

            Returns:
                Some bytes for a western tile, ``None`` for an eastern one.
            """
            return b"tile" if x == 0 else None

        box = Bounds(-10.0, -10.0, 10.0, 10.0, crs=4326)
        root = tmp_path / "pyramid"
        written = _written_pyramid(
            root, _western_only, bounds=box, zooms=(1, 1), caller="field()"
        )
        assert written == 2, (
            f"two of the level's four addresses have values; got {written}"
        )
        on_disk = sorted(
            path.relative_to(root).as_posix() for path in root.rglob("*.png")
        )
        assert on_disk == ["1/0/0.png", "1/0/1.png"], on_disk

    def test_a_range_that_writes_no_tile_at_all_is_refused(self, tmp_path):
        """An empty directory beside a page that points into it is a map that draws nothing.

        Args:
            tmp_path: pytest's temporary directory.

        Test scenario:
            The complement of the skip above: a skip is normal, and a range where *every* address skips is
            not, because it has no other symptom than a page that renders blank.
        """
        from digitalearth.base.spec import Bounds
        from digitalearth.web.raster import _written_pyramid

        def _nothing_anywhere(zoom, x, y):
            """Answer for no tile at all.

            Args:
                zoom: Unused — the level being written.
                x: Unused — the tile column.
                y: Unused — the tile row.

            Returns:
                ``None``, always.
            """
            return None

        box = Bounds(-10.0, -10.0, 10.0, 10.0, crs=4326)
        root = tmp_path / "empty"
        with pytest.raises(ValueError, match="produced no tile with any value"):
            _written_pyramid(
                root, _nothing_anywhere, bounds=box, zooms=(1, 1), caller="field()"
            )


class TestTheWindowedTileReads:
    """A tiled route reads one window per tile, and a window off the raster is not a failed read."""

    def test_pyramids_out_of_bounds_report_is_matched_by_its_type_name(self):
        """Matched by type so the tier needs no import of it — and not by message, so a real bug shows.

        Test scenario:
            "out of bounds" is also the text of the commonest indexing bug in Python, so an ``IndexError``
            carrying those very words must not read as a missed tile. The two are built from different
            exception types and checked against opposite answers.
        """
        from digitalearth.web.raster import _off_tile

        assert _off_tile(_OutOfBoundsReport("tile 3/1/1")) is True, (
            "pyramids' own report must read as a missed tile"
        )
        assert _off_tile(IndexError("index 4 is out of bounds")) is False, (
            "an indexing defect must not be swallowed as a missed tile"
        )

    def test_a_band_tile_off_the_raster_answers_none(self):
        """A pyramid legitimately has holes, so a missed window is an answer rather than a failure."""
        from digitalearth.web.raster import _tile_values

        reader = _TileReader(_OutOfBoundsReport("no window there"))
        assert _tile_values(reader, 3, 1, 1, band=1) is None, (
            "a missed tile must answer None rather than raise"
        )

    def test_a_band_read_that_really_broke_is_raised_unchanged(self):
        """Swallowing a real failure here would turn it into a silently absent tile."""
        from digitalearth.web.raster import _tile_values

        reader = _TileReader(MemoryError("the window did not fit"))
        with pytest.raises(MemoryError, match="did not fit"):
            _tile_values(reader, 3, 1, 1, band=1)

    def test_a_composite_tile_off_the_raster_answers_none(self):
        """The three-band read answers a missed window the same way the single-band one does."""
        from digitalearth.web.raster import _tile_stack

        reader = _TileReader(_OutOfBoundsReport("no window there"))
        assert _tile_stack(reader, 3, 1, 1, bands=(1, 2, 3)) is None, (
            "a missed tile must answer None rather than raise"
        )

    def test_a_composite_read_that_really_broke_is_raised_unchanged(self):
        """And it re-raises a real failure the same way, for the same reason."""
        from digitalearth.web.raster import _tile_stack

        reader = _TileReader(MemoryError("the window did not fit"))
        with pytest.raises(MemoryError, match="did not fit"):
            _tile_stack(reader, 3, 1, 1, bands=(1, 2, 3))

    def test_a_multiband_read_is_indexed_by_band_not_by_row(self):
        """pyramids answers a multiband tile band-first, so a band is a whole plane and not a row.

        Test scenario:
            Three 2x2 planes of distinct values, read as bands 1 and 3. Indexed as rows, the first channel
            would carry the first *row* of the first plane; indexed as planes it carries that plane whole,
            and the second channel starts where the third plane does. Each channel is compared against the
            plane it was read from rather than against a literal.
        """
        import numpy as np

        from digitalearth.web.raster import _tile_stack

        planes = np.arange(12.0).reshape(3, 2, 2)
        reader = _TileReader(planes, no_data_value=[-9999.0, -9999.0, -9999.0])
        stack = _tile_stack(reader, 0, 0, 0, bands=(1, 3))
        assert stack.shape == (2, 2, 2), stack.shape
        assert stack[..., 0].tolist() == planes[0].tolist(), stack[..., 0]
        assert stack[..., 1].tolist() == planes[2].tolist(), stack[..., 1]

    def test_a_dataset_reporting_no_sentinel_keeps_every_value_it_read(self):
        """A raster with no NoData has no hole in it, and one must not be invented for it.

        Test scenario:
            The complement of the sentinel path: read through a dataset reporting ``-9999`` that cell would
            come back NaN and be drawn transparent. Reported by a dataset with no sentinel at all, it is a
            value like any other.
        """
        import numpy as np

        from digitalearth.web.raster import _tile_stack

        reader = _TileReader(np.array([[-9999.0, 1.0], [2.0, 3.0]]))
        stack = _tile_stack(reader, 0, 0, 0, bands=(1, 1, 1))
        assert np.isfinite(stack).all(), stack
        assert stack[0, 0, 0] == -9999.0, stack[0, 0, 0]
