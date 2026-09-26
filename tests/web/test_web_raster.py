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
        with pytest.raises(ValueError, match=r"tiles=") as refusal:
            WebMap().field(dataset)
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
        with pytest.raises(ValueError, match="pmtiles"):
            WebMap().field(dataset, tiles="pmtiles", tiles_path=tmp_path / "out")

    def test_a_route_with_no_destination_is_refused(self, dataset):
        """A tiled route writes files, so it has to be told where.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        with pytest.raises(ValueError, match="tiles_path"):
            WebMap().field(dataset, tiles="xyz")

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

    @pytest.mark.parametrize("given", [(3, 1), (-1, 2), (1, 2, 3), 4, (1.5, 2.0)])
    def test_a_range_that_writes_nothing_is_refused(self, given):
        """A reversed or malformed range would write an empty pyramid and look like a silent failure.

        Args:
            given: A ``zooms=`` value no pyramid can be written for.
        """
        from digitalearth.base.spec import Bounds
        from digitalearth.web.raster import _zoom_range

        with pytest.raises(ValueError, match="zooms"):
            _zoom_range(Bounds(0.0, 0.0, 1.0, 1.0, crs=4326), None, given, "field()")


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
