"""DI.7 + DI.8 — tap-to-inspect / hover and draw-AOI / linked selection.

Streams are driven in-process (``stream.event(...)``) — no browser/server. The tap-profile and draw
round-trips are exercised by firing the stream and asserting the callback / captured geometry. Runs in
the ``interactive`` pixi env.
"""

import pytest

from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")


@pytest.fixture()
def m(dataset) -> InteractiveMap:
    """A map carrying one raster layer (a tap/hover source)."""
    return InteractiveMap().image(dataset)


class TestHoverAndTap:
    """``hover`` / ``on_tap`` / ``tap_profile`` (DI.7)."""

    def test_hover_sets_custom_tooltips(self, m):
        m.hover(tooltips=[("value", "@value")])
        plot = hv.Store.lookup_options("bokeh", m.layers[-1], "plot").kwargs
        assert any(
            getattr(t, "tooltips", None) for t in plot.get("tools", [])
        ), "custom HoverTool missing"

    def test_hover_without_layers_raises(self):
        with pytest.raises(ValueError, match="at least one layer"):
            InteractiveMap().hover()

    def test_on_tap_returns_dynamicmap_and_fires_callback(self, m):
        seen = {}

        def _cb(x, y):
            seen["xy"] = (x, y)
            return hv.Points([(x, y)])

        dmap = m.on_tap(_cb)
        assert isinstance(dmap, hv.DynamicMap)
        from holoviews import streams

        tap = next(s for s in dmap.streams if isinstance(s, streams.Tap))
        tap.event(x=12.0, y=34.0)
        dmap[()]
        assert seen["xy"] == (
            12.0,
            34.0,
        ), "tap callback must receive the clicked coords"

    def test_on_tap_without_source_raises(self):
        with pytest.raises(ValueError, match="needs a source layer"):
            InteractiveMap().on_tap(lambda x, y: None)

    def test_tap_profile_reads_collection_at_clicked_cell(self, m, dataset):
        from pyramids.dataset.collection import DatasetCollection

        dc = DatasetCollection.from_files(["examples/data/acc4000.tif"] * 3)
        dmap = m.tap_profile(dc)
        from holoviews import streams

        tap = next(s for s in dmap.streams if isinstance(s, streams.Tap))
        # tap at the centre of the dataset's extent (display CRS)
        merc = dataset.to_crs(3857)
        xmin, ymin, xmax, ymax = merc.bbox if hasattr(merc, "bbox") else (0, 0, 1, 1)
        tap.event(x=(xmin + xmax) / 2, y=(ymin + ymax) / 2)
        curve = dmap[()]
        assert isinstance(curve, hv.Curve)
        assert len(curve) == 3, "the profile must have one value per collection member"


class TestDrawAOI:
    """``draw`` / ``drawn_geometry`` / ``aoi_crop`` / ``cross_filter`` (DI.8)."""

    def test_draw_box_registers_rectangles(self, m):
        m.draw("box")
        assert isinstance(m.layers[-1], gv.Rectangles), f"got {type(m.layers[-1])}"

    @pytest.mark.parametrize(
        "kind, element",
        [("poly", "Polygons"), ("point", "Points"), ("freehand", "Path")],
    )
    def test_draw_other_kinds(self, m, kind, element):
        m.draw(kind)
        assert type(m.layers[-1]).__name__ == element, f"{kind} → {type(m.layers[-1])}"

    def test_drawn_geometry_returns_raw_data_for_non_box(self, m):
        """A non-box draw stream returns its raw column dict (not a bbox tuple)."""
        m.draw("point")
        m._draw_stream.event(data={"Longitude": [1.0], "Latitude": [2.0]})
        assert m.drawn_geometry == {"Longitude": [1.0], "Latitude": [2.0]}

    def test_unknown_draw_kind_raises(self, m):
        with pytest.raises(ValueError, match="unknown draw kind"):
            m.draw("blob")

    def test_drawn_geometry_none_before_drawing(self, m):
        m.draw("box")
        assert m.drawn_geometry is None, "nothing drawn yet → None"

    def test_box_event_yields_display_crs_bbox(self, m):
        m.draw("box")
        m._draw_stream.event(data={"x0": [1.0], "y0": [2.0], "x1": [3.0], "y1": [4.0]})
        assert m.drawn_geometry == (
            1.0,
            2.0,
            3.0,
            4.0,
        ), f"bbox not read back: {m.drawn_geometry}"

    def test_aoi_crop_calls_pyramids_crop(self, m):
        m.draw("box")
        m._draw_stream.event(
            data={"x0": [-1e5], "y0": [-1e5], "x1": [1e5], "y1": [1e5]}
        )
        captured = {}

        class _FakeDS:
            def crop(self, *, bbox, epsg):
                captured["bbox"] = bbox
                captured["epsg"] = epsg
                return "cropped"

        assert m.aoi_crop(_FakeDS()) == "cropped"
        assert captured["bbox"] == (-1e5, -1e5, 1e5, 1e5) and captured["epsg"] == 3857

    def test_aoi_crop_without_drawing_raises(self, m):
        m.draw("box")
        with pytest.raises(ValueError, match="needs a drawn box"):
            m.aoi_crop(object())

    def test_cross_filter_links_panels(self, m):
        from pyramids.feature import FeatureCollection

        fc = FeatureCollection.read_file("tests/data/points.geojson")
        a = InteractiveMap().points(fc).render()
        b = InteractiveMap().points(fc).render()
        linker = m.cross_filter(a, b)
        assert hasattr(linker, "selection_expr"), "link_selections instance expected"
