"""Selection/reset streams, multiple draw tools, the annotator, and richer cross-filter (IN-5/8/11).

These round-trip to Python only under a live kernel/server; the tests assert the *wiring* — which stream is
attached, that several draw tools coexist, that the annotator carries its read-back objects, and that the
cross-filter parameters reach ``link_selections`` — the way the rest of the interactive suite does.
"""

import pytest

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")

from holoviews import streams  # noqa: E402

from digitalearth.interactive import InteractiveMap  # noqa: E402


def _blank(**kwargs):
    """Return an empty overlay — a stand-in stream/selection callback.

    Args:
        **kwargs: The stream's keyword payload, ignored.

    Returns:
        An empty ``hv.Overlay``.
    """
    return hv.Overlay([])


@pytest.fixture
def m(dataset) -> InteractiveMap:
    """Return a Web-Mercator map carrying one raster layer.

    Args:
        dataset: The raster fixture.

    Returns:
        The map.
    """
    return InteractiveMap().field(dataset)


@pytest.fixture
def point_fc():
    """Return the point fixture as a pyramids ``FeatureCollection``.

    Returns:
        The feature collection.
    """
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


class TestSelectionStreams:
    """``on_select`` binds box/lasso/index selections; ``on_reset`` binds the reset tool (IN-8)."""

    @pytest.mark.parametrize(
        ("kind", "stream_type"),
        [
            ("box", streams.BoundsXY),
            ("lasso", streams.Lasso),
            ("index", streams.Selection1D),
        ],
    )
    def test_on_select_binds_the_right_stream(self, m, kind, stream_type):
        """Each selection kind attaches its matching HoloViews stream.

        Args:
            m: The map fixture.
            kind: The selection kind asked for.
            stream_type: The stream class it must bind.
        """
        dmap = m.on_select(_blank, kind=kind)
        assert isinstance(dmap, hv.DynamicMap)
        assert any(isinstance(s, stream_type) for s in dmap.streams)

    def test_on_select_unknown_kind_raises(self, m):
        """A kind that is not box/lasso/index is refused.

        Args:
            m: The map fixture.
        """
        with pytest.raises(ValueError, match="unknown selection kind"):
            m.on_select(_blank, kind="rectangle")

    def test_on_reset_binds_plotreset(self, m):
        """``on_reset`` attaches a ``PlotReset`` stream.

        Args:
            m: The map fixture.
        """
        dmap = m.on_reset(_blank)
        assert any(isinstance(s, streams.PlotReset) for s in dmap.streams)

    def test_on_select_without_a_layer_raises(self):
        """With no layer and no explicit source there is nothing to listen on."""
        empty = InteractiveMap()
        with pytest.raises(ValueError, match="needs a source layer"):
            empty.on_select(_blank)


class TestSeveralDrawTools:
    """A map can carry more than one draw tool now (IN-11)."""

    def test_two_draw_calls_both_register(self, m):
        """A second ``draw`` no longer replaces the first tool's binding.

        Args:
            m: The map fixture.
        """
        m.draw("box").draw("point")
        assert len(m._draw_streams) == 2

    def test_drawn_geometries_reads_every_tool(self, m):
        """``drawn_geometries`` returns one entry per tool that captured something, oldest first.

        Args:
            m: The map fixture.
        """
        m.draw("box").draw("point")
        m._draw_streams[0].event(
            data={"x0": [1.0], "y0": [2.0], "x1": [3.0], "y1": [4.0]}
        )
        m._draw_streams[1].event(data={"Longitude": [5.0], "Latitude": [6.0]})
        captured = m.drawn_geometries
        assert captured[0] == (1.0, 2.0, 3.0, 4.0)
        assert captured[1] == {"Longitude": [5.0], "Latitude": [6.0]}


class TestAnnotator:
    """``annotate`` opens HoloViews' annotator with its read-back objects (IN-11)."""

    def test_annotate_returns_a_readback_capable_annotator(self, m):
        """The annotator carries ``.annotated`` and ``.selected`` for the edits to come back through.

        Args:
            m: The map fixture.
        """
        annotator = m.annotate(annotations=["label"])
        assert hasattr(annotator, "annotated")
        assert hasattr(annotator, "selected")


class TestCrossFilterParameters:
    """``cross_filter`` exposes the ``link_selections`` knobs that decide how a brush selects (IN-5)."""

    def test_the_parameters_reach_the_linker(self, m, point_fc):
        """A caller's selection/cross-filter modes and alpha are set on the returned linker.

        Args:
            m: The map fixture.
            point_fc: A point layer to link.
        """
        other = InteractiveMap().points(point_fc).render()
        linker = m.cross_filter(
            other,
            selection_mode="intersect",
            cross_filter_mode="overwrite",
            unselected_alpha=0.3,
        )
        assert linker.selection_mode == "intersect"
        assert linker.cross_filter_mode == "overwrite"
        assert linker.unselected_alpha == 0.3

    def test_the_linker_exposes_selection_expr(self, m):
        """The returned linker carries ``selection_expr`` — the live predicate read back into Python.

        Args:
            m: The map fixture.
        """
        linker = m.cross_filter()
        assert hasattr(linker, "selection_expr")

    def test_the_optional_colour_and_index_knobs_reach_the_linker(self, m):
        """``index_cols``/``selected_color``/``unselected_color`` are forwarded only when given.

        Args:
            m: The map fixture.
        """
        linker = m.cross_filter(
            index_cols=["fid"],
            selected_color="#ff0000",
            unselected_color="#cccccc",
        )
        assert linker.index_cols == ["fid"]
        assert linker.selected_color == "#ff0000"
        assert linker.unselected_color == "#cccccc"


class TestStreamSourceResolution:
    """A stream listens on the layer added last, or on an explicit ``source=`` (IN-8)."""

    def test_on_tap_honours_an_explicit_source(self, m):
        """Passing ``source=`` binds the tap there rather than to the last layer.

        Args:
            m: The map fixture.
        """
        element = m.layers[0]
        dmap = m.on_tap(lambda x, y: hv.Overlay([]), source=element)
        assert isinstance(dmap, hv.DynamicMap)

    def test_on_select_honours_an_explicit_source(self, m):
        """``on_select(source=...)`` binds the selection there (the ``_source_layer`` explicit branch).

        Args:
            m: The map fixture.
        """
        dmap = m.on_select(_blank, source=m.layers[0])
        assert any(isinstance(s, streams.BoundsXY) for s in dmap.streams)


class TestDrawnGeometriesEdgeCases:
    """``drawn_geometries`` skips tools that captured nothing (IN-11)."""

    def test_a_tool_with_no_capture_is_skipped(self, m):
        """A draw tool that never fired contributes nothing, so the list is empty.

        Args:
            m: The map fixture.
        """
        m.draw("box")
        assert m.drawn_geometries == []

    def test_a_box_with_empty_data_is_skipped(self, m):
        """A box whose captured ``x0`` is an empty list is skipped, not returned as a degenerate bbox.

        Args:
            m: The map fixture.
        """
        m.draw("box")
        m._draw_streams[0].event(data={"x0": [], "y0": [], "x1": [], "y1": []})
        assert m.drawn_geometries == []
