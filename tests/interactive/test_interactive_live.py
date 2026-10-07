"""Streaming / live-updating layers (IN-17).

A live layer is a DynamicMap backed by a Pipe (replace) or Buffer (append). The push updates the stream in
process, so these assert the data path end to end; only the browser repaint needs a live kernel.
"""

import pytest

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")
pd = pytest.importorskip("pandas")

from holoviews import streams  # noqa: E402

from digitalearth.interactive import InteractiveMap  # noqa: E402


def _xy(*rows) -> "pd.DataFrame":
    """Return a small x/y DataFrame for a point feed.

    Args:
        *rows: ``(x, y)`` tuples.

    Returns:
        A DataFrame with ``x``/``y`` columns.
    """
    return pd.DataFrame(rows, columns=["x", "y"])


class TestPipeLiveLayer:
    """A Pipe-backed live layer: each push replaces the data."""

    def test_live_points_registers_a_dynamicmap_layer(self):
        """``live`` adds a DynamicMap layer and chains (returns the map)."""
        m = InteractiveMap()
        assert m.live(kind="points", name="cars") is m
        assert isinstance(m.layers[-1], hv.DynamicMap)

    def test_the_stream_is_a_pipe_reachable_by_id(self):
        """``live_stream`` hands back the Pipe for the layer, empty before any push."""
        m = InteractiveMap().live(kind="points", name="cars")
        assert isinstance(m.live_stream("cars"), streams.Pipe)
        assert m.live_stream("cars").data is None

    def test_push_replaces_the_data_and_redraws(self):
        """A push sends new data to the Pipe; the current frame reflects it."""
        m = InteractiveMap().live(kind="points", name="cars")
        assert m.push("cars", _xy((0.0, 0.0), (1.0, 1.0))) is m
        assert len(m.live_stream("cars").data) == 2
        assert len(m.layers[-1][()]) == 2


class TestBufferLiveLayer:
    """A Buffer-backed live layer: each push appends, keeping the last ``length`` rows."""

    def test_buffer_tails_the_last_length_rows(self):
        """Two pushes past a length-2 window keep exactly the last two rows."""
        m = InteractiveMap().live(
            kind="points", data=_xy((0.0, 0.0)), buffer=True, length=2, name="track"
        )
        m.push("track", _xy((1.0, 1.0)))
        m.push("track", _xy((2.0, 2.0)))
        assert len(m.live_stream("track").data) == 2

    def test_a_path_kind_renders_empty_and_after_a_push(self):
        """``kind="path"`` renders an empty frame and a pushed track (N1 — exercise the draw/push path)."""
        m = InteractiveMap().live(kind="path", name="route")
        assert isinstance(m.live_stream("route"), streams.Pipe)
        assert len(m.layers[-1][()].dimension_values(0)) == 0  # empty path renders
        m.push("route", _xy((0.0, 0.0), (1.0, 1.0)))
        assert len(m.live_stream("route").data) == 2
        assert len(m.layers[-1][()].dimension_values(0)) == 2  # pushed track renders


class TestLiveLayerLifecycle:
    """A live stream is dropped when its layer goes, so a removed/reused id is no longer live (M1)."""

    def test_remove_layer_drops_the_live_stream(self):
        """After ``remove_layer`` the id is not a live layer, so ``push`` refuses it."""
        m = InteractiveMap().live(kind="points", name="cars")
        m.remove_layer("cars")
        frame = _xy((0.0, 0.0))
        assert "cars" not in m._live_streams
        with pytest.raises(KeyError, match="no live layer"):
            m.push("cars", frame)

    def test_a_reused_id_is_not_live_after_remove(self):
        """Re-adding a non-live layer with a freed id does not inherit the old live stream."""
        m = InteractiveMap().live(kind="points", name="cars")
        m.remove_layer("cars")
        m.add_layer(gv.Points([], crs=gv.util.process_crs(m.crs)), name="cars")
        frame = _xy((0.0, 0.0))
        with pytest.raises(KeyError, match="no live layer"):
            m.push("cars", frame)

    def test_close_clears_the_live_streams(self):
        """``close()`` lets go of every live stream with the rest of the engine state."""
        m = InteractiveMap().live(kind="points", name="cars")
        m.close()
        assert m._live_streams == {}

    def test_draw_figure_that_drops_a_live_layer_drops_its_stream(self):
        """A `_change`/`draw_figure` that drops the live layer must not leave a detached stream (round-2 M1)."""
        m = InteractiveMap().live(kind="points", name="cars")
        m.draw_figure(InteractiveMap().figure_spec)  # an empty figure drops every layer
        frame = _xy((0.0, 0.0))
        assert "cars" not in m._live_streams
        with pytest.raises(KeyError, match="no live layer"):
            m.push("cars", frame)


class TestLiveKeying:
    """Each live layer's stream is keyed by its own id, even when ids are auto-generated (I1)."""

    def test_two_auto_named_live_layers_key_independently(self):
        """Two unnamed live layers get distinct ids, and a push reaches only the one addressed."""
        m = InteractiveMap().live(kind="points").live(kind="path")
        first, second = m.layer_ids[-2], m.layer_ids[-1]
        assert first != second
        assert {first, second} <= set(m._live_streams)
        m.push(first, _xy((0.0, 0.0), (1.0, 1.0)))
        assert len(m.live_stream(first).data) == 2
        assert m.live_stream(second).data is None  # untouched


class TestLiveRefusals:
    """The refusals, each with a single throwing call in the ``pytest.raises`` block."""

    def test_unknown_kind_is_refused(self):
        """A kind outside points/path is refused."""
        m = InteractiveMap()
        with pytest.raises(ValueError, match="unknown live kind"):
            m.live(kind="blob")

    def test_buffer_without_initial_data_is_refused(self):
        """A Buffer cannot infer its columns with no seed data."""
        m = InteractiveMap()
        with pytest.raises(ValueError, match="needs an initial"):
            m.live(buffer=True)

    def test_push_to_an_unknown_layer_is_refused(self):
        """Pushing to an id that is not a live layer is a clear KeyError."""
        m = InteractiveMap()
        frame = _xy((0.0, 0.0))
        with pytest.raises(KeyError, match="no live layer"):
            m.push("nope", frame)

    def test_live_stream_for_an_unknown_layer_is_refused(self):
        """Asking for the stream of a non-live layer is refused."""
        m = InteractiveMap()
        with pytest.raises(KeyError, match="no live layer"):
            m.live_stream("nope")
