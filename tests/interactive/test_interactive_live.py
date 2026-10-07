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

    def test_a_path_kind_is_accepted(self):
        """``kind="path"`` builds a growing-track live layer."""
        m = InteractiveMap().live(kind="path", name="route")
        assert isinstance(m.live_stream("route"), streams.Pipe)


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
