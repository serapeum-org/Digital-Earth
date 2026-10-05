"""`close()`/`__exit__` spare a figure the scene did not create — #371, resolved.

A scene closes the figure only when it made the figure itself. A figure handed to it through `ax=`/`fig=`
(and so every `grid` panel and every `inset` locator) belongs to whoever laid it out, and tearing it down on
`close()` was the #371 footgun: `with Map(ax=ax) as m:` or `host.inset().locator.close()` closed the caller's
whole figure. The gate is `Scene._owns_fig`, which `__init__` already sets (`True` only when it built the
figure), and it brings `Scene` into line with `TexturedGlobe.close`, which has always gated on it.

The pick-handler disconnect is *not* gated: a borrowed-axes scene still connected its own click handler to the
canvas, and that must come off on close regardless of who owns the figure, or it answers for a scene whose data
has just been let go.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseEvent

from digitalearth.static import Map, Scene


class TestOwnedFigureStillCloses:
    """A scene that built its own figure closes it, exactly as before."""

    def test_close_closes_the_scenes_own_figure(self):
        """`m.close()` removes the figure `m` created from the open set.

        Test scenario:
            A `Map(crs=...)` makes its own figure, so it owns it and must still close it.
        """
        m = Map(crs=4326)
        number = m.fig.number
        m.close()
        assert number not in plt.get_fignums(), (
            f"an owned figure must still close; open: {plt.get_fignums()}"
        )

    def test_with_block_closes_an_owned_figure(self):
        """A `with Map(crs=...)` block closes its own figure on exit."""
        with Map(crs=4326) as m:
            number = m.fig.number
            assert number in plt.get_fignums()
        assert number not in plt.get_fignums(), (
            f"a with-block over an owned figure must close it; open: {plt.get_fignums()}"
        )


class TestBorrowedFigureIsSpared:
    """A figure the caller laid out survives the scene that borrowed it."""

    def test_close_spares_a_borrowed_figure(self):
        """`Map(crs=..., ax=ax).close()` leaves the caller's figure open (#371).

        Test scenario:
            The caller owns `fig`; the scene only borrowed its axes, so `close()` must not touch it.
        """
        fig, ax = plt.subplots()
        number = fig.number
        m = Map(crs=4326, ax=ax)
        assert m._owns_fig is False
        m.close()
        assert number in plt.get_fignums(), (
            f"a borrowed figure must be spared; open: {plt.get_fignums()}"
        )
        plt.close(fig)

    def test_with_block_spares_a_borrowed_figure(self):
        """A `with Map(ax=ax)` block leaves the caller's figure open on exit."""
        fig, ax = plt.subplots()
        number = fig.number
        with Map(crs=4326, ax=ax):
            assert number in plt.get_fignums()
        assert number in plt.get_fignums(), (
            f"a with-block over a borrowed figure must spare it; open: {plt.get_fignums()}"
        )
        plt.close(fig)

    def test_closing_an_inset_locator_spares_the_host(self):
        """`host.inset(); host.locator.close()` leaves the host's figure open.

        Test scenario:
            The locator is drawn on an inset axes of the host's own figure, so it borrows that figure;
            closing the locator must not close the map it sits in — the exact #371 case.
        """
        host = Map(crs=4326)
        host.set_bounds([2.0, 3.0, 8.0, 9.0])
        host.inset(size=0.2)
        locator = host.locator
        assert locator._owns_fig is False
        locator.close()
        assert host.fig.number in plt.get_fignums(), (
            f"closing an inset locator must spare the host figure; open: {plt.get_fignums()}"
        )
        host.close()


class TestPickHandlerComesOffEitherWay:
    """The click handler is disconnected on close whether or not the figure is owned."""

    def test_a_borrowed_scene_stops_answering_clicks_after_close(self):
        """A borrowed-axes `on_pick` map delivers nothing after `close()`, though its figure lives on.

        Test scenario:
            The handler is the scene's own, connected to the shared canvas; it must come off on close even
            though the figure is spared, or it goes on firing for a scene whose data has been let go.
        """
        fig, ax = plt.subplots()
        m = Map(crs=4326, ax=ax)
        m.field(np.arange(16.0).reshape(4, 4), name="grid")
        seen = []
        m.on_pick(seen.append)
        assert m._pick_cid is not None
        m.close()
        assert m._pick_cid is None, (
            "the pick handler must be disconnected on close, owned or borrowed"
        )
        assert fig.number in plt.get_fignums(), "the borrowed figure is still spared"
        m.fig.canvas.draw()
        px, py = ax.transData.transform((1.5, 2.0))
        click = MouseEvent("button_press_event", fig.canvas, px, py, button=1)
        fig.canvas.callbacks.process("button_press_event", click)
        assert seen == [], f"a closed scene must not answer clicks; got {seen}"
        plt.close(fig)


class TestBareSceneToo:
    """The gate holds for a bare `Scene`, not only a `Map`."""

    def test_scene_given_only_an_axes_spares_it(self):
        """`Scene(ax=ax).close()` spares the caller's figure."""
        fig, ax = plt.subplots()
        number = fig.number
        Scene(ax=ax).close()
        assert number in plt.get_fignums(), (
            f"a bare Scene on a borrowed axes must spare the figure; open: {plt.get_fignums()}"
        )
        plt.close(fig)
