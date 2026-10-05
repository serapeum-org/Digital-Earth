"""The static tier's decoration methods chain like its data builders do (round-1 L6).

ST-20 made all 27 *data* builders return ``Self``; the decoration methods kept handing back an ``Axes``
(the Natural-Earth layers), a ``Text``/``Annotation`` (``text``/``annotate``) or a ``PolyCollection``
(``nightshade``/``tissot``/``basemap``). So the tier had four return conventions at once, and the repo's
own ``CLAUDE.md`` example — ``m.field(ds).coastlines().colorbar()`` — raised
``AttributeError: 'Axes' object has no attribute 'colorbar'``.

These tests pin the two halves of the fix together, because either alone is a regression:

* every decoration method hands back **the composed map**, so the chain runs; and
* the artist it drew is still reachable, through :meth:`~digitalearth.static.scene.Scene.artist` — the
  public route ST-20 added for exactly this — and a layer that drew nothing still registers no id, which
  is what each method's ``None`` used to say.

``stock_img`` is deliberately **not** here: it keeps returning its backdrop artist (ST-20's own
follow-up decided that), because handing that one artist back for restyling is the method's purpose
rather than a side effect of how it was written. :class:`TestStockImgStillAnswersWithItsArtist` pins that
exception so it reads as a decision rather than an oversight.
"""

import matplotlib
import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalearth.static import Map

#: The bundled raster every map here is framed on, read relative to the repo root (as the suite runs).
RASTER_PATH = "examples/data/acc4000.tif"

#: A lon/lat point on the near side of every map below — Amsterdam.
NEAR_LON, NEAR_LAT = 4.9, 52.4

#: The instant the night shade is drawn at: the March equinox at noon UTC.
EQUINOX_NOON = "2026-03-20T12:00:00+00:00"

#: One Natural-Earth resolution, named once so the calls below read as the chain and not the keywords.
COARSE = "110m"

#: Every decoration method that must chain, as ``name -> (call, layer name)``. The layer name is what the
#: call asks for, so the artist can be read back by id rather than by "the last one".
CHAINING_CALLS = {
    "coastlines": (lambda canvas: canvas.coastlines(COARSE, name="deco"), "deco"),
    "borders": (lambda canvas: canvas.borders(COARSE, name="deco"), "deco"),
    "land": (lambda canvas: canvas.land(COARSE, name="deco"), "deco"),
    "ocean": (lambda canvas: canvas.ocean(COARSE, name="deco"), "deco"),
    "lakes": (lambda canvas: canvas.lakes(COARSE, name="deco"), "deco"),
    "rivers": (lambda canvas: canvas.rivers(COARSE, name="deco"), "deco"),
}


@pytest.fixture
def dataset():
    """The bundled raster, as the maps below frame themselves on.

    Returns:
        The pyramids ``Dataset`` read from :data:`RASTER_PATH`.
    """
    return Dataset.read_file(RASTER_PATH)


@pytest.fixture
def canvas():
    """A flat lon/lat map, closed when the test ends.

    Yields:
        A ``Map`` in EPSG:4326, which is where the Natural-Earth layers and the overlays all draw.
    """
    scene = Map(crs=4326)
    yield scene
    scene.close()


class TestEveryDecorationMethodHandsBackTheMap:
    """Tests that each decoration method returns the composed map, so the calls chain."""

    @pytest.mark.parametrize("method", sorted(CHAINING_CALLS))
    def test_the_call_returns_the_very_map_it_drew_on(self, method, canvas):
        """The return value is the same object the call was made on.

        Args:
            method: The entry in :data:`CHAINING_CALLS` under test.
            canvas: The flat map it draws on.

        Test scenario:
            Identity rather than type: a method that built a second map would satisfy
            ``isinstance(..., Map)`` and still break every chain, which is the failure
            ``tests/test_mixin_contract.py`` describes for a `-> Self` annotation that does not return
            `self`.
        """
        call, _ = CHAINING_CALLS[method]
        assert call(canvas) is canvas, f"{method}() did not hand back the map it drew on"

    @pytest.mark.parametrize("method", sorted(CHAINING_CALLS))
    def test_the_artist_is_still_reachable_by_the_layer_s_name(self, method, canvas):
        """What the method drew is still reachable, through the public accessor rather than its return.

        Args:
            method: The entry in :data:`CHAINING_CALLS` under test.
            canvas: The flat map it draws on.

        Test scenario:
            Chaining must not cost a caller the artist. ``Map.artist(layer_id)`` is where it moved, and
            it must answer with something matplotlib drew — not ``None``, and not the map.
        """
        call, layer = CHAINING_CALLS[method]
        call(canvas)
        drawn = canvas.artist(layer)
        assert drawn is not canvas, f"{method}() filed the map as its own artist"

    @pytest.mark.parametrize("method", sorted(CHAINING_CALLS))
    def test_the_layer_is_registered_under_the_name_it_was_given(self, method, canvas):
        """Each call describes one layer, named as the caller asked.

        Args:
            method: The entry in :data:`CHAINING_CALLS` under test.
            canvas: The flat map it draws on.

        Test scenario:
            The layer id is now the only handle on what was drawn, so a method that chained without
            registering anything would hand back a map with nothing to ask about.
        """
        call, layer = CHAINING_CALLS[method]
        call(canvas)
        assert canvas.layer_ids == [layer], canvas.layer_ids


class TestTheDocumentedChainRuns:
    """Tests for the exact expression ``CLAUDE.md`` promises, end to end."""

    def test_field_then_coastlines_then_colorbar_is_the_map(self, dataset):
        """``m.field(ds).coastlines().colorbar()`` runs and answers with the map.

        Args:
            dataset: The raster the field is drawn from.

        Test scenario:
            This is the repo's own documented example. Before L6 it raised
            ``AttributeError: 'Axes' object has no attribute 'colorbar'`` at the second link, because
            ``coastlines()`` handed back the axes.
        """
        with Map(crs=dataset.epsg) as scene:
            assert scene.field(dataset).coastlines(COARSE).colorbar() is scene

    def test_the_chain_draws_every_link_it_names(self, dataset):
        """A chained figure holds one layer per call, in the order they were made.

        Args:
            dataset: The raster the field is drawn from.

        Test scenario:
            A chain that returned the map without drawing would pass the identity test above and leave
            an empty figure, so the layers are counted as well.
        """
        with Map(crs=dataset.epsg) as scene:
            scene.field(dataset, name="grid").coastlines(COARSE, name="coast")
            assert scene.layer_ids == ["grid", "coast"], scene.layer_ids

class TestStockImgStillAnswersWithItsArtist:
    """Tests for the one decoration method that keeps its artist return, deliberately."""

    def test_the_backdrop_artist_comes_back_rather_than_the_map(self, dataset):
        """``stock_img`` hands back the backdrop itself, which is what the method is for.

        Args:
            dataset: The raster drawn as the backdrop.

        Test scenario:
            A backdrop is one artist a caller restyles afterwards — an alpha, a uniform fill, a zorder —
            so its return value is the method's purpose. L6 left it alone on that ground; this pins the
            exception so the next sweep reads it as a decision.
        """
        with Map(crs=dataset.epsg) as scene:
            backdrop = scene.stock_img(dataset)
            assert isinstance(backdrop, matplotlib.image.AxesImage), type(backdrop)

    def test_the_backdrop_is_the_artist_the_layer_filed(self, dataset):
        """What it hands back is the drawn layer's own artist, not a second object.

        Args:
            dataset: The raster drawn as the backdrop.

        Test scenario:
            ``stock_img`` reads its return value off ``Scene.artist`` rather than off the ``field`` call
            it delegates to; assert the two agree, so the shortcut and the public route cannot drift.
        """
        with Map(crs=dataset.epsg) as scene:
            backdrop = scene.stock_img(dataset, name="back")
            assert backdrop is scene.artist("back")


class TestTheDrawnGeometrySurvivedTheMigration:
    """Tests that moving the return value did not change what reaches the axes."""

    def test_the_reference_layer_put_finite_geometry_on_the_axes(self, canvas):
        """A chained Natural-Earth call still draws, with no non-finite vertex reaching matplotlib.

        Args:
            canvas: The flat lon/lat map.

        Test scenario:
            The Natural-Earth methods answered with the axes before L6, so nothing a caller held proved
            the draw happened. The axes' own contents do.
        """
        canvas.land(COARSE, name="fill")
        drawn = np.concatenate(
            [
                collection.get_paths()[0].vertices
                for collection in canvas.ax.collections
                if collection.get_paths()
            ]
        )
        assert np.isfinite(drawn).all()
