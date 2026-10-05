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

#: What the stubbed tile fetch hands back, so the accessor can be asserted against the engine's own object.
TILE_ARTIST = object()

#: Every decoration method that must chain, as ``name -> (call, layer name)``. The layer name is what the
#: call asks for, so the artist can be read back by id rather than by "the last one".
CHAINING_CALLS = {
    "coastlines": (lambda canvas: canvas.coastlines(COARSE, name="deco"), "deco"),
    "borders": (lambda canvas: canvas.borders(COARSE, name="deco"), "deco"),
    "land": (lambda canvas: canvas.land(COARSE, name="deco"), "deco"),
    "ocean": (lambda canvas: canvas.ocean(COARSE, name="deco"), "deco"),
    "lakes": (lambda canvas: canvas.lakes(COARSE, name="deco"), "deco"),
    "rivers": (lambda canvas: canvas.rivers(COARSE, name="deco"), "deco"),
    "text": (
        lambda canvas: canvas.text(NEAR_LON, NEAR_LAT, "Amsterdam", name="deco"),
        "deco",
    ),
    "annotate": (
        lambda canvas: canvas.annotate(NEAR_LON, NEAR_LAT, "Amsterdam", name="deco"),
        "deco",
    ),
    "nightshade": (lambda canvas: canvas.nightshade(EQUINOX_NOON, name="deco"), "deco"),
    "tissot": (lambda canvas: canvas.tissot(name="deco"), "deco"),
}

#: The orthographic globe whose far side the skipped-label tests place a point on.
FAR_SIDE_GLOBE = "+proj=ortho +lon_0=0 +lat_0=0"


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
        assert call(canvas) is canvas, (
            f"{method}() did not hand back the map it drew on"
        )

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

    def test_a_decoration_chain_mixes_the_overlays_and_the_reference_layers(self):
        """Several decoration calls chain into one another, across the kinds.

        Test scenario:
            The six Natural-Earth layers, the two text builders and the two solar overlays were three
            separate return conventions before L6 — an ``Axes``, a ``Text`` and a ``PolyCollection``.
            Chaining all three kinds in one expression is what proves they are now one.
        """
        with Map(crs=4326) as scene:
            chained = (
                scene.ocean(COARSE)
                .land(COARSE)
                .coastlines(COARSE)
                .tissot()
                .nightshade(EQUINOX_NOON)
                .text(NEAR_LON, NEAR_LAT, "Amsterdam")
            )
            assert len(chained.layer_ids) == 6, chained.layer_ids

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


class TestALayerThatDrewNothingStillChains:
    """Tests for what replaced the ``None`` ``text``/``annotate`` used to answer with."""

    @pytest.mark.parametrize("method", ["text", "annotate"])
    def test_a_far_side_label_registers_no_layer(self, method):
        """A point on the far side of a globe draws nothing, and describes nothing.

        Args:
            method: ``text`` or ``annotate`` — both reproject their point the same way.

        Test scenario:
            Both answered ``None`` for this before L6. The successor signal is the absent layer id —
            the reading ``digitalearth.base.crs`` already uses — and the chain still runs, so a caller
            who did not care never has to branch on it.
        """
        with Map(crs=FAR_SIDE_GLOBE, globe=True) as scene:
            getattr(scene, method)(180.0, 0.0, "hidden", name="far")
            assert scene.layer_ids == [], scene.layer_ids

    @pytest.mark.parametrize("method", ["text", "annotate"])
    def test_a_far_side_label_still_hands_back_the_map(self, method):
        """The skipped call chains like a drawn one.

        Args:
            method: ``text`` or ``annotate``.

        Test scenario:
            The point of the uniform return: a chain does not break on the one link whose data the
            display CRS could not place.
        """
        with Map(crs=FAR_SIDE_GLOBE, globe=True) as scene:
            assert getattr(scene, method)(180.0, 0.0, "hidden") is scene

    def test_the_skipped_layer_s_artist_is_refused_by_name(self):
        """Asking for the artist of a layer that drew nothing is refused, not answered ``None``.

        Test scenario:
            ``Map.artist`` raises for an unknown id, and a skipped layer is dropped from the
            description — so that refusal is how "that layer drew nothing" now reads to a caller who
            does go looking for the artist. ``None`` would have been indistinguishable from a drawn
            layer whose drawer produced nothing.
        """
        with Map(crs=FAR_SIDE_GLOBE, globe=True) as scene:
            scene.text(180.0, 0.0, "hidden", name="far")
            with pytest.raises(KeyError, match="no layer 'far' on this figure"):
                scene.artist("far")


class TestBasemapChainsWithoutTheNetwork:
    """Tests for ``basemap``, which the tables above leave out because it fetches tiles."""

    @pytest.fixture
    def tiled(self, mocker):
        """A framed map whose tile fetch is a stand-in, closed when the test ends.

        Args:
            mocker: Stubs ``cleopatra.basemap.tiles.add_tiles`` where the tier calls it.

        Yields:
            The map, framed on a lon/lat region so the extent the tiles are asked for exists.
        """
        mocker.patch(
            "digitalearth.static.maps.decoration.add_tiles", return_value=TILE_ARTIST
        )
        scene = Map(domain=(-10.0, 35.0, 5.0, 45.0))
        scene.ax.set_xlim(-10.0, 5.0)
        scene.ax.set_ylim(35.0, 45.0)
        yield scene
        scene.close()

    def test_the_call_returns_the_very_map_it_drew_on(self, tiled):
        """``basemap()`` hands back the map, so a basemap can open a chain.

        Args:
            tiled: The framed map with a stubbed tile fetch.

        Test scenario:
            ``WebMap().basemap().points(...)`` is how the web tier already reads; this is the static
            tier catching up, which is the cross-tier half of L6.
        """
        assert tiled.basemap(name="tiles") is tiled

    def test_the_tile_artist_is_reachable_by_the_layer_s_name(self, tiled):
        """What ``add_tiles`` returned is still reachable, by layer id.

        Args:
            tiled: The framed map with a stubbed tile fetch.

        Test scenario:
            The stand-in's own object is asserted, so this fails if the accessor hands back anything
            other than exactly what the engine produced.
        """
        tiled.basemap(name="tiles")
        assert tiled.artist("tiles") is TILE_ARTIST

    def test_a_basemap_opens_a_chain_of_decoration(self, tiled):
        """A basemap, then reference geography, then a label — one expression.

        Args:
            tiled: The framed map with a stubbed tile fetch.

        Test scenario:
            The chain spans all three former return conventions and starts at the one that fetches,
            which is the shape a user writes.
        """
        chained = (
            tiled.basemap(name="tiles")
            .coastlines(COARSE)
            .text(NEAR_LON, NEAR_LAT, "Amsterdam")
        )
        assert chained.layer_ids == ["tiles", "coastlines-1", "text-1"], (
            chained.layer_ids
        )


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

    def test_the_night_shade_covers_midnight_and_not_noon(self, canvas):
        """The shade read off ``artist()`` is the same geometry the old return value was.

        Args:
            canvas: The flat lon/lat map.

        Test scenario:
            The sharpest check available on that artist: at the March equinox at noon UTC lon 170 is at
            midnight and lon 0 is at noon, so the shade must contain the one and not the other. Reading
            it through ``artist()`` proves the accessor hands back the real ``PolyCollection`` rather
            than something that merely answers ``get_paths``.
        """
        canvas.nightshade(EQUINOX_NOON, name="night")
        shade = canvas.artist("night")
        covered = [
            any(path.contains_point(xy) for path in shade.get_paths())
            for xy in ((170.0, 0.0), (0.0, 0.0))
        ]
        assert covered == [True, False], covered

    def test_the_world_grid_of_indicatrices_is_sixty_rings(self, canvas):
        """``tissot()``'s default world grid is still 60 rings, read off the artist.

        Args:
            canvas: The flat lon/lat map.

        Test scenario:
            Five latitudes by twelve longitudes. That count is what the old return value was read for,
            so it is read the new way here.
        """
        canvas.tissot(name="rings")
        assert len(canvas.artist("rings").get_paths()) == 60

    def test_the_label_keeps_the_text_it_was_given(self, canvas):
        """The ``Text`` ``text()`` drew is reachable and says what it was asked to.

        Args:
            canvas: The flat lon/lat map.

        Test scenario:
            ``text()`` used to hand the ``Text`` straight back, and a caller's one use for it was to
            read or restyle it. That must still be possible, by layer id.
        """
        canvas.text(NEAR_LON, NEAR_LAT, "Amsterdam", name="label")
        assert canvas.artist("label").get_text() == "Amsterdam"

    def test_the_annotation_is_the_one_on_the_axes(self, canvas):
        """``annotate()``'s ``Annotation`` is reachable by id, and is the artist matplotlib holds.

        Args:
            canvas: The flat lon/lat map.

        Test scenario:
            Identity against ``ax.texts`` rather than a type check, so an accessor that built a second
            artist would fail — the drawn one is what a caller restyles.
        """
        canvas.annotate(NEAR_LON, NEAR_LAT, "Amsterdam", name="arrow")
        assert canvas.artist("arrow") in canvas.ax.texts

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
