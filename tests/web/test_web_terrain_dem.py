"""DE-28 — the web tier's ``terrain_tiles`` takes a pyramids DEM, not only a URL template.

MapLibre draws 3-D terrain from a served ``raster-dem`` tile source, so the tier took a tile-URL template and
its docstrings told the caller to encode and serve the tiles themselves. pyramids encodes them —
``Dataset.to_terrain_rgb`` writes a ``{z}/{x}/{y}.png`` terrain-RGB pyramid — so the wiring was the only thing
missing, and the advice was pointing at work the caller did not have to do.

The tests read what the widget is actually handed, by replaying the map's queue against a recorder — the same
way `_build_map_widget` replays it. Terrain is described like every other kind now, and its drawer answers on
the terrain route: a ``raster-dem`` source and a ``setTerrain`` call, which is what MapLibre takes in place of
a style layer. Reading the widget calls rather than the description is what holds the two to each other.
"""

import pytest

pytest.importorskip("maplibre")

from digitalearth.web import WebMap  # noqa: E402


def _recorded_terrain(web_map):
    """Return what a map's terrain hands the widget.

    Args:
        web_map: A ``WebMap`` ``terrain_tiles`` has been called on.

    Returns:
        ``(source, terrain)`` — the ``RasterDEMSource`` registered and the ``(source_id, exaggeration)`` pair
        ``set_terrain`` was called with.
    """
    seen = {}

    class Recorder:
        """Captures what the terrain closure registers instead of building a MapLibre widget."""

        def add_source(self, src_id, source):
            """Record the DEM source.

            Args:
                src_id: The id the source is registered under.
                source: The ``RasterDEMSource`` itself.
            """
            seen["source"] = source

        def set_terrain(self, src_id, exaggeration):
            """Record the terrain call.

            Args:
                src_id: The source the terrain reads.
                exaggeration: The vertical exaggeration factor.
            """
            seen["terrain"] = (src_id, exaggeration)

        def add_layer(self, layer):
            """Ignore layers; only the DEM source is under test.

            Args:
                layer: The MapLibre layer, unused.
            """

    # The queue, not `layers`: `_apply_layer` takes the entries the widget build hands it, and `layers`
    # reports each described entry already resolved to the object it drew.
    for entry in web_map._queued:
        web_map._apply_layer(Recorder(), entry)
    assert "source" in seen, "the terrain drawer registered no DEM source"
    return seen["source"], seen.get("terrain")


class TestATileTemplateStillWorks:
    """The route the tier already had must not move, since every map using it names a served pyramid."""

    def test_a_template_is_passed_through_as_it_was_given(self):
        """A ``{z}/{x}/{y}`` template is a served pyramid, and is used unchanged."""
        template = "https://tiles.example/terrain/{z}/{x}/{y}.png"
        source, _ = _recorded_terrain(WebMap().terrain_tiles(template))
        assert source.tiles == [template], source.tiles

    def test_no_argument_takes_the_public_default(self):
        """A bare call still draws terrain, from the open-data tiles the tier documents."""
        from digitalearth.web.threed import _DEFAULT_TERRAIN_TILES

        source, terrain = _recorded_terrain(WebMap().terrain_tiles())
        assert source.tiles == [_DEFAULT_TERRAIN_TILES], source.tiles
        assert terrain[1] == 1.0, terrain


class TestAPyramidsDemIsEncodedHere:
    """DE-28 — ``terrain_tiles(dem)`` encodes the DEM rather than telling the caller to do it."""

    def test_a_dataset_becomes_a_terrain_rgb_pyramid_on_disk(self, dataset, tmp_path):
        """The tiles have to exist and decode, not merely be pointed at.

        Args:
            dataset: The shared pyramids raster fixture, read as an elevation band.
            tmp_path: pytest's temporary directory, where the pyramid is written.
        """
        import io

        from matplotlib import image as mpimage

        m = WebMap().terrain_tiles(dataset, tiles_path=tmp_path / "dem", zooms=(0, 5))
        source, _ = _recorded_terrain(m)
        assert source.tiles == ["dem/{z}/{x}/{y}.png"], source.tiles
        written = sorted((tmp_path / "dem").rglob("*.png"))
        assert written, "the DEM was not encoded to tiles"
        tile = mpimage.imread(io.BytesIO(written[0].read_bytes()))
        assert tile.shape[:2] == (256, 256), tile.shape

    def test_the_page_decodes_with_the_encoding_the_tiles_were_written_with(
        self, dataset, tmp_path
    ):
        """Terrain-RGB has two schemes and they are not interchangeable — a mismatch misreads every height.

        Args:
            dataset: The shared pyramids raster fixture.
            tmp_path: pytest's temporary directory.

        Test scenario:
            The tier defaults to ``terrarium`` and pyramids' encoder to ``mapbox``, so leaving the encoder to
            its own default would have written Mapbox tiles and told MapLibre to read them as terrarium. The
            two halves are compared rather than each checked against a literal.
        """
        m = WebMap().terrain_tiles(
            dataset, tiles_path=tmp_path / "dem", encoding="mapbox", zooms=(0, 4)
        )
        source, _ = _recorded_terrain(m)
        assert source.encoding == "mapbox", source.encoding

    def test_a_dem_with_no_destination_is_refused(self, dataset):
        """Encoding a pyramid writes files, so it has to be told where.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        m = WebMap()
        with pytest.raises(ValueError, match="tiles_path"):
            m.terrain_tiles(dataset)

    def test_the_template_carries_no_local_path_into_the_page(self, dataset, tmp_path):
        """A saved page is shared, and an absolute path names the machine that made it.

        Args:
            dataset: The shared pyramids raster fixture.
            tmp_path: pytest's temporary directory, whose own name must not reach the page.
        """
        out = tmp_path / "terrain.html"
        page = (
            WebMap()
            .terrain_tiles(dataset, tiles_path=tmp_path / "dem", zooms=(0, 4))
            .save(str(out))
            .read_text("utf-8")
        )
        assert "dem/{z}/{x}/{y}.png" in page, (
            "the page must address the pyramid it sits beside"
        )
        assert str(tmp_path) not in page, (
            "an absolute path names the machine that made the page"
        )


class TestWhatTerrainRefuses:
    """A DEM source is written into a page, so what it may be is worth being strict about."""

    def test_a_tile_provider_mapping_is_refused_rather_than_serialised(self):
        """An ``xyzservices.TileProvider`` is a ``dict``, and one carrying a key would be written out whole.

        Test scenario:
            The tier's basemap path takes a provider and pulls the URL out of it; terrain took its argument
            straight into the source's ``tiles`` list, so a mapping would have landed in the saved page with
            every field it holds — an API key among them.
        """
        provider = {
            "url": "https://tiles.example/{z}/{x}/{y}.png?key={apikey}",
            "apikey": "s3cr3t",
        }
        m = WebMap()
        with pytest.raises(TypeError, match="tile-URL template"):
            m.terrain_tiles(provider)

    def test_a_string_that_is_no_template_is_read_as_a_dem_path(self, tmp_path):
        """A plain path is a DEM to encode, and a missing one says so rather than becoming a tile URL.

        Args:
            tmp_path: pytest's temporary directory, naming a file that does not exist.
        """
        m = WebMap()
        nowhere = str(tmp_path / "nowhere.tif")
        with pytest.raises(FileNotFoundError):
            m.terrain_tiles(nowhere, tiles_path=tmp_path / "dem")

    @pytest.mark.parametrize("given", [(4, 1), (-1, 2), (1, 2, 3), 7])
    def test_a_zoom_range_that_writes_nothing_is_refused(
        self, dataset, tmp_path, given
    ):
        """A reversed or malformed range would write an empty pyramid and look like a silent failure.

        Args:
            dataset: The shared pyramids raster fixture.
            tmp_path: pytest's temporary directory.
            given: A ``zooms=`` value no pyramid can be written for.
        """
        m = WebMap()
        with pytest.raises(ValueError, match="zooms"):
            m.terrain_tiles(dataset, tiles_path=tmp_path / "dem", zooms=given)
