"""The static ``field`` builder adopts the Core channel keywords (#324, #332, order 27a).

``Map.field`` grew named ``band`` / ``cmap`` / ``limits`` / ``opacity`` parameters rather than reaching them
through ``**kwargs``, so the same channel is spelled the same way on every tier. ``limits`` and ``cmap`` reach
the ``ArrayGlyph`` artist directly — ``limits`` as its ``vmin``/``vmax``, read back as the image's colour
limits — while ``opacity`` and ``band`` are recorded in the layer's description under the engine's own
spelling (cleopatra folds ``alpha`` into the image data, so the artist exposes no scalar for it).

The dataset is built in memory with ``Dataset.from_array`` so the checks neither touch the filesystem nor
depend on the test's working directory.
"""

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

from digitalearth.static import Map

#: The colour limits the checks pin. Chosen well inside the data range so an auto-scaled default differs.
LIMITS = (0.0, 50.0)

#: The opacity the checks ask for — not ``1.0``, matplotlib's own default, which a dropped keyword would
#: still read as.
OPACITY = 0.25


@pytest.fixture
def dem() -> Dataset:
    """Return a small in-memory single-band raster in EPSG:4326.

    Returns:
        A pyramids ``Dataset`` whose one band holds 0..99 over a 10x10 grid.
    """
    array = np.arange(100, dtype="float32").reshape(10, 10)
    reference = GeoReference(geo=(0.0, 1.0, 0.0, 10.0, 0.0, -1.0), epsg=4326)
    return Dataset.from_array(array, geo_ref=reference)


@pytest.fixture
def drawn():
    """Yield an empty static map in EPSG:4326, closed on the way out.

    Yields:
        The tier's ``Map``.
    """
    scene = Map(crs=4326)
    yield scene
    scene.close()


def _props(scene):
    """Return the most recently drawn layer's symbology props.

    Args:
        scene: The map whose last layer is inspected.

    Returns:
        The ``Symbology.props`` mapping of the layer.
    """
    layer_id = scene.layer_ids[-1]
    return scene.figure_spec.layers.get(layer_id).symbology.props


class TestLimitsAndCmapReachTheArtist:
    """``limits`` and ``cmap`` arrive on the image, which is the only place they can colour it."""

    def test_limits_set_the_image_colour_limits(self, drawn, dem):
        """``limits=(vmin, vmax)`` is what the image is coloured through.

        Args:
            drawn: The map under test.
            dem: The raster fixture.
        """
        drawn.field(dem, limits=LIMITS)
        assert drawn.artist().get_clim() == LIMITS, (
            f"limits did not reach the image: {drawn.artist().get_clim()}"
        )

    def test_limits_are_auto_scaled_by_default(self, drawn, dem):
        """With no ``limits`` the image auto-scales, so the default must differ from the pinned pair.

        Args:
            drawn: The map under test.
            dem: The raster fixture.

        Test scenario:
            The complement that stops the check above passing for a builder that pins ``(0, 50)`` always.
        """
        drawn.field(dem)
        assert drawn.artist().get_clim() != LIMITS, (
            f"an unpinned field coincidentally matched {LIMITS}"
        )

    def test_cmap_reaches_the_image(self, drawn, dem):
        """``cmap`` names the colormap the image draws with.

        Args:
            drawn: The map under test.
            dem: The raster fixture.
        """
        drawn.field(dem, cmap="magma")
        assert drawn.artist().cmap.name == "magma", (
            f"cmap did not reach the image: {drawn.artist().cmap.name}"
        )


class TestOpacityAndBandReachTheEngineDescription:
    """``opacity`` and ``band`` are handed to ``ArrayGlyph`` under its own spelling, recorded on the layer."""

    def test_opacity_is_recorded_as_the_engines_alpha(self, drawn, dem):
        """``opacity`` travels on as matplotlib's ``alpha``, the spelling ``ArrayGlyph`` is handed.

        Args:
            drawn: The map under test.
            dem: The raster fixture.

        Test scenario:
            cleopatra folds ``alpha`` into the rendered image rather than onto a scalar the artist exposes,
            so the engine-facing record is where the channel is checked.
        """
        drawn.field(dem, opacity=OPACITY)
        assert _props(drawn)["opts"]["alpha"] == OPACITY, (
            f"opacity did not reach the engine opts: {_props(drawn)['opts']}"
        )

    def test_opacity_is_absent_by_default(self, drawn, dem):
        """A caller who asks for no opacity must not have one invented in the engine opts.

        Args:
            drawn: The map under test.
            dem: The raster fixture.
        """
        drawn.field(dem)
        engine_opts = _props(drawn).get("opts") or {}
        assert engine_opts.get("alpha") is None, (
            f"an unstyled field invented an alpha: {engine_opts}"
        )

    def test_band_is_recorded(self, drawn, dem):
        """``band`` is a named Core keyword now, so it is written into the description, not lost in kwargs.

        Args:
            drawn: The map under test.
            dem: The raster fixture.
        """
        drawn.field(dem, band=1)
        assert _props(drawn)["band"] == 1, (
            f"band was not recorded on the layer: {_props(drawn)['band']}"
        )

    def test_limits_travel_as_vmin_vmax(self, drawn, dem):
        """``limits`` reaches ``ArrayGlyph`` under its own ``vmin``/``vmax`` spelling.

        Args:
            drawn: The map under test.
            dem: The raster fixture.

        Test scenario:
            The Core channel is ``limits``; the engine is handed the two ends separately, which is where a
            caller's own ``vmin=``/``vmax=`` always went.
        """
        drawn.field(dem, limits=LIMITS)
        opts = _props(drawn)["opts"]
        assert (opts["vmin"], opts["vmax"]) == LIMITS, (
            f"limits did not travel as vmin/vmax: {opts}"
        )
