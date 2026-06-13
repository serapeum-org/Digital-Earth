"""DI.12 — autostyle defaults + Source ingestion + quickplot(backend="interactive") wiring.

Confirms the interactive builders pull their colormap from ``autostyle.auto_style`` when not overridden,
accept bare ``Source`` objects, and that ``quickplot(backend="interactive")`` returns an InteractiveMap.
Runs in the ``interactive`` pixi env.
"""

import numpy as np
import pytest

from digitalearth.interactive import InteractiveMap
from digitalearth.sources import Source
from digitalearth.sources.dimension import DimensionInfo

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")


@pytest.fixture()
def m() -> InteractiveMap:
    """A map already in EPSG:3857 so a bare Source needs no reprojection."""
    return InteractiveMap(crs=3857)


def _source(variable: str) -> Source:
    """A tiny 3857 raster Source carrying a variable name (drives the autostyle lookup)."""
    z = DimensionInfo(np.arange(6.0).reshape(2, 3), "z")
    x = DimensionInfo(np.array([0.0, 1.0, 2.0]), "x")
    y = DimensionInfo(np.array([0.0, 1.0]), "y")
    return Source(z, x, y, crs=3857, metadata={"variable": variable})


class TestAutostyleDefaults:
    """``image`` resolves its colormap from ``auto_style`` when ``cmap`` is not given (DI.12)."""

    def test_temperature_variable_picks_magics_cmap(self, m):
        from digitalearth.autostyle import auto_style

        src = _source("t2m")
        expected = auto_style(src)["cmap"]
        m.image(src)
        style = hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs
        assert (
            style["cmap"] == expected
        ), f"autostyle cmap not applied: {style.get('cmap')} != {expected}"

    def test_unknown_variable_falls_back_to_viridis(self, m):
        m.image(_source("totally-unknown-field"))
        style = hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs
        assert style["cmap"] == "viridis", f"fallback cmap wrong: {style.get('cmap')}"

    def test_explicit_cmap_overrides_autostyle(self, m):
        m.image(_source("t2m"), cmap="bone")
        style = hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs
        assert style["cmap"] == "bone", "explicit cmap must win over autostyle"


class TestSourceIngestion:
    """The builders accept a bare ``Source`` (decoupled from Dataset/FeatureCollection)."""

    def test_image_accepts_bare_source(self, m):
        src = _source("rain")
        m.image(src)
        assert isinstance(m.layers[0], hv.Image)
        x_samples = m.layers[0].dimension_values("x", expanded=False)
        assert np.allclose(
            np.sort(x_samples), src.x.values
        ), "Source coords must pass through"

    def test_source_input_matches_dataset_input(self, m, dataset):
        """A Source extracted from the dataset renders the same element type as the dataset."""
        from digitalearth.sources import get_source

        src = get_source(dataset.to_crs(3857))
        InteractiveMap(crs=3857).image(src)
        m.image(dataset)
        assert isinstance(m.layers[0], hv.Image)


class TestQuickplotBackend:
    """``quickplot(backend="interactive")`` dispatch (DX.1 data half)."""

    def test_raster_returns_interactive_map(self, dataset):
        from digitalearth.api import quickplot

        out = quickplot(dataset, crs=dataset.epsg, backend="interactive")
        assert isinstance(
            out, InteractiveMap
        ), f"expected InteractiveMap, got {type(out)}"
        assert isinstance(out.layers[0], hv.Image)

    def test_vector_returns_interactive_points(self):
        from pyramids.feature import FeatureCollection

        from digitalearth.api import quickplot

        fc = FeatureCollection.read_file("tests/data/points.geojson")
        out = quickplot(fc, backend="interactive")
        assert isinstance(out, InteractiveMap)
        assert isinstance(out.layers[0], gv.Points)

    def test_matplotlib_backend_still_returns_static_map(self, dataset):
        from digitalearth.api import quickplot
        from digitalearth.scene import Map

        out = quickplot(dataset, crs=dataset.epsg)
        assert isinstance(out, Map), "default backend must remain the static Map"

    def test_unknown_backend_raises(self, dataset):
        from digitalearth.api import quickplot

        with pytest.raises(ValueError, match="unknown backend"):
            quickplot(dataset, backend="webgl")

    def test_colorbar_false_drops_colorbar_on_data_layer(self, dataset):
        """quickplot(backend="interactive", colorbar=False) toggles the colorbar off (L2)."""
        from digitalearth.api import quickplot

        out = quickplot(
            dataset, crs=dataset.epsg, backend="interactive", colorbar=False
        )
        plot = hv.Store.lookup_options("bokeh", out.layers[0], "plot").kwargs
        assert plot["colorbar"] is False, "colorbar=False must reach the data layer"

    def test_colorbar_true_keeps_builder_default(self, dataset):
        """The default colorbar=True leaves the builder's colorbar in place."""
        from digitalearth.api import quickplot

        out = quickplot(dataset, crs=dataset.epsg, backend="interactive")
        plot = hv.Store.lookup_options("bokeh", out.layers[0], "plot").kwargs
        assert plot.get("colorbar") is not False, "default must keep the colorbar"
