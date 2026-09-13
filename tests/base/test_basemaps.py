"""Tests for #236 — the actionable errors ``base/basemaps`` produces must survive the trip through the API.

``digitalearth.base.basemaps`` is written to fail *loudly*: a keyed preset that cannot find its credential
raises a ``ValueError`` naming the environment variable to set, and an unknown preset name lists the ones that
exist. Reached through :func:`digitalearth.api.quickmap`, all of that used to vanish — three decoration steps
were wrapped in ``except Exception: pass``, and ``basemap`` was typed ``bool`` and only tested for truth, so a
preset name was truthy, dropped on the floor, and the *default* basemap drawn in its place.

These tests pin both halves: the request reaches ``Map.basemap`` as given, and only a genuine unavailability
(a network/tile failure, a missing optional extra) is tolerated — with a warning, never in silence.
"""

import logging

import numpy as np
import pytest

from digitalearth.api import quickmap
from digitalearth.base import basemaps
from digitalearth.base.basemaps import KeyedTileSource

#: A preset with no required keywords, so a test can reach the credential check without a ``preset`` dict.
#: (``Planet.NICFI`` needs a ``date``, so naming it alone raises about the keyword, not the key.)
_TEST_ENV = "DIGITALEARTH_TEST_TILE_KEY"


def _test_preset() -> KeyedTileSource:
    """A keyed tile source whose credential comes from an environment variable no test sets.

    Returns:
        KeyedTileSource: the preset, with world-wide bounds so the coverage guard never fires first.
    """
    return KeyedTileSource(
        name="Test.Keyed",
        url_template="https://tiles.invalid/{z}/{x}/{y}.png?key={api_key}",
        attribution="© test",
        credential_env=_TEST_ENV,
    )


@pytest.fixture
def keyed_preset(monkeypatch):
    """Register ``Test.Keyed`` in the keyed-basemap registry for the duration of one test.

    Args:
        monkeypatch: pytest's monkeypatch fixture.

    Returns:
        str: the preset's name, as a caller would spell it.
    """
    monkeypatch.setitem(basemaps.KEYED_BASEMAPS, "test.keyed", _test_preset)
    monkeypatch.setitem(basemaps.KEYED_BASEMAP_NAMES, "test.keyed", "Test.Keyed")
    monkeypatch.delenv(_TEST_ENV, raising=False)
    return "Test.Keyed"


@pytest.fixture
def raster():
    """A small lon/lat raster to draw under the basemap.

    Returns:
        pyramids.dataset.Dataset: a 20x20 EPSG:4326 raster around (4E, 53N).
    """
    from pyramids.dataset import Dataset, GeoReference

    return Dataset.from_array(
        np.ones((20, 20), "float32"),
        geo_ref=GeoReference(geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326),
    )


class TestKeyedBasemapRequest:
    """A keyed preset named through ``quickmap`` must reach ``Map.basemap`` and keep its error."""

    def test_credential_error_reaches_the_caller(self, raster, keyed_preset):
        """The credential ``ValueError`` names the variable to set instead of being swallowed.

        Test scenario:
            The reported repro, with a preset that needs nothing but a key: before the fix the name was
            discarded (``basemap`` was only tested for truth) and the default basemap drawn in silence.
        """
        with pytest.raises(ValueError, match=_TEST_ENV):
            quickmap(raster, crs=4326, basemap=keyed_preset, colorbar=False)

    def test_unknown_preset_is_reported(self, raster):
        """An unknown preset name raises rather than silently drawing the default.

        Test scenario:
            ``is_keyed_basemap`` says no for an unknown name, so this lands in cleopatra's provider lookup —
            either way the caller is told, which is the behaviour that was missing.
        """
        with pytest.raises((ValueError, TypeError)):
            quickmap(raster, crs=4326, basemap="NoSuch.Preset.Name", colorbar=False)

    def test_true_still_means_the_default_source(self, raster, monkeypatch):
        """``basemap=True`` keeps meaning "the backend's default", i.e. no source is forwarded.

        Test scenario:
            Widening the argument must not change what the boolean meant.
        """
        seen = []
        monkeypatch.setattr(
            "digitalearth.static.Map.basemap",
            lambda self, source=None, **kwargs: seen.append(source),
        )
        quickmap(raster, crs=4326, basemap=True, colorbar=False)
        assert seen == [None], f"basemap=True should pass no source, got {seen!r}"


class TestBestEffortDecoration:
    """Only an unavailability is tolerated — and it is logged, not silently dropped."""

    @pytest.mark.parametrize("step", ["basemap", "coastlines"])
    def test_unreachable_assets_warn_but_do_not_raise(
        self, raster, monkeypatch, caplog, step
    ):
        """A network failure leaves the figure undecorated and says so in a warning.

        Args:
            step: The decoration step being made to fail.

        Test scenario:
            ``ConnectionError`` is what cleopatra raises after exhausting its tile/asset retries.
        """

        def _unreachable(self, *args, **kwargs):
            raise ConnectionError("tile server unreachable")

        monkeypatch.setattr(f"digitalearth.static.Map.{step}", _unreachable)
        with caplog.at_level(logging.WARNING, logger="digitalearth.api"):
            scene = quickmap(raster, crs=4326, colorbar=False, **{step: True})
        assert scene is not None, (
            "an unreachable decoration must not fail the whole call"
        )
        assert step in caplog.text, (
            f"the warning should name the {step} step; got {caplog.text!r}"
        )

    @pytest.mark.parametrize("step", ["basemap", "coastlines"])
    def test_actionable_errors_propagate(self, raster, monkeypatch, step):
        """A ``ValueError`` from a decoration step is the caller being told what to fix — it must not be eaten.

        Args:
            step: The decoration step being made to fail.

        Test scenario:
            ``Map.basemap`` raises exactly this for an empty extent ("Axes have no data extent. Plot data
            before adding a basemap."), and ``quickmap`` used to answer it with a blank figure.
        """

        def _actionable(self, *args, **kwargs):
            raise ValueError(
                "Axes have no data extent. Plot data before adding a basemap."
            )

        monkeypatch.setattr(f"digitalearth.static.Map.{step}", _actionable)
        with pytest.raises(ValueError, match="no data extent"):
            quickmap(raster, crs=4326, colorbar=False, **{step: True})

    def test_colorbar_failure_is_narrowed_and_logged(self, raster, monkeypatch, caplog):
        """An unmappable layer still skips the colorbar, but now leaves a warning behind.

        Test scenario:
            ``AttributeError`` is what matplotlib raises for an outline-only artist (no ``cmap``).
        """

        def _unmappable(self, *args, **kwargs):
            raise AttributeError("'Line2D' object has no attribute 'cmap'")

        monkeypatch.setattr("digitalearth.static.Map.colorbar", _unmappable)
        with caplog.at_level(logging.WARNING, logger="digitalearth.api"):
            scene = quickmap(raster, crs=4326)
        assert scene is not None, "an unmappable layer must not fail the call"
        assert "colorbar" in caplog.text, (
            f"the skip should be logged; got {caplog.text!r}"
        )

    def test_unexpected_colorbar_error_propagates(self, raster, monkeypatch):
        """An error that is not "this layer cannot carry a colorbar" is no longer swallowed.

        Test scenario:
            A ``RuntimeError`` out of the colorbar step is a bug, not a best-effort miss.
        """

        def _boom(self, *args, **kwargs):
            raise RuntimeError("something genuinely broke")

        monkeypatch.setattr("digitalearth.static.Map.colorbar", _boom)
        with pytest.raises(RuntimeError, match="genuinely broke"):
            quickmap(raster, crs=4326)
