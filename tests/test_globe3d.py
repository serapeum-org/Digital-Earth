"""Tests for the D3.5 textured-globe renderer (digitalearth.three_d.GlobeMixin).

The globe path is gated on the optional, lazily-imported ``geovista``: the render tests skip when geovista is
absent, but the lazy-import **error** path is always tested (it must raise a clear, actionable message).
"""
import builtins

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.sources import get_source
from digitalearth.three_d import Scene3D
from digitalearth.three_d import globe as globe_mod


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def _global_field():
    """A cos(latitude) field on a 1-D lon/lat grid (a smooth global field to drape)."""
    lon = np.linspace(-180.0, 180.0, 37)
    lat = np.linspace(-90.0, 90.0, 19)
    field = np.add.outer(np.cos(np.deg2rad(lat)), np.zeros(len(lon)))
    return get_source(field, x=lon, y=lat)


def test_globe_missing_geovista_raises_actionable_error(monkeypatch):
    """When geovista is not importable, globe() raises ImportError naming the install command."""
    real_import = builtins.__import__

    def _no_geovista(name, *args, **kwargs):
        if name == "geovista" or name.startswith("geovista."):
            raise ImportError("No module named 'geovista'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_geovista)
    scene = Scene3D(off_screen=True)
    with pytest.raises(ImportError, match=r"digitalearth\[3d\]"):
        scene.globe(_global_field(), coastlines=False)
    scene.close()


# --- real-render tests below need geovista installed ---
geovista = pytest.importorskip("geovista")


def test_globe_renders_field_sphere():
    """globe() drapes the field on a sphere and registers a layer (no coastlines = offline-safe)."""
    scene = Scene3D(off_screen=True)
    actor = scene.globe(_global_field(), coastlines=False)
    assert actor is not None and len(scene.layers) == 1
    assert bool(scene.screenshot().any())
    scene.close()


def test_globe_with_coastlines_adds_a_second_layer():
    """With coastlines=True the globe adds the field sphere plus a coastline overlay (two layers)."""
    scene = Scene3D(off_screen=True)
    scene.globe(_global_field(), coastlines=True, coastline_resolution="110m")
    assert len(scene.layers) == 2
    scene.close()


def test_globe_accepts_a_source_directly():
    """globe() takes an already-built Source without re-sourcing it."""
    scene = Scene3D(off_screen=True)
    scene.globe(_global_field(), coastlines=False, cmap="cividis")
    assert globe_mod._GEOVISTA_DATA in scene.layers[0][0].point_data
    scene.close()
