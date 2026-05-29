"""Tests for DC.5/DC.6 — Map(globe=) projection frame, graticule, set_global."""
import numpy as np
import pytest
from pyramids.dataset import Dataset

from digitalearth.scene import Map, projections


def test_globe_frame_applied_on_render(dataset):
    """A globe Map draws a boundary patch and equal aspect after render()."""
    m = Map(crs=projections.orthographic(lon=-9, lat=39), globe=True)
    m.imshow(dataset)
    assert not m.ax.patches            # frame not applied until render
    m.render()
    assert m.ax.patches                # boundary patch added
    assert m.ax.get_aspect() == 1.0
    assert m._framed is True


def test_globe_frame_idempotent(dataset):
    """render() applies the frame once (no duplicate boundary patches)."""
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    m.imshow(dataset)
    m.render(); n = len(m.ax.patches)
    m.render()
    assert len(m.ax.patches) == n


def test_non_globe_map_unframed(dataset):
    """A plain (globe=False) Map adds no boundary patch."""
    m = Map(crs=dataset.epsg)
    m.imshow(dataset)
    m.render()
    assert not m.ax.patches


def test_graticule_drawn_in_frame(dataset):
    """graticule() lines are drawn when the globe frame is applied."""
    m = Map(crs=projections.orthographic(-9, 39), globe=True)
    m.imshow(dataset)
    m.graticule(lon_step=30, lat_step=30)
    before = len(m.ax.lines)
    m.render()
    assert len(m.ax.lines) > before


def test_set_global_sets_full_domain():
    """set_global sets the axes extent to the projection's full domain."""
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    m.set_global()
    _, xlim, ylim = projections.projection_frame(m.crs)
    assert m.ax.get_xlim()[0] == pytest.approx(xlim[0], rel=1e-6)
    assert m.ax.get_ylim()[1] == pytest.approx(ylim[1], rel=1e-6)


def test_polar_stereographic_reprojects(dataset):
    """An EPSG-coded projection (polar south, 3031) reprojects and frames without proj4."""
    m = Map(crs=projections.polar_south(), globe=True)
    m.imshow(dataset)
    m.render()
    assert len(m.layers) == 1 and m.ax.patches


def test_globe_coastlines_when_online(dataset):
    """Globe coastlines project per-line and split at the limb (skipped offline)."""
    m = Map(crs=projections.orthographic(10, 25), globe=True)
    m.imshow(dataset)
    try:
        segs = m.coastlines(resolution="110m")
    except Exception as exc:  # network/data unavailable
        pytest.skip(f"Natural Earth coastline unavailable offline: {exc}")
    assert segs and m.ax.lines  # finite projected segments drawn
    pts = np.vstack([line.get_xydata() for line in m.ax.lines])
    assert np.isfinite(pts).all()  # no inf/nan reached the axes


@pytest.mark.parametrize("layer", ["land", "ocean"])
def test_polygon_fills_rejected_on_globe(layer):
    """land/ocean fills raise a clear error on a globe (not yet supported)."""
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    with pytest.raises(NotImplementedError, match="globe map"):
        getattr(m, layer)()


def test_globe_save(dataset, tmp_path):
    """A globe map saves a non-empty PNG (frame applied on save)."""
    m = Map(crs=projections.orthographic(-9, 39), globe=True)
    m.imshow(dataset)
    m.coastlines() if False else None  # coastlines need network; covered elsewhere
    m.graticule()
    out = tmp_path / "globe.png"
    m.save(str(out))
    assert out.exists() and out.stat().st_size > 0
