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


def test_globe_coastlines(dataset):
    """Globe coastlines project per-line and split at the limb (real 110m data, seeded cache)."""
    m = Map(crs=projections.orthographic(10, 25), globe=True)
    m.imshow(dataset)
    segs = m.coastlines(resolution="110m")
    assert segs and m.ax.lines  # finite projected segments drawn
    pts = np.vstack([line.get_xydata() for line in m.ax.lines])
    assert np.isfinite(pts).all()  # no inf/nan reached the axes


def test_ocean_fills_disc_on_globe():
    """ocean() fills the whole projection disc on a globe with a single finite ring (no network)."""
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    pc = m.ocean()
    assert pc is not None and len(pc.get_paths()) == 1
    verts = np.vstack([p.vertices for p in pc.get_paths()])
    assert np.isfinite(verts).all()


@pytest.fixture
def global_field():
    """A 2-degree global lon/lat raster (covers the whole sphere, so a globe has a far side).

    Returns:
        Dataset: a global EPSG:4326 field.
    """
    ny, nx = 90, 180
    lat = np.linspace(90, -90, ny)[:, None]
    z = (np.cos(np.deg2rad(lat)) * 30) * np.ones((ny, nx), "float32")
    return Dataset.create_from_array(arr=z.astype("float32"), geo=(-180.0, 2.0, 0.0, 90.0, 0.0, -2.0), epsg=4326)


def test_finite_polygons_drops_nonfinite():
    """_finite_polygons drops rings with any inf/nan vertex and the matching values."""
    good = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    bad = np.array([[0.0, 0.0], [np.inf, 0.0], [1.0, 1.0]])
    polys, vals = Map._finite_polygons([good, bad, good], np.array([10.0, 20.0, 30.0]))
    assert len(polys) == 2
    assert vals.tolist() == [10.0, 30.0]
    polys_only, none_vals = Map._finite_polygons([good, bad])
    assert len(polys_only) == 1 and none_vals is None


def test_globe_choropleth_drops_far_side():
    """choropleth on a globe drops polygons whose vector reprojection lands off the disc (no inf vertices)."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    fc = fc.to_crs(4326)
    fc["geometry"] = fc.geometry.buffer(2.0)  # ~2-degree polygons in lon/lat
    fc["val"] = range(len(fc))
    m = Map(crs=projections.orthographic(lon=-9, lat=39), globe=True)
    pc = m.choropleth(fc, column="val")
    verts = np.vstack([p.vertices for p in pc.get_paths()]) if pc.get_paths() else np.zeros((1, 2))
    assert np.isfinite(verts).all()  # no inf reached the PolyCollection


def test_globe_grid_cells_finite(global_field):
    """grid_cells on a globe renders finite cells (raster reprojection yields a finite projected grid)."""
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    pc = m.grid_cells(global_field)
    verts = np.vstack([p.vertices for p in pc.get_paths()])
    assert len(pc.get_paths()) > 0 and np.isfinite(verts).all()


def test_globe_tricontourf_finite(global_field):
    """tricontourf on a globe triangulates only finite points."""
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    m.tricontourf(global_field)
    assert len(m.layers) == 1


def test_globe_save(dataset, tmp_path):
    """A globe map saves a non-empty PNG (frame applied on save)."""
    m = Map(crs=projections.orthographic(-9, 39), globe=True)
    m.imshow(dataset)
    m.coastlines() if False else None  # coastlines need network; covered elsewhere
    m.graticule()
    out = tmp_path / "globe.png"
    m.save(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_globe_show_applies_frame(dataset):
    """show() applies the projection frame for a globe map (Agg backend -> no window)."""
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    m.imshow(dataset)
    assert not m.ax.patches
    m.show()  # MPLBACKEND=Agg makes this a no-op draw, but the frame must still be applied
    assert m.ax.patches and m._framed is True


def test_frame_is_memoised(dataset):
    """_frame caches projection_frame per CRS: a second call returns the same cached tuple."""
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    first = m._frame()
    assert m._frame_cache is not None and m._frame_cache[0] == m.crs
    second = m._frame()  # cache hit branch — no recompute
    assert second is first


def test_grid_cells_without_nodata(global_field, mocker):
    """grid_cells skips the nodata mask when the dataset declares no nodata value.

    Uses a matching display CRS (no reprojection) and a ``no_data_value`` of ``None`` so the
    ``nodata is None`` branch (which leaves every cell value intact) is exercised.
    """
    m = Map(crs=4326)  # matches the field's CRS -> _reproject returns it unchanged
    reprojected = m._reproject(global_field)
    mocker.patch.object(type(reprojected), "no_data_value",
                        new_callable=mocker.PropertyMock, return_value=[None])
    mocker.patch.object(m, "_reproject", return_value=reprojected)
    pc = m.grid_cells(global_field)
    assert len(pc.get_paths()) > 0


def test_project_line_features_skips_empty(mocker):
    """_project_line_features ignores empty coordinate arrays and projects the rest."""
    parts = [np.empty((0, 2)), np.array([(-9, 39), (-8, 40)], dtype=float)]
    m = Map(crs=projections.orthographic(-9, 39), globe=True)
    segs = m._project_line_features(parts)
    assert all(s.shape[1] == 2 and len(s) > 1 for s in segs)


def _add_features_drawing(*verts):
    """Build an ``add_features`` stand-in that draws ``verts`` as a LineCollection, preserving the view.

    Mirrors the real ``cleopatra.basemap.reference.add_features`` contract used by the flat ``_natural_earth`` path:
    it adds an artist to the axes and holds the current limits, so the decoration's own autoscale-when-empty
    logic is what gets exercised — all without touching the network.
    """
    from matplotlib.collections import LineCollection

    def fake(ax, layer="coastline", resolution="110m", *, crs=None, zorder=0, **style):
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.add_collection(LineCollection([np.asarray(v, dtype=float) for v in verts]))
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        return ax

    return fake


def test_natural_earth_flat_without_data_autoscales_to_layer(mocker):
    """On a flat map with nothing drawn yet, a Natural Earth layer autoscales the view to itself."""
    mocker.patch(
        "digitalearth.scene.maps.decoration.add_features",
        side_effect=_add_features_drawing([(-50, -20), (50, 20)]),
    )
    m = Map(crs=4326)  # flat, no imshow -> had_data is False
    m.coastlines()
    assert m.ax.collections  # the layer drew something
    assert m.ax.get_xlim()[0] <= -50 and m.ax.get_xlim()[1] >= 50  # fitted to the layer, not pinned


# --------------------------------------------------------------------- #43 globe land/ocean fills


@pytest.fixture
def land_fc():
    """Synthetic lon/lat exterior rings (a limb-crossing 'continent'), as ``natural_earth`` now returns.

    Returns:
        list[numpy.ndarray]: closed `(N, 2)` lon/lat rings spanning both hemispheres of an ortho globe —
            the coordinate-array shape ``cleopatra.basemap.reference.natural_earth`` yields for a polygon layer.
    """
    near = np.array([(-20, -20), (20, -20), (20, 20), (-20, 20), (-20, -20)], float)      # near side
    straddle = np.array([(60, -30), (120, -30), (120, 30), (60, 30), (60, -30)], float)   # crosses the limb
    return [near, straddle]


def test_project_polygon_features_finite_and_closed(land_fc):
    """_project_polygon_features returns finite, closed projected rings for a limb-crossing layer."""
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    rings = m._project_polygon_features(land_fc)
    assert rings, "expected at least one fill ring"
    verts = np.vstack(rings)
    assert np.isfinite(verts).all(), "fill rings must not contain inf/nan"
    assert all(np.allclose(r[0], r[-1]) for r in rings), "fill rings must be closed"


def test_project_polygon_features_skips_empty():
    """_project_polygon_features ignores empty rings and projects the rest."""
    parts = [np.empty((0, 2)), np.array([(-10, -10), (10, -10), (10, 10), (-10, 10), (-10, -10)], float)]
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    rings = m._project_polygon_features(parts)
    assert len(rings) == 1 and np.isfinite(np.vstack(rings)).all()


def test_project_polygon_features_handles_multiple_parts():
    """Each ring part (e.g. an exploded MultiPolygon) contributes one fill ring."""
    parts = [
        np.array([(-20, -20), (-10, -20), (-10, -10), (-20, -10), (-20, -20)], float),
        np.array([(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)], float),
    ]
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    rings = m._project_polygon_features(parts)
    assert len(rings) == 2 and np.isfinite(np.vstack(rings)).all()


def test_land_fill_finite_on_globe(land_fc, mocker):
    """land() on a globe draws a finite, closed PolyCollection (Natural Earth mocked, no network)."""
    mocker.patch("digitalearth.scene.maps.decoration.natural_earth", return_value=land_fc)
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    pc = m.land()
    assert pc is not None and pc.get_paths()
    verts = np.vstack([p.vertices for p in pc.get_paths()])
    assert np.isfinite(verts).all()


def test_land_fill_preserves_extent_and_zorder(land_fc, dataset, mocker):
    """land() keeps the axes limits and sits below the data raster (background z-order)."""
    mocker.patch("digitalearth.scene.maps.decoration.natural_earth", return_value=land_fc)
    m = Map(crs=projections.orthographic(-75, 42), globe=True)
    img = m.imshow(dataset)
    xlim0, ylim0 = m.ax.get_xlim(), m.ax.get_ylim()
    pc = m.land()
    assert m.ax.get_xlim() == xlim0 and m.ax.get_ylim() == ylim0, "land() blew out the extent"
    assert pc.get_zorder() < img.get_zorder(), "land must draw beneath the data raster"


def test_ocean_below_land_zorder():
    """ocean() draws below land() (ocean is the deepest background layer)."""
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    ocean = m.ocean()
    assert ocean.get_zorder() < -1.5, f"ocean zorder should sit below land (-1.5), got {ocean.get_zorder()}"


def test_ocean_flat_uses_add_features(mocker):
    """On a flat map, ocean() draws the Natural-Earth ocean layer via add_features (not the disc shortcut)."""
    spy = mocker.patch(
        "digitalearth.scene.maps.decoration.add_features",
        side_effect=_add_features_drawing([(-10, -10), (10, -10), (10, 10), (-10, 10)]),
    )
    m = Map(crs=4326)  # flat
    m.ocean()
    spy.assert_called_once()
    assert spy.call_args.args[1] == "ocean"  # add_features(ax, "ocean", ...)
    assert m.ax.collections


def test_fill_globe_polygons_empty_returns_none():
    """_fill_globe_polygons short-circuits to None when there are no rings to draw."""
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    assert m._fill_globe_polygons([], facecolor="#ccc", zorder=-1.0) is None


def test_lakes_fill_on_globe_above_land(land_fc, mocker):
    """lakes() fills polygons on a globe and sits just above land (so lakes show on the land)."""
    mocker.patch("digitalearth.scene.maps.decoration.natural_earth", return_value=land_fc)
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    pc = m.lakes()
    assert pc is not None and pc.get_paths()
    assert np.isfinite(np.vstack([p.vertices for p in pc.get_paths()])).all()
    assert pc.get_zorder() > -1.5, "lakes should draw above land (-1.5)"


def test_rivers_drawn_as_lines_on_globe(mocker):
    """rivers() draws projected line segments (split at the limb) on a globe."""
    rv = [np.array([(-9, 39), (-8, 40), (-7, 41)], float)]
    mocker.patch("digitalearth.scene.maps.decoration.natural_earth", return_value=rv)
    m = Map(crs=projections.orthographic(-9, 39), globe=True)
    artists = m.rivers()
    assert artists and m.ax.lines


def test_land_fill_finite_on_cylindrical_frame(land_fc, mocker):
    """land() fills finite rings on a cylindrical (rectangular-boundary) framed map, not just a disc."""
    mocker.patch("digitalearth.scene.maps.decoration.natural_earth", return_value=land_fc)
    m = Map(crs=3857, globe=True)  # Web-Mercator boundary is a rectangle, not a circle
    pc = m.land()
    assert pc is not None and pc.get_paths()
    assert np.isfinite(np.vstack([p.vertices for p in pc.get_paths()])).all()


def test_land_flat_uses_add_features(mocker):
    """On a flat map, land() draws the Natural-Earth polygons via add_features (not the globe path)."""
    spy = mocker.patch(
        "digitalearth.scene.maps.decoration.add_features",
        side_effect=_add_features_drawing([(-10, -10), (10, -10), (10, 10), (-10, 10)]),
    )
    m = Map(crs=4326)  # flat -> _natural_earth flat branch even with polygon=True
    m.land()
    spy.assert_called_once()
    assert spy.call_args.args[1] == "land"
    assert m.ax.collections


def test_project_polygon_features_single_exterior_ring(mocker):
    """_project_polygon_features emits one finite ring per exterior ring (holes are dropped at the source)."""
    # natural_earth returns exterior rings only, so a holed polygon arrives here as its exterior alone.
    exterior = np.array([(-20, -20), (20, -20), (20, 20), (-20, 20), (-20, -20)], float)
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    rings = m._project_polygon_features([exterior])
    assert len(rings) == 1, f"a single exterior ring should yield one ring, got {len(rings)}"
    assert np.isfinite(np.vstack(rings)).all(), "exterior ring must be finite"


def test_globe_basemap_with_fills_saves_png(land_fc, dataset, tmp_path, mocker):
    """A globe base map (ocean + land + coastlines + data) frames and saves a non-empty PNG."""
    mocker.patch("digitalearth.scene.maps.decoration.natural_earth", return_value=land_fc)
    m = Map(crs=projections.orthographic(-30, 20), globe=True)
    m.ocean()
    m.imshow(dataset)
    m.land()
    out = tmp_path / "globe_fills.png"
    m.save(str(out))
    assert out.exists() and out.stat().st_size > 0
    assert m._framed is True
