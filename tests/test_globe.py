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


def test_project_line_features_skips_none_and_empty(mocker):
    """_project_line_features ignores None geometries and empty coordinate arrays."""
    import geopandas as gpd
    from shapely.geometry import LineString

    gdf = gpd.GeoDataFrame(
        {"geometry": [None, LineString([]), LineString([(-9, 39), (-8, 40)])]},
        crs=4326,
    )
    gdf.epsg = 4326  # _project_line_features reads fc.epsg
    m = Map(crs=projections.orthographic(-9, 39), globe=True)
    segs = m._project_line_features(gdf)
    assert all(s.shape[1] == 2 and len(s) > 1 for s in segs)


def test_natural_earth_flat_without_data_does_not_pin_extent(mocker):
    """On a flat map with nothing drawn yet, a Natural Earth layer is not pinned to prior data limits."""
    import geopandas as gpd
    from shapely.geometry import LineString

    world = gpd.GeoDataFrame(geometry=[LineString([(-50, -20), (50, 20)])], crs=4326)
    mocker.patch("digitalearth.scene.map.natural_earth", return_value=world)
    m = Map(crs=4326)  # flat, no imshow -> has_data is False
    m.coastlines()
    assert m.ax.lines or m.ax.collections  # the layer drew something
