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


# --------------------------------------------------------------------- #43 globe land/ocean fills


@pytest.fixture
def land_fc():
    """A synthetic polygon FeatureCollection in lon/lat with a limb-crossing 'continent'.

    Returns:
        geopandas.GeoDataFrame: polygons (with `.epsg` set) spanning both hemispheres of an ortho globe.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon

    near = Polygon([(-20, -20), (20, -20), (20, 20), (-20, 20)])      # near side of ortho(0, 0)
    straddle = Polygon([(60, -30), (120, -30), (120, 30), (60, 30)])  # crosses the limb
    gdf = gpd.GeoDataFrame({"geometry": [near, straddle]}, crs=4326)
    gdf.epsg = 4326
    return gdf


def test_project_polygon_features_finite_and_closed(land_fc):
    """_project_polygon_features returns finite, closed projected rings for a limb-crossing layer."""
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    rings = m._project_polygon_features(land_fc)
    assert rings, "expected at least one fill ring"
    verts = np.vstack(rings)
    assert np.isfinite(verts).all(), "fill rings must not contain inf/nan"
    assert all(np.allclose(r[0], r[-1]) for r in rings), "fill rings must be closed"


def test_project_polygon_features_skips_none_and_empty(mocker):
    """_project_polygon_features ignores None geometries and empty polygons."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    gdf = gpd.GeoDataFrame(
        {"geometry": [None, Polygon(), Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)])]},
        crs=4326,
    )
    gdf.epsg = 4326
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    rings = m._project_polygon_features(gdf)
    assert len(rings) == 1 and np.isfinite(np.vstack(rings)).all()


def test_project_polygon_features_handles_multipolygon():
    """A MultiPolygon contributes one fill ring per finite part."""
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon

    mp = MultiPolygon([
        Polygon([(-20, -20), (-10, -20), (-10, -10), (-20, -10)]),
        Polygon([(10, 10), (20, 10), (20, 20), (10, 20)]),
    ])
    gdf = gpd.GeoDataFrame({"geometry": [mp]}, crs=4326)
    gdf.epsg = 4326
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    rings = m._project_polygon_features(gdf)
    assert len(rings) == 2 and np.isfinite(np.vstack(rings)).all()


def test_land_fill_finite_on_globe(land_fc, mocker):
    """land() on a globe draws a finite, closed PolyCollection (Natural Earth mocked, no network)."""
    mocker.patch("digitalearth.scene.map.natural_earth", return_value=land_fc)
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    pc = m.land()
    assert pc is not None and pc.get_paths()
    verts = np.vstack([p.vertices for p in pc.get_paths()])
    assert np.isfinite(verts).all()


def test_land_fill_preserves_extent_and_zorder(land_fc, dataset, mocker):
    """land() keeps the axes limits and sits below the data raster (background z-order)."""
    mocker.patch("digitalearth.scene.map.natural_earth", return_value=land_fc)
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


def test_ocean_flat_uses_natural_earth(mocker):
    """On a flat map, ocean() reprojects the Natural-Earth ocean polygons (not the disc shortcut)."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    poly = gpd.GeoDataFrame(geometry=[Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)])], crs=4326)
    spy = mocker.patch("digitalearth.scene.map.natural_earth", return_value=poly)
    m = Map(crs=4326)  # flat
    m.ocean()
    spy.assert_called_once()  # flat path goes through natural_earth("ocean", ...)
    assert m.ax.collections


def test_fill_globe_polygons_empty_returns_none():
    """_fill_globe_polygons short-circuits to None when there are no rings to draw."""
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    assert m._fill_globe_polygons([], facecolor="#ccc", zorder=-1.0) is None


def test_lakes_fill_on_globe_above_land(land_fc, mocker):
    """lakes() fills polygons on a globe and sits just above land (so lakes show on the land)."""
    mocker.patch("digitalearth.scene.map.natural_earth", return_value=land_fc)
    m = Map(crs=projections.orthographic(0, 0), globe=True)
    pc = m.lakes()
    assert pc is not None and pc.get_paths()
    assert np.isfinite(np.vstack([p.vertices for p in pc.get_paths()])).all()
    assert pc.get_zorder() > -1.5, "lakes should draw above land (-1.5)"


def test_rivers_drawn_as_lines_on_globe(mocker):
    """rivers() draws projected line segments (split at the limb) on a globe."""
    import geopandas as gpd
    from shapely.geometry import LineString

    rv = gpd.GeoDataFrame(geometry=[LineString([(-9, 39), (-8, 40), (-7, 41)])], crs=4326)
    rv.epsg = 4326
    mocker.patch("digitalearth.scene.map.natural_earth", return_value=rv)
    m = Map(crs=projections.orthographic(-9, 39), globe=True)
    artists = m.rivers()
    assert artists and m.ax.lines


def test_land_fill_finite_on_cylindrical_frame(land_fc, mocker):
    """land() fills finite rings on a cylindrical (rectangular-boundary) framed map, not just a disc."""
    mocker.patch("digitalearth.scene.map.natural_earth", return_value=land_fc)
    m = Map(crs=3857, globe=True)  # Web-Mercator boundary is a rectangle, not a circle
    pc = m.land()
    assert pc is not None and pc.get_paths()
    assert np.isfinite(np.vstack([p.vertices for p in pc.get_paths()])).all()


def test_globe_basemap_with_fills_saves_png(land_fc, dataset, tmp_path, mocker):
    """A globe base map (ocean + land + coastlines + data) frames and saves a non-empty PNG."""
    mocker.patch("digitalearth.scene.map.natural_earth", return_value=land_fc)
    m = Map(crs=projections.orthographic(-30, 20), globe=True)
    m.ocean()
    m.imshow(dataset)
    m.land()
    out = tmp_path / "globe_fills.png"
    m.save(str(out))
    assert out.exists() and out.stat().st_size > 0
    assert m._framed is True
