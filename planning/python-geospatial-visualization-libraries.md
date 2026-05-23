# Python Geospatial Visualization Libraries — Comprehensive Survey

A deep survey of the Python ecosystem for visualizing geospatial data: maps, rasters, vectors,
choropleths, big-data point clouds, interactive web maps, dashboards, and 3D globes.

- **Research date:** 2026-05-22.
- **Method:** live web research against PyPI, GitHub, and pypistats.org.
- **Figures:** "stars" = GitHub stargazers; "downloads/mo" = PyPI installs in the last 30 days
  (pypistats.org). Download counts are inflated by CI/mirrors and bundling (e.g. pydeck/altair
  ship inside Streamlit), so treat them as relative popularity signals, not human-user counts.
  All numbers are approximate snapshots and drift over time.

---

## 1. Executive summary

The ecosystem splits into **rendering tiers** that compose, rather than competing one-to-one:

1. **Static / publication maps** — matplotlib is the universal canvas; Cartopy adds true
   projections; GeoPandas/geoplot/rasterio/rioxarray put vectors and rasters on it; contextily
   adds basemap tiles; mapclassify supplies choropleth class breaks.
2. **Interactive web maps** — Folium (Leaflet, static HTML) and ipyleaflet (Leaflet, bidirectional
   Jupyter widget) are the bases; leafmap and geemap are high-level multi-backend aggregators;
   xyzservices/mercantile are the tile plumbing; maplibre brings WebGL/3D.
3. **GPU / big-data** — pydeck, lonboard, and keplergl all render on the deck.gl WebGL engine;
   datashader rasterizes billions of rows server-side; h3-py provides hexagonal binning.
4. **Plotting ecosystems & dashboards** — Plotly (+ Dash) and the HoloViz stack
   (hvPlot → HoloViews → GeoViews + Datashader → Panel); Bokeh underpins both HoloViews and Panel;
   Altair offers declarative choropleths.
5. **3D globes & scientific viz** — PyVista/VTK for genuine interactive 3D; PyGMT for perspective
   relief; Plotly/Cartopy orthographic for a 2D "globe look"; xarray/Iris for climate rasters.

**Key takeaways**
- There is no single "best" library — production stacks combine 4–6 of these.
- For a *true, interactive 3D globe* in maintained Python today, **PyVista** is the realistic choice;
  most "globe" results elsewhere are 2D orthographic projections (a disc, not a sphere).
- Several once-popular tools are now **dormant**: basemap (deprecated → Cartopy), geoplot, earthpy,
  cesiumpy, ipygany, and largely vaex.
- This repo (**Digital-Earth**) already sits in tier 1: its `StaticGlyph` builds on matplotlib via
  **cleopatra** and reads rasters via **pyramids**. See §8 for integration relevance.

---

## 2. Master popularity ranking (by PyPI downloads/month)

Ranked across all surveyed libraries. Use as a relative "most used" signal only.

| Rank | Library          | Tier / role                         | Downloads/mo | Stars  | Maintained   |
|-----:|------------------|-------------------------------------|-------------:|-------:|--------------|
| 1    | matplotlib       | Static base engine                  | ~217M        | ~22.8K | Yes          |
| 2    | plotly           | Interactive charts + maps           | ~65M         | ~18.4K | Yes          |
| 3    | altair           | Declarative charts/choropleth       | ~54M         | ~10.4K | Yes          |
| 4    | pydeck           | deck.gl GPU maps (Streamlit-bundled)| ~27.3M       | ~14.2K | Yes (moderate)|
| 5    | geopandas        | Vector dataframe + quick maps       | ~17.6M       | ~5.1K  | Yes          |
| 6    | xarray           | N-D / climate raster + plotting     | ~14.8M       | ~4.2K  | Yes          |
| 7    | h3               | Hex spatial index (binning)         | ~13.5M       | ~770   | Yes          |
| 8    | dash             | Plotly dashboard framework          | ~9.4M        | ~24.2K | Yes          |
| 9    | xyzservices      | Tile-provider registry              | ~9.2M        | ~230   | Yes          |
| 10   | bokeh            | Interactive charts + app server     | ~7.3M        | ~20.4K | Yes          |
| 11   | rasterio         | Raster I/O + plotting               | ~4.4M        | ~2.5K  | Yes          |
| 12   | panel            | HoloViz dashboard framework         | ~3.2M        | ~5.6K  | Yes          |
| 13   | folium           | Leaflet → static HTML maps          | ~3.0M        | ~7.4K  | Yes          |
| 14   | holoviews        | Declarative viz layer               | ~1.9M        | ~2.9K  | Yes          |
| 15   | hvplot           | High-level `.hvplot` API            | ~1.3M        | ~1.3K  | Yes          |
| 16   | cartopy          | Projection-aware static maps        | ~1.28M       | ~1.6K  | Yes          |
| 17   | rioxarray        | CRS-aware xarray raster             | ~1.1M        | ~620   | Yes          |
| 18   | mercantile       | Web Mercator tile math              | ~1.04M       | ~450   | Maintenance  |
| 19   | pyvista          | 3D scientific viz / globes          | ~915K        | ~3.7K  | Yes          |
| 20   | mapclassify      | Choropleth classification           | ~836K        | ~150   | Yes          |
| 21   | keplergl         | deck.gl GUI map widget              | ~670K        | ~11.8K | JS yes/py lag|
| 22   | contextily       | Basemap tiles for static maps       | ~563K        | ~585   | Yes          |
| 23   | osmnx            | OSM street-network analysis/viz     | ~556K        | ~5.7K  | Yes          |
| 24   | streamlit-folium | Folium-in-Streamlit bridge          | ~433K        | ~580   | Yes          |
| 25   | datashader       | Big-data rasterizer                 | ~430K        | ~3.5K  | Yes          |
| 26   | ipyleaflet       | Leaflet Jupyter widget              | ~363K        | ~1.5K  | Yes          |
| 27   | whitebox         | Geoprocessing backend (analysis)    | ~132K        | ~410   | Yes          |
| 28   | leafmap          | Multi-backend mapping aggregator    | ~114K        | ~3.7K  | Yes (active) |
| 29   | basemap          | Legacy projections (DEPRECATED)     | ~107K        | ~810   | No           |
| 30   | geemap           | Google Earth Engine maps            | ~103K        | ~4.0K  | Yes (active) |
| 31   | maplibre         | MapLibre GL WebGL/3D bindings       | ~76K         | ~100   | Yes          |
| 32   | geoviews         | Geographic HoloViews extension      | ~56K         | ~628   | Yes          |
| 33   | lonboard         | Arrow-native GPU vector maps        | ~39K         | ~950   | Yes (active) |
| 34   | earthpy          | Teaching raster helpers             | ~11K         | ~535   | Dormant      |
| 35   | geoplot          | High-level convenience maps         | ~6.1K        | ~1.2K  | Dormant      |

> Not download-ranked (no clean PyPI figure or distributed differently): **VTK** (very high,
> bundled binaries; ~3.2K stars; v9.6.2), **Iris** (~717 stars; v3.15.0), **cesiumpy** (dormant
> since 2016), **ipyvolume** (~2K stars, semi-active), **ipygany** (~494 stars, inactive),
> **vaex** (~8.5K stars, mostly inactive), **whiteboxgui** (stale since 2023).

---

## 3. Decision guide — which tool for which job

| Goal                                                        | Recommended primary tool(s)                          |
|-------------------------------------------------------------|------------------------------------------------------|
| Publication-quality static map with projection             | matplotlib + Cartopy (+ contextily basemap)          |
| Quick choropleth from a GeoDataFrame                        | GeoPandas `.plot(scheme=...)` (uses mapclassify)     |
| Raster / DEM / satellite imagery display                   | rioxarray / rasterio (+ Cartopy for projection)      |
| Shareable interactive map as a single HTML file            | Folium                                               |
| Interactive notebook EDA with Python feedback              | ipyleaflet (or leafmap default backend)              |
| Rapid multi-format / COG / STAC mapping, minimal code      | leafmap                                              |
| Google Earth Engine / remote sensing                       | geemap                                               |
| Millions of vector features, fast pan/zoom                 | lonboard (GeoArrow) or pydeck                        |
| Billions of points (density), server-side                  | datashader (+ HoloViews/Panel)                       |
| No-code GUI exploration of geo data                        | keplergl                                             |
| Hexbin aggregation / spatial indexing                      | h3-py (render via kepler/pydeck/lonboard)            |
| Interactive dashboard, production, rich components          | Dash + Plotly (+ dash-leaflet for tiles)             |
| Interactive dashboard mixing many plot libs, pure-Python    | Panel (+ hvPlot/GeoViews)                            |
| Declarative statistical + small choropleth views           | Altair (mind 5K-row default limit; no web tiles)     |
| True interactive 3D globe / planet                          | PyVista (`examples.planets`) / VTK                   |
| Perspective 3D relief, publication-grade                    | PyGMT `grdview`                                      |
| 2D "globe look" figure                                      | Plotly orthographic or Cartopy Orthographic          |
| Climate / NetCDF multidimensional fields                    | xarray (+ Cartopy) / Iris                            |
| Street networks / urban analysis                            | OSMnx                                                |
| GPS / vessel / wildlife trajectories                        | MovingPandas                                         |

---

## 4. Static / cartographic plotting

The classic publication-map stack. matplotlib renders; everything else feeds or projects onto it.

### 4.1 matplotlib
- **Summary / category:** general-purpose 2D plotting; the foundational engine for nearly all
  static maps.
- **Core purpose:** publication-quality static figures via a flexible Artist/Figure/Axes model.
- **Key features:** OO + pyplot APIs; vector geometry via `Path`/`PatchCollection`/`LineCollection`;
  raster via `imshow`/`pcolormesh`/`contourf`; colormaps, `Normalize`/`BoundaryNorm`, colorbars;
  export to PNG/PDF/SVG/EPS at any DPI; LaTeX text. No geodesy/CRS by itself.
- **Strengths:** universal substrate; total control; mature, stable, vast docs.
- **Limitations:** not map-aware (no projections/CRS); verbose; slow on very large vector sets.
- **Use cases:** base canvas for Cartopy/GeoPandas/geoplot; any precise print-quality figure.
- **Popularity:** v3.10.9; ~22.8K stars; ~217M dl/mo; actively maintained; License: Matplotlib
  (BSD-style).
- **Deps:** numpy, contourpy, cycler, fonttools, kiwisolver, pillow, pyparsing, python-dateutil.
- **Interop:** the render target for every other static-map library here.
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 5))
ax.imshow(raster_array, cmap="terrain")
fig.savefig("map.png", dpi=200, bbox_inches="tight")
```

### 4.2 Cartopy
- **Summary / category:** cartographic toolkit adding true map projections to matplotlib.
- **Core purpose:** define geographic CRS, reproject geometries/rasters, draw projected maps with
  coastlines/features/gridlines on matplotlib axes.
- **Key features:** `GeoAxes` via `projection=ccrs.<Projection>`; 30+ PROJ projections; on-the-fly
  reprojection with `transform=`; Natural Earth + GSHHS features; raster via `imshow`/`pcolormesh`;
  WMS/WMTS/web-tile basemaps; labeled gridlines; 10m/50m/110m feature resolutions.
- **Strengths:** correct geodesy; seamless matplotlib integration; free vector base features.
- **Limitations:** steeper learning curve (transform vs projection); slow on dense vectors; depends
  on GEOS/PROJ/Shapely build chain.
- **Use cases:** climate/earth-science maps, global/regional projected figures, gridded output.
- **Popularity:** v0.25.0; ~1.6K stars; ~1.28M dl/mo; actively maintained (UK Met Office/SciTools);
  License: BSD-3-Clause.
- **Deps:** numpy, matplotlib, shapely, pyproj, pyshp.
- **Interop:** extends matplotlib Axes; GeoPandas/geoplot can draw onto a `GeoAxes`.
```python
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
ax = plt.axes(projection=ccrs.Robinson())
ax.coastlines()
ax.scatter(lons, lats, transform=ccrs.PlateCarree())
```

### 4.3 GeoPandas (`.plot` / `.explore`)
- **Summary / category:** pandas extension for vector geodata with built-in plotting.
- **Core purpose:** hold geometries in a `GeoDataFrame`, run spatial ops, produce choropleth/symbol
  maps via `.plot()` (static, matplotlib) or `.explore()` (interactive, Leaflet/folium).
- **Key features:** `.plot(column=...)` choropleths; `scheme=` (quantiles, fisher_jenks, …) via
  mapclassify; legends/colorbars/`missing_kwds`/`markersize`; multi-layer overlays; reprojection via
  `to_crs`; reads GPKG/Shapefile/GeoJSON (pyogrio/GDAL).
- **Strengths:** fastest vector-to-map path; pandas-familiar; excellent classification + legends.
- **Limitations:** `.plot()` not projection-aware (pair with Cartopy); limited fine cartographic
  control; slow on very large data.
- **Use cases:** choropleths, exploratory vector mapping, multi-layer thematic maps.
- **Popularity:** v1.1.3; ~5.1K stars; ~17.6M dl/mo; actively maintained (NumFOCUS); BSD-3-Clause.
- **Deps:** pandas, shapely, pyproj, pyogrio, numpy.
- **Interop:** central hub — feeds geoplot, draws onto matplotlib/Cartopy, uses mapclassify for
  `scheme=`, adds basemaps via contextily.
```python
import geopandas as gpd
gdf = gpd.read_file("regions.gpkg")
gdf.plot(column="pop", scheme="quantiles", k=5, legend=True, cmap="viridis")
```

### 4.4 geoplot
- **Summary / category:** high-level declarative cartographic plotting over matplotlib + Cartopy +
  GeoPandas.
- **Core purpose:** one-call functions for common map types with projection support.
- **Key features:** `polyplot`, `choropleth`, `kdeplot`, `pointplot`, `voronoi`, `cartogram`,
  `quadtree`, `sankey`; Cartopy projections via `gcrs`; mapclassify schemes; contextily basemaps.
- **Strengths:** concise seaborn-like API for attractive maps.
- **Limitations:** **dormant** (last release 0.5.1, Mar 2022); small community; can lag upstream APIs.
- **Use cases:** quick exploratory thematic maps, teaching, KDE/cartogram/voronoi.
- **Popularity:** v0.5.1; ~1.2K stars; ~6.1K dl/mo; minimally maintained; License: MIT.
- **Deps:** matplotlib, geopandas, cartopy, mapclassify, contextily, seaborn, pandas.
- **Interop:** thin layer over Cartopy/GeoPandas/mapclassify/contextily; outputs matplotlib axes.
  *(Note: this repo's `StaticGlyph.plotCatchment` uses geoplot.)*
```python
import geoplot as gplt, geoplot.crs as gcrs
gplt.choropleth(gdf, hue="pop", scheme="quantiles", cmap="Greens",
                projection=gcrs.AlbersEqualArea(), legend=True)
```

### 4.5 contextily
- **Summary / category:** fetches web-map basemap tiles and adds them under static matplotlib maps.
- **Core purpose:** download XYZ/WMTS tiles for a bbox and overlay as a static basemap behind
  vector layers.
- **Key features:** `add_basemap(ax, source=..., crs=...)`; tile catalog via xyzservices (OSM,
  CartoDB, Esri, …); `bounds2img`, `Place`, tile caching, zoom control; reprojects to layer CRS.
- **Strengths:** trivially adds basemap context; large swappable catalog; caching.
- **Limitations:** requires internet; provider terms/attribution; static raster only.
- **Use cases:** street/satellite/terrain context behind choropleths and point maps.
- **Popularity:** v1.7.0; ~585 stars; ~563K dl/mo; actively maintained (GeoPandas org); BSD-3-Clause.
- **Deps:** matplotlib, rasterio, mercantile, xyzservices, requests, pillow, geopy, joblib.
- **Interop:** adds basemaps onto matplotlib/GeoPandas axes; used internally by geoplot's webmap.
```python
import contextily as cx
ax = gdf.to_crs(epsg=3857).plot(figsize=(8, 8), alpha=0.6)
cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
```

### 4.6 rasterio + rioxarray
- **Summary / category:** GDAL-backed raster I/O with matplotlib `show` (rasterio); CRS-aware raster
  on xarray (rioxarray).
- **Core purpose:** read/write/manipulate rasters and display them; rioxarray adds labeled-array,
  reprojection, clipping, and lazy/dask reads.
- **Key features:** rasterio `plot.show()`/`show_hist()`, windowed reads, masks/nodata, affine + CRS,
  reproject/resample/merge/clip, GeoTIFF/COG/many GDAL formats. rioxarray `.rio` accessor:
  `reproject`, `clip`, `reproject_match`, `write_crs`; plot via `DataArray.plot()`.
- **Strengths:** authoritative raster handling; xarray ergonomics + lazy ops; coordinate-correct axes.
- **Limitations:** plotting is utilitarian (no projection grids — pair with Cartopy); GDAL build
  sensitivity.
- **Use cases:** DEMs, satellite imagery, NDVI, masking, reprojection pipelines.
- **Popularity:** rasterio v1.5.0 (~2.5K stars, ~4.4M dl/mo, BSD-3); rioxarray v0.22.0 (~620 stars,
  ~1.1M dl/mo, Apache-2.0); both actively maintained.
- **Deps:** rasterio: numpy, affine, attrs, click, cligj, certifi (system GDAL). rioxarray: rasterio,
  xarray, pyproj, numpy.
- **Interop:** raster layers for Cartopy/matplotlib; basemaps for GeoPandas; rioxarray builds on
  rasterio; contextily uses rasterio for tile rasters.
```python
import rioxarray
da = rioxarray.open_rasterio("dem.tif").rio.reproject("EPSG:3857")
da.plot(cmap="terrain")
```

### 4.7 mapclassify
- **Summary / category:** classification-scheme library that bins values for choropleths
  (not a plotter).
- **Core purpose:** compute class breaks and assign observations to classes; powers `scheme=` in
  GeoPandas/geoplot.
- **Key features:** Quantiles, EqualInterval, FisherJenks(+Sampled), NaturalBreaks, StdMean, BoxPlot,
  Percentiles, MaximumBreaks, HeadTailBreaks, JenksCaspall, …; `classify()` API + diagnostics
  (ADCM, GADF); numpy/pandas-friendly.
- **Strengths:** comprehensive, well-tested; the de facto choropleth binning lib.
- **Limitations:** not a visualization tool; some schemes (FisherJenks) costly on large arrays.
- **Use cases:** choosing choropleth breaks; comparing classification strategies.
- **Popularity:** v2.10.0; ~150 stars; ~836K dl/mo; actively maintained (PySAL); BSD-3-Clause.
- **Deps:** numpy, pandas, scipy, scikit-learn, networkx.
- **Interop:** backs `scheme=` in GeoPandas/geoplot; breaks consumed for `BoundaryNorm`/legends.
```python
import mapclassify as mc
print(mc.Quantiles(gdf["pop"], k=5).bins)
```

### 4.8 earthpy
- **Summary / category:** educational helper toolkit for spatial / remote-sensing plotting.
- **Core purpose:** simplify plotting raster bands, RGB composites, histograms, hillshades, and
  fetching teaching datasets.
- **Key features:** `earthpy.plot` (`plot_bands`, `plot_rgb`, `hist`, `draw_legend`);
  `earthpy.spatial` (`stack`, `crop_image`, `normalized_diff` (NDVI), `hillshade`); data downloader.
- **Strengths:** lowers boilerplate for multispectral imagery; convenient RGB/NDVI helpers.
- **Limitations:** **dormant** (0.9.4, Oct 2021); teaching-oriented; superseded by rioxarray/rasterio.
- **Use cases:** Earth Lab courses, quick Landsat/NDVI/hillshade demos.
- **Popularity:** v0.9.4; ~535 stars; ~11K dl/mo; low/dormant; BSD-3-Clause.
- **Deps:** numpy, matplotlib, rasterio, geopandas, scikit-image, requests.
- **Interop:** wraps rasterio + matplotlib; increasingly superseded by rioxarray.

### 4.9 matplotlib-basemap (DEPRECATED)
- **Status:** officially deprecated; the matplotlib team recommends **Cartopy** as its successor. A
  community fork lingers on PyPI in maintenance-only mode — not recommended for new projects.
- **What it did:** many projections (Mercator, Lambert, orthographic, polar stereographic, …),
  `drawcoastlines`/`drawcountries`/`fillcontinents`/`drawmeridians`, `contourf`/`pcolormesh`,
  `bluemarble`/shaded relief; manual lon/lat→projection conversion via the `Basemap` callable.
- **Why migrate:** awkward API separate from matplotlib transforms; no active development; install
  friction; fully superseded by Cartopy.
- **Popularity:** v2.0.0 (2021, community fork); ~810 stars; ~107K dl/mo (legacy/CI); unmaintained;
  License: MIT (GEOS LGPL-2.1).
```python
from mpl_toolkits.basemap import Basemap          # legacy only — prefer Cartopy
m = Basemap(projection="ortho", lat_0=20, lon_0=10)
m.drawcoastlines(); m.fillcontinents()
```

**Typical static stack:** matplotlib (canvas) + Cartopy (projection) + GeoPandas (vectors via
mapclassify) + contextily (basemap) + rasterio/rioxarray (rasters). geoplot wraps several; basemap
and earthpy are legacy/teaching.

---

## 5. Interactive web maps (Leaflet / MapLibre / tiles)

### 5.1 Folium (Leaflet.js)
- **Summary / category:** Python→Leaflet.js bridge rendering interactive maps as standalone HTML.
- **Core purpose:** wrangle data in Python, visualize on a Leaflet map serialized to self-contained
  HTML (or inline in Jupyter); no JS required.
- **Key features:** many tile basemaps + custom XYZ/WMS; markers/popups/tooltips, circle markers,
  clustering; GeoJSON/TopoJSON, PolyLine/Polygon, `Choropleth` class with data joins; plugins (Draw,
  HeatMap, MarkerCluster, Fullscreen, MiniMap, TimestampedGeoJson, Search); LayerControl; dual maps;
  `Map.save()` to HTML; optional Selenium PNG export.
- **Strengths:** very mature, huge community; portable static HTML; tight pandas/GeoPandas integration.
- **Limitations:** one-way (Python→JS), limited bidirectional interaction; DOM/SVG rendering slow on
  big data; no native raster/COG or 3D.
- **Use cases:** shareable maps, reports, blog embeds, choropleths.
- **Popularity:** v0.20.0; ~7.4K stars; ~3.0M dl/mo; actively maintained; License: MIT.
- **Deps:** branca, jinja2, numpy, requests (bundles Leaflet.js).
- **Interop:** leafmap/`geemap.foliumap` expose folium backends; streamlit-folium embeds folium maps;
  uses xyzservices providers.
```python
import folium
m = folium.Map(location=[52.0, 4.3], zoom_start=8, tiles="CartoDB positron")
folium.Marker([52.1, 4.3], popup="Site A").add_to(m)
m.save("map.html")
```

### 5.2 ipyleaflet
- **Summary / category:** Jupyter widget bringing fully interactive, bidirectional Leaflet maps to
  notebooks.
- **Core purpose:** link a Leaflet map to the Python kernel for two-way interaction (clicks, draws,
  viewport changes) in Jupyter/JupyterLab/Voila.
- **Key features:** bidirectional sync of bounds/zoom/clicks; layers (TileLayer, WMSLayer, GeoJSON,
  Choropleth, Heatmap, MarkerCluster, ImageOverlay, VelocityLayer, AntPath, vector tiles, GeoData);
  controls (Draw, Measure, SplitMap, Layers, Search); link to ipywidgets via `link`/`observe`.
- **Strengths:** true interactivity/bidirectional flow; first-class ipywidgets citizen; clean
  Traitlets API (Project Jupyter).
- **Limitations:** needs a live kernel (not standalone HTML); heavier setup (widget extensions); DOM
  rendering on large vectors.
- **Use cases:** interactive EDA, AOI digitizing, Voila dashboards.
- **Popularity:** v0.20.0; ~1.5K stars; ~363K dl/mo; actively maintained; License: MIT.
- **Deps:** ipywidgets, traitlets, xyzservices, branca, jupyter-leaflet (JS).
- **Interop:** the engine under leafmap (default) and geemap; consumes xyzservices.
```python
from ipyleaflet import Map, Marker, DrawControl
m = Map(center=(52.0, 4.3), zoom=8)
m.add(Marker(location=(52.1, 4.3)))
dc = DrawControl(); m.add(dc)
m   # dc.last_draw holds drawn GeoJSON back in Python
```

### 5.3 leafmap
- **Summary / category:** high-level, low-code Jupyter mapping toolkit unifying multiple backends.
- **Core purpose:** interactive mapping + geospatial analysis with minimal code; backend-agnostic
  over ipyleaflet, folium, MapLibre, pydeck, kepler.gl, plotly, bokeh.
- **Key features:** pluggable backends (`leafmap`, `.foliumap`, `.maplibregl`, `.deck`, `.kepler`,
  `.plotlymap`, `.bokehmap`); add vector/raster/COG/PMTiles/point clouds; hundreds of basemaps via
  xyzservices; WMS/XYZ/TMS; split-map; drawing, clusters, choropleth, heatmaps, legends, time slider;
  COG/STAC via TiTiler, local raster via localtileserver; DuckDB/PostGIS queries; HTML export;
  3D MapLibre globe/terrain via the maplibre extra.
- **Strengths:** one API across many engines; strong cloud-native raster (COG/STAC/PMTiles) support;
  excellent docs; rapid release cadence.
- **Limitations:** large optional-dependency surface; feature parity varies by backend;
  Jupyter-oriented.
- **Use cases:** rapid EDA, teaching, COG/STAC visualization, cloud raster workflows.
- **Popularity:** v0.62.0; ~3.7K stars; ~114K dl/mo; very actively maintained; License: MIT.
- **Deps:** ipyleaflet, folium, xyzservices, whiteboxgui, ipywidgets; optional maplibre, titiler,
  localtileserver, pmtiles, duckdb, pydeck, keplergl, gdal, laspy.
- **Interop:** wraps ipyleaflet (default) + folium; shares codebase with geemap; bridges to MapLibre,
  pydeck, kepler.gl.
```python
import leafmap
m = leafmap.Map(center=[52, 4.3], zoom=8)
m.add_basemap("Esri.WorldImagery")
m.add_cog_layer("https://example.com/scene.tif", name="COG")
m   # m.to_html("out.html") to export
```

### 5.4 geemap (Google Earth Engine)
- **Summary / category:** interactive mapping/analysis of Google Earth Engine data in Jupyter.
- **Core purpose:** turn GEE's JavaScript-centric API into a Python + widgets workflow.
- **Key features:** `Map.addLayer(ee_object, vis_params, name)`; layer manager, inspector, drawing,
  time-series animation; JS→Python conversion; export EE images/tables to Drive/local; split-map,
  legends, charts, zonal stats; ipyleaflet + folium backends; cartoee for publication maps; timelapse
  (Landsat/Sentinel/MODIS/NAIP).
- **Strengths:** the de facto Python interface to interactive GEE; rich remote-sensing/time-series
  tooling; large community.
- **Limitations:** requires a GEE account/auth (service terms/quotas); heavy deps; Jupyter-centric;
  newer releases need Python ≥3.12.
- **Use cases:** land-cover change, vegetation/water indices, timelapse, EE prototyping.
- **Popularity:** v0.37.3; ~4.0K stars; ~103K dl/mo; very actively maintained; License: MIT.
- **Deps:** earthengine-api, ipyleaflet, ipywidgets, folium, xyzservices, geopandas.
- **Interop:** built on ipyleaflet/ipywidgets; leafmap is geemap's non-GEE spin-off (shared codebase).
```python
import ee, geemap
ee.Initialize()
m = geemap.Map(center=[52, 4.3], zoom=6)
m.addLayer(ee.Image("USGS/SRTMGL1_003"), {"min": 0, "max": 3000}, "DEM")
m
```

### 5.5 mercantile / xyzservices (tile utilities)
- **Summary / category:** low-level plumbing — mercantile for Web Mercator tile math, xyzservices for
  tile-provider specs. Neither renders maps.
- **mercantile features:** `tile(lng, lat, zoom)`, `bounds`, `xy_bounds`, `parent`/`children`,
  `quadkey`, `tiles(...)` enumeration; CLI. Web Mercator only; **maintenance mode** (see morecantile
  for other grids). v1.2.1; ~450 stars; ~1.04M dl/mo; BSD-3-Clause.
- **xyzservices features:** `xyzservices.providers` tree (OSM, CartoDB, Esri, Stamen, Stadia, …);
  URL templates, attribution, max_zoom, API-key placeholders; `.build_url()`, `requires_token()`;
  JSON-backed, **zero dependencies**. v2026.3.0; ~230 stars; ~9.2M dl/mo; actively maintained;
  BSD-3-Clause.
- **Interop:** xyzservices feeds basemaps into folium/ipyleaflet/leafmap/geemap/contextily; mercantile
  underpins tiling/raster pipelines (e.g. rio-tiler/titiler family).
```python
import mercantile, xyzservices.providers as xyz
t = mercantile.tile(4.3, 52.0, 8)
url = xyz.CartoDB.Positron.build_url()
```

### 5.6 maplibre (pymaplibregl)
- **Summary / category:** Python bindings for MapLibre GL JS — WebGL vector/3D maps, optionally with
  Deck.GL layers. (PyPI package `maplibre`; project historically "pymaplibregl".)
- **Key features:** vector/raster/GeoJSON sources; data-driven styling/expressions; layers (fill,
  line, circle, symbol, fill-extrusion 3D, heatmap, raster); 3D terrain, globe projection,
  pitch/bearing camera; mix Deck.GL layers over MapLibre basemaps; renders in Jupyter/JupyterLite/
  Marimo (anywidget) and Shiny for Python; HTML export.
- **Strengths:** WebGL performance for large/vector data; true 3D + globe; open-source (no Mapbox
  token lock-in); works across Shiny/Marimo/Jupyter.
- **Limitations:** smaller community/ecosystem; pre-1.0 API (0.3.x); MapLibre style-spec learning
  curve.
- **Use cases:** high-performance vector/3D web maps, Deck.GL big-data viz, Shiny/Marimo dashboards.
- **Popularity:** v0.3.6; ~100 stars; ~76K dl/mo; actively maintained; License: MIT.
- **Deps:** pydantic, jinja2; extras `[ipywidget]`/`[shiny]`/`[all]`.
- **Interop:** leafmap `.maplibregl` and geemap wrap it as their MapLibre backend.

### 5.7 streamlit-folium
- **Summary / category:** Streamlit component rendering Folium maps with bidirectional interaction.
- **Key features:** `st_folium(map, ...)` returns interaction state (`last_clicked`,
  `last_object_clicked`, `bounds`, `zoom`, `all_drawings`); `folium_static(map)` for display-only;
  supports Folium plugins; key-based caching to limit reruns.
- **Strengths:** drop-in bridge making Folium interactive in Streamlit; rich returned interaction data.
- **Limitations:** `st_folium` reruns can be costly on big maps; inherits Folium DOM-rendering limits.
- **Use cases:** interactive Streamlit geo dashboards, click-to-select, AOI-drawing apps.
- **Popularity:** v0.27.2; ~580 stars; ~433K dl/mo; actively maintained; License: MIT.
- **Deps:** streamlit, folium, branca, jinja2.

### 5.8 whitebox / whiteboxgui (analysis tier, not a renderer)
- **What they are:** whitebox wraps the WhiteboxTools (Rust) engine for terrain/hydrology/LiDAR/RS
  geoprocessing (500+ tools); whiteboxgui is an ipywidgets GUI to run them in Jupyter.
- **Relevance:** they produce data layers (GeoTIFF/Shapefile), not web maps. leafmap/geemap embed
  whiteboxgui so analysis outputs feed ipyleaflet/folium/MapLibre layers. Treat as the analysis tier.
- **Popularity:** whitebox v2.3.6 (~410 stars, ~132K dl/mo, MIT, active); whiteboxgui v2.3.0 (stale
  since 2023).
```python
import whitebox
wbt = whitebox.WhiteboxTools()
wbt.fill_depressions("dem.tif", "dem_filled.tif")
wbt.d8_flow_accumulation("dem_filled.tif", "facc.tif")   # then show in leafmap
```

**Rendering engines:** Folium/ipyleaflet/streamlit-folium = Leaflet (DOM/SVG); maplibre = MapLibre GL
(WebGL/3D); leafmap/geemap = multi-backend aggregators (shared codebase; geemap adds Earth Engine).

---

## 6. GPU-accelerated, big-data & deck.gl-style viz

### 6.1 pydeck
- **Summary / category:** official Python bindings for deck.gl; declarative WebGL/GPU layered maps.
- **Key features:** full deck.gl layer catalog (Scatterplot, Hexagon, Grid, Arc, Path, Polygon,
  GeoJson, Heatmap, Column, Trips, …); WebGL2 GPU rendering of 100K–millions of rows; 3D extrusion;
  time animation (TripsLayer); GPU/CPU aggregation; tooltips/picking; Carto/Mapbox/Google basemaps;
  consumes pandas/GeoJSON. No native GeoArrow path.
- **Strengths:** complete deck.gl access; well documented; integrates with Streamlit/Panel/Jupyter;
  JSON spec portable to JS.
- **Limitations:** serializes data as JSON to the browser (heavy for big frames); no Arrow zero-copy
  (unlike lonboard); moderate release cadence; minimal styling helpers.
- **Use cases:** dashboards, 3D city/hexbin viz, trip/flow animation, Streamlit/Panel map panes.
- **Popularity:** v0.9.2; lives in `visgl/deck.gl` (~14.2K stars); ~27.3M dl/mo (Streamlit-bundled);
  actively maintained; License: Apache-2.0.
- **Deps:** deck.gl (JS, bundled), pandas (typical), ipywidgets (optional `jupyter` extra).
- **Interop:** same engine as lonboard/kepler; h3-py IDs feed H3HexagonLayer; datashader rasters via
  BitmapLayer.
```python
import pydeck as pdk
layer = pdk.Layer("HexagonLayer", data=df, get_position=["lon", "lat"],
                  radius=200, elevation_scale=4, extruded=True, pickable=True)
view = pdk.ViewState(latitude=37.78, longitude=-122.42, zoom=11, pitch=45)
pdk.Deck(layers=[layer], initial_view_state=view).to_html("map.html")
```

### 6.2 lonboard
- **Summary / category:** GPU-accelerated, Arrow-native deck.gl binding for Jupyter (Development Seed).
- **Key features:** GeoArrow + GeoParquet path (binary, columnar, near zero-copy); deck.gl GPU
  rendering of millions of features; layers (Scatterplot, Path, Polygon) + extensions (Heatmap,
  Bitmap, Trips, Brushing, DataFilter); one-line `viz()` auto-detects geometry; reads GeoPandas/
  shapely/pyarrow (geopandas/pandas/shapely now optional → Pyodide-friendly); matplotlib/palettable
  colormaps.
- **Strengths:** best-in-class for huge vector data (Arrow transport ≫ pydeck JSON); lean install;
  works in lite/Pyodide; modern, active.
- **Limitations:** smaller layer surface than raw deck.gl; Jupyter-centric; less turnkey HTML export;
  younger/smaller community.
- **Use cases:** millions of GeoParquet features, lidar/point clouds, large trajectory/AIS data,
  GeoArrow pipelines.
- **Popularity:** v0.16.0; ~950 stars; ~39K dl/mo; actively maintained; License: MIT.
- **Deps:** arro3/pyarrow, anywidget + ipywidgets, deck.gl (bundled); optional geopandas, pandas,
  shapely, matplotlib, palettable.
- **Interop:** shares deck.gl with pydeck/kepler; consumes GeoArrow/GeoParquet; h3-py boundaries →
  PolygonLayer; complements datashader (vector vs raster).
```python
import geopandas as gpd
from lonboard import viz
gdf = gpd.read_parquet("large_dataset.parquet")   # millions of rows
viz(gdf)
```

### 6.3 keplergl
- **Summary / category:** Python/Jupyter widget for kepler.gl — a no-code geospatial GUI on deck.gl.
- **Key features:** rich GUI (style/color/size-by-field, filters, tooltips, split maps); layers
  (point, arc, line, hexbin, grid, heatmap, cluster, polygon/GeoJSON, trip, icon, 3D buildings, H3);
  time playback; GPU rendering of millions of points; accepts DataFrames/GeoDataFrames/GeoJSON/CSV via
  `add_data()`; save/load config JSON; HTML export; bidirectional config state in Jupyter.
- **Strengths:** powerful exploratory UI with zero rendering code; native H3 layer; reproducible
  config JSON.
- **Limitations:** Python widget lags the JS core (0.4.0 still RC); JSON/CSV data (no GeoArrow);
  heavyweight front end; limited programmatic fine-control.
- **Use cases:** visual EDA, trip/flow exploration, H3 hexbin dashboards, stakeholder demos.
- **Popularity:** v0.3.7 (0.4.0rc1 pre-release); `keplergl/kepler.gl` ~11.8K stars; ~670K dl/mo;
  JS core very active, Python widget moderate; License: MIT.
- **Deps:** kepler.gl (JS, bundled), ipywidgets/traitlets, pandas; geopandas/shapely for geo input.
- **Interop:** same deck.gl core as pydeck/lonboard; native H3 layer consumes h3-py; config JSON
  portable to the kepler.gl web app.

### 6.4 datashader
- **Summary / category:** server-side rasterization engine aggregating huge datasets into images
  (not a map widget).
- **Key features:** aggregates billions of rows into a canvas-sized 2D array (count/mean/max/
  categorical); Numba-accelerated CPU + **GPU via cuDF/CUDA**; out-of-core via Dask; glyphs (points,
  lines/time-series/trajectories, areas, polygons, rasters, quadmesh); perceptually uniform colormaps,
  log/eq-hist normalization, spreading; integrates with xarray/rasterio; outputs PIL/numpy images.
- **Strengths:** scales to billions where vector renderers choke; data-faithful (no overplot); tight
  HoloViz integration (zoom-triggered re-rasterization); GPU + Dask.
- **Limitations:** produces rasters/images, not interactive vector features (no per-feature picking);
  needs a plotting/tile layer for interactivity/basemaps; steeper conceptual model.
- **Use cases:** census/GPS/point-cloud density maps, network/edge bundling, trajectory heat, raster
  aggregation, dynamic re-rendering in HoloViews/Panel.
- **Popularity:** v0.19.1; ~3.5K stars; ~430K dl/mo; actively maintained (HoloViz); BSD-3-Clause.
- **Deps:** numpy, pandas, dask, numba, xarray, pillow, colorcet, param, toolz; optional cudf/cupy.
- **Interop:** rasters overlay in pydeck (BitmapLayer)/Folium/Bokeh/HoloViews/Panel; consumes Dask/
  pandas/Arrow; H3-binned aggregates as inputs.
```python
import datashader as ds, datashader.transfer_functions as tf
cvs = ds.Canvas(plot_width=1000, plot_height=1000)
agg = cvs.points(df, "x", "y", ds.count())
img = tf.shade(agg, cmap=["lightblue", "darkblue"], how="eq_hist")
```

### 6.5 h3 / h3-py
- **Summary / category:** Python bindings for Uber's H3 hierarchical hexagonal spatial index
  (enables viz, not a renderer).
- **Key features:** `latlng_to_cell`/`cell_to_latlng`/`cell_to_boundary`; 16 resolutions (~1000 km →
  <1 m); hierarchical parent/child; grid algorithms (`grid_disk`, `grid_distance`, `grid_path`,
  neighbors, compaction); polygon↔cells; vectorized NumPy/pandas/Arrow bulk ops; built on the C H3 v4
  core.
- **Strengths:** fast, uniform, hierarchical equal-area bins; language-agnostic standard supported by
  kepler.gl/deck.gl H3 layers, BigQuery, etc.; avoids grid/quadkey distortion.
- **Limitations:** not a viz tool (IDs/geometries only); 12 pentagons can't perfectly tile a sphere;
  v3→v4 API rename was breaking.
- **Use cases:** hex-bin density maps, demand aggregation, spatial joins, ML features, H3 layer inputs.
- **Popularity:** v4.4.2; ~770 stars; ~13.5M dl/mo; actively maintained (Uber); License: Apache-2.0.
- **Deps:** effectively none beyond NumPy (bundled C extension).
- **Interop:** cell IDs feed kepler/deck.gl/pydeck H3 layers; boundaries → PolygonLayer in lonboard/
  pydeck; aggregations join to pandas/Dask for datashader/lonboard.
```python
import h3
cell = h3.latlng_to_cell(37.78, -122.42, 9)
boundary = h3.cell_to_boundary(cell)
```

### 6.6 vaex (limited geo/viz relevance)
- **Summary:** out-of-core, billion-row Arrow/NumPy DataFrame with basic viz; tangential to geo.
- **Geo relevance:** fast N-D binning (`df.count`, `viz.heatmap`) gives datashader-like density
  rasters for lon/lat; a thin `vaex.geo` module offers coordinate transforms — but no GeoArrow
  geometry types, no map widget, no deck.gl.
- **Limitations:** **mostly inactive** (umbrella had no PyPI release in the prior ~12 months as of
  late 2025); thin geo features; rudimentary visualization.
- **Popularity:** v4.19.0 (umbrella); ~8.5K stars; declining; License: MIT.
- **Interop:** emits Arrow tables consumable by lonboard/GeoArrow pipelines; binned outputs → pydeck/
  datashader; coordinates → h3-py.

**Recommended big-data pattern:** index/aggregate with **h3-py** (and/or pre-aggregate huge tables in
vaex) → rasterize density with **datashader** OR render vectors directly with **lonboard** (large) /
**pydeck** (declarative) → use **keplergl** for no-code GUI exploration. All three renderers share the
deck.gl GPU engine.

---

## 7. Plotting ecosystems & dashboards

The HoloViz stack is **layered**, which clarifies the per-library notes:

```
hvPlot          (.hvplot high-level API; geo=True, rasterize=True)
   |  builds
HoloViews       (declarative annotated data -> plots; backend-agnostic)
   |  extends                         renders via
GeoViews ------ Cartopy/pyproj        Bokeh / Matplotlib / Plotly backends
   |  big-data overlay
Datashader      (server-side rasterization of huge data)
   |  app/dashboard layer
Panel           (serves any of the above as interactive web apps)
```

Dash is the parallel, Plotly-centric dashboard framework. Bokeh underpins both HoloViews' default
backend and Panel's server.

### 7.1 Plotly
- **Category:** browser-based interactive charting (plotly.js wrapper) with first-class map traces.
- **Map features:** tile maps via MapLibre `*_map` (`scatter_map`, `line_map`, `choropleth_map`,
  `density_map`); **legacy `*_mapbox` deprecated** → use `*_map`, `map_style`, `layout.map`; outline/
  geo maps via d3-geo (`choropleth`, `scatter_geo`) with real projections (natural earth, orthographic
  globe, mercator, albers); GeoJSON/feature-id choropleths; built-in country/US-state geometries;
  color scales, hover, animation frames, faceting; WebGL acceleration; selection/zoom/lasso; true
  interactivity via Dash callbacks.
- **Strengths:** beautiful output out of the box; no JS; huge feature surface; tile + projected-geo
  both supported; pairs natively with Dash.
- **Limitations:** large HTML payloads; no server-side rasterization (downsample/Datashader for big
  data); projected `choropleth` (d3-geo) and tile `choropleth_map` (MapLibre) are separate subsystems;
  verbose at the `go` level.
- **Popularity:** v6.7.0; ~18.4K stars; ~65M dl/mo; actively maintained; License: MIT.
- **Deps:** narwhals, packaging; optional pandas/geopandas, kaleido (static export), anywidget.
- **Interop:** embeds in Dash and in Panel (`pn.pane.Plotly`); independent of HoloViz.
```python
import plotly.express as px
fig = px.choropleth(df, locations="iso_alpha", color="gdp",
                    projection="natural earth", color_continuous_scale="Viridis")
```

### 7.2 Bokeh
- **Category:** Python→BokehJS interactive viz with a Tornado-based app server.
- **Map features:** `figure(x_axis_type="mercator")` + `add_tile("CartoDB Positron"|"OSM")` (data in
  EPSG:3857); `GeoJSONDataSource`; polygons via `patches`/`multi_polygons`; linked brushing/selection,
  hover, `ColumnDataSource` linking, Python+JS callbacks; WebGL glyphs; big data via Datashader.
- **Strengths:** fine-grained control; genuine app server with Python callbacks; composable linked
  views; foundation for HoloViews/Panel.
- **Limitations:** no built-in projections (reproject to 3857 yourself); more boilerplate than
  `px`/hvPlot; no native choropleth helper; tile maps only (no globe/ortho).
- **Popularity:** v3.9.0 (Python ≥3.10); ~20.4K stars; ~7.3M dl/mo; actively maintained; BSD-3-Clause.
- **Deps:** numpy, pandas, tornado, Jinja2, pillow, PyYAML, xyzservices, narwhals, contourpy.
- **Interop:** default HoloViews backend; Panel embeds Bokeh models and shares its server.

### 7.3 HoloViews
- **Category:** declarative, backend-agnostic viz layer ("annotate data, get plots").
- **Features:** wrap data in Elements (`Points`, `Path`, `Image`, `Polygons`, `Curve`), render via
  Bokeh/Matplotlib/Plotly; composition (`*`/`+`), `DynamicMap`/streams, linked selections; first-class
  Datashader integration (`datashade`/`rasterize`/`spread`). No projections itself (that's GeoViews).
- **Strengths:** concise, composable, backend-agnostic; superb big-data via Datashader; the
  conceptual core of the stack.
- **Limitations:** learning curve; needs GeoViews for CRS/projections; abstraction can obscure
  low-level styling.
- **Popularity:** v1.22.x; ~2.9K stars; ~1.9M dl/mo; actively maintained; BSD-3-Clause.
- **Deps:** numpy, pandas, param, pyviz_comms, panel, colorcet.
- **Interop:** engine beneath GeoViews and hvPlot; served by Panel.

### 7.4 GeoViews
- **Category:** geographic extension of HoloViews adding Cartopy projections + features.
- **Features:** Cartopy-backed projections (PlateCarree, Mercator, Orthographic/globe) with on-the-fly
  reprojection; geo Elements (`gv.Points/Polygons/Path/Image/QuadMesh`), `gv.feature` (coastline/
  borders/land/ocean), `gv.tile_sources`; reads GeoPandas/shapely/xarray; choropleths via
  `gv.Polygons`; Datashader integration retains CRS.
- **Strengths:** true cartographic projections (rare in interactive-JS stacks); GeoPandas/xarray
  native; inherits HoloViews composition + Datashader; Bokeh (interactive) + Matplotlib (publication)
  backends.
- **Limitations:** heavy install (Cartopy/GEOS/PROJ; usually conda-forge); smallest community of the
  group; docs assume HoloViews familiarity.
- **Popularity:** v1.15.x; ~628 stars; ~56K dl/mo; actively maintained; BSD-3-Clause.
- **Deps:** holoviews, cartopy (→ GEOS/PROJ), pyproj, shapely, numpy, param; optional geopandas,
  xarray, datashader.
- **Interop:** subclass of HoloViews; produced by `hvplot(geo=True)`; served by Panel.
```python
import geoviews as gv, geoviews.feature as gf
gv.extension("bokeh")
polys = gv.Polygons(gdf, vdims=["pop"]).opts(cmap="viridis", tools=["hover"])
(gf.coastline * polys).opts(projection="GOOGLE_MERCATOR", width=700)
```

### 7.5 hvPlot
- **Category:** high-level `.hvplot()` API for pandas/dask/xarray/geopandas returning HoloViews objects.
- **Features:** `df.hvplot.points(..., geo=True, tiles="CartoDB Positron")`,
  `gdf.hvplot.polygons(geo=True, c="value")` choropleths, `xr.hvplot.image(geo=True, projection=...)`;
  `geo=True` activates GeoViews/Cartopy; `tiles=` adds basemaps; `rasterize=`/`datashade=` activate
  Datashader; inherits HoloViews interactivity.
- **Strengths:** minimal code for full interactive geo maps; one API across data types; seamless
  big-data + projections by flag.
- **Limitations:** convenience layer (less low-level control); geo needs GeoViews/Cartopy installed.
- **Popularity:** v0.12.x; ~1.3K stars; ~1.3M dl/mo; actively maintained; BSD-3-Clause.
- **Interop:** thin API over HoloViews; pulls in GeoViews + Datashader; embeds in Panel.
```python
import hvplot.pandas  # noqa
gdf.hvplot.points(geo=True, tiles="CartoDB Positron", c="magnitude", rasterize=True)
```

### 7.6 Panel
- **Category:** HoloViz dashboard / web-app framework serving arbitrary Python objects and plots.
- **Features:** panes for Plotly, Bokeh, HoloViews/GeoViews/hvPlot, Folium, deck.gl/pydeck,
  Vega/Altair, Matplotlib, PyVista; reactive `param` callbacks, widgets, templates; `.servable()`;
  Bokeh/Tornado server; WebAssembly (Pyodide) export; linked interactivity + Datashader streaming.
- **Strengths:** library-agnostic (mix Plotly + GeoViews + Folium in one app); pure-Python reactivity;
  notebook→server continuity; strong big-data geo story.
- **Limitations:** overlapping API concepts (callbacks vs `param` vs `interact`); smaller component
  marketplace than Dash.
- **Popularity:** v1.8.x; ~5.6K stars; ~3.2M dl/mo; actively maintained; BSD-3-Clause.
- **Interop:** native host for the HoloViz stack and for Plotly/Altair/Folium/deck.gl; shares Bokeh's
  server.

### 7.7 Dash
- **Category:** Plotly's React+Flask framework for analytical web apps (no JS).
- **Features:** Plotly map figures in `dcc.Graph`; click/select/hover/relayout events drive callbacks;
  cross-filtering, linked maps; component ecosystem (`dash-leaflet` for Leaflet/tiles, `dash-deck` for
  deck.gl); server-side + clientside + pattern-matching callbacks; enterprise deploy.
- **Strengths:** production-grade, scalable, well documented; rich third-party components; tight Plotly
  integration.
- **Limitations:** more boilerplate than Panel/hvPlot; map abilities bounded by Plotly unless using
  dash-leaflet/dash-deck; no built-in server-side rasterization.
- **Popularity:** v3.x; ~24.2K stars (most-starred here); ~9.4M dl/mo; actively maintained; License: MIT.
- **Deps:** plotly, Flask, Werkzeug, (bundled) dash-html/core/table components.
- **Interop:** Plotly.js + Flask + React; can embed HoloViews via the HoloViews/Dash bridge.

### 7.8 Altair (Vega-Altair)
- **Category:** declarative statistical viz on Vega-Lite with basic geo support.
- **Features:** `mark_geoshape()` renders GeoJSON/TopoJSON/GeoDataFrames; choropleths via
  `encode(color=...)`; `project(type=...)` d3-geo projections (mercator, albersUsa, orthographic);
  declarative selections, pan/zoom, linked brushing, tooltips, layering.
- **Strengths:** clean terse grammar; real cartographic projections; great for linked statistical+map
  views; pairs with Panel/Streamlit.
- **Limitations:** default 5K-row limit (opt out / pre-aggregate); **no tile/basemap layers**; shape
  loading boilerplate; not a dashboard framework; whole spec ships to browser.
- **Popularity:** v5.x (6.x in dev); ~10.4K stars; ~54M dl/mo (Streamlit/JupyterLab bundling); actively
  maintained; BSD-3-Clause.
- **Interop:** standalone Vega-Lite; embeddable as `pn.pane.Vega` in Panel; common in Streamlit.

---

## 8. 3D globes, planetary rendering & scientific 3D viz

### 8.1 PyVista
- **Category:** Pythonic, NumPy-native API over VTK for 3D mesh viz — the realistic route to a true
  interactive globe.
- **3D/globe features:** texture-map a 2D Earth image onto a sphere; `examples.planets` ships real
  planet textures (Earth, Mars, Moon); load DEMs as `StructuredGrid`/`warp_by_scalar` for 3D relief;
  perspective + orthographic cameras, lighting, picking, animation; Jupyter (trame), headless/
  off-screen, GPU volume rendering.
- **Strengths:** far simpler than raw VTK; excellent gallery; true interactive 3D (rotate/zoom/dolly);
  publication-quality; large active community.
- **Limitations:** not geospatially aware (no CRS/projection — you do lon/lat→XYZ math); heavy VTK
  dependency; web embedding less seamless than JS tools.
- **Use cases:** FEA/CFD/medical meshes, DEM terrain, custom 3D planet/globe renders.
- **Popularity:** v0.48.4; ~3.7K stars; ~915K dl/mo; actively maintained; License: MIT.
- **Deps:** vtk, numpy, matplotlib, pillow, scooby, pooch; trame for Jupyter.
- **Interop:** built on VTK; consumes xarray/rasterio DEMs; complements Cartopy/PyGMT (2D) with real 3D.
```python
import pyvista as pv
from pyvista import examples
earth = examples.planets.load_earth(radius=6378)
pl = pv.Plotter()
pl.add_mesh(earth, texture=examples.load_globe_texture())
pl.show()
```

### 8.2 VTK
- **Category:** the foundational C++ 3D graphics/viz engine (Python bindings) that PyVista wraps.
- **Features:** full 3D rendering (meshes, textures, volumes, glyphs, streamlines, contours, cameras);
  texture-mapped spheres for globes; hundreds of filters; OpenGL/GPU volume rendering; off-screen.
- **Strengths:** battle-tested 30+ years; extremely capable; powers ParaView/3D Slicer/MayaVi/PyVista.
- **Limitations:** verbose, steep, un-Pythonic API; large install; no native geographic projections.
- **Popularity:** v9.6.2; ~3.2K stars (Kitware mirror); actively maintained (Kitware); BSD-3-Clause.
- **Interop:** underlies PyVista/MayaVi/ParaView; reads/writes VTK formats.

### 8.3 cesiumpy (Cesium.js bindings)
- **Category:** Python wrapper around Cesium.js for a true 3D WGS84 globe in Jupyter.
  **Note: effectively unmaintained (last release 0.3.3, 2016).**
- **Features:** genuine interactive 3D globe (terrain providers, imagery layers, time/CZML); entities
  (points/polylines/polygons); GeoJSON/KML.
- **Limitations:** long-stale; targets old Cesium; needs a Cesium Ion token; fragile with modern
  Jupyter. Prefer modern web tools (pydeck, lonboard, leafmap, kepler.gl, or Cesium.js directly).
- **Popularity:** v0.3.3; ~80 stars; very low downloads; dormant; License: Apache-2.0.

### 8.4 ipyvolume / ipygany
- **Category:** Jupyter widgets for 3D volume/glyph (ipyvolume) and mesh (ipygany) rendering.
  **Both largely dormant.**
- **Features:** ipyvolume — WebGL 3D scatter/quiver/volume/isosurface as ipywidgets; ipygany — VTK-
  backed mesh/scalar-field viz. Neither has native globe features (build a textured sphere manually).
- **Limitations:** ipyvolume last release 0.6.3 (Jun 2023, Jupyter-7 friction); ipygany inactive/
  discontinued; no geospatial awareness; small communities. PyVista (via trame) is the active
  alternative for in-notebook 3D.
- **Popularity:** ipyvolume ~2K stars (MIT, semi-active); ipygany ~494 stars (BSD-3, inactive).

### 8.5 "Globe look" via Plotly / Cartopy orthographic
- **Category:** 2D cartographic projection that *mimics* a 3D globe (a disc, not a rotatable sphere).
- **Options:** Plotly `geo.projection.type="orthographic"` (`Scattergeo`/`choropleth`); Cartopy
  `ccrs.Orthographic`, `ccrs.Geostationary`, `ccrs.NearsidePerspective` axes.
- **Behavior:** Orthographic = hemispheric disc; NearsidePerspective = satellite-altitude perspective;
  Geostationary = full-disc. Plotly's orthographic is drag-rotatable but still a 2D projection.
- **Strengths:** easy, lightweight, publication-quality; Cartopy adds coastlines/CRS reprojection;
  Plotly is web-interactive and shareable with no 3D engine.
- **Limitations:** not a real sphere — no perspective depth, no terrain relief, no mesh texture; only
  one hemisphere; no true 3D camera dolly.
```python
import matplotlib.pyplot as plt, cartopy.crs as ccrs
ax = plt.axes(projection=ccrs.Orthographic(0, 30))
ax.coastlines(); ax.stock_img()
```

### 8.6 PyGMT (Generic Mapping Tools)
- **Category:** Python interface to GMT for publication-quality maps, including 3D/perspective relief.
- **3D/globe features:** `Figure.grdview` 3D perspective surface/terrain of grids (DEMs) with drapes;
  `perspective=[azimuth, elevation]`; globe/perspective projection `G` (orthographic/perspective);
  relief shading, contouring, coastlines, CPTs; reads xarray DataArrays.
- **Strengths:** superb cartographic quality; native perspective-globe + 3D relief; geophysics-grade;
  numpy/pandas/xarray/geopandas integration.
- **Limitations:** static output (PNG/PDF/SVG) — not interactively rotatable; needs the external GMT C
  library; GMT-convention learning curve.
- **Use cases:** earth/ocean science maps, seismicity, bathymetry/topography 3D relief, globe
  perspective.
- **Popularity:** v0.18.0; ~859 stars; actively maintained; BSD-3-Clause.
- **Deps:** GMT (C lib) ≥6, numpy, pandas, xarray, netCDF4; optional geopandas.
```python
import pygmt
grid = pygmt.datasets.load_earth_relief(resolution="10m")
fig = pygmt.Figure()
fig.grdview(grid, projection="G30/20/12c", perspective=[150, 45], surftype="i", cmap="geo")
```

### 8.7 OSMnx
- **Category:** download/model/analyze/visualize OpenStreetMap street networks (mostly 2D viz).
- **Features:** fetch OSM networks as NetworkX graphs + GeoDataFrames; routing; building footprints;
  amenities; stats; `plot_graph`/`plot_figure_ground`/`plot_graph_route` (matplotlib); interactive via
  `.explore()` (2D Leaflet). No native 3D (export to PyVista/pydeck for 3D cities).
- **Strengths:** best-in-class OSM retrieval/analysis; concise API; strong docs/papers.
- **Limitations:** 2D-focused viz; Overpass rate limits; not a rendering engine.
- **Popularity:** v2.1.0; ~5.7K stars; ~556K dl/mo; actively maintained; License: MIT.
- **Interop:** outputs GeoPandas/NetworkX → MovingPandas/Cartopy/pydeck/PyVista.

### 8.8 MovingPandas
- **Category:** pandas/GeoPandas-based trajectory structures and movement analysis.
- **Features:** `Trajectory`/`TrajectoryCollection`; stop detection, generalization, smoothing,
  cleaning, splitting, aggregation; interactive maps via HoloViz (GeoViews/hvplot) and Folium (2D).
  No native 3D globe.
- **Strengths:** purpose-built trajectory API; rich analytics; good HoloViz integration.
- **Limitations:** 2D/2.5D viz; depends on heavier HoloViz stack for interactivity.
- **Popularity:** v0.22.4; ~1.4K stars; actively maintained; BSD-3-Clause.
- **Interop:** built on GeoPandas; consumes OSMnx context; plots via GeoViews/Folium.

### 8.9 xarray plotting + Iris
- **Category:** labeled N-D array (xarray) and Earth-science cube (Iris) libs for climate raster viz.
- **Features:** xarray `.plot()` → 2D pcolormesh/contour; with `subplot_kws` + Cartopy CRS → globe-look
  maps. Iris `iris.plot`/`quickplot` wrap matplotlib + Cartopy for projected maps. Both handle
  time/level/ensemble dims, faceting, lazy/dask arrays. No true 3D themselves ("globe" via Cartopy
  orthographic).
- **Strengths:** xarray = ecosystem standard for gridded climate data, dask scaling, huge adoption;
  Iris = strong CF/metadata handling, regridding, GRIB/NetCDF (UK Met Office grade).
- **Limitations:** plotting is 2D/projection-based; for interactive 3D fields export to PyVista
  (pyvista-xarray) or use hvplot/Datashader for large 2D.
- **Popularity:** xarray v2026.4.0 (~4.2K stars, ~14.8M dl/mo, Apache-2.0, very active); Iris v3.15.0
  (~717 stars, BSD-3, active).
- **Interop:** xarray↔Iris convert via `to_iris()`/`from_iris`; both plot through Cartopy; xarray feeds
  PyGMT and PyVista.
```python
import xarray as xr, cartopy.crs as ccrs, matplotlib.pyplot as plt
ds = xr.tutorial.open_dataset("air_temperature")
ax = plt.axes(projection=ccrs.Orthographic(-100, 40))
ds.air.isel(time=0).plot(ax=ax, transform=ccrs.PlateCarree())
ax.coastlines()
```

### 8.10 Building a globe in Python — realistic options

| Tool                          | Globe type              | Interactive 3D?    | Relief    | Texture |
|-------------------------------|-------------------------|--------------------|-----------|---------|
| PyVista / VTK                 | True 3D textured sphere | Yes (rotate/zoom)  | Yes (DEM) | Yes     |
| cesiumpy (Cesium.js)          | True 3D WGS84 globe     | Yes (lib dormant)  | Yes       | Yes     |
| PyGMT                         | Perspective globe (`G`) | No (static)        | Yes       | Drape   |
| Plotly orthographic           | 2D globe-look disc      | Rotatable, still 2D| No        | No      |
| Cartopy Ortho/NearsidePersp.  | 2D globe-look disc      | No (static mpl)    | No        | stock   |
| xarray/Iris (+Cartopy)        | 2D globe-look disc      | No                 | No        | No      |
| ipyvolume/ipygany             | Manual textured sphere  | Yes (dormant)      | Manual    | Manual  |

- **Genuine interactive 3D globe:** PyVista (recommended, maintained) or raw VTK — texture a sphere,
  warp DEMs for relief. Cesium-based stacks give a streaming WGS84 globe, but cesiumpy is dormant —
  prefer pydeck/lonboard/leafmap/kepler.gl or Cesium.js directly for production web globes.
- **2D "globe look" (a disc):** Plotly orthographic (web-interactive re-centering), Cartopy
  Orthographic/Geostationary/NearsidePerspective, xarray/Iris via Cartopy. Great for figures, not 3D.
- **3D perspective relief (static, cartographic-grade):** PyGMT `grdview` with `perspective=` and `G`.

---

## 9. Relevance to Digital-Earth (this repo)

Digital-Earth's `StaticGlyph` (in `src/digitalearth/static.py`) is a **tier-1 static plotter**: it
reads rasters via **pyramids** `Dataset` and renders arrays through **cleopatra** (`Array.plot`), which
itself sits on matplotlib. `plotCatchment` uses **geoplot** (now dormant). Mapping the README's stated
roadmap ("dynamic/interactive maps", and the `examples/` dirs: cartopy, contextily, holoview, webmap,
3d-maps) onto this survey:

- **Stay in the static tier (lowest effort, matches current design):** lean on **Cartopy** for true
  projections and **contextily** for basemaps behind `StaticGlyph` output; both compose with the
  existing matplotlib axes.
- **Interactive/web maps (the `webmap`/`holoview` examples):** **Folium** for shareable static HTML,
  or **leafmap** for a one-API multi-backend layer (and COG/STAC support that pairs naturally with
  pyramids rasters). **hvPlot + GeoViews** is the option that keeps projections *and* interactivity.
- **3D maps (the `3d-maps` examples):** for a real interactive globe/terrain, **PyVista** (DEM
  `warp_by_scalar`, `examples.planets`) is the maintained choice; **PyGMT** for static perspective
  relief. Avoid cesiumpy/ipygany (dormant).
- **Migrate away from geoplot** in `plotCatchment` if interactivity or longevity matters — its dormancy
  is a maintenance risk. GeoViews or a Folium/leafmap path are live alternatives.
- **Big-data point overlays:** if point counts grow, **datashader** (raster) or **lonboard**/**pydeck**
  (GPU vector) are the scale paths; **h3-py** for hexbin aggregation.

---

## 10. Sources

PyPI (`pypi.org`), GitHub (per-project repos), and pypistats.org, accessed 2026-05-22. Selected:

- matplotlib · github.com/matplotlib/matplotlib
- Cartopy · github.com/SciTools/cartopy · scitools.org.uk/cartopy
- GeoPandas · github.com/geopandas/geopandas
- geoplot · github.com/ResidentMario/geoplot
- contextily · github.com/geopandas/contextily
- rasterio · github.com/rasterio/rasterio · rioxarray · github.com/corteva/rioxarray
- mapclassify · github.com/pysal/mapclassify
- earthpy · github.com/earthlab/earthpy
- basemap · matplotlib.org/basemap (deprecated → Cartopy)
- Folium · github.com/python-visualization/folium
- ipyleaflet · github.com/jupyter-widgets/ipyleaflet
- leafmap · github.com/opengeos/leafmap · geemap · github.com/gee-community/geemap
- mercantile · github.com/mapbox/mercantile · xyzservices · github.com/geopandas/xyzservices
- maplibre (pymaplibregl) · github.com/eodaGmbH/py-maplibregl
- streamlit-folium · github.com/randyzwitch/streamlit-folium
- whitebox · github.com/opengeos/whitebox-python
- pydeck · github.com/visgl/deck.gl · deck.gl
- lonboard · github.com/developmentseed/lonboard
- keplergl · github.com/keplergl/kepler.gl
- datashader · github.com/holoviz/datashader
- h3-py · github.com/uber/h3-py
- vaex · github.com/vaexio/vaex
- Plotly · github.com/plotly/plotly.py · Plotly Mapbox→MapLibre migration guide
- Bokeh · github.com/bokeh/bokeh
- HoloViews · holoviews.org · GeoViews · geoviews.org · hvPlot · github.com/holoviz/hvplot
- Panel · github.com/holoviz/panel · Dash · github.com/plotly/dash
- Altair · github.com/vega/altair
- PyVista · github.com/pyvista/pyvista · VTK · github.com/Kitware/VTK
- cesiumpy · github.com/sinhrks/cesiumpy · ipyvolume · github.com/widgetti/ipyvolume
- PyGMT · github.com/GenericMappingTools/pygmt
- OSMnx · github.com/gboeing/osmnx · MovingPandas · github.com/movingpandas/movingpandas
- xarray · github.com/pydata/xarray · Iris · github.com/SciTools/iris
