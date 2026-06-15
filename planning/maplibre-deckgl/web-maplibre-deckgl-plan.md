# Tier Web — MapLibre + deck.gl Plan (interactive web maps, big-data & 3-D in the browser)

The **web** rendering tier — slippy/3-D web maps, choropleths, heatmaps, clustering, point clouds, 3-D tiles,
extrusions, terrain/globe, time animation, and self-contained shareable HTML — built on **MapLibre GL JS** (the
WebGL basemap/vector engine) + **deck.gl** (GPU layers for big-data & 3-D), driven from Python through their
**anywidget** bindings. This is the modern, dependency-light "shareable browser map without VTK" lane (the lonboard
/ MapLibre path the master plan listed as the optional web companion). It supersedes the older Plotly sketch
(`../interactive-3d/web-plotly-plan.md`): MapLibre + deck.gl cover the same web niche far more capably (it is the
exact stack the GeoLibre app uses).

> **This file IS Area D (the visualization web-map tier) of the GeoLibre-parity plan**
> (`../geolibre-parity/00-overview.md`) — the parity overview delegates the whole web-tier detail here. The GIS
> features the tier consumes are tracked (verified against pyramids 0.33.0) in
> `../geolibre-parity/pyramids-gis-io-and-conversions.md`; the shared symbology classifier lives in Area E
> (`../geolibre-parity/digitalearth-charts-and-symbology.md`, `DC.7`). Charts and non-map viz are Area E, not here.

> **Read `../interactive-3d/00-architecture-and-ingestion.md` first** — it owns the 🚫 HARD RULE (pyramids is the
> only GIS dep), the architecture (engines live in `digitalearth/web/`, reuse the `Source` abstraction), and the
> cross-cutting concerns. **One-line restatement:** all data comes from pyramids as NumPy + GeoDataFrame; **no**
> xarray/rasterio/any GIS competitor — enforced by `test_no_competitor_imports` (DX.3), which must be extended to
> scan `web/`.

- **Engine:** MapLibre GL JS + deck.gl, via the **`maplibre`** Python binding (py-maplibregl — wraps *both*
  MapLibre and deck.gl in one anywidget) ; optional **`lonboard`** (GeoArrow-fast deck.gl) for very large vector.
- **Module:** `src/digitalearth/web/` — base + capability **mixins** composed into `WebMap`, mirroring the v0.4.0
  `Map(GeoLayerBase, …)`, the M1 `Scene3D(Scene3DBase, …)` and the M2 `InteractiveMap(InteractiveMapBase, …)`.
- **Status:** in progress. **DW.0 is ✅ done (PR #96).** Sizes use the same XS/S/M/L scale as M1/M2.
- **Tier goal (one line):** `WebMap().basemap().choropleth(gdf, column="pop").save("map.html")` writes one
  self-contained, pan/zoom/hover web map (and, with deck.gl layers, 3-D) — from pyramids data, opening in any
  browser with nothing installed.

---

## ▶ DW.0 — implementation kickoff (start a fresh conversation here)

> This section is the hand-off for an implementation conversation. The detail is in **DW.0** under §Tasks below;
> this is the crisp entry point + facts verified in the planning thread (2026-06-14). Read DW.0 and the
> `three_d/` / `interactive/` tiers, then build.

**Paste-able starting prompt:**
> *Implement DW.0 of the web tier per `planning/maplibre-deckgl/web-maplibre-deckgl-plan.md`. Branch
> `feat/web-maplibre-deckgl-tier` from `origin/main`. Scaffold the `[web]` extra + `web` pixi env (do NOT add a
> `[tool.pixi.feature.web.pypi-dependencies]` table — pixi derives the feature from the extra), create
> `src/digitalearth/web/` (`__init__.py`, `base.py` `WebMapBase`, `map.py` `WebMap` composing empty mixins), extend
> `tests/test_no_competitor_imports.py` `_GUARDED_ROOTS` to include `src/digitalearth/web`, add tests
> (`tests/test_web_base.py`), run them in the `web` env, push, and open a PR using the `.github` PR template.*

**Verified facts to carry over (don't re-derive):**
1. **Bindings exist on PyPI:** `maplibre` **0.3.6** (py-maplibregl — MapLibre GL JS via `anywidget`) and `lonboard`
   **0.16.0** (deck.gl via `anywidget`). The `[web]` extra = `["maplibre >=0.3", "lonboard >=0.10"]`. **First
   step: `pixi install -e web`, then introspect the real `maplibre` API** (`from maplibre import …`) before writing
   `render()`/`save()` — do not guess the API surface.
2. **Mirror the existing tiers** exactly: `WebMap(WebMapBase, …empty mixins…)` like
   `Scene3D(Scene3DBase, …)` / `InteractiveMap(InteractiveMapBase, …)`; lazy engine import with an actionable
   `ImportError("install digitalearth[web]")`; `_to_display_source(data)` reprojects via pyramids `to_crs` to
   3857/4326 (the single CRS choke point — option A, no cartopy).
3. **pixi lesson (hard-won):** the extra auto-derives the `web` feature; re-declaring it as a pixi feature table
   double-declares every package and breaks `pixi update` ("<pkg> is already a dependency"). The `web` env =
   `{ features = ["dev", "web"], solve-group = "web" }`. Commit the `pixi.lock` editable-hash after the pyproject
   edit.
4. **DX.3 guard:** add `"src/digitalearth/web"` to `_GUARDED_ROOTS` in `tests/test_no_competitor_imports.py`.
   Tests: import-guard negative fixture + a smoke test (`WebMap().render()` is the widget; `.save(tmp/"m.html")`
   writes a file) + a lazy-import test (bare `import digitalearth.web` works without the extra).
5. **CI (later, not DW.0):** add `tests-web` / `notebooks-web` jobs symmetric with the `tests-3d` / `interactive`
   jobs (anywidget renders to HTML headless — **no browser/xvfb needed** for structural tests).

**Git state at hand-off (2026-06-14):** branch from `origin/main` (`6bcf457`). **PR #95 (`quickplot(backend="3d")`)
is still open/unmerged** — consider merging it first; it's independent of the web tier. Leave any stash on
`feat/quickplot-3d-backend` alone.

---

## 🎯 Scope — plot types, NOT a GIS app (read this before anything)

GeoLibre is a 40k-line **application** (desktop/web/Android). This tier ports its **visualization** only; the rest
is either pyramids' job or out of a viz library's charter. Set expectations here so the tier stays a *thin
renderer*, per `CLAUDE.md`.

**✅ In scope (this tier renders it):** basemaps/tiles · raster (COG/GeoTIFF/PMTiles/MBTiles) · vector
points/lines/polygons · choropleth + graduated/categorized symbology · heatmaps · point clustering · point clouds
· 3-D tiles · glTF models · polygon **extrusions** (3-D choropleth) · 3-D **terrain** + **globe** · time-slider
animation · hover/popups/legends · self-contained HTML export.

**❌ Out of scope (do NOT build here):**
- **GIS analysis** — SQL workspace, buffer/clip/union/spatial-join, hillshade/slope/contours, isochrones/OD
  matrices, H3, geocoding → **pyramids** (file an upstream issue for any gap; never reimplement here).
- **Data acquisition** — S3/GCS/Azure, Earth Engine, Planetary Computer, WMS/WFS → **pyramids' cloud/STAC readers**.
- **Application features** — AI segmentation (SAM), NL assistant, real-time collaboration, desktop/Android builds,
  print layouts, story-map presenter mode, project files → **not a Python plotting library's job.** (A minimal
  scroll/story *export* is a maybe-later nicety, not a goal.)

If a task seems to need an out-of-scope item, stop and route it to pyramids or drop it — exactly as M1/M2 did.

---

## Dependencies (extra `digitalearth[web]`)

```toml
[project.optional-dependencies]
# Web tier: MapLibre GL JS + deck.gl via their anywidget Python bindings. `maplibre` (py-maplibregl) drives both
# the MapLibre basemap/vector layers AND deck.gl overlays from one widget. lonboard is the GeoArrow-fast deck.gl
# path for very large vector/point data. None is a GIS competitor — they are renderers; pyramids stays the only GIS.
web = [
    "maplibre >=0.3",          # py-maplibregl: MapLibre GL JS + deck.gl + anywidget  (verify name/version at DW.0)
    "lonboard >=0.10",         # optional big-data deck.gl (GeoArrow); fold in like geovista in [3d]
    "pmtiles >=3",             # read/serve PMTiles archives for raster/vector tiles (if not transitive)
]
```
- Pure-pip, anywidget-based — **no native deps** (the whole point vs PyVista/VTK). Renders in Jupyter and exports a
  self-contained `.html`.
- **Pixi feature + env** `web`, separate solve-group (mirror the `3d`/`interactive` blocks — and per the pixi
  lesson, **do not** re-declare these as a `[tool.pixi.feature.web.pypi-dependencies]` table; pixi derives the
  `web` feature from this extra, the `web` env references it directly). Add `test-web` + `notebooks-web` tasks.
- **Headless export** may want a browser (selenium/playwright) for PNG snapshots in tests — gate that on a single
  smoke test; structural tests need no browser (assert the widget/spec, not pixels). Expect a small "one more
  backend" follow-up here, like M1's trame-vtk.

---

## Ingestion recipes (MapLibre/deck.gl, pyramids-native — NO xarray)

Each recipe consumes a Digital-Earth `Source` (or a pyramids object) and emits a layer the widget renders. MapLibre
+ tile basemaps render in **EPSG:3857 / lon-lat (EPSG:4326)** only — **reproject in pyramids first**
(`Dataset.to_crs(3857)` / vectors already lon-lat), exactly like the M2 projection rule. The IDs (W1–W7) are
referenced by the tasks.

### W1 — raster → tiles (the browser-native raster path)
Browsers stream tiles, not whole rasters. Convert in **pyramids** to a cloud-tiled form, then point the map at it:
```python
# pyramids does the GIS; we only hand the renderer a URL/array.
ds.to_crs(3857)                                  # MapLibre = Web-Mercator                  (pyramids ✅)
ds.to_cog("dem_cog.tif")                         # COG + XYZ tiling/overviews                (pyramids ✅ — to_cog/to_xyz/get_tile)
m.add_raster("dem_cog.tif")                      # MapLibre raster source (COG via titiler/local, or XYZ tiles)
# small rasters: normalize -> RGBA PNG data-URI as an image source (no tiling); large -> COG/XYZ.
# NB (verified 0.33.0): COG/XYZ/overviews + PMTiles *read* already exist; only a vector-tile/PMTiles *writer*
# and `to_terrain_rgb` are pyramids gaps — see ../geolibre-parity/pyramids-gis-io-and-conversions.md (PY-IO.8/9).
```

### W2 — vector → GeoJSON / GeoArrow source (points / lines / polygons)
```python
gdf = fc.to_crs(4326)                            # pyramids reproject; FeatureCollection *is a* GeoDataFrame
m.add_geojson(gdf, ...)                           # MapLibre fill/line/circle layer with a paint spec
# choropleth: a data-driven paint expression on `column` (step/interpolate) -> categorized/graduated symbology.
```

### W3 — big vector → deck.gl via lonboard (GeoArrow, millions of features)
```python
import lonboard                                   # deck.gl, zero-copy GeoArrow
layer = lonboard.ScatterplotLayer.from_geopandas(gdf)   # or PolygonLayer / PathLayer
# overlaid on the MapLibre basemap; no per-feature DOM — GPU-rendered.
```

### W4 — density → MapLibre heatmap / deck.gl aggregation
```python
m.add_heatmap(points_gdf, weight="value")         # MapLibre heatmap layer
m.add_cluster(points_gdf)                          # MapLibre clustered circle layer (count labels)
# or deck.gl HexagonLayer / ScreenGridLayer for GPU spatial binning.
```

### W5 — 3-D → deck.gl layers + MapLibre 3-D
```python
m.add_extrusion(gdf, height="pop", column="pop")  # fill-extrusion (3-D choropleth)
deck.PointCloudLayer(...) / deck.Tile3DLayer(url) / deck.ScenegraphLayer(gltf_url)   # point cloud / 3D tiles / glTF
m.set_terrain(dem_tiles) ; m.set_globe(True)       # 3-D terrain & globe projection (MapLibre)
```

### W6 — time → a slider that swaps the active source/filter
```python
# anywidget value -> update the layer's source/filter per frame (a DatasetCollection member or a time field).
m.timeslider(collection, kdim="time")
```

### W7 — export → one self-contained HTML
```python
m.save("map.html")                                 # embeds the widget state + data inline; opens offline, no server.
```

---

## Architecture — `WebMap` (base + mixins), mirroring `Map` / `Scene3D` / `InteractiveMap`

```
src/digitalearth/web/
  __init__.py        # exports WebMap (lazy — importing the package needs no maplibre)
  base.py            # WebMapBase — layer registry, display CRS + reproject->Source plumbing, the MapLibre widget,
                     #   add_layer/render/save/_repr_*; lazy `import maplibre` guarded with an actionable error
  raster.py          # RasterMixin     — add_raster (COG/PMTiles/image), basemap raster sources
  vector.py          # VectorMixin     — points / lines / polygons / choropleth (data-driven paint)
  bigdata.py         # BigDataMixin    — lonboard/deck.gl layers + heatmap/cluster + the size threshold
  threed.py          # ThreeDMixin     — extrusion / point cloud / 3D tiles / glTF / terrain / globe
  temporal.py        # TemporalMixin   — time-slider over a DatasetCollection
  decoration.py      # DecorationMixin — basemaps (tile_sources), legend, colorbar, popups/tooltips, controls
  export.py          # ExportMixin     — save() to self-contained HTML (+ optional PNG via headless browser)
  map.py             # WebMap(WebMapBase, RasterMixin, VectorMixin, BigDataMixin, ThreeDMixin, TemporalMixin,
                     #   DecorationMixin, ExportMixin) — thin composition
```
- **Reuse, don't rebuild:** consume `digitalearth/sources/` `Source` + extractors; add engine-specific *builders*
  (Source → MapLibre source/paint spec or deck.gl layer). No new data model.
- **One widget, two layer systems:** `maplibre` hosts MapLibre style layers *and* deck.gl layers in a single
  anywidget — so `WebMap` overlays both (MapLibre basemap + deck.gl big-data/3-D) without a second engine, matching
  GeoLibre's MapLibre+deck.gl architecture.
- **Lazy import** of `maplibre`/`lonboard` inside the methods that need them (like geovista in M1), so
  `import digitalearth.web` works without the extra; a missing extra raises a clear, actionable error.
- **CRS upstream in pyramids** — reproject to 3857/4326 before building a layer; never reproject in the renderer.

---

## Tasks

Each task has an **Objective**, a numbered build breakdown, the **public API** it adds, verifiable **acceptance
criteria**, and a **test list**. Sub-task IDs (DW.2a…) mark separable commits. Builders return `self` for chaining,
mirroring `Map`/`InteractiveMap`.

---

### DW.0 — packaging + `WebMap` skeleton + CRS plumbing + DX.3 · **M** · depends: none · ✅ **done (PR #96)**

**Objective.** Stand up the `web` extra/env and a `WebMap` host the mixins compose into — layer registry, the
MapLibre anywidget, reproject→display-CRS plumbing, `render()`/`save()`, lazy engine import — and extend the import
guard to `web/`, so every later task only adds a builder.

**Why it matters.** Foundation for DW.1+, and it settles the projection choice (reproject in pyramids → 3857/4326)
in one place.

**Files**
- *create:* `web/__init__.py`, `base.py`, `map.py`.
- *edit:* `pyproject.toml` — the `[web]` extra + pixi `web` feature/env (no duplicate feature table — see the deps
  note) + `test-web`/`notebooks-web` tasks; `pixi.lock` (regenerate, commit the hash).
- *edit:* `tests/test_no_competitor_imports.py` — add `"src/digitalearth/web"` to `_GUARDED_ROOTS`.

**Design & implementation**
1. **Packaging** mirroring the `interactive` block (extra → auto-feature → env); verify `maplibre`/`lonboard`
   resolve; re-confirm the real package name/version (it may be `maplibre`/`py-maplibregl`/`leafmap`).
2. **`WebMapBase`**: `self._layers` (ordered), display config (`center`, `zoom`, `crs`, `style`); `add_layer`
   returns `self`; `render()` returns the configured MapLibre widget (empty map if no layers); `save(path)` →
   self-contained HTML; `_repr_mimebundle_` for inline notebook display; `_require_maplibre()` lazy import with an
   actionable error.
3. **Display-CRS plumbing**: `_to_display_source(data, *, band=1)` builds a `Source` and reprojects through pyramids
   to 3857/4326 — the single choke point every builder calls. No cartopy/pyproj import.
4. **`WebMap`** composes the (initially empty) mixins; export from `__init__` and `digitalearth.web`.

**Public API (this task)**
```python
class WebMap(WebMapBase, ...):
    def __init__(self, *, center=None, zoom=2, style="dark", crs=3857, height=500) -> None: ...
    def add_layer(self, layer) -> "WebMap": ...
    def render(self) -> Any: ...                  # the maplibre anywidget
    def save(self, path: str, **kw) -> str: ...    # self-contained .html
    def show(self) -> Any: ...
```

**Acceptance criteria**
- [x] `from digitalearth.web import WebMap` works **without** the extra (lazy); calling a builder without it raises
      a clear `ImportError("install digitalearth[web]")`.
- [x] with the extra, `WebMap().render()` returns the MapLibre widget and `.save("m.html")` writes a >1 KB HTML
      page embedding the map. (Softened: maplibre's `to_html` CDN-references `maplibre-gl`, so it is not yet fully
      offline — true offline bundling is deferred to DW.6. See `upstream-maplibre-issues.md` MPL-2.)
- [x] `_to_display_source` reprojects a non-3857 raster via pyramids (no cartopy).

**Tests** (`tests/test_web_base.py`)
- import-guard negative fixture (a temp `web/` module importing `xarray` fails `test_no_competitor_imports`).
- smoke: `WebMap().render()` is the expected widget type; `.save(tmp/"m.html")` writes a file with the inline data.
- lazy import: monkeypatch the engine import to fail → builder raises; bare `import digitalearth.web` still works.

**Risks / notes** — pin the exact binding (the MapLibre-Python ecosystem has a few packages); confirm it bundles
deck.gl (or whether deck.gl needs `lonboard` separately). Self-contained HTML size for inline data — document the
practical limit.

---

### DW.1 — raster layers + basemaps · **M** · depends: DW.0

**Objective.** Put pyramids rasters on the web map as tiled/image sources, with tile basemaps underneath.

#### DW.1a — basemaps & tiles (`DecorationMixin`)
- *Implement:* `basemap(provider="CartoDark")`, `tiles(url, attribution=…)` (XYZ/WMTS), `pmtiles(path)`.
- *Acceptance:* `WebMap().basemap()` renders a pan/zoom slippy basemap.

#### DW.1b — raster sources (`RasterMixin`, recipe W1)
- *Implement:* `add_raster(data, *, band=1, cmap=…, opacity=…)` — small rasters → normalized RGBA image source;
  large → route to a **COG + XYZ tiles produced by pyramids** (`to_cog`/`to_xyz`/`get_tile` — all ✅ in 0.33.0).
  NoData → transparent.
- *Acceptance:* `m.add_raster(dem).basemap()` overlays a colour-mapped raster aligned on the basemap; opacity works.

**Public API**
```python
def basemap(self, provider="CartoDark") -> "WebMap"
def tiles(self, url, *, attribution="") -> "WebMap"
def add_raster(self, data, *, band=1, cmap="viridis", opacity=1.0, **opts) -> "WebMap"
```
**Acceptance / Tests** — each builder registers a layer of the expected type and returns `self`; assert the
generated MapLibre source/layer spec (type=`raster`/`image`, the url/data, paint opacity); mock any tile fetch.

**Risks** — true COG tiling in-browser usually needs a tile server (titiler) **or** PMTiles. Decide DW.1: ship the
**image-source** path first (works offline, size-limited), then the COG/XYZ path (pyramids ✅). A PMTiles/
vector-tile *writer* is the one pyramids gap (PY-IO.8) — only needed for very large layers; not a DW.1 blocker.

---

### DW.2 — vector layers + symbology · **M** · depends: DW.0

**Objective.** Render point/line/polygon GeoDataFrames with hover/popups and full thematic symbology
(single / categorized / graduated) via MapLibre data-driven paint.

- *Implement (`VectorMixin`, recipe W2):* `points`/`lines`/`polygons`; `choropleth(gdf, column, *, scheme=…,
  k=…, cmap=…)` that **calls the shared classifier** (Area E `DC.7`, `digitalearth/_classify.py`) to get the
  breaks/categories, then *compiles them into a MapLibre data-driven paint expression* (step/interpolate for
  graduated, match for categorized); `DecorationMixin.popup(fields)`/`tooltip(fields)`/`legend()`.
- *Acceptance:* `m.choropleth(gdf, column="pop").basemap()` is an interactive filled-polygon map with a legend +
  click popup; `points(gdf, value_column=…)` a graduated circle layer; the same `scheme`/`k` produce the same
  breaks as the static/interactive tiers' `choropleth`.
- *Tests:* assert the generated layer type + that the paint expression encodes the classifier's breaks/categories;
  legend entries match; mpl-free (inspect the spec, render a static PNG only in the one export smoke test).

**Risks** — **do not re-implement classification here.** The break/category *algorithms* (quantile/equal-interval/
Jenks/categorized) are the **single shared helper** owned by Area E (`DC.7`), reused by the static, interactive and
web `choropleth`. This tier only **compiles** the resulting breaks into a MapLibre paint expression — it owns no
classification logic of its own.

---

### DW.3 — big-data: heatmap, clustering, deck.gl · **M** · depends: DW.2

**Objective.** Render million-plus-feature layers as GPU heatmaps/clusters or deck.gl layers (via lonboard) instead
of one DOM glyph per feature.

- *Implement (`BigDataMixin`, recipe W3/W4):* `heatmap(gdf, *, weight=…)`, `cluster(gdf)`,
  `deck_scatter`/`deck_polygons`(gdf) via lonboard GeoArrow; an auto-threshold (feature count) that routes large
  `points`/`polygons` through deck.gl/cluster — **logged, never silent** (the M2 principle).
- *Acceptance:* a ~1e6-row GeoDataFrame renders as a smooth GPU layer; zoom stays responsive; the aggregator/weight
  changes the output.
- *Tests:* 1e5–1e6 synthetic points → a lonboard/deck layer (not raw circles) above the threshold; the threshold
  switch is logged; deck layer carries the GeoArrow table.

**Risks** — lonboard pulls `pyarrow`/GeoArrow; confirm it's a renderer dep (not used by us as a GIS engine). Pin
canvas size in any pixel test.

---

### DW.4 — 3-D: extrusions, point clouds, 3D tiles, glTF, terrain, globe · **M** · depends: DW.2

**Objective.** The GeoLibre 3-D surface — without VTK: 3-D choropleth extrusions, point clouds, 3D-tiles scenes,
glTF models, draped terrain, and globe projection.

- *Implement (`ThreeDMixin`, recipe W5):* `extrusion(gdf, height=…, column=…)` (fill-extrusion), `point_cloud(xyz)`
  (deck PointCloudLayer), `tiles_3d(url)` (deck Tile3DLayer), `gltf(url, lng, lat)` (deck ScenegraphLayer),
  `terrain(dem_tiles, exaggeration=…)` + `globe(True)` (MapLibre).
- *Acceptance:* `m.extrusion(gdf, height="pop", column="pop")` is an interactive 3-D choropleth; `point_cloud` and
  `tiles_3d` render in the browser; `globe(True)` switches to the spherical projection.
- *Tests:* each registers the expected MapLibre fill-extrusion / deck layer; extrusion height + colour encode the
  column; terrain/globe flags set on the map spec.

**Risks** — terrain tiles need a terrain-RGB source. **`to_terrain_rgb` is a confirmed pyramids gap** (verified
0.33.0 — see `../geolibre-parity/pyramids-gis-io-and-conversions.md` PY-IO.9): file it upstream, or use a public
terrain-RGB demo source for examples until it ships. 3D-tiles/glTF examples may need bundled/remote assets — keep
them optional in notebooks.

---

### DW.5 — temporal animation · **S** · depends: DW.2

**Objective.** Scrub a `DatasetCollection`/time-field through a slider on the web map (the GeoLibre time-slider).

- *Implement (`TemporalMixin`, recipe W6):* `timeslider(collection, *, kdim="time", labels=…)` — an anywidget slider
  whose value swaps the active source (per member) or sets a MapLibre filter on a time field; shared colour range.
- *Acceptance:* a multi-step stack renders with a working slider; the colour scale is stable across frames.
- *Tests:* a 3-step synthetic stack → a slider widget of length 3; first/last frame share the classification.

---

### DW.6 — export & sharing · **M** · depends: DW.1–DW.5

**Objective.** Produce a **single self-contained HTML** (and optional PNG) of any `WebMap` — the tier's headline
(shareable, server-free, no install).

- *Implement (`ExportMixin`, recipe W7):* `save(path)` → inline-data standalone HTML; `save(path, fmt="png")` via a
  headless browser (optional dep, gated). Document Panel-style inline-size limits; large data → reference a
  COG/PMTiles URL instead of inlining.
- *Acceptance:* `save("map.html")` opens offline with working pan/zoom/hover; PNG export works when the browser dep
  is present.
- *Tests:* `save(tmp/"m.html")` writes a >1 KB file containing the layer data; PNG path skipped when the browser
  dep is absent (clear skip, not a failure).

**Out of scope here:** a full GeoLibre "story map" with presenter mode — note as a possible future nicety, not a
goal.

---

## Tier-level Definition of Done

- [ ] DW.0–DW.6 acceptance criteria pass; `pixi run -e web test-web` green; `web/` is DX.3-clean (guard extended).
- [ ] A runnable `docs/examples/web/` notebook renders headless in a `notebooks-web` CI job (DX.2) — Bokeh-style:
      structural/mpl baselines, no browser needed for the suite.
- [ ] `quickplot(data, backend="web")` returns a `WebMap` (DX.1) — wired into `api._quickmap_web`, alongside
      `matplotlib`/`interactive`/`3d`.
- [ ] Docstrings on the full public surface; README/docs note the `[web]` extra and the in-scope/out-of-scope line.

## Suggested implementation order (PR slicing, mirroring M1/M2)

1. **DW.0** — extra/env + skeleton + CRS plumbing + DX.3 guard extension. *(one PR)*
2. **DW.1 + DW.2** — raster/basemaps, then vector + symbology (the core map). *(one PR)*
3. **DW.3** — big-data (heatmap/cluster/deck.gl). *(one PR)*
4. **DW.4** — 3-D layers. *(one PR)*
5. **DW.5 + DW.6** — temporal + export. *(one PR)*
6. **DX.1 + DX.2** — `quickplot(backend="web")` + docs notebook + `notebooks-web`/`test-web` CI jobs (symmetric
   with the `interactive`/`tests-3d` jobs). *(one PR)*

Build on `feat/web-maplibre-deckgl-tier`; commit+push per task, run `/test` + `/docstring` every two tasks,
`/review` at the end — the workflow that shipped M1 and M2.

---

## Cross-cutting (this tier)

- **HARD RULE / DX.3:** no xarray/rasterio/pyproj/cartopy/shapely/geopandas-as-engine imports in `web/`. All
  ingestion via W1–W7; **all CRS/reproject/resample/tiling/COG/PMTiles via pyramids**. Classification/binning for
  symbology is pure-numpy *styling*, allowed here.
- **Reproject upstream:** MapLibre/deck.gl + tiles are Web-Mercator/lon-lat; `_to_display_source` reprojects in
  pyramids before any layer is built.
- **Headless testing:** assert the generated **MapLibre style spec / deck layer objects** (structural), not pixels —
  no browser for the suite; one gated PNG-export smoke test for the headless-browser path. No network — mock tiles.
- **Reuse, don't rebuild:** consume `Source`/extractors; only add builders. cleopatra/pyramids are never edited —
  file upstream issues for gaps (COG/PMTiles/terrain-RGB exports are the likely ones).
- **Conventions:** max line 120; Black/isort 88; Conventional Commits; **no AI/Claude attribution**;
  `--no-gpg-sign`; feature branch only; commit the `pixi.lock` editable-hash after any pyproject edit.

---

## Checklist

| ID   | Status   | Title                                                   | Depends   | Effort |
|------|----------|---------------------------------------------------------|-----------|-------:|
| DW.0 | ✅ done   | packaging + `WebMap` skeleton + CRS + DX.3 guard        | —         |      M |
| DW.1 | 🔜 ready | raster layers + basemaps/tiles                          | DW.0      |      M |
| DW.2 | 🔜 ready | vector layers + symbology (choropleth/graduated/categ.) | DW.0      |      M |
| DW.3 | ⛔        | big-data: heatmap / cluster / deck.gl (lonboard)        | DW.2      |      M |
| DW.4 | ⛔        | 3-D: extrusion / point cloud / 3D tiles / glTF / globe  | DW.2      |      M |
| DW.5 | ⛔        | temporal animation (time-slider)                        | DW.2      |      S |
| DW.6 | ⛔        | export & sharing (self-contained HTML + PNG)            | DW.1–DW.5 |      M |

Status legend: ✅ done · 🔜 ready · ⛔ blocked · ⏸ deferred.

---

## Cross-tier follow-ups (DX.*)

- **DX.1 — `quickplot(backend="web")`:** add `_quickmap_web` dispatch in `api.py` (raster→`add_raster`, points→
  `points`/`heatmap`, polygons→`choropleth`), beside the existing `matplotlib`/`interactive`/`3d` branches. Now all
  four backends share one entry point.
- **DX.2 — docs + notebook:** `docs/examples/web/` gallery + a `notebooks-web` CI job in the `web` env (mirror
  `notebooks-interactive`).
- **DX.3 — import guard:** extend `_GUARDED_ROOTS` to include `src/digitalearth/web` in DW.0.

---

## Risks / open decisions

1. **Which MapLibre-Python binding** — `maplibre` (py-maplibregl) vs `leafmap.maplibregl` (built on it, more
   batteries) vs raw anywidget. Recommend **py-maplibregl** (it bundles MapLibre + deck.gl, the lean choice);
   confirm the package name + that deck.gl layers are first-class at DW.0.
2. **Raster delivery** — in-browser COG tiling needs a tile server (titiler) or PMTiles. Default to the **inline
   image-source** path, then COG/XYZ (pyramids ✅). **Confirmed pyramids gaps** (verified 0.33.0,
   `../geolibre-parity/pyramids-gis-io-and-conversions.md`): only `to_terrain_rgb` (PY-IO.9) and a vector-tile/
   PMTiles **writer** convenience (PY-IO.8) — PMTiles *read*, COG, XYZ tiling and overviews already exist.
3. **Self-contained HTML size** — inlining big data bloats the file; document the threshold and steer large data to
   a referenced COG/PMTiles URL.
4. **Scope creep toward "the GeoLibre app"** — the standing risk (see §Scope, and the GeoLibre-parity overview's
   ownership split). Every analysis/acquisition/app feature is a redirect to pyramids or a hard "no", not a task here.
5. **PNG export** needs a headless browser (selenium/playwright) — optional, gated; the test suite stays
   structural/browserless.

---

## Sources

- `../interactive-3d/00-architecture-and-ingestion.md` (HARD RULE, ingestion contract, DX.* tasks);
  `../interactive-3d/web-plotly-plan.md` (superseded — Plotly covered a subset of this niche less capably);
  `backend-selection-for-3d-geospatial-visualization.md` (lonboard/MapLibre listed as the web companions).
- Engines: MapLibre GL JS, deck.gl; Python bindings py-maplibregl (`maplibre`) / `leafmap.maplibregl`, `lonboard`.
- GeoLibre (opengeos) as the capability reference for the in-scope plot-type list. **Re-verify all package
  names/versions at DW.0** (the MapLibre-Python ecosystem moves fast).
```
