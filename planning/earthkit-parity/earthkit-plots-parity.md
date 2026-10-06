# earthkit-plots parity gap analysis

What `earthkit-plots` provides that the Digital-Earth static (matplotlib) tier does not.

| | |
|---|---|
| **Compared** | `earthkit-plots` **1.0.4** (released 2026-08-26, ECMWF, Apache-2.0) |
| **Against** | `digitalearth.static` (local `main` + `claude/compassionate-mayer-2qyn9n`), `cleopatra` **0.41.0**, `pyramids-gis` **0.65.0** |
| **Date** | 2026-10-06 |
| **Method** | AST extraction of all public names from both sides, name-pairing across the full Digital-Earth stack, then manual adjudication. 447 `earthkit-plots` public names compared. |
| **Scope** | Rendering, layout, styling, metadata, geography. Data ingestion (`earthkit-data` vs `pyramids`) is excluded — see [Out of scope](#out-of-scope). |

> **Why the comparison covers three packages.** Digital-Earth delegates all data IO to `pyramids-gis` and its matplotlib glyphs/styling to `cleopatra`. A capability that `digitalearth.static` does not define may still be present in the stack, so every candidate gap was checked against all three before being recorded. One preliminary finding (`convert_units`) was withdrawn this way — `pyramids` handles unit conversion in 3 modules. Names that survive below are absent from **all three**, unless marked *Buried*.

---

## Summary

| Area | Gaps | Severity |
|---|---|---|
| [1. Axis control](#1-axis-control) | 11 | **High** — entire subsystem absent |
| [2. Mixed-panel figures](#2-mixed-panel-figures) | 7 | **High** — architectural |
| [3. Map reference layers](#3-map-reference-layers) | 8 | Medium — mechanical to add |
| [4. Style system](#4-style-system) | 14 + library | Medium–High |
| [5. CRS & domain intelligence](#5-crs--domain-intelligence) | 9 | Medium — high demo value |
| [6. Metadata templating](#6-metadata-templating) | 3 | Medium |
| [7. Smaller items](#7-smaller-items) | 8 | Low |
| **Total** | **~60 public names** | |

**No plot type is missing.** Every drawing method `earthkit-plots` offers exists in Digital-Earth, under the same or a different name. The gaps are in axis handling, figure composition, styling infrastructure, and projection/metadata intelligence — not in what can be drawn.

---

## Not gaps — name differences

These appeared in the raw diff but are renames or siblings. Listed so they are not chased twice.

| earthkit-plots | Digital-Earth | Where |
|---|---|---|
| `contourf` | `contours` (filled) | `static/maps/raster.py` |
| `imshow` | `field` | `static/maps/raster.py` |
| `gridlines` | `graticule` | `static/maps/projection.py` |
| `gridpoints` | `grid_points` | `static/maps/vector.py` |
| `shapes` | `polygons` | `static/maps/vector.py` |
| `scatter` | `points` / `charts.scatter` | `static/maps/vector.py`, `static/charts.py` |
| `quantiles` | `quantile_band` | `static/series.py` |
| `fill_between` | `envelope` | `static/series.py` |
| `bar`, `line`, `stripes`, `multiboxplot`, `envelope`, `boxplot` | same names | `static/charts.py`, `static/series.py` |
| `ax`, `fig`, `figure`, `current_ax` | `.ax`, `.fig` attributes | `static/scene.py` |
| `attribution` | declared **absent by design** | `static/capabilities.py` |

## Digital-Earth only

Capabilities Digital-Earth has that `earthkit-plots` does not. Recorded because parity runs both ways and these are the differentiation argument.

`choropleth` · `voronoi` · `cartogram` · `hexbin` · `quadtree` · `kde` · `sankey` · `nightshade` · `tissot` · `mesh`/`block` · `inset`/`mark_extent`/`locator` · `TexturedGlobe` · `rotate` · `facet` · `shared_colorbar` · serialisable `FigureSpec`/`LayerTree` with cross-backend replay · per-tier `Capabilities` declarations · `Scene.on_pick` · `OffLimbError` strict mode · windowed/decimated raster reads

---

## 1. Axis control

**Absent from all three packages — 0 references anywhere in the Digital-Earth stack.** The largest single gap. Callers currently reach `Map.ax` and use raw matplotlib.

| Method | Signature | Source | What it does |
|---|---|---|---|
| `Subplot.xlabel` | `(label, **kwargs)` | `components/subplots.py:785` | Set x-axis label |
| `Subplot.ylabel` | `(label, side, **kwargs)` | `components/subplots.py:710` | Set y-axis label, left or right |
| `Subplot.xticks` | `(frequency, minor_frequency, format, minor_format, period, labels, **kwargs)` | `components/subplots.py:333` | Tick locations + major/minor formatting, period-aware |
| `Subplot.yticks` | `(frequency, minor_frequency, format, minor_format, labels, **kwargs)` | `components/subplots.py:383` | As above for y |
| `Subplot.format_x_ticks` | `(format_spec)` | `components/subplots.py:234` | Apply a tick formatter |
| `Subplot.format_y_ticks` | `(format_spec)` | `components/subplots.py:260` | As above for y |
| `Subplot.fix_x_units` | `(units)` | `components/subplots.py:204` | Pin a unit to the axis; later layers convert into it |
| `Subplot.fix_y_units` | `(units)` | `components/subplots.py:219` | As above for y |
| `Subplot.axis` | `(key)` | `components/subplots.py:665` | Look up a registered axis **by display units or assigned name** |
| `Subplot.twinx` | `()` | `components/subplots.py:583` | Twin subplot sharing x, independent right y |
| `Subplot.twin_axis` | `(position, index)` | `components/subplots.py:480` | Context manager redirecting subsequent plot calls to a twin axes |
| `Figure.xlabel` / `ylabel` / `xticks` / `yticks` | `(*args, **kwargs)` | `components/figures.py:300–329` | Broadcast the above to every subplot |

There is also a dedicated `AxisView` class (`components/_axis_view.py`) carrying the per-axis versions.

**Note on `fix_*_units` + `axis`.** These are not cosmetic. Together they make a subplot unit-aware: an axis is registered under a unit, a later layer in a different unit is converted into it rather than silently mis-plotted. Digital-Earth has unit conversion in `pyramids` but no axis-side registry to route it through.

## 2. Mixed-panel figures

`earthkit-plots` `Figure` holds **heterogeneous** subplots — a map beside a timeseries beside a climatology, in one figure, through one API. Digital-Earth's `grid()` binds a `Map` to every axes (`static/figure.py`), so every panel is a map; `charts.py` and `series.py` are module-level functions taking an `ax`, not panel types.

**Consequence: a map-plus-timeseries figure cannot currently be built through the package.**

| Method | Signature | Source |
|---|---|---|
| `Figure.add_subplot` | `(row, column, **kwargs)` | `components/figures.py:337` |
| `Figure.add_map` | `(row, column, domain, crs, **kwargs)` | `components/figures.py:356` |
| `Figure.add_timeseries` | `(row, column, **kwargs)` | `components/figures.py:395` |
| `Figure.add_climatology` | `(row, column, **kwargs)` | `components/figures.py:487` |
| `Figure.subplot_titles` | `(*args, **kwargs)` | `components/figures.py:531` |
| `Figure.apply_to_subplots` | `(method)` | `components/figures.py:231` — decorator broadcasting a method to all subplots |
| `Figure.iterate_subplots` | `(method)` | `components/figures.py:249` — decorator iterating data and subplots together |

Closing this needs a panel protocol that `Map`, a chart panel and a temporal panel all satisfy, which `grid()` then lays out. It is the most architectural item on this page.

## 3. Map reference layers

Digital-Earth has `coastlines`, `borders`, `land`, `ocean`, `lakes`, `rivers` (`static/maps/decoration.py`). These are the additional ones, sourced from Natural Earth or GISCO. Mechanical to add — `cleopatra.basemap.reference` already reads Natural Earth.

| Method | Signature | Source | What it adds |
|---|---|---|---|
| `Map.cities` | `(density, labels, capital_cities, capital_cities_kwargs, medium_cities_kwargs, small_cities_kwargs, adjust_labels, **kwargs)` | `components/maps.py:1292` | Populated-place markers, tiered by size, with label collision adjustment |
| `Map.urban_areas` | `(*args, **kwargs)` | `components/maps.py:1252` | Urban-area polygons |
| `Map.administrative_areas` | `(*args, **kwargs)` | `components/maps.py:967` | Sub-national (admin level 1) boundary lines |
| `Map.disputed_boundaries` | `(*args, **kwargs)` | `components/maps.py:936` | Disputed and breakaway territory boundaries |
| `Map.nuts_regions` | `(level, *args, **kwargs)` | `components/maps.py:1024` | EU NUTS statistical regions from GISCO, by level |
| `Map.countries` | `(*args, **kwargs)` | `components/maps.py:1002` | Country **polygons** (Digital-Earth's `borders` draws lines) |
| `Map.map_units` / `unit_boundaries` | `(*args, **kwargs)` | `components/maps.py:1278`, `:898` | Natural Earth map-unit polygons / boundary lines |
| `Map.standard_layers` | `()` | `components/maps.py:1832` | One call applying the conventional layer set |

`Figure` mirrors `cities`, `countries`, `urban_areas`, `administrative_areas` and `standard_layers` across all map subplots (`components/figures.py:650–715`).

Also `Map.ancillary_layer` (`components/maps.py`) — the decorator these are built from, which resolves layers from several data sources. Worth reading before implementing the above.

## 4. Style system

### 4a. `Schema` — global style configuration (absent)

A process-wide style config with a context manager and matplotlib-stylesheet export. Digital-Earth's `base/autostyle/` resolves a style per field; it has no global schema layer.

| Method | Signature | Source |
|---|---|---|
| `Schema.set` | `(**kwargs)` | `schemas.py:301` |
| `Schema.use` | `(name)` | `schemas.py:350` — switch to a named schema |
| `Schema.reset` | `()` | `schemas.py:339` |
| `Schema.style_context` | `()` | `schemas.py:380` — matplotlib style context manager |
| `Schema.to_stylesheet` | `()` | `schemas.py:188` — export as a matplotlib stylesheet |

### 4b. `Style` methods (absent)

| Method | Signature | Source | What it does |
|---|---|---|---|
| `Style.save_legend` | `(data, label, filename, transparent, **kwargs)` | `styles/__init__.py:1196` | Write a standalone legend image with no figure |
| `Style.values_to_colors` | `(values, data)` | `styles/__init__.py:1129` | Resolve values to colours outside a draw |
| `Style.with_overrides` | `(**overrides)` | `styles/__init__.py:290` | Copy a style with parameters replaced |
| `Style.levels` | `(data)` | `styles/__init__.py:331` | Generate levels adapted to the data |
| `Style.disjoint` | `(*args, **kwargs)` | `styles/__init__.py:1184` | Disjoint legend variant |
| `Style.vector` | `(*args, **kwargs)` | `styles/__init__.py:1188` | Vector legend variant |
| `Style.extend` | `()` | `styles/__init__.py:352` | Colorbar extend handling |
| `Style.apply_scale_factor` | `(values)` | `styles/__init__.py:364` | Scale factor applied in the style layer |
| `Style.convert_units` | `(values, source_units, short_name)` | `styles/__init__.py:370` | Unit conversion in the style layer — *Buried:* `pyramids` converts units, but not through the style/colorbar path |

Plus the `to_*_kwargs` family (`to_contour_kwargs`, `to_contourf_kwargs`, `to_pcolormesh_kwargs`, `to_quiver_kwargs`, `to_scatter_kwargs`, `to_matplotlib_kwargs`, `to_add_geometries_kwargs`) — one style object rendering itself into any backend call's keywords. Digital-Earth's equivalent is `static/render_compat.py` + `style_fold.py`, which folds in the other direction. Comparable capability, different shape; recorded for completeness rather than as a gap.

### 4c. The style library — 134 field identities vs ~9

| | earthkit-plots | Digital-Earth |
|---|---|---|
| Field identities | **134** (`data/styles/identities/*.yml`) | **~9** (`base/autostyle/library/magics.yml`) |
| Paired styles | **135** (`data/styles/auto-styles/*.yml`) | same file |
| Total YAMLs | 270 | 2 |

Structure is a pair per field: `identities/<id>.yml` carries `criteria` (GRIB keys such as `levelist`, parameter names, units) and `auto-styles/<id>.yml` carries an `optimal` style name plus one or more named `styles` with explicit colour lists.

`cleopatra`'s 8 preset JSONs (`styling/data/*_presets.json` — weather, terrain, scientific, ncl, radar, ocean, builtin) are **colormap presets**, a different kind of object: they do not map a field identity to a canonical style. They do not close this gap.

This is a grind, not a design problem: ~125 YAML files, each independently addable. `magics.yml`'s header already says it is "a representative, extensible subset — add fields here, not in code."

## 5. CRS & domain intelligence

### `CRSOptimiser` — automatic projection selection (absent)

Hand it a data extent, get an appropriate projection. No user CRS choice required. Digital-Earth's `Map(crs=...)` always needs one and defaults to `3857`.

Classification predicates (`geography/optimisers.py:65–185`):

| Predicate | Rule |
|---|---|
| `is_global()` | area > 60% of the globe |
| `is_large()` | 20–60% |
| `is_small()` | < 20% |
| `is_polar()` | abs(central latitude) > 75° |
| `is_equatorial()` | abs(central latitude) < 25° |
| `is_landscape()` / `is_portrait()` / `is_square()` | aspect ratio ±20% |
| `standard_parallels()` | `geography/optimisers.py:106` |
| `mutate()` | `geography/optimisers.py:176` — walk to the better-fitting optimiser |

Dispatching to: `PlateCarree`, `LambertAzimuthalEqualArea`, `TransverseMercator`, `AlbersEqualArea`, `NorthPolarStereo`, `SouthPolarStereo`, via optimiser classes `Global`, `Equatorial`, `NorthPolar`, `SouthPolar`, `LargeEqatorial`, `Square`, `Landscape`, `Portrait`.

Self-contained, ~200 lines, no earthkit-specific dependencies. High demonstration value.

### `Domain` and `BoundingBox` (absent)

Digital-Earth's `static/domains.py` is 10 lines; `set_domain` resolves named regions only.

| Method | Signature | Source |
|---|---|---|
| `Domain.from_data` | `(cls, data)` | `geography/domains.py:265` — derive a domain from a dataset |
| `Domain.from_string` | `(cls, string, crs)` | `geography/domains.py:195` |
| `Domain.can_bbox` | `()` | `geography/domains.py:346` — whether the domain can slice data |
| `BoundingBox.from_geometry` | `(cls, geometry, source_crs, target_crs)` | `geography/bounds.py:25` |
| `BoundingBox.to_optimised_bbox` | `()` | `geography/bounds.py:190` — rebuild in the optimal CRS |
| `BoundingBox.to_bbox` | `(target_crs)` | `geography/bounds.py:255` — CRS-converting |
| `BoundingBox.contains_point` | `(point, crs)` | `geography/bounds.py:280` |

## 6. Metadata templating

Titles such as `"{variable_name} at {time}"` filled from data metadata. Digital-Earth's `Scene._title_for` takes literal strings; there is no placeholder expansion anywhere in the stack (0 references to `format_string`).

| Method | Signature | Source |
|---|---|---|
| `Subplot.format_string` | `(string, unique, grouped, axis)` | `components/subplots.py:2396` |
| `Figure.format_string` | `(string, unique, grouped)` | `components/figures.py:1464` |
| `Layer.format_string` | — | `components/layers.py` |

Supporting helper `has_placeholders(template)` in `_base.py`. The `unique`/`grouped` arguments handle multi-layer figures where a token resolves differently per layer.

## 7. Smaller items

| Missing | Source | Note |
|---|---|---|
| `LayerGroup` + `distinct_legend_layers` | `components/layers.py`, `subplots.py:831`, `figures.py:553` | Group layers sharing a style into a single legend entry |
| `Layer.reset_facecolors` | `components/layers.py` | — |
| `Figure.add_logo` / `Subplot.add_logo` | `figures.py:1651`, `subplots.py:321` | Organisation logo placement, backed by `ancillary.find_logo`. Digital-Earth has `stamp` / `WatermarkMixin` — related but not equivalent |
| `Map.image` | `components/maps.py:1419` | `(img, extent, origin, transform)` — place an arbitrary image |
| `Map.crs_name` / `Map.domain_name` | `maps.py:203`, `:212` | Human-readable names for auto-titles |
| `Subplot.wrap_longitudes` | `components/subplots.py:182` | **Buried.** Digital-Earth has `base/preprocess.wrap_longitude`; it is simply not exposed as a `Map` method. Cheapest item in this document |
| `Subplot.multiboxplot_legend` | `components/subplots.py` | Visual legend for the most recent multiboxplot |
| `Source.u` / `v` / `update_units` / `source_units` / `gridspec` | `sources/__init__.py` | Vector-source abstraction carrying u/v components with unit conversion |

---

## Broken upstream — do not implement

**`Figure.add_hovmoller` does not work in the shipped 1.0.4 wheel.**

`components/figures.py:473` executes `from earthkit.plots.temporal.hovmoller import Hovmoller`, but `earthkit/plots/temporal/` contains only `__init__.py`, `anchors.py`, `climatology.py` and `timeseries.py`. There is no `hovmoller.py` anywhere in the distribution. The call raises `ModuleNotFoundError`.

Hovmöller diagrams are documented in the method's docstring (time on one axis, pressure/height on the other, automatic axis inversion for pressure coordinates) but are not shipped. Treat as a non-gap until a release contains the module.

---

## Out of scope

Data ingestion was excluded: `earthkit-plots` delegates to `earthkit-data` (GRIB, BUFR, netCDF, zarr, HEALPix, reduced Gaussian) and Digital-Earth delegates to `pyramids-gis` (GDAL raster/vector, netCDF, STAC, processing). These are different bets by both projects, neither would adopt the other's, and comparing them produces rows nobody would action.

Likewise `earthkit.plots.interactive` (plotly `Chart`) — that is Digital-Earth's `interactive` tier, not the static one.

---

## Suggested order of work

Ranked by effort against value.

1. **Expose `wrap_longitudes` on `Map`** — minutes. The function already exists in `base/preprocess.py`.
2. **Natural Earth reference layers** (§3) — mechanical; `cleopatra.basemap.reference` already does the reading. `cities` with tiered density and label adjustment is the most involved; `administrative_areas`, `disputed_boundaries`, `countries`, `urban_areas` are straightforward. `nuts_regions` needs a GISCO fetch path.
3. **Axis control** (§1) — large, but it is what users hit first, and `Map.ax` escape-hatching is the current workaround. `fix_*_units` + `axis` depend on routing `pyramids`' unit conversion through a new registry.
4. **`CRSOptimiser`** (§5) — ~200 lines, self-contained, no new dependencies, strong demo value ("hand it data, get the right projection").
5. **Metadata templating** (§6) — moderate; `base/autostyle` already resolves the metadata the tokens need.
6. **Mixed-panel figures** (§2) — architectural. Needs a panel protocol satisfied by `Map`, charts and temporal panels, with `grid()` laying them out. Largest design change here.
7. **Style library** (§4c) — grind. ~125 YAML files, independently addable, no design work.
8. **`Schema`** (§4a) — a new subsystem; worth doing only after 1–6.

---

## Appendix — reproducing this

No GitHub or readthedocs access is needed; those hosts are blocked by the session's egress proxy. PyPI is reachable.

```bash
pip download earthkit-plots --no-deps -d ./wheels
pip download cleopatra pyramids-gis --no-deps -d ./wheels
cd ekp_src && unzip -q ../wheels/earthkit_plots-1.0.4-py3-none-any.whl
```

Then AST-walk `earthkit/plots/**/*.py` for public names, AST-walk `src/digitalearth/`, `cleopatra/` and `pyramids/` for the same (excluding `_vendor/`, `_licenses/`, `*.libs/`), pair by method name, and adjudicate the residue by hand.

**Caveats.**

- Name-pairing is a first pass only. It produced false positives in both directions: five apparent gaps (`stripes`, `envelope`, `quantiles`, `boxplot`, `climatology`) were present in `static/series.py` and `static/temporal/`, and `convert_units` was present in `pyramids`. Every row above survived manual adjudication, but **behavioural equivalence was not tested** — no Python environment was available in the session (`matplotlib`, `cleopatra` and `pyramids` are not installed; `pixi install` has not run). Rows marked as present may still differ in depth.
- Pinned to `earthkit-plots` 1.0.4. Re-run against any later release before acting on it.
