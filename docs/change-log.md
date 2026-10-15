# Change log

## 0.13.0 (2026-10-08)

### Feat

- **three_d**: freeze the colour scale across animation frames with record(clim=) (#448)
- **interactive**: complete the eight partial HoloViz capability rows (#447)
- **three_d**: clear the unblocked 3-D capability backlog (#429)
- **web**: add the web tier's own standalone-HTML export (swipe, minimap, measure) (#427)

## 0.12.0 (2026-10-07)

### Feat

- **interactive**: streaming / live-updating layers (IN-17) (#425)
- **interactive**: clear the HoloViz capability backlog (IN-1..IN-18) (#408)
- **web**: persist the vector time-slider in exported HTML and add a basemap gallery (#407)
- **web**: add vector tiles, controls, reference lines, MP4 export and cluster styling (#397)

### Fix

- **static**: warn on off-band center=, normalise blank titles, validate inset extent (#394)

## 0.11.0 (2026-10-05)

### BREAKING CHANGE

- WebMap.field, rgb_composite and hsv_composite take
tiles=TileRoute(kind, path, zooms=None) in place of tiles=, tiles_path=
and zooms=. A bare route name such as tiles="xyz" raises a TypeError
naming the new spelling. WebMap.terrain_tiles is unchanged.
- every deprecated spelling is removed rather than aliased.                                        
  Methods: `imshow`, `scatter`, `shapes`, `set_extent`, `contour` and `contourf`                                    
  on the static tier; `image`, `path`, `add_element` and `filled_contours` on the                                   
  interactive tier; `add_raster`, `fit_bounds`, `title`, `terrain`, `globe`,                                        
  `animate`, `to_gif` and `save_gif` on the web tier; `animate` on the 3-D tier.                                    
  Keywords: `alpha`, `duration`, `framerate`, `layer_ids`, `point_size`, `radius`,                                  
  `rasterize_threshold`, `scale`, `size` on label builders, `string` and                                            
  `value_column`. Also removed: the `StaticGlyph` entry point, the                                                  
  `digitalearth.scene` forwarding module, the moved-submodule aliases, and the                                      
  `api.imshow`, `api.scatter`, `api.contour` and `api.contourf` wrappers, which                                     
  become `api.field`, `api.points` and one `api.contours(filled=)`. The static                                      
  `set_bounds` now reads a bare sequence as `(west, south, east, north)` rather                                     
  than matplotlib's `[xmin, xmax, ymin, ymax]`, and its first parameter is named                                    
  `bounds` rather than `bbox`.                                                                                      
                                                                                                                    
  Closes #189, #207, #262, #264, #266, #333, #337
- Scene3DBase.display_crs is no longer always None, and a
globe refuses a scene drawn in another CRS. Camera.vertical_exaggeration
defaults to None rather than 1.0, so to_dict() writes null for a camera
that names none. WebMap.add_layer draws a layer under the id its object
carries: a colliding id is refused rather than suffixed, and a name=
that contradicts the object's id is refused rather than quietly winning.
WebMap.graticule() defaults to 30 degrees rather than 10.
LayerTree.replace re-places a layer whose new description changes its
band, at the top of the band it now belongs to. WebMap gains close() and
context-manager support, and a closed map lets go of the in-memory data
it registered.
- cleopatra[tiles] now requires >=0.39.0, the release that
introduced WatermarkMixin. Several inputs that were silently accepted now
raise: Map.set_extent rejects a sequence that is not exactly four values
(extras were ignored), Map.set_domain rejects a back-to-front bbox
(it produced a globe-spanning extent), an explicit vmin or vmax that
excludes every value in the data is refused rather than widened away,
and point styling passed with no points= array is named rather than
dropped.
- TexturedGlobe.points() returns a collection holding
every point given, with the ones facing away drawn at size zero, where
it used to return only the visible ones. Code counting
scatter.get_offsets() to find how many markers show should ask
TexturedGlobe.visible() instead.
- orbit(viewup=...) now shapes the orbital path as well
as the camera, so an existing caller that passed it renders a different
fly-through; and orbit(n_frames=...) below 3 raises ValueError where
pyvista previously clamped it to 3.

### Feat

- **static**: clear the static tier's remaining capability rows and make the tier chain (#389)
- clear the capability hit-list across the static, web and 3-D tiers (#372)
- **web**: classify raster fields with scheme=/k= on every 2-D tier (#361)
- complete the S7 spine -- the conformance to_backend probe (U-4) and the three ratchets (U-5) (#358)
- add to_backend to render a FigureSpec on any backend, and draw it cross-tier across the 2-D tiers (U-6) (#355)
- land Wave 8 — register installed plugins, collapse api.py, and drive every backend from ops (#352)
- finish Wave 7 — the colour key becomes a guide on its layer's encoding (#346)
- land Wave 7's layer management, framing and large-data paths, and delete every deprecation path (#339)
- open Wave 7 with the behavioural conformance suite, and settle the contract decisions it was blocked on     (#318)
- render every tier from its FigureSpec, and refuse from the declarations (#311)
- render every tier from a FigureSpec, and declare what each draws (#304)
- **base**: describe a scene as layers, views and targets that round-trip (#285)
- **base**: add a re-readable data tier and fold three duplications (#280)
- **base**: one shared vocabulary for bounds, scales, slices, data and style (#275)
- close out Wave 0 and hold the four backends to one contract (#227)
- **web**: legend, framing, layer control, labels, composites, contours and animation export (#211)
- **static**: give the 3-D globe reference layers that turn with it (#212)
- add an all extra, drop pyramids-eo, and mirror src/ in tests/ (#197)
- **3d**: let orbit shape the path its camera flies (#182)
- **basemaps**: add keyed-XYZ basemaps, starting with Planet NICFI  (#180)
- **static**: animate RGB/HSV composites on one frozen stretch (#162)
- **web**: scrub a DatasetCollection with the time-slider (#163)

### Fix

- close out Wave 0 — one stack colour-range rule, one meaning for colorbar= (#269)
- **static**: draw empty instead of crashing when data is off the view (#181)
- **3d**: take pyvista's trame stack from pyvista[jupyter] (#166)
- **web**: encode date-like columns before the frame becomes GeoJSON (#177)
- **web**: encode date-like columns before the frame becomes GeoJSON                                               
                                                                                                                    
  WebMap hands the caller's GeoDataFrame straight to maplibre's                                                     
  add_source, which serialises it with json.dumps -- and GeoJSON has no                                             
  date type, so any datetime column raised                                                                          
                                                                                                                    
      TypeError: Object of type Timestamp is not JSON serializable                                                  
                                                                                                                    
  and took the whole map with it. That is not an edge case: a timestamp                                             
  is the norm for the vector data this tier renders -- alerts,                                                      
  detections, observations -- and timeslider is hit hardest, since the                                              
  kdim it scrubs is usually the very column that breaks the                                                         
  serialisation. All seven vector builders failed on the same frame.                                                
                                                                                                                    
  Coerce date-like columns to ISO-8601 in _display_gdf, the choke point                                             
  every vector builder already shares. ISO is what GeoJSON consumers                                                
  expect, it sorts lexicographically so timeslider's ordering still                                                 
  holds, and MapLibre expressions compare the strings directly. Missing                                             
  values become null rather than the text "NaT", which would otherwise                                              
  show up in a popup as a real value. Frames carrying no date-like                                                  
  column are returned untouched, and the coercion copies rather than                                                
  mutating the caller's frame.                                                                                      
                                                                                                                    
  - Covers every spelling that fails json.dumps: datetime64 tz-naive and                                            
    tz-aware, timedelta64 and numpy.timedelta64 as ISO durations, period,                                           
    category (whose real dtype hides in .categories), datetime.time,                                                
    numpy.datetime64, and object columns holding any of them.                                                       
  - Scans an object column for any date-like value rather than sampling                                             
    the first non-null one, so ["n/a", date(...)] and its reverse behave                                            
    the same. A mixed column keeps its non-date values; only the                                                    
    date-like ones encode.                                                                                          
  - Builds the encoded column as an explicit object Series. Under                                                   
    pandas 3 .map/.where infer a str dtype whose missing marker is a                                                
    float nan, and json.dumps writes that as the bare literal NaN --                                                
    invalid JSON, not null.
- timeslider drops steps whose kdim is missing. Encoding NaT to None                                              
    otherwise broke its own sort with "'<' not supported between                                                    
    instances of 'NoneType' and 'str'"; a column of nothing but NaT                                                 
    still raises.                                                                                                   
  - Keeps the encoder's helpers at module level so no single function                                               
    carries the whole dispatch, and leaves the tier's lazy numpy/pandas                                             
    imports intact.                                                                                                 
                                                                                                                    
  Behaviour worth knowing, none of it breaking a previously-working                                                 
  call: a feature with no time value is still drawn, so it hides behind                                             
  the live slider and shows in a saved page that carries no filter; and                                             
  colouring by an encoded column needs scheme="categorical", since the                                              
  graduated schemes parse the property as a float. Colouring by a date                                              
  column never worked -- before this encoding it failed later, in                                                   
  json.dumps.                                                                                                       
                                                                                                                    
  Closes #156

### Refactor

- **deps**: drop the unused geostatista dependency and the map presets (#183)

## 0.10.0 (2026-09-09)

### Refactor

- **package**: split digitalearth into one subpackage per backend plus a shared base (#161)

## 0.9.0 (2026-09-06)

### Feat

- **geostatistics**: visualize geostatista results in the map tiers (#146)

### Fix

- **render**: recognise the lognorm colour scale (cleopatra 0.36.0)

## 0.8.0 (2026-09-04)

### Feat

- **scene**: add a 3-D textured globe, figure watermarks and clip saving (#136)
- **scene**: add categorical symbology to the static choropleth (#122)
- add field charts, categorized symbology, and web map controls (#110)

### Fix

- **deps,web**: upgrade pyramids 0.46.0 / cleopatra 0.25.0 and harden the maplibre UTF-8 shim (#119)
- **deps**: upgrade to pyramids 0.44.0 / cleopatra 0.24.0 and adapt to their API changes (#117)

## 0.7.0 (2026-06-15)

### Feat

- **web**: add the MapLibre + deck.gl web rendering tier (#96)
- **api**: add backend="3d" to quickplot/quickmap (dispatch to Scene3D) (#95)

## 0.6.0 (2026-06-14)

### Feat

- **interactive**: add the HoloViz interactive-2D web-map tier (#75)

## 0.5.0 (2026-06-11)

### Feat

- **three_d**: add the true-3D PyVista rendering tier (Scene3D) (#66)

### Refactor

- **decoration**: source Natural-Earth basemaps from cleopatra.reference (#63)

## 0.4.0 (2026-06-10)

### Feat

- decompose Map into a base + mixin architecture and tidy the package (#55)

## 0.3.0 (2026-06-07)

### Feat

- **vector**: add Voronoi, cartogram, quadtree, KDE and flow/Sankey map plots (#49)

## 0.2.0 (2026-06-02)

### Feat

- geospatial plotting suite (maps, fields, vectors, mesh, temporal, projected/globe, animation, charts, auto-styling, batch/CLI) (#36)


- migrate to pyproject/pixi/mkdocs and modernize the toolchain (#23)

## 0.1.11 (2023-01-31)

- bump up cleopatra version
- bump up pyramids version
- add github action for creating a release from tags

## 0.1.10 (2022-12-27)

- fix pypi package names in the requirements.txt
- fix python version number

## 0.1.9 (2022-12-19)

- lock numpy to version 1.23.5

## 0.1.8 (2022-12-19)

- use environment.yaml and requirements.txt instead of pyproject.toml and replace poetry env by conda env

## 0.1.0 (2022-05-24)

- First release on PyPI.
