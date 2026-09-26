# Renamed names

One idea should have one spelling on every backend. A marker's visual size is `size=` whether matplotlib,
HoloViz, PyVista or MapLibre draws it; a frame rate is `fps=`; the feature count at which a builder switches
to a big-data renderer is `big_data_threshold=`. Getting there meant renaming the keywords that had drifted
apart tier by tier, and then the **methods** themselves, so that a raster field is `field()` on all four —
plus one return type, which is the only change here no alias can soften.

**Every old spelling still works for one release.** Calling or passing one emits a `DeprecationWarning` that
names the method *and* its replacement, located at your call rather than inside the package. Passing both
spellings of the same parameter in a single call is a `TypeError`, not a silent winner — a contradictory call
is a mistake, and guessing which half was meant is worse than refusing it. The rules live in one place,
`digitalearth.base.deprecation`, so the message reads the same on all four tiers: `renamed_parameter` for a
keyword, `renamed_method` for a method.

The **Backend** column in both tables below is the string `quickmap(backend=...)` takes, which is also the
key the tables in `digitalearth.base.contract` are grouped by.

## The renamed methods

Every row here is a live alias: the old name exists, warns once, and names its replacement. The table is
held against `digitalearth.base.contract.ALIASES` by `tests/test_renamed_names_page.py`, in both directions —
an alias missing from this page fails, and so does a row this page invents.

| Backend | Old spelling | Current spelling |
|---|---|---|
| matplotlib | `contour` | `contours` |
| matplotlib | `contourf` | `contours` |
| matplotlib | `imshow` | `field` |
| matplotlib | `scatter` | `points` |
| matplotlib | `set_extent` | `set_bounds` |
| matplotlib | `shapes` | `polygons` |
| interactive | `add_element` | `add_layer` |
| interactive | `image` | `field` |
| interactive | `path` | `lines` |
| 3d | `animate` | `record` |
| web | `add_raster` | `field` |
| web | `animate` | `save_animation` |
| web | `fit_bounds` | `set_bounds` |
| web | `globe` | `projection` |
| web | `save_gif` | `save_animation` |
| web | `terrain` | `terrain_tiles` |
| web | `title` | `set_title` |
| web | `to_gif` | `save_animation` |

Four of those are worth a sentence beyond the table.

- **`contour` / `contourf` fold into one method with a switch.** Two names for filled and unfilled isolines
  became `contours(filled=...)`, which is why these two aliases *translate* rather than merely forward:
  `contour(...)` calls `contours(..., filled=False)` and `contourf(...)` calls `contours(..., filled=True)`.
  Pass `filled=` yourself and one method covers both.
- **`globe` folds the same way.** `WebMap.globe()` is `projection("globe")`, and `projection(name=...)` takes
  any projection the tier can draw — so the name that meant one projection now names the choice.
- **Three web names all meant "write an animation".** `animate`, `to_gif` and `save_gif` are aliases of
  `save_animation`, which is the Core name and the one the other tiers answer to. Or let the suffix pick for
  you: `WebMap.save("clip.gif")` routes to `save_animation()` on its own.
- **`set_extent` was never quite `set_bounds`.** It framed the axes and returned nothing in particular;
  `set_bounds` frames on a region, takes `padding=`, accepts `None` for "fit the data", and returns `self` so
  it chains. The old name is a live alias, not a synonym of the old behaviour.

A method rename changes nothing underneath. The figure declarations still record the layer kind they always
did (`"imshow"`, `"scatter"`, `"shapes"`, `"image"`), so a figure saved before the rename reads back into the
same drawer.

## The renamed keywords

| Backend | Method | Old spelling | Current spelling |
|---|---|---|---|
| matplotlib | `Map.points`, `Map.grid_points`, `Map.point_cloud` | `point_size=` | `size=` |
| matplotlib | `Map.points` | `scale=` | `size_column=` |
| matplotlib | `Map.choropleth` | `alpha=` | `opacity=` |
| interactive | `InteractiveMap.points`, `.polygons`, `.trimesh` | `rasterize_threshold=` | `big_data_threshold=` |
| interactive | `InteractiveMap.points`, `.polygons`, `.choropleth` | `alpha=` | `opacity=` |
| interactive | `InteractiveMap.points` | `value_column=` | `column=` |
| 3d | `Scene3D.point_cloud` | `point_size=` | `size=` |
| 3d | `Scene3D.orbit` | `framerate=` | `fps=` |
| 3d | `Scene3D.record` | `framerate=` | `fps=` |
| web | `WebMap.points` | `radius=` | `size=` |
| web | `WebMap.deck_scatter` | `radius=` | `size=` |
| web | `WebMap.point_cloud` | `point_size=` | `size=` |
| web | `WebMap.labels` | `size=` | `text_size=` |
| web | `WebMap.text` | `size=` | `text_size=` |
| web | `WebMap.text` | `string=` | `s=` |
| web | `WebMap.layer_control` | `layer_ids=` | `layers=` |
| web | `WebMap.save_animation` | `duration=` | `fps=` |

Three of those deserve a word beyond the table:

- **`scale=` never scaled anything.** It named the *column* whose values vary the marker size, which is why
  it is now `size_column=`. The single uniform size a layer draws when no column is given is `size=`, the
  same keyword every other tier uses.
- **`size=` on `WebMap.labels` / `WebMap.text` is a font size, not a marker size.** It is now `text_size=`,
  so `size=` can mean the one thing it means everywhere else. On those two methods only, the old and the new
  spelling are both `size`-shaped — read the warning, it names which method it came from.
- **`value_column=` is still current on two methods.** `InteractiveMap.trimesh` and `Scene3D.point_cloud`
  take a `value_column=` that has never been renamed; only `InteractiveMap.points` spells it `column=` now,
  because that is the tier's classified-points builder and `column=` is the Core keyword for it.

### `duration=` and `fps=` measure opposite things

`WebMap.save_animation(duration=0.5)` meant *half a second held per frame*; `fps=` is *frames per second*, so
the same clip is now `fps=2`. The deprecated keyword is converted for you (`fps = 1 / duration`), and a
non-positive `duration` is refused rather than turned into a negative rate.

## `Scene3D.save()` returns a path, not a frame

`Scene3D.save()` used to hand back the RGB frame for a raster suffix and `None` for an export — a return
type that changed with the file extension. It now returns the `pathlib.Path` it wrote on every branch, like
`Map.save`, `InteractiveMap.save` and `WebMap.save`.

Unlike the keyword and method renames, a return *type* has no alias to offer, so this one is a break rather
than a deprecation. Ask for the pixels by name instead — `screenshot()` writes the same file **and** returns
the array:

```python
frame = scene.save(out)        # before: an ndarray for a .png, None for a .gltf
frame = scene.screenshot(out)  # now: writes the file and returns the RGB frame
path = scene.save(out)         # now: writes the file and returns its Path
```

## Shared defaults

The renames came with one shared default per idea, so the same call renders the same way on every backend:

| Constant (in `digitalearth.base`) | Value | What it sets |
|---|---|---|
| `animation.DEFAULT_FPS` | `3.0` | Frame rate for `Scene3D.orbit` / `.record` and `WebMap.save_animation` |
| `bigdata.DEFAULT_BIG_DATA_THRESHOLD` | `50_000` | Feature count that switches to the big-data renderer |
| `basemaps.DEFAULT_BASEMAP_PROVIDER` | `"CartoLight"` | Basemap a bare `basemap()` draws |
| `symbology.MISSING_COLOR` | `"#cccccc"` | Colour a classified layer paints a missing value |

`Scene3D.orbit` (12 fps), `Scene3D.record` (8 fps) and `WebMap.save_animation` (`duration=0.8`, i.e. 1.25 fps)
each had a default of their own before; all three now start from `DEFAULT_FPS`. An existing call that relied
on the old default renders at a different speed — pass `fps=` explicitly to pin it.
