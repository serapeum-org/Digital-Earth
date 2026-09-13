# Renamed keywords

One idea should have one spelling on every backend. A marker's visual size is `size=` whether matplotlib,
HoloViz, PyVista or MapLibre draws it; a frame rate is `fps=`; the feature count at which a builder switches
to a big-data renderer is `big_data_threshold=`. Getting there meant renaming twelve keywords that had
drifted apart tier by tier, plus one method and one return type.

**Every old spelling still works for one release.** Passing one emits a `DeprecationWarning` that names the
method *and* the new keyword, located at your call rather than inside the package. Passing both spellings of
the same parameter in a single call is a `TypeError`, not a silent winner — a contradictory call is a
mistake, and guessing which half was meant is worse than refusing it. The rule lives in one place,
`digitalearth.base.deprecation.renamed_parameter`, so the message reads the same on all four tiers.

## The twelve keywords

| Backend | Method | Old spelling | Current spelling |
|---|---|---|---|
| matplotlib | `Map.scatter`, `Map.grid_points`, `Map.point_cloud` | `point_size=` | `size=` |
| matplotlib | `Map.scatter` | `scale=` | `size_column=` |
| interactive | `InteractiveMap.points`, `.polygons`, `.trimesh` | `rasterize_threshold=` | `big_data_threshold=` |
| 3-D | `Scene3D.point_cloud` | `point_size=` | `size=` |
| 3-D | `Scene3D.orbit` | `framerate=` | `fps=` |
| 3-D | `Scene3D.animate` | `framerate=` | `fps=` |
| web | `WebMap.points` | `radius=` | `size=` |
| web | `WebMap.deck_scatter` | `radius=` | `size=` |
| web | `WebMap.point_cloud` | `point_size=` | `size=` |
| web | `WebMap.labels` | `size=` | `text_size=` |
| web | `WebMap.text` | `size=` | `text_size=` |
| web | `WebMap.animate` | `duration=` | `fps=` |

Two of those deserve a word beyond the table:

- **`scale=` never scaled anything.** It named the *column* whose values vary the marker size, which is why
  it is now `size_column=`. The single uniform size a layer draws when no column is given is `size=`, the
  same keyword every other tier uses.
- **`size=` on `WebMap.labels` / `WebMap.text` is a font size, not a marker size.** It is now `text_size=`,
  so `size=` can mean the one thing it means everywhere else. On those two methods only, the old and the new
  spelling are both `size`-shaped — read the warning, it names which method it came from.

### `duration=` and `fps=` measure opposite things

`WebMap.animate(duration=0.5)` meant *half a second held per frame*; `fps=` is *frames per second*, so the
same clip is now `fps=2`. The deprecated keyword is converted for you (`fps = 1 / duration`), and a
non-positive `duration` is refused rather than turned into a negative rate.

## `WebMap.to_gif()` is now `WebMap.animate()`

`animate()` is the real method and the name the other tiers already used (`Scene3D.animate`, `Map.animate`).
`to_gif()` survives as a deprecated alias with the identical signature and result — calling it warns once,
at your line. Prefer `animate()`, or let the suffix pick for you: `WebMap.save("clip.gif")` routes to
`animate()` on its own.

## `Scene3D.save()` returns a path, not a frame

`Scene3D.save()` used to hand back the RGB frame for a raster suffix and `None` for an export — a return
type that changed with the file extension. It now returns the `pathlib.Path` it wrote on every branch, like
`Map.save`, `InteractiveMap.save` and `WebMap.save`.

Unlike the keyword renames, a return *type* has no alias to offer, so this one is a break rather than a
deprecation. Ask for the pixels by name instead — `screenshot()` writes the same file **and** returns the
array:

```python
frame = scene.save(out)        # before: an ndarray for a .png, None for a .gltf
frame = scene.screenshot(out)  # now: writes the file and returns the RGB frame
path = scene.save(out)         # now: writes the file and returns its Path
```

## Shared defaults

The renames came with one shared default per idea, so the same call renders the same way on every backend:

| Constant (in `digitalearth.base`) | Value | What it sets |
|---|---|---|
| `animation.DEFAULT_FPS` | `3.0` | Frame rate for `Scene3D.orbit` / `.animate` and `WebMap.animate` |
| `bigdata.DEFAULT_BIG_DATA_THRESHOLD` | `50_000` | Feature count that switches to the big-data renderer |
| `basemaps.DEFAULT_BASEMAP_PROVIDER` | `"CartoLight"` | Basemap a bare `basemap()` draws |
| `symbology.MISSING_COLOR` | `"#cccccc"` | Colour a classified layer paints a missing value |

`Scene3D.orbit` (12 fps), `Scene3D.animate` (8 fps) and `WebMap.animate` (`duration=0.8`, i.e. 1.25 fps) each
had a default of their own before; all three now start from `DEFAULT_FPS`. An existing call that relied on
the old default renders at a different speed — pass `fps=` explicitly to pin it.
