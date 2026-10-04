# 3-D scenes (PyVista)

`Scene3D` is the 3-D tier's host: one PyVista plotter and the layers drawn into it. It answers to the same
method names as the 2-D tiers wherever it can draw the same thing, and adds what only a 3-D scene has — relief,
volumes and isosurfaces, extruded polygons, a globe, and a camera that can orbit.

```python
from digitalearth.three_d import Scene3D

with Scene3D(off_screen=True) as scene:
    scene.terrain(dem, z_exaggeration=3.0)
    scene.colorbar()
    scene.screenshot("relief.png")
```

The tier needs the `3d` extra (`pip install digitalearth[3d]`). On a Linux machine with no display, see
[Headless 3-D rendering](headless_3d.md) before the first render.

## Where the methods come from

`Scene3D` composes a base with one mixin per capability, mirroring how the static `Map` is built, and every
method below is listed on `Scene3D` itself whichever class defines it:

| Capability              | Methods                                         | Defined on        |
|-------------------------|-------------------------------------------------|-------------------|
| Relief                  | `terrain`                                       | `TerrainMixin`    |
| Points                  | `point_cloud`                                   | `PointCloudMixin` |
| 3-D fields              | `volume`, `isosurface`                          | `VolumeMixin`     |
| Vectors and extrusions  | `vectors`, `extruded_polygons`                  | `VectorMixin`     |
| Data on a sphere        | `globe`                                         | `GlobeMixin`      |
| Decoration              | `set_title`, `text`, `axes`, `orientation_axes` | `DecorationMixin` |
| Colour keys             | `colorbar`, `legend`                            | `GuideMixin`      |
| Camera motion           | `orbit`, `record`, `jupyter`                    | `AnimationMixin`  |
| Layers, view and output | everything else                                 | `Scene3DBase`     |

## The plotter is built on first use

Creating a scene does not open a render window or import PyVista; the plotter is built the first time
something is drawn or the `plotter` property is read. A scene can therefore be described — its layers listed,
its `figure_spec` taken — without a renderer, and `Scene3D.from_figure` draws a figure described elsewhere.

::: digitalearth.three_d.scene3d.Scene3D
    options:
      inherited_members: true
      docstring_options:
        warn_missing_types: false
      members:
        - terrain
        - point_cloud
        - volume
        - isosurface
        - vectors
        - extruded_polygons
        - globe
        - set_title
        - text
        - axes
        - orientation_axes
        - colorbar
        - legend
        - orbit
        - record
        - jupyter
        - layer_ids
        - layers
        - get_layer
        - move_layer
        - remove_layer
        - replace_layer
        - set_visible
        - figure_spec
        - from_figure
        - draw_figure
        - add_mesh
        - add_volume
        - actor_of
        - mesh_of
        - held_objects
        - camera
        - view
        - vertical_exaggeration
        - plotter
        - renderer
        - render
        - screenshot
        - save
        - export_html
        - show
        - close
