# Headless 3-D rendering

The 3-D tier draws through PyVista, and PyVista draws through VTK, which needs an **OpenGL context** even when
no window is shown. On Windows and macOS the platform provides one. On Linux the stock `vtk` wheel gets it from
an X server, so a machine with no screen — a server, a container, a remote Jupyter kernel — cannot render a
`Scene3D` until it is given one. There are three ways to do that.

## 1. Render off-screen

Whatever else you do, ask for no window:

```python
from digitalearth.three_d import Scene3D

with Scene3D(off_screen=True) as scene:
    scene.terrain(dem)
    scene.screenshot("relief.png")
```

`off_screen=None` (the default) follows `pyvista.OFF_SCREEN`, so setting that once at the top of a script does
the same for every scene. Off-screen stops a window being opened; it does not remove the need for an OpenGL
context on Linux — the next two sections are where that comes from.

## 2. A virtual X server (Xvfb)

**Around the whole process.** `xvfb-run` starts a virtual X server, sets `DISPLAY`, runs the command and stops
the server afterwards. It is what this project's CI does for every 3-D job:

```bash
sudo apt-get install -y xvfb libgl1 libglx-mesa0 libgl1-mesa-dri
xvfb-run -a python render_scene.py
```

**From inside Python.** When you cannot wrap the process — a notebook kernel someone else started, say —
call `start_xvfb()` once before the first scene is drawn. It needs the `Xvfb` binary installed (the `xvfb`
package above) and does nothing where a display already exists, so it is safe to leave in a script that also
runs on a desktop, under `xvfb-run`, or on Windows and macOS:

```python
from digitalearth.three_d import Scene3D, start_xvfb

server = start_xvfb()          # None when no server was needed
with Scene3D(off_screen=True) as scene:
    scene.terrain(dem)
    scene.screenshot("relief.png")
if server is not None:
    server.terminate()
```

PyVista used to ship a helper of this name and no longer does — the PyVista this tier is tested against has
none — which is why Digital-Earth carries its own.

## 3. EGL or OSMesa — no X server at all

VTK can also create its context through **EGL** (GPU drivers, no X server) or **OSMesa** (Mesa's software
renderer). On Linux, VTK picks a backend at run time and lets you override the choice with the
`VTK_DEFAULT_OPENGL_WINDOW` environment variable, set before VTK is imported:

| Value                     | Backend                                                          |
|---------------------------|------------------------------------------------------------------|
| `vtkXOpenGLRenderWindow`  | X server (the default when one is reachable)                     |
| `vtkEGLRenderWindow`      | EGL — needs a VTK built with EGL and a driver that provides it   |
| `vtkOSOpenGLRenderWindow` | OSMesa — needs `libOSMesa.so` installed                          |

```bash
VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow python render_scene.py
```

Which of these a given `vtk` build supports, and what each needs installed, is VTK's to document — see
[VTK runtime settings](https://docs.vtk.org/en/latest/advanced/runtime_settings.html). Choosing one is an
installation decision rather than something Digital-Earth switches for you; nothing in the `3d` extra pins a
particular backend.

::: digitalearth.three_d.headless.start_xvfb
