"""three_d — Digital-Earth's true-3D visualization tier, built on PyVista.

Public surface::

    from digitalearth.three_d import Scene3D
    scene = Scene3D(off_screen=True)
    scene.add_mesh(mesh, scalars="z", cmap="terrain")
    scene.screenshot("frame.png")          # or scene.export_html("scene.html")

PyVista is a **renderer, not a GIS engine**: every mesh is built from pyramids-sourced numpy arrays + geometry
(``read_array``, ``Dataset.x/.y``, ``geotransform``, ``to_geodataframe`` …) — **never** ``xarray``/``rasterio``/
``pyvista-xarray`` or any GIS competitor (enforced by ``tests/test_no_competitor_imports.py``). All CRS/reproject
work stays in pyramids.

Requires the optional ``3d`` extra (``pip install digitalearth[3d]`` → pyvista + trame; geovista for the globe).
"""
from digitalearth.three_d.scene3d import Scene3D

__all__ = ["Scene3D"]
