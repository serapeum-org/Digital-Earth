"""Shared UGRID-mesh construction for the 3-D tier's tests.

The 3-D mesh suites (`test_mesh3d`, `test_capabilities3d`, `test_kwargs_builders3d`) all need the same bare
``pyramids.netcdf.ugrid.mesh.Mesh2d`` — a two-face mesh (a quad and a triangle) that fan-cuts into three
triangles — rather than the ``MeshDataset`` wrapper the shared ``ugrid_mesh`` conftest fixture returns. Defining
that construction once here keeps the three suites from each carrying their own copy of it.
"""

from typing import Any

import numpy as np


def bare_ugrid_mesh(node_x: Any = None, node_y: Any = None, epsg: Any = None) -> Any:
    """Build a two-face UGRID ``Mesh2d`` (a quad and a triangle) directly, without the ``MeshDataset`` wrapper.

    Args:
        node_x: Node x coordinates; ``None`` uses ``[0, 1, 1, 0, 2]`` (pass non-finite ones to make a mesh a
            projection drops).
        node_y: Node y coordinates; ``None`` uses ``[0, 0, 1, 1, 0.5]``.
        epsg: An EPSG code the mesh declares as its CRS; ``None`` leaves it unreferenced.

    Returns:
        A pyramids ``Mesh2d`` whose quad and triangle fan-cut into three triangles.
    """
    from pyramids.netcdf.ugrid.connectivity import Connectivity
    from pyramids.netcdf.ugrid.mesh import Mesh2d

    mesh = Mesh2d(
        node_x=(
            np.array([0.0, 1.0, 1.0, 0.0, 2.0])
            if node_x is None
            else np.asarray(node_x, dtype="float64")
        ),
        node_y=(
            np.array([0.0, 0.0, 1.0, 1.0, 0.5])
            if node_y is None
            else np.asarray(node_y, dtype="float64")
        ),
        face_node_connectivity=Connectivity(
            data=np.array([[0, 1, 2, 3], [1, 4, 2, -1]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
    )
    if epsg is not None:
        from pyramids.base.crs import crs_from_user_input

        mesh.crs = crs_from_user_input(epsg)
    return mesh
