"""MeshSource — a :class:`Source` whose geometry is an explicit triangulation.

A raster ``Source`` carries a grid and two axis vectors; a point ``Source`` carries three parallel 1-D
arrays. An **unstructured mesh** is neither: its cells are listed, not implied by a grid, so the geometry
only exists once the connectivity travels with the coordinates. That is what this class adds — the
triangles, and *where* on them the values sit.

``triangles``, ``location`` and the node coordinates are born together, travel together and die together,
which is why they are one value object rather than a tuple threaded through a parameter list.

This module is a **leaf** like :mod:`digitalearth.base.sources.source`: it imports no renderer, so the
engine-neutrality of ``base/`` holds (a UGRID mesh reaches matplotlib through cleopatra's ``MeshGlyph``,
which is the static tier's business, not this layer's).
"""

from typing import Any

import numpy as np

from digitalearth.base.sources.dimension import DimensionInfo
from digitalearth.base.sources.source import Source

__all__ = ["MESH_LOCATIONS", "MeshSource"]

#: Where a mesh variable's values sit. CF-UGRID names more locations (``edge``, ``volume``); these are the
#: two a 2-D surface render can colour, so a variable on any other location is refused by the extractor
#: rather than guessed at.
MESH_LOCATIONS: tuple[str, ...] = ("node", "face")


class MeshSource(Source):
    """An unstructured-mesh source: node coordinates, an explicit triangulation, and values on it.

    Args:
        z: The values (a :class:`DimensionInfo`) — one per node for ``location="node"``, one per triangle
            for ``location="face"``.
        x: The node x / longitude coordinates (a :class:`DimensionInfo`), 1-D, one per node.
        y: The node y / latitude coordinates (a :class:`DimensionInfo`), 1-D, one per node.
        triangles: The triangulation as an ``(n_triangles, 3)`` array of node indices. A mesh face with
            more than three nodes is already fan-triangulated by the extractor, so this is always
            triangles — the one shape every triangular renderer takes.
        location: ``"node"`` or ``"face"`` — which of the two ``z`` is aligned with.
        crs: The CRS the coordinates are expressed in, **as given**.
        metadata: Free-form metadata (``kind``, ``variable``, ``n_faces``, …).
        units: Unit string for the values, or ``None``.

    Examples:
        - A two-triangle mesh with a value on each face:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import DimensionInfo, MeshSource
            >>> nodes_x = DimensionInfo(np.array([0.0, 1.0, 0.0, 1.0]), "x")
            >>> nodes_y = DimensionInfo(np.array([0.0, 0.0, 1.0, 1.0]), "y")
            >>> faces = np.array([[0, 1, 2], [1, 3, 2]])
            >>> src = MeshSource(DimensionInfo(np.array([10.0, 20.0]), "z"), nodes_x, nodes_y,
            ...                  triangles=faces, location="face", crs=4326)
            >>> src.triangles.shape
            (2, 3)
            >>> src.location
            'face'
            >>> src.n_triangles
            2

            ```
        - The values may instead sit on the nodes, in which case there is one per coordinate:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import DimensionInfo, MeshSource
            >>> src = MeshSource(DimensionInfo(np.arange(4.0), "z"),
            ...                  DimensionInfo(np.zeros(4), "x"), DimensionInfo(np.zeros(4), "y"),
            ...                  triangles=np.array([[0, 1, 2]]), location="node")
            >>> src.location, src.z.values.size
            ('node', 4)

            ```
        - An unknown location is refused, because a renderer would otherwise guess which array to colour:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import DimensionInfo, MeshSource
            >>> try:
            ...     MeshSource(None, DimensionInfo(np.zeros(3), "x"), DimensionInfo(np.zeros(3), "y"),
            ...                triangles=np.array([[0, 1, 2]]), location="edge")
            ... except ValueError as error:
            ...     print(error)
            location='edge' is not one of ('node', 'face')

            ```
    """

    def __init__(
        self,
        z: DimensionInfo | None,
        x: DimensionInfo,
        y: DimensionInfo,
        *,
        triangles: Any,
        location: str,
        crs: Any = None,
        metadata: dict | None = None,
        units: str | None = None,
    ):
        if location not in MESH_LOCATIONS:
            raise ValueError(f"location={location!r} is not one of {MESH_LOCATIONS}")
        super().__init__(z, x, y, crs, metadata, units)
        self._triangles = np.asarray(triangles)
        self._location = location

    @property
    def triangles(self) -> np.ndarray:
        """The triangulation as an ``(n_triangles, 3)`` array of node indices."""
        return self._triangles

    @property
    def location(self) -> str:
        """Which of :data:`MESH_LOCATIONS` the values in ``z`` are aligned with."""
        return self._location

    @property
    def n_triangles(self) -> int:
        """How many triangles the mesh draws as."""
        return int(self._triangles.shape[0]) if self._triangles.ndim == 2 else 0
